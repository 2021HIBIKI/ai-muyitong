"""主动巡检（阶段 7 A3）：采集指标 → 确定性评估 → 告警落库 → SSE 推送大屏。

「你问才答」到「自己盯着」：定时（或手动）采集草情指标，按确定性规则判定是否需要
告警，需要则落库并通过 SSE 推给大屏；LLM 只负责在**升级**时补充建议文字（省配额）。

**巡检单位 = 草场**：一轮巡检逐个检查每个草场（各自的编号照片聚合状况），逐场判定、
逐场告警——草场之间彼此独立，冷却与升级判定互不影响，本轮每个草场的结果都会返回
（含未告警的草场，见 ``run_patrol`` 的 ``results``）。

评估规则（确定性纯函数，见 ``evaluate``）：
- 升级必发（不受冷却限制）；降级发一条恢复通知；
- 同级：仅在冷却期外补发提醒，冷却期内抑制；
- 首次巡检：非 normal 必发；
- 阈值复用 ``app.vision.grassland.alert_level_from_coverage``（不重写）。
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from app.config import settings
from app.data import repository
from app.data.db import connect
from app.vision.grassland import alert_level_from_coverage

logger = logging.getLogger(__name__)

# 等级严重度（用于升级/降级判断）
SEVERITY: dict[str, int] = {"normal": 0, "warning": 1, "danger": 2}
LEVEL_LABEL: dict[str, str] = {"normal": "正常", "warning": "预警", "danger": "危险"}

_SOURCE_LABEL: dict[str, str] = {
    "pasture": "本草场编号照片聚合",
    "image": "巡检点位图片",
    "metric": "最新覆盖度记录",
    None: "无数据",
}


def _pasture_label(metrics: dict[str, Any]) -> str:
    """告警对象标签：优先「编号 + 名称」，无草场信息时退回点位名。"""
    code = str(metrics.get("pasture_code") or "").strip()
    name = str(metrics.get("pasture_name") or "").strip()
    label = f"{code} {name}".strip()
    if label:
        return label
    return "巡检点位" if metrics.get("source") == "image" else "草场"


def _source_text(metrics: dict[str, Any]) -> str:
    """数据来源文案（草场独立口径：注明本草场的分区/照片规模）。"""
    if metrics.get("source") == "pasture":
        return (
            f"本草场编号照片聚合（{metrics.get('analyzed_plot_count') or 0}/"
            f"{metrics.get('plot_count') or 0} 个分区 · {metrics.get('image_count') or 0} 张照片）"
        )
    return _SOURCE_LABEL.get(metrics.get("source"), str(metrics.get("source")))


def parse_ts(value: str | None) -> datetime | None:
    """ISO8601（含 Z 后缀）→ aware datetime；解析失败返回 None。"""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass
class PatrolDecision:
    """一次巡检的判定结果（纯数据，便于测试与落库）。"""

    level: str
    should_alert: bool
    escalated: bool
    recovered: bool
    reason: str  # escalated / recovered / first / cooldown_suppressed / repeat_reminder / normal
    title: str
    message: str


def _format_message(level: str, metrics: dict[str, Any], note: str) -> str:
    coverage = metrics.get("coverage")
    source = _source_text(metrics)
    if coverage is None:
        body = "暂无覆盖度数据，无法给出量化判定"
    else:
        warn = settings.alert_warning_threshold
        danger = settings.alert_danger_threshold
        if level == "danger":
            body = (
                f"覆盖度 {coverage:.1%} 低于危险阈值 {danger:.0%}，草场退化风险高，"
                "建议禁牧/休牧并补播牧草或人工补饲"
            )
        elif level == "warning":
            body = (
                f"覆盖度 {coverage:.1%} 处于预警区间（{danger:.0%}–{warn:.0%}），"
                "建议降低放牧强度、缩短放牧周期"
            )
        else:
            body = f"覆盖度 {coverage:.1%} 已回到正常区间（≥{warn:.0%}），可恢复正常放牧并持续监测"
    text = f"{body}。数据来源：{source}。{note}"
    health = metrics.get("health") or {}
    if health.get("score") is not None:
        text += f"多模态草情评分 {health['score']:.0f}/100（覆盖度·绿度·环境融合）。"
    return text.strip()


def evaluate(
    metrics: dict[str, Any],
    prev_level: str | None,
    last_alert_at: datetime | None,
    now: datetime,
    cooldown_seconds: float,
) -> PatrolDecision:
    """巡检判定（纯函数）：是否告警、等级、原因与文案。

    ``prev_level`` 为上一次告警等级（None = 首次巡检）；``last_alert_at`` 为上次告警时间
    （用于同级冷却）；``now`` 为当前时间（注入以便测试）。
    """
    level = alert_level_from_coverage(metrics.get("coverage"))
    escalated = prev_level is not None and SEVERITY[level] > SEVERITY[prev_level]
    recovered = prev_level is not None and SEVERITY[level] < SEVERITY[prev_level]

    if prev_level is None:
        should_alert = level != "normal"
        reason = "first" if should_alert else "normal"
    elif escalated:
        should_alert = True
        reason = "escalated"
    elif recovered:
        should_alert = True
        reason = "recovered"
    elif level != "normal":
        in_cooldown = (
            last_alert_at is not None
            and (now - last_alert_at).total_seconds() < cooldown_seconds
        )
        should_alert = not in_cooldown
        reason = "cooldown_suppressed" if in_cooldown else "repeat_reminder"
    else:
        should_alert = False
        reason = "normal"

    note = ""
    if reason == "escalated":
        note = f"等级由{LEVEL_LABEL.get(prev_level or 'normal')}升级为{LEVEL_LABEL[level]}，已即时告警。"
    elif reason == "recovered":
        note = f"等级由{LEVEL_LABEL[prev_level or 'normal']}回落至{LEVEL_LABEL[level]}，本条为恢复通知。"
    elif reason == "cooldown_suppressed":
        note = f"同级告警处于冷却期（{cooldown_seconds:.0f}s）内，已抑制重复推送。"

    who = _pasture_label(metrics)
    title = f"{who} · {LEVEL_LABEL[level]}告警" if level != "normal" else f"{who} · 状态恢复"
    return PatrolDecision(
        level=level,
        should_alert=should_alert,
        escalated=escalated,
        recovered=recovered,
        reason=reason,
        title=title,
        message=_format_message(level, metrics, note),
    )


def pasture_metrics(conn, pasture) -> dict[str, Any]:
    """单个草场的巡检指标：本草场编号照片聚合状况（草场之间互不影响）。"""
    from app.data.pasture_service import pasture_summary

    summary = pasture_summary(conn, pasture)
    return {
        "pasture_id": summary["id"],
        "pasture_code": summary["code"],
        "pasture_name": summary["name"],
        "coverage": summary["coverage"],
        "greenness": summary["greenness"],
        "alert_level": summary["alert_level"],
        "plot_count": summary["plot_count"],
        "analyzed_plot_count": summary["analyzed_plot_count"],
        "image_count": summary["image_count"],
        "updated_at": summary["updated_at"],
        "source": "pasture",
    }


def collect_metrics(conn) -> list[dict[str, Any]]:
    """采集本轮巡检指标（**按草场独立**：每个草场一条，各自用自身照片聚合状况）。

    草场之间互不影响——没有跨草场平均。库中尚无草场时退回「巡检点位图片」
    （``PATROL_IMAGE_PATH``，模拟固定摄像头）或最新覆盖度记录，返回单条「巡检点位」
    指标（兼容旧库与单点位场景）。
    """
    rows = repository.list_pastures(conn)
    if rows:
        return [pasture_metrics(conn, row) for row in rows]

    return [_point_metrics(conn)]


def _point_metrics(conn) -> dict[str, Any]:
    """兜底单点位指标：配置了巡检图片则实测该图，否则取最新覆盖度记录。"""
    path = (settings.patrol_image_path or "").strip()
    if path:
        import cv2

        from app.data.pasture_service import local_image_path

        local = local_image_path(path)
        image = cv2.imread(str(local)) if local else None
        if image is None:
            raise ValueError(f"巡检图片无法读取：{path}")
        from app.vision.grassland import coverage

        result = coverage(image)
        return {
            "coverage": round(result["coverage"], 4),
            "greenness": round(result["greenness"], 4),
            "source": "image",
            "image": path,
        }

    latest = repository.list_grassland_metrics(conn, limit=1)
    if latest:
        return {
            "coverage": latest[0]["coverage"],
            "greenness": latest[0]["greenness"],
            "source": "metric",
        }
    return {"coverage": None, "greenness": None, "source": None}


def alert_payload(row) -> dict[str, Any]:
    """告警记录 → 接口/SSE 载荷（含 metrics 解析与草场归属）。"""
    metrics = {}
    try:
        metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
    except (json.JSONDecodeError, TypeError):
        metrics = {}
    pasture_id = None
    try:
        pasture_id = row["pasture_id"]
    except (IndexError, KeyError):
        pasture_id = None
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "level": row["level"],
        "title": row["title"],
        "message": row["message"],
        "metrics": metrics,
        "acknowledged": bool(row["acknowledged"]),
        "pasture_id": pasture_id if pasture_id is not None else metrics.get("pasture_id"),
        "pasture_code": metrics.get("pasture_code"),
        "pasture_name": metrics.get("pasture_name"),
    }


def run_patrol(
    conn,
    *,
    now: datetime | None = None,
    metrics: dict[str, Any] | list[dict[str, Any]] | None = None,
    advice_fn: Callable[[dict[str, Any], PatrolDecision], str | None] | None = None,
) -> dict[str, Any]:
    """执行一轮巡检：**逐草场**采集 → 评估 → （需告警时）落库。

    每个草场独立判定：上一等级/上次告警时间取自**该草场自己**的告警历史
    （升级必发、同级冷却互不影响其他草场）。``metrics`` 可注入单条或一组指标
    （测试用）；``advice_fn`` 仅在该草场**升级**时被调用（LLM 补充建议，省配额）。

    返回 ``{"results": [逐场结果（含未告警的草场）], "alerts": [本轮新落库告警],
    "summary": {...}}``。
    """
    moment = now or datetime.now(timezone.utc)
    items = metrics if metrics is not None else collect_metrics(conn)
    if isinstance(items, dict):  # 单条注入（兼容旧用法）
        items = [items]

    results: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    for current in items:
        previous = repository.list_alerts(conn, limit=1, pasture_id=current.get("pasture_id"))
        prev_level = previous[0]["level"] if previous else None
        last_alert_at = parse_ts(previous[0]["created_at"]) if previous else None

        decision = evaluate(
            current,
            prev_level=prev_level,
            last_alert_at=last_alert_at,
            now=moment,
            cooldown_seconds=settings.patrol_cooldown_seconds,
        )

        item: dict[str, Any] = {
            "pasture_id": current.get("pasture_id"),
            "pasture_code": current.get("pasture_code"),
            "pasture_name": current.get("pasture_name"),
            "level": decision.level,
            "coverage": current.get("coverage"),
            "source": current.get("source"),
            "should_alert": decision.should_alert,
            "escalated": decision.escalated,
            "recovered": decision.recovered,
            "reason": decision.reason,
            "title": decision.title,
            "message": decision.message,
            "alert": None,
        }
        if decision.should_alert:
            message = decision.message
            if decision.escalated and advice_fn is not None:
                advice = advice_fn(current, decision)
                if advice:
                    message = f"{message}\nAI 建议：{advice}"

            alert_id = repository.insert_alert(
                conn,
                level=decision.level,
                title=decision.title,
                message=message,
                metrics=current,
                pasture_id=current.get("pasture_id"),
                created_at=moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            )
            conn.commit()
            payload = alert_payload(repository.get_alert(conn, alert_id))
            item["message"] = message
            item["alert"] = payload
            alerts.append(payload)
        results.append(item)

    return {
        "results": results,
        "alerts": alerts,
        "summary": {
            "total": len(results),
            "alerted": len(alerts),
            "suppressed": sum(1 for r in results if r["reason"] == "cooldown_suppressed"),
            "normal": sum(1 for r in results if r["reason"] == "normal"),
            "recovered": sum(1 for r in results if r["recovered"]),
        },
    }


# ---------------------------------------------------------------------------
# 告警推送中枢（SSE 订阅者）与后台巡检调度
# ---------------------------------------------------------------------------

class AlertHub:
    """告警推送中枢：SSE 订阅者集合 + 广播。

    广播**须在事件循环内**调用（``publish``）；从 ``asyncio.to_thread`` 的工作线程里
    推送请用 ``publish_threadsafe``（内部走 ``loop.call_soon_threadsafe``）。
    慢消费者（队列满）直接丢弃本条，绝不阻塞巡检。
    """

    def __init__(self, maxsize: int = 32) -> None:
        self._maxsize = maxsize
        self._subscribers: set[asyncio.Queue] = set()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def publish(self, payload: dict[str, Any]) -> int:
        """向所有订阅者广播（事件循环内调用）；返回成功投递数。"""
        delivered = 0
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(payload)
                delivered += 1
            except asyncio.QueueFull:
                continue
        return delivered

    def publish_threadsafe(self, payload: dict[str, Any], loop: asyncio.AbstractEventLoop) -> None:
        """跨线程安全广播（工作线程内调用）。"""
        loop.call_soon_threadsafe(self.publish, payload)


# 全局告警中枢（大屏 SSE 端点与巡检调度共用）
alert_hub = AlertHub()


def _patrol_round() -> dict[str, Any]:
    """执行一轮巡检（同步，含 DB 与视觉工作）——由线程池调用。"""
    conn = connect()
    try:
        return run_patrol(conn)
    finally:
        conn.close()


async def patrol_loop(interval_seconds: float | None = None) -> None:
    """后台定时巡检任务（asyncio）：单轮失败不影响后续轮次，取消时干净退出。

    - 同步工作（视觉/DB）走 ``asyncio.to_thread``，不阻塞事件循环；
    - 广播在事件循环内进行（await 线程结果后再 publish，无跨线程 put）；
    - 循环体捕获 Exception 但**不吞** CancelledError。
    """
    interval = float(
        interval_seconds if interval_seconds is not None else settings.patrol_interval_seconds
    )
    while True:
        try:
            payload = await asyncio.to_thread(_patrol_round)
            for alert in payload.get("alerts") or []:  # 逐场告警分别推送
                delivered = alert_hub.publish(alert)
                logger.info(
                    "巡检告警已推送：%s level=%s 订阅者=%d",
                    alert.get("pasture_code") or "点位",
                    alert["level"],
                    delivered,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # 单轮失败不终止巡检
            logger.warning("巡检轮次失败：%s", exc)
        await asyncio.sleep(interval)


async def start_patrol_task() -> asyncio.Task | None:
    """按配置启动巡检任务（在 lifespan 内调用，不能 import 期创建）。"""
    if not settings.patrol_enabled:
        return None
    return asyncio.create_task(patrol_loop(), name="patrol_loop")


async def stop_patrol_task(task: asyncio.Task | None) -> None:
    """取消并等待巡检任务退出（必须 await，否则 pytest 报 pending 任务）。"""
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# 阶段 7.7：升级时的 LLM 建议（省配额：只在升级时调用）
# ---------------------------------------------------------------------------

_ADVICE_PROMPT = """你是牧场草情巡检助手。请针对本轮巡检结果给出**一句话**（40 字以内）的
紧要处理建议，不要重复数据、不要使用 Markdown、不要寒暄。

巡检指标：{metrics}
判定等级：{level}
告警标题：{title}
"""


def patrol_advice(metrics: dict[str, Any], decision: PatrolDecision) -> str | None:
    """升级告警时用 LLM 补充一句建议；失败/无 key 返回 None（告警仍按模板落库）。

    专用客户端 ``max_retries=0`` + 15s 超时：默认 2 次重试 + 指数退避会把巡检轮次
    拖到分钟级；``complete()`` 永不抛错（返回 fallback 哨兵），天然可测。
    """
    from app.agent.llm_client import LLMClient

    client = LLMClient(timeout=15.0, max_retries=0)
    if not client.enabled:
        return None
    prompt = _ADVICE_PROMPT.format(
        metrics=json.dumps(metrics, ensure_ascii=False),
        level=LEVEL_LABEL.get(decision.level, decision.level),
        title=decision.title,
    )
    result = client.complete([{"role": "user", "content": prompt}])
    if result.get("fallback"):
        return None
    text = (result.get("content") or "").strip()
    if not text:
        return None
    return text.splitlines()[0].strip()[:80]
