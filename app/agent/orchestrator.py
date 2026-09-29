"""Agent 编排器：NL → 规划 → 循环调用工具 → 结构化决策 JSON。

确定性优先：告警等级由覆盖度阈值规则算，载畜量由 NY/T 635 公式算，
LLM 只负责补充建议文字；无 key / LLM 失败时走关键字规则降级。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Literal

from pydantic import BaseModel

from app.agent.llm_client import LLMClient
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import dispatch_tool, tool_schemas
from app.config import settings
# alert_level_from_coverage 的重导出：口径统一在 app.vision.grassland（草场聚合/大屏/
# 编排器共用同一阈值规则），导入到本模块供既有调用方与测试继续使用。
from app.vision.grassland import (
    ALERT_CAPACITY_FACTOR,
    LEVEL_SEVERITY,
    REFERENCE_COVERAGE,
    alert_level_from_coverage,
    carrying_capacity,
    coverage_linked_yield,
)


class DecisionJSON(BaseModel):
    """Agent 输出的结构化决策。"""

    alert_level: Literal["normal", "warning", "danger"]
    recommended_actions: list[str]
    recommended_stocking: float
    rotation_plan: str
    evidence: list[str]


# 事件回调：供 SSE 思考流逐步推送 {type: thinking|analysis|tool_call|tool_result|answer|error, ...}
EventCallback = Callable[[dict[str, Any]], None]


def _pastures_analysis(pastures: list[dict[str, Any]]) -> str | None:
    """各草场独立覆盖度的一句话解读（确定性；不做跨草场平均）。"""
    if not pastures:
        return None
    parts = []
    for p in pastures:
        who = f"{p.get('code') or ''} {(p.get('name') or '').strip()}".strip() or "草场"
        if p.get("coverage") is None:
            parts.append(f"{who} 未分析")
        else:
            level = _ALERT_LABEL.get(p.get("alert_level"), p.get("alert_level"))
            parts.append(f"{who} {p['coverage']:.1%}（{level}）")
    return f"共 {len(pastures)} 个草场（各自独立判定）：" + "；".join(parts)


def analyze_tool_result(name: str, result: dict[str, Any]) -> str | None:
    """工具结果的确定性解读（供思考流展示的「分析」行；规则计算，非 LLM 生成）。"""
    if not result or result.get("error"):
        return None
    if name == "grassland_coverage":
        if result.get("source") == "pastures":
            # 各草场独立：逐场列出，不给跨草场平均（草场是独立管理单元）
            return _pastures_analysis(result.get("pastures") or [])
        c = result.get("coverage")
        if c is None:
            return None
        warn, danger = settings.alert_warning_threshold, settings.alert_danger_threshold
        if c >= warn:
            return f"覆盖度 {c:.1%} 高于正常阈值 {warn:.0%}，草情良好"
        if c >= danger:
            return f"覆盖度 {c:.1%} 处于预警区间（{danger:.0%}–{warn:.0%}），需控制放牧强度"
        return f"覆盖度 {c:.1%} 低于危险阈值 {danger:.0%}，草场处于危险区间"
    if name == "sheep_count":
        return f"识别到 {result.get('count', 0)} 只羊，与草场承载能力进行比对评估"
    if name == "carrying_capacity":
        return f"理论载畜量 {result.get('recommended_stocking')} 羊单位，作为草畜平衡上限参考"
    if name == "pasture_status":
        pasture = result.get("pasture")
        if pasture:
            value = pasture.get("coverage")
            if value is None:
                return f"「{pasture.get('name')}」尚无照片分析数据，建议先分析该草场照片"
            level = _ALERT_LABEL.get(pasture.get("alert_level"), pasture.get("alert_level"))
            return (
                f"「{pasture.get('name')}」草场覆盖度 {value:.1%}"
                f"（{pasture.get('analyzed_plot_count')}/{pasture.get('plot_count')} 个分区照片聚合），"
                f"处于{level}区间"
            )
        pastures = result.get("pastures") or []
        if pastures:
            return _pastures_analysis(pastures)
        return None
    if name == "query_history":
        return f"获取到 {result.get('count', 0)} 条历史记录，用于趋势判断"
    return None


_ACTIONS_BY_LEVEL: dict[str, list[str]] = {
    "normal": ["草地状况良好，可按推荐载畜量正常放牧", "保持轮牧并持续监测草情变化"],
    "warning": ["草地覆盖度偏低，建议适度降低放牧强度", "缩短放牧周期，增加休牧时间"],
    "danger": ["草地退化风险高，建议禁牧或休牧", "采取补播牧草或人工补饲措施"],
}


def actions_for_level(level: str) -> list[str]:
    """该告警等级的确定性建议措施（编排器与演示数据共用同一文案）。"""
    return list(_ACTIONS_BY_LEVEL.get(level, _ACTIONS_BY_LEVEL["normal"]))


def _is_short_suggestion(text: str) -> bool:
    """LLM 结论文本仅在简短且单行时并入「建议措施」。

    大模型常输出整篇 Markdown 报告；这类长文保留在思考流中完整展示，
    避免把整篇报告塞进措施列表（列表项应是简明措施）。
    """
    stripped = text.strip()
    return 0 < len(stripped) <= 80 and "\n" not in stripped

# 建议载畜量 = 理论载畜量 × 告警调整系数（口径统一在 app.vision.grassland）
_ALERT_CAPACITY_FACTOR = ALERT_CAPACITY_FACTOR
_ALERT_LABEL: dict[str, str] = {"normal": "正常", "warning": "预警", "danger": "危险"}

# 未提供草场参数时的默认口径（演示用：100 亩、利用率 50%、放牧 30 天）
_DEFAULT_CAPACITY_PARAMS: dict[str, float] = {"area": 100.0, "utilization": 0.5, "grazing_days": 30.0}

_DANGER_ROTATION_PREFIX = (
    "【建议禁牧】覆盖度处于危险区间，应立即停止放牧并安排补播/补饲，"
    "待覆盖度恢复至正常阈值以上再重新核定载畜量。"
)

_PASTURE_ROTATION_ACTION: dict[str, str] = {
    "danger": "禁牧/休牧并补播补饲",
    "warning": "降低放牧强度、缩短放牧周期",
    "normal": "按建议载畜量正常放牧",
}


def rotation_for_pastures(pastures: list[dict[str, Any]]) -> str:
    """多草场轮牧方案：各草场按自身等级独立执行（确定性，可单测）。"""
    parts = [
        f"{p.get('code') or '草场'} {_PASTURE_ROTATION_ACTION.get(p.get('alert_level'), '持续监测')}"
        for p in pastures
    ]
    return "各草场独立执行：" + "；".join(parts) + "。" if parts else ""


def link_stocking(
    coverage: float | None,
    theoretical: float | None = None,
    rotation: str | None = None,
    area_mu: float | None = None,
) -> dict[str, Any]:
    """载畜量 × 告警等级联动（确定性，编排器与大屏共用）。

    - theoretical 为空且覆盖度已知 → 按折算亩产自动计算理论载畜量
      （面积优先用 area_mu，缺省口径 100 亩、利用率 0.5、放牧 30 天）；
    - 建议载畜量 = 理论载畜量 × 告警系数（正常 1.0 / 预警 0.6 / 危险禁牧 0）；
    - 危险等级轮牧文本以禁牧说明整体替换，避免残留大数字自相矛盾。

    返回 {"alert_level", "theoretical", "factor", "recommended_stocking",
          "rotation_plan", "source"}，source ∈ {"tool", "coverage", None}。
    """
    params = dict(_DEFAULT_CAPACITY_PARAMS)
    if area_mu is not None:
        params["area"] = float(area_mu)

    alert = alert_level_from_coverage(coverage)
    factor = _ALERT_CAPACITY_FACTOR[alert]
    source: str | None = "tool" if theoretical is not None else None
    generated = False

    if theoretical is None and coverage is not None:
        linked_yield = coverage_linked_yield(coverage)
        theoretical = carrying_capacity(linked_yield, **params)["recommended_stocking"]
        source = "coverage"
        generated = True
        if factor > 0:
            rotation = carrying_capacity(linked_yield * factor, **params)["rotation_plan"]
    if theoretical is None:
        theoretical = 0.0

    stocking = round(theoretical * factor, 2)
    rotation = rotation or ""

    if alert == "danger":
        note = f"恢复后可参考理论载畜量 {theoretical} 羊单位重新核定。" if theoretical > 0 else ""
        rotation = _DANGER_ROTATION_PREFIX + note  # 整体替换，不残留原轮牧文本
    elif not generated and 0.0 < factor < 1.0 and rotation:
        rotation += (
            f"【告警折减】按{_ALERT_LABEL[alert]}等级折减至 {factor:.0%}，"
            f"建议载畜量 {stocking} 羊单位。"
        )

    return {
        "alert_level": alert,
        "theoretical": theoretical,
        "factor": factor,
        "recommended_stocking": stocking,
        "rotation_plan": rotation,
        "source": source,
    }


class AgentOrchestrator:
    """轻量 Agent 循环：按需调用工具，聚合结果，输出决策 JSON。"""

    def __init__(
        self,
        llm: LLMClient | None = None,
        dispatch: Callable[..., dict[str, Any]] | None = None,
        max_rounds: int | None = None,
    ) -> None:
        self.llm = llm if llm is not None else LLMClient()
        self.dispatch = dispatch if dispatch is not None else dispatch_tool
        self.max_rounds = max_rounds if max_rounds is not None else settings.agent_max_rounds
        self.last_tool_calls: list[str] = []

    def run(self, question: str, on_event: EventCallback | None = None) -> DecisionJSON:
        """处理一次自然语言问题，返回结构化决策。

        on_event 提供时逐步推送事件（供 SSE 思考流）：
        thinking / analysis / tool_call / tool_result / answer；每个事件带 ts（毫秒耗时）。
        """
        sink: EventCallback = on_event or (lambda event: None)
        started = time.monotonic()

        def emit(event: dict[str, Any]) -> None:
            event = dict(event)
            event["ts"] = int((time.monotonic() - started) * 1000)
            sink(event)

        if self.llm.enabled:
            results, final_text = self._run_with_llm(question, emit)
        else:
            emit(
                {
                    "type": "thinking",
                    "text": "未配置大模型 API Key，使用本地关键词规则匹配问题意图…",
                }
            )
            results, final_text = self._run_with_rules(question, emit), None

        self.last_tool_calls = list(results.keys())
        decision = self._build_decision(question, results, final_text)
        emit({"type": "answer", "decision": decision.model_dump()})
        return decision

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------

    def _dispatch_with_events(
        self, name: str, arguments: dict[str, Any], emit: EventCallback
    ) -> dict[str, Any]:
        """调用工具并推送 tool_call / tool_result / analysis 事件。"""
        emit({"type": "tool_call", "name": name, "arguments": arguments})
        result = self.dispatch(name, arguments)
        emit({"type": "tool_result", "name": name, "result": result})
        note = analyze_tool_result(name, result)
        if note:
            emit({"type": "analysis", "name": name, "text": note})
        return result

    def _run_with_llm(
        self, question: str, emit: EventCallback
    ) -> tuple[dict[str, dict[str, Any]], str | None]:
        """LLM function calling 循环（上限 max_rounds）。"""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        results: dict[str, dict[str, Any]] = {}
        final_text: str | None = None

        for round_index in range(self.max_rounds):
            if round_index > 0:
                emit({"type": "thinking", "text": f"第 {round_index + 1} 轮：结合工具结果继续分析…"})
            resp = self.llm.complete(messages, tools=tool_schemas())
            if resp.get("fallback"):
                # LLM 中途失败 → 用关键字规则补齐剩余工具
                emit(
                    {
                        "type": "thinking",
                        "text": f"大模型调用失败（{resp.get('reason')}），切换到本地规则降级…",
                    }
                )
                rule_results = self._run_with_rules(question, emit)
                results.update(rule_results)
                final_text = None
                break

            if resp.get("content"):
                emit({"type": "thinking", "text": str(resp["content"])})

            tool_calls = resp.get("tool_calls") or []
            if not tool_calls:
                final_text = resp.get("content")
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": resp.get("content"),
                    # OpenAI 规范线格式：tool_calls[].type=function + function.{name,arguments(JSON 字符串)}
                    "tool_calls": [
                        {
                            "id": tc.get("id") or f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                            },
                        }
                        for i, tc in enumerate(tool_calls)
                    ],
                }
            )
            for tc in tool_calls:
                name = tc.get("name")
                result = self._dispatch_with_events(name, tc.get("arguments") or {}, emit)
                results[name] = result
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id") or "0",
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        if not results:
            # 模型只作答未调工具时用规则补齐工具调用：决策数字必须来自工具，不采信模型叙述
            emit({"type": "thinking", "text": "大模型未调用工具，按问题意图补齐数据查询…"})
            results.update(self._run_with_rules(question, emit))

        return results, final_text

    def _run_with_rules(
        self, question: str, emit: EventCallback | None = None
    ) -> dict[str, dict[str, Any]]:
        """无 key / LLM 失败时的关键字规则降级。"""
        emit = emit or (lambda event: None)
        results: dict[str, dict[str, Any]] = {}

        if any(k in question for k in ("覆盖", "草", "绿", "植被", "长势")):
            results["grassland_coverage"] = self._dispatch_with_events(
                "grassland_coverage", {}, emit
            )
        if any(k in question for k in ("羊", "牛", "计数", "多少只", "数量", "几头")):
            results["sheep_count"] = self._dispatch_with_events("sheep_count", {}, emit)
        # 载畜量：问题带明确参数（含数字，如「亩产500公斤、100亩」）时按参数试算；
        # 否则交给系统按草场/全区聚合口径自动折算（link_stocking），与驾驶舱数字一致
        if any(k in question for k in ("载畜", "放牧", "能放", "容量", "养多少")) and re.search(
            r"\d", question
        ):
            results["carrying_capacity"] = self._dispatch_with_events(
                "carrying_capacity", {}, emit
            )
        # 草场口径：涉及草场/放牧/载畜量的问题查询草场状况（未命中具体草场时返回全部草场，逐场独立）
        if any(
            k in question
            for k in ("草场", "牧场", "分区", "照片", "亩", "载畜", "放牧", "能放", "容量", "养多少")
        ):
            results["pasture_status"] = self._dispatch_with_events(
                "pasture_status", {"pasture": question}, emit
            )

        return results

    # ------------------------------------------------------------------
    # 决策组装
    # ------------------------------------------------------------------

    def _build_decision(
        self,
        question: str,
        results: dict[str, dict[str, Any]],
        final_text: str | None,
    ) -> DecisionJSON:
        coverage_result = results.get("grassland_coverage") or {}
        coverage_val = coverage_result.get("coverage")
        area_mu = coverage_result.get("total_area_mu")

        # 草场口径优先（与草场分析视图/驾驶舱共用同一份结果）。**草场彼此独立**：
        # 1) 命中单一草场 → 用该草场自身的照片聚合覆盖度与面积核定；
        # 2) 只有全草场概览 → 不取覆盖度均值：等级取最严重、载畜量取各场之和（可加量）、
        #    覆盖度逐场列出。
        pasture_result = results.get("pasture_status") or {}
        pasture = pasture_result.get("pasture")
        pastures = pasture_result.get("pastures") or coverage_result.get("pastures") or []
        analyzed_pastures = [p for p in pastures if p.get("coverage") is not None]
        stocking_sum: float | None = None
        worst: dict[str, Any] | None = None
        if pasture and pasture.get("coverage") is not None:
            coverage_val = pasture["coverage"]
            area_mu = pasture.get("area_mu")
        elif analyzed_pastures:
            worst = max(
                analyzed_pastures,
                key=lambda p: LEVEL_SEVERITY.get(p.get("alert_level") or "normal", 0),
            )
            coverage_val = None  # 跨草场没有「一个覆盖度」，不给均值
            area_mu = round(sum(float(p.get("area_mu") or 0) for p in analyzed_pastures), 1)
            if all(p.get("recommended_stocking") is not None for p in analyzed_pastures):
                stocking_sum = round(
                    sum(float(p["recommended_stocking"]) for p in analyzed_pastures), 2
                )

        capacity = results.get("carrying_capacity") or {}
        linked = link_stocking(
            coverage_val,
            theoretical=float(capacity["recommended_stocking"]) if capacity else None,
            rotation=str(capacity.get("rotation_plan", "") or "") if capacity else None,
            area_mu=area_mu,
        )
        if stocking_sum is None and coverage_result.get("source") == "pastures":
            stocking_sum = coverage_result.get("recommended_stocking_sum")
        if stocking_sum is not None and not capacity:
            # 无参数试算时，建议载畜量直接取各草场建议值之和（与驾驶舱数字一致）
            linked["recommended_stocking"] = stocking_sum
        if worst is not None and not capacity:
            # 多草场状态汇总：最严重等级 + 逐场轮牧；理论值同为各场之和
            linked["alert_level"] = worst.get("alert_level") or "normal"
            linked["factor"] = None
            linked["theoretical"] = round(
                sum(float(p.get("theoretical") or 0) for p in analyzed_pastures), 2
            )
            linked["rotation_plan"] = rotation_for_pastures(analyzed_pastures)
            linked["source"] = "pastures"

        evidence = self._evidence_from_results(results)
        if linked["source"] is not None:
            if worst is not None and not capacity:
                # 多草场状态汇总（不做覆盖度平均）：等级取最严重，载畜量取各场之和
                evidence.append(
                    f"告警等级取各草场中最严重：{worst.get('code')} {worst.get('name')}"
                    f"（{_ALERT_LABEL.get(worst.get('alert_level'), worst.get('alert_level'))}，"
                    f"覆盖度 {float(worst.get('coverage') or 0):.1%}）"
                )
                if stocking_sum is not None:
                    evidence.append(
                        f"建议载畜量合计 {stocking_sum} 羊单位"
                        "（各草场按自身告警等级独立核定后求和，与驾驶舱口径一致）"
                    )
                if area_mu:
                    evidence.append(f"面积合计按 {area_mu} 亩核定（来自草场数据）")
            elif stocking_sum is not None and not capacity:
                # 全区口径：各草场按自身告警系数折减后汇总（与驾驶舱同一数字）
                evidence.append(
                    f"建议载畜量为各草场建议值之和 {stocking_sum} 羊单位"
                    "（各草场按自身告警等级折减后汇总，与驾驶舱口径一致）"
                )
                if area_mu:
                    evidence.append(f"面积合计按 {area_mu} 亩核定（来自草场数据）")
            else:
                evidence.append(
                    self._capacity_evidence(
                        linked["theoretical"], linked["factor"],
                        linked["recommended_stocking"], linked["alert_level"],
                    )
                )
                if linked["source"] == "coverage":
                    evidence.append(
                        f"亩产按覆盖度折算为 {coverage_linked_yield(coverage_val)} kg/亩"
                        f"（以 {REFERENCE_COVERAGE:.0%} 覆盖度为满产参照）"
                    )
                    if area_mu:
                        evidence.append(f"面积按 {area_mu} 亩核定（来自草场数据）")
        if not evidence:
            evidence.append("未调用到任何数据工具，无法给出量化依据。")

        actions = actions_for_level(linked["alert_level"])
        if final_text and _is_short_suggestion(final_text):
            actions.insert(0, final_text.strip())

        return DecisionJSON(
            alert_level=linked["alert_level"],
            recommended_actions=actions,
            recommended_stocking=linked["recommended_stocking"],
            rotation_plan=linked["rotation_plan"],
            evidence=evidence,
        )

    def _capacity_evidence(
        self, theoretical: float, factor: float, stocking: float, alert: str
    ) -> str:
        """载畜量的确定性依据说明（数字均有公式出处）。"""
        if factor == 1.0:
            return f"建议载畜量 {stocking} 羊单位"
        if factor == 0.0:
            return (
                f"理论载畜量 {theoretical} 羊单位；当前告警为{_ALERT_LABEL[alert]}，"
                f"建议禁牧，实际载畜 {stocking} 羊单位"
            )
        return (
            f"理论载畜量 {theoretical} 羊单位；按{_ALERT_LABEL[alert]}等级折减 {factor:.0%}"
            f" → 建议载畜量 {stocking} 羊单位"
        )

    def _evidence_from_results(self, results: dict[str, dict[str, Any]]) -> list[str]:
        evidence: list[str] = []
        pasture = (results.get("pasture_status") or {}).get("pasture")
        if pasture:
            # 字段容错：工具结果可能精简（只带关键字段），缺失时用默认值而非抛错
            evidence.append(
                f"「{pasture.get('name') or '草场'}」（{pasture.get('code') or '-'}）"
                f"{pasture.get('plot_count') or 0} 个分区 · "
                f"{pasture.get('image_count') or 0} 张编号照片，"
                f"{pasture.get('analyzed_plot_count') or 0} 个分区已分析"
            )
        if "grassland_coverage" in results:
            c = results["grassland_coverage"]
            if c.get("source") == "pastures":
                # 各草场独立：逐场列覆盖度，不给跨草场均值
                for p in c.get("pastures") or []:
                    level = _ALERT_LABEL.get(p.get("alert_level"), p.get("alert_level"))
                    evidence.append(
                        f"{p.get('code')} {p.get('name')} 覆盖度 "
                        f"{float(p.get('coverage') or 0):.2%}（{level}）"
                    )
            else:
                evidence.append(
                    f"草地覆盖度 {float(c.get('coverage', 0)):.2%}，"
                    f"平均绿度 {float(c.get('greenness', 0)):.2f}"
                )
        if "sheep_count" in results:
            evidence.append(f"羊群数量 {int(results['sheep_count'].get('count', 0))} 只")
        return evidence
