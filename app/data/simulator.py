"""模拟数据生成：温湿度/覆盖度/羊数时间序列，可入库用于大屏演示。

曲线均为固定种子的确定性随机序列（测试可复现）：
- 温湿度：昼夜正弦曲线 + 小噪声；
- 覆盖度：30 天缓慢波动（0.05~0.95）；草场版按「草场基准草情 × 分区」生成；
- 羊数：围绕基础规模小幅波动（非负整数）。

草场演示数据：编号/名称/照片来自 data/samples/grassland/manifest.json（单一来源），
面积等演示参数见 DEMO_PASTURE_META。
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
from datetime import datetime, timedelta, timezone

from app.config import BASE_DIR
from app.data import repository

DEFAULT_SEED = 42

# 草场样例清单（照片编号与实测覆盖度）；缺失时跳过草场播种
GRASSLAND_MANIFEST = BASE_DIR / "data" / "samples" / "grassland" / "manifest.json"

# 演示草场的补充参数（清单里没有的：区域、面积、备注）
DEMO_PASTURE_META: dict[str, dict] = {
    "P1": {"region": "东区", "area_mu": 120.0, "note": "灌溉条件好，草情常年良好"},
    "P2": {"region": "西区", "area_mu": 90.0, "note": "降水依赖型草场，草情中等"},
    "P3": {"region": "北坡", "area_mu": 60.0, "note": "坡地草场，本轮处于轮休观察期"},
}
DEFAULT_PASTURE_AREA_MU = 100.0


def _iso(ts: datetime) -> str:
    """ISO8601 UTC 毫秒（与建表默认时间格式一致）。"""
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _end_anchor(end: datetime | None) -> datetime:
    return end or datetime.now(timezone.utc)


def generate_environment(
    hours: int = 48, interval_hours: int = 1, seed: int = DEFAULT_SEED, end: datetime | None = None
) -> list[dict]:
    """生成温湿度时间序列：午后最高温、凌晨最低温，湿度与温度负相关。

    默认 1 小时一点（48 点 = 近 48 小时），时序曲线更细。
    """
    rng = random.Random(seed)
    anchor = _end_anchor(end)
    start = anchor - timedelta(hours=interval_hours * (hours - 1))
    out: list[dict] = []
    for i in range(hours):
        ts = start + timedelta(hours=interval_hours * i)
        phase = (ts.hour - 9) / 24 * 2 * math.pi  # 14 时左右峰值
        temperature = 16 + 8 * math.sin(phase) + rng.uniform(-1, 1)
        humidity = 55 - 20 * math.sin(phase) + rng.uniform(-3, 3)
        out.append(
            {
                "ts": _iso(ts),
                "temperature": round(temperature, 1),
                "humidity": round(min(max(humidity, 20), 95), 1),
            }
        )
    return out


def generate_coverage_series(
    days: int = 30, seed: int = DEFAULT_SEED, end: datetime | None = None
) -> list[dict]:
    """生成覆盖度/绿度日序列：缓慢波动 + 噪声，值域 0.05~0.95。"""
    rng = random.Random(seed)
    anchor = _end_anchor(end)
    start = anchor - timedelta(days=days - 1)
    out: list[dict] = []
    for i in range(days):
        ts = start + timedelta(days=i)
        base = 0.55 + 0.12 * math.sin(i / 5) + rng.uniform(-0.05, 0.05)
        coverage = round(min(max(base, 0.05), 0.95), 4)
        greenness = round(min(max(coverage * 0.25 + rng.uniform(-0.02, 0.02), 0.0), 1.0), 4)
        out.append({"ts": _iso(ts), "coverage": coverage, "greenness": greenness})
    return out


def generate_sheep_series(
    days: int = 30, seed: int = DEFAULT_SEED, end: datetime | None = None
) -> list[dict]:
    """生成羊数日序列：围绕基础规模小幅波动，非负整数。"""
    rng = random.Random(seed)
    anchor = _end_anchor(end)
    start = anchor - timedelta(days=days - 1)
    out: list[dict] = []
    for i in range(days):
        ts = start + timedelta(days=i)
        count = max(int(28 + 5 * math.sin(i / 4) + rng.randint(-4, 4)), 0)
        out.append({"ts": _iso(ts), "count": count})
    return out


# ---------------------------------------------------------------------------
# 草场（多草场管理）演示数据
# ---------------------------------------------------------------------------

def demo_pastures() -> list[dict]:
    """演示草场定义：编号/名称/编号照片来自样例清单，面积等为演示参数。

    清单缺失（如裁剪过的部署环境）时返回空列表，草场播种自动跳过。
    """
    if not GRASSLAND_MANIFEST.exists():
        return []
    try:
        manifest = json.loads(GRASSLAND_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    pastures: list[dict] = []
    for entry in manifest.get("pastures", []):
        meta = DEMO_PASTURE_META.get(entry.get("code"), {})
        expected = entry.get("expected") or {}
        images = []
        for image in entry.get("images", []):
            segments = str(image.get("code", "")).split("-")
            if len(segments) < 3:
                continue  # 编号需形如 草场-分区-序号
            images.append(
                {
                    "plot_code": segments[1],
                    "image_code": image["code"],
                    "image_path": f"/samples/grassland/{image.get('file') or image['code'] + '.jpg'}",
                }
            )
        if not images:
            continue
        pastures.append(
            {
                "code": entry["code"],
                "name": entry["name"],
                "region": meta.get("region"),
                "area_mu": float(meta.get("area_mu", DEFAULT_PASTURE_AREA_MU)),
                "note": meta.get("note"),
                "base": float(expected.get("mean_coverage") or 0.55),
                "images": images,
            }
        )
    return pastures


def generate_pasture_coverage_series(
    pasture: dict, days: int = 30, seed: int = DEFAULT_SEED, end: datetime | None = None
) -> list[dict]:
    """草场各分区照片的覆盖度日序列：围绕草场基准草情波动，分区/照片间略有差异。"""
    rng = random.Random(seed + sum(ord(c) for c in pasture["code"]))
    anchor = _end_anchor(end)
    start = anchor - timedelta(days=days - 1)
    out: list[dict] = []
    for i in range(days):
        ts = start + timedelta(days=i)
        for index, image in enumerate(pasture["images"]):
            base = pasture["base"] + 0.05 * math.sin(i / 5) - 0.02 * index + rng.uniform(-0.03, 0.03)
            value = round(min(max(base, 0.02), 0.98), 4)
            greenness = round(min(max(value * 0.25 + rng.uniform(-0.02, 0.02), 0.0), 1.0), 4)
            out.append(
                {
                    "ts": _iso(ts),
                    "plot_code": image["plot_code"],
                    "image_code": image["image_code"],
                    "image_path": image["image_path"],
                    "coverage": value,
                    "greenness": greenness,
                }
            )
    return out


def populate(
    conn: sqlite3.Connection,
    *,
    days: int = 30,
    hours: int = 48,
    seed: int = DEFAULT_SEED,
    region: str = "东区",
    end: datetime | None = None,
) -> dict:
    """将模拟序列写入三张业务表（含历史时间戳），返回写入条数。"""
    result = {
        **_populate_environment(conn, hours=hours, seed=seed, region=region, end=end),
        **_populate_coverage(conn, days=days, seed=seed, region=region, end=end),
        **_populate_sheep(conn, days=days, seed=seed, region=region, end=end),
    }
    conn.commit()
    return result


def _populate_environment(conn, *, hours, seed, region, end) -> dict:
    for row in generate_environment(hours=hours, seed=seed, end=end):
        repository.insert_environment_record(
            conn, region=region, temperature=row["temperature"], humidity=row["humidity"],
            created_at=row["ts"],
        )
    return {"environment": hours}


def _populate_coverage(conn, *, days, seed, region, end) -> dict:
    for row in generate_coverage_series(days=days, seed=seed, end=end):
        repository.insert_grassland_metric(
            conn, region=region, coverage=row["coverage"], greenness=row["greenness"],
            created_at=row["ts"],
        )
    return {"coverage": days}


def _populate_sheep(conn, *, days, seed, region, end) -> dict:
    for row in generate_sheep_series(days=days, seed=seed, end=end):
        repository.insert_count_record(
            conn, region=region, count=row["count"],
            per_class={"sheep": row["count"]}, annotated_image=None, created_at=row["ts"],
        )
    return {"sheep": days}


def _populate_pastures(conn, *, days, seed, end) -> dict:
    """写入演示草场及其编号照片（编号/名称/照片来自样例清单）。"""
    count = 0
    for pasture in demo_pastures():
        pasture_id = repository.insert_pasture(
            conn,
            code=pasture["code"],
            name=pasture["name"],
            region=pasture["region"],
            area_mu=pasture["area_mu"],
            note=pasture["note"],
        )
        for image in pasture["images"]:
            repository.insert_pasture_image(
                conn,
                pasture_id=pasture_id,
                plot_code=image["plot_code"],
                image_code=image["image_code"],
                image_path=image["image_path"],
            )
        count += 1
    return {"pastures": count} if count else {}


def _populate_pasture_metrics(conn, *, days, seed, end) -> dict:
    """为「尚无覆盖度记录」的草场播种分区照片日序列（老库升级后同样补齐）。"""
    total = 0
    demo_by_code = {p["code"]: p for p in demo_pastures()}
    for row in repository.list_pastures(conn):
        if repository.list_metrics_for_pasture(conn, row["id"], limit=1):
            continue  # 已有数据（含真实分析结果）的草场不覆盖
        images = repository.list_pasture_images(conn, row["id"])
        if not images:
            continue
        spec = {
            "code": row["code"],
            "base": float(demo_by_code.get(row["code"], {}).get("base", 0.55)),
            "images": [
                {
                    "plot_code": image["plot_code"],
                    "image_code": image["image_code"],
                    "image_path": image["image_path"],
                }
                for image in images
            ],
        }
        for point in generate_pasture_coverage_series(spec, days=days, seed=seed, end=end):
            repository.insert_grassland_metric(
                conn,
                region=row["code"],
                coverage=point["coverage"],
                greenness=point["greenness"],
                created_at=point["ts"],
                pasture_id=row["id"],
                plot_code=point["plot_code"],
                image_code=point["image_code"],
                image_path=point["image_path"],
            )
            total += 1
    return {"pasture_metrics": total} if total else {}


def _populate_decisions(conn, *, end=None) -> dict:
    """为决策日志播种演示问答（含思考事件），供大屏决策流与历史对话演示。

    事件与决策由编排器的确定性函数生成（工具结果解读、告警系数联动、措施文案），
    与真实问答的输出保持一致。
    """
    from app.agent.orchestrator import actions_for_level, analyze_tool_result, link_stocking
    from app.data.pasture_service import pasture_summary
    from app.vision.grassland import carrying_capacity, coverage_linked_yield

    summaries = [
        pasture_summary(conn, row) for row in repository.list_pastures(conn)
    ]
    analyzed = [s for s in summaries if s["coverage"] is not None]
    if not analyzed:
        return {}

    counts = repository.list_count_records(conn, limit=1)
    sheep_count = counts[0]["count"] if counts else 28
    anchor = _end_anchor(end)
    seeded = 0

    for index, summary in enumerate(analyzed):
        coverage = summary["coverage"]
        coverage_result = {"coverage": coverage, "greenness": summary["greenness"]}
        sheep_result = {"count": sheep_count, "per_class": {"sheep": sheep_count}}
        capacity_result = {
            "recommended_stocking": summary["theoretical"],
            "rotation_plan": carrying_capacity(
                coverage_linked_yield(coverage), float(summary["area_mu"] or 0), 0.5, 30.0
            )["rotation_plan"],
        }
        linked = link_stocking(
            coverage,
            theoretical=summary["theoretical"],
            rotation=capacity_result["rotation_plan"],
        )
        decision = {
            "alert_level": linked["alert_level"],
            "recommended_actions": actions_for_level(linked["alert_level"]),
            "recommended_stocking": linked["recommended_stocking"],
            "rotation_plan": linked["rotation_plan"],
            "evidence": [
                f"草地覆盖度 {coverage:.2%}，平均绿度 {float(summary['greenness'] or 0):.2f}",
                f"羊群数量 {sheep_count} 只",
                f"理论载畜量 {summary['theoretical']} 羊单位（按草场面积 {summary['area_mu']} 亩核定）",
            ],
        }

        events: list[dict] = [
            {
                "type": "thinking",
                "text": f"分析问题：先读取「{summary['name']}」各分区照片的覆盖度，"
                "再结合羊群数量评估草畜平衡…",
                "ts": 180,
            },
        ]
        # 与真实 SSE 流同序：tool_call → tool_result → 确定性解读（analysis）
        steps = [
            ("grassland_coverage", coverage_result, {}, 1900),
            ("sheep_count", sheep_result, {}, 6100),
            (
                "carrying_capacity",
                capacity_result,
                {"area": summary["area_mu"], "grazing_days": 30},
                6190,
            ),
        ]
        for name, result, arguments, result_ts in steps:
            events.append(
                {
                    "type": "tool_call",
                    "name": name,
                    "arguments": arguments,
                    "ts": max(result_ts - 130, 200),
                }
            )
            events.append({"type": "tool_result", "name": name, "result": result, "ts": result_ts})
            note = analyze_tool_result(name, result)
            if note:
                events.append(
                    {"type": "analysis", "name": name, "text": note, "ts": result_ts + 10}
                )
        events.append({"type": "answer", "decision": decision, "ts": 6300})

        repository.insert_decision_log(
            conn,
            question=f"{summary['name']}草情如何？能放多少羊？",
            tool_calls=["grassland_coverage", "sheep_count", "carrying_capacity"],
            decision_json=decision,
            events=events,
            created_at=_iso(anchor - timedelta(hours=(len(analyzed) - index) * 3)),
        )
        seeded += 1

    return {"decisions": seeded} if seeded else {}


def table_empty(conn: sqlite3.Connection, table: str) -> bool:
    """指定表是否为空（表名仅限代码内常量，不接收外部输入）。"""
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def _decision_events_missing(conn: sqlite3.Connection) -> bool:
    """是否还没有带思考事件的决策记录（老库升级后补齐演示历史对话）。"""
    return (
        conn.execute(
            "SELECT COUNT(*) FROM decision_logs WHERE events IS NOT NULL"
        ).fetchone()[0]
        == 0
    )


def populate_if_empty(
    conn: sqlite3.Connection,
    *,
    days: int = 30,
    hours: int = 48,
    seed: int = DEFAULT_SEED,
    region: str = "东区",
    end: datetime | None = None,
) -> dict | None:
    """逐表/逐草场判断：仅对空表与无数据的草场播种模拟数据。

    草场覆盖度按「草场 → 分区照片」播种（不再播种未关联草场的全局序列），
    大屏与草场管理共用同一份按草场聚合的口径。
    """
    seeded: dict = {}
    if table_empty(conn, "environment_records"):
        seeded.update(_populate_environment(conn, hours=hours, seed=seed, region=region, end=end))
    if table_empty(conn, "pastures"):
        seeded.update(_populate_pastures(conn, days=days, seed=seed, end=end))
    seeded.update(_populate_pasture_metrics(conn, days=days, seed=seed, end=end))
    if table_empty(conn, "count_records"):
        seeded.update(_populate_sheep(conn, days=days, seed=seed, region=region, end=end))
    if _decision_events_missing(conn):
        seeded.update(_populate_decisions(conn, end=end))
    if not seeded:
        return None
    conn.commit()
    return seeded
