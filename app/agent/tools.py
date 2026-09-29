"""Agent 工具注册表：草地覆盖度 / 草场状况 / 羊群计数 / 载畜量 / 历史查询。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from app.config import BASE_DIR
from app.data import repository
from app.data.db import connect
from app.data.pasture_service import pasture_summary, plots_detail
from app.vision.counter import count_sheep
from app.vision.detector import SheepDetector
from app.vision.grassland import carrying_capacity, coverage

# 默认样例图：库中尚无草场数据时用于演示覆盖度/计数。
DEFAULT_SAMPLE_IMAGE = str(BASE_DIR / "data" / "samples" / "mauron_crop_example.png")

# 模块级检测器（惰性加载）；测试中可替换。
detector = SheepDetector()


def _load_image(path: str | None) -> np.ndarray:
    p = path or DEFAULT_SAMPLE_IMAGE
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {p}")
    return img


def _match_pasture(rows, keyword: str):
    """按编号或名称定位草场（名称支持包含匹配，供自由文本问题使用）。"""
    key = keyword.strip()
    if not key:
        return None
    upper = key.upper()
    for row in rows:
        if row["code"].upper() == upper:
            return row
    for row in rows:
        name = row["name"] or ""
        if name and name in key:
            return row
        if row["code"] and row["code"].upper() in upper:
            return row
    return None


def tool_pasture_status(pasture: str | None = None) -> dict[str, Any]:
    """查询草场状况：一个草场由多个分区的编号照片构成，本草场状况由各分区聚合成。

    pasture 可为草场编号（如 P1）或名称（如「北坡轮休草场」），也接受包含它们的
    自由文本（如整句问题）；缺省/未匹配到时返回全部草场概览。
    """
    conn = connect()
    try:
        rows = repository.list_pastures(conn)
        if not rows:
            return {"error": "库中尚无草场数据"}
        target = _match_pasture(rows, pasture or "") if pasture else None
        if target is not None:
            summary = pasture_summary(conn, target)
            summary["plots"] = [
                {
                    "plot_code": plot["plot_code"],
                    "coverage": plot["coverage"],
                    "alert_level": plot["alert_level"],
                    "image_codes": [i["image_code"] for i in plot["images"]],
                }
                for plot in plots_detail(conn, target)
            ]
            return {"pasture": summary}
        summaries = [pasture_summary(conn, row) for row in rows]
        payload: dict[str, Any] = {
            "count": len(summaries),
            "matched": target is not None,
            "pastures": [
                {
                    "code": s["code"],
                    "name": s["name"],
                    "area_mu": s["area_mu"],
                    "coverage": s["coverage"],
                    "alert_level": s["alert_level"],
                    "plot_count": s["plot_count"],
                    "analyzed_plot_count": s["analyzed_plot_count"],
                    "recommended_stocking": s["recommended_stocking"],
                    "worst_plot": s["worst_plot"],
                }
                for s in summaries
            ],
        }
        if pasture and target is None:
            payload["note"] = f"未匹配到草场「{pasture}」，以下为全部草场概览"
        return payload
    finally:
        conn.close()


def tool_grassland_coverage(image_path: str | None = None) -> dict[str, Any]:
    """分析草场图像，返回草地覆盖度与平均绿度。

    未指定图像时返回**各草场独立**的覆盖度列表（草场是独立管理单元：跨草场不做
    覆盖度平均，仅汇总建议载畜量等可加量）；库中尚无草场分析数据时退回样例图
    分析（兼容未建立草场的库）。
    """
    if image_path:
        return coverage(_load_image(image_path))

    conn = connect()
    try:
        summaries = [pasture_summary(conn, row) for row in repository.list_pastures(conn)]
    finally:
        conn.close()
    analyzed = [s for s in summaries if s["coverage"] is not None]
    if not analyzed:
        return coverage(_load_image(None))

    return {
        "source": "pastures",
        "pasture_count": len(summaries),
        "analyzed_pasture_count": len(analyzed),
        "total_area_mu": round(sum(float(s["area_mu"] or 0) for s in analyzed), 1),
        # 各草场建议载畜量之和（可加量：各场独立核定后求和，与驾驶舱同一口径）
        "recommended_stocking_sum": round(
            sum(float(s["recommended_stocking"]) for s in analyzed), 2
        ),
        "pastures": [
            {
                "code": s["code"],
                "name": s["name"],
                "area_mu": s["area_mu"],
                "coverage": s["coverage"],
                "greenness": s["greenness"],
                "alert_level": s["alert_level"],
                "theoretical": s["theoretical"],
                "recommended_stocking": s["recommended_stocking"],
            }
            for s in analyzed
        ],
    }


def tool_sheep_count(image_path: str | None = None) -> dict[str, Any]:
    """识别并计数草场图像中的羊群数量。"""
    img = _load_image(image_path)
    return count_sheep(detector.detect(img))


def tool_carrying_capacity(
    yield_per_area: float = 800.0,
    area: float = 100.0,
    utilization: float = 0.5,
    grazing_days: float = 30.0,
    daily_intake: float | None = None,
) -> dict[str, Any]:
    """按 NY/T 635 公式计算合理载畜量（羊单位）。"""
    return carrying_capacity(
        yield_per_area=yield_per_area,
        area=area,
        utilization=utilization,
        grazing_days=grazing_days,
        daily_intake=daily_intake,
    )


def tool_query_history(entity: str = "count", limit: int = 10) -> dict[str, Any]:
    """查询历史记录（羊群计数 / 覆盖度 / 载畜量 / 决策）。"""
    entity = (entity or "count").lower()
    conn = connect()
    try:
        if entity in ("count", "count_records", "sheep"):
            rows = repository.list_count_records(conn, limit)
        elif entity in ("coverage", "grassland", "metric"):
            rows = repository.list_grassland_metrics(conn, limit)
        elif entity in ("carrying", "capacity"):
            rows = repository.list_carrying_results(conn, limit)
        elif entity in ("decision", "agent", "log"):
            rows = repository.list_decision_logs(conn, limit)
        else:
            rows = []
        records = [dict(r) for r in rows]
    finally:
        conn.close()
    return {"entity": entity, "count": len(records), "records": records}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., dict[str, Any]]


TOOLS: dict[str, Tool] = {
    "grassland_coverage": Tool(
        name="grassland_coverage",
        description=(
            "获取草地覆盖度（0-1）与平均绿度 GLI：不指定图像时返回各草场**独立**的覆盖度"
            "（草场之间不做平均），指定 image_path 时分析该图像"
        ),
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "草场图像路径，缺省返回各草场独立覆盖度"}
            },
        },
        func=tool_grassland_coverage,
    ),
    "pasture_status": Tool(
        name="pasture_status",
        description=(
            "查询草场状况（本草场各分区编号照片聚合的覆盖度/告警等级/建议载畜量/"
            "多模态草情评分与分区明细）；草场由多个分区照片组成，编号形如 P1-A-01"
        ),
        parameters={
            "type": "object",
            "properties": {
                "pasture": {
                    "type": "string",
                    "description": "草场编号（如 P1）或名称（如「北坡轮休草场」），缺省返回全部草场",
                }
            },
        },
        func=tool_pasture_status,
    ),
    "sheep_count": Tool(
        name="sheep_count",
        description="识别并计数草场图像中的羊群数量",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "草场图像路径，缺省使用样例图"}
            },
        },
        func=tool_sheep_count,
    ),
    "carrying_capacity": Tool(
        name="carrying_capacity",
        description="按 NY/T 635 公式计算合理载畜量（羊单位）；仅在用户给出明确参数时用于参数试算",
        parameters={
            "type": "object",
            "properties": {
                "yield_per_area": {"type": "number", "description": "亩均可食产量 kg/亩"},
                "area": {"type": "number", "description": "草场面积 亩"},
                "utilization": {"type": "number", "description": "牧草利用率 0-1（用户未给出时不要传，缺省 0.5）"},
                "grazing_days": {"type": "number", "description": "放牧天数（用户未给出时不要传，缺省 30）"},
            },
            "required": ["yield_per_area", "area"],
        },
        func=tool_carrying_capacity,
    ),
    "query_history": Tool(
        name="query_history",
        description="查询历史记录（羊群计数 / 草地覆盖度 / 载畜量 / 决策日志）",
        parameters={
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "enum": ["count", "coverage", "carrying", "decision"],
                    "description": "实体类型",
                },
                "limit": {"type": "integer", "description": "返回条数"},
            },
        },
        func=tool_query_history,
    ),
}


def tool_schemas() -> list[dict[str, Any]]:
    """返回 OpenAI function calling 格式的工具清单。"""
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in TOOLS.values()
    ]


def dispatch_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """按名称分发工具调用。未知工具 / 执行异常均返回错误字典而非抛错。"""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return tool.func(**(arguments or {}))
    except Exception as e:
        return {"error": str(e)}
