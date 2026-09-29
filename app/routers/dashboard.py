"""数据大屏路由：概览 / 环境 / 趋势 / 决策流。

草场彼此独立（跨草场不做覆盖度平均）：概览 KPI 逐场返回，等级取各场中最严重、
载畜量取各场之和；库中无已分析草场时回退最近一条覆盖度记录（单点口径，兼容旧库）。
"""
from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter, Query
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.agent.orchestrator import link_stocking
from app.data import repository
from app.data.db import connect
from app.data.pasture_service import pasture_summary
from app.schemas import err, ok
from app.vision.grassland import LEVEL_SEVERITY

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


def _row_dict(row: sqlite3.Row) -> dict:
    return dict(row)


def _coverage_kpis(conn) -> dict:
    """草场 KPI（**各草场独立**：跨草场不做覆盖度平均）。

    - ``pastures``：每个草场各自的编号照片聚合状况（一个草场内部才做分区/照片聚合）；
    - ``alert_level``：各草场独立判定后取**最严重**等级（附该草场），不是覆盖度均值判定；
    - ``recommended_stocking`` / ``theoretical``：各草场独立核定后求和（可加量）；
    - 库中无已分析草场时回退最近一条覆盖度记录（单点口径 ``fallback``，仍非跨草场平均）。
    """
    summaries = [pasture_summary(conn, p) for p in repository.list_pastures(conn)]
    analyzed = [s for s in summaries if s["coverage"] is not None]
    pastures = [
        {
            "id": s["id"],
            "code": s["code"],
            "name": s["name"],
            "area_mu": s["area_mu"],
            "coverage": s["coverage"],
            "alert_level": s["alert_level"],
            "greenness": s["greenness"],
            "plot_count": s["plot_count"],
            "analyzed_plot_count": s["analyzed_plot_count"],
            "image_count": s["image_count"],
            "theoretical": s["theoretical"],
            "recommended_stocking": s["recommended_stocking"],
            "updated_at": s["updated_at"],
        }
        for s in summaries
    ]
    if analyzed:
        worst = max(analyzed, key=lambda s: LEVEL_SEVERITY.get(s["alert_level"] or "normal", 0))
        return {
            "pastures": pastures,
            "pasture_count": len(summaries),
            "analyzed_pasture_count": len(analyzed),
            "alert_level": worst["alert_level"],
            "alert_pasture": {
                "code": worst["code"],
                "name": worst["name"],
                "coverage": worst["coverage"],
                "alert_level": worst["alert_level"],
            },
            "recommended_stocking": round(
                sum(float(s["recommended_stocking"]) for s in analyzed), 2
            ),
            "theoretical": round(sum(float(s["theoretical"]) for s in analyzed), 2),
            "fallback": None,
        }

    metrics = repository.list_grassland_metrics(conn, limit=1)
    carries = repository.list_carrying_results(conn, limit=1)
    linked = link_stocking(
        metrics[0]["coverage"] if metrics else None,
        theoretical=float(carries[0]["recommended_stocking"]) if carries else None,
        rotation=carries[0]["rotation_plan"] if carries else None,
    )
    return {
        "pastures": pastures,
        "pasture_count": len(summaries),
        "analyzed_pasture_count": 0,
        "alert_level": linked["alert_level"] if metrics else None,
        "alert_pasture": None,
        "recommended_stocking": linked["recommended_stocking"],
        "theoretical": linked["theoretical"],
        "fallback": {
            "coverage": metrics[0]["coverage"] if metrics else None,
            "greenness": metrics[0]["greenness"] if metrics else None,
            "rotation_plan": linked["rotation_plan"] or None,
        },
    }


@router.get("/overview")
def overview():
    """大屏顶部 KPI：最新羊数、覆盖度、告警等级、建议载畜量、温湿度。"""
    try:
        conn = connect()
        try:
            counts = repository.list_count_records(conn, limit=1)
            envs = repository.list_environment_records(conn, limit=1)
            kpis = _coverage_kpis(conn)
            return ok(
                {
                    "sheep_count": counts[0]["count"] if counts else 0,
                    "temperature": envs[0]["temperature"] if envs else None,
                    "humidity": envs[0]["humidity"] if envs else None,
                    **kpis,
                }
            )
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))


@router.get("/environment")
def environment(limit: int = Query(48, ge=1, le=168)):
    """近 N 条环境记录（升序，供折线图）。"""
    try:
        conn = connect()
        try:
            rows = repository.list_environment_records(conn, limit=limit, ascending=True)
            return ok({"records": [_row_dict(r) for r in rows]})
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))


@router.get("/trends")
def trends(days: int = Query(30, ge=1, le=365)):
    """覆盖度趋势（**各草场独立**：每个草场一条按日序列）与羊数趋势（最近 days 天，升序）。

    覆盖度不做跨草场平均——各草场是独立管理单元；计数记录中的标注图（base64 大字段）
    不下发，避免响应体积膨胀。
    """
    try:
        conn = connect()
        try:
            pastures = []
            for pasture in repository.list_pastures(conn):
                rows = repository.daily_coverage_series(
                    conn, days=days, pasture_id=pasture["id"]
                )
                pastures.append(
                    {
                        "id": pasture["id"],
                        "code": pasture["code"],
                        "name": pasture["name"],
                        "coverage": [
                            {
                                "day": r["day"],
                                "coverage": round(r["coverage"], 4)
                                if r["coverage"] is not None
                                else None,
                                "greenness": round(r["greenness"], 4)
                                if r["greenness"] is not None
                                else None,
                                "n": r["n"],
                            }
                            for r in rows
                        ],
                    }
                )
            counts = repository.list_count_records(conn, limit=days, ascending=True)
            sheep = [
                {
                    "created_at": r["created_at"],
                    "region": r["region"],
                    "count": r["count"],
                    "per_class": json.loads(r["per_class"]) if r["per_class"] else None,
                }
                for r in counts
            ]
            return ok({"pastures": pastures, "sheep": sheep})
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))


@router.get("/decision-flow")
def decision_flow(limit: int = Query(10, ge=1, le=100)):
    """Agent 决策历史（供大屏决策流面板）。"""
    try:
        conn = connect()
        try:
            rows = repository.list_decision_logs(conn, limit=limit)
            logs = []
            for r in reversed(rows):  # 时间正序展示
                decision = {}
                try:
                    decision = json.loads(r["decision_json"] or "{}")
                except json.JSONDecodeError:
                    pass
                logs.append(
                    {
                        "id": r["id"],  # 供大屏决策流点击跳转到对应回答
                        "created_at": r["created_at"],
                        "question": r["question"],
                        "tool_calls": json.loads(r["tool_calls"] or "[]"),
                        "alert_level": decision.get("alert_level"),
                        "recommended_stocking": decision.get("recommended_stocking"),
                        "recommended_actions": decision.get("recommended_actions", []),
                    }
                )
            return ok({"logs": logs})
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))


# ---------------------------------------------------------------------------
# 阶段 7：巡检告警（列表 / 确认 / SSE 实时推送）
# ---------------------------------------------------------------------------

@router.get("/alerts")
def alerts_endpoint(limit: int = Query(50, ge=1, le=200)):
    """告警历史（最新在前）——大屏告警面板与 5 秒轮询兜底共用。"""
    try:
        from app.agent import patrol

        conn = connect()
        try:
            rows = repository.list_alerts(conn, limit=limit)
            return ok({"alerts": [patrol.alert_payload(r) for r in rows]})
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))


@router.post("/alerts/{alert_id}/ack")
def acknowledge_alert_endpoint(alert_id: int):
    """确认告警（大屏点「已知悉」）。"""
    try:
        from app.agent import patrol

        conn = connect()
        try:
            if not repository.acknowledge_alert(conn, alert_id):
                return err(404, f"告警不存在：{alert_id}")
            conn.commit()
            return ok(patrol.alert_payload(repository.get_alert(conn, alert_id)))
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))


def alert_event_stream():
    """SSE 事件生成器工厂（模块级名字：便于测试 monkeypatch 成有限生成器）。

    - 订阅/退订通过 AlertHub；断开清理放在生成器的 ``finally``；
    - 只推新事件（历史由 /alerts 拉取），事件名 ``alert``，载荷为告警 JSON。
    """

    async def _generator():
        from app.agent import patrol

        hub = patrol.alert_hub
        queue = hub.subscribe()
        try:
            while True:
                payload = await queue.get()
                yield ServerSentEvent(event="alert", data=payload, retry=5000)
        finally:
            hub.unsubscribe(queue)

    return _generator()


@router.get("/alerts/stream", response_class=EventSourceResponse)
async def alerts_stream_endpoint():
    """告警实时推送（SSE）：事件名 ``alert``；编码由 FastAPI 路由层完成。

    ``EventSourceResponse`` 必须作为 ``response_class`` 用在「本身是 async 生成器」
    的路由上（由路由层把 ServerSentEvent 编码成 text/event-stream 并带 15s 心跳），
    不能手工构造 Response(content=生成器) —— 那样会绕过编码层。
    大屏另以 5 秒轮询 /alerts 作为永久兜底。
    """
    async for event in alert_event_stream():
        yield event
