"""巡检路由：手动触发一轮巡检（演示现场 100% 可控，不依赖定时器）。"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.agent import patrol
from app.data.db import connect
from app.schemas import err, ok

router = APIRouter(prefix="/api/patrol", tags=["patrol"])


@router.post("/run")
async def run_patrol_endpoint():
    """手动巡检一轮：**逐草场**采集 → 评估 → （需告警时）落库并推送到大屏。

    返回本轮每个草场的结果（含未告警的草场，供大屏逐场展示）。同步工作（视觉/DB）
    在线程池中执行，不阻塞事件循环；广播在事件循环内进行。
    """
    def _round() -> dict:
        conn = connect()
        try:
            return patrol.run_patrol(conn, advice_fn=patrol.patrol_advice)
        finally:
            conn.close()

    try:
        payload = await asyncio.to_thread(_round)
    except Exception as e:  # 采集失败（如巡检图片缺失）也应给出可读错误
        return err(500, f"巡检执行失败：{e}")

    for alert in payload.get("alerts") or []:  # 逐场告警分别推送
        patrol.alert_hub.publish(alert)
    return ok(payload)
