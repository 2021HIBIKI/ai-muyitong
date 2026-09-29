"""系统信息接口：运行时配置的安全只读子集（供前端「系统设置」页展示）。

只暴露对演示/排障有意义且**不含密钥**的字段：
- 阈值/巡检/视觉/大模型/视频/载畜量口径（模型只给文件名，不给路径）；
- LLM 是否已配置只给布尔值；数据存储只给表名与记录数（不给文件路径）；
- 绝不下发 GLM key、鉴权口令、令牌密钥、DB 路径。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter

from app.config import settings
from app.data.db import connect
from app.schemas import ok

router = APIRouter(prefix="/api/system", tags=["system"])

# 数据存储统计（表名 → 中文标签；只统计数量，不暴露路径）
_STORAGE_TABLES = {
    "pastures": "草场",
    "pasture_images": "草场照片",
    "grassland_metrics": "覆盖度记录",
    "alerts": "巡检告警",
    "decision_logs": "决策日志",
    "count_records": "羊群识别记录",
    "environment_records": "环境记录",
    "carrying_results": "载畜量计算",
    "users": "账号",
}


def _storage_stats() -> dict[str, dict[str, int | str]]:
    """各业务表记录数（老库缺表时跳过，不报错）。"""
    stats: dict[str, dict[str, int | str]] = {}
    conn = connect()
    try:
        for table, label in _STORAGE_TABLES.items():
            try:
                row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
                stats[table] = {"label": label, "count": int(row["n"])}
            except sqlite3.OperationalError:
                continue
    finally:
        conn.close()
    return stats


@router.get("/info")
def system_info() -> dict:
    """运行时配置摘要（受登录鉴权保护；不含任何密钥）。"""
    return ok(
        {
            "app_name": settings.app_name,
            "api_prefix": settings.api_prefix,
            "alert": {
                "warning_threshold": settings.alert_warning_threshold,
                "danger_threshold": settings.alert_danger_threshold,
            },
            "patrol": {
                "enabled": settings.patrol_enabled,
                "interval_seconds": settings.patrol_interval_seconds,
                "cooldown_seconds": settings.patrol_cooldown_seconds,
            },
            "vision": {
                "model_file": Path(settings.vision_model_path).name,
                "fallback_model": Path(settings.vision_fallback_model).name,
                "device": settings.vision_device,
                "confidence": settings.vision_confidence,
                "max_det": settings.vision_max_det,
            },
            "llm": {
                "model": settings.glm_model,
                "configured": bool(settings.glm_api_key),
                "max_retries": settings.llm_max_retries,
                "retry_backoff": settings.llm_retry_backoff,
            },
            "agent_max_rounds": settings.agent_max_rounds,
            "video": {
                "max_bytes": settings.video_max_bytes,
                "max_seconds": settings.video_max_seconds,
                "stride": settings.video_stride,
            },
            "daily_intake_per_sheep": settings.daily_intake_per_sheep,
            "storage": _storage_stats(),
        }
    )
