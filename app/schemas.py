"""统一响应与 Pydantic 数据模型。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


def ok(data: Any = None) -> dict[str, Any]:
    """成功响应统一包裹。"""
    return {"code": 0, "data": data, "message": "ok"}


def err(code: int, message: str) -> dict[str, Any]:
    """错误响应统一包裹（code 非 0）。"""
    return {"code": code, "data": None, "message": message}


class CarryingCapacityRequest(BaseModel):
    """载畜量计算入参（NY/T 635）。边界值 0 允许，由计算函数兜底处理。"""

    yield_per_area: float = Field(..., ge=0, description="亩均可食产量（kg/亩）")
    area: float = Field(..., ge=0, description="草场面积（亩）")
    utilization: float = Field(0.5, ge=0, le=1, description="牧草利用率（0-1）")
    grazing_days: float = Field(..., ge=0, description="放牧天数")
    daily_intake: float | None = Field(None, gt=0, description="日食量（kg/羊单位·天）")


class PastureCreate(BaseModel):
    """新建草场入参（编号唯一；照片通过 /api/pastures/{id}/images 按编号登记）。"""

    code: str = Field(..., min_length=1, max_length=20, description="草场编号（唯一，如 P4）")
    name: str = Field(..., min_length=1, max_length=50, description="草场名称")
    region: str | None = Field(None, max_length=50, description="所在区域/位置")
    area_mu: float = Field(0.0, ge=0, description="草场面积（亩）")
    note: str | None = Field(None, max_length=200, description="备注")
