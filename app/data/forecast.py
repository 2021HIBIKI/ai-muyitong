"""覆盖度时序预测：Holt 线性趋势外推 + 残差置信带。

对草场按日覆盖度序列做 Holt 双参数指数平滑（水平 + 趋势），外推未来 ``horizon``
天的预测值，并给出随步长加宽的置信带（基于一步预测残差的标准差）。

纯手写实现，无第三方依赖；全流程确定性、可复现、可单测。

口径（如实说明）：覆盖度受季节、放牧强度、气象等因素影响，Holt 只是短期外推
工具——本模块在系统中定位为**趋势预警辅助**：置信带随预测步长加宽，表达
「越远期越不确定」。预测值按覆盖度的物理边界截断到 [0, 1]。
"""
from __future__ import annotations

import math

# 置信带系数：1.96 对应 95% 置信区间（正态假设）
BAND_FACTOR = 1.96


def holt_linear(
    values: list[float],
    horizon: int,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> dict:
    """Holt 线性趋势外推。

    ``values``：等间隔时序（至少 2 个点，均为数值）；``horizon``：预测步数（≥1）；
    ``alpha``/``beta``：水平与趋势平滑系数（0–1）。

    返回 ``{"forecast": 预测值列表(长度 horizon), "level": 末期水平,
    "trend": 末期趋势, "band": 各步置信半宽, "residual_std": 一步残差标准差,
    "fitted": 一步拟合值}``。预测值截断到 [0, 1]。
    """
    if not values or len(values) < 2:
        raise ValueError("至少需要 2 个观测点才能做趋势外推")
    if horizon < 1:
        raise ValueError(f"预测步数需 ≥1，收到 {horizon}")
    series = [float(v) for v in values]
    if not all(math.isfinite(v) for v in series):
        raise ValueError("序列包含非有限值")

    level = series[0]
    trend = series[1] - series[0]
    fitted = [series[0]]
    for y in series[1:]:
        prev_level, prev_trend = level, trend
        fitted.append(prev_level + prev_trend)
        level = alpha * y + (1 - alpha) * (prev_level + prev_trend)
        trend = beta * (level - prev_level) + (1 - beta) * prev_trend

    residuals = [y - f for y, f in zip(series[1:], fitted[1:])]
    variance = sum(r * r for r in residuals) / (len(residuals) - 1) if len(residuals) > 1 else 0.0
    residual_std = math.sqrt(max(variance, 0.0))

    forecast, band = [], []
    for step in range(1, horizon + 1):
        forecast.append(max(0.0, min(1.0, level + step * trend)))  # 物理边界 [0,1]
        band.append(round(BAND_FACTOR * residual_std * math.sqrt(step), 4))

    return {
        "forecast": [round(v, 4) for v in forecast],
        "level": round(level, 4),
        "trend": round(trend, 6),
        "band": band,
        "residual_std": round(residual_std, 4),
        "fitted": [round(v, 4) for v in fitted],
    }
