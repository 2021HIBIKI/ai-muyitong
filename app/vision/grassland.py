"""草地覆盖度与载畜量：ExG+HSV 植被分割 + GLI 绿度 + NY/T 635 载畜量。

`exg()` / `exg_standardised()` / `exg_standardised_hue()` 移植自 OpenWeedLocator
（geezacoleman/OpenWeedLocator，MIT License，utils/algorithms.py），算法参数保持一致，
仅去掉与本项目无关的显示/检测逻辑。GLI 与载畜量公式为本项目实现。
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from app.config import settings


# ---------------------------------------------------------------------------
# 植被分割（OpenWeedLocator 移植，MIT）
# ---------------------------------------------------------------------------

def exg(image: np.ndarray) -> np.ndarray:
    """Excess Green：ExG = 2G - R - B，输出灰度图（uint8）。"""
    blue, green, red = cv2.split(image)
    blue = blue.astype(np.float32)
    green = green.astype(np.float32)
    red = red.astype(np.float32)

    out = 2 * green - red - blue
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def exg_standardised(image: np.ndarray) -> np.ndarray:
    """标准化 ExG：(2G-R-B)/(R+G+B)，缩放至 0-255（uint8）。"""
    blue, green, red = cv2.split(image)
    blue = blue.astype(np.float32)
    green = green.astype(np.float32)
    red = red.astype(np.float32)

    channel_sum = red + green + blue
    np.maximum(channel_sum, 1.0, out=channel_sum)

    out = 2 * green - red - blue
    out /= channel_sum
    out *= 255.0
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def exg_standardised_hue(
    image: np.ndarray,
    hue_min: int = 30,
    hue_max: int = 90,
    brightness_min: int = 10,
    brightness_max: int = 220,
    saturation_min: int = 30,
    saturation_max: int = 255,
    invert_hue: bool = False,
) -> np.ndarray:
    """ExG + HSV 双阈值植被分割。

    返回掩膜：0 = 非植被，>0 = 植被（值为标准化 ExG）。绿色像素需同时满足
    HSV（hue/saturation/brightness）区间与 ExG 门限。
    """
    blue, green, red = cv2.split(image)
    blue = blue.astype(np.float32)
    green = green.astype(np.float32)
    red = red.astype(np.float32)

    channel_sum = red + green + blue
    np.maximum(channel_sum, 1.0, out=channel_sum)

    out = 2 * green - red - blue
    out /= channel_sum
    out *= 255.0
    np.clip(out, 0, 255, out=out)
    image_out = out.astype(np.uint8)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if not invert_hue:
        lower = np.array([hue_min, saturation_min, brightness_min], dtype=np.uint8)
        upper = np.array([hue_max, saturation_max, brightness_max], dtype=np.uint8)
        hsv_thresh = cv2.inRange(hsv_image, lower, upper)
    else:
        lower1 = np.array([0, saturation_min, brightness_min], dtype=np.uint8)
        upper1 = np.array([hue_min, saturation_max, brightness_max], dtype=np.uint8)
        lower2 = np.array([hue_max, saturation_min, brightness_min], dtype=np.uint8)
        upper2 = np.array([180, saturation_max, brightness_max], dtype=np.uint8)
        mask1 = cv2.inRange(hsv_image, lower1, upper1)
        mask2 = cv2.inRange(hsv_image, lower2, upper2)
        hsv_thresh = cv2.bitwise_or(mask1, mask2)

    return hsv_thresh & image_out


def gli(image: np.ndarray) -> np.ndarray:
    """Green Leaf Index：GLI = (2G - R - B) / (2G + R + B)，输出 float32（[-1, 1]）。"""
    blue = image[:, :, 0].astype(np.float32)
    green = image[:, :, 1].astype(np.float32)
    red = image[:, :, 2].astype(np.float32)

    num = 2 * green - red - blue
    den = 2 * green + red + blue
    den = np.maximum(den, 1e-6)
    return num / den


# ---------------------------------------------------------------------------
# 覆盖度
# ---------------------------------------------------------------------------

def coverage(image: np.ndarray, **kwargs) -> dict[str, float]:
    """计算草地覆盖度与平均绿度。

    coverage：植被（绿色）像素占比，0-1；
    greenness：植被区域的平均 GLI（[-1, 1]），无植被时为 0。
    """
    mask = exg_standardised_hue(image, **kwargs)
    veg = mask > 0
    coverage_ratio = float(veg.mean())
    greenness = float(gli(image)[veg].mean()) if veg.any() else 0.0
    return {"coverage": coverage_ratio, "greenness": greenness}


# ---------------------------------------------------------------------------
# 载畜量（NY/T 635）
# ---------------------------------------------------------------------------

def carrying_capacity(
    yield_per_area: float,
    area: float,
    utilization: float,
    grazing_days: float,
    daily_intake: float | None = None,
) -> dict:
    """NY/T 635 载畜量：载畜量 = 产量 × 利用率 ÷ (日食量 × 放牧天数)。

    yield_per_area：亩均可食产量（kg/亩）；area：面积（亩）；
    utilization：利用率（0-1，超界自动截断）；grazing_days：放牧天数；
    daily_intake：日食量（kg/羊单位·天），缺省取配置 daily_intake_per_sheep。

    返回 {"recommended_stocking": 羊单位, "rotation_plan": 轮牧建议文本}。
    0 面积 / 0 天数等边界不抛错，载畜量按 0 处理。
    """
    intake = daily_intake if daily_intake is not None else settings.daily_intake_per_sheep
    if intake <= 0:
        intake = settings.daily_intake_per_sheep

    total_yield = max(0.0, yield_per_area) * max(0.0, area)
    usable_yield = total_yield * min(max(utilization, 0.0), 1.0)

    if grazing_days > 0:
        sheep_units = usable_yield / (intake * grazing_days)
    else:
        sheep_units = 0.0  # 0 天数边界：不抛错

    return {
        "recommended_stocking": round(sheep_units, 2),
        "rotation_plan": _rotation_plan(sheep_units, area, grazing_days),
    }


def _rotation_plan(sheep_units: float, area: float, grazing_days: float) -> str:
    """生成确定性轮牧建议文本（1 牛 ≈ 5 羊单位）。"""
    if area > 0 and grazing_days > 0:
        cow_units = sheep_units / 5.0
        # 每小区放牧约 7 天，预留 1 个休牧小区
        paddocks = max(2, math.ceil(grazing_days / 7.0) + 1)
        return (
            f"建议载畜 {sheep_units:.1f} 羊单位（约 {cow_units:.1f} 头牛）。"
            f"将草场划分为 {paddocks} 个轮牧小区，每区放牧约 7 天、休牧约 {7 * (paddocks - 1)} 天，"
            f"共放牧 {grazing_days:.0f} 天，避免过度啃食、促进牧草恢复。"
        )
    return "载畜量为 0：草场面积或放牧天数不足，请先确认草场面积与放牧周期。"


# ---------------------------------------------------------------------------
# 覆盖度 ↔ 载畜量联动（本项目实现，确定性折算规则）
# ---------------------------------------------------------------------------

DEFAULT_MAX_YIELD = 800.0  # 参照覆盖度下的亩均可食产量（kg/亩）
REFERENCE_COVERAGE = 0.60  # 正常阈值（与告警阈值一致）作为满产参照覆盖度


def coverage_linked_yield(
    coverage: float | None,
    max_yield: float = DEFAULT_MAX_YIELD,
    reference_coverage: float = REFERENCE_COVERAGE,
) -> float:
    """按覆盖度折算亩均可食产量：以 60% 覆盖度为满产参照线性折算。

    覆盖度 ≥ 参照值 → 满产 max_yield；低于则等比例下降（覆盖度越低产草越少）；
    覆盖度未知（None）→ 取 max_yield 兜底。
    """
    if coverage is None:
        return round(max_yield, 1)
    ratio = min(max(coverage, 0.0) / reference_coverage, 1.0)
    return round(max_yield * ratio, 1)


# ---------------------------------------------------------------------------
# 告警等级与草场聚合（全站唯一口径，编排器/大屏/草场管理共用）
#
# 口径边界：**草场内**（多个分区/多张编号照片 → 一个草场）做聚合；
# **草场之间彼此独立**，只做状态汇总（最严重等级 / 载畜量求和），不做覆盖度平均。
# ---------------------------------------------------------------------------

def alert_level_from_coverage(coverage: float | None) -> str:
    """覆盖度 → 告警等级（确定性阈值规则，阈值取自配置）。

    ≥warning 阈值 → normal；[danger, warning) → warning；<danger 阈值 → danger；
    覆盖度未知（未分析）→ normal（不臆断告警）。
    """
    if coverage is None:
        return "normal"
    if coverage >= settings.alert_warning_threshold:
        return "normal"
    if coverage >= settings.alert_danger_threshold:
        return "warning"
    return "danger"


# 等级严重度（跨草场只做「状态汇总」，绝不做覆盖度平均——各草场是独立管理单元）
LEVEL_SEVERITY: dict[str, int] = {"normal": 0, "warning": 1, "danger": 2}


def worst_level(levels) -> str | None:
    """多个等级里最严重的一个；无有效等级返回 None（无数据不臆断）。"""
    known = [level for level in levels if level in LEVEL_SEVERITY]
    if not known:
        return None
    return max(known, key=lambda level: LEVEL_SEVERITY[level])


def pasture_aggregate(plot_coverages: list[float]) -> dict:
    """草场状况（**本草场内部**聚合）：各分区最新覆盖度等权平均（确定性，可复现）。

    一个草场由多个分区的照片集合构成，草场覆盖度取各分区最新分析值的均值；
    告警等级按草场覆盖度走统一阈值口径（草场之间彼此独立，不做跨草场平均）。无任何已分析分区时 coverage=None
    （草场「未分析」），不臆断为正常或危险。

    返回 {"coverage", "alert_level", "plot_count", "min_coverage", "max_coverage"}。
    """
    values = [float(v) for v in plot_coverages if v is not None]
    if not values:
        return {
            "coverage": None,
            "alert_level": "normal",
            "plot_count": 0,
            "min_coverage": None,
            "max_coverage": None,
        }
    mean = sum(values) / len(values)
    return {
        "coverage": round(mean, 4),
        "alert_level": alert_level_from_coverage(mean),
        "plot_count": len(values),
        "min_coverage": round(min(values), 4),
        "max_coverage": round(max(values), 4),
    }


def plot_coverage_map(metric_rows) -> dict[str, float]:
    """覆盖度记录 → {分区编号: 分区覆盖度}（「图片集合 → 草场」的归并口径）。

    两级归并（确定性）：同一分区内每张照片取各自最新一次分析值，分区值 = 这些
    照片最新值的均值；没有图片级记录的分区退回其分区级最新值兜底。
    入参行需含 plot_code / image_code / coverage / id 字段（dict 或 sqlite3.Row）。
    """
    latest_image: dict[tuple[str, str], tuple[int, float]] = {}
    latest_plot: dict[str, tuple[int, float]] = {}
    for row in metric_rows:
        plot_code = row["plot_code"]
        coverage = row["coverage"]
        if plot_code is None or coverage is None:
            continue
        record_id = int(row["id"])
        image_code = row["image_code"]
        if image_code:
            key = (plot_code, image_code)
            if key not in latest_image or record_id > latest_image[key][0]:
                latest_image[key] = (record_id, float(coverage))
        elif plot_code not in latest_plot or record_id > latest_plot[plot_code][0]:
            latest_plot[plot_code] = (record_id, float(coverage))

    grouped: dict[str, list[float]] = {}
    for (plot_code, _image), (_mid, value) in latest_image.items():
        grouped.setdefault(plot_code, []).append(value)

    values = {plot: sum(vals) / len(vals) for plot, vals in grouped.items()}
    for plot, (_mid, value) in latest_plot.items():
        values.setdefault(plot, value)  # 仅当该分区无图片级记录时兜底
    return values


# 建议载畜量 = 理论载畜量 × 告警调整系数（确定性：正常不折减、预警折减、危险禁牧）
ALERT_CAPACITY_FACTOR: dict[str, float] = {"normal": 1.0, "warning": 0.6, "danger": 0.0}


def pasture_recommendation(
    coverage: float | None,
    area_mu: float,
    utilization: float = 0.5,
    grazing_days: float = 30.0,
) -> dict:
    """草场级载畜量建议（确定性）：按草场自有面积计算理论值，再乘告警系数。

    理论载畜量 = 覆盖度折算亩产 × 面积 × 利用率 ÷ (日食量 × 放牧天数)（NY/T 635）；
    建议载畜量 = 理论值 × 告警系数（正常 1.0 / 预警 0.6 / 危险 0，即禁牧）。
    覆盖度未知（未分析）时不臆断，理论值与建议值均取 0。
    """
    if coverage is None or area_mu <= 0:
        theoretical = 0.0
    else:
        yield_per_mu = coverage_linked_yield(coverage)
        theoretical = carrying_capacity(yield_per_mu, area_mu, utilization, grazing_days)[
            "recommended_stocking"
        ]
    level = alert_level_from_coverage(coverage)
    factor = ALERT_CAPACITY_FACTOR[level]
    return {
        "alert_level": level,
        "factor": factor,
        "theoretical": theoretical,
        "recommended_stocking": round(theoretical * factor, 2),
    }


# ---------------------------------------------------------------------------
# 多模态融合草情评分（图像 + 环境）：确定性加权，缺项自动归一化权重
# ---------------------------------------------------------------------------

def comfort_score(temperature: float | None = None, humidity: float | None = None) -> float | None:
    """温湿度适宜度（0–1）：分段线性（与概览「适宜」口径一致）。

    温度：12–25℃ 满分；0℃→12℃ 线性爬升；25℃→40℃ 线性衰减（两端外截断）。
    湿度：35–75% 满分；10%→35% 线性爬升；75%→100% 线性衰减。缺项返回 None。
    """
    if temperature is None and humidity is None:
        return None
    scores = []
    if temperature is not None:
        t = float(temperature)
        t_score = 1.0 if 12 <= t <= 25 else max(0.0, min(1.0, t / 12.0)) if t < 12 else max(0.0, min(1.0, (40 - t) / 15.0))
        scores.append(t_score)
    if humidity is not None:
        h = float(humidity)
        h_score = 1.0 if 35 <= h <= 75 else max(0.0, min(1.0, (h - 10) / 25.0)) if h < 35 else max(0.0, min(1.0, (100 - h) / 25.0))
        scores.append(h_score)
    return round(sum(scores) / len(scores), 4)


def grass_health_score(
    coverage: float | None = None,
    greenness: float | None = None,
    temperature: float | None = None,
    humidity: float | None = None,
) -> dict:
    """多模态融合草情评分（0–100，确定性）。

    各模态的单项分（0–100）：
    - 覆盖度：min(coverage / 0.6, 1) × 100（60% 为满产参照，与折算亩产同一口径）
    - 绿度：max(GLI, 0) × 100（GLI ∈ [-1, 1]，负值按 0）
    - 环境：comfort_score × 100（温湿度适宜度，缺项时该项不参与）

    权重：覆盖度 60% / 绿度 15% / 环境 25%；**缺失模态自动按剩余权重归一化**。
    等级仍与全站唯一口径一致（``alert_level_from_coverage``）——评分是补充量化
    信息（展示与趋势预警辅助），不改变告警判定。

    返回 {"score", "level", "contributions": {来源: 单项分}, "missing": [...]}。
    """
    weights = {"coverage": 0.6, "greenness": 0.15, "environment": 0.25}
    parts: dict[str, float] = {}
    if coverage is not None:
        parts["coverage"] = min(max(float(coverage), 0.0) / REFERENCE_COVERAGE, 1.0) * 100
    if greenness is not None:
        parts["greenness"] = max(min(float(greenness), 1.0), 0.0) * 100
    comfort = comfort_score(temperature, humidity)
    if comfort is not None:
        parts["environment"] = comfort * 100

    active = {k: w for k, w in weights.items() if k in parts}
    total_weight = sum(active.values())
    if not active:
        return {"score": None, "level": alert_level_from_coverage(coverage),
                "contributions": {}, "missing": list(weights)}
    score = sum(parts[k] * (active[k] / total_weight) for k in active)

    return {
        "score": round(score, 1),
        "level": alert_level_from_coverage(coverage),
        "contributions": {k: round(v, 1) for k, v in parts.items()},
        "missing": [k for k in weights if k not in parts],
    }
