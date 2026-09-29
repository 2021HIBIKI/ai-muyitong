"""阶段 2：草地覆盖度 + 载畜量测试。"""
from __future__ import annotations

import numpy as np
import pytest

from app.vision.grassland import (
    carrying_capacity,
    coverage,
    coverage_linked_yield,
    exg,
    exg_standardised,
    exg_standardised_hue,
    gli,
)

GREEN = (0, 160, 0)      # BGR 纯绿（hue=60、brightness=160，落在植被阈值内）
SOIL = (128, 128, 128)   # BGR 中灰（饱和度为 0，判为非植被）


def _img(w=200, h=100):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_exg_and_exg_standardised_shapes():
    img = _img()
    assert exg(img).shape == img.shape[:2]
    assert exg_standardised(img).shape == img.shape[:2]


def test_exg_standardised_hue_mask():
    img = _img()
    mask = exg_standardised_hue(img)
    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.uint8


def test_coverage_pure_green():
    img = _img()
    img[:] = GREEN
    result = coverage(img)
    assert result["coverage"] == pytest.approx(1.0, abs=0.05)
    assert result["greenness"] > 0.5  # 纯绿 GLI 趋近 1


def test_coverage_pure_soil():
    img = _img()
    img[:] = SOIL
    result = coverage(img)
    assert result["coverage"] == pytest.approx(0.0, abs=0.05)


def test_coverage_half_green_half_soil():
    img = _img()
    img[:, :100] = GREEN
    img[:, 100:] = SOIL
    result = coverage(img)
    assert result["coverage"] == pytest.approx(0.5, abs=0.05)


def test_gli_green_positive_and_grey_zero():
    green = _img()
    green[:] = GREEN
    assert gli(green).mean() == pytest.approx(1.0, abs=0.05)

    grey = _img()
    grey[:] = SOIL
    assert gli(grey).mean() == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# 载畜量（NY/T 635）
# ---------------------------------------------------------------------------

def test_carrying_capacity_hand_calc():
    """产量 1000kg/亩 × 100 亩 × 0.5 利用率 ÷ (2kg × 100 天) = 250 羊单位。"""
    result = carrying_capacity(
        yield_per_area=1000.0,
        area=100.0,
        utilization=0.5,
        grazing_days=100.0,
        daily_intake=2.0,
    )
    assert result["recommended_stocking"] == pytest.approx(250.0)
    assert isinstance(result["rotation_plan"], str)
    assert result["rotation_plan"]


def test_carrying_capacity_uses_default_intake():
    """未传 daily_intake 时采用配置默认 2.0。"""
    result = carrying_capacity(
        yield_per_area=400.0,
        area=10.0,
        utilization=1.0,
        grazing_days=10.0,
    )
    # 400*10*1 / (2*10) = 200
    assert result["recommended_stocking"] == pytest.approx(200.0)


def test_carrying_capacity_zero_area_no_error():
    result = carrying_capacity(1000.0, area=0.0, utilization=0.5, grazing_days=100.0)
    assert result["recommended_stocking"] == 0.0


def test_carrying_capacity_zero_days_no_error():
    result = carrying_capacity(1000.0, area=100.0, utilization=0.5, grazing_days=0.0)
    assert result["recommended_stocking"] == 0.0


def test_carrying_capacity_utilization_clamped():
    lo = carrying_capacity(1000.0, 100.0, utilization=-0.5, grazing_days=100.0)
    hi = carrying_capacity(1000.0, 100.0, utilization=2.0, grazing_days=100.0)
    assert lo["recommended_stocking"] == 0.0
    # 利用率 2.0 截断为 1.0：1000*100*1/(2*100) = 500
    assert hi["recommended_stocking"] == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# 覆盖度联动折算亩产（阶段 5）
# ---------------------------------------------------------------------------

def test_coverage_linked_yield():
    assert coverage_linked_yield(None) == 800.0  # 未知覆盖度兜底满产
    assert coverage_linked_yield(0.60) == 800.0  # 参照覆盖度满产
    assert coverage_linked_yield(0.90) == 800.0  # 超过参照封顶
    assert coverage_linked_yield(0.30) == 400.0  # 等比例折算
    assert coverage_linked_yield(0.00) == 0.0


def test_coverage_linked_yield_monotonic_capacity():
    """覆盖度越低 → 折算亩产越低 → 载畜量越低。"""
    low = carrying_capacity(coverage_linked_yield(0.30), 100.0, 0.5, 30.0)
    mid = carrying_capacity(coverage_linked_yield(0.45), 100.0, 0.5, 30.0)
    high = carrying_capacity(coverage_linked_yield(0.60), 100.0, 0.5, 30.0)
    assert low["recommended_stocking"] < mid["recommended_stocking"] < high["recommended_stocking"]


# ---------------------------------------------------------------------------
# 多模态融合草情评分
# ---------------------------------------------------------------------------

def test_comfort_score_ranges_and_missing():
    """温湿度适宜度：舒适区间满分、两端线性衰减、缺项返回 None。"""
    from app.vision.grassland import comfort_score

    assert comfort_score(20, 50) == 1.0
    assert comfort_score(12, 75) == 1.0  # 边界含
    assert comfort_score(0, None) == 0.0
    assert comfort_score(40, None) == 0.0
    assert comfort_score(6, None) == 0.5  # 线性爬升中点
    assert comfort_score(30, 50) == 0.8333  # 温度 30℃（0.667）+ 湿度 50%（1.0）的均值
    assert comfort_score(None, 50) == 1.0  # 只有湿度：满分
    assert comfort_score(None, None) is None


def test_grass_health_score_weights_and_normalization():
    """加权融合：覆盖度 60% / 绿度 15% / 环境 25%；缺项自动归一化。"""
    from app.vision.grassland import grass_health_score

    full = grass_health_score(0.60, 0.50, 20, 50)
    assert full["score"] == 92.5  # 0.6×100 + 0.15×50 + 0.25×100
    assert full["contributions"] == {"coverage": 100.0, "greenness": 50.0, "environment": 100.0}

    only_coverage = grass_health_score(0.60, None, None, None)
    assert only_coverage["score"] == 100.0  # 权重归一化到 1 → 单项即总分
    assert only_coverage["missing"] == ["greenness", "environment"]

    none = grass_health_score(None, None, None, None)
    assert none["score"] is None and none["contributions"] == {}


def test_grass_health_score_monotonic_and_aligned_with_alert():
    """覆盖度越高评分越高；评分等级与全站告警口径一致（不另立判定）。"""
    from app.vision.grassland import grass_health_score

    low = grass_health_score(0.30, 0.40, 20, 50)
    mid = grass_health_score(0.50, 0.40, 20, 50)
    high = grass_health_score(0.70, 0.40, 20, 50)
    assert low["score"] < mid["score"] < high["score"]
    assert low["level"] == "danger"
    assert mid["level"] == "warning"
    assert high["level"] == "normal"
    # 覆盖度缺失时等级为 normal（与 alert_level_from_coverage(None) 一致）
    assert grass_health_score(None, 0.9, 20, 50)["level"] == "normal"


def test_grass_health_score_bad_input_clipped():
    """绿度越界按 [0,1] 截断；覆盖度越界按参照值封顶。"""
    from app.vision.grassland import grass_health_score

    assert grass_health_score(0.9, 1.5, 20, 50)["contributions"]["greenness"] == 100.0
    assert grass_health_score(0.9, -0.5, 20, 50)["contributions"]["greenness"] == 0.0
    assert grass_health_score(2.0, None, None, None)["contributions"]["coverage"] == 100.0
