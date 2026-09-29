"""覆盖度时序预测测试：Holt 线性趋势、置信带、边界与坏输入。"""
from __future__ import annotations

import pytest

from app.data.forecast import holt_linear


def test_linear_series_extrapolates_slope():
    """线性序列 0.2→0.35（斜率 0.05/步）：一步预测 ≈ 0.40，残差≈0。"""
    result = holt_linear([0.20, 0.25, 0.30, 0.35], horizon=1)
    assert result["trend"] == pytest.approx(0.05, abs=1e-6)
    assert result["forecast"][0] == pytest.approx(0.40, abs=0.002)
    assert result["residual_std"] == pytest.approx(0.0, abs=1e-6)


def test_constant_series_has_zero_trend():
    """常数序列：趋势≈0，预测≈常数，置信带≈0。"""
    result = holt_linear([0.5, 0.5, 0.5, 0.5], horizon=3)
    assert result["trend"] == pytest.approx(0.0, abs=1e-6)
    for v in result["forecast"]:
        assert v == pytest.approx(0.5, abs=0.001)
    assert max(result["band"]) == pytest.approx(0.0, abs=1e-6)


def test_band_widens_with_horizon():
    """置信带随预测步长加宽：越远期越不确定。"""
    noisy = [0.45, 0.52, 0.44, 0.55, 0.47, 0.53]
    result = holt_linear(noisy, horizon=4)
    for i in range(1, len(result["band"])):
        assert result["band"][i] > result["band"][i - 1]
    assert result["residual_std"] > 0.01  # 噪声序列残差非零


def test_forecast_clipped_to_physical_bounds():
    """覆盖度物理上 ∈ [0,1]：陡峭上升/下降序列的预测值被截断。"""
    up = holt_linear([0.2, 0.5, 0.8, 0.95], horizon=3)
    assert max(up["forecast"]) <= 1.0
    down = holt_linear([0.8, 0.5, 0.2, 0.05], horizon=3)
    assert min(down["forecast"]) >= 0.0


def test_deterministic_and_self_contained():
    """同输入两次运行结果一致；返回字段完整。"""
    values = [0.4, 0.42, 0.45, 0.41, 0.46]
    a = holt_linear(values, horizon=2)
    b = holt_linear(values, horizon=2)
    assert a == b
    assert set(a) == {"forecast", "level", "trend", "band", "residual_std", "fitted"}
    assert len(a["forecast"]) == 2 and len(a["band"]) == 2
    assert len(a["fitted"]) == len(values)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        holt_linear([], horizon=3)
    with pytest.raises(ValueError):
        holt_linear([0.5], horizon=3)  # 少于 2 点
    with pytest.raises(ValueError):
        holt_linear([0.4, 0.5], horizon=0)
    with pytest.raises(ValueError):
        holt_linear([0.4, float("nan")], horizon=2)
