"""阶段 1.2：计数去重测试 —— 唯一 ID 计数、重复框去重。"""
from __future__ import annotations

import pytest

from app.vision.counter import count_sheep
from app.vision.detector import Detection


def _det(cls=18, name="sheep"):
    return Detection((0, 0, 10, 10), cls, name, 0.9)


def test_count_without_track_ids():
    dets = [_det() for _ in range(3)]
    result = count_sheep(dets)
    assert result["count"] == 3
    assert result["per_class"] == {"sheep": 3}


def test_count_unique_track_ids_dedup():
    """同一只羊（相同 track id）跨帧重复检测只计一次。"""
    dets = [_det() for _ in range(4)]
    ids = [1, 1, 2, 3]
    result = count_sheep(dets, ids)
    assert result["count"] == 3
    assert result["per_class"] == {"sheep": 3}


def test_count_per_class():
    dets = [
        Detection((0, 0, 1, 1), 18, "sheep", 0.9),
        Detection((0, 0, 1, 1), 18, "sheep", 0.9),
        Detection((0, 0, 1, 1), 19, "cow", 0.8),
    ]
    result = count_sheep(dets)
    assert result["count"] == 3
    assert result["per_class"] == {"sheep": 2, "cow": 1}


def test_count_untracked_none_ids():
    """track_id 为 None 的框按独立个体计。"""
    dets = [_det() for _ in range(3)]
    ids = [1, None, 2]
    result = count_sheep(dets, ids)
    assert result["count"] == 3


def test_count_length_mismatch_raises():
    dets = [_det() for _ in range(2)]
    with pytest.raises(ValueError):
        count_sheep(dets, [1, 2, 3])
