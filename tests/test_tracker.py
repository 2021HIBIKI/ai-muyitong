"""阶段 6.1：跟踪适配层测试 —— 脚本化检测序列，不碰 cv2 / torch。"""
from __future__ import annotations

import pytest

from app.vision.detector import Detection
from app.vision.track import ObjectTracker


def _det(x1, y1, x2, y2, conf=0.9, class_id=18, class_name="sheep"):
    return Detection((x1, y1, x2, y2), class_id, class_name, conf)


def _box_at(cx, cy, size=40):
    return _det(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def test_same_object_keeps_track_id_across_frames():
    """同目标连续多帧（含小幅位移）ID 稳定。"""
    tracker = ObjectTracker()
    ids = []
    for step in range(8):
        tracked = tracker.update([_box_at(100 + step * 6, 200 + step * 3)])
        assert len(tracked) == 1
        ids.append(tracked[0].track_id)
    assert len(set(ids)) == 1  # 全程同一个 ID
    assert all(isinstance(i, int) and i > 0 for i in ids)


def test_two_objects_get_distinct_ids():
    tracker = ObjectTracker()
    tracked = tracker.update([_box_at(100, 200), _box_at(500, 200)])
    assert len(tracked) == 2
    assert len({t.track_id for t in tracked}) == 2


def test_object_reappearing_after_long_absence_gets_new_id():
    """消失超过 lost_track_buffer 后重新出现：分配新 ID（不把新目标算成老目标）。"""
    tracker = ObjectTracker(frame_rate=30, lost_track_buffer=5)  # 5 帧后轨迹被移除
    first = tracker.update([_box_at(100, 200)])
    assert len(first) == 1
    first_id = first[0].track_id

    for _ in range(10):  # 空帧若干轮（超期）
        tracker.update([])

    # 新轨迹：建立帧不出现在输出（ByteTrack 需连续 2 帧确认）
    assert tracker.update([_box_at(100, 200)]) == []
    again = tracker.update([_box_at(100, 200)])
    assert len(again) == 1
    assert again[0].track_id != first_id


def test_new_track_needs_two_consecutive_frames():
    """轨迹确认机制：跟踪器第 1 帧立即输出，此后的新目标从第 2 帧起进入输出。

    逐帧计数因此最多滞后 1 帧，对「同帧最大并发」主指标无影响。
    """
    tracker = ObjectTracker()
    assert len(tracker.update([_box_at(100, 200)])) == 1  # 第 1 帧：立即确认
    assert tracker.update([_box_at(600, 200)]) == []  # 后来的新目标：建立轨迹
    assert len(tracker.update([_box_at(600, 200)])) == 1  # 第 2 帧起输出


def test_short_gap_keeps_id():
    """短暂漏检（未超期）后 ID 保持。"""
    tracker = ObjectTracker(frame_rate=30, lost_track_buffer=30)
    first_id = tracker.update([_box_at(100, 200)])[0].track_id
    tracker.update([])
    tracker.update([])
    again = tracker.update([_box_at(104, 202)])
    assert again[0].track_id == first_id


def test_empty_frame_does_not_raise():
    tracker = ObjectTracker()
    assert tracker.update([]) == []
    assert tracker.update([]) == []


def test_class_name_and_centroid_preserved():
    tracker = ObjectTracker()
    tracked = tracker.update([_det(10, 20, 50, 80, class_id=18, class_name="sheep")])
    assert tracked[0].class_name == "sheep"
    assert tracked[0].class_id == 18
    assert tracked[0].centroid == (30.0, 50.0)


def test_reset_clears_state():
    """reset() 后跟踪器重建：计数从头开始（换视频不串号）。"""
    tracker = ObjectTracker()
    tracker.update([_box_at(100, 200)])
    assert tracker._tracker is not None

    tracker.reset()
    assert tracker._tracker is None  # 惰性重建
    tracked = tracker.update([_box_at(100, 200), _box_at(400, 200)])
    assert len(tracked) == 2
    # 重建后新 ID 从 1 起（不与上一段视频的 ID 延续）
    assert min(t.track_id for t in tracked) == 1


def test_lazy_construction():
    """构造时不应引入 supervision（import 期无重依赖）。"""
    tracker = ObjectTracker()
    assert tracker._tracker is None


def test_unsupported_confidence_missing_not_applicable():
    """置信度由 Detection 强制提供（数据集契约），此处仅确认异常路径可读。"""
    with pytest.raises(TypeError):
        Detection((0, 0, 1, 1), 18, "sheep")  # 缺少 confidence
