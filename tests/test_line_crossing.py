"""阶段 6.2：越线统计测试 —— 合成轨迹（上/下、抖动、折返），不碰 cv2 / torch。"""
from __future__ import annotations

from app.vision.line_crossing import LineCrossingCounter, line_from_ratio
from app.vision.track import TrackedDetection

LINE = ((0.0, 100.0), (300.0, 100.0))  # 从左到右的水平线 y=100


def _tracked(track_id: int, cx: float, cy: float, size: float = 40.0) -> TrackedDetection:
    return TrackedDetection(
        track_id=track_id,
        bbox=(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2),
        class_id=18,
        class_name="sheep",
        confidence=0.9,
    )


def _run(ys: list[float], track_id: int = 1, **kwargs) -> LineCrossingCounter:
    counter = LineCrossingCounter(*LINE, **kwargs)
    for y in ys:
        counter.update([_tracked(track_id, 100.0, y)])
    return counter


def test_crossing_down_counts_one_outbound():
    """自下向上/自上向下的方向语义：从上走到下 = 1 次 outbound。"""
    result = LineCrossingCounter(*LINE).update([_tracked(1, 100.0, 50.0)])
    counter = _run([50.0, 50.0, 150.0, 150.0])
    assert counter.update([]).outbound == 1  # 需连续 2 帧位于另一侧
    assert counter.update([]).inbound == 0
    assert counter.update([]).net == -1
    assert result.inbound == 0 and result.outbound == 0  # 首帧在线一侧，未穿越


def test_crossing_up_counts_one_inbound():
    counter = _run([150.0, 150.0, 50.0, 50.0])
    result = counter.update([])
    assert result.inbound == 1
    assert result.outbound == 0
    assert result.net == 1


def test_jitter_on_line_counts_nothing():
    """贴线抖动（±3 / ±15 像素来回）不计数。"""
    for amplitude in (3.0, 6.0, 15.0):
        counter = _run([100.0 - amplitude, 100.0 + amplitude] * 4)
        result = counter.update([])
        assert result.inbound == 0 and result.outbound == 0, amplitude
        assert result.net == 0


def test_crossing_window_suppresses_edge_flapping():
    """判定窗口（默认 3 帧）：左右横跳式贴线抖动不计入任何方向。"""
    counter = _run([97.0, 103.0, 97.0, 103.0, 97.0, 103.0])
    result = counter.update([])
    assert result.inbound == 0 and result.outbound == 0


def test_crossing_window_counts_real_crossings():
    """真实穿越（连续多帧行进到另一侧）在窗口满足后计 1 次。"""
    counter = _run([40.0, 40.0, 60.0, 160.0, 160.0])
    result = counter.update([])
    assert result.outbound == 1
    assert result.inbound == 0


def test_cross_then_return_then_cross_again_counts_each_direction_once():
    """穿越后折返再穿越：每 ID 每方向只计一次（门禁：再进计 0 次）。"""
    counter = _run([50.0, 50.0, 150.0, 150.0, 50.0, 50.0])
    after_return = counter.update([])
    assert after_return.inbound == 1
    assert after_return.outbound == 1

    counter.update([_tracked(1, 100.0, 150.0)])
    result = counter.update([_tracked(1, 100.0, 150.0)])
    assert result.outbound == 1  # 同一 ID 同方向不重复计入
    assert result.inbound == 1
    assert result.net == 0


def test_dedup_can_be_disabled():
    """关闭去重时每次真实穿越都累加（对照口径）。"""
    counter = _run(
        [50.0, 50.0, 150.0, 150.0, 50.0, 50.0], dedup_per_direction=False
    )
    assert counter.update([]).outbound == 1
    counter.update([_tracked(1, 100.0, 150.0)])
    result = counter.update([_tracked(1, 100.0, 150.0)])
    assert result.outbound == 2  # 真实折返：再次出去再计一次
    assert result.inbound == 1


def test_net_equals_in_minus_out():
    counter = LineCrossingCounter(*LINE)
    for track_id, ys in ((1, [50.0, 50.0, 150.0, 150.0]), (2, [150.0, 150.0, 50.0, 50.0])):
        for y in ys:
            counter.update([_tracked(track_id, 100.0, y)])
    result = counter.update([])
    assert (result.inbound, result.outbound) == (1, 1)
    assert result.net == 0


def test_empty_frame_and_reset_are_safe():
    counter = _run([50.0, 50.0, 150.0, 150.0])
    assert counter.update([]).outbound == 1
    counter.reset()
    result = counter.update([])
    assert (result.inbound, result.outbound) == (0, 0)  # 重置后重新计数


def test_crossing_ids_reported():
    """本帧穿越事件带出 track_id（前端高亮用）。"""
    counter = LineCrossingCounter(*LINE)
    counter.update([_tracked(7, 100.0, 50.0), _tracked(9, 200.0, 150.0)])
    counter.update([_tracked(7, 100.0, 150.0), _tracked(9, 200.0, 150.0)])
    result = counter.update([_tracked(7, 100.0, 160.0), _tracked(9, 200.0, 150.0)])
    assert result.crossed_out == (7,)
    assert result.crossed_in == ()


def test_line_from_ratio_scales_to_frame():
    start, end = line_from_ratio(720, 1280, 0.0, 0.5, 1.0, 0.5)
    assert start == (0.0, 640.0)
    assert end == (720.0, 640.0)


def test_line_descriptor():
    counter = LineCrossingCounter((1.0, 2.0), (3.0, 4.0))
    assert counter.line() == {"start": [1.0, 2.0], "end": [3.0, 4.0]}
