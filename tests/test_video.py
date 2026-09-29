"""阶段 6.3：视频处理核心测试 —— 假帧源（帧内容即帧号）+ 假检测函数，不碰 cv2/torch。"""
from __future__ import annotations

from app.vision.detector import Detection
from app.vision.line_crossing import LineCrossingCounter
from app.vision.track import ObjectTracker
from app.vision.video import process_frames


def _det(cx: float, cy: float, class_id: int = 18, class_name: str = "sheep", size: float = 60.0) -> Detection:
    return Detection((cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2), class_id, class_name, 0.9)


def _scripted(script: dict[int, list[Detection]]):
    """帧号 → 该帧检测结果；帧内容直接用帧号（video.py 不关心帧的具体类型）。"""

    def detect(frame):
        return script.get(frame, [])

    return detect


def _detect_calls(script):
    calls = []

    def detect(frame):
        calls.append(frame)
        return script.get(frame, [])

    return detect, calls


def test_three_metrics_and_frame_series():
    """三条口径：同帧最大并发 / 累计出现过 / 逐帧序列长度。"""
    script = {
        **{i: [_det(100, 100)] for i in range(20)},  # A：全程
        **{i: [_det(100, 100), _det(300, 100)] for i in range(4, 20)},  # B：第 4 帧起
        **{i: [_det(100, 100), _det(300, 100), _det(500, 100)] for i in range(8, 14)},  # C
    }
    summary = process_frames(range(20), _scripted(script))

    assert summary.total_frames == 20
    assert summary.processed_frames == 20
    assert len(summary.frames) == 20
    assert summary.max_simultaneous == 3  # 帧 9–14 三只同在
    assert summary.unique_count == 3
    assert summary.per_class == {"sheep": 3}
    # 逐帧计数序列与事件一致
    assert summary.frames[0].count == 1
    assert summary.frames[19].count == 2
    # 无越线器时不产生进/出
    assert (summary.inbound, summary.outbound, summary.net) == (0, 0, 0)


def test_stride_skips_detection_but_keeps_total():
    detect, calls = _detect_calls({i: [_det(100, 100)] for i in range(20)})
    summary = process_frames(range(20), detect, stride=2)

    assert calls == list(range(0, 20, 2))  # 只对采样帧做检测
    assert summary.total_frames == 20
    assert summary.processed_frames == 10
    assert len(summary.frames) == 10
    assert summary.max_simultaneous == 1


def test_previews_evenly_sampled_and_capped():
    """标注帧按整段视频均匀采样且有上限（响应体积可控）。"""
    encoded: list[tuple[int, int]] = []

    def encode(frame, tracked):
        encoded.append((frame, len(tracked)))
        return f"jpeg-{frame}"

    summary = process_frames(
        range(20),
        _scripted({i: [_det(100, 100)] for i in range(20)}),
        encode_fn=encode,
        total_frames=20,
        max_previews=4,
    )
    assert [f.frame_index for f in summary.frames if f.preview_jpeg] == [0, 5, 10, 15]
    assert len(encoded) == 4  # 编码函数只在采样帧被调用
    assert summary.frames[0].preview_jpeg == "jpeg-0"


def test_line_crossing_accumulates_into_summary():
    """越线统计：自上向下平滑走过水平线 → outbound 累计进汇总。

    轨迹按 15px/帧 平滑移动（IoU 足够高），模拟真实跟踪输入。
    """
    line = LineCrossingCounter((0.0, 100.0), (300.0, 100.0))
    script = {i: [_det(100, 40 + i * 15)] for i in range(9)}  # cy: 40 → 160
    summary = process_frames(range(9), _scripted(script), tracker=ObjectTracker(), line=line)

    assert summary.unique_count == 1  # 全程同一目标（未跟丢）
    assert summary.outbound == 1
    assert summary.inbound == 0
    assert summary.net == -1
    assert summary.frames[-1].net == -1
    assert summary.frames[0].outbound == 0  # 首帧尚未穿越


def test_empty_video_is_safe():
    summary = process_frames([], _scripted({}))
    assert summary.total_frames == 0
    assert summary.processed_frames == 0
    assert summary.frames == []
    assert (summary.max_simultaneous, summary.unique_count) == (0, 0)
    assert (summary.inbound, summary.outbound, summary.net) == (0, 0, 0)


def test_per_class_uses_last_seen_class_per_track():
    """每 ID 类别取最近一次检测的类别。"""
    script = {
        0: [_det(100, 100), _det(300, 100, class_id=19, class_name="cow")],
        1: [_det(100, 100), _det(300, 100, class_id=19, class_name="cow")],
    }
    summary = process_frames(range(2), _scripted(script))
    assert summary.per_class == {"sheep": 1, "cow": 1}
    assert summary.unique_count == 2
