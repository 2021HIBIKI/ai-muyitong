"""视频帧处理核心（阶段 6.3）：纯累积逻辑，帧源与检测/编码函数全部注入。

本模块**不 import cv2**：调用方（路由层）负责读帧、检测与标注编码，
这里只做「逐帧累加 + 三条统计口径」，因此可以用假帧源脚本化测试。

三条统计口径（分开返回，避免重复计数质疑，见 开发方案/V2开发计划.md 2.3）：
- ``max_simultaneous``：同帧最大并发 —— 对 ID switch 免疫，「这群有多少只」的主指标；
- ``unique_count``：累计出现过的目标数（受跟踪抖动影响会偏大，需标注说明）；
- ``inbound`` / ``outbound`` / ``net``：越线进/出/净（仅固定机位有意义，流量语义）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from app.vision.detector import Detection
from app.vision.line_crossing import LineCrossingCounter
from app.vision.track import ObjectTracker, TrackedDetection

DetectFn = Callable[[Any], list[Detection]]
EncodeFn = Callable[[Any, list[TrackedDetection]], str]


@dataclass
class FrameStats:
    """单帧统计（frame_index 为原始视频帧序号）。"""

    frame_index: int
    count: int  # 本帧跟踪确认的目标数
    tracker_ids: tuple[int, ...]
    inbound: int
    outbound: int
    net: int
    preview_jpeg: str | None = None


@dataclass
class VideoSummary:
    """整段视频的汇总（三条口径 + 稀有类别 + 逐帧序列）。"""

    processed_frames: int  # 实际送入检测的帧数
    total_frames: int  # 读取到的帧总数
    stride: int
    max_simultaneous: int
    unique_count: int
    inbound: int
    outbound: int
    net: int
    per_class: dict[str, int] = field(default_factory=dict)
    frames: list[FrameStats] = field(default_factory=list)


def process_frames(
    frames: Iterable[Any],
    detect_fn: DetectFn,
    *,
    encode_fn: EncodeFn | None = None,
    tracker: ObjectTracker | None = None,
    line: LineCrossingCounter | None = None,
    stride: int = 1,
    max_previews: int = 24,
    total_frames: int | None = None,
) -> VideoSummary:
    """逐帧累积：检测 → 跟踪 → 越线 → 采样标注帧。

    - ``stride``：检测步长（默认逐帧）。跳过的帧不进跟踪器（ByteTrack 自带漏检容忍）；
    - ``max_previews``：标注帧数量上限（按整段视频均匀采样，避免响应体积过大）；
    - ``tracker`` 必须与 ``line`` 配套使用（越线依赖 track_id）。
    """
    tracker = tracker if tracker is not None else ObjectTracker()
    stride = max(1, int(stride))

    preview_interval = stride
    if encode_fn is not None and max_previews > 0 and total_frames:
        preview_interval = max(stride, math.ceil(total_frames / max_previews))

    frame_stats: list[FrameStats] = []
    class_by_track: dict[int, str] = {}
    max_simultaneous = 0
    processed = 0
    total = 0
    previews_taken = 0

    for index, frame in enumerate(frames):
        total = index + 1
        if index % stride != 0:
            continue

        detections = detect_fn(frame)
        tracked = tracker.update(detections)
        processed += 1

        for item in tracked:
            class_by_track[item.track_id] = item.class_name

        if line is not None:
            crossing = line.update(tracked)
            inbound, outbound, net = crossing.inbound, crossing.outbound, crossing.net
        else:
            inbound = outbound = net = 0

        count = len(tracked)
        max_simultaneous = max(max_simultaneous, count)

        preview = None
        if (
            encode_fn is not None
            and max_previews > 0
            and previews_taken < max_previews
            and index % preview_interval == 0
        ):
            preview = encode_fn(frame, tracked)
            previews_taken += 1

        frame_stats.append(
            FrameStats(
                frame_index=index,
                count=count,
                tracker_ids=tuple(item.track_id for item in tracked),
                inbound=inbound,
                outbound=outbound,
                net=net,
                preview_jpeg=preview,
            )
        )

    per_class: dict[str, int] = {}
    for class_name in class_by_track.values():
        per_class[class_name] = per_class.get(class_name, 0) + 1

    last = frame_stats[-1] if frame_stats else None
    return VideoSummary(
        processed_frames=processed,
        total_frames=total,
        stride=stride,
        max_simultaneous=max_simultaneous,
        unique_count=len(class_by_track),
        inbound=last.inbound if last else 0,
        outbound=last.outbound if last else 0,
        net=last.net if last else 0,
        per_class=per_class,
        frames=frame_stats,
    )
