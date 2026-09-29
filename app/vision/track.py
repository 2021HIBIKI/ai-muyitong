"""多目标跟踪适配层（阶段 6.1）：Detection → supervision.Detections → ByteTrack。

为什么用 supervision 而不是 ultralytics 的 ``model.track()``：
ultralytics 跟踪器在 import 期会「运行时静默 pip install lap」（离线演示直接 500），
supervision 用 scipy 做匹配、无此问题（见 开发方案/V2开发计划.md 1.1）。

本模块只做「检测列表 ↔ 跟踪器」的适配，不碰 cv2 / torch，便于脚本化序列测试。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

from app.vision.detector import Detection


@dataclass
class TrackedDetection:
    """带跟踪 ID 的检测框（同一段视频内 track_id 保持稳定）。"""

    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2（像素坐标）
    class_id: int
    class_name: str
    confidence: float

    @property
    def centroid(self) -> tuple[float, float]:
        """检测框中心点（越线判定用）。"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class ObjectTracker:
    """ByteTrack 适配：跨帧保持目标 ID（跟踪去重计数与越线统计的基础）。

    内部跟踪器惰性构造（模块 import 期不引入 supervision）；``update()`` 输入一帧的
    Detection 列表，输出带 track_id 的目标列表。跟踪器返回的是「匹配上的检测子集」，
    未匹配上一轨迹的检测不会出现在输出中。
    """

    def __init__(
        self,
        frame_rate: float = 30.0,
        lost_track_buffer: int = 30,
        track_activation_threshold: float = 0.25,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        self.frame_rate = frame_rate
        self.lost_track_buffer = lost_track_buffer
        self.track_activation_threshold = track_activation_threshold
        self.minimum_matching_threshold = minimum_matching_threshold
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self._tracker = None
        self._class_names: dict[int, str] = {}

    def _ensure_tracker(self):
        if self._tracker is None:
            import supervision as sv

            with warnings.catch_warnings():
                # supervision 0.30 起 ByteTrack 标记为弃用（替代品在独立 trackers 包）；
                # 本项目锁定 supervision==0.30.3，这里静默该提示
                warnings.simplefilter("ignore", FutureWarning)
                self._tracker = sv.ByteTrack(
                    track_activation_threshold=self.track_activation_threshold,
                    lost_track_buffer=self.lost_track_buffer,
                    minimum_matching_threshold=self.minimum_matching_threshold,
                    frame_rate=self.frame_rate,
                    minimum_consecutive_frames=self.minimum_consecutive_frames,
                )
        return self._tracker

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        """喂入一帧检测结果，返回带 track_id 的目标（未命中轨迹的检测被丢弃）。

        轨迹确认机制：新目标在其出现后的**第 2 帧**起进入输出（首帧仅建立轨迹；
        跟踪器自身的第 1 帧例外，会立即输出）。因此逐帧计数最多滞后 1 帧，
        对「同帧最大并发」指标无影响。
        """
        import numpy as np
        import supervision as sv

        tracker = self._ensure_tracker()
        for detection in detections:
            self._class_names.setdefault(detection.class_id, detection.class_name)

        if detections:
            frame = sv.Detections(
                xyxy=np.array([d.bbox for d in detections], dtype=np.float32),
                confidence=np.array([d.confidence for d in detections], dtype=np.float32),
                class_id=np.array([d.class_id for d in detections], dtype=np.int32),
            )
        else:
            frame = sv.Detections.empty()

        tracked = tracker.update_with_detections(frame)
        results: list[TrackedDetection] = []
        if tracked.tracker_id is None:
            return results
        for index, track_id in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[index])
            class_id = (
                int(tracked.class_id[index]) if tracked.class_id is not None else -1
            )
            confidence = (
                float(tracked.confidence[index]) if tracked.confidence is not None else 0.0
            )
            results.append(
                TrackedDetection(
                    track_id=int(track_id),
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                    class_name=self._class_names.get(class_id, str(class_id)),
                    confidence=confidence,
                )
            )
        return results

    def reset(self) -> None:
        """清空跟踪状态（换视频时调用，避免 ID 串号）。"""
        self._tracker = None
        self._class_names = {}
