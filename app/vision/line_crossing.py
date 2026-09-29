"""越线进出统计（阶段 6.2）：supervision.LineZone 适配 + 配置化线位。

方向约定（对**从左到右**的水平线）：目标自下向上穿越记为 **inbound**，自上向下记为
**outbound**；把线位端点顺序反过来即可互换方向。

去抖由 supervision 内建提供，无需自研（判定窗口 = ``minimum_crossing_threshold + 1`` 帧）：
- 目标锚点需跨到线的另一侧并保持窗口一致才计一次穿越（默认窗口 3 帧 ≈ 30fps 下 0.1s），
  因而左右横跳式的贴线抖动不会反复计数；
- 检测框跨越线体（各锚点分处两侧）的帧不参与计数；
- 锚点默认取检测框**下边中点**（地面视角下的「落脚点」；可传入自定义锚点集合，
  例如 ``[sv.Position.BOTTOM_CENTER, sv.Position.CENTER]``）。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

from app.vision.track import TrackedDetection

# 默认判定窗口：3 帧（≈30fps 下 0.1s）
DEFAULT_CROSSING_THRESHOLD = 2


@dataclass
class CrossingResult:
    """越线统计结果（累计值 + 本帧事件）。"""

    inbound: int
    outbound: int
    net: int
    crossed_in: tuple[int, ...] = ()
    crossed_out: tuple[int, ...] = ()


class LineCrossingCounter:
    """越线进出统计：LineZone 负责「去抖后判定穿越事件」，本类负责计数策略。

    计数策略（默认开启 ``dedup_per_direction``）：同一 track_id 在同一方向只计一次，
    即「进出/折返」不会让同一只羊把 in（或 out）刷成多次（与 2.2 回退方案的
    「每 ID 每方向最多计一次」一致，避免折返抖动虚高）。
    """

    def __init__(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        anchors: list | None = None,
        minimum_crossing_threshold: int = DEFAULT_CROSSING_THRESHOLD,
        dedup_per_direction: bool = True,
    ) -> None:
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))
        self._anchors = anchors
        self.minimum_crossing_threshold = minimum_crossing_threshold
        self.dedup_per_direction = dedup_per_direction
        self._zone = None
        self._inbound = 0
        self._outbound = 0
        self._counted: dict[int, set[str]] = {}

    def _ensure_zone(self):
        if self._zone is None:
            import supervision as sv

            anchors = self._anchors
            if anchors is None:
                anchors = [sv.Position.BOTTOM_CENTER]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                self._zone = sv.LineZone(
                    start=sv.Point(x=self.start[0], y=self.start[1]),
                    end=sv.Point(x=self.end[0], y=self.end[1]),
                    triggering_anchors=anchors,
                    minimum_crossing_threshold=self.minimum_crossing_threshold,
                )
        return self._zone

    def update(self, tracked: list[TrackedDetection]) -> CrossingResult:
        """喂入一帧跟踪结果，返回累计进/出/净与本帧新计入的 track_id。"""
        import numpy as np
        import supervision as sv

        zone = self._ensure_zone()
        if tracked:
            detections = sv.Detections(
                xyxy=np.array([t.bbox for t in tracked], dtype=np.float32),
                confidence=np.array([t.confidence for t in tracked], dtype=np.float32),
                class_id=np.array([t.class_id for t in tracked], dtype=np.int32),
                tracker_id=np.array([t.track_id for t in tracked], dtype=np.int32),
            )
        else:
            detections = sv.Detections.empty()

        crossed_in, crossed_out = zone.trigger(detections=detections)
        ids = [t.track_id for t in tracked] if tracked else []
        raw_in = [ids[i] for i in np.flatnonzero(crossed_in)]
        raw_out = [ids[i] for i in np.flatnonzero(crossed_out)]

        counted_in = self._count("in", raw_in)
        counted_out = self._count("out", raw_out)
        return CrossingResult(
            inbound=self._inbound,
            outbound=self._outbound,
            net=self._inbound - self._outbound,
            crossed_in=tuple(counted_in),
            crossed_out=tuple(counted_out),
        )

    def _count(self, direction: str, track_ids: list[int]) -> list[int]:
        """按计数策略累计（默认每 ID 每方向一次）。"""
        counted: list[int] = []
        for track_id in track_ids:
            seen = self._counted.setdefault(track_id, set())
            if self.dedup_per_direction and direction in seen:
                continue
            seen.add(direction)
            counted.append(track_id)
        if direction == "in":
            self._inbound += len(counted)
        else:
            self._outbound += len(counted)
        return counted

    def reset(self) -> None:
        self._zone = None
        self._inbound = 0
        self._outbound = 0
        self._counted = {}

    def line(self) -> dict:
        """线位（供前端叠加显示）。"""
        return {"start": list(self.start), "end": list(self.end)}


def line_from_ratio(
    width: float, height: float, x1: float, y1: float, x2: float, y2: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """归一化线位（0–1）× 画面尺寸 → 像素线位（配置项到实际帧的换算）。"""
    return (
        (float(x1) * width, float(y1) * height),
        (float(x2) * width, float(y2) * height),
    )
