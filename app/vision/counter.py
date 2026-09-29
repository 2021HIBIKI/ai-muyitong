"""计数去重：按唯一 track ID 计数，避免同一只羊被重复计数。"""
from __future__ import annotations

from collections import Counter

from app.vision.detector import Detection


def count_sheep(detections: list[Detection], track_ids: list[int | None] | None = None) -> dict:
    """
    统计检测结果数量。

    - 无 track_ids：直接按检测框计数。
    - 有 track_ids：按唯一 track ID 去重（跨帧同一只羊只计一次）；
      track_id 为 None 的框各自按独立个体计一次。

    返回 ``{"count": int, "per_class": {class_name: int}}``。
    """
    if track_ids is not None:
        if len(track_ids) != len(detections):
            raise ValueError("track_ids 长度必须与 detections 一致")
        seen_ids: set[int] = set()
        per_class: dict[str, int] = {}
        for det, tid in zip(detections, track_ids):
            if tid is None:
                per_class[det.class_name] = per_class.get(det.class_name, 0) + 1
            elif tid not in seen_ids:
                seen_ids.add(tid)
                per_class[det.class_name] = per_class.get(det.class_name, 0) + 1
        return {"count": sum(per_class.values()), "per_class": per_class}

    per_class = dict(Counter(d.class_name for d in detections))
    return {"count": len(detections), "per_class": per_class}
