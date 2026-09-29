"""羊群检测：YOLO 加载（羊专用权重优先 → COCO 兜底；CUDA→CPU 回退）与推理。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import settings


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2（像素坐标）
    class_id: int
    class_name: str
    confidence: float


INPUT_SIZE = 640  # 固定推理输入边长（见 _letterbox 说明）


def _letterbox(
    image: np.ndarray, size: int = INPUT_SIZE
) -> tuple[np.ndarray, float, int, int]:
    """等比缩放 + 灰边填充到固定 size×size，返回 (画布, 缩放比, 左填充, 上填充)。

    固定输入形状是必要的：ROCm/MIOpen 对未编译过的张量形状可能长时间编译
    甚至挂起，统一形状后内核只编译一次、延迟稳定（CPU 同样受益于该约定）。
    """
    import cv2

    h, w = image.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interp)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # 114 与 ultralytics 默认填充一致
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def draw_annotations(
    image: np.ndarray, detections: list[Detection], color: tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """在图像上绘制检测框与类别标签，返回新图像（不修改原图）。"""
    import cv2

    out = image.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


class SheepDetector:
    """惰性加载 YOLO 模型；羊专用权重优先，COCO 兜底；设备 CUDA→CPU 自动回退。"""

    def __init__(
        self,
        model_path: str | Path | None = None,
        fallback_model: str | None = None,
        device: str | None = None,
        confidence: float | None = None,
    ) -> None:
        self._model = None
        self._model_path = Path(model_path) if model_path else settings.vision_model_path_resolved
        self._fallback_model = fallback_model if fallback_model else settings.vision_fallback_model
        self._device = device if device else settings.vision_device
        self._confidence = confidence if confidence else settings.vision_confidence

    def _resolve_device(self) -> str:
        """解析推理设备：cuda 不可用 / auto 时回退 cpu。"""
        import torch

        if self._device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self._device

    def _resolve_model_path(self) -> str:
        """羊专用权重存在则优先，否则用 COCO 兜底权重。"""
        if self._model_path.exists():
            return str(self._model_path)
        return self._fallback_model

    def _load_model(self):
        from ultralytics import YOLO

        self._model = YOLO(self._resolve_model_path())
        self._warmup()
        return self._model

    def _warmup(self) -> None:
        """固定尺寸空图预热一次：让 CUDA/HIP 内核提前编译（真实推理形状一致）。"""
        try:
            blank = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
            self._model.predict(
                blank, device=self._resolve_device(), conf=self._confidence, verbose=False
            )
        except Exception:
            # 预热属尽力而为：失败不阻塞加载，真实推理时再按需编译
            pass

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def detect(
        self, image: np.ndarray | str | Path, max_det: int | None = None
    ) -> list[Detection]:
        """对单张图像做目标检测，返回原图坐标系下的 Detection 列表。

        输入统一 letterbox 到 INPUT_SIZE×INPUT_SIZE（固定张量形状，规避
        ROCm/MIOpen 对任意形状的编译卡死），检测框再映射回原图坐标。
        接受 ndarray，或文件路径（str/Path，自动读取）。
        max_det 为单帧最大检测数（缺省取配置；航拍大群会超过 ultralytics 默认 300）。
        """
        if not isinstance(image, np.ndarray):
            import cv2

            loaded = cv2.imread(str(image))
            if loaded is None:
                raise ValueError(f"无法读取图片: {image}")
            image = loaded

        model = self.model
        device = self._resolve_device()
        limit = int(max_det) if max_det is not None else settings.vision_max_det
        canvas, scale, pad_x, pad_y = _letterbox(image)
        results = model.predict(
            canvas, device=device, conf=self._confidence, max_det=limit, verbose=False
        )
        h, w = image.shape[:2]
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            names = r.names
            for box in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                # 画布坐标 → 原图坐标（去填充、除缩放、裁剪到边界）
                x1, x2 = (x1 - pad_x) / scale, (x2 - pad_x) / scale
                y1, y2 = (y1 - pad_y) / scale, (y2 - pad_y) / scale
                x1, y1 = max(0.0, min(x1, w)), max(0.0, min(y1, h))
                x2, y2 = max(0.0, min(x2, w)), max(0.0, min(y2, h))
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=cls,
                        class_name=str(names.get(cls, cls)),
                        confidence=conf,
                    )
                )
        return detections
