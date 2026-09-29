"""阶段 1.1：检测器测试 —— bbox 结构、设备回退、权重优先级（不下载权重）。"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from app.vision.detector import Detection, SheepDetector, draw_annotations


class FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = torch.tensor([list(xyxy)])
        self.cls = torch.tensor([cls])
        self.conf = torch.tensor([conf])


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeModel:
    def __init__(self, results):
        self._results = results
        self.calls: list = []

    def predict(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return self._results


def test_resolve_device_auto_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    d = SheepDetector(device="auto")
    # CUDA 不可用时 auto 回退 cpu（mock，不依赖本机环境）
    assert d._resolve_device() == "cpu"


def test_resolve_device_auto_to_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    d = SheepDetector(device="auto")
    # CUDA/HIP（含 AMD ROCm SDK）可用时 auto 优先 GPU
    assert d._resolve_device() == "cuda"


def test_resolve_device_cuda_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    d = SheepDetector(device="cuda")
    assert d._resolve_device() == "cpu"  # 无 CUDA 时回退 cpu


def test_resolve_device_cpu():
    d = SheepDetector(device="cpu")
    assert d._resolve_device() == "cpu"


def test_model_path_priority(tmp_path):
    p = tmp_path / "custom.pt"
    p.write_bytes(b"x")
    d = SheepDetector(model_path=p, fallback_model="yolov8s.pt")
    assert d._resolve_model_path() == str(p)


def test_model_path_fallback(tmp_path):
    d = SheepDetector(model_path=tmp_path / "missing.pt", fallback_model="yolov8s.pt")
    assert d._resolve_model_path() == "yolov8s.pt"


def test_lazy_load():
    d = SheepDetector()
    assert d._model is None  # 未调用前不加载模型


def test_load_model_runs_warmup(monkeypatch):
    """加载模型后执行一次空图预热，且使用解析后的设备。"""
    import ultralytics

    calls: list = []

    class FakeYOLO:
        def __init__(self, path):
            self.path = path

        def predict(self, source, **kwargs):
            calls.append((source, kwargs))
            return []

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    d = SheepDetector()
    d._load_model()
    assert calls  # 预热执行了一次 predict
    source, kwargs = calls[0]
    assert source.shape == (640, 640, 3)  # 固定尺寸空图（与真实推理形状一致）
    assert (source == 114).all()  # 填充灰度
    assert kwargs["device"] == "cpu"  # 预热使用解析后的设备
    assert kwargs["conf"] == d._confidence


def test_load_model_warmup_failure_does_not_block(monkeypatch):
    """预热抛错时加载流程不受影响。"""
    import ultralytics

    class BrokenYOLO:
        def __init__(self, path):
            self.path = path

        def predict(self, source, **kwargs):
            raise RuntimeError("hip warmup failed")

    monkeypatch.setattr(ultralytics, "YOLO", BrokenYOLO)
    d = SheepDetector()
    assert d._load_model() is not None  # 预热失败仅被吞掉


def test_detect_bbox_structure(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # 本测试固定走 cpu 路径
    d = SheepDetector()
    d._model = FakeModel(
        [
            FakeResult(
                [FakeBox((0, 0, 10, 10), 18, 0.9), FakeBox((5, 5, 15, 15), 19, 0.8)],
                {18: "sheep", 19: "cow"},
            )
        ]
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)  # 与固定输入同尺寸：映射为恒等变换
    dets = d.detect(img)
    assert len(dets) == 2
    assert dets[0].bbox == (0, 0, 10, 10)
    assert dets[0].class_name == "sheep"
    assert dets[0].class_id == 18
    assert dets[0].confidence == pytest.approx(0.9)
    assert dets[1].class_name == "cow"
    # predict 传入设备与置信度
    assert d._model.calls[0][1]["device"] == "cpu"
    assert d._model.calls[0][1]["conf"] == d._confidence


def test_draw_annotations():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    dets = [Detection((0, 0, 10, 10), 18, "sheep", 0.9)]
    out = draw_annotations(img, dets)
    assert out.shape == img.shape
    assert out is not img  # 不修改原图


# ---------------------------------------------------------------------------
# 固定输入 letterbox（修复：特定尺寸图导致 ROCm/MIOpen 张量形状编译卡死）
# ---------------------------------------------------------------------------

def test_letterbox_shape_and_padding():
    from app.vision.detector import INPUT_SIZE, _letterbox

    # 320×640 竖图：保持比例，左右各填 160px 灰边
    img = np.zeros((640, 320, 3), dtype=np.uint8)
    canvas, scale, pad_x, pad_y = _letterbox(img)
    assert canvas.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert scale == 1.0 and pad_x == 160 and pad_y == 0
    assert (canvas[:, 0] == 114).all()  # 填充为灰

    # 小图放大：长边贴合 640
    small = np.zeros((64, 32, 3), dtype=np.uint8)
    canvas2, scale2, px2, py2 = _letterbox(small)
    assert canvas2.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert scale2 == 10.0 and px2 == 160 and py2 == 0


def test_detect_maps_boxes_back_to_original_coords(monkeypatch):
    """画布坐标 → 原图坐标映射：去填充、除缩放、裁剪到边界；模型收到固定 640×640 输入。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    d = SheepDetector()
    img = np.zeros((640, 320, 3), dtype=np.uint8)  # scale=1, pad_x=160

    d._model = FakeModel([FakeResult([FakeBox((170, 10, 270, 60), 0, 0.9)], {0: "sheep"})])
    dets = d.detect(img)
    assert dets[0].bbox == (10.0, 10.0, 110.0, 60.0)

    src, _ = d._model.calls[0]
    assert src.shape == (640, 640, 3)  # 固定输入形状

    # 含填充区的越界框被裁剪到原图内
    d._model = FakeModel([FakeResult([FakeBox((0, 0, 640, 700), 0, 0.8)], {0: "sheep"})])
    dets2 = d.detect(img)
    assert dets2[0].bbox == (0.0, 0.0, 320.0, 640.0)
