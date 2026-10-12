# -*- coding: utf-8 -*-
"""私有数据集上的模型评估：公开权重 vs 微调权重。

输出 data/private_dataset/eval_metrics.json（报告书表 3-2 的数据来源）并打印摘要。

用法（有 GPU 更快；纯 CPU 亦可，约 1 分钟）：
    python scripts/eval_private_dataset.py

前置：微调权重 runs/private_ft/weights/best.pt（由 scripts/train_private_ft.py 生成；
      随包交付的等价文件是 models/yolo11n_private.pt，499 个张量逐元素相同）。

两处刻意的写法，别改：
- `workers=0`：Windows 下 ultralytics 的 val 默认 spawn DataLoader 子进程，子进程会重新
  导入 __main__，顶层脚本必须带 `if __name__ == "__main__"` 守卫；验证集只有 28 张，
  直接在主进程取数即可，顺带绕开这一整类问题。
- `rect=False, batch=8`：固定 640×640 输入形状，规避 ROCm/MIOpen 对未编译张量形状的
  长时间编译（曾出现单次 val 卡住 20 分钟）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = str(ROOT / "data" / "private_dataset" / "dataset.yaml")
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main() -> None:
    print(f"device = {DEVICE}（{'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}）", flush=True)

    out: dict = {}
    for tag, weights in (
        ("baseline", ROOT / "models" / "yolo11n_all.pt"),
        ("ft", ROOT / "runs" / "private_ft" / "weights" / "best.pt"),
    ):
        model = YOLO(str(weights))
        r = model.val(data=DATA, split="val", device=DEVICE, verbose=False, plots=False,
                      rect=False, batch=8, workers=0,
                      project=str(ROOT / "runs" / "eval"), name=tag, exist_ok=True)
        out[tag] = {
            "weights": str(weights.relative_to(ROOT)),
            "mAP50": round(float(r.box.map50), 4),
            "mAP50_95": round(float(r.box.map), 4),
            "precision": round(float(r.box.mp), 4),
            "recall": round(float(r.box.mr), 4),
        }
        print(f"{tag}: mAP50={out[tag]['mAP50']} mAP50-95={out[tag]['mAP50_95']} "
              f"P={out[tag]['precision']} R={out[tag]['recall']}", flush=True)

    out["val_images"] = len(list((ROOT / "data" / "private_dataset" / "images" / "val").glob("*.jpg")))
    out["note"] = (
        "ft 权重 = runs/private_ft/weights/best.pt（由 scripts/train_private_ft.py 生成）；"
        "随源码交付的等价文件是 models/yolo11n_private.pt（499 个张量逐元素相同，仅清除过内嵌本机路径）。"
        "两组权重在同一验证集、同一推理设置下评测。"
    )
    dst = ROOT / "data" / "private_dataset" / "eval_metrics.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
