# -*- coding: utf-8 -*-
"""在私有派生数据集上微调羊群检测模型（yolo11n，从项目基线权重出发）。

用法::

    python scripts/train_private_ft.py            # 30 epochs / imgsz 640 / batch 16

产物：runs/private_ft/ 下的权重与训练曲线（results.png 等）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "yolo11n_all.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "private_dataset" / "dataset.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="private_ft")
    args = parser.parse_args()

    import torch

    from ultralytics import YOLO

    # ROCm 兼容：Windows TheRock 轮子的 MIOpen 无法 JIT 编译 BatchNorm 训练核
    # （MIOpenBatchNormFwdTrainSpatialHIP.cpp: 'type_traits' not found），
    # 且已实测本机 native ATen 卷积反而更快；故训练进程整体退回 ATen native 核。
    torch.backends.cudnn.enabled = False

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=42,
        project=str(ROOT / "runs"),
        name=args.name,
        workers=2,
        verbose=False,
        plots=True,
        exist_ok=True,
    )
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
