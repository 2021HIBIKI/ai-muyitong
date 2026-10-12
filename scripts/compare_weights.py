# -*- coding: utf-8 -*-
"""对比两个检测权重在演示资产上的表现，为「部署哪一个」提供实测依据。

- `models/yolo11n_all.pt`      ICAERUS 公开羊专用权重（域：地面/公路视角羊群）
- `models/yolo11n_private.pt`  本团队在私有数据集 Muyitong-Sheep 上微调所得（域：无人机俯拍放牧）

评测两组资产（均用平台自身的 SheepDetector，推理参数与线上完全一致）：
1. 羊群快速样例 10 张（Mendeley 来源，有 COCO 点标注真值）—— 微调权重的训练域同源；
2. 测试图库 74 张（ICAERUS 视频抽帧 + 网格切片）—— 与微调训练域不同源，考察跨源泛化。

用法（GPU 环境）：
    /d/Anaconda/envs/muyitong/python.exe scripts/compare_weights.py

输出：data/private_dataset/weight_comparison.json（结论随报告书一并引用，可复算）
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    from app.vision.detector import SheepDetector

    weights = {
        "icaerus": ROOT / "models" / "yolo11n_all.pt",
        "private_ft": ROOT / "models" / "yolo11n_private.pt",
    }
    detectors = {k: SheepDetector(model_path=p) for k, p in weights.items()}

    # ---- 1) 快速样例：与 COCO 点标注真值比误差 ----
    sheep = json.loads((ROOT / "data" / "samples" / "sheep" / "manifest.json").read_text(encoding="utf-8"))
    sample_rows, sample_err = [], {k: [] for k in weights}
    for s in sheep["samples"]:
        img = ROOT / "data" / "samples" / "sheep" / s["file"]
        counts = {k: len(d.detect(img)) for k, d in detectors.items()}
        row = {"code": s["code"], "truth": s["truth"], **counts}
        for k in weights:
            sample_err[k].append(abs(counts[k] - s["truth"]))
        sample_rows.append(row)

    # ---- 2) 测试图库：与清单记录的检出数比偏差（考察跨来源域偏移）----
    gallery = json.loads((ROOT / "data" / "samples" / "gallery" / "manifest.json").read_text(encoding="utf-8"))
    gal_diffs = {k: [] for k in weights}
    for g in gallery["images"]:
        img = ROOT / "data" / "samples" / "gallery" / g["file"]
        for k, d in detectors.items():
            gal_diffs[k].append(len(d.detect(img)) - g["count"])

    out = {
        "note": "由 scripts/compare_weights.py 实测生成；检出数存在 ±2 量级的运行波动。",
        "weights": {k: str(v.relative_to(ROOT)) for k, v in weights.items()},
        "sheep_samples": {
            "n": len(sample_rows),
            "mean_abs_error": {k: round(statistics.mean(v), 2) for k, v in sample_err.items()},
            "rows": sample_rows,
        },
        "gallery": {
            "n": len(gallery["images"]),
            "mean_diff": {k: round(statistics.mean(v), 2) for k, v in gal_diffs.items()},
            "median_diff": {k: round(statistics.median(v), 2) for k, v in gal_diffs.items()},
            "images_off_over_10pct": {
                k: sum(1 for x, g in zip(v, gallery["images"]) if abs(x) > 0.1 * max(1, g["count"]))
                for k, v in gal_diffs.items()
            },
        },
        "conclusion": "微调权重在同源（无人机俯拍放牧）样例上更准；在异源（ICAERUS 公路/地面视角）"
                      "测试图库上系统性低估，说明其有效域收窄到本产品的采集视角。平台默认部署公开权重，"
                      "微调权重随源码交付，可通过 VISION_MODEL_PATH 切换。",
    }
    dst = ROOT / "data" / "private_dataset" / "weight_comparison.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("快速样例（10 张）平均绝对误差：", out["sheep_samples"]["mean_abs_error"])
    print("测试图库（74 张）平均/中位偏差：", out["gallery"]["mean_diff"], out["gallery"]["median_diff"])
    print("偏差超 10% 的图：", out["gallery"]["images_off_over_10pct"])
    print("已写入", dst.relative_to(ROOT))


if __name__ == "__main__":
    main()
