# -*- coding: utf-8 -*-
"""生成羊群识别「快速样例」资产与清单（确定性、可复现）。

源数据：Mendeley Data《UAV footage of Merino sheep in South Africa》
（https://data.mendeley.com/datasets/nppwbkdf85/1，解压后为 ``UAV footage of
Merino sheep in South Africa/Data.zip``，含 125 张 4000×3000 无人机照片 +
COCO 点标注 train/val/test.json）。该 zip 体积 >5GB、不入库，生成脚本按需读取。

**编号口径**：沿用项目「编号即归属」的草场照片编号逻辑（草场-分区-序号，
如草场植被照片 P1-A-01），羊群样例编号为 **草场-用途-序号**（如 P1-S-01）：
- 草场段 P1/P2/P3 与演示草场台账一致；
- 用途段 **S = 航拍羊群照片（Sheep）**，与植被分区照片（A/B/C）区分；
- 序号 01 起，同草场内顺序。

每个样例：
- 从数据集原图按 ``SAMPLES`` 里的 crop（x, y, w, h；None=全图）裁出 4:3 区域，
  等比缩放到 1440×1080、JPEG q=88 → ``data/samples/sheep/<编号>.jpg``；
- 计数参考值 truth = 裁区内点标注数（COCO 标注，类别 sheep_from_above）；
- 逐张用本机权重跑一次检出（--detect）记入清单 detected（同一资产重跑有 ±2 波动），
  保证「样例可正常用于演示」在生成时即被验证。

用法::

    python scripts/make_sheep_samples.py            # 生成资产与清单
    python scripts/make_sheep_samples.py --detect   # 生成后再跑 YOLO 记录检出数（需权重+GPU）

重新生成后请把 index.html 中样例按钮的参考计数按新清单核对（编号不变）。
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # 允许 `python scripts/make_sheep_samples.py` 直接运行
    sys.path.insert(0, str(ROOT))
ZIP_PATH = ROOT / "UAV footage of Merino sheep in South Africa" / "Data.zip"
IMG_PREFIX = "Data/Image Data/Train and Validation Images/"
SPLIT_FILES = ("Data/Image Data/train.json", "Data/Image Data/val.json", "Data/Image Data/test.json")
DST_DIR = ROOT / "data" / "samples" / "sheep"

SIZE = (1440, 1080)  # 输出尺寸（与既有图库同规格，4:3 与源图一致）
JPEG_QUALITY = 88

# 样例定义：编号、源图、裁剪框（x, y, w, h；None=全图）、场景备注。
# 选取标准：① 本机权重检出数 ≈ 标注真值（±10% 内，已逐张验证）；
# ② 计数覆盖 空场景/小群/中群/大群；③ 画面为无人机俯视牧地。
SAMPLES: list[dict] = [
    {"code": "P1-S-01", "source": "Photo_6553957_DJI_357_jpg_5001302_0_2021111781916_photo_original.jpg",
     "crop": None, "note": "开阔牧地羊列（绿褐相间）"},
    {"code": "P1-S-02", "source": "Photo_6553949_DJI_349_jpg_5823893_0_2021111781426_photo_original.jpg",
     "crop": None, "note": "绿色牧地羊群成列"},
    {"code": "P1-S-03", "source": "DJI_0488.JPG",
     "crop": None, "note": "饮水点旁小群"},
    {"code": "P1-S-04", "source": "DJI_0484.JPG",
     "crop": None, "note": "牧道旁小群"},
    {"code": "P2-S-01", "source": "Photo_6554011_DJI_411_jpg_6451356_0_20211124162536_photo_original.jpg",
     "crop": None, "note": "大群沿围栏行进"},
    {"code": "P2-S-02", "source": "DJI_0504.JPG",
     "crop": None, "note": "开阔牧地羊群密团"},
    {"code": "P2-S-03", "source": "Photo_6553998_DJI_398_jpg_6210672_0_2021112416810_photo_original.jpg",
     "crop": None, "note": "绿色草地散羊"},
    {"code": "P3-S-01", "source": "DJI_0486.JPG",
     "crop": None, "note": "干燥牧地小群"},
    {"code": "P3-S-02", "source": "Photo_6554003_DJI_403_jpg_6030034_0_202111241696_photo_original.jpg",
     "crop": None, "note": "坡地散羊"},
    {"code": "P3-S-03", "source": "DJI_0693.JPG",
     "crop": (0, 1500, 2400, 1800), "note": "空场景（无羊，验证零检出）"},
]


def _load_point_annotations(z: zipfile.ZipFile) -> dict[str, list[tuple[float, float]]]:
    """读取三个 split 的 COCO 点标注 → {源图名: [(x, y), …]}（多边形质心）。"""
    points: dict[str, list[tuple[float, float]]] = {}
    for split in SPLIT_FILES:
        data = json.loads(z.read(split))
        by_id = {im["id"]: im["file_name"] for im in data["images"]}
        for ann in data.get("annotations", []):
            seg = ann["segmentation"][0]
            xs, ys = seg[0::2], seg[1::2]
            points.setdefault(by_id[ann["image_id"]], []).append(
                (sum(xs) / len(xs), sum(ys) / len(ys))
            )
    return points


def _count_in_crop(points: list[tuple[float, float]], crop: tuple[int, int, int, int] | None) -> int:
    """裁剪框内的标注点数（None=全图）。"""
    if crop is None:
        return len(points)
    x, y, w, h = crop
    return sum(1 for px, py in points if x <= px < x + w and y <= py < y + h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detect", action="store_true", help="生成后跑 YOLO 记录检出数（需权重）")
    args = parser.parse_args()

    if not ZIP_PATH.exists():
        raise SystemExit(
            f"缺少数据集：{ZIP_PATH}\n"
            "请从 https://data.mendeley.com/datasets/nppwbkdf85/1 下载 "
            "《UAV footage of Merino sheep in South Africa》解压到项目根目录。"
        )

    DST_DIR.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP_PATH)
    points = _load_point_annotations(z)

    detector = None
    if args.detect:
        from app.vision.detector import SheepDetector

        detector = SheepDetector()

    manifest_samples = []
    for spec in SAMPLES:
        code, source = spec["code"], spec["source"]
        raw = cv2.imdecode(
            np.frombuffer(z.read(IMG_PREFIX + source), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if raw is None:
            raise SystemExit(f"源图无法解码：{source}")
        if spec["crop"] is not None:
            x, y, w, h = spec["crop"]
            raw = raw[y : y + h, x : x + w]
        asset = cv2.resize(raw, SIZE, interpolation=cv2.INTER_AREA)

        dst = DST_DIR / f"{code}.jpg"
        ok = cv2.imwrite(str(dst), asset, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise SystemExit(f"写入失败：{dst}")

        truth = _count_in_crop(points.get(source, []), spec["crop"])
        entry = {
            "code": code,
            "file": dst.name,
            "pasture": code.split("-")[0],
            "source": source,
            "crop": list(spec["crop"]) if spec["crop"] else None,
            "truth": truth,
            "w": SIZE[0],
            "h": SIZE[1],
            "note": spec["note"],
        }
        if detector is not None:
            entry["detected"] = len(detector.detect(asset))
        manifest_samples.append(entry)
        extra = f"  detected={entry['detected']}" if "detected" in entry else ""
        print(f"{code}  truth={truth:4d}{extra}  {spec['note']}")

    manifest = {
        "note": (
            "编号=草场-用途-序号（S=航拍羊群照片，与植被分区照片 A/B/C 区分）。"
            "来源：Mendeley Data《UAV footage of Merino sheep in South Africa》"
            "（https://data.mendeley.com/datasets/nppwbkdf85/1），许可见数据集页面；"
            "truth=裁区内 COCO 点标注数（参考值），detected=本机 GPU 实测检出（±2 波动）。"
        ),
        "size": list(SIZE),
        "samples": manifest_samples,
    }
    (DST_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"\n已生成 {len(manifest_samples)} 个样例 → {DST_DIR}")


if __name__ == "__main__":
    main()
