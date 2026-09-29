# -*- coding: utf-8 -*-
"""生成草场分析「分区编号照片」样例资产与清单（确定性、可复现）。

源数据：Mendeley Data《UAV footage of Merino sheep in South Africa》
（https://data.mendeley.com/datasets/nppwbkdf85/1，解压后为项目根目录下
``UAV footage of Merino sheep in South Africa/Data.zip``；与羊群样例
``scripts/make_sheep_samples.py`` 同源，zip 不入库）。

**编号口径**：草场-分区-序号（如 P1-A-01），编号即归属证明；羊群样例用
S 段（P1-S-01）区分，二者不冲突。

每个样例：
- 数据集原图（4000×3000）→ 1440×1080、JPEG q=88 → ``data/samples/grassland/<编号>.jpg``；
- 覆盖度/绿度用项目算法（ExG+HSV，``app.vision.grassland.coverage``）对**最终资产**
  实测后写入清单——与前端展示、告警档位同一口径（清单即演示真值）。

**三档草情演示护栏**：脚本自检 P1=正常 / P2=预警 / P3=危险 三档齐全（阈值取配置），
不满足时直接报错退出（说明选图或阈值口径变了，需调整选图）。

用法::

    python scripts/make_grassland_samples.py

重新生成后：① 重启服务让示范库按新清单重播草场日序列；② 对三个草场各跑一次
「一键分析全部照片」刷新分区照片实测值（演示库本地状态，不入库）。
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # 允许 `python scripts/make_grassland_samples.py` 直接运行
    sys.path.insert(0, str(ROOT))

from app.vision.grassland import alert_level_from_coverage, coverage  # noqa: E402

ZIP_PATH = ROOT / "UAV footage of Merino sheep in South Africa" / "Data.zip"
IMG_PREFIX = "Data/Image Data/Train and Validation Images/"
DST_DIR = ROOT / "data" / "samples" / "grassland"

SIZE = (1440, 1080)
JPEG_QUALITY = 88

# 演示草场与编号照片（编号=草场-分区-序号；选取标准见各 note）：
# 覆盖度档位按「灌溉良好 → 降水依赖 → 轮休退化」递减，保证三档草情演示齐全。
PASTURES: list[dict] = [
    {
        "code": "P1",
        "name": "东区示范牧场",
        "images": [
            ("P1-A-01", "DJI_0514.JPG", "灌溉草场（长势良好）"),
            ("P1-A-02", "DJI_0625.JPG", "灌溉草场（条带刈割）"),
            ("P1-B-01", "DJI_0641.JPG", "放牧地（返青良好）"),
            ("P1-C-01", "DJI_0624.JPG", "放牧地（刈割后再生）"),
        ],
    },
    {
        "code": "P2",
        "name": "西区生态牧场",
        "images": [
            ("P2-A-01", "DJI_0642.JPG", "降水依赖草场（草情中等）"),
            ("P2-B-01", "Photo_6553951_DJI_351_jpg_6135078_0_2021111781522_photo_original.jpg",
             "羊群卧息地（草情中等）"),
            ("P2-C-01", "Photo_6553956_DJI_356_jpg_4977748_0_202111178198_photo_original.jpg",
             "散牧地（草情中等偏下）"),
        ],
    },
    {
        "code": "P3",
        "name": "北坡轮休草场",
        "images": [
            ("P3-A-01", "Photo_6553950_DJI_350_jpg_6070519_0_2021111781516_photo_original.jpg",
             "轮休坡地（退化迹象）"),
            ("P3-B-01", "DJI_0628.JPG", "轮休坡地（干枯）"),
            ("P3-C-01", "DJI_0627.JPG", "轮休坡地（重度退化）"),
        ],
    },
]


def main() -> None:
    if not ZIP_PATH.exists():
        raise SystemExit(
            f"缺少数据集：{ZIP_PATH}\n"
            "请从 https://data.mendeley.com/datasets/nppwbkdf85/1 下载 "
            "《UAV footage of Merino sheep in South Africa》解压到项目根目录。"
        )

    DST_DIR.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP_PATH)
    names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg"))]

    manifest_pastures, levels = [], {}
    for pasture in PASTURES:
        images = []
        for code, source, note in pasture["images"]:
            matches = [n for n in names if n.split("/")[-1] == source]
            if not matches:
                raise SystemExit(f"数据集中找不到原图：{source}")
            raw = cv2.imdecode(np.frombuffer(z.read(matches[0]), np.uint8), cv2.IMREAD_COLOR)
            if raw is None:
                raise SystemExit(f"原图无法解码：{source}")
            asset = cv2.resize(raw, SIZE, interpolation=cv2.INTER_AREA)

            dst = DST_DIR / f"{code}.jpg"
            if not cv2.imwrite(str(dst), asset, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                raise SystemExit(f"写入失败：{dst}")

            # 存盘回读后再实测：与前端/巡检的分析路径（读文件）完全同源——
            # JPEG 重编码会让边缘少量像素漂移，内存值直接入库会与实测差 ~1%。
            saved = cv2.imread(str(dst), cv2.IMREAD_COLOR)
            measured = coverage(saved)
            images.append(
                {
                    "code": code,
                    "file": dst.name,
                    "source": source,
                    "note": note,
                    "w": SIZE[0],
                    "h": SIZE[1],
                    "coverage": round(measured["coverage"], 4),
                    "greenness": round(measured["greenness"], 4),
                }
            )
            print(f"{code}  覆盖度 {measured['coverage']:.3f}  绿度 {measured['greenness']:.3f}  {note}")

        mean = round(sum(i["coverage"] for i in images) / len(images), 4)
        level = alert_level_from_coverage(mean)
        levels[pasture["code"]] = level
        manifest_pastures.append(
            {
                "code": pasture["code"],
                "name": pasture["name"],
                "images": images,
                "expected": {"mean_coverage": mean, "level": level},
            }
        )
        print(f"  → {pasture['code']} 均值 {mean:.3f}（{level}）\n")

    if set(levels.values()) != {"normal", "warning", "danger"}:
        raise SystemExit(
            f"三档草情演示护栏未通过：P1/P2/P3 等级为 {levels}，应为 normal/warning/danger 各一。"
        )

    manifest = {
        "note": (
            "来源：Mendeley Data《UAV footage of Merino sheep in South Africa》"
            "（https://data.mendeley.com/datasets/nppwbkdf85/1）；"
            "覆盖度/绿度为入库前实测值（算法：ExG+HSV 双阈值），复现见 scripts/make_grassland_samples.py"
        ),
        "pastures": manifest_pastures,
    }
    (DST_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"已生成 {sum(len(p['images']) for p in manifest_pastures)} 张编号照片 → {DST_DIR}")


if __name__ == "__main__":
    main()
