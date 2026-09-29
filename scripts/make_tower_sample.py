# -*- coding: utf-8 -*-
"""生成塔基斜拍模拟样例（演示资产，确定性、可复现）。

原理：把 ``data/samples/mauron_example1.JPG``（ICAERUS 真实无人机正俯拍牧地，
含羊群/树篱/割草带）经已知单应变换「斜拍化」——地面四边形在画面中呈梯形
（远边窄、近边宽），四周留黑边模拟天空/空地，输出到
``static/assets/samples/tower_oblique.jpg``。

前端「载入样例」会把 QUAD（归一化地面四边形角点）自动填入 4 个标定点——
一键演示「斜拍直测低估 → 四点标定校正恢复真值」。
**重新生成后请把 index.html 中样例 URL 的 ?v= 版本号 +1**（绕开浏览器缓存）。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "samples" / "mauron_example1.JPG"
DST = ROOT / "static" / "assets" / "samples" / "tower_oblique.jpg"

SIZE = (960, 540)
# 取牧地主体（裁掉顶部大块翻耕地，突出草场与羊群）
CROP = (0, 700, 4000, 3000)  # x0, y0, x1, y1
# 斜拍画面中地面四边形（归一化，TL/TR/BR/BL）：远边窄（塔基俯视透视）
QUAD = [[0.34, 0.10], [0.66, 0.13], [0.96, 0.87], [0.04, 0.85]]


def main() -> None:
    img = cv2.imread(str(SRC))
    assert img is not None, f"样例源图缺失：{SRC}"
    x0, y0, x1, y1 = CROP
    img = img[y0:y1, x0:x1]
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[q[0] * SIZE[0], q[1] * SIZE[1]] for q in QUAD])
    homography = cv2.getPerspectiveTransform(src, dst)
    oblique = cv2.warpPerspective(
        img, homography, SIZE, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    DST.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DST), oblique, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(json.dumps(QUAD))
    print(f"已生成 {DST}（{SIZE[0]}x{SIZE[1]}）")


if __name__ == "__main__":
    main()
