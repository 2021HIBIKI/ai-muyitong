"""草场分区自动分割：植被掩膜 → 形态学去噪 → 连通域分析 → 空间聚类成管理分区。

塔基/无人机的大场景草场图里，一张图常覆盖多个管理分区。覃星力等
（《AI 驱动的塔基农情遥感监测新技术》，中科院空天院 CropWatch，2026）用空间信息
引导的 SAM 视频图像分割自动划分种植地块；本模块实现其思路的**轻量版**：
ExG+HSV 植被掩膜 + 形态学开闭去噪 + 连通域分析 + 空间聚类（草斑分组 / 长轴条带），
把一张草场图自动划分为 K 个管理分区（编号 A/B/C…），并给出每个分区的植被覆盖度。

与论文的差异（如实说明）：论文用 GIS 耕地边界矢量生成 SAM 引导点做语义分割；
本项目为**无地理参考下的纯图像连通域聚类**，适用于「一张图覆盖若干草斑」的场景，
全流程确定性、可复现、可单测。
"""
from __future__ import annotations

import cv2
import numpy as np

from app.vision.grassland import exg_standardised_hue

# 分区叠加图的边框颜色（BGR，与 UI 主题一致）
PLOT_COLORS = [(0, 114, 227), (255, 149, 0), (52, 199, 89), (175, 82, 222), (255, 59, 48)]


def vegetation_mask(image: np.ndarray, *, open_ksize: int = 5, close_ksize: int = 13) -> np.ndarray:
    """植被掩膜：ExG+HSV 双阈值 → 开运算去噪点 → 闭运算填小空洞/裂缝（uint8 0/1）。"""
    mask = (exg_standardised_hue(image) > 0).astype(np.uint8)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def _band_split(points: np.ndarray, k: int) -> tuple[np.ndarray, ...]:
    """坐标点集 → k 个**沿长轴等分的平行条带**（确定性、轴对齐）。

    为什么不用 k-means：k-means 的边界是两个质心的垂直平分线——在整片草场上
    会切出没有物理含义的斜线（演示实测「被几条斜线切过」）；且多次迭代结果
    随质心漂移。对单一大片草场，工程上合理的划分是**沿长轴等分条带**
    （对应条带轮牧的管理惯例），边界整齐、结果可预期、可复现。
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return tuple(np.array([], dtype=int) for _ in range(k))
    xs, ys = pts[:, 0], pts[:, 1]
    axis = 0 if (xs.max() - xs.min()) >= (ys.max() - ys.min()) else 1
    order = np.argsort(pts[:, axis], kind="stable")
    edges = np.linspace(0, n, k + 1).astype(int)
    return tuple(order[edges[j] : edges[j + 1]] for j in range(k))


def _component(mask: np.ndarray, label: int, labels: np.ndarray, stats, centroids) -> dict:
    x, y, bw, bh, area = [int(v) for v in stats[label]]
    cx, cy = [float(v) for v in centroids[label]]
    return {
        "mask": labels == label,
        "area": int(area),
        "bbox": [x, y, bw, bh],
        "centroid": (cx, cy),
    }


def _merge_pair(a: dict, b: dict) -> dict:
    """把连通域 b 并入 a（掩膜并集、包围盒并集、面积加权质心）。"""
    mask = a["mask"] | b["mask"]
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    wa, wb = a["area"], b["area"]
    total = wa + wb
    cx = (a["centroid"][0] * wa + b["centroid"][0] * wb) / total
    cy = (a["centroid"][1] * wa + b["centroid"][1] * wb) / total
    return {"mask": mask, "area": total, "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1], "centroid": (cx, cy)}


def _zone_from_indices(mask: np.ndarray, indices: np.ndarray) -> dict:
    """连通域内的像素子集（聚类结果）→ 分区（掩膜、包围盒、覆盖度）。"""
    ys, xs = np.nonzero(mask)
    sub = np.zeros_like(mask, dtype=bool)
    sub[ys[indices], xs[indices]] = True
    y0, y1 = int(ys[indices].min()), int(ys[indices].max())
    x0, x1 = int(xs[indices].min()), int(xs[indices].max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    return {
        "mask": sub,
        "area": int(sub.sum()),
        "bbox": [x0, y0, bw, bh],
        "centroid": (float(xs[indices].mean()), float(ys[indices].mean())),
        "coverage": float(sub.sum() / max(bw * bh, 1)),
    }


def draw_plots(image: np.ndarray, plots: list[dict]) -> np.ndarray:
    """绘制分区叠加图：按分区**真实像素**半透明填充 + 分区轮廓 + 「编号 覆盖率」标注。

    刻意不画外接矩形实心块——矩形必然互相交叠，会让分区看起来"重叠"；
    分区像素集合本身互不重叠（Voronoi 归属），按像素填充如实呈现各分区范围。
    """
    out = image.copy()
    for i, plot in enumerate(plots):
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        mask = plot.get("mask")
        if mask is None:
            continue
        tint = out.copy()
        tint[mask] = color
        out = cv2.addWeighted(tint, 0.28, out, 0.72, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 3)
        cx, cy = int(plot["centroid"][0]), int(plot["centroid"][1])
        label = f"{plot['code']} {plot['coverage'] * 100:.0f}%"
        cv2.putText(out, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.putText(out, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
    return out


def segment_plots(image: np.ndarray, n_plots: int = 3, min_area_ratio: float = 0.03) -> dict:
    """把草场图自动划分为 K 个管理分区并给出分区覆盖度。

    流程：植被掩膜 → 形态学去噪 → 连通域 → 过滤小斑块 → 合并/拆分到 K 个分区：
    - 草斑多于 K：最小者并入质心最近者（自然草斑**空间聚类**，分区=连片草斑集合）；
    - 草斑少于 K：最大斑**沿长轴等分为平行条带**（近似条带轮牧小区，边界轴对齐、
      确定性；刻意不用 k-means——其垂直平分线边界会在整片草场上切出无意义的斜线）；
    最后按质心 x 排序编号 A/B/C…。各分区的像素集合互不重叠（分属不同连通域/条带）。

    返回 ``{"plots": [{code, coverage, area, bbox, centroid}, …],
    "mask_ratio": 全图植被占比, "n_components": 有效连通域数,
    "image": 分区叠加图}``；无植被时 plots 为空、image 为 None。
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("图像为空或格式不正确")
    if not 1 <= n_plots <= 9:
        raise ValueError(f"分区数需在 1–9 之间，收到 {n_plots}")
    h, w = image.shape[:2]
    total_pixels = h * w

    mask = vegetation_mask(image)
    mask_ratio = float(mask.sum() / total_pixels) if total_pixels else 0.0
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    components = []
    for label in range(1, num):
        if stats[label, cv2.CC_STAT_AREA] < min_area_ratio * total_pixels:
            continue  # 过小斑块不参与分区
        components.append(_component(mask, label, labels, stats, centroids))
    valid_components = len(components)  # 参与分区的有效草斑数（面积达标）

    if not components:
        return {"plots": [], "mask_ratio": round(mask_ratio, 4), "n_components": 0, "image": None}

    # 连通域多于目标分区数：最小者并入质心最近者
    while len(components) > n_plots:
        smallest = min(range(len(components)), key=lambda i: components[i]["area"])
        small = components.pop(smallest)
        nearest = min(
            range(len(components)),
            key=lambda i: (
                (components[i]["centroid"][0] - small["centroid"][0]) ** 2
                + (components[i]["centroid"][1] - small["centroid"][1]) ** 2
            ),
        )
        components[nearest] = _merge_pair(components[nearest], small)

    # 连通域少于目标分区数：对最大斑**一次**沿长轴等分为若干平行条带
    # （一次到位；若递归二分，子条带长轴会翻转、切出 T 形，观感混乱）
    if len(components) < n_plots:
        biggest = max(range(len(components)), key=lambda i: components[i]["area"])
        comp = components.pop(biggest)
        ys, xs = np.nonzero(comp["mask"])
        pts = np.column_stack([xs, ys])
        need = n_plots - len(components)  # 还需补几个分区
        for indices in _band_split(pts, need):
            if len(indices):
                components.append(_zone_from_indices(comp["mask"], indices))

    plots = []
    for comp in components:
        bw, bh = comp["bbox"][2], comp["bbox"][3]
        comp["coverage"] = float(comp["area"] / max(bw * bh, 1))
        plots.append(comp)

    # 按质心 x 从左到右编号 A/B/C…
    plots.sort(key=lambda p: p["centroid"][0])
    for i, plot in enumerate(plots):
        plot["code"] = chr(ord("A") + i)

    overlay = draw_plots(image, plots)  # 需要掩膜（此后才剔除大数组）
    for plot in plots:
        plot.pop("mask")

    return {
        "plots": [
            {
                "code": p["code"],
                "coverage": round(p["coverage"], 4),
                "area": p["area"],
                "bbox": p["bbox"],
                "centroid": [round(p["centroid"][0], 1), round(p["centroid"][1], 1)],
            }
            for p in plots
        ],
        "mask_ratio": round(mask_ratio, 4),
        "n_components": valid_components,
        "image": overlay,
    }
