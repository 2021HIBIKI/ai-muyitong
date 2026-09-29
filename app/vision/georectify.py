"""塔基图像几何校正：四点标定 → 单应变换 → 正射视图。

塔基摄像头俯角大、缺少地理参考，斜拍图像难以与卫星/无人机正射影像及 GIS 数据
协同分析（见覃星力等《AI 驱动的塔基农情遥感监测新技术》，中科院空天院 CropWatch）。
本模块实现其【关键技术 1：塔基视频图像几何校正】的**轻量实现**：由 4 个已知地面
参考点（地块角点等）解算单应矩阵 H，把斜拍视角校正为正射视图。

与论文的差异（如实说明）：论文通过云台参数 + 四元数模型自动完成高精度地理配准；
本项目采用四点标定的单应变换——无云台参数时的标准做法，精度取决于参考点选取。
"""
from __future__ import annotations

import cv2
import numpy as np

# 四边形面积占比下限：低于此值视为退化（点太近或近似共线）
MIN_AREA_RATIO = 0.005


def _parse_points(points, *, width: int | None = None, height: int | None = None) -> np.ndarray:
    """校验并归一化输入点：必须 4 个 (x, y)；值域全在 [0, 1] 时按图像尺寸放大。"""
    if points is None:
        raise ValueError("需要 4 个参考点")
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape != (4, 2):
        raise ValueError(f"参考点必须是 4 个 (x, y) 坐标，收到形状 {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("参考点包含非有限值")
    if width and height and float(np.max(arr)) <= 1.0 and float(np.min(arr)) >= 0.0:
        arr = arr * np.array([width, height], dtype=np.float64)  # 归一化坐标 → 像素坐标
    return arr


def order_points(points) -> np.ndarray:
    """把 4 个点排成 左上 → 右上 → 右下 → 左下，保证单应解算稳定。

    排序依据坐标和差：左上 x+y 最小、右下最大；右上 x−y 最大、左下最小。
    """
    arr = np.asarray(points, dtype=np.float64).reshape(4, 2)
    order = np.zeros((4, 2), dtype=np.float64)
    s = arr.sum(axis=1)
    d = arr[:, 0] - arr[:, 1]  # x - y
    order[0] = arr[np.argmin(s)]  # 左上
    order[2] = arr[np.argmax(s)]  # 右下
    order[1] = arr[np.argmax(d)]  # 右上
    order[3] = arr[np.argmin(d)]  # 左下
    return order


def _quad_area(quad: np.ndarray) -> float:
    """四边形面积（鞋带公式绝对值）。"""
    x, y = quad[:, 0], quad[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def default_dst_points(src: np.ndarray) -> np.ndarray:
    """按源四边形边长生成保持长宽比的正射矩形（左上角为原点）。"""
    tl, tr, br, bl = src
    width = max(float(np.linalg.norm(tr - tl)), float(np.linalg.norm(br - bl)))
    height = max(float(np.linalg.norm(bl - tl)), float(np.linalg.norm(br - tr)))
    width = max(width, 2.0)
    height = max(height, 2.0)
    return np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )


def orthorectify(image: np.ndarray, src_points, dst_points=None) -> dict:
    """按四点标定做透视校正（斜拍 → 正射）。

    ``src_points``：源图像上的 4 个地面参考点；像素坐标或归一化坐标（0–1）皆可，
    顺序任意（内部会排序）。``dst_points``：正射平面的对应 4 点，缺省按源四边形
    边长生成保持长宽比的矩形。

    返回 ``{"image": 校正后图像, "matrix": 3×3 单应矩阵 H, "src_points",
    "dst_points", "size": (宽, 高)}``；点数不对、退化四边形（面积过小/近似共线）
    抛 ``ValueError``。
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("图像为空或格式不正确")
    h, w = image.shape[:2]

    src = order_points(_parse_points(src_points, width=w, height=h))
    if _quad_area(src) < MIN_AREA_RATIO * float(w * h):
        raise ValueError("参考点过于接近或近似共线，无法解算单应矩阵")
    if _quad_area(src) < 1.0:
        raise ValueError("参考点过于接近或近似共线，无法解算单应矩阵")

    if dst_points is None:
        dst = default_dst_points(src)
    else:
        # 目标点使用像素坐标（其坐标系是正射平面本身，无图像尺寸可依）
        dst = order_points(_parse_points(dst_points))

    matrix = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    if not np.isfinite(matrix).all():
        raise ValueError("单应矩阵解算失败：参考点退化")

    out_w = max(int(round(float(dst[:, 0].max()) + 1.0)), 2)
    out_h = max(int(round(float(dst[:, 1].max()) + 1.0)), 2)
    warped = cv2.warpPerspective(image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)

    return {
        "image": warped,
        "matrix": [[float(v) for v in row] for row in matrix],
        "src_points": [[float(x), float(y)] for x, y in src],
        "dst_points": [[float(x), float(y)] for x, y in dst],
        "size": (out_w, out_h),
    }


def reprojection_error(matrix, src_points, dst_points) -> float:
    """单应矩阵在 4 个参考点上的平均重投影误差（像素）——校正质量的量化指标。

    供报告/调试验证使用：理想标定下应接近 0。
    """
    h = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    src = np.asarray(src_points, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_points, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((src.shape[0], 1))
    proj = (h @ np.hstack([src, ones]).T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return float(np.mean(np.linalg.norm(proj - dst, axis=1)))
