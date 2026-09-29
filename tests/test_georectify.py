"""塔基图像几何校正（单应变换）测试：四点标定、正射恢复、退化输入与重投影误差。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.vision.georectify import (
    default_dst_points,
    order_points,
    orthorectify,
    reprojection_error,
)

TARGET_W, TARGET_H = 320, 240


def _target_image() -> np.ndarray:
    """高对比测试图：左半黑右半白 + 左上一块绿色（便于验证变换后结构是否还原）。"""
    img = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    img[:, TARGET_W // 2 :] = 255
    img[20:60, 20:60] = (0, 160, 0)  # BGR 绿
    return img


def _tilted_source(target: np.ndarray):
    """把正射目标图做一次已知透视变换，模拟塔基斜拍。返回 (斜拍图, 目标角点, 源角点)。"""
    dst_corners = np.array(
        [[0.0, 0.0], [TARGET_W - 1.0, 0.0], [TARGET_W - 1.0, TARGET_H - 1.0], [0.0, TARGET_H - 1.0]],
        dtype=np.float32,
    )
    src_corners = np.array(
        [[26.0, 34.0], [300.0, 12.0], [315.0, 214.0], [12.0, 196.0]], dtype=np.float32
    )
    h_true = cv2.getPerspectiveTransform(dst_corners, src_corners)
    tilted = cv2.warpPerspective(target, h_true, (340, 240))
    return tilted, dst_corners.tolist(), src_corners.tolist()


def test_order_points_sorts_quadrant():
    """乱序输入 → 左上 / 右上 / 右下 / 左下。"""
    quad = [[311, 208], [8, 22], [18, 200], [300, 10]]
    ordered = order_points(quad)
    assert ordered[0].tolist() == [8, 22]  # 左上
    assert ordered[1].tolist() == [300, 10]  # 右上
    assert ordered[2].tolist() == [311, 208]  # 右下
    assert ordered[3].tolist() == [18, 200]  # 左下


def test_orthorectify_recovers_known_transform():
    """已知透视变换的斜拍图 → 用 4 个对应点校正 → 结构还原且重投影误差≈0。"""
    target = _target_image()
    tilted, dst_corners, src_corners = _tilted_source(target)

    result = orthorectify(tilted, src_corners, dst_corners)
    restored = result["image"]

    assert restored.shape[:2] == (TARGET_H, TARGET_W)
    # 左右半区明暗结构还原（双线性重采样会有边缘模糊，取半区均值判断）
    left = float(restored[:, : TARGET_W // 2 - 8].mean())
    right = float(restored[:, TARGET_W // 2 + 8 :].mean())
    assert left < 60, f"左半区应接近黑，实测均值 {left}"
    assert right > 190, f"右半区应接近白，实测均值 {right}"
    # 左上绿色块还原到原位
    block = restored[26:54, 26:54]
    assert block[:, :, 1].mean() > block[:, :, 0].mean() + 60  # 绿通道显著高于蓝通道
    assert reprojection_error(result["matrix"], src_corners, dst_corners) < 0.5


def test_orthorectify_default_dst_preserves_aspect():
    """未给目标点时按源四边形边长生成正射矩形：尺寸比例与源四边形一致。"""
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    # 源四边形：水平边长 160、斜边长 √(20²+80²)≈82.46 的平行四边形（模拟斜拍矩形）
    quad = [[20, 40], [180, 40], [200, 120], [40, 120]]
    result = orthorectify(img, quad)
    w, h = result["size"]
    assert h > 0 and w > 0
    assert abs(w - 160) <= 2
    assert abs(h - 82.46) <= 2  # 取两条斜边的较长者
    assert abs(w / h - 160 / 82.46) < 0.05


def test_orthorectify_accepts_normalized_source_points():
    """源点支持归一化坐标（0–1）：与像素坐标应得到等价结果。"""
    target = _target_image()
    tilted, dst_corners, src_corners = _tilted_source(target)
    h, w = tilted.shape[:2]

    pixels = orthorectify(tilted, src_corners, dst_corners)
    normalized = orthorectify(tilted, [[x / w, y / h] for x, y in src_corners], dst_corners)
    assert normalized["size"] == pixels["size"]
    diff = np.abs(normalized["image"].astype(int) - pixels["image"].astype(int)).mean()
    assert diff < 2.0, f"归一化与像素坐标结果应一致，实测平均差 {diff}"


def test_orthorectify_rejects_bad_input():
    """点数不对 / 退化四边形 / 空图 —— 一律 ValueError（路由层转 400）。"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        orthorectify(img, [[0, 0], [10, 0], [10, 10]])  # 只有 3 点
    with pytest.raises(ValueError):
        orthorectify(img, [[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]])  # 5 点
    with pytest.raises(ValueError):
        orthorectify(img, [[0, 0], [10, 0], [20, 0], [30, 0]])  # 共线 → 面积 0
    with pytest.raises(ValueError):
        orthorectify(img, [[50, 50], [50.1, 50], [50.1, 50.1], [50, 50.1]])  # 面积过小
    with pytest.raises(ValueError):
        orthorectify(None, [[0, 0], [100, 0], [100, 100], [0, 100]])
    with pytest.raises(ValueError):
        orthorectify(img, [[0, 0], [100, 0], [100, 100], [0, np.nan]])  # 非有限值


def test_default_dst_points_matches_quad_edges():
    quad = np.array([[10.0, 20.0], [110.0, 20.0], [110.0, 70.0], [10.0, 70.0]])
    dst = default_dst_points(quad)
    assert dst[0].tolist() == [0.0, 0.0]
    assert dst[2].tolist() == [99.0, 49.0]  # 宽 100、高 50 → 末点 99/49


def test_orthorectify_matrix_roundtrip_maps_src_to_dst():
    """返回的 H 应把每个源点精确映射到目标点（3×3、有限值）。"""
    img = np.full((120, 160, 3), 200, dtype=np.uint8)
    src = [[20, 30], [140, 10], [150, 100], [10, 110]]
    dst = [[0, 0], [119, 0], [119, 89], [0, 89]]
    result = orthorectify(img, src, dst)
    assert len(result["matrix"]) == 3 and len(result["matrix"][0]) == 3
    assert reprojection_error(result["matrix"], result["src_points"], result["dst_points"]) < 1e-6
