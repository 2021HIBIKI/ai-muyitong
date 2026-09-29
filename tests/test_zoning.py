"""草场分区自动分割测试：连通域合并/拆分、斑块过滤、编号排序与确定性。"""
from __future__ import annotations

import numpy as np
import pytest

from app.vision.zoning import segment_plots, vegetation_mask

GREEN = (0, 165, 0)
SOIL = (120, 120, 130)
W, H = 300, 200


def _img() -> np.ndarray:
    return np.full((H, W, 3), SOIL, dtype=np.uint8)


def test_vegetation_mask_opens_and_closes():
    """掩膜去噪：小噪点被开运算移除，植被区内的细小裂缝被闭运算填上。"""
    img = _img()
    img[:, :250] = GREEN
    img[:, 160:163] = SOIL  # 植被区内部的 3px 宽裂缝
    img[10:13, 280:283] = GREEN  # 3×3 噪点
    mask = vegetation_mask(img)
    # 裂缝在植被区内部：闭运算后连通，裂缝处也为植被
    assert mask[100, 161] == 1
    # 噪点被打开：3×3 绿点消失
    assert mask[11, 281] == 0


def test_single_blob_splits_into_even_zones():
    """只有一块大植被区：沿长轴拆成 2 个分区，各自覆盖度接近 1、面积各半。"""
    img = _img()
    img[40:160, 20:260] = GREEN  # 240×120 横宽草带（长轴为 x → 左右二分）
    result = segment_plots(img, n_plots=2)
    assert len(result["plots"]) == 2
    assert [p["code"] for p in result["plots"]] == ["A", "B"]
    for p in result["plots"]:
        assert p["coverage"] > 0.95  # 分区框内几乎全植被
        bw, bh = p["bbox"][2], p["bbox"][3]
        assert 0.22 * W * H <= bw * bh <= 0.38 * W * H  # 各约一半
    # 左分区在左、右分区在右（沿长轴 x 二分）
    left, right = result["plots"]
    assert left["centroid"][0] < right["centroid"][0]


def test_two_blobs_map_to_two_zones():
    """左右两块草斑（中间裸地间隔）→ 两个分区分别命中。"""
    img = _img()
    img[:, 20:140] = GREEN  # 左带
    img[:, 170:270] = GREEN  # 右带（中间 30px 裸地间隔，闭运算不连接）
    result = segment_plots(img, n_plots=2)
    assert len(result["plots"]) == 2
    left, right = result["plots"]
    assert left["centroid"][0] < W // 2 < right["centroid"][0]
    assert abs(left["centroid"][0] - 80) < 12
    assert abs(right["centroid"][0] - 220) < 12
    assert left["coverage"] > 0.95 and right["coverage"] > 0.95


def test_extra_blobs_merge_into_nearest_zone():
    """连通域多于分区数：小斑并入最近者，最终分区数 = n_plots。"""
    img = _img()
    img[:, 20:120] = GREEN   # 大斑 1
    img[:, 140:180] = GREEN  # 小斑（靠近大斑 1）
    img[:, 220:280] = GREEN  # 大斑 2
    result = segment_plots(img, n_plots=2)
    assert len(result["plots"]) == 2
    # 三个草斑中心都必须落在某个分区框内
    for cx in (70, 160, 250):
        inside = any(
            p["bbox"][0] <= cx <= p["bbox"][0] + p["bbox"][2] for p in result["plots"]
        )
        assert inside, f"草斑中心 x={cx} 未被任何分区覆盖"


def test_small_noise_blobs_are_ignored():
    """面积小于阈值的绿点不参与分区（含在 mask_ratio 里，但不单独成区）。"""
    img = _img()
    img[:, 20:280] = GREEN
    img[40:45, 290:295] = GREEN  # 5×5=25px 噪点 << 3% × 60000
    result = segment_plots(img, n_plots=1)
    assert len(result["plots"]) == 1
    x, y, bw, bh = result["plots"][0]["bbox"]
    assert x + bw <= 288  # 分区框不含右侧噪点


def test_all_bare_returns_no_plots():
    result = segment_plots(_img(), n_plots=3)
    assert result["plots"] == []
    assert result["mask_ratio"] == 0.0
    assert result["image"] is None


def test_plot_codes_sorted_left_to_right():
    """编号按质心 x 从左到右：A/B/C 空间有序。"""
    img = _img()
    for x0, x1 in ((10, 90), (110, 190), (210, 290)):
        img[:, x0:x1] = GREEN
    result = segment_plots(img, n_plots=3)
    assert [p["code"] for p in result["plots"]] == ["A", "B", "C"]
    xs = [p["centroid"][0] for p in result["plots"]]
    assert xs == sorted(xs)


def test_segment_is_deterministic():
    """同输入两次运行结果完全一致（无随机成分）。"""
    rng = np.random.default_rng(7)
    img = _img()
    for _ in range(6):
        x0, y0 = int(rng.integers(10, 220)), int(rng.integers(10, 120))
        img[y0 : y0 + 40, x0 : x0 + 60] = GREEN
    a = segment_plots(img, n_plots=3)
    b = segment_plots(img, n_plots=3)
    assert a["plots"] == b["plots"]
    assert a["mask_ratio"] == b["mask_ratio"]


def test_segment_rejects_bad_params():
    img = _img()
    with pytest.raises(ValueError):
        segment_plots(img, n_plots=0)
    with pytest.raises(ValueError):
        segment_plots(img, n_plots=10)
    with pytest.raises(ValueError):
        segment_plots(None)


def test_overlay_image_drawn_with_zone_frames():
    """叠加图与输入同尺寸，且包含分区边框颜色。"""
    img = _img()
    img[:, 30:270] = GREEN
    result = segment_plots(img, n_plots=3)
    overlay = result["image"]
    assert overlay is not None and overlay.shape == img.shape
    # 边框色（蓝 BGR 0,114,227）应出现在叠加图中
    blue_pixels = int(((overlay == (0, 114, 227)).all(axis=2)).sum())
    assert blue_pixels > 100


def test_zones_are_disjoint_and_cover_all_valid_vegetation():
    """分区互不重叠且覆盖全部有效草斑：各分区面积之和 == 有效植被像素总数
    （分区像素分属不同连通域/条带，无重复计数、无遗漏）。"""
    img = _img()
    img[:, 20:140] = GREEN
    img[:, 170:270] = GREEN
    result = segment_plots(img, n_plots=2)
    total = sum(p["area"] for p in result["plots"])
    assert total == int(vegetation_mask(img).sum())


def test_single_blob_splits_into_parallel_bands():
    """单一大草场 → 沿长轴等分为**平行条带**（轴对齐、x 区间互不重叠）——
    修复 k-means 垂直平分线在整片草场上切出斜线的观感问题。"""
    img = _img()
    img[40:160, 20:260] = GREEN  # 宽 > 高 → 长轴为 x → 竖向条带
    result = segment_plots(img, n_plots=3)
    assert len(result["plots"]) == 3
    boxes = [p["bbox"] for p in result["plots"]]  # 已按质心 x 排序
    for i in range(len(boxes) - 1):
        x0, _, bw, _ = boxes[i]
        nx0 = boxes[i + 1][0]
        assert x0 + bw <= nx0 + 1, f"条带 x 区间不得重叠：{boxes[i]} vs {boxes[i+1]}"
    for _, y, _, bh in boxes:
        assert y <= 45 and bh >= 110, f"每条带应近乎全高（平行条带而非斜切）：{y},{bh}"


def test_overlay_fills_only_zone_pixels_not_bbox():
    """绘制按分区真实像素填充：外接矩形完全重叠的两个分区互不染色——
    修复「矩形块交叠看起来分区重叠」的观感问题。"""
    from app.vision.zoning import draw_plots

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    left = np.zeros((100, 100), dtype=bool)
    left[:, 0:40] = True
    right = np.zeros((100, 100), dtype=bool)
    right[:, 60:100] = True
    plots = [
        {"code": "A", "coverage": 1.0, "centroid": (20, 20), "bbox": [0, 0, 100, 100], "mask": left},
        {"code": "B", "coverage": 1.0, "centroid": (80, 20), "bbox": [0, 0, 100, 100], "mask": right},
    ]
    out = draw_plots(img, plots)
    assert out[80, 80].tolist() != [0, 0, 0]   # 右侧（B 的像素）被 B 染色
    assert out[80, 20].tolist() != [0, 0, 0]   # 左侧（A 的像素）被 A 染色
    assert out[80, 50].tolist() == [0, 0, 0]   # 中间（不属于任何分区的像素）保持原样
