"""阶段 1.4：样例资产测试 —— 公开羊群/草场样例图存在且可被 OpenCV 解码。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"

# 草场样例集：多草场 × 多分区编号（反馈迭代：草场分析的 10 个样例）
GRASSLAND_DIR = SAMPLES_DIR / "grassland"

# 羊专用权重应存在（gitignored，需手动下载；仅断言路径约定与文件非空）。
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def _sample_images() -> list[Path]:
    return sorted(
        p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def test_sample_images_present():
    """至少准备 2 张公开羊群样例图。"""
    assert len(_sample_images()) >= 2


def test_sample_images_decodable():
    """每张样例图都能被 OpenCV 解码为非空彩色图像。"""
    for p in _sample_images():
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        assert img is not None, f"{p.name} 无法解码"
        assert img.size > 0
        assert img.ndim == 3


def test_sheep_weight_path_convention():
    """羊专用权重路径约定为 models/yolo11n_all.pt（文件由 .gitignore 排除，需手动下载）。"""
    from app.config import settings

    assert settings.vision_model_path == "models/yolo11n_all.pt"
    assert settings.vision_model_path_resolved.name == "yolo11n_all.pt"


# ---------------------------------------------------------------------------
# 草场样例集：多草场 · 分区编号照片（反馈迭代需求）
# ---------------------------------------------------------------------------

def _grassland_manifest() -> dict:
    path = GRASSLAND_DIR / "manifest.json"
    assert path.exists(), "草场样例清单缺失 data/samples/grassland/manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_images(manifest: dict):
    for pasture in manifest["pastures"]:
        for img in pasture["images"]:
            yield pasture, img


def test_grassland_samples_numbered_and_decodable():
    """10 张编号照片分属多个草场：编号唯一、文件名=编号、可解码且尺寸与清单一致。"""
    manifest = _grassland_manifest()
    assert len(manifest["pastures"]) >= 2  # 演示「哪些图片属于哪个草场」至少两个草场

    codes, total = [], 0
    for pasture, img in _iter_images(manifest):
        total += 1
        code = img["code"]
        codes.append(code)
        # 编号自证归属：草场编号-分区-序号（如 P1-A-01）
        assert code.startswith(pasture["code"] + "-"), f"{code} 与草场 {pasture['code']} 不匹配"
        assert len(code.split("-")) == 3, f"编号格式应为 草场-分区-序号：{code}"
        p = GRASSLAND_DIR / img["file"]
        assert p.exists(), f"{p.name} 缺失"
        assert img["file"] == f"{code}.jpg"
        decoded = cv2.imread(str(p), cv2.IMREAD_COLOR)
        assert decoded is not None, f"{p.name} 无法解码"
        assert (decoded.shape[1], decoded.shape[0]) == (img["w"], img["h"])

    assert total == 10  # 需求：10 个草场样例
    assert len(codes) == len(set(codes)), "编号必须唯一"


def test_grassland_manifest_coverage_matches_algorithm():
    """清单覆盖度与算法实测一致（±0.01），且均值档位与告警阈值一致。"""
    from app.config import settings
    from app.vision.grassland import coverage

    manifest = _grassland_manifest()
    for pasture, img in _iter_images(manifest):
        real = coverage(cv2.imread(str(GRASSLAND_DIR / img["file"]), cv2.IMREAD_COLOR))
        assert abs(real["coverage"] - img["coverage"]) < 0.01, img["code"]

        # 均值档位（与前端聚合口径一致：各分区等权平均）
        covs = [i["coverage"] for i in pasture["images"]]
        mean = sum(covs) / len(covs)
        if mean >= settings.alert_warning_threshold:
            level = "normal"
        elif mean >= settings.alert_danger_threshold:
            level = "warning"
        else:
            level = "danger"
        assert abs(mean - pasture["expected"]["mean_coverage"]) < 0.01
        assert level == pasture["expected"]["level"], pasture["code"]


def test_grassland_samples_cover_three_alert_levels():
    """样例覆盖 正常/预警/危险 三个档位，保证演示能看到三种草场状态。"""
    manifest = _grassland_manifest()
    levels = {p["expected"]["level"] for p in manifest["pastures"]}
    assert levels == {"normal", "warning", "danger"}


# ---------------------------------------------------------------------------
# 羊群识别「快速样例」（编号即归属：草场-用途-序号，S=航拍羊群）
#  源= Mendeley Data《UAV footage of Merino sheep in South Africa》，由
#  scripts/make_sheep_samples.py 确定性生成；truth=点标注数，detected=生成时实测检出。
# ---------------------------------------------------------------------------

SHEEP_DIR = SAMPLES_DIR / "sheep"
SHEEP_CODE_RE = re.compile(r"^P\d+-S-\d{2}$")


def _sheep_manifest() -> dict:
    path = SHEEP_DIR / "manifest.json"
    assert path.exists(), "缺羊群样例清单 data/samples/sheep/manifest.json（运行 scripts/make_sheep_samples.py）"
    return json.loads(path.read_text(encoding="utf-8"))


def test_sheep_samples_numbered_and_decodable():
    """10 个快速样例按「草场-用途-序号」编号（如 P1-S-01）：编号唯一、
    文件名=编号、归属段与草场一致、可解码且尺寸与清单一致。"""
    manifest = _sheep_manifest()
    samples = manifest["samples"]
    assert len(samples) == 10  # 前端快速样例按钮 10 个

    codes = [s["code"] for s in samples]
    assert len(codes) == len(set(codes)), "编号必须唯一"
    for s in samples:
        code = s["code"]
        assert SHEEP_CODE_RE.match(code), f"编号格式应为 草场-S-序号（如 P1-S-01）：{code}"
        assert s["pasture"] == code.split("-")[0]
        assert s["file"] == f"{code}.jpg"
        p = SHEEP_DIR / s["file"]
        assert p.exists(), f"{p.name} 缺失"
        decoded = cv2.imread(str(p), cv2.IMREAD_COLOR)
        assert decoded is not None, f"{p.name} 无法解码"
        assert (decoded.shape[1], decoded.shape[0]) == (s["w"], s["h"]) == tuple(manifest["size"])


def test_sheep_samples_cover_pastures_and_count_spread():
    """覆盖 P1/P2/P3 三个演示草场；计数档位齐全：含空场景 0 与 ≥100 的大群。"""
    samples = _sheep_manifest()["samples"]
    assert {s["pasture"] for s in samples} == {"P1", "P2", "P3"}
    truths = [s["truth"] for s in samples]
    assert truths.count(0) == 1, "应恰有一个空场景样例（零检出验证）"
    assert max(truths) >= 100, "应有大群样例"
    assert len({t for t in truths if t}) >= 5, "小群/中群档位应有区分度"


def test_sheep_samples_detection_verified_at_generation():
    """demo 可用性护栏：生成时本机权重实测检出 detected 与标注真值同量级
    （max(2, 10%) 内；空场景必须 0 检出）——样例不是「声称可用」而是已验证。"""
    samples = _sheep_manifest()["samples"]
    for s in samples:
        assert "detected" in s, f"{s['code']} 缺 detected（生成时请带 --detect 重跑）"
        tol = max(2, 0.1 * s["truth"])
        assert abs(s["detected"] - s["truth"]) <= tol, (
            f"{s['code']} 检出 {s['detected']} 与真值 {s['truth']} 偏离超过容差"
        )
        if s["truth"] == 0:
            assert s["detected"] == 0, f"{s['code']} 空场景出现误检"


# ---------------------------------------------------------------------------
# 塔基斜拍模拟样例（演示资产：由 scripts/make_tower_sample.py 确定性生成）
# ---------------------------------------------------------------------------

TOWER_SAMPLE = Path(__file__).resolve().parent.parent / "static" / "assets" / "samples" / "tower_oblique.jpg"


def test_tower_sample_asset_present_and_decodable():
    """斜拍模拟样例存在且可解码（960×540；黑边 + 梯形地面区域）。

    源为 ICAERUS 真实无人机俯拍牧地（mauron_example1）经已知单应变换生成：
    中心区域应为绿色植被（B 通道显著高于 R 通道）。"""
    assert TOWER_SAMPLE.exists(), "缺 static/assets/samples/tower_oblique.jpg（运行 scripts/make_tower_sample.py）"
    img = cv2.imread(str(TOWER_SAMPLE), cv2.IMREAD_COLOR)
    assert img is not None
    assert (img.shape[1], img.shape[0]) == (960, 540)
    # 四周黑边（模拟天空/空地）确实存在：左上角近乎纯黑
    assert img[0:8, 0:8].mean() < 10
    # 中心为绿色牧地（黄绿草）：绿通道高于红通道，且显著高于蓝通道
    center = img[200:340, 320:640].astype(float)
    blue, green, red = (center[:, :, i].mean() for i in range(3))
    assert green > red and green > blue + 40, f"中心应为绿色植被（BGR {blue:.0f}/{green:.0f}/{red:.0f}）"
