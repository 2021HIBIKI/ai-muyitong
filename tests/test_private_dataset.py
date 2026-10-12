"""私有数据集管线守卫（scripts/make_private_dataset.py 与其产物 manifest.json）。

管线依赖 >5GB 原始数据包（不入库），无法在测试里整跑；因此守两件事：
1) 纯函数行为——质量指标、点标注→YOLO 框、确定性增强、分层划分；
2) 已产出清单的内部一致性，尤其是报告书明示的「同源图的切片与增强变体不跨划分」
   这条防泄漏保证（清单即证据）。
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "scripts" / "make_private_dataset.py"
DATA_DIR = ROOT / "data" / "private_dataset"
MANIFEST = DATA_DIR / "manifest.json"
YAML = DATA_DIR / "dataset.yaml"


def _pipeline():
    spec = importlib.util.spec_from_file_location("make_private_dataset", PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _pipeline()


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------
def test_point_annotation_becomes_uniform_yolo_box():
    """点标注 → 以质心为中心、边长 BOX_SIZE 的 YOLO 归一化框；越界点 clamp 到图内。"""
    # 画面中心：框居中；归一化边长 = BOX_SIZE / 源图边长（缩放不改变该比值）
    cx, cy, bw, bh = P._yolo_boxes([(720.0, 540.0)], 1440, 1080)[0]
    assert (round(cx, 4), round(cy, 4)) == (0.5, 0.5)
    assert abs(bw - P.BOX_SIZE / 1440) < 1e-9 and abs(bh - P.BOX_SIZE / 1080) < 1e-9
    # 4:3 源图（4000×3000）缩放到 OUT_SIZE 同理
    _, _, bw2, bh2 = P._yolo_boxes([(2000.0, 1500.0)], 4000, 3000)[0]
    assert abs(bw2 - P.BOX_SIZE / 4000) < 1e-9 and abs(bh2 - P.BOX_SIZE / 3000) < 1e-9
    # 角点：框被裁掉一半（x1 clamp 到 0），中心落在 1/4 框长处、宽度为半框
    cx3, cy3, bw3, _ = P._yolo_boxes([(0.0, 0.0)], 1440, 1080)[0]
    assert abs(cx3 - (P.BOX_SIZE / 4) / 1440) < 1e-9
    assert abs(cy3 - (P.BOX_SIZE / 4) / 1080) < 1e-9
    assert abs(bw3 - (P.BOX_SIZE / 2) / 1440) < 1e-9
    # 极端越界点仍产出合法范围内的框
    for b in P._yolo_boxes([(-500.0, 9999.0), (1440.0, 1080.0)], 1440, 1080):
        assert 0 < b[0] < 1 and 0 < b[1] < 1 and 0 < b[2] <= 1 and 0 < b[3] <= 1


def test_laplacian_var_separates_sharp_from_blurred():
    """清晰度指标能把纹理图与重模糊图分到阈值两侧（质量筛除依据）。"""
    yy, xx = np.indices((1080, 1440))
    board = np.repeat(np.repeat(((xx // 4 + yy // 4) % 2) * 255, 1, axis=0), 1, axis=1).astype(np.uint8)
    sharp = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=9, sigmaY=9)
    v_sharp, v_blur = P._laplacian_var(sharp), P._laplacian_var(blurred)
    assert v_sharp > P.BLUR_THRESHOLD > v_blur, (v_sharp, v_blur)


def test_splits_are_stratified_deterministic_and_leak_free():
    """按羊数分桶分层 70/15/15：同种子结果一致、每张源图只落一个划分、每个桶都分到三个划分。"""
    per_bucket = 20
    value_of_bucket = {0: 0, 1: 5, 2: 30, 3: 80, 4: 150}
    counts = {f"img{i}_{j}": value_of_bucket[i]
              for i in range(len(P.BUCKETS)) for j in range(per_bucket)}
    kept = sorted(counts)

    a = P._assign_splits(counts, kept, P.SEED)
    b = P._assign_splits(counts, kept, P.SEED)
    assert a == b, "同种子必须给出同一划分"
    assert sorted(a) == kept, "有源图被漏分或重复"
    n = len(kept)
    assert abs(sum(v == "train" for v in a.values()) / n - P.SPLIT_RATIO[0]) < 0.05
    assert abs(sum(v == "val" for v in a.values()) / n - P.SPLIT_RATIO[1]) < 0.05
    for i in range(len(P.BUCKETS)):
        got = {a[f"img{i}_{j}"] for j in range(per_bucket)}
        assert got == {"train", "val", "test"}, f"第 {i} 桶未分层：{got}"


def test_augmentation_is_deterministic_and_keeps_boxes_valid():
    """增强可复现：翻转镜像 x、亮度双向、旋转后框仍在画面内（中心转出者丢弃）。"""
    img = np.full((P.OUT_SIZE[1], P.OUT_SIZE[0], 3), 120, dtype=np.uint8)
    boxes = [(0.25, 0.5, 0.05, 0.06), (0.9, 0.5, 0.05, 0.06)]

    f1, b1 = P._augment(img, boxes, "flip", +1)
    f2, b2 = P._augment(img, boxes, "flip", +1)
    assert np.array_equal(f1, f2) and b1 == b2, "同输入必须产出同结果"
    assert np.array_equal(f1, cv2.flip(img, 1))
    assert abs(b1[0][0] - 0.75) < 1e-9 and abs(b1[1][0] - 0.10) < 1e-9

    up, _ = P._augment(img, boxes, "bright", +1)
    dn, _ = P._augment(img, boxes, "bright", -1)
    assert abs(up.mean() - 120 * 1.2) < 3 and abs(dn.mean() - 120 * 0.8) < 3, (up.mean(), dn.mean())

    for sign in (+1, -1):
        r_img, r_boxes = P._augment(img, boxes, "rot", sign)
        assert r_img.shape == img.shape and len(r_boxes) <= len(boxes)
        for cx, cy, bw, bh in r_boxes:
            assert 0 < cx < 1 and 0 < cy < 1 and 0 < bw <= 1 and 0 < bh <= 1


# ---------------------------------------------------------------------------
# 产物清单
# ---------------------------------------------------------------------------
def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_proves_no_source_leakage():
    """每张源图只记录一次、只归属一个划分，且全部产物名唯一、跟随源图命名。"""
    srcs = _manifest()["sources"]
    files = [s["file"] for s in srcs]
    assert len(files) == len(set(files)), "源图重复登记"
    outputs = [o for s in srcs for o in s["outputs"]]
    assert len(outputs) == len(set(outputs)), "产物名冲突（不同源图的产物重名）"
    for s in srcs:
        assert s["split"] in ("train", "val", "test")
        stem = Path(s["file"]).stem.replace(" ", "_")  # 管线把文件名里的空格换成下划线
        assert s["outputs"] and all(o.startswith(stem) for o in s["outputs"]), s


def test_manifest_counts_follow_pipeline_rules():
    """清单统计必须与管线规则自洽：切片新增数、各划分图片数 = 切片数 ×（训练集含增强）、
    保留图过阈值、切片图达到密集阈值。"""
    m = _manifest()
    st, srcs, params = m["stats"], m["sources"], m["params"]
    assert params["seed"] == P.SEED and params["aug_factor"] == st["aug_factor"]
    assert params["blur_threshold"] == P.BLUR_THRESHOLD and params["box_size"] == P.BOX_SIZE
    assert st["raw_images"] == len(srcs)
    assert st["kept_images"] + st["dropped_images"] == st["raw_images"]
    assert st["patches_added"] == sum(s["n_slices"] - 1 for s in srcs if s["kept"])

    for split, total_key, inst_key in (("train", "total_train", "instances_train"),
                                       ("val", "total_val", "instances_val"),
                                       ("test", "total_test", "instances_test")):
        ss = [s for s in srcs if s["split"] == split]
        factor = st["aug_factor"] if split == "train" else 1
        assert sum(s["n_slices"] * factor for s in ss) == st[total_key], split
        assert sum(len(s["outputs"]) for s in ss) == st[total_key], split
        assert st[inst_key] > 0
        if split != "train":  # 验证/测试集保持原始，不得出现增强变体
            suffixes = tuple("_" + v for v in P.AUG_SUFFIX.values())
            assert not any(o.endswith(suffixes) for s in ss for o in s["outputs"]), split

    for s in srcs:
        assert s["laplacian_var"] >= P.BLUR_THRESHOLD if s["kept"] else bool(s["reason"])
        if s["mode"] == "sliced":
            assert s["sheep"] >= P.DENSE_SHEEP_THRESHOLD
            assert s["n_slices"] == P.SLICE_GRID[0] * P.SLICE_GRID[1]
        else:
            assert s["n_slices"] == 1


def test_dataset_yaml_points_at_manifest_paths():
    """dataset.yaml 的划分路径与类别名和管线约定一致（训练脚本据此加载）。"""
    text = YAML.read_text(encoding="utf-8")
    for key in ("train: images/train", "val: images/val", "test: images/test", "0: sheep"):
        assert key in text, key
    assert "path:" not in text, "不写绝对 path，换台机器才能直接加载"


# 与 scripts/strip_weight_paths.py 的 DRIVE_HINT 同形：必须“像路径”（盘符之后还有
# ≥4 个路径字符、盘符前不接字母数字），否则浮点张量里的随机字节会误报。
_ABS_PATH = re.compile(rb"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/Anaconda/)[A-Za-z0-9_.\- /\\]{4,}")


def test_shipped_artifacts_carry_no_local_absolute_paths():
    """发布信息卫生：随包的清单/配置/评测结果/权重里不得内嵌本机绝对路径。

    pickle 会把训练机的 model / data / save_dir 原样写进 checkpoint，发布前需用
    scripts/strip_weight_paths.py 清除——这条守卫防止重新训练或重新生成后再漏出去。
    """
    artifacts = [MANIFEST, YAML, DATA_DIR / "eval_metrics.json", DATA_DIR / "weight_comparison.json"]
    artifacts += sorted((ROOT / "models").glob("*.pt"))
    checked = 0
    for path in artifacts:
        if not path.exists():
            continue
        checked += 1
        hits = _ABS_PATH.findall(path.read_bytes())
        assert not hits, f"{path.name} 内嵌本机绝对路径：{[h[:60] for h in hits[:3]]}"
    assert checked >= 3, "随包产物缺失，检查无意义"
