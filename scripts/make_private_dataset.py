# -*- coding: utf-8 -*-
"""构建 AI牧易通「私有派生数据集」：公开无人机羊群数据集 → 确定性处理管线。

源数据：Mendeley Data《UAV footage of Merino sheep in South Africa》
（https://data.mendeley.com/datasets/nppwbkdf85/1），即项目根目录
``UAV footage of Merino sheep in South Africa/Data.zip``（>5GB，不入库；脚本用
zipfile 流式读取，不整包解压）。含 125 张 4000×3000 无人机照片与 COCO 点标注
train/val/test.json（类别 sheep_from_above；另有 19 段视频，不使用）。

处理管线（全部阈值见下方常量；随机种子默认 42，产物可复现）：
1) 读取 125 张带标注原图与三份 COCO json；
2) 质量筛除：缩至 1440×1080 灰度后算 Laplacian 方差，< BLUR_THRESHOLD 判为模糊剔除；
3) 标注转换：COCO 点标注（多边形质心）→ YOLO 目标框，框统一 BOX_SIZE×BOX_SIZE
   原图像素、以质心为中心、clamp 到图内；
4) 密集切片：羊数 ≥ DENSE_SHEEP_THRESHOLD 的图切 2×2 无重叠切片（2000×1500），
   质心在片内的框随切片平移并 clamp；羊数 < 阈值的图整体保留；
5) 统一分辨率：所有输出图（整图与切片）缩放到 OUT_SIZE，框坐标同步缩放；
6) 数据增强（仅 train 划分，确定性）：每张训练图生成 aug_factor-1 个变体——
   水平翻转 / 亮度±20%（HSV V 通道）/ 旋转±10°（图像与框一同旋转，边角填白）；
7) 划分：按源图 70/15/15 分层（羊数分桶 0 / 1-10 / 11-50 / 51-100 / >100），
   随机种子 42；同源图的切片与增强变体必落在同一划分（防泄漏）；
8) 输出 ``data/private_dataset/``：images|labels/{train,val,test}、dataset.yaml、
   manifest.json（处理参数、全部统计、每张源图记录）。

口径约定：
- ``patches_added`` = 因密集切片新增的图片数（每张密集源图 +3）；
- ``instances_*`` = 对应划分 label 文件总行数（train 含增强副本）；
- ``aug_factor`` = 每张训练基础图的份数（原图 + 变体）。

用法::

    python scripts/make_private_dataset.py                  # 构建（默认增强系数 4）
    python scripts/make_private_dataset.py --aug-factor 2   # 训练集过大时降增强系数
    python scripts/make_private_dataset.py --check-only     # 仅抽样画框自检（已有数据集）
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "UAV footage of Merino sheep in South Africa" / "Data.zip"
IMG_PREFIX = "Data/Image Data/Train and Validation Images/"
SPLIT_FILES = (
    "Data/Image Data/train.json",
    "Data/Image Data/val.json",
    "Data/Image Data/test.json",
)
DST_DIR = ROOT / "data" / "private_dataset"
CLASS_NAME = "sheep"

# ---------- 处理参数（阈值与规格，全部写为脚本常量） ----------
SEED = 42                    # 随机种子（划分 / 增强符号 / 自检抽样）
THUMB_SIZE = (1440, 1080)    # 质量评估缩放尺寸（与源图同为 4:3）
BLUR_THRESHOLD = 120.0       # 灰度 Laplacian 方差阈值：低于此值判为模糊剔除
BOX_SIZE = 96                # 目标框统一边长（原图像素），以质心为中心
DENSE_SHEEP_THRESHOLD = 60   # 羊数 ≥ 阈值 → 2×2 密集切片
SLICE_GRID = (2, 2)          # 切片网格（无重叠）
OUT_SIZE = (1440, 1080)      # 全部输出图的统一分辨率
JPEG_QUALITY = 88            # 输出 JPEG 质量
SPLIT_RATIO = (0.70, 0.15, 0.15)  # train/val/test 划分比例（按源图分层）
BUCKETS = ((0, 0), (1, 10), (11, 50), (51, 100), (101, 10 ** 9))  # 羊数分桶
VARIANTS = ("flip", "bright", "rot")   # 增强：水平翻转 / 亮度±20% / 旋转±10°
AUG_SUFFIX = {"flip": "f", "bright": "b", "rot": "r"}


def _load_point_annotations(z: zipfile.ZipFile) -> dict[str, list[tuple[float, float]]]:
    """读取三个 split 的 COCO 标注 → {源图名: [(质心 x, y), …]}。

    质心 = 多边形（segmentation）顶点均值；缺失时回退 bbox 中心。
    """
    points: dict[str, list[tuple[float, float]]] = {}
    for split in SPLIT_FILES:
        data = json.loads(z.read(split))
        by_id = {im["id"]: im["file_name"] for im in data["images"]}
        for ann in data.get("annotations", []):
            seg = ann.get("segmentation")
            if seg and seg[0]:
                flat = seg[0]
                xs, ys = flat[0::2], flat[1::2]
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            else:
                bx, by, bw, bh = ann["bbox"]
                cx, cy = bx + bw / 2.0, by + bh / 2.0
            points.setdefault(by_id[ann["image_id"]], []).append((cx, cy))
    return points


def _read_image(z: zipfile.ZipFile, fname: str) -> np.ndarray | None:
    """从 zip 流式读取并解码一张原图（BGR）。"""
    data = z.read(IMG_PREFIX + fname)
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def _laplacian_var(img_bgr: np.ndarray) -> float:
    """缩放到 THUMB_SIZE 灰度后的 Laplacian 方差（清晰度指标）。"""
    thumb = cv2.resize(img_bgr, THUMB_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _assign_splits(counts: dict[str, int], kept: list[str], seed: int) -> dict[str, str]:
    """按羊数分桶分层做 70/15/15 划分（同源图全部产物跟随同一划分，防泄漏）。"""
    rng = random.Random(seed)
    assign: dict[str, str] = {}
    for lo, hi in BUCKETS:
        group = sorted(f for f in kept if lo <= counts[f] <= hi)
        rng.shuffle(group)
        n = len(group)
        n_val = int(round(n * SPLIT_RATIO[1]))
        n_test = int(round(n * SPLIT_RATIO[2]))
        n_train = n - n_val - n_test
        for i, f in enumerate(group):
            assign[f] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return assign


def _yolo_boxes(pts: list[tuple[float, float]], cw: int, ch: int) -> list[tuple[float, float, float, float]]:
    """质心列表 → 统一 BOX_SIZE 的 YOLO 归一化框（cx, cy, w, h）。

    框以质心为中心、clamp 到画布（cw×ch）内，再同步缩放到 OUT_SIZE 并归一化。
    """
    res: list[tuple[float, float, float, float]] = []
    half = BOX_SIZE / 2.0
    sx, sy = OUT_SIZE[0] / cw, OUT_SIZE[1] / ch
    for px, py in pts:
        px = min(max(float(px), 0.0), float(cw))
        py = min(max(float(py), 0.0), float(ch))
        x1 = min(max(px - half, 0.0), cw)
        x2 = min(max(px + half, 0.0), cw)
        y1 = min(max(py - half, 0.0), ch)
        y2 = min(max(py + half, 0.0), ch)
        x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        res.append(
            (
                (x1 + x2) / 2.0 / OUT_SIZE[0],
                (y1 + y2) / 2.0 / OUT_SIZE[1],
                (x2 - x1) / OUT_SIZE[0],
                (y2 - y1) / OUT_SIZE[1],
            )
        )
    return res


def _augment(
    img: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    variant: str,
    sign: int,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    """确定性增强：flip=水平翻转；bright=亮度±20%（HSV V）；rot=旋转±10°（边角填白）。

    符号 sign 由源图处理序号奇偶决定（+1 / -1），保证可复现。
    旋转时框中心随图像旋转（保持轴对齐近似），中心转出画面的框丢弃。
    """
    cw, ch = OUT_SIZE
    if variant == "flip":
        out = cv2.flip(img, 1)
        return out, [(1.0 - cx, cy, bw, bh) for cx, cy, bw, bh in boxes]
    if variant == "bright":
        factor = 1.2 if sign > 0 else 0.8
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 2] = np.clip(hsv[..., 2].astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), list(boxes)
    # variant == "rot"
    angle = 10.0 if sign > 0 else -10.0
    m = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), angle, 1.0)
    out = cv2.warpAffine(img, m, (cw, ch), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    res: list[tuple[float, float, float, float]] = []
    for cx, cy, bw, bh in boxes:
        px, py = cx * cw, cy * ch
        nx = m[0, 0] * px + m[0, 1] * py + m[0, 2]
        ny = m[1, 0] * px + m[1, 1] * py + m[1, 2]
        if not (0.0 <= nx < cw and 0.0 <= ny < ch):
            continue  # 中心旋转出图 → 丢弃该框
        pw, ph = bw * cw, bh * ch
        x1 = min(max(nx - pw / 2.0, 0.0), float(cw))
        x2 = min(max(nx + pw / 2.0, 0.0), float(cw))
        y1 = min(max(ny - ph / 2.0, 0.0), float(ch))
        y2 = min(max(ny + ph / 2.0, 0.0), float(ch))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        res.append(
            ((x1 + x2) / 2.0 / cw, (y1 + y2) / 2.0 / ch, (x2 - x1) / cw, (y2 - y1) / ch)
        )
    return out, res


def _write_sample(
    dst: Path,
    split: str,
    name: str,
    img: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
) -> None:
    """写出一张图（JPEG）与对应 YOLO 标签（txt，可空=纯背景）。"""
    ok = cv2.imwrite(
        str(dst / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ok:
        raise SystemExit(f"写入失败：{dst / 'images' / split / (name + '.jpg')}")
    lines = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cx, cy, bw, bh in boxes]
    (dst / "labels" / split / f"{name}.txt").write_text(
        ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
    )


def build(zip_path: Path, dst: Path, seed: int, aug_factor: int, limit: int | None) -> dict:
    """执行完整处理管线并写出数据集、dataset.yaml、manifest.json，返回 stats。"""
    t0 = time.time()
    # 清理上次产物的 images/labels（保证重跑幂等，避免降增强后残留旧文件）
    for sub in ("images", "labels"):
        p = dst / sub
        if p.exists():
            shutil.rmtree(p)
    for split in ("train", "val", "test"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(zip_path)
    points = _load_point_annotations(z)
    files = sorted(points)
    if limit:
        files = files[:limit]
        print(f"[warn] 调试模式：仅处理前 {limit} 张源图，产物不可用于正式交付")
    counts = {f: len(points[f]) for f in files}

    # ---- 质量筛除：Laplacian 方差 < 阈值 → 模糊剔除 ----
    quality: dict[str, float] = {}
    dropped: dict[str, str] = {}
    for i, fname in enumerate(files, start=1):
        raw = _read_image(z, fname)
        if raw is None:
            quality[fname] = 0.0
            dropped[fname] = "decode_error"
            continue
        quality[fname] = _laplacian_var(raw)
        if quality[fname] < BLUR_THRESHOLD:
            dropped[fname] = f"blur(lap_var={quality[fname]:.1f})"
        if i % 25 == 0 or i == len(files):
            print(f"[质量筛除] {i}/{len(files)}", flush=True)
    kept = [f for f in files if f not in dropped]
    print(f"质量筛除完成：保留 {len(kept)}，剔除 {len(dropped)}")
    for f in dropped:
        print(f"  - 剔除 {f}（{dropped[f]}）")

    # ---- 划分（分层 + 同源防泄漏） ----
    assign = _assign_splits(counts, kept, seed)
    if limit is None:
        assert set(assign) == set(kept), "划分必须覆盖全部保留源图"

    # ---- 逐源图处理：切片/整图 → 统一分辨率 → 写出（train 附增强变体） ----
    idx_all = {f: i for i, f in enumerate(files)}
    variants = VARIANTS[: max(0, aug_factor - 1)]
    records: list[dict] = []
    for i, fname in enumerate(files, start=1):
        rec: dict = {
            "file": fname,
            "sheep": counts[fname],
            "laplacian_var": round(quality[fname], 1),
            "kept": fname not in dropped,
            "reason": dropped.get(fname),
            "split": assign.get(fname),
            "n_slices": 0,
            "mode": None,
            "outputs": [],
        }
        if fname in dropped:
            records.append(rec)
            continue
        split = assign[fname]
        raw = _read_image(z, fname)
        h0, w0 = raw.shape[:2]
        pts = points[fname]
        stem = Path(fname).stem.replace(" ", "_")
        sliced = counts[fname] >= DENSE_SHEEP_THRESHOLD
        if sliced:
            cw, ch = w0 // SLICE_GRID[0], h0 // SLICE_GRID[1]
            tiles = [
                (c * cw, r * ch, cw, ch)  # (x0, y0, 宽, 高)
                for r in range(SLICE_GRID[0])
                for c in range(SLICE_GRID[1])
            ]
        else:
            tiles = [(0, 0, w0, h0)]
        rec["n_slices"] = len(tiles)
        rec["mode"] = "sliced" if sliced else "whole"
        sign = 1 if idx_all[fname] % 2 == 0 else -1
        for x0, y0, tw, th in tiles:
            sub = raw[y0 : y0 + th, x0 : x0 + tw]
            sub_pts = [
                (px - x0, py - y0)
                for px, py in pts
                if x0 <= px < x0 + tw and y0 <= py < y0 + th
            ]
            img_out = cv2.resize(sub, OUT_SIZE, interpolation=cv2.INTER_AREA)
            boxes = _yolo_boxes(sub_pts, tw, th)
            name = stem if not sliced else f"{stem}_s{y0 // ch}{x0 // cw}"
            _write_sample(dst, split, name, img_out, boxes)
            rec["outputs"].append(name)
            if split == "train":
                for variant in variants:
                    a_img, a_boxes = _augment(img_out, boxes, variant, sign)
                    vname = f"{name}_{AUG_SUFFIX[variant]}"
                    _write_sample(dst, split, vname, a_img, a_boxes)
                    rec["outputs"].append(vname)
        records.append(rec)
        if i % 20 == 0 or i == len(files):
            print(f"[处理输出] {i}/{len(files)}", flush=True)
    z.close()

    # ---- 统计（从磁盘实际产物计数，保证数字真实） ----
    def _count_files(p: Path, pattern: str) -> int:
        return len(list(p.glob(pattern)))

    def _count_instances(p: Path) -> int:
        return sum(
            len([ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()])
            for f in p.glob("*.txt")
        )

    patches_added = sum(r["n_slices"] - 1 for r in records if r["kept"])
    stats = {
        "raw_images": len(files),
        "kept_images": len(kept),
        "dropped_images": len(dropped),
        "patches_added": patches_added,
        "total_train": _count_files(dst / "images" / "train", "*.jpg"),
        "total_val": _count_files(dst / "images" / "val", "*.jpg"),
        "total_test": _count_files(dst / "images" / "test", "*.jpg"),
        "instances_train": _count_instances(dst / "labels" / "train"),
        "instances_val": _count_instances(dst / "labels" / "val"),
        "instances_test": _count_instances(dst / "labels" / "test"),
        "aug_factor": aug_factor,
    }

    # ---- dataset.yaml ----
    # 不写 path：ultralytics 缺省以 yaml 所在目录为数据集根，仓库换台机器也能直接跑
    yaml_text = (
        f"# AI牧易通私有羊群检测数据集（scripts/make_private_dataset.py 生成，seed={seed}，aug_factor={aug_factor}）\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )
    (dst / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    # ---- manifest.json ----
    manifest = {
        "name": "AI牧易通私有羊群检测数据集（UAV Merino 公开数据集确定性派生）",
        "source": {
            "dataset": "UAV footage of Merino sheep in South Africa",
            "authors": "Theart, Rensu; Schreve, Kristiaan",
            "publisher": "Mendeley Data, 2024-03-28",
            "doi_url": "https://data.mendeley.com/datasets/nppwbkdf85/1",
            "license": "CC BY 4.0（Creative Commons Attribution 4.0 International）",
            # 只记相对位置，不写入本机绝对路径（清单会随包公开发布）
            "archive": str(Path("UAV footage of Merino sheep in South Africa") / zip_path.name),
            "note": (
                "公开数据集 → 本团队私有派生集：质量筛除 + 点标注转统一框 + 密集切片 + "
                "统一分辨率 + train 确定性增强 + 分层划分（同源切片/变体不跨划分，防泄漏）"
            ),
        },
        "params": {
            "seed": seed,
            "blur_threshold": BLUR_THRESHOLD,
            "box_size": BOX_SIZE,
            "dense_sheep_threshold": DENSE_SHEEP_THRESHOLD,
            "slice_grid": list(SLICE_GRID),
            "output_size": list(OUT_SIZE),
            "jpeg_quality": JPEG_QUALITY,
            "split_ratio": list(SPLIT_RATIO),
            "buckets": ["0", "1-10", "11-50", "51-100", ">100"],
            "aug_factor": aug_factor,
            "aug_variants": [f"{v}({AUG_SUFFIX[v]})" for v in variants],
        },
        "stats": stats,
        "definitions": {
            "patches_added": "因密集切片新增的图片数（Σ(片数-1)，仅统计保留图）",
            "instances_train/val/test": "对应划分 label 总框数（train 含增强副本）",
            "aug_factor": "每张训练基础图的份数（原图 + aug_factor-1 个变体）",
        },
        "sources": records,
    }
    (dst / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    print(f"\n=== 私有数据集统计（{dst}） ===")
    for k in (
        "raw_images", "kept_images", "dropped_images", "patches_added",
        "total_train", "total_val", "total_test",
        "instances_train", "instances_val", "instances_test", "aug_factor",
    ):
        print(f"{k}: {stats[k]}")
    print(f"耗时 {time.time() - t0:.1f}s")
    return stats


def run_check(dst: Path, seed: int = SEED, count: int = 3) -> None:
    """自检：随机抽 count 张训练图，把 YOLO 标注框画上去另存 _check_sample*.jpg。"""
    train_dir = dst / "images" / "train"
    files = sorted(p.name for p in train_dir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"没有可自检的训练图：{train_dir}")
    rng = random.Random(seed)
    for i, name in enumerate(rng.sample(files, min(count, len(files))), start=1):
        img = cv2.imread(str(train_dir / name))
        if img is None:
            raise SystemExit(f"自检读取失败：{train_dir / name}")
        h, w = img.shape[:2]
        label = (dst / "labels" / "train" / f"{Path(name).stem}.txt").read_text(encoding="utf-8")
        n = 0
        for line in label.splitlines():
            if not line.strip():
                continue
            _, cx, cy, bw, bh = (float(v) for v in line.split())
            x1, y1 = int((cx - bw / 2.0) * w), int((cy - bh / 2.0) * h)
            x2, y2 = int((cx + bw / 2.0) * w), int((cy + bh / 2.0) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            n += 1
        out = dst / f"_check_sample{i}.jpg"
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[自检] {out}  <- {name}  框数={n}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--zip", type=Path, default=ZIP_PATH, help="公开数据集 zip 路径")
    parser.add_argument("--dst", type=Path, default=DST_DIR, help="输出目录")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子（默认 42）")
    parser.add_argument(
        "--aug-factor", type=int, default=4, help="train 增强系数（默认 4：原图+3 变体；可选 2）"
    )
    parser.add_argument("--limit", type=int, default=0, help="调试：仅处理前 N 张源图（0=全部）")
    parser.add_argument("--check", action="store_true", help="构建后抽样画框自检")
    parser.add_argument("--check-only", action="store_true", help="跳过构建，仅对已有数据集自检")
    args = parser.parse_args()

    if args.check_only:
        run_check(args.dst, seed=args.seed)
        return
    if not args.zip.exists():
        raise SystemExit(
            f"缺少数据集：{args.zip}\n"
            "请从 https://data.mendeley.com/datasets/nppwbkdf85/1 下载 "
            "《UAV footage of Merino sheep in South Africa》解压到项目根目录。"
        )
    build(args.zip, args.dst, args.seed, args.aug_factor, args.limit or None)
    if args.check:
        run_check(args.dst, seed=args.seed)


if __name__ == "__main__":
    main()
