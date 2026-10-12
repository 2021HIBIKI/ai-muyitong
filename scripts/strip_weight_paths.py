# -*- coding: utf-8 -*-
"""清除权重文件中内嵌的本机绝对路径（公开发布前的信息卫生处理）。

ultralytics 训练时把 model / data / project / save_dir 等参数原样写进 checkpoint，
于是交付出去的 .pt 会带着训练机器上的绝对路径（含 Windows 用户名与目录结构）。
本脚本把这些字符串改写为**仓库相对路径**（不在仓库内则只留文件名），
模型权重本身一个字节都不动，因此精度不受影响。

用法::

    python scripts/strip_weight_paths.py models/yolo11n_private.pt           # 就地清洗
    python scripts/strip_weight_paths.py models/yolo11n_private.pt --check   # 只报告，不写回

重新训练后请再跑一次；tests/test_private_dataset.py 有守卫会拦住带绝对路径的产物。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
# 盘符路径（C:\ 或 C:/）、UNC 路径、以及本机 Git-Bash 风格的解释器路径
ABS_PATTERN = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/[a-z]+/Anaconda/)")
# 字节层面的痕迹检测：必须“像路径”——盘符/Anaconda 之后要有 ≥4 个路径字符，
# 且盘符前不能紧跟字母数字（否则 https:// 里的 "s://" 会被当成盘符路径）。
DRIVE_HINT = re.compile(rb"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/Anaconda/)[A-Za-z0-9_.\- /\\]{4,}")


def _neutral(s: str) -> str:
    """仓库内 → 相对路径（正斜杠）；仓库外 → 仅文件名。"""
    try:
        return Path(s).resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return Path(s).name


def scrub(obj, found: list[str]):
    """递归遍历 pickle 出来的对象图，就地改写绝对路径字符串。"""
    if isinstance(obj, str):
        if ABS_PATTERN.match(obj):
            found.append(obj)
            return _neutral(obj)
        return obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = scrub(v, found)
        return obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = scrub(v, found)
        return obj
    if isinstance(obj, tuple):
        return tuple(scrub(v, found) for v in obj)
    if hasattr(obj, "__dict__"):  # ultralytics 的 IterableSimpleNamespace 等
        for k, v in list(vars(obj).items()):
            setattr(obj, k, scrub(v, found))
    return obj


def count_abs_paths(path: Path) -> int:
    """不看 pickle 结构，直接在字节层面数一下盘符路径痕迹（用于校验与测试）。"""
    return len(DRIVE_HINT.findall(path.read_bytes()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights", type=Path, nargs="+")
    ap.add_argument("--check", action="store_true", help="只报告，不写回")
    args = ap.parse_args()

    for path in args.weights:
        before = count_abs_paths(path)
        if not before:
            print(f"{path}: 无绝对路径痕迹，跳过")
            continue
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        found: list[str] = []
        ckpt = scrub(ckpt, found)
        if not found:
            print(f"{path}: 结构内无可改写的绝对路径（{before} 处字节痕迹为误报）")
            continue
        print(f"{path}: 命中 {len(found)} 处")
        for s in sorted(set(found)):
            print("   ", s, "→", _neutral(s))
        if args.check:
            continue
        torch.save(ckpt, str(path))
        after = count_abs_paths(path)
        print(f"{path}: 写回完成，残留痕迹 {after}（应为 0）")
        if after:
            raise SystemExit("仍有绝对路径残留，请检查 ABS_PATTERN")


if __name__ == "__main__":
    main()
