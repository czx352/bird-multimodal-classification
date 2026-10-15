"""01_prepare_data.py —— 数据准备（手册 §1.3）。

把「类别文件夹 + 同名 txt」的原始结构，按 6:2:2 分层划分，
生成一个冻结划分的 annotations.json。

用法：
    python 01_prepare_data.py                       # 用 config.yaml 里的路径
    python 01_prepare_data.py --data-root datasets --out datasets/annotations.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本无论从哪个目录运行都能 import birdmm
sys.path.insert(0, str(Path(__file__).resolve().parent))

from birdmm.data import build_annotations          # noqa: E402
from birdmm.utils import load_config               # noqa: E402


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="生成 annotations.json（6:2:2 划分）")
    parser.add_argument("--data-root", default=cfg["data"]["root"], help="原始数据根目录")
    parser.add_argument("--out", default=cfg["data"]["annotations"], help="输出 json 路径")
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--captions", type=int, default=cfg["data"]["captions_per_image"])
    args = parser.parse_args()

    ratio = tuple(cfg["data"]["split"])
    print(f"[数据] 扫描目录：{args.data_root}")
    print(f"[数据] 划分比例：{ratio[0]}:{ratio[1]}:{ratio[2]}，随机种子：{args.seed}")

    ann = build_annotations(
        data_root=args.data_root,
        out_path=args.out,
        split=ratio,
        seed=args.seed,
        captions_per_image=args.captions,
    )

    meta = ann["meta"]
    print(f"[数据] 类别数：{meta['num_classes']}")
    print(f"[数据] 样本数：{meta['counts']}")
    print(f"[数据] 已写出标注文件：{args.out}")
    print("[数据] 类别示例：")
    for i, c in enumerate(ann["classes"][:5]):
        print(f"    {i:>3}  {c}   ->  prompt: \"{ann['class_prompts'][c]}\"")


if __name__ == "__main__":
    main()