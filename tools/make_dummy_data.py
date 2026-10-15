"""make_dummy_data.py —— 生成一份「合成鸟类数据」，用于在没有 CUB 数据集时跑通整条流水线。

目录结构与真实数据完全一致：
    <out>/<类别文件夹>/<图片.jpg>  与同名 <图片.txt>（10 行描述）

每类用一个固定的主打色 + 纹理，使 image-only 也能学到东西，便于验证代码正确性。

用法：
    python tools/make_dummy_data.py --out datasets --classes 5 --per-class 20 --size 224
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

# 每类一套（颜色, 描述词），保证"图-文-标签"三者一致，便于验证对齐是否有效
PALETTE = [
    ((210, 60, 60), ["red", "crimson", "scarlet"]),
    ((60, 120, 210), ["blue", "azure", "cobalt"]),
    ((70, 170, 90), ["green", "emerald", "olive"]),
    ((220, 190, 60), ["yellow", "golden", "amber"]),
    ((150, 90, 200), ["purple", "violet", "lavender"]),
    ((230, 130, 60), ["orange", "tangerine", "rust"]),
    ((60, 60, 60), ["black", "dark", "charcoal"]),
    ((240, 240, 240), ["white", "pale", "ivory"]),
]


def make_image(color, size: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), color)
    draw = ImageDraw.Draw(img)
    # 随机画几条同色系深浅的条纹，制造一点纹理变化
    for _ in range(rng.randint(3, 6)):
        shade = tuple(min(255, max(0, ch + rng.randint(-50, 50))) for ch in color)
        x0, y0 = rng.randint(0, size), rng.randint(0, size)
        x1, y1 = rng.randint(0, size), rng.randint(0, size)
        draw.line((x0, y0, x1, y1), fill=shade, width=rng.randint(4, 14))
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="生成合成鸟类数据")
    parser.add_argument("--out", default="datasets")
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--captions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[合成数据] 输出目录：{out}")

    for ci in range(args.classes):
        color, words = PALETTE[ci % len(PALETTE)]
        cname = f"{ci + 1:03d}.{words[0].capitalize()}_bird"
        cdir = out / cname
        cdir.mkdir(parents=True, exist_ok=True)
        for j in range(args.per_class):
            stem = f"{words[0].capitalize()}_bird_{ci + 1:03d}_{j:04d}"
            make_image(color, args.size, rng.randint(0, 10 ** 6)).save(
                cdir / f"{stem}.jpg", quality=85
            )
            lines = []
            for _ in range(args.captions):
                w = rng.choice(words)
                lines.append(f"a {w} bird with {rng.choice(['a short beak', 'long wings', 'a small body'])}")
            (cdir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"    类别 {ci + 1}/{args.classes}: {cname}  写入 {args.per_class} 张图片 + 描述")

    print("[合成数据] 完成。下一步：python 01_prepare_data.py")


if __name__ == "__main__":
    main()