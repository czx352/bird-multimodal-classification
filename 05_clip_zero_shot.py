"""05_clip_zero_shot.py —— CLIP 思想实验（手册 §5）。

    image ─► 图像向量 ┐
                       ├──► 余弦相似度最大者 = 匹配
    text  ─► 文本向量 ┘

没有分类器：类别不是被"输出"，而是通过"和类别描述的相似度"被比较出来。
本脚本验证三种能力中的两种：
    1. 零样本分类：类别名写成提示词 -> 比相似度 -> 零样本 Accuracy
    2. 图文检索  ：image->text 排序 -> R@1 / R@5

用法：
    python 05_clip_zero_shot.py
    python 05_clip_zero_shot.py --model-name openai/clip-vit-base-patch32 --limit 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from birdmm.clip_utils import (encode_images, encode_texts, retrieval_recall,  # noqa: E402
                               zero_shot_classify)
from birdmm.metrics import classification_metrics, format_metrics       # noqa: E402
from birdmm.utils import ensure_dir, get_device, load_config, set_seed  # noqa: E402
from birdmm.data import load_annotations                                # noqa: E402


def main() -> None:
    cfg = load_config()
    c = cfg["clip"]
    parser = argparse.ArgumentParser(description="CLIP 零样本分类 + 图文检索")
    parser.add_argument("--model-name", default=c["model_name"])
    parser.add_argument("--batch-size", type=int, default=c["batch_size"])
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 条（0 表示全部）")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="runs/clip")
    args = parser.parse_args()

    set_seed(cfg["seed"])
    device = get_device(args.device)
    print(f"[CLIP] device={device}, model={args.model_name}")

    try:
        from birdmm.clip_utils import load_clip
        model, processor = load_clip(args.model_name, str(device))
    except Exception as exc:
        print(f"[CLIP] 无法加载 {args.model_name}（需要安装 transformers 且能联网下载权重）：{exc}")
        return

    ann = load_annotations(cfg["data"]["annotations"])
    data_root = Path(cfg["data"]["root"])
    classes = ann["classes"]
    prompts = ann["class_prompts"]

    # ---- 1) 零样本分类：把所有类别提示词编码成文本向量 ----
    class_texts = [prompts[c_] for c_ in classes]
    class_emb = encode_texts(model, processor, class_texts, str(device))
    print(f"[CLIP] 类别提示词：{len(class_texts)} 条，文本向量 {tuple(class_emb.shape)}")

    test = ann["splits"]["test"]
    if args.limit:
        test = test[: args.limit]

    y_true, y_pred = [], []
    img_embs, txt_embs = [], []
    for start in range(0, len(test), args.batch_size):
        chunk = test[start : start + args.batch_size]
        images = [Image.open(data_root / s["image_path"]).convert("RGB") for s in chunk]
        img_emb = encode_images(model, processor, images, str(device))
        pred = zero_shot_classify(img_emb, class_emb)
        y_pred.extend(pred.cpu().tolist())
        y_true.extend(s["label"] for s in chunk)
        # 检索用：每张图配它的第一条描述
        captions = [s["captions"][0] if s["captions"] else "" for s in chunk]
        img_embs.append(img_emb)
        txt_embs.append(encode_texts(model, processor, captions, str(device)))
        print(f"    零样本评测进度：{min(start + args.batch_size, len(test))}/{len(test)}", end="\r")
    print()

    print("\n[CLIP] 零样本分类（无分类器，靠相似度比较）：")
    zero_shot = classification_metrics(y_true, y_pred)
    print(format_metrics(zero_shot))

    # ---- 2) 图文检索：image->text 的 R@1 / R@5 ----
    img_embs = torch.cat(img_embs, dim=0)
    txt_embs = torch.cat(txt_embs, dim=0)
    recall = retrieval_recall(img_embs, txt_embs, ks=(1, 5))
    print("[CLIP] 图文检索（image -> text）：")
    for k, v in recall.items():
        print(f"    {k}: {v:.4f}")

    out_dir = ensure_dir(Path(__file__).resolve().parent / args.out_dir)
    with open(Path(out_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model_name, "zero_shot": zero_shot,
                   "retrieval": recall, "num_eval": len(test)},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[CLIP] 结果已保存到：{out_dir}")


if __name__ == "__main__":
    main()