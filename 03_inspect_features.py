"""03_inspect_features.py —— 模型拆解与特征分析（手册 §3）。

把两个编码器拆开，逐层观察「张量是怎么流动的」：

    图片 [3,224,224] ─► ResNet 主干（去掉 avgpool/fc）─► 特征图 [512,7,7]
                                ├── 全局平均池化 ─► [512]        （Early/Late/Gated 用）
                                └── 展平 ─► [49, 512]            （Attention 用）

    文本 "a small blue bird" ─► tokenizer ─► ids ─► embedding 表(30522×768)
                             ─► Transformer ─► [L, 768] ─► pooling ─► [768]

用法：
    python 03_inspect_features.py                      # 打印图像 / 文本特征流程
    python 03_inspect_features.py --checkpoint runs/multimodal_cross_attention/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from birdmm.data import build_image_transform, load_annotations        # noqa: E402
from birdmm.utils import ensure_dir, get_device, load_config, set_seed  # noqa: E402


def inspect_image_encoder(device, backbone: str, image_size: int = 224) -> None:
    """拆开图像编码器：特征图 -> token -> 池化向量。"""
    from birdmm.models.image_encoder import build_image_encoder

    print("\n" + "=" * 68)
    print("① Image Encoder —— 一张图如何变成向量 / 视觉 token")
    print("=" * 68)
    enc = build_image_encoder(backbone=backbone, pretrained=False, freeze=False,
                              out_dim=None).to(device).eval()
    x = torch.randn(2, 3, image_size, image_size, device=device)
    with torch.no_grad():
        fmap = enc.body(x)
        tokens = fmap.flatten(2).transpose(1, 2)
        pooled = enc.pool(fmap).flatten(1)
        vec = enc(x)
        tok = enc(x, return_tokens=True)
    print(f"    输入图像              : {tuple(x.shape)}")
    print(f"    主干输出特征图        : {tuple(fmap.shape)}   # [B, C, h, w]")
    print(f"    flatten(2)            : {tuple(fmap.flatten(2).shape)}   # [B, C, h*w]")
    print(f"    transpose(1,2)        : {tuple(tokens.shape)}   # [B, h*w, C] —— {tokens.shape[1]} 个'视觉单词'")
    print(f"    AdaptiveAvgPool2d(1)  : {tuple(pooled.shape)}   # [B, C] 全局向量（Early/Late/Gated 用）")
    print(f"    encoder(x)            : {tuple(vec.shape)}")
    print(f"    encoder(x, tokens)    : {tuple(tok.shape)}")
    print(f"    维度一致性检查        : out_dim={enc.out_dim}, feature_dim={enc.feature_dim}")


def inspect_text_encoder(device, model_name: str, sample_text: str) -> None:
    """拆开文本编码器：tokenizer -> embedding -> Transformer -> pooling。"""
    try:
        from birdmm.models.text_encoder import build_text_encoder, load_tokenizer
    except ImportError as exc:
        print(f"\n[跳过文本编码器] 未安装 transformers（{exc}）。"
              f"\n    安装：pip install transformers，并联网下载 {model_name}")
        return

    print("\n" + "=" * 68)
    print("② Text Encoder —— 一句话如何变成向量")
    print("=" * 68)
    try:
        tokenizer = load_tokenizer(model_name)
        enc = build_text_encoder(model_name=model_name, pretrained=True,
                                 freeze=True, out_dim=None, pooling="mean").to(device).eval()
    except Exception as exc:
        print(f"    [跳过] 加载 {model_name} 失败：{exc}")
        return

    batch = tokenizer([sample_text, "a bird with a long beak"], padding="max_length",
                      truncation=True, max_length=16, return_tensors="pt")
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)

    embed = enc.bert.get_input_embeddings()
    with torch.no_grad():
        token_emb = embed(ids)
        tokens, pooled = enc.encode_tokens(ids, mask)

    print(f"    词表示例              : {tokenizer.convert_ids_to_tokens(ids[0].tolist())}")
    print(f"    input_ids             : {tuple(ids.shape)}   # [B, L]，L={ids.shape[1]}")
    print(f"    embedding 表大小      : {tuple(embed.weight.shape)}   # 30522 × 768（bert-base-uncased）")
    print(f"    Token Embedding 输出  : {tuple(token_emb.shape)}   # [B, L, 768]")
    print(f"    Transformer 输出       : {tuple(tokens.shape)}   # 每个 token 一个 768 维上下文向量")
    print(f"    Mean Pooling 输出      : {tuple(pooled.shape)}   # [B, 768] 句向量")

    # 演示 attention_mask 的作用：不过滤 padding 会把句向量拉向 0
    from birdmm.models.text_encoder import masked_mean
    naive = tokens.mean(dim=1)
    masked = masked_mean(tokens, mask)
    print(f"    不做 mask 的平均       : norm={naive.norm(dim=-1).mean():.4f}（被 [PAD] 稀释）")
    print(f"    按 mask 求平均         : norm={masked.norm(dim=-1).mean():.4f}（只用真实 token）")


def visualize_attention(checkpoint: str, device, cfg, out_dir: Path) -> None:
    """从训练好的 Cross-Attention 模型里导出注意力权重并（尽力）可视化。"""
    try:
        from birdmm.data import MultimodalBirdDataset
        from birdmm.models.multimodal import build_multimodal_model
        from birdmm.models.text_encoder import load_tokenizer
    except ImportError as exc:
        print(f"[跳过注意力可视化] {exc}")
        return

    print("\n" + "=" * 68)
    print("③ Cross-Attention —— 模型在看哪个词")
    print("=" * 68)
    try:
        tokenizer = load_tokenizer(cfg["text_encoder"]["model_name"])
        ann = load_annotations(cfg["data"]["annotations"])
        model = build_multimodal_model(cfg, ann["meta"]["num_classes"]).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
        model.eval()
        ds = MultimodalBirdDataset(cfg["data"]["annotations"], "test", cfg["data"]["root"],
                                   mode="multimodal", tokenizer=tokenizer,
                                   max_length=cfg["text_encoder"]["max_length"])
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=1)))
        with torch.no_grad():
            out = model(batch["image"].to(device), batch["input_ids"].to(device),
                        batch["attention_mask"].to(device))
    except Exception as exc:
        print(f"    [跳过] 加载/patch 失败：{exc}")
        return

    attn = out.get("attn_weights")
    if attn is None:
        print("    该 checkpoint 的融合方式不是 cross_attention，无注意力权重。")
        return
    attn = attn[0]                                   # [heads, N_img, L]
    words = tokenizer.convert_ids_to_tokens(batch["input_ids"][0].tolist())
    mean_attn = attn.mean(0).mean(0)                 # 平均到 [L]：每个词被关注的强度
    print(f"    注意力张量形状        : {tuple(attn.shape)}  # [heads, 49, L]")
    top = torch.topk(mean_attn, k=min(5, mean_attn.numel()))
    print("    图像最关注的 5 个词：")
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"        {words[idx]:<15} {score:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(range(len(words)), mean_attn.numpy())
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=90, fontsize=7)
        ax.set_title("Cross-Attention: image tokens -> text words (mean over heads & positions)")
        fig.tight_layout()
        out_png = Path(out_dir) / "cross_attention_weights.png"
        fig.savefig(out_png, dpi=150)
        print(f"    注意力可视化已保存：{out_png}")
    except ImportError:
        print("    （未安装 matplotlib，仅打印数值。安装后可生成注意力热图。）")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="编码器拆解与特征分析")
    parser.add_argument("--backbone", default=cfg["image_encoder"]["backbone"])
    parser.add_argument("--text-model", default=cfg["text_encoder"]["model_name"])
    parser.add_argument("--text", default="a small blue bird with a long beak")
    parser.add_argument("--checkpoint", default=None, help="可选的 multimodal best.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(cfg["seed"])
    device = get_device(args.device)
    print(f"[Inspect] device={device}")

    inspect_image_encoder(device, args.backbone, cfg["data"]["image_size"])
    inspect_text_encoder(device, args.text_model, args.text)

    out_dir = ensure_dir(Path(__file__).resolve().parent / "runs/inspect")
    if args.checkpoint:
        visualize_attention(args.checkpoint, device, cfg, out_dir)
    print(f"\n[Inspect] 分析完成，产物目录：{out_dir}")


if __name__ == "__main__":
    main()