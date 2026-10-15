"""04_train_multimodal.py —— 多模态训练 / 融合方式消融（手册 §4）。

   图片 ─► ImageEncoder ─► 图像向量 / token ─┐
                                              ├─► Fusion ─► Classifier ─► 鸟的种类
   描述 ─► TextEncoder  ─► 文本向量 / token ─┘

四种融合接口完全一致，换融合方式只需要改配置里的 fusion.name 一个字段：
    early | late | cross_attention | gated
（也可以用命令行 --fusion 临时覆盖，用于一次性跑完消融对比。）

用法：
    python 04_train_multimodal.py --fusion cross_attention
    python 04_train_multimodal.py --fusion early --epochs 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from birdmm.data import build_dataloaders, load_annotations             # noqa: E402
from birdmm.metrics import classification_metrics, format_metrics       # noqa: E402
from birdmm.models.multimodal import build_multimodal_model             # noqa: E402
from birdmm.models.text_encoder import load_tokenizer                   # noqa: E402
from birdmm.utils import count_parameters, ensure_dir, get_device, load_config, set_seed  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:                                                       # pragma: no cover
    def tqdm(x, **k):
        return x


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """评测；同时把 Cross-Attention 的注意力权重 / Gated 的门控值收集起来做分析。"""
    model.eval()
    y_true, y_pred = [], []
    collected = {"attn_weights": None, "gate": None, "weight": None}
    for batch in loader:
        images = batch["image"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(images, ids, mask)
        y_pred.extend(out["logits"].argmax(dim=-1).cpu().tolist())
        y_true.extend(batch["label"].tolist())
        for key in collected:
            if key in out and collected[key] is None:
                collected[key] = out[key].detach().cpu()
    metrics = classification_metrics(y_true, y_pred)
    metrics["_probe"] = collected
    return metrics


def main() -> None:
    cfg = load_config()
    t = cfg["train"]
    parser = argparse.ArgumentParser(description="多模态训练（融合方式可切换）")
    parser.add_argument("--fusion", default=cfg["fusion"]["name"],
                        choices=["early", "late", "cross_attention", "gated"])
    parser.add_argument("--epochs", type=int, default=t["epochs"])
    parser.add_argument("--batch-size", type=int, default=t["batch_size"])
    parser.add_argument("--lr", type=float, default=t["lr"])
    parser.add_argument("--device", default=t.get("device", "auto"))
    parser.add_argument("--out-dir", default=None, help="默认 runs/multimodal_<fusion>")
    args = parser.parse_args()

    cfg["fusion"]["name"] = args.fusion
    set_seed(cfg["seed"])
    device = get_device(args.device)
    ann = load_annotations(cfg["data"]["annotations"])
    num_classes = ann["meta"]["num_classes"]
    out_dir = args.out_dir or f"runs/multimodal_{args.fusion}"
    print(f"[MultiModal] fusion={args.fusion}, device={device}, num_classes={num_classes}")

    try:
        tokenizer = load_tokenizer(cfg["text_encoder"]["model_name"])
    except ImportError as exc:
        print(f"[MultiModal] {exc}")
        return
    loaders = build_dataloaders(cfg, mode="multimodal", tokenizer=tokenizer,
                                batch_size=args.batch_size)
    print(f"[MultiModal] 样本数：train={len(loaders['train'].dataset)}, "
          f"val={len(loaders['val'].dataset)}, test={len(loaders['test'].dataset)}")

    model = build_multimodal_model(cfg, num_classes).to(device)
    trainable, total = count_parameters(model)
    print(f"[MultiModal] 参数量：可训练 {trainable:,} / 总 {total:,}")

    # 只优化需要梯度的参数（主干冻结时即"只训练融合层 + 分类头"）
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=t["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    history = []
    print("\n[MultiModal] 开始训练 …")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total_n = 0.0, 0, 0
        for batch in tqdm(loaders["train"], desc=f"Epoch {epoch}/{args.epochs}"):
            images = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            out = model(images, ids, mask)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out["logits"].argmax(dim=-1) == labels).sum().item()
            total_n += labels.size(0)

        train_acc = correct / max(total_n, 1)
        val_metrics = evaluate(model, loaders["val"], device)
        val_metrics.pop("_probe", None)
        history.append({"epoch": epoch, "loss": total_loss / max(len(loaders["train"]), 1),
                        "train_acc": train_acc, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"    epoch {epoch:>2}  loss={history[-1]['loss']:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
              f"val_f1={val_metrics['f1']:.4f}")
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n[MultiModal] 最优验证准确率：{best_val:.4f}，在测试集上评估：")
    test_metrics = evaluate(model, loaders["test"], device)
    probe = test_metrics.pop("_probe")
    print(format_metrics(test_metrics))

    # 融合模块的可解释性输出
    if probe.get("weight") is not None:
        print(f"    [Late Fusion] 学到的模态权重 w（越大越信文本）：{probe['weight'].mean():.3f}")
    if probe.get("gate") is not None:
        print(f"    [Gated Fusion] 门控 a 的逐样本均值：{probe['gate'].mean():.3f}，"
              f"std={probe['gate'].std():.3f}")
    if probe.get("attn_weights") is not None:
        print(f"    [Cross-Attention] 注意力权重形状 [B, heads, N_img, L] = "
              f"{tuple(probe['attn_weights'].shape)}（可用 03_inspect_features.py 可视化）")

    out_path = ensure_dir(Path(__file__).resolve().parent / out_dir)
    torch.save({"model": model.state_dict(), "fusion": args.fusion,
                "num_classes": num_classes, "args": vars(args)}, Path(out_path) / "best.pt")
    with open(Path(out_path) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"fusion": args.fusion, "test": test_metrics, "best_val_acc": best_val,
                   "history": history}, f, ensure_ascii=False, indent=2)
    print(f"[MultiModal] 结果已保存到：{out_path}")


if __name__ == "__main__":
    main()