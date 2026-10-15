"""02_singleModal_TextOnly.py —— 单模态 Text-Only 训练（手册 §2.2）。

输入：描述文本（每张图 10 条，训练时随机取一条）+ 类别标签
模型：BERT 文本编码器 + 分类头（手册 §2.2 提到 CLIP —— 其文本塔同样是 Transformer 编码器，
      本脚本用 BERT 复现同一套"分词 -> 上下文向量 -> 池化 -> 分类"的流程；
      CLIP 的"共同语义空间 / 零样本"用法见 05_clip_zero_shot.py）
输出：类别

参考原始版本：02_singleModal_TextOnly.py（已改为读取 annotations.json 的分层划分）

用法：
    python 02_singleModal_TextOnly.py --epochs 10
    python 02_singleModal_TextOnly.py --freeze-bert --epochs 3     # 更快
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from birdmm.data import MultimodalBirdDataset, load_annotations          # noqa: E402
from birdmm.metrics import classification_metrics, format_metrics        # noqa: E402
from birdmm.models.text_encoder import build_text_encoder, load_tokenizer  # noqa: E402
from birdmm.utils import count_parameters, ensure_dir, get_device, load_config, set_seed  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:                                                       # pragma: no cover
    def tqdm(x, **k):
        return x


class BertClassifier(nn.Module):
    """BERT 句向量 + 线性分类头（对应原始 02_singleModal_TextOnly.py）。"""

    def __init__(self, model_name: str, num_classes: int, freeze_bert: bool = False,
                 pooling: str = "mean", dropout: float = 0.1):
        super().__init__()
        self.encoder = build_text_encoder(
            model_name=model_name, pretrained=True, freeze=freeze_bert,
            out_dim=None, pooling=pooling, dropout=0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.encoder.out_dim, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        sentence_embedding = self.encoder(input_ids, attention_mask)   # [B, 768]
        return self.classifier(sentence_embedding)


def show_tokenization(tokenizer, text: str) -> None:
    """手册 §3：展示 tokenizer 如何把一句话变成 id（[CLS] … [SEP]）。"""
    enc = tokenizer(text)
    ids = enc["input_ids"]
    toks = tokenizer.convert_ids_to_tokens(ids)
    print("\n[分词] 示例：", text)
    print("    tokens:", toks)
    print("    ids   :", ids)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(ids, mask)
        y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
        y_true.extend(batch["label"].tolist())
    return classification_metrics(y_true, y_pred)


def main() -> None:
    cfg = load_config()
    to = cfg["text_only"]
    parser = argparse.ArgumentParser(description="Text-Only 单模态训练")
    parser.add_argument("--model-name", default=to["model_name"])
    parser.add_argument("--epochs", type=int, default=to["epochs"])
    parser.add_argument("--batch-size", type=int, default=to["batch_size"])
    parser.add_argument("--lr", type=float, default=to["lr"])
    parser.add_argument("--max-length", type=int, default=to["max_length"])
    parser.add_argument("--freeze-bert", action="store_true", default=to["freeze_bert"])
    parser.add_argument("--pooling", default="mean", choices=["cls", "mean"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="runs/text_only")
    args = parser.parse_args()

    set_seed(cfg["seed"])
    device = get_device(args.device)
    ann = load_annotations(cfg["data"]["annotations"])
    num_classes = ann["meta"]["num_classes"]
    print(f"[Text-Only] device={device}, model={args.model_name}, num_classes={num_classes}")

    try:
        tokenizer = load_tokenizer(args.model_name)
    except ImportError as exc:
        print(f"[Text-Only] {exc}")
        return

    # 数据里取一条真实描述做分词演示
    sample_caps = next((s["captions"] for s in ann["splits"]["train"] if s["captions"]), [])
    if sample_caps:
        show_tokenization(tokenizer, sample_caps[0])

    d = cfg["data"]
    loaders = {
        split: DataLoader(
            MultimodalBirdDataset(
                annotations_path=d["annotations"], split=split, data_root=d["root"],
                mode="text", tokenizer=tokenizer, max_length=args.max_length,
                train=(split == "train"),
                caption_policy="random" if split == "train" else "first",
            ),
            batch_size=args.batch_size, shuffle=(split == "train"),
            num_workers=cfg["train"]["num_workers"],
        )
        for split in ("train", "val", "test")
    }
    print(f"[Text-Only] 样本数：train={len(loaders['train'].dataset)}, "
          f"val={len(loaders['val'].dataset)}, test={len(loaders['test'].dataset)}")

    model = BertClassifier(args.model_name, num_classes, freeze_bert=args.freeze_bert,
                           pooling=args.pooling).to(device)
    trainable, total = count_parameters(model)
    print(f"[Text-Only] 参数量：可训练 {trainable:,} / 总 {total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=cfg["train"]["weight_decay"],
    )

    best_val, best_state = -1.0, None
    history = []
    print("\n[Text-Only] 开始训练 …")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total_n = 0.0, 0, 0
        for batch in tqdm(loaders["train"], desc=f"Epoch {epoch}/{args.epochs}"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_n += labels.size(0)

        train_acc = correct / max(total_n, 1)
        val_metrics = evaluate(model, loaders["val"], device)
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
    print(f"\n[Text-Only] 最优验证准确率：{best_val:.4f}，在测试集上评估：")
    test_metrics = evaluate(model, loaders["test"], device)
    print(format_metrics(test_metrics))

    out_dir = ensure_dir(Path(__file__).resolve().parent / args.out_dir)
    torch.save({"model": model.state_dict(), "model_name": args.model_name,
                "num_classes": num_classes, "args": vars(args)}, Path(out_dir) / "best.pt")
    with open(Path(out_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test": test_metrics, "best_val_acc": best_val, "history": history},
                  f, ensure_ascii=False, indent=2)
    print(f"[Text-Only] 结果已保存到：{out_dir}")


if __name__ == "__main__":
    main()