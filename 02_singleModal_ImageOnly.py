"""02_singleModal_ImageOnly.py —— 单模态 Image-Only 训练（手册 §2.1「请补全…」）。

输入：图像 + 类别标签
模型：ResNet18 / ResNet34 / ResNet50（可换 MobileNetV2 见文末注释）
输出：类别

学习目标：
    1. 实现训练、测试和分类性能指标验证；
    2. 把模型拆开，查看对图像处理后的特征（特征图 -> token -> 池化向量）。

用法：
    python 02_singleModal_ImageOnly.py --epochs 10
    python 02_singleModal_ImageOnly.py --backbone resnet18 --freeze-backbone   # CPU 快速跑通
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
from birdmm.models.image_encoder import build_image_encoder              # noqa: E402
from birdmm.utils import count_parameters, ensure_dir, get_device, load_config, set_seed  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:                                                       # pragma: no cover
    def tqdm(x, **k):
        return x


class ImageOnlyClassifier(nn.Module):
    """图像编码器 + 一个全连接分类头（对应笔记：512 维后接 FC 得到类别概率）。"""

    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True,
                 freeze_backbone: bool = False, dropout: float = 0.1):
        super().__init__()
        self.encoder = build_image_encoder(
            backbone=backbone, pretrained=pretrained,
            freeze=freeze_backbone, out_dim=None, dropout=0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.encoder.out_dim, num_classes)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(images)                     # [B, 512]
        return self.classifier(feat)


def describe_image_features(model: ImageOnlyClassifier, images: torch.Tensor, device) -> None:
    """拆开图像模型：打印 特征图 -> 49 个 token -> 池化向量 的形状与统计量。"""
    model.eval()
    with torch.no_grad():
        x = images[:2].to(device)
        fmap = model.encoder.body(x)                    # [B, C, h, w]
        tokens = fmap.flatten(2).transpose(1, 2)        # [B, h*w, C]
        pooled = model.encoder.pool(fmap).flatten(1)    # [B, C]
    print("\n[特征拆解] 一次前向的张量流动：")
    print(f"    输入图像        : {tuple(x.shape)}")
    print(f"    ResNet 特征图    : {tuple(fmap.shape)}   （去掉 avgpool/fc 后的 [B, C, h, w]）")
    print(f"    视觉 token       : {tuple(tokens.shape)}   （flatten+transpose：每张图切成 {tokens.shape[1]} 个'视觉单词'）")
    print(f"    全局池化向量     : {tuple(pooled.shape)}")
    print(f"    向量统计         : mean={pooled.mean():.3f}, std={pooled.std():.3f}, "
          f"min={pooled.min():.3f}, max={pooled.max():.3f}")


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
        y_true.extend(batch["label"].tolist())
    return classification_metrics(y_true, y_pred)


def main() -> None:
    cfg = load_config()
    io = cfg["image_only"]
    parser = argparse.ArgumentParser(description="Image-Only 单模态训练")
    parser.add_argument("--backbone", default=io["backbone"])
    parser.add_argument("--epochs", type=int, default=io["epochs"])
    parser.add_argument("--batch-size", type=int, default=io["batch_size"])
    parser.add_argument("--lr", type=float, default=io["lr"])
    parser.add_argument("--freeze-backbone", action="store_true", default=io["freeze_backbone"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="runs/image_only")
    args = parser.parse_args()

    set_seed(cfg["seed"])
    device = get_device(args.device)
    ann = load_annotations(cfg["data"]["annotations"])
    num_classes = ann["meta"]["num_classes"]
    print(f"[Image-Only] device={device}, backbone={args.backbone}, num_classes={num_classes}")

    # ---- 数据 ----
    d = cfg["data"]
    loaders = {
        split: DataLoader(
            MultimodalBirdDataset(
                annotations_path=d["annotations"], split=split, data_root=d["root"],
                mode="image", image_size=d["image_size"], train=(split == "train"),
            ),
            batch_size=args.batch_size, shuffle=(split == "train"),
            num_workers=cfg["train"]["num_workers"],
        )
        for split in ("train", "val", "test")
    }
    print(f"[Image-Only] 样本数：train={len(loaders['train'].dataset)}, "
          f"val={len(loaders['val'].dataset)}, test={len(loaders['test'].dataset)}")

    # ---- 模型 ----
    model = ImageOnlyClassifier(
        backbone=args.backbone, num_classes=num_classes,
        pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone,
    ).to(device)
    trainable, total = count_parameters(model)
    print(f"[Image-Only] 参数量：可训练 {trainable:,} / 总 {total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=cfg["train"]["weight_decay"],
    )

    # 展示一次特征拆解（手册学习目标 2）
    describe_image_features(model, next(iter(loaders["train"]))["image"], device)

    # ---- 训练 ----
    best_val, best_state = -1.0, None
    history = []
    print("\n[Image-Only] 开始训练 …")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total_n = 0.0, 0, 0
        for batch in tqdm(loaders["train"], desc=f"Epoch {epoch}/{args.epochs}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(images)
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

    # ---- 测试 ----
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n[Image-Only] 最优验证准确率：{best_val:.4f}，在测试集上评估：")
    test_metrics = evaluate(model, loaders["test"], device)
    print(format_metrics(test_metrics))

    # ---- 保存 ----
    out_dir = ensure_dir(Path(__file__).resolve().parent / args.out_dir)
    torch.save({"model": model.state_dict(), "backbone": args.backbone,
                "num_classes": num_classes, "args": vars(args)}, Path(out_dir) / "best.pt")
    with open(Path(out_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test": test_metrics, "best_val_acc": best_val, "history": history},
                  f, ensure_ascii=False, indent=2)
    print(f"[Image-Only] 结果已保存到：{out_dir}")


if __name__ == "__main__":
    main()

# 想换 MobileNetV2：把 ImageOnlyClassifier 里的 build_image_encoder 换成
#   from torchvision import models
#   backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
#   self.encoder = nn.Sequential(backbone, nn.AdaptiveAvgPool2d(1), nn.Flatten())
# 其余训练/评估流程完全一致。