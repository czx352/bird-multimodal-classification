"""data.py —— 数据准备与加载（对应手册 §1.3）。

职责：
1. build_annotations(): 扫描「类别文件夹 + 同名 txt」的原始结构，
   按 6:2:2 分层划分 train/val/test，生成一个 annotations.json（冻结划分）；
2. MultimodalBirdDataset: 统一读 annotations.json 的某个 split，
   支持三种返回模式 —— image（仅图像）/ text（仅文本）/ multimodal（图文）；
3. build_dataloaders(): 依据 config 组装 DataLoader。

annotations.json 结构（与手册一致）：
{
  "meta":    {"source": ..., "num_classes": ..., "num_captions": ...},
  "classes": ["001.Black_footed_Albatross", ...],          # 下标 == label
  "class_prompts": {"001.Black_footed_Albatross": "a photo of a black footed albatross"},
  "splits":  {"train": [{"image_path": ..., "captions": [...], "label": 0}, ...],
              "val": [...], "test": [...]}
}
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# 1. 生成标注文件
# --------------------------------------------------------------------------- #
def clean_class_name(folder_name: str) -> str:
    """把 '001.Black_footed_Albatross' 清洗成 'black footed albatross'。"""
    name = re.sub(r"^\d+[._\-\s]*", "", folder_name)      # 去掉 '001.' 前缀
    name = name.replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", name)


def class_prompt(folder_name: str) -> str:
    """类别 -> 提示词，供 CLIP 零样本分类使用（手册 §5）。"""
    return f"a photo of a {clean_class_name(folder_name)}"


def _read_captions(txt_path: Path, limit: int) -> List[str]:
    """读取同名 txt（每行一条描述），最多 limit 条。"""
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[:limit]


def build_annotations(
    data_root: str | Path,
    out_path: str | Path,
    split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
    captions_per_image: int = 10,
) -> Dict:
    """扫描 data_root 生成 annotations.json 并返回其内容。

    目录约定：data_root/<类别文件夹>/<图片.jpg|png|...> 与同名的 .txt。
    txt 允许缺失（此时 captions 为空列表，仅 image-only 可用）。
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"找不到数据目录：{data_root}")

    # 找出所有类别文件夹（含图片的直接子目录；兼容 images/ 再套一层的情况）
    class_dirs = sorted({p.parent for p in data_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS})
    classes = [d.name for d in class_dirs]
    class_to_label = {c: i for i, c in enumerate(classes)}

    per_class: Dict[int, List[Dict]] = defaultdict(list)
    for cdir in class_dirs:
        label = class_to_label[cdir.name]
        for img in sorted(p for p in cdir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            txt = img.with_suffix(".txt")
            captions = _read_captions(txt, captions_per_image) if txt.exists() else []
            per_class[label].append(
                {
                    "image_path": str(img.relative_to(data_root)).replace("\\", "/"),
                    "captions": captions,
                    "label": label,
                }
            )

    # 分层随机划分：对每个类别内部按比例切分，保证类别分布一致
    rng = random.Random(seed)
    splits: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    for label in sorted(per_class):
        items = per_class[label][:]
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * split[0]))
        n_val = int(round(n * split[1]))
        n_train, n_val = min(n_train, n), min(n_val, n - n_train)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])
    for k in splits:                        # 打乱各 split 内部的顺序
        rng.shuffle(splits[k])

    annotations = {
        "meta": {
            "source": str(data_root),
            "num_classes": len(classes),
            "num_captions": captions_per_image,
            "split_ratio": list(split),
            "seed": seed,
            "counts": {k: len(v) for k, v in splits.items()},
        },
        "classes": classes,
        "class_prompts": {c: class_prompt(c) for c in classes},
        "splits": splits,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    return annotations


def load_annotations(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 2. 数据集
# --------------------------------------------------------------------------- #
def build_image_transform(image_size: int = 224, train: bool = False):
    """ImageNet 标准预处理（与预训练 ResNet 对齐）。"""
    from torchvision import transforms

    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class MultimodalBirdDataset(Dataset):
    """读 annotations.json 的某个 split，按 mode 返回图文样本。

    Args:
        annotations_path: annotations.json 路径
        split:            "train" / "val" / "test"
        data_root:        图片路径的根目录（json 里存的是相对路径）
        mode:             "image" | "text" | "multimodal"
        tokenizer:        文本模式必填（HuggingFace tokenizer）
        max_length:       文本截断长度
        image_size:       图像尺寸
        train:            是否使用训练增强
        caption_policy:   "random" 训练时随机取一条；"first" 评测时固定第一条
    """

    def __init__(
        self,
        annotations_path: str | Path,
        split: str,
        data_root: str | Path,
        mode: str = "multimodal",
        tokenizer=None,
        max_length: int = 64,
        image_size: int = 224,
        train: bool = False,
        caption_policy: str = "random",
    ):
        assert mode in {"image", "text", "multimodal"}
        self.ann = load_annotations(annotations_path)
        self.samples = self.ann["splits"][split]
        self.data_root = Path(data_root)
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.caption_policy = caption_policy
        self.transform = build_image_transform(image_size, train=train)
        if mode in {"text", "multimodal"} and tokenizer is None:
            raise ValueError(f"mode={mode} 需要传入 tokenizer")

    def __len__(self) -> int:
        return len(self.samples)

    def _pick_caption(self, captions: List[str]) -> str:
        if not captions:
            return ""
        if self.caption_policy == "random" and len(captions) > 1:
            return random.choice(captions)
        return captions[0]

    def __getitem__(self, index: int) -> Dict:
        item = self.samples[index]
        out: Dict = {"label": item["label"]}

        if self.mode in {"image", "multimodal"}:
            img = Image.open(self.data_root / item["image_path"]).convert("RGB")
            out["image"] = self.transform(img)

        if self.mode in {"text", "multimodal"}:
            text = self._pick_caption(item["captions"])
            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            out["input_ids"] = enc["input_ids"].squeeze(0)
            out["attention_mask"] = enc["attention_mask"].squeeze(0)

        return out


# --------------------------------------------------------------------------- #
# 3. DataLoader 组装
# --------------------------------------------------------------------------- #
def build_dataloaders(
    cfg: Dict,
    mode: str,
    tokenizer=None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
):
    """根据配置创建 train / val / test 三个 DataLoader。"""
    d = cfg["data"]
    t = cfg["train"]
    batch_size = batch_size or t["batch_size"]
    num_workers = t["num_workers"] if num_workers is None else num_workers

    loaders = {}
    for split, train in (("train", True), ("val", False), ("test", False)):
        ds = MultimodalBirdDataset(
            annotations_path=d["annotations"],
            split=split,
            data_root=d["root"],
            mode=mode,
            tokenizer=tokenizer,
            max_length=cfg.get("text_encoder", {}).get("max_length", 64),
            image_size=d["image_size"],
            train=train,
            caption_policy="random" if split == "train" else "first",
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders