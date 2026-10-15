"""clip_utils.py —— 手册 §5：CLIP 思想实验（从"分类器"到"共同语义空间"）。

    image ─► 图像向量 ┐
                       ├─► 余弦相似度最大者 = 匹配
    text  ─► 文本向量 ┘

这里没有分类器：类别通过"和类别描述的相似度"被比较出来。一旦有了共同语义空间，
立刻得到三种能力：图文检索、零样本分类、通用表示。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


def load_clip(model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
    """加载 CLIP 模型与处理器。国内可先 set HF_ENDPOINT=https://hf-mirror.com。"""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def _feature_tensor(out) -> torch.Tensor:
    """取出投影后的图文向量 [N, d]。

    兼容不同 transformers 版本：旧版 get_*_features 直接返回张量，
    新版（如 5.x）返回 BaseModelOutputWithPooling，投影结果在 pooler_output 里。
    """
    if torch.is_tensor(out):
        return out
    for attr in ("pooler_output", "text_embeds", "image_embeds"):
        value = getattr(out, attr, None)
        if torch.is_tensor(value):
            return value
    raise TypeError(f"无法从 {type(out)} 中取出特征张量")


@torch.no_grad()
def encode_texts(model, processor, texts: Sequence[str], device: str = "cpu") -> torch.Tensor:
    """文本 -> 归一化向量 [N, d]。"""
    inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = _feature_tensor(model.get_text_features(**inputs))
    return F.normalize(feats, dim=-1)


@torch.no_grad()
def encode_images(model, processor, images, device: str = "cpu") -> torch.Tensor:
    """PIL 图片列表 -> 归一化向量 [N, d]。"""
    inputs = processor(images=list(images), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = _feature_tensor(model.get_image_features(**inputs))
    return F.normalize(feats, dim=-1)


def zero_shot_classify(image_emb: torch.Tensor, class_emb: torch.Tensor,
                       logit_scale: float = 100.0) -> torch.Tensor:
    """用相似度排序做零样本分类，返回每个样本的预测类别下标。"""
    sims = logit_scale * image_emb @ class_emb.t()          # [N_img, N_class]
    return sims.argmax(dim=-1)


def retrieval_recall(image_emb: torch.Tensor, text_emb: torch.Tensor,
                     ks: Sequence[int] = (1, 5)) -> Dict[str, float]:
    """图文检索：以对角线为正确配对，计算 R@k（image->text）。

    返回 {"R@1": ..., "R@5": ...}
    """
    sims = image_emb @ text_emb.t()                          # [N, N]
    ranks = sims.argsort(dim=-1, descending=True)
    targets = torch.arange(sims.size(0), device=sims.device).unsqueeze(1)
    hit = (ranks == targets)                                 # 正确项出现在第几列
    out = {}
    for k in ks:
        out[f"R@{k}"] = float(hit[:, :k].any(dim=-1).float().mean())
    return out