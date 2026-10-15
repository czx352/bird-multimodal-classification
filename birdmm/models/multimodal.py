"""multimodal.py —— 把「图像编码器 + 文本编码器 + 融合模块」组装成一个分类器。

   图片 ─► ImageEncoder ─► 图像向量 / token ─┐
                                              ├─► Fusion ─► Classifier ─► 鸟的种类
   描述 ─► TextEncoder  ─► 文本向量 / token ─┘

融合方式决定需要哪种表示：
    Early / Late / Gated        -> 池化向量
    Cross-Attention             -> 图像 token [B, 49, C] 与 文本 token [B, L, C]
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .fusion import build_fusion
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class MultimodalClassifier(nn.Module):
    """图文双塔 + 融合头。"""

    def __init__(self, fusion_name: str, num_classes: int,
                 image_backbone: str = "resnet18", image_pretrained: bool = True,
                 image_freeze: bool = True, image_out_dim: int = 512, image_dropout: float = 0.1,
                 text_model: str = "bert-base-uncased", text_freeze: bool = True,
                 text_out_dim: int = 768, text_pooling: str = "mean", text_dropout: float = 0.1,
                 hidden_dim: int = 512, num_heads: int = 8, fusion_dropout: float = 0.1):
        super().__init__()
        self.image_encoder = ImageEncoder(
            backbone=image_backbone, pretrained=image_pretrained,
            freeze=image_freeze, out_dim=image_out_dim, dropout=image_dropout,
        )
        self.text_encoder = TextEncoder(
            model_name=text_model, freeze=text_freeze,
            out_dim=text_out_dim, pooling=text_pooling, dropout=text_dropout,
        )
        self.fusion = build_fusion(
            fusion_name,
            image_dim=self.image_encoder.out_dim,
            text_dim=self.text_encoder.out_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim, num_heads=num_heads, dropout=fusion_dropout,
        )
        self.fusion_name = fusion_name
        self.needs_tokens = getattr(self.fusion, "needs_tokens", False)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.needs_tokens:
            image_feat = self.image_encoder(images, return_tokens=True)      # [B, 49, C]
            text_feat = self.text_encoder(input_ids, attention_mask, return_tokens=True)
            return self.fusion(image_feat, text_feat, attention_mask)
        image_feat = self.image_encoder(images)                             # [B, C]
        text_feat = self.text_encoder(input_ids, attention_mask)            # [B, H]
        return self.fusion(image_feat, text_feat, attention_mask)


def build_multimodal_model(cfg: Dict, num_classes: int) -> MultimodalClassifier:
    """按 config.yaml 构建多模态模型。"""
    ie, te, fu = cfg["image_encoder"], cfg["text_encoder"], cfg["fusion"]
    return MultimodalClassifier(
        fusion_name=fu["name"],
        num_classes=num_classes,
        image_backbone=ie["backbone"],
        image_pretrained=ie["pretrained"],
        image_freeze=ie["freeze"],
        image_out_dim=ie["out_dim"],
        image_dropout=ie.get("dropout", 0.1),
        text_model=te["model_name"],
        text_freeze=te["freeze"],
        text_out_dim=te["out_dim"],
        text_pooling=te.get("pooling", "mean"),
        text_dropout=te.get("dropout", 0.1),
        hidden_dim=fu.get("hidden_dim", 512),
        num_heads=fu.get("num_heads", 8),
        fusion_dropout=fu.get("dropout", 0.1),
    )