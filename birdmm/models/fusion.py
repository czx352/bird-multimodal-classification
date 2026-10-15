"""fusion.py —— 多模态融合模块（本实验的核心）。

1. Early Fusion (ConcatFusion)
      [图像向量, 文本向量] -> 拼接 -> MLP -> 分类
   优点：简单、参数少；缺点：两个模态从头到尾被当成一串数字，没有交互。

2. Late Fusion (LateFusion)
      图像 -> 图像分类头 -> 概率
      文本 -> 文本分类头 -> 概率
      两者加权平均 -> 最终概率
   优点：任一模态缺失都能工作、可解释；缺点：模态之间从不交流。

3. Cross-Attention Fusion (CrossAttentionFusion)  ← 推荐
      图像 token 作为 Query，文本 token 作为 Key/Value，
      让"图像的每个空间位置"主动去文本里找相关描述（"这里是不是 blue？"）。
   优点：显式建模跨模态对齐；可导出注意力权重，用于可视化"模型在看哪个词"。

4. Gated Fusion (GatedFusion)
      用一个门控网络逐个样本决定"这次更信图像还是更信文本"。
   优点：对"描述没提颜色"这类不可靠文本有天然的抑制能力。

所有融合模块统一返回 dict：
    {"logits": [B, C],
     "image_logits": [...]?,   # 仅 Late Fusion 有
     "text_logits": [...]?,    # 仅 Late Fusion 有
     "gate": [...]?,           # 仅 Gated Fusion 有
     "attn_weights": [...]?}   # 仅 Attention Fusion 有，用于可视化

统一接口：forward(image_feat, text_feat, attention_mask) -> dict
    - needs_tokens=False 的融合：image_feat/text_feat 均为池化向量 [B, d]
    - needs_tokens=True  的融合：image_feat=[B, N_img, d]（视觉 token），
                                 text_feat =[B, L, d]（文本 token）
因此换融合方式只需要改配置里的 fusion.name 一个字段。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """① Early Fusion：直接拼接两个向量。"""

    needs_tokens = False

    def __init__(self, image_dim: int, text_dim: int, num_classes: int,
                 hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        fused = torch.cat([image_feat, text_feat], dim=-1)      # [B, 512+768=1280]
        return {"logits": self.mlp(fused)}


class LateFusion(nn.Module):
    """② Late Fusion：两个模态各自分类，再在概率层面加权。"""

    needs_tokens = False

    def __init__(self, image_dim: int, text_dim: int, num_classes: int,
                 hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.image_head = nn.Linear(image_dim, num_classes)
        self.text_head = nn.Linear(text_dim, num_classes)
        # 可学习的融合权重：w 越接近 1 越信文本，越接近 0 越信图像
        self.logit_w = nn.Parameter(torch.zeros(1))

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        image_logits = self.image_head(image_feat)             # [B, C]
        text_logits = self.text_head(text_feat)                # [B, C]
        w = torch.sigmoid(self.logit_w)                        # 标量权重
        logits = w * text_logits + (1.0 - w) * image_logits
        return {
            "logits": logits,
            "image_logits": image_logits,
            "text_logits": text_logits,
            "weight": w.detach().expand(image_feat.size(0)),
        }


class CrossAttentionFusion(nn.Module):
    """③ Cross-Attention Fusion：图像 token 去文本里找对应描述。

    Query = 图像 token [B, 49, 512]，Key/Value = 文本 token [B, L, 768]。
    """

    needs_tokens = True

    def __init__(self, image_dim: int, text_dim: int, num_classes: int,
                 hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        d = hidden_dim
        self.q_proj = nn.Linear(image_dim, d)
        self.kv_proj = nn.Linear(text_dim, d)
        self.attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d, d)
        )
        self.norm2 = nn.LayerNorm(d)
        self.classifier = nn.Linear(d, num_classes)

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        q = self.q_proj(image_feat)                            # [B, 49, d]
        kv = self.kv_proj(text_feat)                           # [B, L, d]
        # attention_mask: 1=真实 token，0=padding；MultiheadAttention 需要 True 表示"屏蔽"
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        attn_out, attn_w = self.attn(
            q, kv, kv, key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=False,
        )                                                      # attn_w: [B, heads, 49, L]
        x = self.norm1(q + attn_out)                           # 残差
        x = self.norm2(x + self.ffn(x))
        pooled = x.mean(dim=1)                                 # [B, d]
        return {"logits": self.classifier(pooled), "attn_weights": attn_w}


class GatedFusion(nn.Module):
    """④ Gated Fusion：逐样本决定"这次更信图像还是更信文本"。

        a = sigmoid(W[img; txt]);  fusion = a * img + (1 - a) * txt
    """

    needs_tokens = False

    def __init__(self, image_dim: int, text_dim: int, num_classes: int,
                 hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        d = hidden_dim
        self.image_proj = nn.Linear(image_dim, d)
        self.text_proj = nn.Linear(text_dim, d)
        self.gate = nn.Sequential(
            nn.Linear(image_dim + text_dim, d),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, num_classes)
        )

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        img = self.image_proj(image_feat)                      # [B, d]
        txt = self.text_proj(text_feat)                        # [B, d]
        a = torch.sigmoid(self.gate(torch.cat([image_feat, text_feat], dim=-1)))
        fused = a * img + (1.0 - a) * txt
        return {"logits": self.classifier(fused), "gate": a}


FUSION_REGISTRY = {
    "early": ConcatFusion,
    "late": LateFusion,
    "cross_attention": CrossAttentionFusion,
    "gated": GatedFusion,
}


def build_fusion(name: str, image_dim: int, text_dim: int, num_classes: int,
                 hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.1) -> nn.Module:
    """按名字构建融合模块（配置里 fusion.name 的一个字段就够）。"""
    key = name.lower()
    if key not in FUSION_REGISTRY:
        raise ValueError(f"未知融合方式：{name}，可选 {list(FUSION_REGISTRY)}")
    cls = FUSION_REGISTRY[key]
    if cls is CrossAttentionFusion:
        return cls(image_dim, text_dim, num_classes, hidden_dim=hidden_dim,
                   num_heads=num_heads, dropout=dropout)
    return cls(image_dim, text_dim, num_classes, hidden_dim=hidden_dim, dropout=dropout)