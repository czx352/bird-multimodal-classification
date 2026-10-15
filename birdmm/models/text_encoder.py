"""text_encoder.py —— 文本编码器：把一句话变成一个向量。

教学要点
--------
1. 分词：BERT tokenizer 把句子切成 WordPiece token，加上 [CLS]/[SEP]，
   每个 token 映射为一个 id；id 再查 embedding 表得到初始向量。
       "a small blue bird" -> [CLS] 1037 2235 2304 4742 [SEP]
   最终输入向量 = Token Embed + Segment Embed + Position Embed。
2. Transformer 自注意力输出每个 token 的上下文向量 [B, L, 768]。
3. 池化：分类需要"一个句子一个向量"，两种常用做法：
   - cls  ：直接取 [CLS] 位置的向量；
   - mean ：对所有真实 token（按 attention_mask 过滤 padding）取平均 —— 更稳。
   （若不做 mask，[PAD] 会稀释真实单词的向量。）
4. freeze=True 时只训练融合层：BERT 完全不动，速度和显存都友好。
5. return_tokens=True 时额外返回 token 序列 [B, L, out_dim]，
   供 Cross-Attention 融合当 Key/Value 使用（手册 §4.3）。
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn

# 延迟到使用时才需要 transformers：这样没有安装它也能导入本项目的图像部分
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None


def _require_transformers() -> None:
    if AutoModel is None:
        raise ImportError(
            "文本编码器需要 transformers：pip install transformers\n"
            "（离线/无网络时先设置镜像：set HF_ENDPOINT=https://hf-mirror.com）"
        )


def load_tokenizer(model_name: str = "bert-base-uncased"):
    """加载 tokenizer。国内网络不通时可先设置镜像：
    export HF_ENDPOINT=https://hf-mirror.com  （Windows: set HF_ENDPOINT=...）
    """
    _require_transformers()
    return AutoTokenizer.from_pretrained(model_name)


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """只对真实 token 求平均（把 padding 位置屏蔽掉）。

    hidden: [B, L, H]；attention_mask: [B, L]（1=真实 token，0=padding）
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)      # [B, L, 1]
    summed = (hidden * mask).sum(dim=1)                       # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-6)                  # [B, 1]
    return summed / counts


class TextEncoder(nn.Module):
    """BERT 文本编码器。

    Args:
        model_name: HuggingFace 模型名，如 bert-base-uncased / distilbert-base-uncased
        freeze:     是否冻结 BERT 主干
        out_dim:    输出维度。None 表示保持 BERT 原生维度（bert-base = 768）
        pooling:    "cls" 或 "mean"
        dropout:    投影层 dropout
    """

    def __init__(self, model_name: str = "bert-base-uncased", pretrained: bool = True,
                 freeze: bool = False, out_dim: int = None,
                 pooling: str = "mean", dropout: float = 0.0):
        super().__init__()
        _require_transformers()
        if pretrained:
            self.bert = AutoModel.from_pretrained(model_name)
        else:
            from transformers import AutoConfig

            self.bert = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        self.hidden_size = self.bert.config.hidden_size   # bert-base -> 768
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling 只能是 'cls' 或 'mean'，收到：{pooling}")
        self.pooling = pooling
        self.out_dim = self.hidden_size if out_dim is None else out_dim
        if out_dim is None:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.hidden_size, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        self.freeze = freeze
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> "TextEncoder":
        super().train(mode)
        if self.freeze:
            self.bert.eval()
        return self

    def encode_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (token 序列 [B, L, out_dim], 池化向量 [B, out_dim])。"""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state                     # [B, L, H]
        if self.pooling == "cls":
            pooled = hidden[:, 0]                          # [CLS] 位置
        else:
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                pooled = masked_mean(hidden, attention_mask)
        return self.proj(hidden), self.proj(pooled)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_tokens: bool = False,
        return_both: bool = False,
    ):
        """前向。

        return_both=True  -> (tokens, pooled)，Cross-Attention 一次拿到两种表示；
        return_tokens=True-> 只返回 tokens [B, L, out_dim]；
        默认               -> 只返回 pooled   [B, out_dim]。
        """
        tokens, pooled = self.encode_tokens(input_ids, attention_mask)
        if return_both:
            return tokens, pooled
        if return_tokens:
            return tokens
        return pooled

    def extra_repr(self) -> str:
        return (f"hidden_size={self.hidden_size}, out_dim={self.out_dim}, "
                f"pooling={self.pooling}, frozen={'(bert)' if self.freeze else 'no'}")


def build_text_encoder(model_name: str = "bert-base-uncased", pretrained: bool = True,
                       freeze: bool = False, out_dim: int = None, pooling: str = "mean",
                       dropout: float = 0.0) -> TextEncoder:
    """构建文本编码器；预训练权重下载失败（如离线环境）时退化为随机初始化。"""
    _require_transformers()
    try:
        return TextEncoder(model_name, pretrained, freeze, out_dim, pooling, dropout)
    except Exception as exc:  # 下载权重失败 / 无网络
        if not pretrained:
            raise
        warnings.warn(f"加载 {model_name} 预训练权重失败（{exc}），改用随机初始化训练。")
        return TextEncoder(model_name, pretrained=False, freeze=freeze, out_dim=out_dim,
                           pooling=pooling, dropout=dropout)