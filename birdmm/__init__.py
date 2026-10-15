"""birdmm —— 鸟类多模态分类实验的可复用库。

模块划分与实验手册的章节一一对应：
    birdmm.data                       §1 数据与标注（6:2:2 划分、annotations.json）
    birdmm.models.image_encoder       §3 Image Encoder（ResNet 特征图 -> 向量 / token）
    birdmm.models.text_encoder        §3 Text Encoder（BERT -> 句向量 / token）
    birdmm.models.fusion              §4 四种融合（Early / Late / Cross-Attention / Gated）
    birdmm.models.multimodal          §4 组装：两个编码器 + 一个融合模块
    birdmm.clip_utils                 §5 CLIP 共同语义空间（零样本 / 检索）
"""
from .utils import load_config, set_seed, get_device, project_root

__all__ = ["load_config", "set_seed", "get_device", "project_root"]