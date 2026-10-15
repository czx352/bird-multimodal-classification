"""birdmm.models —— 编码器与融合模块（手册 §3、§4）。"""
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder, load_tokenizer, masked_mean
from .fusion import build_fusion, FUSION_REGISTRY
from .multimodal import MultimodalClassifier, build_multimodal_model

__all__ = [
    "ImageEncoder",
    "TextEncoder",
    "load_tokenizer",
    "masked_mean",
    "build_fusion",
    "FUSION_REGISTRY",
    "MultimodalClassifier",
    "build_multimodal_model",
]