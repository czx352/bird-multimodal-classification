"""utils.py —— 配置加载、随机种子、设备选择等通用工具。"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

# 项目根目录（本文件位于 <root>/birdmm/utils.py）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_root() -> Path:
    """返回项目根目录，供脚本拼接相对路径时使用。"""
    return PROJECT_ROOT


def load_config(path: str | None = None) -> Dict[str, Any]:
    """读取 YAML 配置；path 为空时读取项目根目录下的 config.yaml。

    配置里所有相对路径（如 datasets）都会被解析为相对「项目根目录」的绝对路径，
    这样无论从哪个工作目录运行脚本，结果都一致。
    """
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 把关键路径字段解析为绝对路径
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    cfg["data"]["root"] = str(root)

    ann = Path(cfg["data"]["annotations"])
    if not ann.is_absolute():
        ann = PROJECT_ROOT / ann
    cfg["data"]["annotations"] = str(ann)

    cfg["_config_path"] = str(path)
    return cfg


def set_seed(seed: int = 42) -> None:
    """固定随机种子，保证划分 / 训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """选择设备：auto 时优先 cuda/mps，否则 cpu。"""
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(prefer)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """返回 (可训练参数量, 总参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def ensure_dir(path: str | os.PathLike) -> str:
    """确保目录存在（用于保存 checkpoint / 报告）。"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)