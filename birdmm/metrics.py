"""metrics.py —— 统一的分类指标：Accuracy / 宏平均 Precision、Recall、F1。

手册要求「训练、测试和分类性能指标验证」，三种单模态/多模态脚本共用这里的实现。
sklearn 缺失时退化为纯 torch/numpy 版本（数值一致），保证脚本永远能跑。
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """计算准确率与宏平均 P/R/F1。

    Args:
        y_true / y_pred: 一维标签序列
    Returns:
        {"accuracy", "precision", "recall", "f1"}
    """
    y_true = _to_np(y_true).astype(np.int64)
    y_pred = _to_np(y_pred).astype(np.int64)

    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
    except ImportError:
        return _metrics_numpy(y_true, y_pred)


def _metrics_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """不依赖 sklearn 的宏平均实现（教学演示用）。"""
    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    labels = np.unique(np.concatenate([y_true, y_pred])) if y_true.size else np.array([])
    precisions, recalls, f1s = [], [], []
    for c in labels:
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return {
        "accuracy": accuracy,
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """把指标字典格式化成手册里那种打印块。"""
    return (
        "====================\n"
        f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n"
        f"宏平均精确率 (Precision): {metrics['precision']:.4f}\n"
        f"宏平均召回率 (Recall): {metrics['recall']:.4f}\n"
        f"宏平均 F1 分数 (F1-Score): {metrics['f1']:.4f}\n"
        "===================="
    )