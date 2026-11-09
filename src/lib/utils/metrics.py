"""
点云分割指标: IoU, Precision/Recall/F1, Accuracy.
"""

import numpy as np
import torch


@torch.no_grad()
def compute_iou(pred: torch.Tensor, target: torch.Tensor, class_names: dict) -> dict:
    """返回 {class_name: IoU, 'mIoU': mean}.

    Args:
        pred: (B, N, C) logits 或 (B*N,) argmax 标签
        target: (B, N) 标签
        class_names: {0: "噪声", 1: "混凝土", 2: "钢筋"} 等
    """
    if pred.dim() > 1 and pred.shape[-1] > 1:
        pred = pred.argmax(dim=-1)
    pred = pred.flatten()
    target = target.flatten()
    ious = {}
    for c, name in class_names.items():
        inter = ((pred == c) & (target == c)).sum().item()
        union = ((pred == c) | (target == c)).sum().item()
        ious[name] = inter / (union + 1e-6)
    vals = [v for k, v in ious.items() if k != "mIoU"]
    ious["mIoU"] = float(np.mean(vals)) if vals else 0.0
    return ious


@torch.no_grad()
def compute_precision_recall_f1(
    pred: torch.Tensor, target: torch.Tensor, class_names: dict
) -> dict:
    """返回 {class_name: {precision, recall, f1}}."""
    if pred.dim() > 1 and pred.shape[-1] > 1:
        pred = pred.argmax(dim=-1)
    pred = pred.flatten()
    target = target.flatten()
    metrics = {}
    for c, name in class_names.items():
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        metrics[name] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


@torch.no_grad()
def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.dim() > 1 and pred.shape[-1] > 1:
        pred = pred.argmax(dim=-1)
    return (pred == target).float().mean().item()


def cm_iou(*class_ious: float) -> float:
    """cmIoU 唯一实现 (稿件 Eq. 9): 工程相关各类 IoU 的算术平均, 恒除以 3.

    三处调用点必须经由本函数, 口径注释:
      - Stage 1 (噪声/混凝土/金属): cm_iou(混凝土, 金属, 金属)
        —— 金属类同时代表钢筋与套筒两个工程类位置, 与 Eq. 9 的三类均值等价;
      - Stage 2 / 单阶段 4 类 (噪声/混凝土/钢筋/套筒): cm_iou(混凝土, 钢筋, 套筒).
    """
    if len(class_ious) != 3:
        raise ValueError("cm_iou 需要 3 个类 IoU (排除噪声后的工程类)")
    return sum(class_ious) / 3.0
