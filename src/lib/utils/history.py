"""
训练历史记录: 模板构建、保存/加载、checkpoint 管理.
"""

from __future__ import annotations

import os, json
import torch
import numpy as np

from .common import CLASS_NAMES, S1_NAMES, console


# ── 历史记录模板 ──

def build_history_template_s1() -> dict:
    """S1 3类训练历史 key 模板."""
    hist = {"epoch": [], "loss": [], "val_loss": [], "acc": [], "cm": []}
    for cls_name in S1_NAMES.values():
        hist[f"iou_{cls_name}"] = []
    for cls_name in S1_NAMES.values():
        for met in ["precision", "recall", "f1"]:
            hist[f"prf_{cls_name}_{met}"] = []
    return hist


def build_history_template_s2() -> dict:
    """S2 级联训练历史 key 模板."""
    hist = {
        "epoch": [], "loss": [], "val_loss": [],
        "val_acc": [], "val_cm": [], "val_rs_iou": [],
    }
    for cls_name in CLASS_NAMES.values():
        hist[f"iou_{cls_name}"] = []
        for met in ["precision", "recall", "f1"]:
            hist[f"prf_{cls_name}_{met}"] = []
    for tag in ["S2_噪声", "S2_钢筋", "S2_套筒"]:
        hist[f"s2pure_{tag}"] = []
    return hist


def build_history_template_a2() -> dict:
    """A2 单阶段4类训练历史 key 模板."""
    hist = {"epoch": [], "loss": [], "val_loss": [], "acc": [], "cm": []}
    for cls_name in CLASS_NAMES.values():
        hist[f"iou_{cls_name}"] = []
    for cls_name in CLASS_NAMES.values():
        for met in ["precision", "recall", "f1"]:
            hist[f"prf_{cls_name}_{met}"] = []
    return hist


# ── 填充历史 ──

def fill_history_s1(hist: dict, ep: int, tl: float, vl: float,
                    ious: dict, acc: float, prf: dict, cm: float):
    """填充 S1 一轮验证结果到历史记录."""
    hist["epoch"].append(ep)
    hist["loss"].append(tl)
    hist["val_loss"].append(vl)
    hist["acc"].append(acc)
    hist["cm"].append(cm)
    for cls_name in S1_NAMES.values():
        hist[f"iou_{cls_name}"].append(ious.get(cls_name, 0.0))
    for cls_name in S1_NAMES.values():
        pm = prf.get(cls_name, {})
        hist[f"prf_{cls_name}_precision"].append(pm.get("precision", 0.0))
        hist[f"prf_{cls_name}_recall"].append(pm.get("recall", 0.0))
        hist[f"prf_{cls_name}_f1"].append(pm.get("f1", 0.0))


def fill_history_s2(hist: dict, ep: int, s2_tl: float,
                    c_ious: dict, c_prf: dict, c_acc: float,
                    s_ious: dict, rs_iou: float, cm: float):
    """填充 S2 一轮验证结果到历史记录."""
    hist["epoch"].append(ep)
    hist["loss"].append(s2_tl)
    hist["val_loss"].append(c_ious.get("mIoU", 0))
    hist["val_acc"].append(c_acc)
    hist["val_cm"].append(cm)
    hist["val_rs_iou"].append(rs_iou)
    for cls_name in CLASS_NAMES.values():
        hist[f"iou_{cls_name}"].append(c_ious.get(cls_name, 0.0))
    for cls_name in CLASS_NAMES.values():
        pm = c_prf.get(cls_name, {})
        hist[f"prf_{cls_name}_precision"].append(pm.get("precision", 0.0))
        hist[f"prf_{cls_name}_recall"].append(pm.get("recall", 0.0))
        hist[f"prf_{cls_name}_f1"].append(pm.get("f1", 0.0))
    for tag in ["S2_噪声", "S2_钢筋", "S2_套筒"]:
        hist[f"s2pure_{tag}"].append(s_ious.get(tag, 0.0))


def fill_history_a2(hist: dict, ep: int, tl: float, vl: float,
                    ious: dict, acc: float, prf: dict, cm: float):
    """填充 A2 一轮验证结果到历史记录."""
    hist["epoch"].append(ep)
    hist["loss"].append(tl)
    hist["val_loss"].append(vl)
    hist["acc"].append(acc)
    hist["cm"].append(cm)
    for cls_name in CLASS_NAMES.values():
        hist[f"iou_{cls_name}"].append(ious.get(cls_name, 0.0))
    for cls_name in CLASS_NAMES.values():
        pm = prf.get(cls_name, {})
        hist[f"prf_{cls_name}_precision"].append(pm.get("precision", 0.0))
        hist[f"prf_{cls_name}_recall"].append(pm.get("recall", 0.0))
        hist[f"prf_{cls_name}_f1"].append(pm.get("f1", 0.0))


# ── 保存/加载 ──

def save_history(ckpt_dir: str, tag: str, history: dict):
    """每 epoch 即时写盘防丢失 (npz + json)."""
    try:
        np.savez(
            os.path.join(ckpt_dir, f"history_{tag}.npz"),
            **{k: np.array(v) for k, v in history.items()},
        )
        with open(os.path.join(ckpt_dir, f"history_{tag}.json"), "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        console.print(f"[yellow][WARN] 保存历史失败: {e}[/]")


def load_history(ckpt_dir: str, tag: str) -> dict | None:
    """加载历史记录 JSON, 不存在返回 None."""
    path = os.path.join(ckpt_dir, f"history_{tag}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── checkpoint 保存/加载/恢复 ──

def save_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    scheduler=None, epoch: int = 0, **extra):
    """保存完整 checkpoint (模型+优化器+调度器+额外信息)."""
    state = {"model_state": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    state.update(extra)
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    scheduler=None, device=None) -> dict:
    """加载 checkpoint, 恢复 model/optimizer/scheduler, 返回 epoch.

    兼容两种格式:
      - 新格式: {"model_state": ..., "optimizer_state": ..., "scheduler_state": ..., "epoch": ...}
      - 旧格式: 纯 state_dict (torch.save(model.state_dict(), path))
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device, weights_only=False)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
        if optimizer is not None and "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and "scheduler_state" in state:
            scheduler.load_state_dict(state["scheduler_state"])
        return state.get("epoch", 0)
    else:
        # 旧格式: 纯 state_dict
        model.load_state_dict(state)
        return 0


def resume_or_init(ckpt_path: str, model, optimizer, scheduler,
                   ckpt_dir: str, tag: str):
    """
    断点续训入口: 若 checkpoint 存在则恢复, 否则返回初始 epoch=0 + 空历史.

    Returns:
        (start_epoch: int, best_metric: float, history: dict)
    """
    hist_path = os.path.join(ckpt_dir, f"history_{tag}.json")
    if os.path.exists(ckpt_path) and os.path.exists(hist_path):
        epoch = load_checkpoint(ckpt_path, model, optimizer, scheduler)
        history = load_history(ckpt_dir, tag) or {}
        best_epochs = history.get("epoch", [])
        if best_epochs:
            # 从历史记录推断最佳指标
            cm_vals = history.get("cm", []) or history.get("val_cm", [])
            best_metric = max(cm_vals) if cm_vals else 0.0
        else:
            best_metric = 0.0
        console.print(f"  [green]断点续训: 从 epoch {epoch + 1} 继续 (最佳 cm={best_metric:.3f})[/]")
        return epoch + 1, best_metric, history

    return 0, 0.0, None
