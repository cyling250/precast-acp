"""
泛型训练/验证循环: 各阶段通过 forward_and_labels_fn 自定义数据处理.
"""

import random

import torch
import torch.nn as nn
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from .common import console, DEVICE
from .metrics import compute_iou, compute_precision_recall_f1, compute_accuracy


def train_one_epoch(model, loader, criterion, optimizer, epoch, tag="train",
                    device=DEVICE, forward_and_labels_fn=None) -> float:
    """
    训练一轮, 返回平均 loss.

    Args:
        forward_and_labels_fn: callable(model, batch) -> (logits, labels)
    """
    model.train()
    total_loss, valid_batches = 0, 0
    progress = Progress(
        TextColumn(f"[bold]{tag} Epoch {epoch:2d}[/] train"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("loss={task.fields[loss]:.4f}"),
        TimeRemainingColumn(),
        console=console,
    )
    task = progress.add_task("train", total=len(loader), loss=0)
    with progress:
        for batch in loader:
            if forward_and_labels_fn is not None:
                logits, labels = forward_and_labels_fn(model, batch)
            else:
                pts, feat, lbl, scale = batch
                pts, lbl = pts.to(device), lbl.to(device)
                logits = model(pts, scale=None)
                labels = lbl

            loss = criterion(logits.permute(0, 2, 1), labels)
            if torch.isnan(loss) or torch.isinf(loss):
                progress.update(task, advance=1, loss=float("nan"))
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            valid_batches += 1
            progress.update(task, advance=1, loss=loss.item())
    return total_loss / max(valid_batches, 1)


@torch.no_grad()
def validate_one_epoch(model, loader_fn, criterion, class_names, cm_fn=None,
                       device=DEVICE, forward_and_labels_fn=None, batch_stride=1):
    """
    验证一轮, 返回 (avg_loss, ious, acc, prf, cm).

    Args:
        class_names: dict {int: str}
        cm_fn: callable(ious_dict) -> float, 计算 clean mIoU.
        batch_stride: 每 K 批取 1 批评估（v2.1 验证集远大于训练集时控制开销；
            DataLoader 洗牌后抽取对全量无偏）。
    """
    model.eval()
    loader = loader_fn()
    total_loss = 0.0
    n_eval = 0
    all_p, all_t = [], []
    batches = list(loader) if batch_stride > 1 else loader
    if batch_stride > 1:
        offset = random.Random(0).randrange(batch_stride)
        batches = batches[offset::batch_stride]
    progress = Progress(
        TextColumn("[bold]val[/]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("loss={task.fields[loss]:.4f}"),
        TimeRemainingColumn(),
        console=console,
    )
    task = progress.add_task("val", total=len(batches), loss=0)
    with torch.no_grad(), progress:
        for batch in batches:
            if forward_and_labels_fn is not None:
                logits, labels = forward_and_labels_fn(model, batch)
            else:
                pts, feat, lbl, scale = batch
                pts, lbl = pts.to(device), lbl.to(device)
                logits = model(pts, scale=None)
                labels = lbl

            loss = criterion(logits.permute(0, 2, 1), labels)
            if torch.isnan(loss) or torch.isinf(loss):
                total_loss += 0.0
            else:
                total_loss += loss.item()
            n_eval += 1
            all_p.append(logits.cpu())
            all_t.append(labels.cpu())
            progress.update(task, advance=1, loss=loss.item()
                            if not (torch.isnan(loss) or torch.isinf(loss)) else 0.0)
    p, t = torch.cat(all_p), torch.cat(all_t)
    ious = compute_iou(p, t, class_names)
    acc = compute_accuracy(p, t)
    prf = compute_precision_recall_f1(p, t, class_names)
    cm = cm_fn(ious) if cm_fn else ious.get("mIoU", 0)
    return total_loss / max(n_eval, 1), ious, acc, prf, cm
