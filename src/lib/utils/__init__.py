"""
lib/utils — 工具包: 常量、种子、指标、数据加载、训练循环、历史记录.
"""

from .common import (
    console,
    DEVICE,
    LR,
    WEIGHT_DECAY,
    NUM_POINTS,
    BATCH_SIZE,
    NUM_CLASSES,
    KEEP_RATIO,
    SAMPLES_PER_EPOCH,
    CLASS_NAMES,
    S1_NAMES,
    OUT_BASE,
    CHECKPOINT_BASE,
    SENSITIVITY_BASE,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    set_seed,
    anchor_project_path,
    map_labels,
    serialize_metrics,
)

from .metrics import (
    compute_iou,
    compute_precision_recall_f1,
    compute_accuracy,
    cm_iou,
)

from .datasets import (
    GpuCachedTrainLoader,
    RandomValDataset,
    RandomSampleDataset,
    create_dataloaders,
    PATCH_RADIUS_MM,
)

from .training import (
    train_one_epoch,
    validate_one_epoch,
)

from .history import (
    build_history_template_s1,
    build_history_template_s2,
    build_history_template_a2,
    fill_history_s1,
    fill_history_s2,
    fill_history_a2,
    save_history,
    load_history,
    save_checkpoint,
    load_checkpoint,
    resume_or_init,
)

__all__ = [
    # common
    "console",
    "DEVICE",
    "LR",
    "WEIGHT_DECAY",
    "NUM_POINTS",
    "BATCH_SIZE",
    "NUM_CLASSES",
    "KEEP_RATIO",
    "SAMPLES_PER_EPOCH",
    "CLASS_NAMES",
    "S1_NAMES",
    "OUT_BASE",
    "CHECKPOINT_BASE",
    "SENSITIVITY_BASE",
    "TRAIN_DIR",
    "VAL_DIR",
    "TEST_DIR",
    "set_seed",
    "anchor_project_path",
    "map_labels",
    "serialize_metrics",
    # metrics
    "compute_iou",
    "compute_precision_recall_f1",
    "compute_accuracy",
    "cm_iou",
    # datasets
    "GpuCachedTrainLoader",
    "RandomValDataset",
    "RandomSampleDataset",
    "create_dataloaders",
    "PATCH_RADIUS_MM",
    # training
    "train_one_epoch",
    "validate_one_epoch",
    # history
    "build_history_template_s1",
    "build_history_template_s2",
    "build_history_template_a2",
    "fill_history_s1",
    "fill_history_s2",
    "fill_history_a2",
    "save_history",
    "load_history",
    "save_checkpoint",
    "load_checkpoint",
    "resume_or_init",
]
