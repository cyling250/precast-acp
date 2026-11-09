"""
预制混凝土构件点云生成器。

合成原理（2026-10-16 定调，参照真实云实测）：
  - 布局 C 唯一：每构件一端外露纵筋束+垫片（公头）、另一端端面套筒空心圆环阵（母头）；
  - 套筒 = 端面凸起 2~8mm 的平面空心圆环（孔内无点），环标混凝土（决策 A）；
  - 纵筋 = 部分圆弧半圆柱 + 稀疏密度 + 端头垫片，标纵筋；
  - 叠合梁顶面外伸 U 形箍筋标箍筋；
  - 构件平躺地面 z≈0：底面/离地<30mm 结构性缺失，端面随机无数据斑；
  - 噪声降级（实测 σ≈0.4mm）。

标签体系 v2:
  0 = non_member  非预制构件点（地面+背景杂物+噪声）
  1 = concrete    混凝土（含套筒端面环）
  2 = rebar       纵筋（含垫片）
  3 = stirrup     箍筋（叠合梁外伸发夹筋）

公开 API:
  make_point_cloud / build_member — 单构件生成（返回 points/labels/metadata GT）
  DatasetGenerator               — 批量数据集生成器
"""

from ._config import (
    LABEL_NON_MEMBER, LABEL_CONCRETE, LABEL_REBAR, LABEL_STIRRUP,
    NUM_CLASSES, CLASS_NAMES,
    set_seed, get_rng,
)
from ._noise import apply_noise
from ._builder import build_member, make_point_cloud, BUILDERS
from ._dataset import DatasetGenerator, render_qa, test_keys, COMP_TYPES

__all__ = [
    "LABEL_NON_MEMBER", "LABEL_CONCRETE", "LABEL_REBAR", "LABEL_STIRRUP",
    "NUM_CLASSES", "CLASS_NAMES",
    "set_seed", "get_rng",
    "apply_noise",
    "build_member", "make_point_cloud", "BUILDERS",
    "DatasetGenerator", "render_qa", "test_keys", "COMP_TYPES",
]
