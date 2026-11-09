"""
lib/segmentation —— 单阶段四类语义分割。

四类：0 非预制构件点 / 1 混凝土 / 2 纵筋 / 3 箍筋。
骨干 PointMLP，
训练单位 = 150mm 球查询补丁（8192 点），类均衡正样本采样保留。
套筒不设类（点太少学不动，端面空心圆走几何检测——方案 §一）。
"""

from .pointmlp import PointMLPSeg
from ._base import FocalLoss, PerSampleWeightedCE, farthest_point_sample

__all__ = ["PointMLPSeg", "FocalLoss", "PerSampleWeightedCE", "farthest_point_sample"]
