"""
套筒点云生成 v2。

布局 C 母头端：混凝土端面（x=0）上每个纵筋对位处一个灌浆套筒出口，
表现为【平面空心圆环】：端面上凸出 2~8 mm 的金属环 + 中心无点圆孔。
环点标 LABEL_CONCRETE（决策 A：套筒不设 DL 类，靠几何检测空心圆）。
"""

import numpy as np
from typing import Optional

from ._config import (
    LABEL_CONCRETE,
    SLEEVE_RING_PROUD_RANGE,
    SLEEVE_RING_HOLE_MARGIN,
    SLEEVE_RING_DENSITY,
    SLEEVE_SPEC_MAP,
)
from ._geometry import sample_annulus_ring


def generate_sleeve_rings(
    positions_yz: list,
    x_face: float,
    sleeve_outer_diam: float,
    sleeve_inner_diam: float,
    density: float,
    rng: np.random.Generator,
    normal_sign: int = -1,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    端面套筒环阵列。

    Parameters
    ----------
    positions_yz : list of (y, z) 套筒中心（与纵筋对位）。
    x_face : 端面 x 坐标。
    sleeve_outer_diam / sleeve_inner_diam : 环外/内径 mm。
    density : 基准表面密度（环会按 SLEEVE_RING_DENSITY_FACTOR 加密）。
    normal_sign : 端面朝向（-x 端面 = -1）。

    Returns
    -------
    points, labels, sleeve_gt
    """
    r_in = sleeve_inner_diam / 2.0
    r_out = sleeve_outer_diam / 2.0
    normal = np.array([float(normal_sign), 0.0, 0.0])
    ring_dens = rng.uniform(*SLEEVE_RING_DENSITY)  # v2.1 绝对密度（实测端面 0.78）

    pts_list, lbl_list, gt_list = [], [], []
    for (py, pz) in positions_yz:
        proud = rng.uniform(*SLEEVE_RING_PROUD_RANGE)
        # 环正面位于端面外侧 proud 处
        center = np.array([x_face + normal_sign * proud, py, pz])
        pts = sample_annulus_ring(
            center, normal, r_in, r_out, ring_dens, rng,
            thickness=proud,
        )
        pts_list.append(pts)
        lbl_list.append(np.full(len(pts), LABEL_CONCRETE, dtype=np.int32))
        gt_list.append({
            "center_yz": [float(py), float(pz)],
            "center_x": float(x_face),
            "inner_diam": float(sleeve_inner_diam),
            "outer_diam": float(sleeve_outer_diam),
            "proud_mm": float(proud),
        })

    if not pts_list:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int32), []
    return np.vstack(pts_list), np.concatenate(lbl_list), gt_list


def sleeve_hole_radius(sleeve_inner_diam: float) -> float:
    """端面挖孔半径 = 内腔半径 + 余量（孔内无点）。"""
    return sleeve_inner_diam / 2.0 + SLEEVE_RING_HOLE_MARGIN


def sleeve_spec_for_rebar(rebar_diam: float) -> tuple[float, float, float]:
    """按筋径取 (外径, 内腔径, 套筒长) —— JG/T 398-2019 规格映射。"""
    return SLEEVE_SPEC_MAP[rebar_diam]
