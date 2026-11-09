"""
钢筋点云生成 v2。

- 外露纵筋束（布局 C 公头端）：部分圆弧半圆柱采样 + 轻微弯曲 + 端头垫片，
  统一标 LABEL_REBAR（垫片属筋部件——决策 A）。
- 叠合梁顶面外伸 U 形箍筋（发夹筋）：标 LABEL_STIRRUP。
"""

import numpy as np
from typing import Optional

from ._config import (
    LABEL_REBAR, LABEL_STIRRUP,
    REBAR_ARC_SPAN_DEG, REBAR_ARC_UP_BIAS_DEG, REBAR_DENSITY_FACTOR,
    REBAR_VISIBLE_R_FACTOR, REBAR_DENS_START_FACTOR, REBAR_DENS_PEAK_T,
    REBAR_DENS_TIP_FACTOR, REBAR_DENS_DIP_PROB, REBAR_DENS_DIP_FACTOR,
    REBAR_BOW_RANGE, WASHER_DIAM_FACTOR, WASHER_LENGTH_RANGE,
    WASHER_DENSITY_FACTOR, HAIRPIN_DENSITY_FACTOR, HAIRPIN_HEIGHT_RANGE,
)
from ._geometry import sample_cylinder_arc, sample_cylinder_segment


def _arc_center_dir(outward_yz: np.ndarray, up_bias_deg: float) -> np.ndarray:
    """弧中心方向 = 截面外向偏转向上（扫描自上方及四周）。"""
    out3 = np.array([0.0, outward_yz[0], outward_yz[1]])
    out3 = out3 / (np.linalg.norm(outward_yz) + 1e-9)
    w = np.tan(np.radians(up_bias_deg))
    c = out3 + w * np.array([0.0, 0.0, 1.0])
    return c / np.linalg.norm(c)


def generate_rebar_bundle(
    positions_yz: list,
    section_center_yz: tuple,
    x_face: float,
    rebar_diam: float,
    protrusion: float,
    density: float,
    rng: np.random.Generator,
    with_washers: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    布局 C 公头端纵筋束。

    Parameters
    ----------
    positions_yz : list of (y, z) 纵筋在端面上的位置。
    section_center_yz : 截面中心（用于确定各筋的朝外方向）。
    x_face : 混凝土端面 x 坐标（筋由此向外伸）。
    rebar_diam : 筋径 mm。
    protrusion : 外露长 mm。
    density : 基准表面密度 pts/mm²。

    Returns
    -------
    points, labels, rebar_gt : GT 每根含 center_line(两端点)、arc_span_deg、bow、washer
    """
    radius = rebar_diam / 2.0
    cy, cz = section_center_yz
    pts_list, lbl_list, gt_list = [], [], []

    for (py, pz) in positions_yz:
        # v2.1 可见弧半径：MVS 对细筋的有效半径小于名义（实测 5-7mm/名义 10-12mm）
        radius_vis = radius * rng.uniform(*REBAR_VISIBLE_R_FACTOR)
        # v2.1 沿轴密度剖面：面端→峰（距端面 10-35%）→末端衰减（实测 0.68×）
        bar_factor = rng.uniform(*REBAR_DENSITY_FACTOR)
        dens_bar = density * bar_factor
        peak_t = rng.uniform(*REBAR_DENS_PEAK_T)
        tip_f = rng.uniform(*REBAR_DENS_TIP_FACTOR)

        def _prof(t: float) -> float:
            if t <= peak_t:
                k = t / max(peak_t, 1e-9)
                return REBAR_DENS_START_FACTOR + (1.0 - REBAR_DENS_START_FACTOR) * k
            k = (t - peak_t) / max(1.0 - peak_t, 1e-9)
            return 1.0 + (tip_f - 1.0) * k ** 1.2

        outward = np.array([py - cy, pz - cz])
        arc_span = np.radians(rng.uniform(*REBAR_ARC_SPAN_DEG))
        up_bias = rng.uniform(*REBAR_ARC_UP_BIAS_DEG)
        c_dir = _arc_center_dir(outward, up_bias)

        bow = rng.uniform(*REBAR_BOW_RANGE) if REBAR_BOW_RANGE[1] > 0 else 0.0
        # 弯曲方向：垂直于轴的随机方向
        bow_dir = np.array([0.0, rng.uniform(-1, 1), rng.uniform(-0.5, 1)])
        bow_dir /= np.linalg.norm(bow_dir) + 1e-9

        p0 = np.array([x_face, py, pz])
        p1 = np.array([x_face + protrusion, py, pz])

        # 采样中心线上的弧段（把直线离散为 8 段折线，逐段弧采样，叠加抛物线弯曲）；
        # v2.1 每段独立密度：剖面值 × 随机凹陷（凹陷段低于 3mm cap → 细化后仍稀疏）
        n_seg = 8
        seg_ts = np.linspace(0, 1, n_seg + 1)
        dips = [rng.uniform(*REBAR_DENS_DIP_FACTOR)
                if rng.random() < REBAR_DENS_DIP_PROB else 1.0
                for _ in range(n_seg)]
        seg_pts = []
        for i in range(n_seg):
            t0, t1 = seg_ts[i], seg_ts[i + 1]
            s0 = p0 + t0 * (p1 - p0) + bow * 4 * t0 * (1 - t0) * bow_dir
            s1 = p0 + t1 * (p1 - p0) + bow * 4 * t1 * (1 - t1) * bow_dir
            seg = sample_cylinder_arc(
                s0, s1, radius_vis,
                dens_bar * _prof((t0 + t1) / 2.0) * dips[i],
                rng, c_dir, arc_span,
            )
            if len(seg):
                seg_pts.append(seg)
        if seg_pts:
            pts_list.append(np.vstack(seg_pts))
            lbl_list.append(np.full(sum(len(s) for s in seg_pts), LABEL_REBAR, dtype=np.int32))

        gt_list.append({
            "p0": [float(v) for v in p0],
            "p1": [float(v) for v in p1],
            "diam": float(rebar_diam),
            "visible_r": float(radius_vis),
            "dens_peak_t": float(peak_t),
            "dens_tip_factor": float(tip_f),
            "arc_span_deg": float(np.degrees(arc_span)),
            "bow_mm": float(bow),
        })

        # 端头垫片（整周短圆柱，套在筋端部；v2.1 密度=该筋峰段密度×系数）
        if with_washers:
            w_len = rng.uniform(*WASHER_LENGTH_RANGE)
            w_diam = rebar_diam * rng.uniform(*WASHER_DIAM_FACTOR)
            w_dens = dens_bar * rng.uniform(*WASHER_DENSITY_FACTOR)
            w0 = np.array([x_face + protrusion - w_len, py, pz])
            w1 = np.array([x_face + protrusion, py, pz])
            wpts = sample_cylinder_segment(w0, w1, w_diam / 2.0, w_dens, rng)
            # 垫片端面圆盘（朝外一面可见）
            n_disk = max(8, int(np.pi * (w_diam / 2) ** 2 * w_dens))
            th = rng.uniform(0, 2 * np.pi, size=n_disk)
            rr = np.sqrt(rng.uniform(0, 1, size=n_disk)) * (rebar_diam / 2 + 1.0)
            disk = np.column_stack([
                np.full(n_disk, w1[0]), w1[1] + rr * np.cos(th), w1[2] + rr * np.sin(th),
            ])
            pts_list.append(np.vstack([wpts, disk]))
            lbl_list.append(np.full(len(wpts) + n_disk, LABEL_REBAR, dtype=np.int32))
            gt_list[-1]["washer"] = {"diam": float(w_diam), "length": float(w_len)}

    if not pts_list:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int32), []
    return np.vstack(pts_list), np.concatenate(lbl_list), gt_list


def generate_u_hairpins(
    x_min: float,
    x_max: float,
    top_z: float,
    y_inner: float,
    y_outer: float,
    diam: float,
    spacing_x: float,
    density: float,
    rng: np.random.Generator,
    top_face_inner_y: Optional[tuple] = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    叠合梁顶面外伸 U 形箍筋（发夹筋），标 LABEL_STIRRUP。

    U 形在 YZ 截面：两竖腿立于混凝土顶面 + 顶部横杆连接；开口朝下扣在顶面上。
    y_inner/y_outer: 两腿的 y 位置（U 的宽度）。
    只采样朝上/朝外弧（细筋稀疏可见），腿的内侧不可见。

    Returns
    -------
    points, labels, hairpin_gt
    """
    radius = diam / 2.0
    dens = density * rng.uniform(*HAIRPIN_DENSITY_FACTOR)
    height = float(rng.uniform(*HAIRPIN_HEIGHT_RANGE))
    y1, y2 = min(y_inner, y_outer), max(y_inner, y_outer)
    pts_list, lbl_list, gt_list = [], [], []

    # 沿 x 的位置——v2.3 加密/疏松分区（用户 2026-10-24；GB 50011 抗震
    # 构造：端部箍筋加密、跨中可疏松。旧版注释写"端部加密"实为等距）
    dens_mult = rng.uniform(0.45, 0.6)   # 加密区间距系数
    mid_mult = rng.uniform(1.3, 1.8)     # 跨中疏松系数
    zone_len = min(rng.uniform(800.0, 1500.0), (x_max - x_min) * 0.35)
    x_positions = []
    x = x_min + spacing_x * dens_mult * 0.5
    while x < x_max - spacing_x * 0.3:
        x_positions.append(x)
        if x < x_min + zone_len or x > x_max - zone_len:
            x += spacing_x * dens_mult
        else:
            x += spacing_x * mid_mult

    up = np.array([0.0, 0.0, 1.0])
    for xp in x_positions:
        segs = []
        # 左腿（弧心朝 -y 外侧偏上）
        c_l = np.array([0.0, -0.8, 0.6]); c_l /= np.linalg.norm(c_l)
        segs.append((
            np.array([xp, y1, top_z]), np.array([xp, y1, top_z + height]),
            c_l,
        ))
        # 右腿
        c_r = np.array([0.0, 0.8, 0.6]); c_r /= np.linalg.norm(c_r)
        segs.append((
            np.array([xp, y2, top_z]), np.array([xp, y2, top_z + height]),
            c_r,
        ))
        # 顶部横杆（弧心朝 +z）
        segs.append((
            np.array([xp, y1, top_z + height]), np.array([xp, y2, top_z + height]),
            up,
        ))
        for s0, s1, cd in segs:
            # 横杆沿 y 方向时弧心仍可用（垂直分量生效）
            pts = sample_cylinder_arc(s0, s1, radius, dens, rng, cd,
                                      np.radians(200.0))
            if len(pts):
                pts_list.append(pts)
                lbl_list.append(np.full(len(pts), LABEL_STIRRUP, dtype=np.int32))
        gt_list.append({
            "x": float(xp), "y1": float(y1), "y2": float(y2),
            "top_z": float(top_z), "height": float(height), "diam": float(diam),
        })

    if not pts_list:
        return np.empty((0, 3)), np.empty((0,), dtype=np.int32), []
    return np.vstack(pts_list), np.concatenate(lbl_list), gt_list
