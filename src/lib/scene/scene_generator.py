"""
场景级点云生成器（T-037 M0，SDS v1）——编排入口。

复用单构件生成器（lib.pointcloud_generator，不改其内部逻辑，决策 D5）：
逐构件 build_member（翘曲/贴地缺失/周向弧窗/斑等 MVS 级退化件内烘焙）→
密度变化 → 噪声 → 剥单件局部地面（label 0=地面+杂物）→ 面实例就近归属
→ 场景位姿变换（平移+绕 Z 偏航 ±15°+模式滚转）。场景级新增：堆叠接触
求解（稳健分位锚定，翘曲不累积）、垫块盒体（class 4）、场景级地面一 次、
z-buffer 可见性、标签烘焙（points/member_id/face_id/class_id/visible +
face_table + member_table + scan_poses）。

工况编排规格（docs/t037_scene_face_pipeline.md §3，用户按三视图核对）：
  S1 单件 / S2 平铺（间隙 200-400，行 2-4 件 × 1-2 排，异型件仅平铺）/
  S3 柱堆叠（垫块 ×2 @ 1/3、2/3 长度 ±50-200，层间垫块错开 ≥100，层间
  xy 抖 ≤50）/ S4 梁堆叠（滚转 ±90°，梳齿朝侧面，L 梁不参与）/
  S5 墙直叠（无垫块，层间 xy ≤30）/ S6 组合 / D1 通用多面体复杂场景。

已知限制（复用约束的代价，记入决策记录）：单件的贴地缺失/周向弧窗按
构件局部姿态烘焙——S4 滚转后原贴地面朝侧但依然无点（保守方向：该缺的
还是缺）；堆叠接触用点云稳健分位（p0.5/p99.5）锚定，接触带点云自然
互混，面 GT 平面保持名义几何+刚性变换。
"""

import argparse
import hashlib
import json
import os
import zlib

import numpy as np

from ..pointcloud_generator import (
    build_member, LABEL_NON_MEMBER,
)
from ..pointcloud_generator._config import (
    NOISE_LEVELS, CONCRETE_ROUGHNESS, GROUND_Z,
    GRAZING_DROP_Z, GRAZING_RAMP_Z,
    MIDSPAN_VISIBLE_ARC_DEG, MIDSPAN_ARC_CENTER_JITTER, MIDSPAN_FULL_END_ZONE,
    WARP_SIGMA_RANGE, WARP_WAVELENGTH_RANGE, WARP_N_COMPONENTS,
    GROUND_MARGIN_RANGE, GROUND_ROUGHNESS_RANGE, GROUND_HEIGHT_OFFSET,
    GROUND_EDGE_IRREG,
)
from ..pointcloud_generator._noise import apply_noise, apply_density_variation
from ..pointcloud_generator._geometry import sample_ground_plane, sample_box_surface
from ..pointcloud_generator._concrete import (
    sample_prism_side_faces, sample_prism_end_face,
    _ensure_ccw, _polygon_outward_normals,
)
from .visibility import make_scan_poses, compute_visibility
from .face_labels import assign_member_faces

# ── 编排规格常量（docs §3，用户定稿） ──────────────────────────
GAP = (200.0, 400.0)               # S2/S6 构件间扫描通道间隙 mm
YAW_DEG = 15.0                     # 绕 Z 偏航抖动 ±15°
DUN_THICK_X = (80.0, 120.0)        # 垫块厚（构件轴向 X）mm
DUN_HEIGHT_Z = (60.0, 100.0)       # 垫块高（Z）mm
DUN_OVERHANG = (20.0, 60.0)        # 垫块长(Y) = 接触面宽 + 两侧挑出
DUN_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0)   # 架在下层 1/3 与 2/3 长度处
DUN_FRACTION_JIT = (50.0, 200.0)   # 名义位置抖动
DUN_MIN_OFFSET_X = 100.0           # 多层垫块错开 ≥100（绝不对齐）
LAYER_XY_JIT = {"s3": 50.0, "s4": 50.0, "s5": 30.0}  # 层间 xy 抖动上限
S5_LAYER_GAP = (2.0, 8.0)          # S5 直叠层间微缝
STACK_LAYERS = (2, 4)              # 每垛 2-4 件
GROUND_DENS_FACTOR = (0.03, 0.08)  # 场景地面密度 × 构件密度（控 4M 上限）
SCENE_X_STAGGER = (0.0, 400.0)     # S2 排内件 x 向错位

# 场景构件统一密度（单件 low 档；多件场景控点数，偏差见决策记录）
MEMBER_DENSITY = 0.10

MAX_POINTS = 4_000_000             # §4.2 N 上限（超限体素降采样）
PCT_LO, PCT_HI = 0.5, 99.5         # 稳健分位（接触锚定/范围测量）

ALL_TYPES = [
    "rectangular_column", "l_shaped_column", "rectangular_beam", "t_beam",
    "rectangular_composite_beam", "t_composite_beam", "l_composite_beam",
    "shear_wall_panel",
]
FLAT_ONLY = {"l_shaped_column", "t_beam", "l_composite_beam"}  # 异型件仅平铺
S4_TYPES = ["rectangular_beam", "rectangular_composite_beam", "t_composite_beam"]

# 场景 class_id（§4.2）：0 地面/背景 1 混凝土面 2 钢筋/垫板 3 箍筋/发夹
# 4 垫块 5 其他杂物（保留给真实场景）


def _rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _extent(pts: np.ndarray) -> dict:
    """稳健范围（p0.5/p99.5）——翘曲离群不带入接触锚定。"""
    lo = np.percentile(pts, PCT_LO, axis=0)
    hi = np.percentile(pts, PCT_HI, axis=0)
    return {"x_lo": lo[0], "x_hi": hi[0], "y_lo": lo[1],
            "y_hi": hi[1], "z_lo": lo[2], "z_hi": hi[2]}


# ── 单件准备（生成→退化→剥地面→面归属） ─────────────────────────

def _prepare_member(comp_type: str, rng: np.random.Generator) -> dict:
    res = build_member(comp_type, density=MEMBER_DENSITY, rng=rng)
    pts, lbls = res["points"], res["labels"]
    pts, lbls = apply_density_variation(pts, lbls, rng)
    level = "medium" if rng.random() < 0.4 else "low"
    pts, lbls = apply_noise(pts, lbls, NOISE_LEVELS[level], rng)
    keep = lbls != LABEL_NON_MEMBER        # 剥单件局部地面+杂物（场景级另铺）
    pts, lbls = pts[keep], lbls[keep]
    fid, faces = assign_member_faces(pts, lbls, res["metadata"])
    return {"pts": pts.astype(np.float32), "lbls": lbls.astype(np.int32),
            "fid": fid, "faces": faces, "meta": res["metadata"]}


# ── D1 通用多面体（随机截面×拉伸，同一套退化机理） ────────────────

def _poly_largest(poly):
    """buffer(0) 的结果统一取最大单块（自交星形修复回 MultiPolygon，
    2026-10-26 d1×400 规模下实锤崩溃点）。空返回 None。"""
    if poly.is_empty:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return None if poly.is_empty else poly


def _random_section(rng: np.random.Generator) -> np.ndarray:
    """星形随机多边形（5-8 边，半径波动→自然凹）∪ 第二体小多边形（两体
    拼接，D6）；顶点稀疏化到 ≤14 边。自交/退化（buffer(0) 空或 Multi-
    Polygon、简化后 <3 顶点）重抽——重试只消耗失败路径的 rng，已生成
    场景不受影响。"""
    from shapely.geometry import Polygon as ShpPoly

    def _star(cy, cz, r_base, n_min, n_max):
        n = int(rng.integers(n_min, n_max + 1))
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        ang += rng.uniform(0, 0.2)
        r = r_base * rng.uniform(0.45, 1.15, n)
        return [(cy + r[i] * np.cos(ang[i]), cz + r[i] * np.sin(ang[i]))
                for i in range(n)]

    for _retry in range(16):
        poly = _poly_largest(
            ShpPoly(_star(0.0, 0.0, rng.uniform(400, 950), 5, 8)).buffer(0))
        if poly is None:
            continue
        cen = poly.centroid
        blob = _poly_largest(
            ShpPoly(_star(cen.x + rng.uniform(-0.4, 0.4)
                          * poly.exterior.bounds[2],
                          cen.y + rng.uniform(-0.4, 0.4)
                          * poly.exterior.bounds[3],
                          rng.uniform(200, 450), 4, 6)).buffer(0))
        if blob is None:
            continue
        u = _poly_largest(poly.union(blob))
        if u is None:
            continue
        u = u.simplify(12.0, preserve_topology=True)
        coords = np.asarray(u.exterior.coords)[:-1]
        if len(coords) > 14:                     # 边数上限（D1：3-12 边量级）
            keep = np.linspace(0, len(coords), 13, endpoint=False).astype(int)
            coords = coords[keep]
        if len(coords) >= 3:
            return coords
    raise RuntimeError("_random_section: 16 次重抽仍退化（rng 异常）")


def _grazing_filter(pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """贴地/掠射角滤波（与 build_member 同口径的本地复刻——原函数为闭包）。"""
    if len(pts) == 0:
        return pts
    z = pts[:, 2] - GROUND_Z
    below = z < GRAZING_DROP_Z
    ramp = (z >= GRAZING_DROP_Z) & (z < GRAZING_RAMP_Z)
    keep = ~below
    prob = (z[ramp] - GRAZING_DROP_Z) / (GRAZING_RAMP_Z - GRAZING_DROP_Z)
    keep[ramp] = rng.random(int(ramp.sum())) < prob
    return pts[keep]


def _warp_field(rng: np.random.Generator):
    """SfM 低频翘曲场（与 build_member 同配方的本地复刻）。"""
    sigma = rng.uniform(*WARP_SIGMA_RANGE)
    comps = []
    for _ in range(WARP_N_COMPONENTS):
        lam = rng.uniform(*WARP_WAVELENGTH_RANGE)
        k = rng.normal(size=3)
        k /= np.linalg.norm(k)
        comps.append((2 * np.pi / lam * k, rng.uniform(0, 2 * np.pi, 3)))
    amp = sigma / np.sqrt(WARP_N_COMPONENTS) / 1.22

    def _warp(p):
        out = p.astype(np.float64, copy=True)
        for k, ph in comps:
            out = out + amp * np.sin(out @ k[:, None] + ph[None, :])
        return out

    return _warp


def _prepare_generic_body(rng: np.random.Generator) -> dict:
    """D1 体：随机（可凹）截面 × 拉伸；混凝土单类、无附属结构。
    退化保护：采样/滤波后 0 点（细窄截面+掠射滤波可清空）整体重抽，
    2026-10-26 d1×400 规模实锤（_extent 空数组分位数崩溃）。"""
    for _retry in range(16):
        sec = _random_section(rng)
        L = float(rng.uniform(2000.0, 6000.0))
        arc_half = rng.uniform(*MIDSPAN_VISIBLE_ARC_DEG) / 2.0
        arc_center = rng.uniform(-MIDSPAN_ARC_CENTER_JITTER,
                                 MIDSPAN_ARC_CENTER_JITTER)
        con = sample_prism_side_faces(
            sec, 0.0, L, MEMBER_DENSITY, rng, roughness=CONCRETE_ROUGHNESS,
            arc_window_deg=(arc_center, arc_half),
            end_full_zone_mm=MIDSPAN_FULL_END_ZONE)
        e0 = sample_prism_end_face(sec, 0.0, rng.uniform(0.40, 0.90), rng, -1)
        eL = sample_prism_end_face(sec, L, rng.uniform(0.40, 0.90), rng, +1)
        pts = _grazing_filter(np.vstack([con, e0, eL]), rng)
        if len(pts) < 100:                    # 退化体：重抽
            continue
        pts = _warp_field(rng)(pts)
        lbls = np.ones(len(pts), np.int32)
        pts, lbls = apply_density_variation(pts, lbls, rng)
        level = "medium" if rng.random() < 0.4 else "low"
        pts, lbls = apply_noise(pts, lbls, NOISE_LEVELS[level], rng)
        meta = {"type": "generic_body", "length_mm": L,
                "section_polygon_yz": [[float(a), float(b)] for a, b in sec]}
        fid, faces = assign_member_faces(pts, lbls, meta)
        return {"pts": pts.astype(np.float32), "lbls": lbls.astype(np.int32),
                "fid": fid, "faces": faces, "meta": meta}
    raise RuntimeError("_prepare_generic_body: 16 次重抽仍退化")


# ── 垫块接触宽度（名义截面按"朝下面"求和） ────────────────────────

def _contact_width(meta: dict, roll: int) -> float:
    """接触面宽 = 朝下外表面边的边长和（S3 平放：底边；S4 滚转 ±90°：
    local ±y 边）。"""
    sec = _ensure_ccw(np.asarray(meta["section_polygon_yz"], np.float64))
    nrms = _polygon_outward_normals(sec)
    span = 0.0
    for i in range(len(sec)):
        j = (i + 1) % len(sec)
        el = float(np.linalg.norm(sec[j] - sec[i]))
        ny, nz = float(nrms[i, 0]), float(nrms[i, 1])
        if roll == 0:
            down = nz < -0.9
        elif roll > 0:
            down = ny < -0.9
        else:
            down = ny > 0.9
        if down:
            span += el
    return span if span > 0 else float(sec[:, 0].ptp())


# ── 工况编排（返回 placements=[(m,R,t)] 与 dunnages=[(center,size)]）──

def _plan_single(rng, types):
    m = _prepare_member(str(rng.choice(types)), rng)
    R = _rz(rng.uniform(-YAW_DEG, YAW_DEG))
    ext = _extent(m["pts"] @ R.T)
    t = np.array([-ext["x_lo"], -ext["y_lo"],
                  rng.uniform(*GROUND_HEIGHT_OFFSET) - ext["z_lo"]])
    return [(m, R, t)], []


def _plan_flat(rng, n_rows=(1, 2), per_row=(2, 4), types=ALL_TYPES):
    """S2 平铺：排内沿 Y 并列（间隙 200-400），排间沿 X 推进；件长随机
    天然参差，相对面大面积缺失由可见性机理产生。"""
    placements = []
    row_x = 0.0
    scene_x_hi = 0.0
    for _r in range(int(rng.integers(n_rows[0], n_rows[1] + 1))):
        y_cur = 0.0
        row_x_hi = 0.0
        for _j in range(int(rng.integers(per_row[0], per_row[1] + 1))):
            m = _prepare_member(str(rng.choice(types)), rng)
            R = _rz(rng.uniform(-YAW_DEG, YAW_DEG))
            ext = _extent(m["pts"] @ R.T)
            t = np.array([row_x + rng.uniform(*SCENE_X_STAGGER) - ext["x_lo"],
                          y_cur - ext["y_lo"],
                          rng.uniform(*GROUND_HEIGHT_OFFSET) - ext["z_lo"]])
            placements.append((m, R, t))
            y_cur += (ext["y_hi"] - ext["y_lo"]) + rng.uniform(*GAP)
            row_x_hi = max(row_x_hi, ext["x_hi"] + t[0])
        scene_x_hi = row_x_hi
        row_x = row_x_hi + rng.uniform(*GAP)
    return placements, [], scene_x_hi


def _plan_stack(rng, kind: str, types):
    """S3/S4/S5 堆垛。垫块在下层顶面 1/3、2/3 处（±抖动），层间错开
    ≥100（X 向），上层底面=垫块顶面（稳健分位锚定）；S5 无垫块直叠。
    在垛局部系（X=构件轴向）内规划，返回后再整体平移。"""
    roll = int(rng.choice([-1, 1])) if kind == "s4" else 0
    R = _rz(rng.uniform(-YAW_DEG, YAW_DEG))
    if roll:
        R = R @ _rx(90.0 * roll)
    placements, dunnages = [], []
    prev_top_z = rng.uniform(*GROUND_HEIGHT_OFFSET)
    prev_ext = prev_t = None
    prev_dun_x: list[float] = []
    jit = LAYER_XY_JIT[kind]
    for _k in range(int(rng.integers(STACK_LAYERS[0], STACK_LAYERS[1] + 1))):
        m = _prepare_member(str(rng.choice(types)), rng)
        ext = _extent(m["pts"] @ R.T)
        if prev_ext is None:
            t = np.array([-ext["x_lo"], -ext["y_lo"], prev_top_z - ext["z_lo"]])
        else:
            dun_h = 0.0 if kind == "s5" else rng.uniform(*DUN_HEIGHT_Z)
            gap = rng.uniform(*S5_LAYER_GAP) if kind == "s5" else 0.0
            t = np.array([
                prev_ext["x_lo"] + prev_t[0] + rng.uniform(-jit, jit) - ext["x_lo"],
                (prev_ext["y_lo"] + prev_ext["y_hi"]) / 2 + prev_t[1]
                + rng.uniform(-jit, jit) - (ext["y_lo"] + ext["y_hi"]) / 2,
                prev_top_z + dun_h + gap - ext["z_lo"],
            ])
            if kind in ("s3", "s4"):
                # 垫块：公共覆盖区内 1/3、2/3（±抖动），层间错开 ≥100
                wext = {"x_lo": ext["x_lo"] + t[0], "x_hi": ext["x_hi"] + t[0],
                        "y_lo": ext["y_lo"] + t[1], "y_hi": ext["y_hi"] + t[1]}
                x_lo_c = max(prev_ext["x_lo"] + prev_t[0], wext["x_lo"])
                x_hi_c = min(prev_ext["x_hi"] + prev_t[0], wext["x_hi"])
                span = max(x_hi_c - x_lo_c, 100.0)
                len_y = _contact_width(m["meta"], roll) + rng.uniform(*DUN_OVERHANG)
                cy = (wext["y_lo"] + wext["y_hi"]) / 2 + rng.uniform(-30.0, 30.0)
                thick = rng.uniform(*DUN_THICK_X)
                for f in DUN_FRACTIONS:
                    x_d = x_lo_c + f * span + rng.uniform(-DUN_FRACTION_JIT[1],
                                                          DUN_FRACTION_JIT[1])
                    for _try in range(8):   # 层间垫块错开 ≥100（X 向）
                        if all(abs(x_d - xp) >= DUN_MIN_OFFSET_X
                               for xp in prev_dun_x):
                            break
                        sign = 1.0 if x_d >= np.mean(prev_dun_x or [x_d]) else -1.0
                        x_d += sign * (DUN_MIN_OFFSET_X + rng.uniform(0, 120))
                    prev_dun_x.append(x_d)
                    dunnages.append((
                        np.array([x_d, cy, prev_top_z + dun_h / 2.0]),
                        np.array([thick, len_y, dun_h]),
                    ))
        placements.append((m, R, t))
        prev_ext, prev_t = ext, t
        prev_top_z = ext["z_hi"] + t[2]
    return placements, dunnages


def _plan_combo(rng):
    """S6 组合：S2 一排（2-3 件）+ 1-2 垛（S3/S4/S5 随机），通道间隙。"""
    placements, dunnages = [], []
    flat, _, flat_x_hi = _plan_flat(rng, n_rows=(1, 1), per_row=(2, 3))
    placements.extend(flat)
    flat_y = np.mean([_extent(m["pts"] @ R.T)["y_hi"] + t[1]
                      for m, R, t in flat])
    n_stacks = int(rng.integers(1, 3))
    x_cur = flat_x_hi + rng.uniform(*GAP)
    for _s in range(n_stacks):
        kind = str(rng.choice(["s3", "s4", "s5"]))
        types = {"s3": ["rectangular_column"], "s4": S4_TYPES,
                 "s5": ["shear_wall_panel"]}[kind]
        st_pl, st_dun = _plan_stack(rng, kind, types)
        exts = [_extent(m["pts"] @ R.T) for m, R, t in st_pl]
        sx_lo = min(e["x_lo"] + t[0] for e, (m, R, t) in zip(exts, st_pl))
        sy_c = np.mean([(e["y_lo"] + e["y_hi"]) / 2 + t[1]
                        for e, (m, R, t) in zip(exts, st_pl)])
        shift = np.array([x_cur - sx_lo, flat_y - sy_c, 0.0])
        placements.extend([(m, R, t + shift) for m, R, t in st_pl])
        dunnages.extend([(c + shift, s) for c, s in st_dun])
        st_x_hi = max(e["x_hi"] + t[0] + shift[0]
                      for e, (m, R, t) in zip(exts, st_pl))
        x_cur = st_x_hi + rng.uniform(*GAP)
    return placements, dunnages


# ── GT 变换（member_table） ───────────────────────────────────

def _member_table_entry(meta: dict, R: np.ndarray, t: np.ndarray) -> dict:
    def xp(p):
        return (R @ np.asarray(p, np.float64) + t).tolist()

    entry = {
        "type": meta.get("type"),
        "pose_R": R.tolist(), "pose_t": [float(v) for v in t],
        "length_mm": float(meta.get("length_mm", 0.0)),
        "n_rebars": int(meta.get("n_rebars", 0)),
        "rebar_diam": meta.get("rebar_diam"),
        "protrusion_mm": meta.get("protrusion_mm"),
        "sleeve_inner_diam": meta.get("sleeve_inner_diam"),
        "sleeve_outer_diam": meta.get("sleeve_outer_diam"),
        "rebar_layout_mode": meta.get("rebar_layout_mode"),
        "rebar_gt": [
            {**{k: v for k, v in g.items() if k not in ("p0", "p1")},
             "p0": xp(g["p0"]), "p1": xp(g["p1"])}
            for g in meta.get("rebar_gt", [])
        ],
        "sleeve_gt": [
            {**{k: v for k, v in g.items()
                if k not in ("center_x", "center_yz")},
             "center": xp([g["center_x"], g["center_yz"][0], g["center_yz"][1]]),
             "normal": (R @ np.array([-1.0, 0.0, 0.0])).tolist()}
            for g in meta.get("sleeve_gt", [])
        ],
        "hairpin_gt": [
            {"diam": g.get("diam"), "height": g.get("height"),
             "foot_l": xp(g["p_foot_l"]), "foot_r": xp(g["p_foot_r"])}
            for g in meta.get("hairpin_gt", [])
        ],
    }
    wg = meta.get("window_gt")
    if wg:
        xc = (wg["x"][0] + wg["x"][1]) / 2.0
        yc = (wg["y"][0] + wg["y"][1]) / 2.0
        entry["window_gt"] = {
            "center": xp([xc, yc, wg["t"] / 2.0]),
            "dir_x": (R @ np.array([1.0, 0.0, 0.0])).tolist(),
            "dir_y": (R @ np.array([0.0, 1.0, 0.0])).tolist(),
            "half_x": (wg["x"][1] - wg["x"][0]) / 2.0,
            "half_y": (wg["y"][1] - wg["y"][0]) / 2.0,
            "bar_diam": wg.get("bar_diam"), "n_bars": wg.get("n_bars"),
        }
    else:
        entry["window_gt"] = None
    return entry


# ── 烘焙 ─────────────────────────────────────────────────────


def _cap_points(arrays: list, max_n: int = MAX_POINTS) -> list:
    """超 4M 先 2mm 体素降采样，仍超逐级加粗（3/4/5/6/8mm）。"""
    n = len(arrays[0])
    if n <= max_n:
        return arrays
    pts = arrays[0]
    for cell in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        keys = np.floor(pts / cell).astype(np.int64)
        view = np.ascontiguousarray(keys).view(
            [("x", np.int64), ("y", np.int64), ("z", np.int64)]).ravel()
        _, idx = np.unique(view, return_index=True)
        idx.sort()
        if len(idx) <= max_n:
            break
    return [a[idx] for a in arrays]


def generate_scene(mode: str, rng: np.random.Generator,
                   types: list = None) -> dict:
    """生成一个场景（S1-S6/D1），返回烘焙后的数组与表。

    types：S1/S2 的构件类型覆盖（M1 验收需指定失败构型族等价类型）；
    S3-S5 类型集由工况规格固定，不受此参影响。"""
    if mode == "s1":
        placements, dunnages = _plan_single(rng, types or ALL_TYPES)
    elif mode == "s2":
        placements, dunnages, _ = _plan_flat(rng, types=types or ALL_TYPES)
    elif mode == "s3":
        placements, dunnages = _plan_stack(rng, "s3", ["rectangular_column"])
    elif mode == "s4":
        placements, dunnages = _plan_stack(rng, "s4", S4_TYPES)
    elif mode == "s5":
        placements, dunnages = _plan_stack(rng, "s5", ["shear_wall_panel"])
    elif mode == "s6":
        placements, dunnages = _plan_combo(rng)
    elif mode == "d1":
        n = int(rng.integers(2, 7))
        placements, dunnages = [], []
        y_cur, row_x_hi = 0.0, 0.0
        for _i in range(n):
            m = _prepare_generic_body(rng)
            R = _rz(rng.uniform(-YAW_DEG, YAW_DEG))
            ext = _extent(m["pts"] @ R.T)
            t = np.array([rng.uniform(*SCENE_X_STAGGER) - ext["x_lo"],
                          y_cur - ext["y_lo"],
                          rng.uniform(*GROUND_HEIGHT_OFFSET) - ext["z_lo"]])
            placements.append((m, R, t))
            y_cur += (ext["y_hi"] - ext["y_lo"]) + rng.uniform(*GAP)
            row_x_hi = max(row_x_hi, ext["x_hi"] + t[0])
    else:
        raise ValueError(f"未知模式: {mode}")

    # ── 变换烘焙：构件点 / 类别 / 面表 / 构件表 ──
    pts_list, cls_list, mid_list, fid_list = [], [], [], []
    face_rows, member_tables = [], []
    for mi, (m, R, t) in enumerate(placements):
        p = m["pts"] @ R.T + t
        pts_list.append(p)
        lbls = m["lbls"]
        cls = np.zeros(len(lbls), np.uint8)
        cls[lbls == 1] = 1
        cls[lbls == 2] = 2
        cls[lbls == 3] = 3
        cls_list.append(cls)
        mid_list.append(np.full(len(p), mi, np.int16))
        fid_list.append(m["fid"])
        for f in m["faces"]:
            n_sc = R @ f["n_local"]
            c_sc = R @ f["anchor_local"] + t
            face_rows.append({
                "member_id": mi, "face_id": int(f["face_id"]),
                "type": int(f["type"]), "n": n_sc, "c": c_sc,
                "d": float(n_sc @ c_sc),
            })
        member_tables.append(_member_table_entry(m["meta"], R, t))

    for c, s in dunnages:
        p = sample_box_surface(c, s, MEMBER_DENSITY * 0.6, rng)
        pts_list.append(p)
        cls_list.append(np.full(len(p), 4, np.uint8))
        mid_list.append(np.full(len(p), -1, np.int16))
        fid_list.append(np.full(len(p), -1, np.int16))

    # ── 场景级地面一次：连续铺满（含构件下方）——用户 2026-10-26 指正：
    # 真实扫描的地面与构件相接触，白色围空不存在；构件下方/侧后方的地面
    # 遮挡由 z-buffer 可见性统一剔除（先生成全量，再模拟相机绕一圈删） ──
    all_placed = np.concatenate(pts_list)
    margin = rng.uniform(*GROUND_MARGIN_RANGE)
    x_min, x_max = all_placed[:, 0].min(), all_placed[:, 0].max()
    y_min, y_max = all_placed[:, 1].min(), all_placed[:, 1].max()
    gnd = sample_ground_plane(
        y_min=y_min - margin, y_max=y_max + margin,
        x_min=x_min - margin, x_max=x_max + margin,
        density=MEMBER_DENSITY * rng.uniform(*GROUND_DENS_FACTOR),
        roughness=rng.uniform(0.2, 0.8), rng=rng,
        exclude_polygon=None, exclude_margin=0.0,
        irregular_edge_mm=rng.uniform(*GROUND_EDGE_IRREG),
        z_offset=rng.uniform(*GROUND_HEIGHT_OFFSET),
    )
    pts_list.append(gnd)
    cls_list.append(np.zeros(len(gnd), np.uint8))
    mid_list.append(np.full(len(gnd), -1, np.int16))
    fid_list.append(np.full(len(gnd), -1, np.int16))

    points = np.concatenate(pts_list).astype(np.float32)
    class_id = np.concatenate(cls_list)
    member_id = np.concatenate(mid_list)
    face_id = np.concatenate(fid_list)

    points, class_id, member_id, face_id = _cap_points(
        [points, class_id, member_id, face_id])

    # ── 可见性（z-buffer，位姿入档） ──
    lo = np.percentile(points, 0.1, axis=0)
    hi = np.percentile(points, 99.9, axis=0)
    bbox = np.array([lo[0], hi[0], lo[1], hi[1], 0.0, hi[2]])
    poses = make_scan_poses(bbox, rng)
    visible = compute_visibility(points, poses)

    # ── face_table 收尾（点数/AABB/可见比来自降采样后的最终点云） ──
    ft = np.zeros((len(face_rows), 18), np.float32)
    for i, fr in enumerate(face_rows):
        sel = ((member_id == fr["member_id"]) & (face_id == fr["face_id"]))
        n_pts = int(sel.sum())
        n_vis = int((sel & visible).sum())
        if n_pts:
            pp = points[sel]
            aabb = np.concatenate([pp.min(0), pp.max(0)])
        else:
            aabb = np.concatenate([fr["c"], fr["c"]])
        ft[i] = [fr["member_id"], fr["face_id"], fr["type"],
                 *fr["n"], fr["d"], *fr["c"], *aabb, n_pts, n_vis]

    scene_meta = {
        "mode": mode,
        "n_members": len(placements),
        "member_types": [m["meta"]["type"] for m, _R, _t in placements],
        "n_points": int(len(points)),
        "n_faces": int(len(face_rows)),
        "n_dunnage": len(dunnages),
        "density": MEMBER_DENSITY,
        "generator": "scene_v1 (T-037 M0)",
    }
    return {
        "points": points,
        "class_id": class_id,
        "member_id": member_id,
        "face_id": face_id,
        "visible": visible,
        "face_table": ft,
        "member_table_json": json.dumps(member_tables, ensure_ascii=False),
        "scene_meta_json": json.dumps(scene_meta, ensure_ascii=False),
        "scan_poses": poses,
    }


# ── split 纪律：按构型 id 不相交（沿用 load_split 纪律精神） ────────

def _split_of(config_id: str) -> str:
    h = int(hashlib.md5(config_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if h < 6 else ("val" if h < 8 else "test")


def main():
    ap = argparse.ArgumentParser(
        description="场景级点云生成器（T-037 SDS v1）：--mode s1..s6,d1")
    ap.add_argument("--mode", default="s1",
                    help="s1|s2|s3|s4|s5|s6|d1|all")
    ap.add_argument("--count", type=int, default=1, help="每模式场景数")
    ap.add_argument("--out", default=None,
                    help="输出根目录（默认 mock/scene/v1）")
    ap.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args()

    root = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))), "mock", "scene", "v1")
    modes = ["s1", "s2", "s3", "s4", "s5", "s6", "d1"] \
        if args.mode == "all" else [args.mode]

    index_path = os.path.join(root, "meta_index.json")
    index = {"version": "sds_v1", "scenes": []}
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    known = {s["file"] for s in index["scenes"]}

    for mode in modes:
        os.makedirs(os.path.join(root, mode), exist_ok=True)
        base = args.seed + zlib.crc32(mode.encode("utf-8")) % 100000
        for i in range(args.count):
            cfg_idx = i % 12 + 1                     # 构型族 12/模式
            config_id = f"{mode}_cfg{cfg_idx:02d}"
            rng = np.random.default_rng(abs(base + i * 7919) % (2**31))
            sc = generate_scene(mode, rng)
            meta = json.loads(sc["scene_meta_json"])
            meta["config_id"] = config_id
            meta["split"] = _split_of(config_id)
            fname = f"scene_{mode}_cfg{cfg_idx:02d}_{i:04d}.npz"
            fpath = os.path.join(root, mode, fname)
            np.savez_compressed(
                fpath,
                points=sc["points"], class_id=sc["class_id"],
                member_id=sc["member_id"], face_id=sc["face_id"],
                visible=sc["visible"], face_table=sc["face_table"],
                member_table_json=sc["member_table_json"],
                scene_meta_json=json.dumps(meta, ensure_ascii=False),
                scan_poses=sc["scan_poses"],
            )
            rel = os.path.relpath(fpath, root).replace("\\", "/")
            if rel not in known:
                index["scenes"].append({
                    "file": rel, "mode": mode, "config_id": config_id,
                    "split": meta["split"],
                    "member_types": meta["member_types"],
                    "n_points": meta["n_points"],
                })
            vis_r = float(sc["visible"].mean())
            print(f"[{mode}] {fname}: {meta['n_points']} pts, "
                  f"{meta['n_members']} 件, {meta['n_faces']} 面, "
                  f"垫块 {meta['n_dunnage']}, 可见率 {vis_r:.2f}")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=1)
    print(f"完成 → {root}（meta_index.json 共 {len(index['scenes'])} 场景）")


if __name__ == "__main__":
    main()
