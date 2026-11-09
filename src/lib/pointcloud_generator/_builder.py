"""
构件组装 v2：布局 C 唯一 —— 每构件一端（x=L）外露纵筋束+垫片（公头），
另一端（x=0）端面套筒空心圆环阵（母头）。

统一坐标约定：构件沿 X 轴平躺、截面在 YZ 平面、底面贴地 z≈0、
扫描自上方及四周 → 贴地面与离地 <30mm 结构性缺失。

七种构件：矩形柱 / L 形柱 / 矩形梁 / 倒 T 梁 / 三种叠合梁（顶面外伸箍筋）。
"""

import numpy as np
from typing import Optional

from ._config import (
    LABEL_CONCRETE, LABEL_REBAR, LABEL_STIRRUP, LABEL_NON_MEMBER,
    COLUMN_WIDTH_RANGE, COLUMN_DEPTH_RANGE, COLUMN_HEIGHT_RANGE,
    L_SHAPED_COLUMN_FLANGE_RANGE, L_SHAPED_COLUMN_THICK_RANGE,
    BEAM_WIDTH_RANGE, BEAM_HEIGHT_RANGE, BEAM_LENGTH_RANGE,
    T_BEAM_FLANGE_WIDTH_RANGE, T_BEAM_FLANGE_HEIGHT_RANGE,
    T_BEAM_WEB_WIDTH_RANGE, T_BEAM_TOTAL_HEIGHT_RANGE,
    STANDARD_REBAR_DIAMETERS, CONCRETE_COVER,
    CONCRETE_ROUGHNESS, END_FACE_DENSITY_RANGE,
    GRAZING_DROP_Z, GRAZING_RAMP_Z,
    END_FACE_GAP_PROB, END_FACE_GAP_RADIUS, END_FACE_MAX_GAPS,
    END_FACE_ROUGH_SIGMA, FOAM_PROB, FOAM_N_RANGE, FOAM_R_RANGE,
    FOAM_H_RANGE, FOAM_AREA_MAX,
    HAIRPIN_DIAM_RANGE, HAIRPIN_DIAM_SPECS, HAIRPIN_SPACING_RANGE,
    REBAR_PROTRUSION_RANGE,
    MIDSPAN_VISIBLE_ARC_DEG, MIDSPAN_ARC_CENTER_JITTER, MIDSPAN_FULL_END_ZONE,
    WARP_SIGMA_RANGE, WARP_WAVELENGTH_RANGE, WARP_N_COMPONENTS,
    GROUND_Z,
    WALL_HEIGHT_RANGE, WALL_THICK_RANGE, WALL_WIDTH_RANGE,
    WALL_WIN_X_RANGE, WALL_WIN_Y_RANGE, WALL_WIN_EDGE_MIN,
    WALL_EDGE_BAR_DIAMS,
)
from ._concrete import (
    sample_prism_side_faces, sample_prism_end_face,
    carve_circular_holes_2d, carve_ellipse_gap,
)
from ._rebar import generate_rebar_bundle, generate_u_hairpins
from ._sleeve import generate_sleeve_rings, sleeve_hole_radius, sleeve_spec_for_rebar
from ._ground import generate_ground, generate_background_blobs


# ─── 截面定义（YZ 平面，底边贴地 z=0，y 居中） ───────────────


def _section_rect(w: float, h: float) -> np.ndarray:
    return np.array([[-w / 2, 0.0], [w / 2, 0.0], [w / 2, h], [-w / 2, h]])


def _section_l(w: float, tf: float, tw: float, h: float) -> np.ndarray:
    """L 形：翼缘在下（宽 w 厚 tf），腹板在左侧（厚 tw 总高 h）。"""
    return np.array([
        [-w / 2, 0.0], [w / 2, 0.0], [w / 2, tf],
        [-w / 2 + tw, tf], [-w / 2 + tw, h], [-w / 2, h],
    ])


def _section_t_upside_down(B: float, hf: float, bw: float, h: float) -> np.ndarray:
    """倒 T（翼缘平放朝下，腹板朝上）：底面宽 B×hf，腹板 bw×h。"""
    return np.array([
        [-B / 2, 0.0], [B / 2, 0.0], [B / 2, hf],
        [bw / 2, hf], [bw / 2, h], [-bw / 2, h], [-B / 2, hf],
    ])


# ─── 纵筋/套筒位置 ─────────────────────────────────────────


def _poly_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _end_face_bumps(end_pts: np.ndarray, sign: float, section: np.ndarray,
                    rng: np.random.Generator):
    """端面泡沫残留 + 基础凹凸（2026-10-24 用户要求；实测校准见 _config）。

    位移式建模：x += sign·Δ；Δ = 基础半正态粗糙 ∨ 泡沫块高度（逐点取大），
    Δ≥0——凹不低于混凝土真实端面。泡沫点保持混凝土标签；RANSAC 距离
    阈值天然把凸起剔为离群（与部署机制一致）。返回泡沫块 GT 列表。
    """
    delta = np.abs(rng.normal(0.0, rng.uniform(*END_FACE_ROUGH_SIGMA),
                              len(end_pts)))
    foam_gt = []
    if len(end_pts) and rng.random() < FOAM_PROB:
        area_face = max(_poly_area(section), 1.0)
        area_used = 0.0
        n_blk = int(rng.integers(FOAM_N_RANGE[0], FOAM_N_RANGE[1] + 1))
        for _ in range(n_blk):
            if area_used > FOAM_AREA_MAX * area_face:
                break
            i0 = int(rng.integers(len(end_pts)))
            cy, cz = float(end_pts[i0, 1]), float(end_pts[i0, 2])
            r = float(rng.uniform(*FOAM_R_RANGE))
            h = float(rng.uniform(*FOAM_H_RANGE))
            d2 = (end_pts[:, 1] - cy) ** 2 + (end_pts[:, 2] - cz) ** 2
            m = d2 < r * r
            if m.sum() < 5:
                continue
            delta[m] = np.maximum(delta[m],
                                  h * rng.uniform(0.6, 1.0, int(m.sum())))
            area_used += np.pi * r * r
            foam_gt.append({"cy": cy, "cz": cz, "r": r, "h": h})
    end_pts[:, 0] += sign * delta
    return foam_gt


def _positions_rect(w: float, h: float, cover: float, count: int,
                    min_spacing: float = 120.0) -> list:
    """
    矩形截面位置：角部 + 顶/底边中部补足。

    补足位置强制最小间距（套筒外径最大 ~66mm，JGJ 355 净距要求）；
    底边配额 1/3（离地近可见性差）。
    v2.3b：底排 z 下界抬到 100（贴地缺失带 30-80mm 渐变删除——旧
    z0=cover≈50 落死区，35% 的环周围点被模拟缺失吃掉）。
    """
    y0, y1 = -w / 2 + cover, w / 2 - cover
    z0, z1 = max(cover, 100.0), h - cover
    pos = [(y0, z0), (y1, z0), (y0, z1), (y1, z1)]
    if count > 4:
        n_extra = count - 4
        n_bottom = n_extra // 3
        n_top = n_extra - n_bottom
        # 各边容量（角部已有两端点）
        n_slots = lambda span: int(np.floor(span / min_spacing)) - 1
        span = y1 - y0
        n_top = min(n_top, max(0, n_slots(span)))
        n_bottom = min(n_bottom, max(0, n_slots(span)))
        for i in range(n_top):
            y = y0 + (y1 - y0) * (i + 1) / (n_top + 1)
            pos.append((y, z1))
        for i in range(n_bottom):
            y = y0 + (y1 - y0) * (i + 1) / (n_bottom + 1)
            pos.append((y, z0))
    return pos[:count]


def _positions_polygon(polygon: np.ndarray, cover: float, count: int,
                       min_spacing: float = 100.0) -> list:
    """多边形截面位置：内缩多边形顶点 + 长边补足（成对最小间距过滤）。

    v2.3（2026-10-24 用户报告套筒露外修复）：
    - 收缩梯子 cover→0.85c→0.72c，下限 floor=cover−15（grid_cover=套筒
      外半径+25 → floor=外半径+5：环全嵌入且 ≥5mm 保护）。旧版"空则回退
      半 cover"把环放到混凝土表面之外（审计 494/2776 环露出，T 梁薄翼缘/
      L 形薄肢为重灾区）；
    - 候选硬过滤：shapely 距原多边形边界 < floor 一律剔除；
    - 凑不够根数先降间距（→0.8×→0.64×）再减数——保护层优先于根数。
    """
    from shapely.geometry import Polygon as ShapelyPolygon, Point

    poly = ShapelyPolygon(polygon)
    floor = cover - 5.0  # v2.3b：≥外半径+30，环带完整落面类带（inset-30）内
    shrunk = poly.buffer(-cover, resolution=8)
    if shrunk.is_empty or not shrunk.is_valid or shrunk.area <= 0:
        return []  # v2.3b：不回退——面类带约束优先于根数（间距梯子兜底凑数）

    inner = np.array(shrunk.exterior.coords)[:-1]
    # 跳过过近顶点（凹角圆弧伪影）
    min_dist = cover * 0.5
    merged = [inner[0]]
    for i in range(1, len(inner)):
        if np.linalg.norm(inner[i] - inner[i - 1]) >= min_dist:
            merged.append(inner[i])
    inner = np.array(merged)

    # 底边附近的顶点位置（z 小）可见性差 → v2.3b 底界抬到 100（避开
    # 贴地缺失带，与 _positions_rect 同口径）
    z_floor = max(cover * 0.8, 100.0)
    raw = [(float(p[0]), float(p[1])) for p in inner if p[1] > z_floor]
    if count > len(raw):
        edges = []
        for i in range(len(inner)):
            j = (i + 1) % len(inner)
            e_len = np.linalg.norm(inner[j] - inner[i])
            edges.append((inner[i], inner[j], e_len))
        total_len = sum(e[2] for e in edges)
        n_extra = count - len(raw)
        for (e0, e1, el) in sorted(edges, key=lambda e: -e[2]):
            if n_extra <= 0:
                break
            n_e = max(1, int(n_extra * el / total_len))
            for k in range(1, n_e + 1):
                t = k / (n_e + 1)
                pt = e0 + t * (e1 - e0)
                if pt[1] > z_floor:
                    raw.append((float(pt[0]), float(pt[1])))
                    n_extra -= 1
                    if n_extra <= 0:
                        break

    # 硬保护层过滤：距原多边形边界不足 floor 的候选剔除（防凹角圆弧伪影）
    def _ok(p):
        pt = Point(p)
        return poly.contains(pt) and poly.exterior.distance(pt) >= floor
    raw = [p for p in raw if _ok(p)]

    # 间距梯子凑数（保护层优先于根数：先降间距，不够再减根数）
    positions = []
    for ms in (min_spacing, min_spacing * 0.8, min_spacing * 0.64):
        positions = []
        for p in raw:
            if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= ms for q in positions):
                positions.append(p)
            if len(positions) >= count:
                break
        if len(positions) >= count:
            break
    return positions[:count]


def _wall_window(con_pts, L, w_h_wall, t, density, rng):
    """剪力墙板窗洞（v2.3，用户 2026-10-24：板+中间窗子+周围钢筋）。

    剔除大面（z≈t 顶面与 z≈0 底面带）落在窗洞矩形内的点 → 洞壁 4 面
    采样（矩形面 + 法向粗糙，斜视可见口径）→ 洞口周围边筋（贴洞壁凸出
    可见弧，标 LABEL_STIRRUP；保护层薄、扫描可见的工程口径）。
    返回 (con_pts', 边筋点, window_gt)。
    """
    from ._geometry import sample_rectangle_surface, sample_cylinder_arc

    wx = min(rng.uniform(*WALL_WIN_X_RANGE), L - 2 * WALL_WIN_EDGE_MIN)
    wy = min(rng.uniform(*WALL_WIN_Y_RANGE), w_h_wall - 2 * WALL_WIN_EDGE_MIN)
    margin_x = (L - wx) / 2 - WALL_WIN_EDGE_MIN
    margin_y = (w_h_wall - wy) / 2 - WALL_WIN_EDGE_MIN
    xc = L / 2 + rng.uniform(-margin_x, margin_x)
    yc = rng.uniform(-margin_y, margin_y)
    x_lo, x_hi = xc - wx / 2, xc + wx / 2
    y_lo, y_hi = yc - wy / 2, yc + wy / 2

    in_win = ((con_pts[:, 0] > x_lo + 8) & (con_pts[:, 0] < x_hi - 8)
              & (con_pts[:, 1] > y_lo + 8) & (con_pts[:, 1] < y_hi - 8))
    con_pts = con_pts[~in_win]

    zc = t / 2.0
    dens_w = rng.uniform(0.30, 0.60)  # 洞壁绝对密度（端面同级偏稀）
    parts = [con_pts]
    for c, nrm, wq in (((x_lo, yc, 0.0), (1, 0, 0), wy),
                       ((x_hi, yc, 0.0), (-1, 0, 0), wy),
                       ((xc, y_lo, 0.0), (0, 1, 0), wx),
                       ((xc, y_hi, 0.0), (0, -1, 0), wx)):
        nrm = np.asarray(nrm, np.float64)
        e1 = np.cross(nrm, (0.0, 0.0, 1.0))
        e1 /= np.linalg.norm(e1)
        uv = sample_rectangle_surface(wq, t, dens_w, rng)
        p = (np.asarray(c)[None] + uv[:, 0:1] * e1[None]
             + uv[:, 1:2] * np.array([0.0, 0.0, 1.0])[None]
             + nrm[None] * rng.normal(0, CONCRETE_ROUGHNESS, (len(uv), 1)))
        parts.append(p.astype(np.float32))
    con_pts = np.vstack(parts)

    # 洞口周围边筋：四条贴洞壁、凸向洞内的可见弧段（距洞边 40mm）
    bar_d = float(rng.choice(WALL_EDGE_BAR_DIAMS))
    dens_b = density * rng.uniform(0.4, 0.8)
    inset = 40.0
    segs = [
        ((x_lo, y_lo + inset, zc), (x_lo, y_hi - inset, zc), (1, 0, 0)),
        ((x_hi, y_lo + inset, zc), (x_hi, y_hi - inset, zc), (-1, 0, 0)),
        ((x_lo + inset, y_lo, zc), (x_hi - inset, y_lo, zc), (0, 1, 0)),
        ((x_lo + inset, y_hi, zc), (x_hi - inset, y_hi, zc), (0, -1, 0)),
    ]
    bars = []
    for s0, s1, cd in segs:
        p = sample_cylinder_arc(
            np.asarray(s0, np.float64), np.asarray(s1, np.float64),
            bar_d / 2.0, dens_b, rng, np.asarray(cd, np.float64),
            np.radians(200.0))
        if len(p):
            bars.append(p)
    bar_pts = (np.vstack(bars).astype(np.float32) if bars
               else np.zeros((0, 3), np.float32))
    gt = {"x": [float(x_lo), float(x_hi)], "y": [float(y_lo), float(y_hi)],
          "t": float(t), "bar_diam": bar_d, "n_bars": 4}
    return con_pts, bar_pts, gt


def _positions_surface_anchored(section, count, cover, rng,
                                min_spacing: float = 100.0,
                                mode: str = "uniform") -> list:
    """表面锚定布位（v2.4，用户 2026-10-24 定稿生成顺序）：
    混凝土几何 → 钢筋/套筒**锚定到截面边、沿内法向向内偏移 cover**
    （保护层即偏移本身——钢筋在截面内侧靠边，物理图像正确）→ 逐候选
    硬校验（多边形内 & 距边界 ≥ cover−5）→ 最小间距梯子贪心。

    取代 buffer(-cover) 收缩近似（旧法在 L/T 凹角等处仍产生出界布位，
    用户目检实锤）。cover=grid_cover（已含套筒外径口径）。底部 z<100
    候选丢弃（贴地缺失带）。
    """
    from shapely.geometry import Point as _ShpPoint, Polygon as _ShpPolygon

    poly = _ShpPolygon(section)
    sec = np.asarray(section, np.float64)
    n = len(sec)
    cen = sec.mean(0)
    edges = []
    for i in range(n):
        a, b = sec[i], sec[(i + 1) % n]
        e = b - a
        el = float(np.hypot(*e))
        if el < 40.0:
            continue
        tng = np.array([e[1], -e[0]]) / el
        mid = (a + b) / 2.0
        eps = 5.0
        n_in = tng if poly.contains(_ShpPoint(mid + tng * eps)) else -tng
        edges.append((a, b, n_in, el))
    if not edges:
        return []
    total = float(sum(e[3] for e in edges))
    # 每边候选数 ∝ 边长（×2 过采样供间距筛选）。
    # v2.4c 布位多样性（用户：合成复杂→真实鲁棒）：三种锚点分布模式
    #（mode 由 build_member 采样传入，meta 记录）——
    #   uniform  均匀+抖动；corner 角部优先（工程最常见）；
    #   clustered  边权随机重分配（某边密某边疏）。
    weights = np.array([e[3] for e in edges], np.float64)
    if mode == "clustered":
        weights = weights * rng.uniform(0.25, 2.5, len(edges))
    elif mode == "corner":
        weights = np.array([
            2.5 if (np.isclose(e[0][1], 0.0) or np.isclose(e[1][1], 0.0))
            else 1.0 for e in edges], np.float64)
    weights = weights / weights.sum()
    # v2.4d：间距驱动候选（旧"按根数分边+角部分位 0.06/0.94"使角部锚点
    # 距相邻边过近被硬过滤大面积拒绝——70/192 件根数<6 甚至 0）。
    cands = []
    for (a, b, n_in, el), w_e in zip(edges, weights):
        if mode == "clustered" and w_e < 0.04:
            continue  # clustered 疏边直接不出锚（多样性）
        n_c = max(1, int(el // 45.0))  # 45mm 细铺（供给≥2×需求，密度由梯子定）
        margin = max(cover, 45.0)
        if margin / el > 0.5:
            continue  # 短边放不下（margin 冲突）
        # v2.4e：corner 模式=两端保证 + 按边长铺中部（旧"两端交替"每边
        # 只有 2 个不同分位，角部成对锚点 ~90mm 被 100ms 贪心二选一 →
        # 恒 5 根）；uniform=等分
        if mode == "corner":
            fr_list = ([margin / el, 1.0 - margin / el]
                       + [(k + 0.5) / n_c for k in range(n_c)])
        else:
            fr_list = [(k + 0.5) / n_c for k in range(n_c)]
        for frac0 in fr_list:
            frac = min(max(frac0 + rng.uniform(-0.1, 0.1) / max(len(fr_list), 1),
                           margin / el), 1.0 - margin / el)
            anchor = a + (b - a) * frac
            c = anchor + n_in * cover  # 沿内法向向内偏移 cover（保护层）
            pt = _ShpPoint(c)
            if (poly.contains(pt)
                    and poly.exterior.distance(pt) >= cover - 5.0
                    and c[1] > 100.0):  # 避开贴地缺失带
                cands.append((float(c[0]), float(c[1])))
    if not cands:
        return []
    # 全库最小间距梯子（保护层优先于根数）；间距基准也随模式抖动
    ms_base = min_spacing * (rng.uniform(0.85, 1.15) if mode != "corner"
                             else 1.0)
    for ms in (ms_base, ms_base * 0.8, ms_base * 0.64):
        pos = []
        for c in cands:
            if all(np.hypot(c[0] - q[0], c[1] - q[1]) >= ms for q in pos):
                pos.append(c)
            if len(pos) >= count:
                break
        if len(pos) >= count:
            break
    return pos[:count]


# ─── 主构建器 ─────────────────────────────────────────────


def build_member(
    comp_type: str,
    density: float = 0.012,
    rng: Optional[np.random.Generator] = None,
    **overrides,
) -> dict:
    """
    统一构件构建入口（v2，布局 C）。

    Parameters
    ----------
    comp_type : 七种之一。
    density : 基准表面密度 pts/mm²。
    overrides : 长度/截面/筋数等显式参数（测试与 QA 用）。

    Returns
    -------
    dict(points, labels, metadata) — metadata 含拟合层验证所需全部 GT。
    """
    if rng is None:
        rng = np.random.default_rng()

    def R(a, b):
        return rng.uniform(a, b)

    # ── 1. 截面与长度 ──
    if comp_type == "rectangular_column":
        w = overrides.get("width", R(*COLUMN_WIDTH_RANGE))
        h = overrides.get("depth", R(*COLUMN_DEPTH_RANGE))
        L = overrides.get("length", R(*COLUMN_HEIGHT_RANGE))
        n_rebars = overrides.get("n_rebars", int(rng.integers(8, 17)))
        section = _section_rect(w, h)
    elif comp_type == "l_shaped_column":
        w = overrides.get("flange", R(*L_SHAPED_COLUMN_FLANGE_RANGE))
        tf = overrides.get("flange_thick", R(*L_SHAPED_COLUMN_THICK_RANGE))
        h = overrides.get("web", R(500.0, 900.0))
        tw = overrides.get("web_thick", R(200.0, min(400.0, w - 50.0)))
        L = overrides.get("length", R(*COLUMN_HEIGHT_RANGE))
        n_rebars = overrides.get("n_rebars", int(rng.integers(8, 17)))
        section = _section_l(w, tf, tw, h)
        w = max(w, tw)
    elif comp_type == "rectangular_beam":
        w = overrides.get("width", R(*BEAM_WIDTH_RANGE))
        h = overrides.get("height", R(*BEAM_HEIGHT_RANGE))
        L = overrides.get("length", R(*BEAM_LENGTH_RANGE))
        n_rebars = overrides.get("n_rebars", int(rng.integers(4, 13)))
        section = _section_rect(w, h)
    elif comp_type == "t_beam":
        B = overrides.get("flange_width", R(*T_BEAM_FLANGE_WIDTH_RANGE))
        hf = overrides.get("flange_height", R(*T_BEAM_FLANGE_HEIGHT_RANGE))
        bw = overrides.get("web_width", R(*T_BEAM_WEB_WIDTH_RANGE))
        h = overrides.get("total_height", R(*T_BEAM_TOTAL_HEIGHT_RANGE))
        L = overrides.get("length", R(*BEAM_LENGTH_RANGE))
        n_rebars = overrides.get("n_rebars", int(rng.integers(6, 15)))
        section = _section_t_upside_down(B, hf, bw, h)
        w = B
    elif comp_type == "rectangular_composite_beam":
        w = overrides.get("width", R(*BEAM_WIDTH_RANGE))
        h = overrides.get("height", R(300.0, 500.0))
        L = overrides.get("length", R(*BEAM_LENGTH_RANGE))
        n_rebars = overrides.get("n_rebars", int(rng.integers(4, 13)))
        section = _section_rect(w, h)
    elif comp_type == "t_composite_beam":
        B = overrides.get("flange_width", R(*T_BEAM_FLANGE_WIDTH_RANGE))
        hf = overrides.get("flange_height", R(*T_BEAM_FLANGE_HEIGHT_RANGE))
        bw = overrides.get("web_width", R(*T_BEAM_WEB_WIDTH_RANGE))
        h = overrides.get("total_height", R(*T_BEAM_TOTAL_HEIGHT_RANGE))
        L = overrides.get("length", R(*BEAM_LENGTH_RANGE))
        n_rebars = overrides.get("n_rebars", int(rng.integers(6, 15)))
        section = _section_t_upside_down(B, hf, bw, h)
        w = B
    elif comp_type == "l_composite_beam":
        w = overrides.get("flange", R(*L_SHAPED_COLUMN_FLANGE_RANGE))
        tf = overrides.get("flange_thick", R(*L_SHAPED_COLUMN_THICK_RANGE))
        h = overrides.get("web", R(500.0, 900.0))
        tw = overrides.get("web_thick", R(200.0, min(400.0, w - 50.0)))
        L = overrides.get("length", R(*BEAM_LENGTH_RANGE))
        n_rebars = overrides.get("n_rebars", int(rng.integers(8, 17)))
        section = _section_l(w, tf, tw, h)
    elif comp_type == "shear_wall_panel":
        # v2.3 剪力墙板（用户 2026-10-24：板+中间窗洞+洞口周围钢筋）。
        # 摆放：沿 x=墙宽；截面 yz=墙高(w=y 维)×板厚(h=z 维)——大面 z=t
        # 朝上、底面贴地；x=0 套筒端 / x=L 出筋端（布局 C 沿用）；窗洞
        # 在顶面中央、沿厚度贯通（_wall_window 处理剔除+洞壁+洞口边筋）。
        w = overrides.get("height", R(*WALL_HEIGHT_RANGE))
        h = overrides.get("thickness", R(*WALL_THICK_RANGE))
        L = overrides.get("length", R(*WALL_WIDTH_RANGE))
        n_rebars = overrides.get("n_rebars", int(rng.integers(8, 15)))
        section = _section_rect(w, h)
    else:
        raise ValueError(f"不支持的构件类型: {comp_type}")

    is_composite = "composite" in comp_type

    # ── 2. 钢筋/套筒参数 ──
    rebar_diam = overrides.get("rebar_diam", float(rng.choice(STANDARD_REBAR_DIAMETERS)))
    protrusion = overrides.get("protrusion", R(*REBAR_PROTRUSION_RANGE))
    sleeve_outer, sleeve_inner, sleeve_len = sleeve_spec_for_rebar(rebar_diam)
    # 位置网格保护层：须容纳套筒外径（同轴对位）；边距 35mm（v2.3b：
    # 面类带 inset-30——环带外缘须完整落入类内需 边距≥外半径+30，
    # 旧 +25 使 71.6% 环的外半圈落在被排除的壳层、视图上环悬真空带）
    grid_cover = max(CONCRETE_COVER, sleeve_outer / 2 + 35.0)

    _layout_mode = str(rng.choice(["uniform", "corner", "clustered"],
                                  p=[0.4, 0.35, 0.25]))
    positions = _positions_surface_anchored(section, n_rebars, grid_cover,
                                            rng, mode=_layout_mode)
    n_rebars = len(positions)
    # ── 2.5 错位钢筋（施工偏差，用户 2026-10-24）：逐根 yz 偏移（幅值
    #      5~20mm 混合：规范 ±5 内为主 + 少量超差）；重验底线=套筒外半径
    #      +30（环带完整嵌入面类带 inset-30，与布位 floor 同一口径——
    #      v2.4b：旧 grid_cover−5 在 cover=grid_cover 时比物理底线松 5mm）──
    from shapely.geometry import Point as _ShpPoint, Polygon as _ShpPolygon
    _poly = _ShpPolygon(section)
    _floor = sleeve_outer / 2 + 30.0
    _mis_max = rng.uniform(5.0, 20.0)
    _mis_max = min(_mis_max, max(grid_cover - 5.0 - _floor, 0.0))
    _shifted = []
    for (py, pz) in positions:
        q = (py, pz)
        for _ in range(10):
            cand = (py + rng.uniform(-_mis_max, _mis_max),
                    pz + rng.uniform(-_mis_max, _mis_max))
            if (_poly.contains(_ShpPoint(cand))
                    and _poly.exterior.distance(_ShpPoint(cand)) >= _floor):
                q = cand
                break
        _shifted.append(q)
    positions = _shifted
    section_center = (float(section[:, 0].mean()), float(section[:, 1].mean()))

    # ── 3. 混凝土：侧面+顶面（底面贴地不采）+ 两端面（绝对密度） ──
    # v2.1 周向可见窗：中段仅 110°~240° 可见弧（实测 36 箱仅 ~10 箱有占），
    # 端部 MIDSPAN_FULL_END_ZONE 区全可见，中间线性过渡。
    arc_half = rng.uniform(*MIDSPAN_VISIBLE_ARC_DEG) / 2.0
    arc_center = rng.uniform(-MIDSPAN_ARC_CENTER_JITTER, MIDSPAN_ARC_CENTER_JITTER)
    con_pts = sample_prism_side_faces(
        section, 0.0, L, density, rng, roughness=CONCRETE_ROUGHNESS,
        arc_window_deg=(arc_center, arc_half), end_full_zone_mm=MIDSPAN_FULL_END_ZONE,
    )
    dens_end_sleeve = rng.uniform(*END_FACE_DENSITY_RANGE)
    dens_end_rebar = rng.uniform(*END_FACE_DENSITY_RANGE)
    end0 = sample_prism_end_face(section, 0.0, dens_end_sleeve, rng, -1)
    endL = sample_prism_end_face(section, L, dens_end_rebar, rng, +1)

    # ── 3.1 剪力墙板窗洞（v2.3）：大面剔除 + 洞壁 + 洞口边筋 ──
    win_bar = np.zeros((0, 3), np.float32)
    window_gt = None
    if comp_type == "shear_wall_panel":
        con_pts, win_bar, window_gt = _wall_window(con_pts, L, w, h, density, rng)

    # 母头端面挖套筒孔 + 无数据斑
    def _pick_gap_c(x_end: float, max_try: int = 8):
        """无数据斑中心（v2.4 用户顺序要求：噪声/缺失最后且不得盖住连接区——
        斑心距所有钢筋/套筒 yz 位 ≥ 半轴+60，重试失败则本端免斑）。"""
        for _ in range(max_try):
            c = np.array([
                x_end,
                rng.uniform(section[:, 0].min() + 50, section[:, 0].max() - 50),
                rng.uniform(80, section[:, 1].max() - 50),
            ])
            rr = rng.uniform(*END_FACE_GAP_RADIUS, size=2).max()
            ok = all(np.hypot(c[1] - py, c[2] - pz) >= rr + 60.0
                     for (py, pz) in positions)
            if ok:
                return c
        return None

    hole_mask = carve_circular_holes_2d(
        end0, positions, [sleeve_hole_radius(sleeve_inner)] * len(positions),
    )
    if rng.random() < END_FACE_GAP_PROB:
        gap_c = _pick_gap_c(0.0)
        if gap_c is not None:
            gap_r = np.array([30.0, *rng.uniform(*END_FACE_GAP_RADIUS, size=2)])
            hole_mask = carve_ellipse_gap(end0, gap_c, gap_r, rng, hole_mask)
    end0 = end0[hole_mask]

    keepL = np.ones(len(endL), dtype=bool)
    if rng.random() < END_FACE_GAP_PROB:
        gap_c = _pick_gap_c(L)
        if gap_c is not None:
            gap_r = np.array([30.0, *rng.uniform(*END_FACE_GAP_RADIUS, size=2)])
            keepL = carve_ellipse_gap(endL, gap_c, gap_r, rng, keepL)
    endL = endL[keepL]

    # ── 3.5 端面泡沫残留 + 基础凹凸（位移式，Δ≥0；2026-10-24） ──
    foam0 = _end_face_bumps(end0, -1.0, section, rng)
    foamL = _end_face_bumps(endL, +1.0, section, rng)

    # ── 4. 套筒环（标混凝土） ──
    ring_pts, ring_lbls, sleeve_gt = generate_sleeve_rings(
        positions, 0.0, sleeve_outer, sleeve_inner, density, rng, normal_sign=-1,
    )

    # ── 5. 纵筋束 + 垫片（标纵筋） ──
    rebar_pts, rebar_lbls, rebar_gt = generate_rebar_bundle(
        positions, section_center, L, rebar_diam, protrusion, density, rng,
        with_washers=True,
    )

    # ── 6. 叠合梁顶面箍筋（标箍筋） ──
    stirrup_pts = np.empty((0, 3))
    stirrup_lbls = np.empty((0,), dtype=np.int32)
    hairpin_gt = []
    if is_composite:
        hp_diam = float(rng.choice(HAIRPIN_DIAM_SPECS))
        hp_spacing = rng.uniform(*HAIRPIN_SPACING_RANGE)
        top_z = float(section[:, 1].max())
        # 顶面 y 范围：取最顶部水平面的 y 区间
        top_edges = []
        n_pts_poly = len(section)
        for i in range(n_pts_poly):
            j = (i + 1) % n_pts_poly
            if abs(section[i, 1] - section[j, 1]) < 1e-9 and abs(section[i, 1] - top_z) < 1e-9:
                top_edges.append((min(section[i, 0], section[j, 0]),
                                  max(section[i, 0], section[j, 0])))
        if top_edges:
            y_lo = max(e[0] for e in top_edges)
            y_hi = min(e[1] for e in top_edges)
            y_mid = (y_lo + y_hi) / 2
            gap = max(120.0, (y_hi - y_lo) * 0.7)
            stirrup_pts, stirrup_lbls, hairpin_gt = generate_u_hairpins(
                0.15 * L, 0.9 * L, top_z,
                y_mid - gap / 2, y_mid + gap / 2,
                hp_diam, hp_spacing, density, rng,
            )

    if len(win_bar):  # 墙板洞口边筋并入箍筋类（v2.3）
        stirrup_pts = (np.vstack([stirrup_pts, win_bar])
                       if len(stirrup_pts) else win_bar)
        stirrup_lbls = np.concatenate([
            stirrup_lbls,
            np.full(len(win_bar), LABEL_STIRRUP, dtype=np.int32)])

    # ── 7. 贴地/掠射角可见性滤波（仅构件点；地面不受影响） ──
    def _grazing_filter(pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts
        z = pts[:, 2] - GROUND_Z
        below = z < GRAZING_DROP_Z
        ramp = (z >= GRAZING_DROP_Z) & (z < GRAZING_RAMP_Z)
        keep = ~below
        prob = (z[ramp] - GRAZING_DROP_Z) / (GRAZING_RAMP_Z - GRAZING_DROP_Z)
        keep[ramp] = rng.random(ramp.sum()) < prob
        return pts[keep]

    con_pts = _grazing_filter(con_pts)
    end0 = _grazing_filter(end0)
    endL = _grazing_filter(endL)
    ring_pts = _grazing_filter(ring_pts)
    rebar_pts = _grazing_filter(rebar_pts)
    stirrup_pts = _grazing_filter(stirrup_pts)

    # ── 8. 地面 + 杂物（标非预制构件点） ──
    y_min, y_max = float(section[:, 0].min()), float(section[:, 0].max())
    member_bbox = (0.0, L + protrusion, y_min, y_max)
    ground_pts = generate_ground(member_bbox, density, rng)
    blob_pts, blob_gt = generate_background_blobs(member_bbox, density, rng)

    # ── 8.5 SfM 低频翘曲（v2.1）：同一形变场作用于全部点（含地面），
    #      保持相对几何；GT 锚点（筋端点/套筒中心）同步形变保持真值一致 ──
    sigma_warp = rng.uniform(*WARP_SIGMA_RANGE)
    warp_comps = []
    for _ in range(WARP_N_COMPONENTS):
        lam = rng.uniform(*WARP_WAVELENGTH_RANGE)
        k_dir = rng.normal(size=3)
        k_dir = k_dir / np.linalg.norm(k_dir)
        warp_comps.append((2 * np.pi / lam * k_dir,
                           rng.uniform(0, 2 * np.pi, size=3)))
    warp_amp = sigma_warp / np.sqrt(WARP_N_COMPONENTS) / 1.22

    def _warp(p: np.ndarray) -> np.ndarray:
        if len(p) == 0:
            return p
        out = p.astype(np.float64, copy=True)
        for k, ph in warp_comps:
            out = out + warp_amp * np.sin(out @ k[:, None] + ph[None, :])
        return out

    con_pts, end0, endL = _warp(con_pts), _warp(end0), _warp(endL)
    ring_pts, rebar_pts = _warp(ring_pts), _warp(rebar_pts)
    stirrup_pts, ground_pts, blob_pts = _warp(stirrup_pts), _warp(ground_pts), _warp(blob_pts)
    for g in rebar_gt:
        g["p0"] = [float(v) for v in _warp(np.asarray(g["p0"])[None, :])[0]]
        g["p1"] = [float(v) for v in _warp(np.asarray(g["p1"])[None, :])[0]]
    for g in sleeve_gt:
        c3 = _warp(np.array([[g["center_x"], g["center_yz"][0], g["center_yz"][1]]]))[0]
        g["center_x"], g["center_yz"] = float(c3[0]), [float(c3[1]), float(c3[2])]
    for g in hairpin_gt:  # 腿脚锚点同步翘曲（顶面 BEV 腿峰 GT 与点云一致）
        for key, yy in (("p_foot_l", g["y1"]), ("p_foot_r", g["y2"])):
            p = _warp(np.array([[g["x"], yy, g["top_z"]]]))[0]
            g[key] = [float(v) for v in p]

    # ── 9. 组装 ──
    points = np.vstack([
        con_pts, end0, endL, ring_pts, rebar_pts, stirrup_pts,
        ground_pts, blob_pts,
    ])
    labels = np.concatenate([
        np.full(len(con_pts) + len(end0) + len(endL) + len(ring_pts), LABEL_CONCRETE, dtype=np.int32),
        np.full(len(rebar_pts), LABEL_REBAR, dtype=np.int32),
        np.full(len(stirrup_pts), LABEL_STIRRUP, dtype=np.int32),
        np.full(len(ground_pts) + len(blob_pts), LABEL_NON_MEMBER, dtype=np.int32),
    ])

    metadata = {
        "generator": "v2.1",
        "type": comp_type,
        "length_mm": float(L),
        "section_polygon_yz": [[float(a), float(b)] for a, b in section],
        "section_bbox": [float(y_max - y_min), float(section[:, 1].max())],
        "axis": [1.0, 0.0, 0.0],
        "ground_z": GROUND_Z,
        "arc_window_deg": [float(arc_center), float(arc_half)],
        "warp_sigma_mm": float(sigma_warp),
        "voxel_thin_mm_pipeline": 3.0,
        "sleeve_end_x": 0.0,
        "rebar_end_x": float(L),
        "n_rebars": int(n_rebars),
        "rebar_layout_mode": str(_layout_mode),
        "rebar_diam": float(rebar_diam),
        "protrusion_mm": float(protrusion),
        # v2.3：位置 GT 用翘曲后的根部锚点（p0=端面处插入点）——旧版存
        # 未翘曲 positions，与 warped 点云有 σ≤40mm 系统偏差（检测 GT 失真）
        "rebar_positions_yz": [[float(g["p0"][1]), float(g["p0"][2])]
                               for g in rebar_gt],
        "rebar_gt": rebar_gt,
        "sleeve_gt": sleeve_gt,
        "sleeve_inner_diam": float(sleeve_inner),
        "sleeve_outer_diam": float(sleeve_outer),
        "hairpin_gt": hairpin_gt,
        "window_gt": window_gt,
        "foam_gt": {"sleeve_end": foam0, "rebar_end": foamL},
        "bg_blob_gt": blob_gt,
        "density": density,
        "n_points": int(len(points)),
    }
    return {"points": points, "labels": labels, "metadata": metadata}


# 兼容旧入口名
BUILDERS = {t: (lambda ct=t, **kw: build_member(ct, **kw)) for t in [
    "rectangular_column",
    "l_shaped_column",
    "rectangular_beam",
    "t_beam",
    "rectangular_composite_beam",
    "t_composite_beam",
    "l_composite_beam",
]}


def make_point_cloud(
    comp_type: str,
    density: float = 0.012,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> dict:
    """统一入口（v2 与 build_member 等价）。"""
    return build_member(comp_type, density=density, rng=rng, **kwargs)
