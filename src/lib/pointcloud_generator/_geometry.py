"""
基础几何采样工具（纯数学，不做工程假设）。
v2 新增：部分圆弧圆柱采样（半圆柱钢筋）、平端面空心圆环采样（套筒环）、
        地面平面采样、杂物块采样。
"""

import numpy as np
from typing import Optional


def sample_rectangle_surface(
    width: float,
    height: float,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """矩形面内均匀采点 (N, 2)。"""
    area = width * height
    n_points = max(4, int(area * density))
    xs = rng.uniform(-width / 2, width / 2, size=n_points)
    ys = rng.uniform(-height / 2, height / 2, size=n_points)
    return np.column_stack([xs, ys])


def sample_cylinder_segment(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """两三维点之间圆柱侧面均匀采样（整周）。"""
    direction = end - start
    length_val = np.linalg.norm(direction)
    if length_val < 1e-8:
        return np.empty((0, 3))
    dir_norm = direction / length_val
    if abs(dir_norm[0]) < 0.9:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.array([0.0, 1.0, 0.0])
    u -= np.dot(u, dir_norm) * dir_norm
    u /= np.linalg.norm(u)
    v = np.cross(dir_norm, u)
    surface_area = 2 * np.pi * radius * length_val
    n_pts = max(4, int(surface_area * density))
    theta = rng.uniform(0, 2 * np.pi, size=n_pts)
    t = rng.uniform(0, length_val, size=n_pts)
    pts = (
        start[None, :]
        + t[:, None] * dir_norm[None, :]
        + radius * (np.cos(theta)[:, None] * u[None, :] + np.sin(theta)[:, None] * v[None, :])
    )
    return pts


def sample_cylinder_arc(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    density: float,
    rng: np.random.Generator,
    arc_center_dir: np.ndarray,
    arc_span_rad: float,
) -> np.ndarray:
    """
    部分圆弧的圆柱侧面采样（v2 钢筋半圆柱）。

    只采样以 arc_center_dir 为中心、±arc_span_rad/2 的可见弧段，
    模拟扫描只能扫到钢筋朝上/朝外一侧。

    Parameters
    ----------
    start, end : (3,) 轴线两端点。
    radius : 半径 mm。
    density : 表面密度 pts/mm²（整圆柱侧面积口径）。
    arc_center_dir : (3,) 可见弧中心方向（垂直于轴线的分量生效）。
    arc_span_rad : 弧总角度（rad），如 π=半圆柱。
    """
    direction = end - start
    length_val = np.linalg.norm(direction)
    if length_val < 1e-8:
        return np.empty((0, 3))
    dir_norm = direction / length_val
    # 弧中心方向投影到与轴线垂直的平面
    c = arc_center_dir - np.dot(arc_center_dir, dir_norm) * dir_norm
    if np.linalg.norm(c) < 1e-8:
        return sample_cylinder_segment(start, end, radius, density, rng)
    c /= np.linalg.norm(c)
    # 第二基向量
    u = np.cross(dir_norm, c)
    u /= np.linalg.norm(u)

    full_surface = 2 * np.pi * radius * length_val
    frac = np.clip(arc_span_rad / (2 * np.pi), 0.05, 1.0)
    n_pts = max(4, int(full_surface * density * frac))
    if n_pts < 4:
        return np.empty((0, 3))
    # 弧参数：中心 0，±span/2（对 θ=atan2(u 分量, c 分量) 即以 c 为 0 角）
    half = arc_span_rad / 2.0
    theta = rng.uniform(-half, half, size=n_pts)
    t = rng.uniform(0, length_val, size=n_pts)
    offset = radius * (np.cos(theta)[:, None] * c[None, :] + np.sin(theta)[:, None] * u[None, :])
    return start[None, :] + t[:, None] * dir_norm[None, :] + offset


def sample_annulus_ring(
    center: np.ndarray,
    normal: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    density: float,
    rng: np.random.Generator,
    thickness: float = 0.0,
) -> np.ndarray:
    """
    平面圆环（环形板）表面采样（v2 套筒端面环）。

    在 center 处、法向 normal 的平面上，采样 inner~outer 半径之间的环形板正面；
    thickness>0 时额外在环的外侧柱面上稀疏采样（侧面可见性低）。

    返回点位于 center 沿 normal 前方（即 z_local=0 平面）。
    """
    n = normal / np.linalg.norm(normal)
    if abs(n[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    e1 = a - np.dot(a, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    area = np.pi * (outer_radius**2 - inner_radius**2)
    n_pts = max(8, int(area * density))
    # 环内均匀采样（面积加权：r √均匀）
    r = np.sqrt(rng.uniform(inner_radius**2, outer_radius**2, size=n_pts))
    theta = rng.uniform(0, 2 * np.pi, size=n_pts)
    local = np.column_stack([r * np.cos(theta), r * np.sin(theta), np.zeros(n_pts)])
    pts = center[None, :] + local[:, 0:1] * e1[None, :] + local[:, 1:2] * e2[None, :]

    if thickness > 1.0:
        # 外圆柱面（稀疏 30%）
        n_side = max(4, int(2 * np.pi * outer_radius * thickness * density * 0.3))
        th = rng.uniform(0, 2 * np.pi, size=n_side)
        zz = rng.uniform(-thickness, 0.0, size=n_side)
        side_local = np.column_stack([
            outer_radius * np.cos(th), outer_radius * np.sin(th), zz,
        ])
        side_pts = (
            center[None, :]
            + side_local[:, 0:1] * e1[None, :]
            + side_local[:, 1:2] * e2[None, :]
            + side_local[:, 2:3] * n[None, :]
        )
        pts = np.vstack([pts, side_pts])
    return pts


def sample_disk(
    radius: float,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """XY 平面 z=0 圆盘均匀采样 (N, 3)。"""
    collected = []
    remaining = n_points
    while remaining > 0:
        xs = rng.uniform(-radius, radius, size=max(remaining * 2, 64))
        ys = rng.uniform(-radius, radius, size=max(remaining * 2, 64))
        radii = np.sqrt(xs**2 + ys**2)
        mask = radii <= radius
        valid = np.column_stack([xs[mask], ys[mask], np.zeros(mask.sum())])
        collected.append(valid)
        remaining -= len(valid)
    return np.concatenate(collected)[:n_points]


def sample_polygon_surface(
    polygon: np.ndarray,
    density: float,
    rng: np.random.Generator,
    max_points: int = 2_000_000,
) -> np.ndarray:
    """任意多边形内部均匀采样（拒绝采样），返回 (N, 2)。"""
    from shapely.geometry import Polygon as ShapelyPolygon

    poly = ShapelyPolygon(polygon)
    if not poly.is_valid or poly.area <= 0:
        return np.empty((0, 2))

    area = poly.area
    n_points = max(4, min(int(area * density), max_points))
    min_x, min_y, max_x, max_y = poly.bounds

    batch_size = min(max(n_points * 3, 1000), 400000)
    collected: list[np.ndarray] = []
    total_needed = n_points

    while total_needed > 0:
        xs = rng.uniform(min_x, max_x, size=batch_size)
        ys = rng.uniform(min_y, max_y, size=batch_size)
        # 向量化点在多边形内判断（shapely 2.x vectorized）
        from shapely import contains_xy
        mask = contains_xy(poly, xs, ys)
        valid = np.column_stack([xs[mask], ys[mask]])
        collected.append(valid[:total_needed])
        total_needed -= min(len(valid), total_needed)

    out = np.concatenate(collected)[:n_points]
    return out


def sample_polygon_prism_surface(
    polygon: np.ndarray,
    x_min: float,
    x_max: float,
    density: float,
    rng: np.random.Generator,
    bottom_visibility: bool = False,
) -> np.ndarray:
    """
    多边形棱柱表面采样（v2：沿 X 拉伸，截面在 YZ）。

    侧面按面积比例采样；两端面单独高密度（由调用方控制）。
    bottom_visibility=False 时底面（z 最小的水平面）不采样（贴地缺失）。

    返回 (N, 3)，x ∈ [x_min, x_max]。
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    poly = ShapelyPolygon(polygon)
    area = poly.area
    perimeter = poly.length
    length = x_max - x_min

    side_area = perimeter * length
    n_side = max(4, int(side_area * density))

    vertices = polygon
    if not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])
    edge_lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    total_perimeter = edge_lengths.sum()

    side_pts = []
    for i in range(len(vertices) - 1):
        n_edge = max(1, int(n_side * edge_lengths[i] / total_perimeter))
        t_vals = rng.uniform(0, 1, size=n_edge)
        sx = rng.uniform(x_min, x_max, size=n_edge)
        y = vertices[i, 0] + t_vals * (vertices[i + 1, 0] - vertices[i, 0])
        z = vertices[i, 1] + t_vals * (vertices[i + 1, 1] - vertices[i, 1])
        side_pts.append(np.column_stack([sx, y, z]))
    side_pts = np.concatenate(side_pts) if side_pts else np.empty((0, 3))

    # 顶面（多边形最高水平面）——只采多边形上边缘构成的朝上面
    # 通用做法：两端面与顶/底面由调用方处理；这里仅返回侧面 + 水平朝上面
    return side_pts


def sample_horizontal_face(
    polygon_2d: np.ndarray,
    x_min: float,
    x_max: float,
    z_level: float,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    水平朝上面采样：截面多边形中位于 z_level 的上表面。
    polygon_2d 为截面（yz），取其位于 z_level 的上边缘段（可能多段），
    向 y 方向展宽到多边形内部。
    简化实现：在多边形内部均匀采样，仅保留 |z - z_level| < 0.5 的点在面内投形。
    """
    raise NotImplementedError("由 _concrete.py 的专用逻辑替代")


def sample_ground_plane(
    y_min: float, y_max: float,
    x_min: float, x_max: float,
    density: float,
    roughness: float,
    rng: np.random.Generator,
    exclude_polygon: Optional[object] = None,
    exclude_margin: float = 0.0,
    irregular_edge_mm: float = 0.0,
    z_offset: float = 0.0,
) -> np.ndarray:
    """
    地面平面采样（z≈z_offset±roughness）。

    exclude_polygon: shapely Polygon（构件 footprint，y-x 平面），其范围内地面剔除
        （构件遮挡其正下方地面）。
    irregular_edge_mm: v2.1 外边界径向不规则幅度——采样域矩形边界按平滑角向
        起伏向内裁剪（两个谐波），消除规则矩形边缘。
    z_offset: v2.1 构件-地面高度错开（贴平或轻微扫描误差）。
    """
    from shapely import contains_xy
    from shapely.geometry import Polygon as ShapelyPolygon

    area = (y_max - y_min) * (x_max - x_min)
    if area <= 0:
        return np.empty((0, 3))
    n_points = max(16, int(area * density))
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    half_x, half_y = (x_max - x_min) / 2.0, (y_max - y_min) / 2.0
    ph1, ph2 = rng.uniform(0, 2 * np.pi, size=2)
    collected = []
    remaining = n_points
    while remaining > 0:
        batch = min(max(remaining * 2, 256), 400000)
        ys = rng.uniform(y_min, y_max, size=batch)
        xs = rng.uniform(x_min, x_max, size=batch)
        mask = np.ones(batch, dtype=bool)
        if exclude_polygon is not None and exclude_polygon.area > 0:
            pad = ShapelyPolygon(exclude_polygon).buffer(exclude_margin, resolution=4) \
                if exclude_margin > 0 else exclude_polygon
            mask &= ~contains_xy(pad, xs, ys)
        if irregular_edge_mm > 0.0:
            # 矩形边界沿极角的半径，减平滑角向起伏（0~1 × 幅度）→ 不规则边缘
            th = np.arctan2(ys - cy, xs - cx)
            ct, st = np.cos(th), np.sin(th)
            t_x = np.where(np.abs(ct) > 1e-12, half_x / np.maximum(np.abs(ct), 1e-12), 1e18)
            t_y = np.where(np.abs(st) > 1e-12, half_y / np.maximum(np.abs(st), 1e-12), 1e18)
            r_rect = np.minimum(t_x, t_y)
            u = 0.5 + 0.25 * np.sin(3 * th + ph1) + 0.25 * np.sin(7 * th + ph2)
            mask &= np.hypot(xs - cx, ys - cy) < r_rect - irregular_edge_mm * u
        valid = np.column_stack([xs[mask], ys[mask]])
        collected.append(valid[:remaining])
        remaining -= min(len(valid), remaining)
    pts2 = np.concatenate(collected)[:n_points]
    z = z_offset + rng.normal(0.0, roughness, size=len(pts2))
    return np.column_stack([pts2[:, 0], pts2[:, 1], z])


def sample_box_surface(
    center: np.ndarray,
    size: np.ndarray,
    density: float,
    rng: np.random.Generator,
    top_only_bias: float = 0.7,
) -> np.ndarray:
    """
    长方体杂物块表面采样（只采顶面+侧面，底面贴地不采）。
    top_only_bias: 顶面点占比下限。
    """
    cx, cy, cz = center
    sx, sy, sz = size
    # 顶面
    n_top = max(8, int(sx * sy * density))
    top = np.column_stack([
        rng.uniform(cx - sx / 2, cx + sx / 2, size=n_top),
        rng.uniform(cy - sy / 2, cy + sy / 2, size=n_top),
        np.full(n_top, cz + sz / 2),
    ])
    # 四侧面（稀疏）
    sides = []
    for axis, half, sign in [(0, sx / 2, 1), (0, sx / 2, -1), (1, sy / 2, 1), (1, sy / 2, -1)]:
        n_s = max(4, int(sz * (sx if axis == 1 else sy) * density * 0.5))
        a = np.full(n_s, (cx if axis == 0 else cy) + sign * half)
        b = rng.uniform(cy - sy / 2 if axis == 0 else cx - sx / 2,
                        cy + sy / 2 if axis == 0 else cx + sx / 2, size=n_s)
        zz = rng.uniform(cz, cz + sz, size=n_s)
        if axis == 0:
            sides.append(np.column_stack([a, b, zz]))
        else:
            sides.append(np.column_stack([b, a, zz]))
    pts = np.vstack([top] + sides)
    return pts


def sample_line_segment(
    start: np.ndarray,
    end: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """线段上均匀采点。"""
    t = rng.uniform(0, 1, size=n_points)
    return start + t[:, np.newaxis] * (end - start)
