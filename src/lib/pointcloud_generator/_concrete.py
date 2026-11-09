"""
混凝土表面点云生成 v2。

构件统一为沿 X 轴平躺的棱柱（截面在 YZ 平面、底面贴地 z≈0）。
表面 = 竖直侧面 + 水平朝上面 + 两端面；贴地底面不采样（结构化缺失）。
粗糙度沿各面局部法向注入（σ=0.4mm，实测对齐）。
"""

import numpy as np
from typing import Optional

from ._config import LABEL_CONCRETE, CONCRETE_ROUGHNESS, GROUND_Z
from ._geometry import sample_polygon_surface


def _polygon_outward_normals(polygon: np.ndarray) -> np.ndarray:
    """
    截面多边形每条边的外法向 (M, 2)（yz 平面）。
    要求顶点按逆时针排列（shapely 校正后）。
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    poly = ShapelyPolygon(polygon)
    if not poly.exterior.is_ccw:
        polygon = polygon[::-1]
    n = len(polygon)
    normals = np.zeros((n, 2))
    for i in range(n):
        j = (i + 1) % n
        edge = polygon[j] - polygon[i]  # (dy, dz)
        length = np.linalg.norm(edge)
        if length < 1e-9:
            normals[i] = (0, 0)
            continue
        # 逆时针多边形外法向 = 边向量逆时针转 90°
        normals[i] = np.array([edge[1], -edge[0]]) / length
    return normals


def _ensure_ccw(polygon: np.ndarray) -> np.ndarray:
    from shapely.geometry import Polygon as ShapelyPolygon
    if ShapelyPolygon(polygon).exterior.is_ccw:
        return polygon
    return polygon[::-1]


def sample_prism_side_faces(
    polygon_yz: np.ndarray,
    x_min: float,
    x_max: float,
    density: float,
    rng: np.random.Generator,
    roughness: Optional[float] = CONCRETE_ROUGHNESS,
    arc_window_deg: Optional[tuple] = None,
    end_full_zone_mm: float = 0.0,
) -> np.ndarray:
    """
    棱柱侧面 + 水平朝上面采样（不含端面、不含贴地底面）。

    - 竖直边 → 竖直侧面（沿 X 拉伸）
    - 水平边且内部在其下方（外法向 +z）→ 水平朝上面
    - 水平边且内部在其上方（外法向 -z）→ 底面/贴合面，跳过
    - v2.1 周向可见窗 arc_window_deg=(center_deg, halfwidth_deg)：面法向方位角
      （0°=正上）落在窗外时，仅端部 end_full_zone_mm 区保留（向中段概率衰减），
      模拟真实中段周向大面积缺失（36 箱仅 ~10 箱有占）。None=不过滤。
    """
    poly = _ensure_ccw(polygon_yz)
    normals = _polygon_outward_normals(poly)
    n = len(poly)
    length = x_max - x_min
    pts_list = []

    def _arc_keep(pts: np.ndarray, nrm2: np.ndarray) -> np.ndarray:
        if arc_window_deg is None or len(pts) == 0:
            return np.ones(len(pts), dtype=bool)
        c_deg, hw_deg = arc_window_deg
        phi = np.degrees(np.arctan2(nrm2[0], nrm2[1]))  # nrm=(ny,nz)，0°=正上
        d = abs((phi - c_deg + 180.0) % 360.0 - 180.0)
        if d <= hw_deg:
            return np.ones(len(pts), dtype=bool)
        # 窗外面：端部全可见区保留，向中段线性衰减到 0
        span = max(x_max - x_min, 1e-9)
        t = (pts[:, 0] - x_min) / span
        zone = min(end_full_zone_mm / span, 0.45)
        e = np.clip(np.minimum(t, 1.0 - t) / max(zone, 1e-9), 0.0, 1.0)
        return rng.random(len(pts)) < e

    for i in range(n):
        j = (i + 1) % n
        p1, p2 = poly[i], poly[j]
        nrm = normals[i]
        edge_len = np.linalg.norm(p2 - p1)
        if edge_len < 1e-9:
            continue
        is_horizontal = abs(p1[1] - p2[1]) < 1e-9
        if is_horizontal:
            if nrm[1] <= 0.1:
                continue  # 底面/朝下面：贴地缺失
            # 水平朝上面：y ∈ [min,max], x ∈ [x_min,x_max], z = p1[1]
            width = abs(p2[0] - p1[0])
            area = width * length
            n_pts = max(4, int(area * density))
            yy = rng.uniform(min(p1[0], p2[0]), max(p1[0], p2[0]), size=n_pts)
            xx = rng.uniform(x_min, x_max, size=n_pts)
            zz = np.full(n_pts, p1[1])
            pts = np.column_stack([xx, yy, zz])
            if roughness:
                zz_noise = rng.normal(0, roughness, size=n_pts)
                pts[:, 2] += zz_noise
            pts = pts[_arc_keep(pts, nrm)]
            if len(pts):
                pts_list.append(pts)
        else:
            # 竖直（或斜）侧面：沿边 + 沿 X
            area = edge_len * length
            n_pts = max(4, int(area * density))
            t = rng.uniform(0, 1, size=n_pts)
            yy = p1[0] + t * (p2[0] - p1[0])
            zz = p1[1] + t * (p2[1] - p1[1])
            xx = rng.uniform(x_min, x_max, size=n_pts)
            pts = np.column_stack([xx, yy, zz])
            if roughness:
                # 法向在 yz 平面 → 3D 法向
                n3 = np.array([0.0, nrm[0], nrm[1]])
                disp = rng.normal(0, roughness, size=n_pts)
                pts = pts + disp[:, None] * n3[None, :]
            pts = pts[_arc_keep(pts, nrm)]
            if len(pts):
                pts_list.append(pts)

    if not pts_list:
        return np.empty((0, 3))
    return np.vstack(pts_list)


def sample_prism_end_face(
    polygon_yz: np.ndarray,
    x: float,
    density: float,
    rng: np.random.Generator,
    normal_sign: int,
    roughness: Optional[float] = CONCRETE_ROUGHNESS,
) -> np.ndarray:
    """
    端面（x=常数）填充采样；normal_sign=+1 表示端面朝 +x，-1 朝 -x。
    density 由调用方按端面密度提升后的值传入。
    """
    pts2 = sample_polygon_surface(polygon_yz, density, rng)
    if len(pts2) == 0:
        return np.empty((0, 3))
    xx = np.full(len(pts2), x)
    pts = np.column_stack([xx, pts2[:, 0], pts2[:, 1]])
    if roughness:
        disp = rng.normal(0, roughness, size=len(pts))
        pts[:, 0] += normal_sign * disp
    return pts


def carve_circular_holes_2d(
    pts: np.ndarray,
    centers_yz: list,
    radii: list,
    plane_axis: int = 0,
) -> np.ndarray:
    """
    从端面点中挖掉圆孔（套筒内腔）。pts 为 (N,3)；plane_axis=0 → 端面在 yz 平面。
    返回布尔 mask：True = 保留。
    """
    keep = np.ones(len(pts), dtype=bool)
    ax = [1, 2] if plane_axis == 0 else [0, 1]
    for (cy, cz), r in zip(centers_yz, radii):
        d2 = (pts[:, ax[0]] - cy) ** 2 + (pts[:, ax[1]] - cz) ** 2
        keep &= d2 > r**2
    return keep


def carve_ellipse_gap(
    pts: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    rng: np.random.Generator,
    keep_mask: np.ndarray,
) -> np.ndarray:
    """
    端面无数据斑（椭圆区域置 False）。radii 为 3D 各轴半径（第三轴很小→薄盘）。
    直接修改 keep_mask 的副本并返回。
    """
    d = (pts - center) / np.maximum(radii, 1e-9)
    r2 = np.sum(d * d, axis=1)
    out = keep_mask.copy()
    out &= r2 > 1.0
    return out
