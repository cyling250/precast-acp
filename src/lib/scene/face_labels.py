"""
构件面实例标签派生（T-037 M0，docs/t037_scene_face_pipeline.md §4.4/§5.3）。

单件生成器的点不带"出自哪个面"的出处（点云 vstack 后已混合），本模块在
【单件生成后、场景位姿变换前】按名义截面几何就近归属：

  面 = 该构件的全部外表面片——端面 ×2（x=0 套筒端 / x=L 出筋端）+
  截面棱柱边各一面（沿 X 拉伸，竖直侧面与水平顶/底面一体按边计）。
  就近 = 到各面片（有限矩形片）的欧氏距离 argmin。翘曲（σ12-40）与
  端面泡沫（≤70mm）使棱边邻带点的归属存在固有歧义，argmin 即 GT 口径；
  面平面参数（法向/偏移）取名义几何精确值，与 section_polygon_yz 同一
  参照系，场景位姿为刚性变换直接左乘（P-044 教训：GT 与点云同场）。

face type（判型头训练标签，§5.3）：0 身体面 / 1 出筋面(x=L 端) /
2 套筒面(x=0 端) / 3 箍筋顶面（叠合梁截面最高水平边，发夹附着面）。
墙板洞口边筋标箍筋类但附着在大面（身体面）上——保持 type=0，判型
语义"箍筋顶面"专指发夹梳齿面。
"""

import numpy as np

from ..pointcloud_generator._config import LABEL_CONCRETE
from ..pointcloud_generator._concrete import (
    _ensure_ccw, _polygon_outward_normals,
)

# 面类型常量（face_table 第 3 列）
FACE_BODY, FACE_REBAR_END, FACE_SLEEVE_END, FACE_STIRRUP_TOP = 0, 1, 2, 3
FACE_TYPE_NAMES = ["body", "rebar_end", "sleeve_end", "stirrup_top"]


def face_type_name(t: int) -> str:
    return FACE_TYPE_NAMES[t] if 0 <= t < len(FACE_TYPE_NAMES) else str(t)


def _seg_dist2(pts2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """点到 2D 线段的距离（向量化）。"""
    ab = b - a
    l2 = float(ab @ ab)
    if l2 < 1e-12:
        return np.linalg.norm(pts2 - a, axis=1)
    t = np.clip(((pts2 - a) @ ab) / l2, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(pts2 - proj, axis=1)


def _polygon_contains(poly, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """截面向多边形内含测试（shapely 2 向量化，回退 matplotlib.path）。"""
    try:
        from shapely import contains_xy
        return contains_xy(poly, ys, zs)
    except Exception:
        from matplotlib.path import Path
        verts = np.asarray(poly.exterior.coords)[:, :2]
        path = Path(verts)
        return path.contains_points(np.column_stack([ys, zs]))


def assign_member_faces(
    points: np.ndarray,
    labels: np.ndarray,
    meta: dict,
) -> tuple[np.ndarray, list[dict]]:
    """
    逐点面实例归属 + 构件局部面表。

    Parameters
    ----------
    points : (N,3) 构件局部系（翘曲后）。
    labels : (N,) 单件标签（混凝土点参与归属，其余 face_id=-1）。
    meta   : build_member.metadata（section_polygon_yz / length_mm /
             type）。

    Returns
    -------
    face_ids : (N,) int16，-1=非面点（筋/箍筋；地面/杂物已被上层剥掉）。
    faces    : list[dict]，每面 {face_id, type, n_local(3), d_local,
               anchor_local(3), n_pts}——法向/偏移为名义精确值，锚点=
               面片中点（侧面）/截面形心（端面）。
    """
    sec = _ensure_ccw(np.asarray(meta["section_polygon_yz"], np.float64))
    normals2 = _polygon_outward_normals(sec)          # (E,2) yz 外法向
    n_edges = len(sec)
    L = float(meta["length_mm"])
    is_composite = "composite" in str(meta.get("type", ""))
    top_z = float(sec[:, 1].max())

    faces: list[dict] = []
    # 面 0：套筒端（x=0，外法向 -x）；面 1：出筋端（x=L，外法向 +x）
    cy, cz = float(sec[:, 0].mean()), float(sec[:, 1].mean())
    faces.append({"face_id": 0, "type": FACE_SLEEVE_END,
                  "n_local": np.array([-1.0, 0.0, 0.0]), "d_local": 0.0,
                  "anchor_local": np.array([0.0, cy, cz])})
    faces.append({"face_id": 1, "type": FACE_REBAR_END,
                  "n_local": np.array([1.0, 0.0, 0.0]), "d_local": L,
                  "anchor_local": np.array([L, cy, cz])})
    # 面 2+i：棱柱边 i（CCW 序）扫掠面
    for i in range(n_edges):
        j = (i + 1) % n_edges
        ny, nz = float(normals2[i, 0]), float(normals2[i, 1])
        e_mid = (sec[i] + sec[j]) / 2.0
        ftype = FACE_BODY
        if is_composite and abs(float(sec[i, 1]) - top_z) < 1e-6 \
                and abs(float(sec[j, 1]) - top_z) < 1e-6 and nz > 0.1:
            ftype = FACE_STIRRUP_TOP  # 截面最高水平边（发夹附着面）
        faces.append({
            "face_id": 2 + i, "type": ftype,
            "n_local": np.array([0.0, ny, nz]),
            "d_local": ny * float(e_mid[0]) + nz * float(e_mid[1]),
            "anchor_local": np.array([L / 2.0, float(e_mid[0]), float(e_mid[1])]),
        })

    face_ids = np.full(len(points), -1, np.int16)
    con = np.where(labels == LABEL_CONCRETE)[0]
    if len(con) == 0:
        for f in faces:
            f["n_pts"] = 0
        return face_ids, faces

    P = points[con].astype(np.float64)
    x = P[:, 0]
    yz = P[:, 1:3]

    # 各棱柱边的 2D 距离（E, n_con）
    d_edges = np.stack([
        _seg_dist2(yz, sec[i], sec[(i + 1) % n_edges]) for i in range(n_edges)
    ])
    # 端面横向间隙：投影在多边形内=0，在外=到边界距离（噪声点兜底）
    from shapely.geometry import Polygon as ShapelyPolygon
    poly = ShapelyPolygon(sec)
    inside = _polygon_contains(poly, yz[:, 0], yz[:, 1])
    gap = np.where(inside, 0.0, d_edges.min(axis=0))

    d_end0 = np.hypot(x - 0.0, gap)
    d_endL = np.hypot(x - L, gap)
    dx_out = np.maximum(0.0, -x) + np.maximum(0.0, x - L)
    d_side = np.sqrt(d_edges ** 2 + dx_out[None, :] ** 2)   # (E, n_con)

    all_d = np.vstack([d_end0[None, :], d_endL[None, :], d_side])  # (2+E, n_con)
    winner = all_d.argmin(axis=0)
    face_ids[con] = winner.astype(np.int16)

    for f in faces:
        f["n_pts"] = int((face_ids == f["face_id"]).sum())
    return face_ids, faces
