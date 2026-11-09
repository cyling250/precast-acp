"""
地面背景与杂物生成 v2.1（非预制构件点，LABEL_NON_MEMBER）。

v2.1（用户观察修正 2026-10-17）：
- 地面外边界不规则（平滑角向起伏裁剪，非矩形）；
- 地面铺到构件接触线（footprint 仅构件自身 + 0~5mm，删除等宽护城河）；
- 构件-地面高度错开 ±4mm（贴平或轻微扫描误差，二者之一）；
- 底面贴地的杂物块不变。
"""

import numpy as np

from ._config import (
    LABEL_NON_MEMBER,
    GROUND_MARGIN_RANGE, GROUND_DENSITY_FACTOR, GROUND_ROUGHNESS_RANGE,
    GROUND_FOOTPRINT_MARGIN, GROUND_HEIGHT_OFFSET, GROUND_EDGE_IRREG,
    BG_BLOB_COUNT, BG_BLOB_SIZE, BG_BLOB_DIST,
)
from ._geometry import sample_ground_plane, sample_box_surface


def generate_ground(
    member_bbox: tuple,
    density: float,
    rng: np.random.Generator,
    member_footprint_wkt: str = None,
) -> np.ndarray:
    """
    地面点（z≈0±高度错开，加起伏 σ）。

    Parameters
    ----------
    member_bbox : (x_min, x_max, y_min, y_max) 构件包围范围。
    member_footprint_wkt : 保留兼容（未用——遮挡用 bbox footprint）。
    """
    from shapely.geometry import box as shapely_box

    x_min, x_max, y_min, y_max = member_bbox
    margin = rng.uniform(*GROUND_MARGIN_RANGE)
    dens = density * rng.uniform(*GROUND_DENSITY_FACTOR)
    rough = rng.uniform(*GROUND_ROUGHNESS_RANGE)
    z_off = rng.uniform(*GROUND_HEIGHT_OFFSET)
    irregular = rng.uniform(*GROUND_EDGE_IRREG)

    # 构件 footprint：仅构件自身（贴线遮挡——其正下方地面被构件挡住，无等宽留白）
    fp_margin = rng.uniform(*GROUND_FOOTPRINT_MARGIN)
    footprint = shapely_box(x_min - fp_margin, y_min - fp_margin,
                            x_max + fp_margin, y_max + fp_margin)

    pts = sample_ground_plane(
        y_min=y_min - margin, y_max=y_max + margin,
        x_min=x_min - margin, x_max=x_max + margin,
        density=dens, roughness=rough, rng=rng,
        exclude_polygon=footprint, exclude_margin=0.0,
        irregular_edge_mm=irregular, z_offset=z_off,
    )
    return pts


def generate_background_blobs(
    member_bbox: tuple,
    density: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    """
    构件旁杂物块（长方体表面，底面贴地不采）。

    Returns
    -------
    points, blob_gt
    """
    x_min, x_max, y_min, y_max = member_bbox
    n_blobs = rng.integers(BG_BLOB_COUNT[0], BG_BLOB_COUNT[1] + 1)
    pts_list, gt_list = [], []

    for _ in range(n_blobs):
        sx = rng.uniform(*BG_BLOB_SIZE)
        sy = rng.uniform(*BG_BLOB_SIZE)
        sz = rng.uniform(*BG_BLOB_SIZE)
        dist = rng.uniform(*BG_BLOB_DIST)
        side = rng.choice([-1, 1])
        cy = (y_min if side < 0 else y_max) + side * dist
        cx = rng.uniform(x_min, x_max)
        center = np.array([cx, cy, sz / 2])
        pts = sample_box_surface(center, np.array([sx, sy, sz]),
                                 density * 0.6, rng)
        pts_list.append(pts)
        gt_list.append({
            "center": [float(cx), float(cy), float(sz / 2)],
            "size": [float(sx), float(sy), float(sz)],
        })

    if not pts_list:
        return np.empty((0, 3)), []
    return np.vstack(pts_list), gt_list
