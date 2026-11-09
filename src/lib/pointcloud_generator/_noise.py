"""
噪声增强 v2（降级）。

参数全面下调（见 _config.py NOISE_LEVELS）——实测真实云顶面残差 σ≈0.4mm。
应用顺序：多边形遮挡 → 均匀噪点 → 噪点簇 → 边缘噪点 → 随机丢弃 → 高斯抖动。
边缘噪点检测为向量化 kNN 标签混合判定（旧逐点循环在 30 万点下不可用）。
"""

import numpy as np

from scipy.spatial import cKDTree

from ._config import (
    LABEL_NON_MEMBER,
    DENSITY_VARIATION_N_ZONES,
    DENSITY_VARIATION_FACTOR_RANGE,
    CLUSTER_OUTLIER_N_CLUSTERS,
    CLUSTER_OUTLIER_SPREAD,
    EDGE_NOISE_SEARCH_RADIUS,
    EDGE_NOISE_JITTER_SIGMA,
    EDGE_NOISE_LABEL_FLIP_PROB,
)


def apply_noise(
    points: np.ndarray,
    labels: np.ndarray,
    noise_config: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """综合噪声增强（一次性叠加全部噪声类型，参数按构件独立采样）。"""
    occ_range = noise_config.get("occlusion_pct_range", (0, 0))
    occlusion_pct = rng.uniform(*occ_range)

    jit_range = noise_config.get("jitter_sigma_range", (0, 0))
    jitter_sigma = rng.uniform(*jit_range)

    keep_range = noise_config.get("keep_pct_range", (1, 1))
    keep_pct = rng.uniform(*keep_range)

    out_range = noise_config.get("outlier_pct_range", (0, 0))
    outlier_pct = rng.uniform(*out_range)

    clus_range = noise_config.get("cluster_outlier_pct_range", (0, 0))
    cluster_pct = rng.uniform(*clus_range)

    edge_range = noise_config.get("edge_noise_pct_range", (0, 0))
    edge_pct = rng.uniform(*edge_range)

    n_occ_poly = noise_config.get("n_occlusion_polygons", (1, 3))
    n_occ_sides = noise_config.get("n_occlusion_sides", (3, 7))

    pts, lbls = points.copy(), labels.copy()

    if occlusion_pct > 0 and len(pts) > 0:
        pts, lbls = _apply_polygon_occlusion(pts, lbls, occlusion_pct, n_occ_poly, n_occ_sides, rng)
    if outlier_pct > 0 and len(pts) > 0:
        pts, lbls = _apply_outliers(pts, lbls, outlier_pct, rng)
    if cluster_pct > 0 and len(pts) > 0:
        pts, lbls = _apply_clustered_outliers(pts, lbls, cluster_pct, rng)
    if edge_pct > 0 and len(pts) > 0:
        pts, lbls = _apply_edge_noise(pts, lbls, edge_pct, rng)
    if keep_pct < 1.0 and len(pts) > 0:
        pts, lbls = _apply_dropout(pts, lbls, keep_pct, rng)
    if jitter_sigma > 0 and len(pts) > 0:
        pts = _apply_jitter(pts, jitter_sigma, rng)

    return pts, lbls


def apply_density_variation(
    points: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """非均匀密度：Voronoi 分区各保留 0.75~1.25（模拟扫描距离差异）。"""
    N = len(points)
    if N < 2000:
        return points, labels

    n_zones = rng.integers(DENSITY_VARIATION_N_ZONES[0], DENSITY_VARIATION_N_ZONES[1] + 1)
    seed_indices = rng.choice(N, size=n_zones, replace=False)
    seeds = points[seed_indices]

    tree = cKDTree(seeds)
    zone_ids = tree.query(points)[1]

    factors = rng.uniform(*DENSITY_VARIATION_FACTOR_RANGE, size=n_zones)
    factors = factors / factors.mean()

    keep_mask = np.zeros(N, dtype=bool)
    for z in range(n_zones):
        zone_mask = zone_ids == z
        n_zone = zone_mask.sum()
        if n_zone == 0:
            continue
        n_keep = max(1, min(n_zone, int(n_zone * factors[z])))
        keep_idx = rng.choice(np.where(zone_mask)[0], size=n_keep, replace=False)
        keep_mask[keep_idx] = True

    return points[keep_mask], labels[keep_mask]


def _random_convex_polygon(center_xy, max_radius, n_sides, rng):
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_sides))
    radii = max_radius * rng.uniform(0.4, 1.0, n_sides)
    return np.column_stack([
        center_xy[0] + radii * np.cos(angles),
        center_xy[1] + radii * np.sin(angles),
    ])


def _apply_polygon_occlusion(points, labels, occlusion_pct, n_polygons_range, n_sides_range, rng):
    """
    多边形遮挡（模拟邻构件/支撑/杂物挡视线）。
    occlusion_pct 为目标遮挡比例，按比例决定遮挡次数。
    """
    if len(points) == 0:
        return points, labels

    from matplotlib.path import Path

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min

    n_polygons = int(np.clip(
        rng.integers(n_polygons_range[0], n_polygons_range[1] + 1)
        * (1 + occlusion_pct * 5), 1, 5))
    z_min, z_max = bbox_min[2], bbox_max[2]
    mask = np.ones(len(points), dtype=bool)

    for _ in range(n_polygons):
        n_sides = rng.integers(n_sides_range[0], n_sides_range[1] + 1)
        max_radius = np.min(bbox_size[[0, 1]]) * (0.15 + occlusion_pct * 1.5)
        center_xy = bbox_min[:2] + bbox_size[:2] * rng.uniform(0.1, 0.9, size=2)
        vertices = _random_convex_polygon(center_xy, max_radius, n_sides, rng)
        poly_path = Path(vertices, closed=True)
        in_xy = poly_path.contains_points(points[:, :2])
        poly_z_min = rng.uniform(z_min, z_min + bbox_size[2] * 0.5)
        poly_z_max = rng.uniform(poly_z_min + bbox_size[2] * 0.2, z_max)
        in_z = (points[:, 2] >= poly_z_min) & (points[:, 2] <= poly_z_max)
        mask &= ~(in_xy & in_z)

    return points[mask], labels[mask]


def _apply_clustered_outliers(points, labels, cluster_pct, rng):
    """结构性噪点簇：构件表面附近高斯簇（施工浮渣/粉尘聚集）。"""
    n_clusters = rng.integers(CLUSTER_OUTLIER_N_CLUSTERS[0], CLUSTER_OUTLIER_N_CLUSTERS[1] + 1)
    n_outliers = max(1, int(len(points) * cluster_pct))

    member_pts = points[labels != LABEL_NON_MEMBER]
    if len(member_pts) < n_clusters:
        member_pts = points
    idx = rng.choice(len(member_pts), size=n_clusters, replace=False)
    centers = member_pts[idx]

    ratios = rng.dirichlet(np.ones(n_clusters))
    counts = np.maximum(1, (ratios * n_outliers).astype(int))

    outliers_list = []
    for center, count in zip(centers, counts):
        spread = rng.uniform(*CLUSTER_OUTLIER_SPREAD)
        spread_xyz = spread * rng.uniform(0.6, 1.4, size=3)
        outliers_list.append(rng.normal(loc=center, scale=spread_xyz, size=(count, 3)))

    outliers = np.vstack(outliers_list)
    return (
        np.vstack([points, outliers]),
        np.concatenate([labels, np.full(len(outliers), LABEL_NON_MEMBER, dtype=np.int32)]),
    )


def _apply_edge_noise(points, labels, edge_pct, rng):
    """
    边缘噪点（混合像素）：类别边界处坐标抖动 + 少量标签翻转。
    向量化：kNN 邻域标签混合判定。
    """
    N = len(points)
    if N < 100:
        return points, labels

    tree = cKDTree(points)
    _, nbr_idx = tree.query(points, k=6)
    nbr_labels = labels[nbr_idx]           # (N, 6)
    n_unique = (nbr_labels == nbr_labels[:, :1]).sum(axis=1)
    edge_mask = n_unique < 4               # 邻域内混类 → 边缘点
    edge_indices = np.where(edge_mask & (labels != LABEL_NON_MEMBER))[0]
    if len(edge_indices) == 0:
        return points, labels

    # edge_pct 按总点数比例口径，实际腐蚀对象限定为边缘点
    n_corrupt = min(max(1, int(N * edge_pct)), len(edge_indices))
    corrupt_idx = rng.choice(edge_indices, size=n_corrupt, replace=False)

    pts = points.copy()
    lbls = labels.copy()
    jitter = rng.normal(0, EDGE_NOISE_JITTER_SIGMA, size=(n_corrupt, 3))
    pts[corrupt_idx] += jitter

    flip = rng.random(n_corrupt) < EDGE_NOISE_LABEL_FLIP_PROB
    for i in np.where(flip)[0]:
        gi = corrupt_idx[i]
        nbrs = nbr_labels[gi]
        others = nbrs[nbrs != labels[gi]]
        if len(others):
            lbls[gi] = rng.choice(others)
    return pts, lbls


def _apply_dropout(points, labels, keep_pct, rng):
    """随机丢弃。"""
    n = len(points)
    n_keep = max(1, int(n * keep_pct))
    indices = rng.choice(n, size=n_keep, replace=False)
    return points[indices], labels[indices]


def _apply_jitter(points, sigma, rng):
    """高斯抖动（各向独立）。"""
    return points + rng.normal(0, sigma, size=points.shape)


def _apply_outliers(points, labels, outlier_pct, rng):
    """均匀外来噪点。"""
    n_outliers = max(1, int(len(points) * outlier_pct))
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    outliers = rng.uniform(bbox_min, bbox_max, size=(n_outliers, 3))
    return (
        np.vstack([points, outliers]),
        np.concatenate([labels, np.full(n_outliers, LABEL_NON_MEMBER, dtype=np.int32)]),
    )
