"""
场景可见性模拟（T-037 M0）：虚拟视点 z-buffer 手持扫描缺失机理。

规格（docs/t037_scene_face_pipeline.md §3）：手持扫描仪位姿 K=6-10 个，
沿场景上方弧/环布设（高度按场景尺度 1.5-3m 等比、俯角 30-60°）；逐位姿
把场景点投影到该位姿深度栅格（格距 5mm），保留每像素首击深度，点深度
> 首击 + 容差(30mm) 即该位姿不可见；任一位姿可见即保留。

正交投影：遮挡关系由视线方向主导，透视畸变不影响可见性判定（位姿距离
远大于场景尺度时二者一致）；单构件生成器既有的 MVS 级退化（贴地缺失/
周向弧窗/端面斑）已在单件内烘焙，本模块只叠加场景级遮挡。
"""

import numpy as np

# z-buffer 参数（§3 规格）
CELL_MM = 5.0       # 深度栅格格距
TOL_MM = 30.0       # 首击容差（噪声/粗糙口径）
POSE_K = (6, 10)    # 位姿数
ELEV_DEG = (30.0, 60.0)   # 俯角
DIST_FACTOR = (1.05, 1.5) # 水平距离 = 场景半径 × 系数
HEIGHT_LO_MM = 1500.0     # 高度下限 1.5m（场景半径 3m 基准）
HEIGHT_HI_MM = 3000.0     # 高度上限 3.0m——"高度按场景尺度 1.5-3m 等比"


def make_scan_poses(scene_bbox: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    沿场景上方弧/环布设虚拟视点位姿。

    Parameters
    ----------
    scene_bbox : (6,) [x_min, x_max, y_min, y_max, z_min, z_max]（mm）。

    Returns
    -------
    (K, 6) float64：每行 [pos(3), lookat(3)]（复现性烘焙进数据）。
    """
    x0, x1, y0, y1, z0, z1 = scene_bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    radius = 0.5 * max(x1 - x0, y1 - y0)
    center_h = np.array([cx, cy], np.float64)

    k = int(rng.integers(POSE_K[0], POSE_K[1] + 1))
    az = np.sort(rng.uniform(0, 2 * np.pi, k))          # 方位角（环）
    az += rng.uniform(-0.15, 0.15, k)                    # 轻微不均匀
    elev = np.radians(rng.uniform(*ELEV_DEG, size=k))    # 俯角
    dist = radius * rng.uniform(*DIST_FACTOR, size=k) + 500.0
    # 高度按场景尺度等比（1.5-3m @ 半径 3m 基准；全量单位 mm），弧/环上方
    h_scale = max(1.0, radius / 3000.0)
    height = np.clip(dist * np.tan(elev),
                     HEIGHT_LO_MM * h_scale, HEIGHT_HI_MM * h_scale)
    look_z = z0 + 0.35 * (z1 - z0)                       # 看向场景中低处

    poses = np.zeros((k, 6), np.float64)
    for i in range(k):
        px = cx + dist[i] * np.cos(az[i])
        py = cy + dist[i] * np.sin(az[i])
        poses[i, :3] = [px, py, height[i]]
        # lookat 取场景中心附近抖动（视轴不完全过心，弧上自然散布）
        jx, jy = rng.uniform(-0.1, 0.1, 2) * radius
        poses[i, 3:] = [cx + jx, cy + jy, look_z]
    return poses


def compute_visibility(
    points: np.ndarray,
    poses: np.ndarray,
    cell_mm: float = CELL_MM,
    tol_mm: float = TOL_MM,
) -> np.ndarray:
    """
    z-buffer 可见性掩码：任一位姿可见即 True。

    Parameters
    ----------
    points : (N,3) 场景系 mm（构件+垫块+地面全体）。
    poses  : (K,6) [pos(3), lookat(3)]。

    Returns
    -------
    (N,) bool
    """
    pts = np.asarray(points, np.float64)
    n = len(pts)
    visible = np.zeros(n, dtype=bool)
    world_up = np.array([0.0, 0.0, 1.0])

    for pose in np.asarray(poses, np.float64):
        pos, look = pose[:3], pose[3:]
        fwd = look - pos
        nf = np.linalg.norm(fwd)
        if nf < 1e-9:
            continue
        fwd /= nf
        # 相机右/上基（视线近竖直时换轴防退化）
        right = np.cross(fwd, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)

        rel = pts - pos
        w = rel @ fwd                      # 深度（沿视线）
        u = rel @ right                    # 像素横
        v = rel @ up                       # 像素纵

        # 栅格范围：1-99.9 分位（外围散点不进栅格，单列可见）
        u_lo, u_hi = np.percentile(u, [0.1, 99.9])
        v_lo, v_hi = np.percentile(v, [0.1, 99.9])
        nu = max(1, int(np.ceil((u_hi - u_lo) / cell_mm)) + 1)
        nv = max(1, int(np.ceil((v_hi - v_lo) / cell_mm)) + 1)
        iu = np.floor((u - u_lo) / cell_mm).astype(np.int64)
        iv = np.floor((v - v_lo) / cell_mm).astype(np.int64)
        in_grid = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)

        zbuf = np.full(nu * nv, np.inf, np.float64)
        idx = np.where(in_grid)[0]
        flat = iu[idx] * nv + iv[idx]
        np.minimum.at(zbuf, flat, w[idx])
        zbuf = zbuf.reshape(nu, nv)

        vis_pose = np.ones(n, dtype=bool)
        iiu = np.clip(iu[in_grid], 0, nu - 1)
        iiv = np.clip(iv[in_grid], 0, nv - 1)
        vis_pose[in_grid] = w[in_grid] <= zbuf[iiu, iiv] + tol_mm
        visible |= vis_pose

    return visible
