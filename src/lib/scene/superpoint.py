"""
超点过分割（T-037 M1，docs/t037_scene_face_pipeline.md §5.1/决策 D8）——
纯几何、免训练。

规格：8mm 体素（体素心+法向复用 `lib/e2e/face_sem.voxel_normals`，
kNN8 PCA）→ 26 邻域邻接 → 区域生长约束（相邻体素法向夹角 <12° 且
平面度相近）→ 连通分量即超点；最小超点 ≥4 体素（更小并入边界接触最
多的邻接超点——"最近邻"，无法并入者标噪声 -1）。

**默认参数 = D14 定稿（kNN8 / 12° / 0.25 / 4 体素，方案 §5.1/§11）**。
警示：D14 曾只存在于 M1 快验的命令行参数里、模块默认值停留在旧基线
kNN16/20°（渗漏：最大超点吞 53% 点）——2026-10-26 M2 建缓存前才改回
默认值；此后任何调用方不传参即得到定稿行为。

D8 纪律：平面参数永远出自原始点 5mm 加权 RANSAC（M4 refine 阶段），
本模块绝不输出平面——过分割只负责把"面"切成单元。

输出：逐点/逐体素 superpoint id、超点邻接表（M2 图卷积输入，边权=
接触体素对数）、逐超点统计。缓存进 `out/exp1_train/cache/scene_cache/`（训练每 epoch
不重算，P-033 纪律）。
"""

import numpy as np

VOXEL_MM = 8.0        # D8：8mm（16mm 会把 20-80mm 缝碎成退化超点）
KNN = 8               # D14 定稿：kNN8（kNN16 棱角法向渐变带跨面渗漏）
ANG_THR_DEG = 12.0    # D14 定稿：法向夹角 12°（基线 20° 同渗漏）
PLAN_THR = 0.25       # 区域生长：平面度（w0/w2）相近容差
MIN_VOX = 4           # 最小超点体素数（更小并入邻接/标噪声）

# 26 郎域偏移
_OFFSETS = np.array([(dx, dy, dz)
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)],
                    np.int64)


def _void_view(keys: np.ndarray):
    """int64 (N,3) → 可 searchsorted 的 void 视图。"""
    return np.ascontiguousarray(keys).view(
        [("x", keys.dtype), ("y", keys.dtype), ("z", keys.dtype)]).ravel()


def voxelize(points: np.ndarray, voxel_mm: float = VOXEL_MM):
    """点 → 体素：返回 (voxel keys, 点→体素 inverse, 体素质心=点均值)。"""
    key = np.floor(points / voxel_mm).astype(np.int64)
    uk, inv, cnt = np.unique(key, axis=0, return_inverse=True,
                             return_counts=True)
    cents = np.zeros((len(uk), 3), np.float64)
    np.add.at(cents, inv, points)
    cents /= cnt[:, None]
    return uk, inv, cents


def _neighbor_edges(uk: np.ndarray) -> np.ndarray:
    """26 邻域体素对（每对一次）。"""
    ukv = _void_view(uk)
    n = len(uk)
    pairs = []
    for off in _OFFSETS:
        sk = _void_view(uk + off[None, :])
        pos = np.searchsorted(ukv, sk)
        pos = np.clip(pos, 0, n - 1)
        hit = ukv[pos] == sk
        j = np.arange(n)[hit]
        i = pos[hit]
        pairs.append(np.column_stack([np.minimum(i, j), np.maximum(i, j)]))
    return np.unique(np.vstack(pairs), axis=0)


def _voxel_normals(cents: np.ndarray, knn: int = KNN, workers: int = -1):
    """与 `face_sem.voxel_normals` 同式（kNN PCA：法向 + 平面度 w0/w2 +
    z+ 定向翻转），仅 cKDTree 查询并行化（workers，默认 -1 全核；多进程
    建缓存时传 cores//nproc 防超订）——场景级体素量大（百万级），单线程
    查询是分钟级瓶颈；face_sem 属共享代码不动，若彼处公式更新须同步此处。"""
    from scipy.spatial import cKDTree
    tree = cKDTree(cents)
    d_k, ix = tree.query(cents, k=knn, workers=workers)
    nb = cents[ix].astype(np.float64)
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", nb, nb)
    w, v = np.linalg.eigh(cov)
    nrm = v[:, :, 0]
    plan = (w[:, 0] / (w[:, 2] + 1e-9)).astype(np.float32)
    flip = (nrm[:, 2] < 0) | ((np.abs(nrm[:, 2]) < 0.05)
                              & ((nrm[:, 0] < 0) | ((np.abs(nrm[:, 0]) < 0.05)
                                                    & (nrm[:, 1] < 0))))
    nrm = np.where(flip[:, None], -nrm, nrm)
    return nrm.astype(np.float32), plan


def _restrained_components(edges: np.ndarray, dots: np.ndarray,
                           dplan: np.ndarray, nrm: np.ndarray,
                           plan: np.ndarray, n_vox: int,
                           ang_thr_deg: float, plan_thr: float) -> np.ndarray:
    """受限并查集组件：边按法向一致性（dots）降序合并，新边须与【两端
    组件的运行均值法向】都在夹角阈内、与两端组件均值平面度都在容差内。

    为什么不用"逐对阈 + 连通分量"（M1 口径）：翘曲让法向沿面渐变，
    11°<12° 的边链可以逐步爬过棱边（跨面蠕行）——把邻面的点整片吞进
    一个巨型超点（M2 失败分解：整面桶 77% 失败=被寄宿的面）。均值参照
    让蠕行在均值被拉动之前就被拒。先按逐对阈预滤（蠕行链的每条边必然
    逐对合法，预滤不改变可达集），再对通过者做受限合并。"""
    keep = (dots > np.cos(np.radians(ang_thr_deg))) & (dplan < plan_thr)
    e = edges[keep]
    dn = dots[keep]
    order = np.argsort(-dn, kind="stable")

    parent = np.arange(n_vox, dtype=np.int64)
    cnt = np.ones(n_vox, np.float64)
    nx = nrm[:, 0].astype(np.float64).copy()
    ny = nrm[:, 1].astype(np.float64).copy()
    nz = nrm[:, 2].astype(np.float64).copy()
    px = plan.astype(np.float64).copy()
    cos_thr = float(np.cos(np.radians(ang_thr_deg)))

    ea = e[:, 0].tolist()
    eb = e[:, 1].tolist()
    ordl = order.tolist()

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k in ordl:
        ra, rb = find(ea[k]), find(eb[k])
        if ra == rb:
            continue
        # 组件均值法向互检（蠕行在此被拦：新边可逐对合法，但两端组件的
        # 均值一旦分属棱边两侧，均值夹角立即超阈）
        if nx[ra] * nx[rb] + ny[ra] * ny[rb] + nz[ra] * nz[rb] < cos_thr:
            continue
        if abs(px[ra] - px[rb]) > plan_thr:
            continue
        if cnt[ra] < cnt[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        w = cnt[rb] / (cnt[ra] + cnt[rb])
        nx[ra] += w * (nx[rb] - nx[ra])
        ny[ra] += w * (ny[rb] - ny[ra])
        nz[ra] += w * (nz[rb] - nz[ra])
        _nl = np.sqrt(nx[ra] * nx[ra] + ny[ra] * ny[ra] + nz[ra] * nz[ra])
        nx[ra] /= _nl
        ny[ra] /= _nl
        nz[ra] /= _nl
        px[ra] += w * (px[rb] - px[ra])
        cnt[ra] += cnt[rb]

    comp = parent.copy()
    while True:                                   # 指针倍跳，向量化寻根
        nxt = parent[comp]
        if np.array_equal(nxt, comp):
            break
        comp = nxt
    return comp


def build(points: np.ndarray, voxel_mm: float = VOXEL_MM,
          ang_thr_deg: float = ANG_THR_DEG,
          plan_thr: float = PLAN_THR, min_vox: int = MIN_VOX,
          knn: int = KNN, workers: int = -1) -> dict:
    """
    场景点云 → 超点（组件阶段=受限并查集，见 _restrained_components）。

    Returns
    -------
    dict(sp_point (N,) int32（噪声=-1）, sp_vox (V,) int32, comp (V,),
    vox_cents/vox_nrm/vox_plan, adj (E,2) int32 超点邻接（不含噪声）,
    adj_w (E,) 接触体素对数, sp_npts (K,), sp_nvox (K,))
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    uk, inv, cents = voxelize(points, voxel_mm)
    nv = len(uk)
    nrm, plan = _voxel_normals(cents, knn=knn, workers=workers)

    edges = _neighbor_edges(uk)
    dots = np.abs(np.einsum("ij,ij->i", nrm[edges[:, 0]], nrm[edges[:, 1]]))
    dplan = np.abs(plan[edges[:, 0]] - plan[edges[:, 1]])

    comp = _restrained_components(edges, dots, dplan, nrm, plan, nv,
                                  ang_thr_deg, plan_thr)

    # 小超点（<min_vox 体素）并入边界接触最多的邻接超点；无邻接或循环
    # → 噪声。按边一次聚合分组（禁逐小超点全量扫边——K×E 二次方陷阱），
    # 映射+链解析防幽灵 id
    comp_sizes = np.bincount(comp, minlength=nv)
    ci, cj = comp[edges[:, 0]], comp[edges[:, 1]]
    ne = ci != cj
    # 每条跨界边的"较小端"（两端皆小取更小者，链解析兜底）
    side = np.where(ne, np.where(comp_sizes[ci] <= comp_sizes[cj], ci, cj),
                    -1)
    sel = (side >= 0) & (comp_sizes[side] < min_vox)
    e_ids = np.where(sel)[0]
    small_ids = np.where(comp_sizes < min_vox)[0]
    merge_map: dict = {int(c): -1 for c in small_ids}
    if len(e_ids):
        side_s = side[e_ids]
        order = np.argsort(side_s, kind="stable")
        sorted_ids = side_s[order]
        uniq_s, start = np.unique(sorted_ids, return_index=True)
        starts = np.append(start, len(sorted_ids))
        for k, c in enumerate(uniq_s):
            e = e_ids[order[starts[k]:starts[k + 1]]]
            other = np.where(ci[e] == c, cj[e], ci[e])
            big = other[comp_sizes[other] >= min_vox]
            if len(big) == 0:
                big = other                     # 周边全小：并给接触最多者
            uc, cc = np.unique(big, return_counts=True)
            merge_map[int(c)] = int(uc[np.argmax(cc)])

    resolved = {}
    for c in small_ids:
        seen = set()
        t = c
        while (t in merge_map) and (t not in seen):
            seen.add(t)
            t = merge_map[t]
            if t == -1:
                break
        # 落到合格超点 → 并入；噪声链或循环（互指）→ 噪声
        resolved[c] = t if (t != -1 and t not in merge_map) else -1
    # 应用合并：查表向量化（禁逐小超点全数组扫描——30 万小超点 ×
    # 80 万体素的 comp==c 扫描是小时级陷阱，M1 实测第三次踩）
    lookup = np.arange(nv, dtype=np.int64)
    for c, t in resolved.items():
        lookup[c] = t
    neg = comp < 0
    comp = np.where(neg, -1, lookup[np.clip(comp, 0, None)])

    # 重编号（噪声=-1）；comp 此时可能指向已改号的旧 id——以并查终态
    # 重新提取连通关系：comp 数组里非 -1 的值域即超点 id（合并目标本身
    # 未被改号，只有小编号被改写，指向安全）
    uniq = np.unique(comp[comp >= 0])
    remap = np.full(nv, -1, np.int64)
    remap[uniq] = np.arange(len(uniq))
    comp_r = np.where(comp >= 0, remap[np.clip(comp, 0, None)], -1)

    sp_point = comp_r[inv].astype(np.int32)
    sp_npts = np.bincount(sp_point[sp_point >= 0],
                          minlength=len(uniq)).astype(np.int64)
    sp_nvox = np.bincount(comp_r[comp_r >= 0],
                          minlength=len(uniq)).astype(np.int64)

    # 超点邻接（M2 图）：全邻接对按超点对聚合，边权=接触体素对数
    a = comp_r[edges[:, 0]]
    b = comp_r[edges[:, 1]]
    ok = (a >= 0) & (b >= 0) & (a != b)
    lo = np.minimum(a[ok], b[ok])
    hi = np.maximum(a[ok], b[ok])
    pair_key = lo * (len(uniq) + 1) + hi
    upk, adj_w = np.unique(pair_key, return_counts=True)
    adj = np.column_stack([upk // (len(uniq) + 1), upk % (len(uniq) + 1)])

    return {
        "sp_point": sp_point,
        "sp_vox": comp_r.astype(np.int32),
        "vox_cents": cents.astype(np.float32),
        "vox_nrm": nrm.astype(np.float32),
        "vox_plan": plan.astype(np.float32),
        "adj": adj.astype(np.int32),
        "adj_w": adj_w.astype(np.int64),
        "sp_npts": sp_npts,
        "sp_nvox": sp_nvox,
        "voxel_mm": voxel_mm,
    }
