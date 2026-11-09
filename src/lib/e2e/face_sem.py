# -*- coding: utf-8 -*-
"""v6 阶段 A——连接面语义识别（2026-10-24 与用户定稿，docs/handover.md T-033）。

逐点语义分类（面带，类少而面实例数可变）→ 分类点集迭代 RANSAC 多平面
→ 每面 (法向, 偏移, 内点集)。

设计要点：
- 输入=整构件粗分辨率点云（体素心，面识别是区域级任务——齿/孔细节
  留给阶段 B 面视图）；仅质心平移，无重力对齐外的变换；
- 四类：0 其他 / 1 出筋面带 / 2 套筒面带 / 3 顶面带（箍筋顶面）。
  带标签从生成器 meta 零成本派生，且必须**特征可判定**（T-036 定稿：
  端面内缩 3mm、顶面整面无窗——特征不可见的标签边界会把 IoU 钉在
  天花板，详见 face_band_labels）。带内外露筋/套筒环是 RANSAC 离群
  点，由拟合剔；
- 网络=mini PointNet（工程红线：简单、鲁棒，不追架构）；
- 增广只做绕 z 旋转+丢体素（保尺度、保重力，方向无关性由标签随点
  旋转教出来——楼梯斜面机制上支持）；
- 迭代 RANSAC：每带串行拟合（面大且互不重叠，用户拍板串行可行），
  每轮取最大平面、剔内点、直到残点不足。

经典基线（本机 2026-10-24 复核，test 前 20 件）：出筋面 teeth 法
15/20·0.87°·13.1mm；套筒面经典法全失（0/20、2/20）；顶面 100%·0.61°·7.0mm。
"""

import numpy as np
import torch
import torch.nn as nn

N_CLASS = 4
# 4 类（2026-10-24 与用户对账定稿）：其他 / 纵筋面 / 套筒面 / 箍筋面。
# 面类 = 面片（端面/顶面平面附近 + 截面内缩的混凝土点）+ 其上附属结构
# 全段（外露纵筋+垫片 / 套筒环 / 发夹箍筋）——"面片带梳子一体"是检测
# 头视图的点集来源（窗退役为深度采样范围）。三面点数量级相近（用户：
# 无样本不平衡）；其他 = 构件身体 + 背景地面合并。
CLASS_NAMES = ("other", "rebar_face", "sleeve_face", "stirrup_face")
PLANE_HALF = 60.0  # 面片带轴向半宽（端面/顶面平面±60mm，含泡沫低凸）


# ── 标签派生（meta 零成本） ─────────────────────────────────────────────
def _section_inside(yz, section, inset=3.0):
    """截面多边形内缩 inset 后包含性（向量化；shapely contains_xy）。

    端面片的"截面内部"判据——排除贴边缘的侧面点（它们法向⊥轴，
    是 face_v2 带类混入 body 的根因）。inset 只需挡住数值上贴着多边形
    边界的侧面点（3mm）；曾经 30mm：内缩环上的面片点与内部面片点局部
    特征完全相同，属"特征不可见标签边界"，把可学 IoU 钉在 ~0.7
    （2026-10-25 诊断：类1"误检"62% 是内缩环真面片点），T-036 起 3mm。
    """
    from shapely.geometry import Polygon
    import shapely
    poly = Polygon(section).buffer(-inset)
    if poly.is_empty:
        poly = Polygon(section)
    return shapely.contains_xy(poly, yz[:, 0], yz[:, 1])


def face_band_labels(pts, lbls, meta):
    """逐点四类面标签。**pts 须为生成器世界系**（meta 的 end_x 同系）。

    标签必须是**特征可判定**的（P-039 教训的泛化，2026-10-25 定稿）：
    局部特征相同的点不许跨类。
    纵筋面(1)=出筋端面片(混凝土+截面内缩3mm+|x−end|<60)+外露纵筋全段
    +垫片(lbls==2 全部——布局 C 纵筋只生成外露段)；
    套筒面(2)=套筒端面片+套筒环(环标混凝土、贴端面)；
    箍筋面(3)=构件顶面片(z>top_z−60 整面)+发夹箍筋(lbls==3)——不再带
    发夹 x/y 窗：窗边界在点云特征上不可见（诊断：漏检 GT 体素深入窗内
    395mm、误检全在身体点），检测视图的窗口由 make_stirrup_crop/
    top_view 自行裁剪（裁剪窗⊂原标签窗，视图内容逐位不变）；
    其他(0)=构件身体+背景合并。泡沫凸起大部分落在平面带外,残余由
    RANSAC 距离阈剔除。
    """
    lab = np.zeros(len(pts), np.int8)
    section = np.asarray(meta.get("section_polygon_yz") or [], np.float64)
    inside = (_section_inside(pts[:, 1:], section) if len(section) >= 3
              else np.ones(len(pts), bool))
    conc = lbls == 1  # 混凝土（端面/顶面/侧面/环）
    # 纵筋面：端面片 + 纵筋全段（含垫片，均 lbls==2）
    ex = float(meta["rebar_end_x"])
    m = (conc & (np.abs(pts[:, 0] - ex) < PLANE_HALF) & inside) \
        | (lbls == 2)
    lab[m] = 1
    # 套筒面：端面片（环点贴端面、在 ±60 窗与截面内,自带包含）
    ex2 = float(meta["sleeve_end_x"])
    m2 = conc & (np.abs(pts[:, 0] - ex2) < PLANE_HALF) & inside
    lab[m2] = 2
    # 箍筋面：顶面整面 + 发夹（lbls==3）
    hps = meta.get("hairpin_gt") or []
    if hps:
        top_z = float(hps[0]["top_z"])
        m3 = (conc | (lbls == 3)) & (pts[:, 2] > top_z - PLANE_HALF)
        lab[m3] = 3
    return lab


# ── 体素化（粗分辨率输入） ──────────────────────────────────────────────
def voxel_normals(cents, knn=16):
    """体素心 kNN 邻域 PCA：法向 + 平面度 + 局部密度。

    平面度=w0/w2（面片点≈0、筋/棱/角大——face_v5 新增判别通道）；
    密度=到第 k 近邻的距离中位（端面 MVS 高密度 vs 侧面）。
    返回 (nrm, planarity, dens)。
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(cents)
    d_k, ix = tree.query(cents, k=knn)
    nb = cents[ix].astype(np.float64)
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", nb, nb)
    w, v = np.linalg.eigh(cov)
    nrm = v[:, :, 0]
    plan = (w[:, 0] / (w[:, 2] + 1e-9)).astype(np.float32)
    dens = d_k[:, -1].astype(np.float32)  # 第 k 近邻距离=局部密度度量
    flip = (nrm[:, 2] < 0) | ((np.abs(nrm[:, 2]) < 0.05)
                              & ((nrm[:, 0] < 0) | ((np.abs(nrm[:, 0]) < 0.05)
                                                    & (nrm[:, 1] < 0))))
    nrm = np.where(flip[:, None], -nrm, nrm)
    return nrm.astype(np.float32), plan, dens


def voxelize_with_labels(pts, lab, vox=16.0, max_pts=32768, rng=None):
    """体素心 + 体素内多数标签 + 法向/平面度/密度；超量随机下采样。

    返回 (P,3), (P,), (P,3), (P,2)——坐标、标签、法向、(平面度,密度)。
    v5：vox 32→16mm、max 16k→32k（面片判别的特征分辨率翻倍）。
    """
    key = np.floor(pts / vox).astype(np.int64)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    nv = len(uk)
    cents = np.zeros((nv, 3))
    np.add.at(cents, inv, pts)
    cents /= np.bincount(inv, minlength=nv)[:, None]
    counts = np.stack([np.bincount(inv[lab == c], minlength=nv)
                       for c in range(N_CLASS)], axis=1)
    vlab = counts.argmax(1).astype(np.int8)
    nrm, plan, dens = voxel_normals(cents)
    geo = np.stack([plan, dens], 1)
    if nv > max_pts:
        idx = (rng or np.random.default_rng(0)).choice(nv, max_pts,
                                                       replace=False)
        cents, vlab, nrm, geo = (cents[idx], vlab[idx], nrm[idx],
                                 geo[idx])
    return cents.astype(np.float32), vlab, nrm.astype(np.float32), geo


# ── 网络：mini PointNet ────────────────────────────────────────────────
class MiniPointNet(nn.Module):
    """逐点分类：shared MLP → 全局 max pool → concat → MLP。

    输入 (B,N,10)=(质心系坐标/1000, 法向, 轴向/径向归一位置, 平面度,
    密度/50)；绕 z 旋转增广下坐标与法向同步旋转、轴向特征重算。
    法向+轴向位置+平面度是面片判别主特征（v1/v3/v4 迭代结论）。
    """

    def __init__(self, n_cls=N_CLASS, ch=(64, 128, 256), in_ch=10):
        super().__init__()
        layers = []
        last = in_ch
        for c in ch:
            layers += [nn.Conv1d(last, c, 1), nn.BatchNorm1d(c),
                       nn.ReLU(inplace=True)]
            last = c
        self.local = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv1d(ch[-1] + in_ch, 128, 1), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, n_cls, 1))

    def forward(self, x):  # x: (B,N,in_ch)
        x = x.transpose(1, 2)
        f = self.local(x)
        g = f.max(2, keepdim=True)[0].expand(-1, -1, x.shape[2])
        return self.head(torch.cat([f, x], 1)).transpose(1, 2)


# ── 迭代 RANSAC 多平面 ──────────────────────────────────────────────────
def _ransac_plane(p, w=None, nrm=None, dist_thr=18.0, ang_thr=25.0,
                  iters=1200, min_inlier=150, seed=0):
    """向量化 RANSAC 单平面（法向一致性约束，v6d 治筋切平面干扰）。

    “平面构成规则体”先验的落地：内点 = |距离|<dist_thr 且（给法向时）
    |cos(点法向, 平面法向)|>cos(ang_thr)——外露筋点法向各向，穿筋切平面
    的法向一致性内点数远少于真面片，无法与面片竞争。
    w 给定时采样概率∝w、得分=内点加权和（概率引导，v6 主线）。
    返回 {n, d, n_in, inliers, mask} 或 None。
    """
    rng = np.random.default_rng(seed)
    p = p.astype(np.float64)
    npt = len(p)
    if w is None:
        w = np.ones(npt)
    w = np.asarray(w, np.float64)
    ws = np.clip(w, 0.05, None)  # 采样保底（one-hot/近零概率时不退化）
    ids = None
    if not np.allclose(ws, ws[0]):
        cum = np.cumsum(ws)
        ids = np.searchsorted(cum, rng.random((iters, 3)) * cum[-1])
        ids = np.clip(ids, 0, npt - 1)
    else:
        ids = rng.integers(0, npt, size=(iters, 3))
    a, b, c = p[ids[:, 0]], p[ids[:, 1]], p[ids[:, 2]]
    nn = np.cross(b - a, c - a)               # (iters,3)
    ln = np.linalg.norm(nn, axis=1, keepdims=True)
    ok_h = (ln[:, 0] > 1e-9) & np.isfinite(ln[:, 0])
    nn = nn[ok_h] / ln[ok_h]
    dd = np.einsum("ij,ij->i", (a[ok_h] + b[ok_h] + c[ok_h]) / 3.0, nn)
    # 距离矩阵 (N,H)：批量 |p·n − d|
    dist = np.abs(p @ nn.T - dd[None, :])     # (N,H)
    inl = dist < dist_thr
    if nrm is not None:
        nrm = np.asarray(nrm, np.float64)
        inl &= np.abs(nrm @ nn.T) > np.cos(np.radians(ang_thr))  # (N,H)
    score = w @ inl                            # (H,) 内点加权和
    best = int(np.argmax(score))
    if inl[:, best].sum() < min_inlier:
        return None
    m = inl[:, best]
    for _ in range(2):  # 等权 SVD 精化两轮
        if m.sum() < min_inlier:
            return None
        c0 = p[m].mean(0)
        _, _, V = np.linalg.svd(p[m] - c0, full_matrices=False)
        nm = V[2]
        m = np.abs((p - c0) @ nm) < dist_thr
        if nrm is not None:
            m &= np.abs(nrm @ nm) > np.cos(np.radians(ang_thr))
    if m.sum() < 3:  # 精化后内点耗尽（退化掩码）——放弃该平面
        return None
    c0 = p[m].mean(0)
    _, _, V = np.linalg.svd(p[m] - c0, full_matrices=False)
    return {"n": V[2], "d": float(c0 @ V[2]), "n_in": int(m.sum()),
            "inliers": p[m], "mask": m}


def iterative_planes(p, prob=None, nrm=None, dist_thr=18.0, n_plane=3,
                     min_inlier=150, iters=1200, prob_thr=0.5,
                     weighted=False, ang_thr=25.0):
    """面类点集 → 串行迭代平面（每轮最大平面、剔内点）——同类多面片
    的实例区分机制（墙 4 出筋面等；用户 2026-10-24 对账）。

    weighted=True：概率引导（采样+打分吃概率，全量点）；否则 prob 按
    prob_thr 过滤后均匀采样。nrm 给定时内点加法向一致性约束（规则体
    先验，防穿筋切平面）。返回 [{n, d, inliers, n_in}]。
    """
    if prob is not None and not weighted:
        sel = prob > prob_thr
        p, prob = p[sel], prob[sel]
        if nrm is not None:
            nrm = nrm[sel]
    work = p.astype(np.float64)
    w_work = (np.asarray(prob, np.float64)
              if (prob is not None and weighted) else None)
    nrm_work = np.asarray(nrm, np.float64) if nrm is not None else None
    out = []
    for k in range(n_plane):
        if len(work) < min_inlier:
            break
        pl = _ransac_plane(work, w=w_work, nrm=nrm_work,
                           dist_thr=dist_thr, ang_thr=ang_thr,
                           iters=iters, min_inlier=min_inlier, seed=k)
        if pl is None:
            break
        out.append(pl)
        keep = ~pl["mask"]
        work = work[keep]
        if w_work is not None:
            w_work = w_work[keep]
        if nrm_work is not None:
            nrm_work = nrm_work[keep]
    return out


@torch.no_grad()
def predict_class_masks(model, pts, device, vox=32.0, max_pts=16384,
                        thr=0.5):
    """整构件点云 → 逐类掩码 (N, N_CLASS) bool（体素心预测 kNN 传播）。

    v6c 干净视图的部署侧：检测视图点集 = 对应面类掩码（面片+附属结构
    一体，窗退役为深度采样范围）。
    """
    from scipy.spatial import cKDTree
    cents, _, prob = predict_bands(model, pts, device, vox, max_pts)
    tree = cKDTree(cents)
    _, ix = tree.query(pts, k=1, workers=-1)
    return prob[ix] > thr


def refine_face_planes(pts_raw, mask, cents, nrm_cents, prob_cents, cls,
                       vox=5.0, n_plane=3, min_inlier=800, dist_thr=15.0,
                       ang_thr=25.0, iters=400):
    """面参数精化：面类原始点（掩码传播）下采样 vox → 法向从体素心
    kNN 传播 → 法向约束 RANSAC（v6d：32mm 体素心上做 RANSAC 面片心仅
    ~200 个统计太弱，出筋面上限仅 45% 的根因修复）。

    返回平面列表（多面实例）。min_inlier 按 5mm 体素量级放大。
    """
    from scipy.spatial import cKDTree
    p = pts_raw[mask]
    if len(p) < min_inlier:
        return []
    key = np.floor(p / vox).astype(np.int64)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    pv = np.zeros((len(uk), 3))
    np.add.at(pv, inv, p)
    pv /= np.bincount(inv, minlength=len(uk))[:, None]
    tree = cKDTree(cents)
    _, ix = tree.query(pv, k=1, workers=-1)
    nrm_pv = nrm_cents[ix]
    w = prob_cents[ix, cls]
    return iterative_planes(pv, prob=w, nrm=nrm_pv, weighted=True,
                            dist_thr=dist_thr, n_plane=n_plane,
                            min_inlier=min_inlier, iters=iters,
                            ang_thr=ang_thr)


def axial_features(cents):
    """轴向/径向归一位置特征（零学习成本的全局几何注入，v6d）。

    轴=质心系 PCA 第一主轴（水平构件=柱轴，域无关）；t_norm∈[0,1]
    沿轴位置、r_norm∈[0,1] 径向位置——端面片=「t_norm 近端 + 法向∥轴 +
    r_norm 内缩」变为逐点局部可判（face_v3 IoU 0.34 的全局上下文瓶颈）。
    返回 (P,2)。
    """
    c = cents - cents.mean(0)
    _, _, V = np.linalg.svd(c[:: max(len(c) // 2000, 1)], full_matrices=False)
    ax = V[0]
    t = c @ ax
    t0, t1 = float(t.min()), float(t.max())
    t_norm = (t - t0) / max(t1 - t0, 1e-6)
    r = np.linalg.norm(c - t[:, None] * ax[None, :], axis=1)
    r_norm = r / max(float(r.max()), 1e-6)
    return np.stack([t_norm, r_norm], 1).astype(np.float32)


@torch.no_grad()
def predict_bands(model, pts, device, vox=16.0, max_pts=32768, rng=None):
    """整构件点云 → 体素心+法向+轴向/平面度/密度特征 → 四类概率。

    返回 (P,3), (P,3), (P,5)——质心系坐标、法向、概率（网络输入 10 通道:
    xyz/1000 + 法向 + t_norm/r_norm + 平面度 + 密度/50）。
    """
    model.eval()
    cents, _, nrm, geo = voxelize_with_labels(
        pts, np.zeros(len(pts), np.int8), vox, max_pts, rng)
    c0 = cents.mean(0)
    cc = cents - c0
    ax_f = axial_features(cc)
    x = np.concatenate([cc / 1000.0, nrm, ax_f, geo[:, :1],
                        geo[:, 1:2] / 50.0], axis=1)
    logit = model(torch.from_numpy(x)[None].float().to(device))[0]
    prob = torch.softmax(logit, 1).cpu().numpy()
    return cc, nrm, prob


def faces_from_bands(cents, prob, nrm=None, weighted=False):
    """分类概率 → 三面类各自的平面列表（多面实例，墙构型扩展用）。

    weighted=True 走概率引导 RANSAC（采样+打分吃概率）；False 先按
    0.5 过滤再均匀采样。nrm=体素心法向，触发内点法向一致性约束
    （规则体先验，防穿筋切平面）。返回 dict{rebar/sleeve/top:
    [plane, ...]}——同类多面由串行 RANSAC 天然分实例（用户 2026-10-24
    对账：聚类的目的就是把同类多个面片区分开），柱/梁构型取首面。
    """
    out = {}
    for key, cls in (("rebar", 1), ("sleeve", 2), ("top", 3)):
        w = prob[:, cls]
        out[key] = iterative_planes(cents, prob=w, nrm=nrm,
                                    weighted=weighted)
    return out
