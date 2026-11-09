"""
超点面网络（T-037 M2，docs/t037_scene_face_pipeline.md §5.2）——
逐超点特征（18 维，FEAT_DIM/维度布局见 FEAT_SLICES）→ 2 层加权图卷积
（邻接图，边权=接触体素对数 log1p；隐藏 64，ReLU）→ 双头：
前景 logit(2) + 关联嵌入(16)。参数 ~1.9 万 << 10 万；禁 transformer
（D2 红线）。

实例解码（§5.2 实例化）：前景概率阈值 → 【邻接 且 嵌入距离<τ】的边做
连通分量 → 邻接共面对合并（法向夹角 <3° 且面偏移 <5mm，仅邻接对）→
面实例；每实例平面 = 成员超点质心的点数加权 PCA（M4 才换原始点 5mm
RANSAC，D8 纪律）。

评估匹配（§7.1）：预测实例 ↔ GT 面：法向夹角 <5° 且 |偏移| <30mm 且
可见区覆盖 ≥30%（覆盖=预测实例承载的 GT 面可见点 / GT 面可见点，由
逐超点多数面可见点数聚合近似）。

损失（§5.2）：前景=类加权 CE（正类=构件面超点）；嵌入=判别损失——
同实例超点对拉近（intra 方差 + sqrt 项），邻接异实例对推离（hinge，
margin），推离只在邻域内算（避免构件两端平行面无谓互斥）。
"""

import numpy as np
import torch
import torch.nn as nn

# ── 特征布局（build_scene_cache 落盘与训练增广共用，改动须两处同步） ──
FEAT_DIM = 18
SL_CENT = slice(0, 3)     # 质心（场景盒 min/max 归一到 [0,1]）
SL_NPTS = slice(3, 4)     # log(点数+1)/8
SL_EXT = slice(4, 7)      # 外接盒三轴 extent：log(mm+1)/7
SL_PCA = slice(7, 9)      # 超点 PCA 特征值比（log）：w0/w2 平面度、w1/w2
SL_NRM = slice(9, 12)     # 平均法向（单位）
SL_NDISP = slice(12, 13)  # 法向离散度：体素法向与超点法向 mean(1-|cos|)
SL_DENS = slice(13, 14)   # log(点数/体素数)
SL_ZQ = slice(14, 16)     # 高度分位 p10/p90（场景高归一）
SL_DEG = slice(16, 17)    # log(邻接数+1)/5
SL_CONS = slice(17, 18)   # 邻接法向一致性均值 |cos|

FG_THR = 0.5        # 前景概率阈值（解码）
EMB_TAU = 0.6       # 实例合并：嵌入距离阈值（§5.2）
COPLA_DEG = 3.0     # 邻接共面合并：法向夹角
COPLA_MM = 5.0      # 邻接共面合并：面偏移
PUSH_MARGIN = 1.5   # 判别损失推离 margin（>EMB_TAU，留解码裕度）
MATCH_DEG = 5.0     # §7.1 匹配：法向夹角
MATCH_MM = 30.0     # §7.1 匹配：偏移
MATCH_COV = 0.30    # §7.1 匹配：可见覆盖


class SceneFaceNet(nn.Module):
    """2 层图卷积（加权均值聚合 + 自信息拼接）→ 多头：
    前景 / 面嵌入（回退路径用）/ 构件嵌入（构件聚类）/ 结构类型档
    （逐构件池化后分类——类型先验只给"RANSAC 边数上限"，尺寸永不入网）。
    构件/类型头 2026-10-27 增补；热启动旧权重 strict=False。"""

    def __init__(self, in_dim: int = FEAT_DIM, hidden: int = 64,
                 emb_dim: int = 16, n_types: int = 4):
        super().__init__()
        self.in_lin = nn.Linear(in_dim, hidden)
        self.l1 = nn.Linear(2 * hidden, hidden)
        self.l2 = nn.Linear(2 * hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.fg_head = nn.Linear(hidden, 2)
        self.emb_head = nn.Linear(hidden, emb_dim)
        self.memb_head = nn.Linear(hidden, emb_dim)
        self.type_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, n_types))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_w: torch.Tensor):
        """返回 (fg_logits, face_emb, memb_emb, trunk)。类型头在主干池化
        之后由调用方施加（池化分组：训练期=GT 构件、推理期=预测簇）。"""
        h0 = torch.relu(self.in_lin(x))
        h1 = torch.relu(self.norm1(self.l1(
            torch.cat([h0, _aggregate(h0, edge_index, edge_w)], dim=1))))
        h2 = torch.relu(self.norm2(self.l2(
            torch.cat([h1, _aggregate(h1, edge_index, edge_w)], dim=1))))
        return self.fg_head(h2), self.emb_head(h2), self.memb_head(h2), h2


TYPE_BUCKETS = [(4, 4), (5, 6), (7, 9), (10, 14)]
TYPE_NAMES = ["box4", "penta_hex", "hepta_nine", "deca_fourteen"]


def type_bucket_of(n_edges: int) -> int:
    """截面边数 → 类型档（头 C 标签/推理先验）。边数=face_table 该构件
    行数−2（2 端面 + N 侧面）。"""
    for k, (lo, hi) in enumerate(TYPE_BUCKETS):
        if lo <= n_edges <= hi:
            return k
    return len(TYPE_BUCKETS) - 1


def _aggregate(h: torch.Tensor, edge_index: torch.Tensor,
               edge_w: torch.Tensor) -> torch.Tensor:
    """dst ← Σ w·h[src]，按入边权和归一（无自环——自信息经拼接携带）。"""
    src, dst = edge_index[0], edge_index[1]
    agg = torch.zeros_like(h)
    agg.index_add_(0, dst, h[src] * edge_w.unsqueeze(1))
    deg = torch.zeros(h.size(0), 1, device=h.device,
                      dtype=h.dtype).index_add_(0, dst, edge_w.unsqueeze(1))
    return agg / deg.clamp_min(1e-6)


def foreground_loss(fg_logits: torch.Tensor, fg_label: torch.Tensor) -> torch.Tensor:
    """类加权 CE：正类（构件面超点）权重 = 负/正样本比，clip 20。"""
    n_pos = int((fg_label == 1).sum())
    n_neg = int((fg_label == 0).sum())
    w_pos = min(max(n_neg / max(n_pos, 1), 1.0), 20.0)
    weight = torch.tensor([1.0, w_pos], device=fg_logits.device)
    return nn.functional.cross_entropy(fg_logits, fg_label, weight=weight)


def embedding_loss(emb: torch.Tensor, inst_id: torch.Tensor,
                   edge_index: torch.Tensor, cent: torch.Tensor = None,
                   margin: float = PUSH_MARGIN,
                   max_push_edges: int = 200_000,
                   spatial_k: int = 4096, spatial_r: float = 150.0
                   ) -> torch.Tensor:
    """判别损失（仅前景超点）。拉近：逐实例方差均值 + sqrt 项；推离：
    ①邻接异实例对 hinge(margin − d)；②空间邻域异实例对（§5.2"推离只在
    邻域内算"——邻接只有边界超点受力，其余塌向零（嵌入 std 0.16 实锤），
    空间 kNN 子采样把推离铺到每个面片附近，spatial_r 内的异实例对互斥）。
    inst_id<0（非前景）不参与。"""
    pos = inst_id >= 0
    if pos.sum() < 2:
        z = emb.sum() * 0.0
        return z
    e_pos, i_pos = emb[pos], inst_id[pos]
    uniq, inv = torch.unique(i_pos, return_inverse=True)
    mu = torch.zeros(len(uniq), e_pos.size(1), device=emb.device)
    mu.index_add_(0, inv, e_pos)
    cnt = torch.bincount(inv, minlength=len(uniq)).clamp_min(1)
    mu = mu / cnt.unsqueeze(1)
    d2 = ((e_pos - mu[inv]) ** 2).sum(1)
    var_k = torch.zeros(len(uniq), device=emb.device).index_add_(0, inv, d2)
    var_k = var_k / cnt
    var_mean = var_k.mean()
    pull = var_mean + var_mean.sqrt()

    src, dst = edge_index[0], edge_index[1]
    both = pos[src] & pos[dst]
    diff = inst_id[src] != inst_id[dst]
    pe = edge_index[:, both & diff]
    push = emb.sum() * 0.0
    if pe.size(1):
        if pe.size(1) > max_push_edges:
            sel = torch.randperm(pe.size(1),
                                 device=emb.device)[:max_push_edges]
            pe = pe[:, sel]
        d = (emb[pe[0]] - emb[pe[1]]).norm(dim=1)
        push = push + torch.relu(margin - d).mean()
    if cent is not None and pos.sum() > 16:
        idx = torch.randperm(int(pos.sum()), device=emb.device)[:spatial_k]
        c_s = cent[pos][idx]
        e_s = e_pos[idx]
        i_s = inv[idx]
        dxy = torch.cdist(c_s, c_s)
        de = torch.cdist(e_s, e_s)
        mask = (dxy < spatial_r) & (dxy > 1e-6) & (i_s[:, None] != i_s[None, :])
        if mask.any():
            push = push + torch.relu(margin - de[mask]).mean()
    return pull + push


# ── 实例解码（numpy/scipy，推理与评估共用） ─────────────────────────


def decode_instances(fg_prob: np.ndarray, emb: np.ndarray,
                     cent_mm: np.ndarray, nrm: np.ndarray,
                     adj: np.ndarray, n_sp: int,
                     n_pts: np.ndarray = None,
                     fg_thr: float = FG_THR, tau: float = EMB_TAU,
                     copla_deg: float = COPLA_DEG,
                     copla_mm: float = COPLA_MM,
                     gap_mm: float = 30.0,
                     merge_deg: float = 12.0,
                     merge_off_mm: float = 30.0,
                     merge_tau: float = 1.2,
                     max_iter: int = 6) -> np.ndarray:
    """前景概率 → 面实例 id（非前景/噪声 = -1）。

    三段：
    ①种子：邻接 ∧ 嵌入距 < τ 的连通分量（τ 内但跨洞的非邻接碎片先各自
      成粒）；
    ②近邻合并（迭代到不动点）：实例对满足【嵌入距 < τ】∧【平面一致：
      法向夹角 < merge_deg 且互到对方平面的点面距 < merge_off_mm】∧
      【空间可桥：两实例的超点质心最近距 < gap_mm（邻接已在①并掉，
      这里桥的是前景小洞隔开的碎片；跨构件缝隙 ≥200mm 天然被拦）】
      → 并。翘曲面被 fg 洞打碎的片在此重聚（2026-10-27 解剖实锤：同面
      两片 n 差 0.01、d 差 8.6mm、嵌入距 0.103，仅因 fg 洞不邻接而未并）。
    ③共面近邻合并：邻接对（无嵌入要求）法向 < copla_deg 且偏移 <
      copla_mm——兜底嵌入未分开的贴面共面片。

    平面一致用互点面距而非 |d_i−d_j|：与 match_faces 同理，倾斜法向的
    d 差含质心杠杆，会误杀同面碎片。"""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    fg = fg_prob >= fg_thr
    inst = np.full(n_sp, -1, np.int32)
    if not fg.any():
        return inst

    a, b = adj[:, 0], adj[:, 1]
    m1 = fg[a] & fg[b]
    edges = []
    if m1.any():
        d = np.linalg.norm(emb[a[m1]] - emb[b[m1]], axis=1)
        sel = d < tau
        edges.append(np.column_stack([a[m1][sel], b[m1][sel]]))
    # ③ 邻接共面兜底
    dots = np.abs((nrm[a] * nrm[b]).sum(1))
    off = np.abs((nrm[a] * cent_mm[a]).sum(1) - (nrm[b] * cent_mm[b]).sum(1))
    m2 = fg[a] & fg[b] & (dots > np.cos(np.radians(copla_deg))) \
        & (off < copla_mm)
    if m2.any():
        edges.append(np.column_stack([a[m2], b[m2]]))
    if edges:
        e = np.unique(np.vstack(edges), axis=0)
        e = np.vstack([e, np.column_stack([np.flatnonzero(fg)] * 2)])
        g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                       shape=(n_sp, n_sp))
        _n, comp = connected_components(g, directed=False)
        inst = np.where(fg, comp, -1).astype(np.int32)
        uniq = np.unique(inst[inst >= 0])
        remap = np.full(n_sp, -1, np.int64)
        remap[uniq] = np.arange(len(uniq))
        inst = np.where(inst >= 0, remap[np.clip(inst, 0, None)],
                        -1).astype(np.int32)

    # ② 迭代近邻合并（嵌入 τ 内 + 平面一致 + 空间可桥 → 并，到不动点）
    fg_ids = np.flatnonzero(fg)
    w_pt = (np.ones(n_sp, np.float64) if n_pts is None
            else n_pts.astype(np.float64))
    for _it in range(max_iter):
        inst_n, inst_d, inst_c = instance_planes(inst, cent_mm, nrm, w_pt)
        ids = np.unique(inst[inst >= 0])
        if len(ids) < 2:
            break
        sel = inst >= 0
        inst_sel = inst[sel]
        emb_mean = np.zeros((len(ids), emb.shape[1]), np.float64)
        emb_mean[:, :] = np.nan
        for k, v in enumerate(ids):
            emb_mean[k] = emb[sel][inst_sel == v].mean(0)
        # 成对嵌入距：‖a‖²+‖b‖²−2a·b 形式（f32），禁止逐差张量
        # (K,K,16) f64 —— 8525 实例时 8.66GiB OOM 实锤（2026-10-27）
        em32 = emb_mean.astype(np.float32)
        sq = (em32 * em32).sum(1)
        de2 = sq[:, None] + sq[None, :] - 2.0 * (em32 @ em32.T)
        np.maximum(de2, 0.0, out=de2)
        de = np.sqrt(de2)
        # ②合并阶段用 merge_tau（松弛到失败面同面相邻嵌入距 p90=1.17，
        # 跨面相邻 p50=1.53——1.2 恰在间隔内）；种子 τ 不动防跨面黏连
        cand = (de < merge_tau) & np.triu(np.ones_like(de, bool), 1)
        if not cand.any():
            break
        # 角度/偏移门同样向量化预滤
        na = inst_n[ids].astype(np.float32)
        nb = na
        cn = np.float32(np.cos(np.radians(merge_deg)))
        dots_m = np.abs(na @ nb.T)
        ca = inst_c[ids].astype(np.float32)
        da, db = inst_d[ids].astype(np.float32), inst_d[ids].astype(np.float32)
        off_ab = np.abs(np.einsum("kd,ld->kl", na, ca) - da[:, None])
        off_ba = np.abs(np.einsum("kd,ld->kl", nb, ca) - db[None, :])
        cand &= (dots_m > cn) & (off_ab < merge_off_mm) \
            & (off_ba < merge_off_mm)
        pairs = np.argwhere(cand)
        if len(pairs) == 0:
            break
        parent = {int(v): int(v) for v in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merged = 0
        sp_of = {int(v): fg_ids[inst_sel == v] for v in ids}
        for ka, kb in pairs:
            ia, ib = int(ids[ka]), int(ids[kb])
            ra, rb = find(ia), find(ib)
            if ra == rb:
                continue
            # 空间可桥：跨实例超点质心最近距 < gap_mm（min 对距离天然对称，
            # 单向查询即全局最近对）
            ta = cKDTree(cent_mm[sp_of[ra]])
            ga, _ = ta.query(cent_mm[sp_of[rb]], k=1,
                             distance_upper_bound=gap_mm)
            if not np.isfinite(ga).any():
                continue
            parent[rb] = ra
            merged += 1
        if merged == 0:
            break
        # 压缩实例 id
        lut = np.full(n_sp, -1, np.int64)
        for v in ids:
            lut[v] = find(int(v))
        u2 = np.unique(lut[lut >= 0])
        r2 = {int(v): k for k, v in enumerate(u2)}
        inst = np.where(inst >= 0,
                        np.array([r2[int(v)]
                                  for v in lut[np.clip(inst, 0, None)]]),
                        -1).astype(np.int32)
    return inst


MEMB_PUSH_MARGIN = 2.5    # 构件推离 margin（>解码 τ 上限，留裕度）
MEMB_PULL_MARGIN = 0.4    # 构件拉拢目标（同构件邻接对距离压到此以下）


def member_embedding_loss(emb: torch.Tensor, gid: torch.Tensor,
                          edge_index: torch.Tensor, cent: torch.Tensor = None,
                          margin: float = MEMB_PUSH_MARGIN,
                          pull_margin: float = MEMB_PULL_MARGIN,
                          max_push_edges: int = 200_000,
                          spatial_k: int = 4096, spatial_r: float = 150.0,
                          hard_neg_frac: float = 0.5,
                          w_pull: float = 3.0
                          ) -> torch.Tensor:
    """构件嵌入判别损失（M3 段②，与 face 级 embedding_loss 分开——
    病因不同不能用同配方，2026-10-27 诊断实锤）。

    face 级塌缩是"推离不足"（std 0.16），方差 pull 够用；构件级是
    **同构件邻接对距离分布过宽**（p50=0.44/p90=1.71，τ=0.6 剪断链式
    连通 → R=0.080）+ 跨构件对只被推到 margin 压线（p50=1.55，解码 τ
    开不大）。方差项按构件平均被大构件稀释、且不直接对齐解码条件，
    故换成：
    - pull = 邻接同构件对 hinge relu(d − pull_margin)：直接优化解码
      条件"邻接边距离<τ"，链上每段压紧，连通分量自然覆盖全构件；
    - push = 邻接异构件对 hinge relu(margin − d)，margin 提到 2.5
      （>τ 上限 2.0）；空间 kNN 推离通道同 face 级（复用已验证配方）。

    2026-10-27 二次诊断（训练集 R=0.58 / P=0.9995 = pull 欠优化）：
    - **难负样本挖掘**：跨构件邻接对里 d 小的少数难样本占 push 梯度
      主体（p10=0.264 vs 同构件 p50=0.306 —— 分布重叠区），只保留
      d 最小的 hard_neg_frac 比例对，梯度集中到真正需要分开的样本；
    - **pull 加权 w_pull**：pull 覆盖百万级对而 push 仅数千，均等权重
      下 pull 单项梯度被淹没。
    gid<0（非前景）不参与。"""
    pos = gid >= 0
    if pos.sum() < 2:
        return emb.sum() * 0.0
    src, dst = edge_index[0], edge_index[1]
    both = pos[src] & pos[dst]
    same = gid[src] == gid[dst]
    pull = emb.sum() * 0.0
    pe_pull = edge_index[:, both & same]
    if pe_pull.size(1):
        if pe_pull.size(1) > max_push_edges:
            sel = torch.randperm(pe_pull.size(1),
                                 device=emb.device)[:max_push_edges]
            pe_pull = pe_pull[:, sel]
        d = (emb[pe_pull[0]] - emb[pe_pull[1]]).norm(dim=1)
        pull = torch.relu(d - pull_margin).mean() * w_pull
    push = emb.sum() * 0.0
    pe = edge_index[:, both & ~same]
    if pe.size(1):
        if pe.size(1) > max_push_edges:
            sel = torch.randperm(pe.size(1),
                                 device=emb.device)[:max_push_edges]
            pe = pe[:, sel]
        d = (emb[pe[0]] - emb[pe[1]]).norm(dim=1)
        if hard_neg_frac < 1.0 and d.numel() > 8:
            k = max(8, int(d.numel() * hard_neg_frac))
            d = torch.topk(-d, k).values.neg()      # 最小的 k 个 = 最难
        push = torch.relu(margin - d).mean()
    if cent is not None and pos.sum() > 16:
        e_pos, i_pos = emb[pos], gid[pos]
        idx = torch.randperm(int(pos.sum()), device=emb.device)[:spatial_k]
        c_s = cent[pos][idx]
        e_s = e_pos[idx]
        i_s = i_pos[idx]
        dxy = torch.cdist(c_s, c_s)
        de = torch.cdist(e_s, e_s)
        mask = (dxy < spatial_r) & (dxy > 1e-6) & (i_s[:, None] != i_s[None, :])
        if mask.any():
            push = push + torch.relu(margin - de[mask]).mean()
    return pull + push


def instance_planes(inst: np.ndarray, cent_mm: np.ndarray, nrm: np.ndarray,
                    n_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray]:
    """每实例平面 = 成员超点【法向的点数加权均值】（重归一）+ 加权质心
    在该法向下的偏移。不水质心 PCA——平面大面≈单个巨型超点，单点实例
    协方差恒为零矩阵，eigh 返回单位向量垃圾法向（2026-10-26 实锤，
    全部实例法向退化成 (1,0,0)）；超点自带体素级法向，聚合即真值。
    法向符号为体素规范朝向（z+/x+/y+），匹配端用无向平面距消符号。
    返回 (n (K_max,3), d (K_max,), c (K_max,3) 加权质心)；未用槽位 0。"""
    k_max = int(inst.max()) + 1 if (inst >= 0).any() else 0
    normals = np.zeros((max(k_max, 1), 3), np.float64)
    ds = np.zeros(max(k_max, 1), np.float64)
    cents = np.zeros((max(k_max, 1), 3), np.float64)
    if k_max == 0:
        return normals, ds, cents
    pos = inst >= 0
    inst_p = inst[pos]
    w = n_pts[pos].astype(np.float64)
    cent_p = cent_mm[pos].astype(np.float64)
    nrm_p = nrm[pos].astype(np.float64)
    sw = np.bincount(inst_p, weights=w, minlength=k_max)
    means = np.zeros((k_max, 3), np.float64)
    n_sum = np.zeros((k_max, 3), np.float64)
    for a in range(3):
        means[:, a] = np.bincount(inst_p, weights=w * cent_p[:, a],
                                  minlength=k_max)
        n_sum[:, a] = np.bincount(inst_p, weights=w * nrm_p[:, a],
                                  minlength=k_max)
    ok = sw > 1e-9
    means[ok] /= sw[ok, None]
    n_fit = n_sum[ok] / sw[ok, None]
    n_len = np.linalg.norm(n_fit, axis=1, keepdims=True)
    n_fit = n_fit / np.maximum(n_len, 1e-9)
    normals[ok] = n_fit
    cents[ok] = means[ok]
    ds[ok] = (n_fit * means[ok]).sum(1)
    return normals, ds, cents


# ── §7.1 评估匹配 ──────────────────────────────────────────────────


def fit_face_planes(pair_sp: np.ndarray, pair_face: np.ndarray,
                    pair_cnt: np.ndarray, cent_mm: np.ndarray,
                    gt_n: np.ndarray, huber_mm: float = 30.0,
                    iters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """逐 GT 面鲁棒拟合"物理平面"（P-045 参照系教训的指标端平移，D17）。

    加权 TLS + IRLS：初始权重=pair_cnt，每轮按质心到当前平面的残差做
    Huber 重加权（huber_mm 内线性、外常数），共 iters 轮。
    为什么必须鲁棒：面行里混着跨界超点——它们只有几个点在本面上、
    质心却站在邻面上，普通加权均值/PCA 被拉歪（2026-10-27 实锤：小端
    面 35% 权重来自跨界超点，平均法向歪 32°、偏移 632mm）；IRLS 把离
    面质心的权重压没，面内质心保留。
    为什么要有拟合面：face_table 的 n/d 是名义模面，点云带 σ12-40mm
    翘曲——小端面的承载质心平均不掉翘曲，对名义面偏移天然 ~σ。部署
    消费的是物理面，判据对拟合面做；名义口径并列报告（buckets_nominal）。
    """
    F = len(gt_n)
    n_fit = np.zeros((F, 3), np.float64)
    d_fit = np.zeros(F, np.float64)
    for fi in range(F):
        m = pair_face == fi
        if not m.any():
            continue
        c = cent_mm[pair_sp[m]].astype(np.float64)
        w0 = pair_cnt[m].astype(np.float64)
        n = None
        d = 0.0
        w = w0 / w0.sum()
        for _ in range(iters):
            mu = (c * w[:, None]).sum(0) / w.sum()
            cov = ((c - mu) * w[:, None]).T @ (c - mu) / w.sum()
            _, vec = np.linalg.eigh(cov)
            n = vec[:, 0]
            if float(n @ gt_n[fi]) < 0:
                n = -n
            d = float(n @ mu)
            res = np.abs(c @ n - d)
            w = w0 * np.minimum(1.0, huber_mm / np.maximum(res, 1e-9))
            w = w / max(w.sum(), 1e-9)
        n_fit[fi] = n
        d_fit[fi] = d
    return n_fit, d_fit

def match_faces(inst: np.ndarray,
                pair_sp: np.ndarray, pair_face: np.ndarray,
                pair_cnt: np.ndarray, cent_mm: np.ndarray,
                nrm: np.ndarray,
                gt_n: np.ndarray, gt_d: np.ndarray,
                offset_gate: bool = False,
                match_deg: float = MATCH_DEG, match_mm: float = MATCH_MM,
                match_cov: float = MATCH_COV) -> list[dict]:
    """预测实例 ↔ GT 面匹配（§7.1，D17 终版证据语义）。

    逐 (实例, 面) 对行聚合承载证据：W=承载点数、C=承载质心、N=承载法向
    （超点 LS 法向加权均值）。判据：
    - 覆盖 cov = W/该面可见承载 ≥30%（点级精确分摊，主判据）；
    - 法向角 = arccos|N·n_nominal| <5°（名义法向精确无估计噪声）；
    - 偏移 = |n_nominal·C − d_nominal|：σ≤40mm 翘曲下与 30mm 门结构性
      冲突（对完美 GT 实例也误杀 26-45%，三次拟合参照均告失败），D17
      终版降为诊断量——offset_gate=True 时恢复旧 §7.1 严格口径并列
      报告（buckets_strict），交用户裁定。

    Returns：每 GT 面一条 dict——face_idx/cov/angle/offset/matched/
    no_support(GT 面缓存零承载)。
    """
    rows = []
    F = len(gt_n)
    n_inst = int(inst.max()) + 1 if (inst >= 0).any() else 0
    if n_inst == 0 or len(pair_face) == 0:
        return [{"face_idx": fi, "cov": 0.0, "angle": 180.0,
                 "offset": np.inf, "matched": False,
                 "no_support": True} for fi in range(F)]
    denom = np.bincount(pair_face, weights=pair_cnt.astype(np.float64),
                        minlength=F)
    sel = inst[pair_sp] >= 0
    ps, pf = pair_sp[sel], pair_face[sel]
    pc = pair_cnt[sel].astype(np.float64)
    pi = inst[ps].astype(np.int64)
    key = pi * F + pf
    ukey, inv = np.unique(key, return_inverse=True)
    W = np.bincount(inv, weights=pc)
    C = np.zeros((len(ukey), 3), np.float64)
    Nv = np.zeros((len(ukey), 3), np.float64)
    for a in range(3):
        C[:, a] = np.bincount(inv, weights=pc * cent_mm[ps, a])
        Nv[:, a] = np.bincount(inv, weights=pc * nrm[ps, a])
    u_inst = ukey // F
    u_face = ukey % F
    C = C / np.maximum(W, 1e-9)[:, None]   # 加权和→加权质心（漏除曾致
    # off~1e8mm、near 排序全废，2026-10-27 单场景手工对照实锤）
    ln = np.linalg.norm(Nv, axis=1, keepdims=True)
    Nv = Nv / np.maximum(ln, 1e-9)
    for fi in range(F):
        row = {"face_idx": fi, "cov": 0.0, "angle": 180.0, "offset": np.inf,
               "matched": False, "no_support": False}
        if denom[fi] <= 0:
            row["no_support"] = True
            rows.append(row)
            continue
        mrow = np.flatnonzero(u_face == fi)
        if len(mrow) == 0:
            rows.append(row)
            continue
        best = None
        for r in mrow:
            nd = abs(float(Nv[r] @ gt_n[fi]))
            ang = float(np.degrees(np.arccos(np.clip(nd, 0.0, 1.0))))
            off = abs(float(gt_n[fi] @ C[r]) - float(gt_d[fi]))
            near = (max(ang / match_deg, 1.0) - 1.0)                 + (max(off / match_mm, 1.0) - 1.0)
            cand = (near, -float(W[r] / denom[fi]), float(ang),
                    float(off), int(u_inst[r]))
            if best is None or cand[:2] < best[:2]:
                best = cand
        _near, ncov, ang, off, _ii = best
        cov = -ncov
        # D17 终版门限：仅支撑覆盖（点级精确、无参照系依赖）。角度/偏移
        # 为诊断量——实测 oracle 证据法向 p90：身体 12°/连接面 46°、偏移
        # 天然 ~σ翘曲，任何固定门都在杀完美实例
        ok = cov >= match_cov
        if offset_gate:                      # 严格 §7.1（旧三判据），并列报告
            ok = ang < match_deg and off < match_mm and cov >= match_cov
        row.update({"cov": cov, "angle": ang, "offset": off,
                    "matched": bool(ok)})
        rows.append(row)
    return rows
