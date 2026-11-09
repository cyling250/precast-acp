# -*- coding: utf-8 -*-
"""点级圆检测头 v2（T-035「v2 打包」，2026-10-24）：v1 未调优基线之上，
调优件 + 架构件一次落地（交接文档 T-035 下一步 1，用户定稿）。

v1→v2 变更：
调优件（治收敛慢，T-035 七病因对症）：
  ① trunk 加 BatchNorm（训练期只统计 valid 行，padding 种子不入统计）；
  ② 双尺度邻域：kNN32 细尺度 + 100mm ball×96 粗尺度（粗尺度提供
    "凸出柱 vs 平面/泡沫"的上下文证据，治 30mm 邻域证据稀释）；
  ③ obj 双半径标签：正类盘 R_OBJ=24mm 收窄（vote/规格/IoU 盘保持 40mm）
    ——24mm 盘基本落在筋上，正负边界与局部证据对齐，治分数两极化；
  ④ focal 按 GT 圈数归一（不再被正种子数稀释，漏一根有梯度压力）；
  ⑤ 前景分割辅助头 fg（标签=GT 盘，免费）概率参与簇心加权投票。
架构件（治天花板）：
  ⑥ 簇精修头（VoteNet proposal 精神=用户"峰验证器"提案）：池化簇成员
    trunk 特征（mean+max）→ Δ中心（tanh×48mm 有界）+ 规格重判 +
    置信度重打分；训练期正池=GT 40mm 盘种子、负池=负种子连续切片
    （体素字典序≈空间扫描序）——主攻规格 16%→90% 与位置 16.9→≤2mm。
  训练入口配套：cosine LR+2 轮 warmup、batch 4×60 轮（步数 ×4）、
  规格平衡采样器（治"记本批的径"）、w_vote=2.0（见 train_e2e_point.py）。

v1 兼容：构造 flags 全关（bn/coarse/refine/fg=False）即复现 v1 架构与
state_dict 键名，供旧权重基线评估。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED_VOX = 10.0   # 种子体素（视图点集 → 种子点；分类拟合分辨率解耦原则）
K_NB = 32         # 细尺度邻域点数
R_OBJ = 24.0      # obj/fg 正种子半径（v2 收窄；< 最小筋距 120mm/2）
R_OBJ_SLEEVE = 36.0  # 套筒视图专用：孔心是空的，环壁在 ID/2~ID/2+壁厚
                     # （18–35mm）——36mm 盘对齐环壁证据边界，24mm 会饿死正类
R_POS = 40.0      # 投票/规格/IoU 正种子盘（保持 v1）
VOTE_RANGE = 48.0  # 投票偏移上界（tanh×此值；≥R_POS 才能覆盖盘边种子）
REF_RANGE = 48.0   # 精修 Δ中心上界（覆盖早期簇心最差残差）
COARSE_R = 100.0   # 粗尺度 ball 半径
K_COARSE = 96      # 粗尺度邻域点数上限
MAX_SEEDS = 8192
NB_CAP = 40000     # 邻域源点上限（内存护栏；超量随机下采样）
REF_POOL = 64      # 精修负池种子数
RADIAL_R_MAX = 36.0  # 径向剖面半径域（> 最大筋半径 16+余量；< 最小筋距/2 60）
RADIAL_BINS = 18     # 2mm 分桶——档距 2mm 的可分辨下限


def radial_profile(pts_uv, center, r_max=RADIAL_R_MAX, nbins=RADIAL_BINS):
    """检测平面原始点 (N,2) + 参考心 (2,) → 径向距离直方图 (nbins,)
    L1 归一化（点级证据，v3 精修头规格输入；T-035 诊断：种子 10mm 体素
    分辨不了 2mm 档距，原始点 2mm 桶可以）。空点/零距离 → 全零。"""
    if pts_uv is None or len(pts_uv) == 0:
        return np.zeros(nbins, np.float32)
    d = np.hypot(pts_uv[:, 0] - center[0], pts_uv[:, 1] - center[1])
    h, _ = np.histogram(d, bins=nbins, range=(0.0, r_max))
    s = int(h.sum())
    return (h.astype(np.float32) / s if s
            else np.zeros(nbins, np.float32))


def _radial_profiles_gpu(pts_uv, n_pts, centers, center_batch,
                         r_max=RADIAL_R_MAX, nbins=RADIAL_BINS):
    """批量径向剖面（损失负池专用，torch no_grad 无梯度路径）。
    pts_uv (B,N,2) 零填充 + n_pts (B,) 有效数；centers (M,2) **全局槽**
    （跨 batch 的扁平槽 id）；center_batch (M,) 各槽所属样本批号
    （只聚合本批样本的点，跨样本不污染）。返回 (M, nbins) L1 归一化。"""
    with torch.no_grad():
        B, N, _ = pts_uv.shape
        M = centers.shape[0]
        dev = pts_uv.device
        valid = (torch.arange(N, device=dev)[None, :]
                 < n_pts[:, None])                              # (B,N)
        d = torch.norm(pts_uv[:, :, None, :] - centers[None, None, :, :],
                       dim=-1)                                   # (B,N,M)
        same = (center_batch[None, None, :]
                == torch.arange(B, device=dev)[:, None, None])
        m = (d < r_max) & valid[:, :, None] & same               # (B,N,M)
        bidx = (d / (r_max / nbins)).clamp(0, nbins - 1).long()
        flat = torch.arange(M, device=dev)[None, None, :].expand(
            B, N, M)[m]
        out = torch.zeros(M, nbins, device=dev)
        out.index_add_(0, flat, torch.ones(int(m.sum()), nbins, device=dev))
        out = out / out.sum(-1).clamp(min=1.0)[:, None]
    return out


# ── 种子与邻域（numpy，Dataset.__getitem__ 内调用） ──────────────────────
def make_seeds(p, seed_vox=SEED_VOX, k=K_NB, max_seeds=MAX_SEEDS,
               nb_cap=NB_CAP, rng=None, coarse_r=COARSE_R, k2=K_COARSE):
    """视图点集 (N,3) → (seeds, nb_rel, extra, nb2_rel)。

    seeds (S,3) 体素心；nb_rel (S,k,3) 细邻域相对坐标/50；nb2_rel (S,k2,3)
    粗邻域（100mm ball 内最近 k2 点，不足重复最近邻填充）相对坐标/100；
    extra (S,7)：[u/600, v/600, t/1000, 邻域凸出高度(t_max−t)/1000,
    邻域t std/100, 邻域均距/50, 面内距离90分位/30]——凸出高度是
    "这附近有没有东西长出来"的直接证据。

    性能实现（2026-10-24 基准 temp_scripts/script_20261024_make_seeds_bench）：
    体素键 int64 编码 + bincount 聚合（替代 axis=0 unique + add.at）；
    细/粗邻域共用一次 k2 近邻查询（细=前 k 列）。与旧双查询实现逐位等价，
    仅第 k/k+1 近邻零距并列（实测 ≤1/7016 种子，gap=0mm）换点——低于
    增广噪声，功能等价；提速 2.4–3.3×。
    """
    from scipy.spatial import cKDTree
    p = np.asarray(p, np.float32)
    if len(p) < 3:
        e = np.zeros((0, 3), np.float32)
        return (e, np.zeros((0, k, 3), np.float32), np.zeros((0, 7), np.float32),
                np.zeros((0, k2, 3), np.float32))
    kf = np.floor(p / seed_vox).astype(np.int64) + (1 << 20)
    key = (kf[:, 0] << 42) | (kf[:, 1] << 21) | kf[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    n = len(uk)
    cnt = np.bincount(inv, minlength=n)
    sums = np.stack([np.bincount(inv, weights=p[:, i], minlength=n)
                     for i in range(3)], 1)
    cents = (sums / cnt[:, None]).astype(np.float32)
    if len(cents) > max_seeds:
        idx = (rng or np.random.default_rng(0)).choice(len(cents), max_seeds,
                                                       replace=False)
        cents = cents[idx]
    src = p if len(p) <= nb_cap else p[
        (rng or np.random.default_rng(1)).choice(len(p), nb_cap,
                                                 replace=False)]
    kq = min(k2, len(src))
    tree = cKDTree(src)
    d2k, ix2 = tree.query(cents, k=kq, workers=-1)
    if kq == 1:
        d2k, ix2 = d2k[:, None], ix2[:, None]
    if kq < k2:
        ix2 = np.concatenate([ix2, np.repeat(ix2[:, -1:], k2 - kq, 1)], 1)
        d2k = np.concatenate([d2k, np.full((len(cents), k2 - kq), np.inf)], 1)
    k_eff = min(k, kq)
    nb = src[ix2[:, :k_eff]]                        # (S,k,3) 前 k 列=细邻域
    d_k = d2k[:, :k_eff]
    rel = (nb - cents[:, None, :]) / 50.0
    rel2 = (src[ix2] - cents[:, None, :]) / coarse_r
    beyond = d2k > coarse_r
    if beyond.any():
        rel2 = np.where(beyond[:, :, None], rel2[:, :1], rel2)
    t_s = cents[:, 0]
    lat = np.linalg.norm(nb[:, :, 1:3] - cents[:, None, 1:3], axis=-1)
    extra = np.stack([cents[:, 1] / 600.0, cents[:, 2] / 600.0,
                      t_s / 1000.0,
                      (nb[:, :, 0].max(1) - t_s) / 1000.0,
                      nb[:, :, 0].std(1) / 100.0,
                      d_k.mean(1) / 50.0,
                      np.percentile(lat, 90, axis=1) / 30.0],
                     1).astype(np.float32)
    return cents, rel.astype(np.float32), extra, rel2.astype(np.float32)


def assign_targets(seeds, gt_uv, r_pos=R_POS, r_obj=R_OBJ):
    """种子 → (gt_idx, vote_target, pos_obj)。gt_idx=-1 为负种子。

    vote_target 单位 mm（= GT 圆心 − 种子(u,v)），训练时再除 VOTE_RANGE。
    pos_obj = 距 GT 圆心 < r_obj（obj/fg 正类）；gt_idx 分配盘保持 r_pos
    （vote/规格/IoU 监督面）。同一 GT 圆盘内多个 GT 时归最近者
    （最小筋距 120mm 下不发生）。
    """
    gt_idx = np.full(len(seeds), -1, np.int64)
    vote = np.zeros((len(seeds), 2), np.float32)
    pos_obj = np.zeros(len(seeds), bool)
    if len(gt_uv) == 0 or len(seeds) == 0:
        return gt_idx, vote, pos_obj
    d2 = ((seeds[:, None, 1:3] - gt_uv[None, :, :]) ** 2).sum(-1)
    best = d2.argmin(1)
    d_min = d2[np.arange(len(seeds)), best]
    within = d_min < r_pos * r_pos
    gt_idx[within] = best[within]
    vote[within] = gt_uv[best[within]] - seeds[within, 1:3]
    pos_obj = d_min < r_obj * r_obj
    return gt_idx, vote, pos_obj


# ── 网络 ────────────────────────────────────────────────────────────────
class PointCircleNet(nn.Module):
    """种子邻域 mini-PointNet（局部聚合）→ 多头（obj/vote/spec/fg/refine）。

    v2：双尺度邻域（coarse）+ trunk BatchNorm（bn，valid 行掩码统计）+
    簇精修头（refine）+ 前景辅助头（fg）。flags 全关即 v1 架构（旧权重
    基线评估用，键名兼容）。工程红线：两层 MLP，无深架构。

    forward 输入 (B,S,k,3)+(B,S,6)+(B,S,k2,3)+valid(B,S)，输出 dict：
    obj (B,S) / vote (B,S,2)（tanh×VOTE_RANGE，mm）/ spec (B,S,n_spec) /
    fg (B,S) 或 None / h (B,S,hidden)（trunk 特征，精修池化用）。
    """

    def __init__(self, n_spec, hidden=128, k_feat=64, in_extra=7,
                 bn=True, coarse=True, refine=True, fg=True, ref_radial=0):
        super().__init__()
        self.n_spec = n_spec
        self.bn_on, self.coarse_on = bn, coarse
        self.refine_on, self.fg_on = refine, fg
        self.ref_radial = int(ref_radial)   # >0: 精修头拼接径向剖面（v3）
        self.nb = nn.Sequential(nn.Linear(3, 32), nn.ReLU(inplace=True),
                                nn.Linear(32, k_feat), nn.ReLU(inplace=True))
        d_in = k_feat + in_extra
        if coarse:
            self.nb2 = nn.Sequential(nn.Linear(3, 32), nn.ReLU(inplace=True),
                                     nn.Linear(32, k_feat), nn.ReLU(inplace=True))
            d_in += k_feat
        # ModuleList 键名与 v1 Sequential 对齐（bn=False 时 trunk.0/2=Linear）
        if bn:
            self.trunk = nn.ModuleList([
                nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True)])
        else:
            self.trunk = nn.ModuleList([
                nn.Linear(d_in, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden), nn.ReLU(inplace=True)])
        self.obj = nn.Linear(hidden, 1)
        self.vote = nn.Linear(hidden, 2)
        self.spec = nn.Linear(hidden, n_spec)
        if fg:
            self.fg = nn.Linear(hidden, 1)
        if refine:
            d_ref = 2 * hidden + (self.ref_radial if refine else 0)
            self.ref = nn.Sequential(nn.Linear(d_ref, hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden, hidden),
                                     nn.ReLU(inplace=True))
            self.ref_delta = nn.Linear(hidden, 2)
            self.ref_spec = nn.Linear(hidden, n_spec)
            self.ref_conf = nn.Linear(hidden, 1)
        if bn:
            # focal 负项 Σ/n_gt 的量纲要求低初始置信（CenterNet 惯例 ŷ₀≈0.1）
            for head in (self.obj, self.fg if fg else None):
                if head is not None:
                    nn.init.constant_(head.bias, -2.19)

    def _bn_rows(self, x, bn, valid_flat):
        """BN；训练期仅统计 valid 行（padding 种子零向量会污染均值方差），
        其余行写 0（损失侧有 valid 掩码）。评估期用 running 统计全量归一。"""
        if valid_flat is None or not self.training or bool(valid_flat.all()):
            return bn(x)
        out = torch.zeros_like(x)
        out[valid_flat] = bn(x[valid_flat])
        return out

    def _trunk(self, x, valid):
        # BatchNorm1d 只吃 2D——先展平 (B*S,C)，末端还原 (B,S,hidden)
        B, S = x.shape[0], x.shape[1]
        x2 = x.reshape(B * S, -1)
        vf = valid.reshape(-1) if valid is not None else None
        if self.bn_on:
            h = self._bn_rows(self.trunk[0](x2), self.trunk[1], vf)
            h = self.trunk[2](h)
            h = self._bn_rows(self.trunk[3](h), self.trunk[4], vf)
            h = self.trunk[5](h)
        else:
            h = self.trunk[1](self.trunk[0](x2))
            h = self.trunk[3](h)
        return h.reshape(B, S, -1)

    def forward(self, nb_rel, extra, nb2_rel=None, valid=None):
        f = self.nb(nb_rel).max(2)[0]
        feats = [f]
        if self.coarse_on:
            assert nb2_rel is not None, "coarse 模式需要 nb2_rel"
            feats.append(self.nb2(nb2_rel).max(2)[0])
        h = self._trunk(torch.cat(feats + [extra], -1), valid)
        out = {"obj": self.obj(h).squeeze(-1),
               "vote": torch.tanh(self.vote(h)) * VOTE_RANGE,
               "spec": self.spec(h), "h": h, "fg": None}
        if self.fg_on:
            out["fg"] = self.fg(h).squeeze(-1)
        return out

    def refine(self, h_pool):
        """池化簇特征 (N,2*hidden[+ref_radial]) → Δ中心 (N,2) tanh×REF_RANGE
        （mm）、规格 logits (N,n_spec)、置信度 logits (N,)。ref_radial>0
        时 h_pool 末尾必须已拼径向剖面 (N,nbins)（维度不符在 ref 层报错）。"""
        z = self.ref(h_pool)
        return {"delta": torch.tanh(self.ref_delta(z)) * REF_RANGE,
                "spec": self.ref_spec(z),
                "conf": self.ref_conf(z).squeeze(-1)}


# ── 损失 ────────────────────────────────────────────────────────────────
def _focal_terms(logit, pos, valid=None, alpha=2.0, beta=4.0):
    """CenterNet 式 focal 的逐点项（P-034 教训逐项对照原式；负项 ŷᵅ 防塌缩）。

    valid 给定时，padding 种子既不进正项也不进负项。聚合方式由调用方
    决定：v1=全局均值（focal_obj_loss）；v2=按圈聚合（point_loss 内段
    聚合，每圈等压、漏圈不被正种子池稀释——T-035 病因⑦；正类是"盘内
    上千种子"而非 CenterNet 单格，直接 Σ/n_gt 会量纲爆炸，须圈内先均值；
    负项 Σ/n_gt 须配 obj/fg 头低偏置初始化 ≈-2.19）。
    """
    pred = torch.sigmoid(logit)
    eps = 1e-6
    w = torch.ones_like(pred) if valid is None else valid.float()
    pos_l = ((1 - pred) ** alpha
             * -torch.log(pred.clamp(min=eps))) * pos * w
    neg_l = ((1 - pos.float()) ** beta * pred.clamp(max=1 - eps) ** alpha
             * -torch.log((1 - pred).clamp(max=1 - eps))) * (~pos) * w
    return pos_l, neg_l


def focal_obj_loss(logit, pos, valid=None, alpha=2.0, beta=4.0, norm=None):
    """v1 口径 focal：Σ/denom（默认=正种子数）。norm 传张量可免同步。"""
    pos_l, neg_l = _focal_terms(logit, pos, valid, alpha, beta)
    if norm is None:
        w = torch.ones_like(pos_l) if valid is None else valid.float()
        norm = (pos.float() * w).sum().clamp(min=1.0)
    return (pos_l.sum() + neg_l.sum()) / norm


def _seg_pool(x, members, seg, n_slot):
    """段聚合 (mean, max, cnt)：x (P,D) 展平特征，members (M,) 行索引，
    seg (M,) 段 id ∈ [0,n_slot)。空段 mean=0/max=0（损失侧 nz 掩码兜底）。"""
    D = x.shape[1]
    cnt = torch.zeros(n_slot, device=x.device, dtype=x.dtype)
    cnt.index_add_(0, seg, torch.ones(len(members), device=x.device,
                                      dtype=x.dtype))
    s = torch.zeros(n_slot, D, device=x.device, dtype=x.dtype)
    s.index_add_(0, seg, x[members])
    mean_ = s / cnt.clamp(min=1.0)[:, None]
    mx = torch.full((n_slot, D), float("-inf"), device=x.device, dtype=x.dtype)
    mx.scatter_reduce_(0, seg[:, None].expand(-1, D), x[members],
                       reduce="amax", include_self=True)
    mx = torch.nan_to_num(mx, neginf=0.0)            # 空段 -inf → 0
    return mean_, mx, cnt


def circle_iou(c1, r1, c2, r2, eps=1e-6):
    """两圆 IoU（闭式，可微）。c: (K,2)，r: (K,)，单位 mm。"""
    d = torch.norm(c1 - c2, dim=-1).clamp(min=eps)
    r1 = r1.clamp(min=eps)
    r2 = r2.clamp(min=eps)
    a1 = torch.acos(((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
                    .clamp(-1 + 1e-4, 1 - 1e-4))
    a2 = torch.acos(((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
                    .clamp(-1 + 1e-4, 1 - 1e-4))
    tri = 0.5 * torch.sqrt(torch.clamp(
        (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2),
        min=0.0))
    inter = r1 * r1 * a1 + r2 * r2 * a2 - tri
    inter = torch.where(d >= r1 + r2, torch.zeros_like(inter), inter)
    inner = torch.pi * torch.minimum(r1, r2) ** 2
    inter = torch.where(d <= (r1 - r2).abs(), inner, inter)
    union = torch.pi * r1 ** 2 + torch.pi * r2 ** 2 - inter
    return inter / union.clamp(min=eps)


def point_loss(out, tgt, spec_table, model=None, w_vote=2.0, w_spec=1.0,
               w_iou=1.0, w_r=0.5, w_fg=0.5, w_ref=1.0, n_ref_neg=4, v1=False):
    """out=PointCircleNet 输出；tgt=dict(seeds_uv, pos, pos_obj, vote_t,
    gt_idx, spec_t, gt_circles (K,4: u,v,r_mm,spec_id), valid)。

    v2 损失面：obj/fg focal 用 pos_obj（24mm 盘）+ 按圈聚合（每圈等压）；
    vote/规格/圆 IoU 保持 40mm 盘（v1 口径）；簇精修头损失（正池=GT 盘
    种子、负池=负种子连续切片）。v1=True 整体退回 v1 损失口径（obj 用
    40mm 盘 + 全局均值，无 fg/ref 项）——--v1 基线复训用。

    性能（2026-10-24 profile：旧逐圈 Python 循环每步 ~68 次 GPU→CPU 同步
    + 数千微内核，loss+backward 占 epoch 73%）：圈级聚合全部改为
    index_add/scatter_reduce 段聚合 + 单次 batched refine 前向，全程无
    逐圈同步。与旧实现数值等价（temp_scripts/script_20261024_point_loss_
    equiv.py 验证），仅两处已记录微差：①圆 IoU 权重和退化的圈由"跳过"
    改为权重和 clamp(1e-3) 参与；②精修 c0 退化回退改为 α=1e-3 混合。
    """
    pos = tgt["pos"] if v1 else tgt["pos_obj"]
    pos_full = tgt["pos"]
    valid = tgt.get("valid")
    gt_idx = tgt["gt_idx"]
    gt_circ = tgt["gt_circles"]
    B, S = gt_idx.shape
    K = gt_circ.shape[1]
    dev = gt_idx.device
    pf = pos_full.float()
    npos = pf.sum().clamp(min=1.0)

    per_v = F.smooth_l1_loss(out["vote"] / VOTE_RANGE,
                             tgt["vote_t"] / VOTE_RANGE,
                             reduction="none").sum(-1)
    l_vote = (per_v * pf).sum() / (2.0 * npos)

    # ── 圈级段结构：直接用 B*K 全局槽 id 聚合（index_add 支持非连续 id；
    #    此前"紧凑 id + [nz] 选择"在圈数不齐的 batch 里行序错位——P-044 家
    #    族教训：聚合 id 与选择掩码必须同一索引空间） ──
    member_mask = (gt_idx >= 0) & valid
    n_slot = B * K
    flat_slot = gt_idx + torch.arange(B, device=dev)[:, None] * K
    counts = torch.bincount(flat_slot[member_mask].reshape(-1),
                            minlength=n_slot)
    nz = counts > 0
    members = torch.where(member_mask.reshape(-1))[0]
    seg = flat_slot.reshape(-1)[members]             # (M,) 槽 id
    n_nz = nz.sum()
    gt_compact = gt_circ.reshape(n_slot, 4)[nz]      # (C,4) 与 [nz] 行序一致
    cnt_c = counts[nz].float()

    if v1:
        denom = (tgt["pos"].float() * valid.float()).sum().clamp(min=1.0)
        l_obj = focal_obj_loss(out["obj"], pos, valid, norm=denom)
        ce = F.cross_entropy(out["spec"].reshape(-1, out["spec"].shape[-1]),
                             tgt["spec_t"].reshape(-1), reduction="none")
        l_spec = (ce * pf.reshape(-1)).sum() / npos
    else:
        pos_l, neg_l = _focal_terms(out["obj"], pos, valid)
        pc = torch.zeros(n_slot, device=dev, dtype=pos_l.dtype)
        pc.index_add_(0, seg, pos_l.reshape(-1)[members])
        l_obj = (pc[nz] / cnt_c).sum() / n_nz.clamp(min=1.0) \
            + neg_l.sum() / n_nz.clamp(min=1.0)
        # v2b：spec CE 收窄到筋心盘并按圈聚合——24-40mm 环带种子站在面片上
        # 无半径证据，40mm 全盘监督=噪声梯度稀释筋心信号（v2 spec 平台根因）
        ce = F.cross_entropy(out["spec"].reshape(-1, out["spec"].shape[-1]),
                             tgt["spec_t"].reshape(-1),
                             reduction="none").reshape(B, S)
        pcs = torch.zeros(n_slot, device=dev, dtype=ce.dtype)
        pcs.index_add_(0, seg, (ce * pos.float()).reshape(-1)[members])
        l_spec = (pcs[nz] / cnt_c).sum() / n_nz.clamp(min=1.0)

    l_iou = torch.zeros((), device=dev)
    l_r = torch.zeros((), device=dev)
    if K and len(members) > 0:
        centers_pred = tgt["seeds_uv"] + out["vote"]
        probs = torch.softmax(out["spec"], -1)
        w_seed = torch.sigmoid(out["obj"]).clamp(min=1e-4).detach()
        # 权重 detach：IoU 只对圆心/半径给梯度，避免 obj 自我强化
        w_m = w_seed.reshape(-1)[members]
        ws = (torch.zeros(n_slot, device=dev).index_add_(0, seg, w_m)
              + 1e-3)                                # 退化圈 clamp（原为跳过）
        cs = torch.zeros(n_slot, 2, device=dev).index_add_(
            0, seg, centers_pred.reshape(-1, 2)[members] * w_m[:, None])
        pp = probs.reshape(-1, probs.shape[-1])
        ps = torch.zeros(n_slot, pp.shape[-1], device=dev).index_add_(
            0, seg, pp[members] * w_m[:, None])
        c_hat = (cs / ws[:, None])[nz]
        r_hat = ((ps / ws[:, None])[nz] @ torch.as_tensor(
            spec_table, dtype=torch.float32, device=dev)) * 0.5
        ious = 1 - circle_iou(c_hat, r_hat, gt_compact[:, :2],
                              gt_compact[:, 2])
        l_iou = ious.sum() / n_nz.clamp(min=1.0)
        if not v1:
            # v2b：软表期望半径 r̂ 的直接监督（档位分类形式不变，非半径回归；
            # 平滑尺度 10mm≈相邻档半距，自带距离感知梯度）
            l_r = F.smooth_l1_loss((r_hat - gt_compact[:, 2]) / 10.0,
                                   torch.zeros_like(r_hat))

    l_fg = torch.zeros((), device=dev)
    if out.get("fg") is not None and not v1:
        pos_l, neg_l = _focal_terms(out["fg"], pos, valid)
        pc = torch.zeros(n_slot, device=dev, dtype=pos_l.dtype)
        pc.index_add_(0, seg, pos_l.reshape(-1)[members])
        l_fg = (pc[nz] / cnt_c).sum() / n_nz.clamp(min=1.0) \
            + neg_l.sum() / n_nz.clamp(min=1.0)

    # ── 簇精修头：全部圈一次 batched 前向 + 负池，无逐圈同步 ──
    z0 = torch.zeros((), device=dev)
    l_rc = l_rd = l_rs = z0
    if model is not None and model.refine_on and not v1:
        H = out["h"].shape[-1]
        h_all = out["h"].reshape(-1, H)
        pool_mean, pool_max, _ = _seg_pool(h_all, members, seg, n_slot)
        # c0=分数加权簇心（α=1e-3 混合无权均值，替代旧退化回退）
        cp = (tgt["seeds_uv"] + out["vote"]).reshape(-1, 2)
        sc = torch.sigmoid(out["obj"]).detach().reshape(-1)
        ws0 = torch.zeros(n_slot, device=dev).index_add_(0, seg, sc[members])
        cs0 = torch.zeros(n_slot, 2, device=dev).index_add_(
            0, seg, cp[members] * sc[members][:, None])
        us0 = torch.zeros(n_slot, 2, device=dev).index_add_(0, seg, cp[members])
        c0 = ((cs0 + 1e-3 * us0) / (ws0[:, None] + 1e-3))[nz]
        # 负池：每样本 n_ref_neg 个连续负种子切片（体素字典序≈空间扫描序）
        neg_mask = (gt_idx == -1) & valid
        cnt_neg = neg_mask.sum(1)                        # (B,)
        base = torch.cumsum(cnt_neg, 0) - cnt_neg
        size = min(REF_POOL, S)
        u = torch.rand(B, n_ref_neg, device=dev)
        start = (u * (cnt_neg - size + 1).clamp(min=1)[:, None]).long()
        ok_neg = (cnt_neg >= size)[:, None].expand(B, n_ref_neg).reshape(-1)
        local = (start[:, :, None]
                 + torch.arange(size, device=dev)[None, None, :])
        local = local.clamp(max=(cnt_neg - 1).clamp(min=0)[:, None, None])
        neg_positions = torch.where(neg_mask.reshape(-1))[0]
        neg_positions = torch.cat([neg_positions,
                                   torch.zeros(1, device=dev, dtype=torch.long)])
        gpos = (base[:, None, None] + local).reshape(-1)
        seed_pos = neg_positions[gpos.clamp(max=len(neg_positions) - 1)]
        hp = h_all[seed_pos]                             # (B*neg*size, H)
        nslot2 = B * n_ref_neg
        seg2 = torch.arange(nslot2, device=dev)[:, None].expand(
            -1, size).reshape(-1)
        nm, mx, _ = _seg_pool(hp, torch.arange(B * n_ref_neg * size,
                                               device=dev), seg2, nslot2)
        # ref_radial>0 时必须走径向分支（ref 层入宽 2H+R）：空圈批
        # （纯负视图）radial numel=0 但 reshape(n_slot,R) 合法
        use_rad = (bool(getattr(model, "ref_radial", 0)) and "radial" in tgt
                   and (tgt["radial"].numel() == 0
                        or tgt["radial"].shape[-1] == model.ref_radial))
        if use_rad:
            # 正槽剖面=GT 心（Dataset 侧 numpy 精算）；全槽位前向（填充槽
            # 剖面=0），掩码在损失侧 [nz]——与 pool/r_neg 同一前向批
            prof_pos = tgt["radial"].to(dev).reshape(n_slot,
                                                     model.ref_radial)
            # 负切片参考心=片内种子 uv 均值（与正池 c0 同族的"当前估计"）
            suv = tgt["seeds_uv"].reshape(-1, 2)
            cs2 = torch.zeros(nslot2, 2, device=dev).index_add_(
                0, seg2, suv[seed_pos.clamp(max=len(suv) - 1)])
            cnt2 = torch.zeros(nslot2, device=dev).index_add_(
                0, seg2, torch.ones(len(seed_pos), device=dev))
            centers2 = cs2 / cnt2.clamp(min=1.0)[:, None]
            prof_neg = _radial_profiles_gpu(
                tgt["pts_uv"].to(dev), tgt["n_pts"].to(dev), centers2,
                center_batch=torch.arange(B, device=dev)[:, None].expand(
                    -1, n_ref_neg).reshape(-1))
            r_pos = model.refine(torch.cat([pool_mean, pool_max, prof_pos],
                                           -1))
            r_neg = model.refine(torch.cat([nm, mx, prof_neg], -1))
        else:
            r_pos = model.refine(torch.cat([pool_mean, pool_max], -1))
            r_neg = model.refine(torch.cat([nm, mx], -1))
        conf_all = torch.cat([r_pos["conf"], r_neg["conf"]])
        # 掩码与目标分离（此前混用导致负池零监督、精修头失控 NaN）：
        # 权重=nz（空槽剔除）+ ok_neg（负池剔除不足样本）；目标=1/0
        w_all = torch.cat([nz.float(), ok_neg.float()])
        t_all = torch.cat([torch.ones(n_slot, device=dev),
                           torch.zeros(B * n_ref_neg, device=dev)])
        l_rc = (F.binary_cross_entropy_with_logits(
            conf_all, t_all, reduction="none") * w_all).sum() \
            / w_all.sum().clamp(min=1.0)
        l_rd = (F.smooth_l1_loss(r_pos["delta"][nz] / REF_RANGE,
                                 (gt_compact[:, :2] - c0) / REF_RANGE,
                                 reduction="none").sum(-1)).sum() \
            / n_nz.clamp(min=1.0)
        l_rs = F.cross_entropy(r_pos["spec"][nz], gt_compact[:, 3].long(),
                               reduction="none").sum() / n_nz.clamp(min=1.0)
    l_ref = l_rc + l_rd + l_rs
    loss = (l_obj + w_vote * l_vote + w_spec * l_spec + w_iou * l_iou
            + w_r * l_r + w_fg * l_fg + w_ref * l_ref)
    parts = {"obj": float(l_obj.detach()), "vote": float(l_vote.detach()),
             "spec": float(l_spec.detach()), "iou": float(l_iou.detach()),
             "r": float(l_r.detach()),
             "fg": float(l_fg.detach()), "ref": float(l_ref.detach()),
             "rc": float(l_rc.detach()), "rd": float(l_rd.detach()),
             "rs": float(l_rs.detach())}
    return loss, parts


# ── 推理 ────────────────────────────────────────────────────────────────
@torch.no_grad()
def detect(model, seeds, nb_rel, extra, nb2_rel, spec_table, thr=0.5,
           merge_r=50.0, topk=None, device="cpu", pos_r=R_POS, pts_uv=None):
    """种子 → (N, 3+n_spec)：(u, v, conf, 规格 softmax)。

    pts_uv：检测平面原始点 (N,2)——ref_radial>0 时必给（每簇精测径向
    剖面，与训练正池同分布：参考心=簇心 c0）。
    base 检出（obj>thr）→ 贪心聚类（锚点=最高分未用种子，吸收 merge_r 内
    全部未用种子）→ 每簇过精修头：池化 c0 40mm 盘内种子 trunk 特征（与
    训练正池同分布）→ center=c0+Δ、conf=精修置信度（再次按 thr 过滤）、
    规格=精修 softmax；簇心与投票用 obj×fg 加权（前景辅助头加权投票）
    → 检出端 50mm NMS（投票散射把同一实例拆成多簇时去重，保最高 conf）。
    topk=None 不设种子配额——P-048 教训：旧默认 64 沿用 BEV decode 的
    峰级配额（每实例 1 峰），到点级头变成种子级配给，每件 580–950 个
    点火种子挤 64 槽、漏检筋 100% 进不了配额，召回钉死 66.1%；反事实
    实测 64→放开：钢筋召回 66.1%→95.3%、精度 99.3% 保持。
    v1 权重（refine_on=False）自动走 v1 路径。
    """
    if len(seeds) == 0:
        return np.zeros((0, 3 + len(spec_table)), np.float32)
    was_training = model.training
    model.eval()
    out = model(torch.from_numpy(nb_rel)[None].to(device),
                torch.from_numpy(extra)[None].to(device),
                torch.from_numpy(nb2_rel)[None].to(device)
                if nb2_rel is not None else None)
    score = torch.sigmoid(out["obj"])[0].cpu().numpy()
    vote = out["vote"][0].cpu().numpy()
    probs = torch.softmax(out["spec"], -1)[0].cpu().numpy()
    fg = (torch.sigmoid(out["fg"])[0].cpu().numpy()
          if out.get("fg") is not None else np.ones_like(score))
    if was_training:
        model.train()
    keep = np.where(score > thr)[0]
    if len(keep) == 0:
        return np.zeros((0, 3 + len(spec_table)), np.float32)
    c = seeds[keep, 1:3] + vote[keep]
    s = score[keep] * fg[keep]                       # 前景概率加权
    order = np.argsort(s)[::-1][:topk]
    used = np.zeros(len(order), bool)
    clusters = []                                    # 每簇=keep 内相对索引
    for a in range(len(order)):
        if used[a]:
            continue
        i = order[a]
        members = [i]
        for b in range(a + 1, len(order)):
            j = order[b]
            if not used[b] and np.linalg.norm(c[j] - c[i]) < merge_r:
                members.append(j)
                used[b] = True
        used[a] = True
        clusters.append(members)
    dets = []
    if model.refine_on:
        h = out["h"][0]
        for members in clusters:
            wm = s[members]
            c0 = (c[members] * wm[:, None]).sum(0) / wm.sum()
            d2 = ((seeds[:, 1:3] - c0) ** 2).sum(-1)
            pool_idx = np.where(d2 < pos_r * pos_r)[0]
            if len(pool_idx) == 0:
                pool_idx = keep[np.asarray(members)]
            hp = h[torch.from_numpy(pool_idx).to(device)]
            if getattr(model, "ref_radial", 0):
                prof = radial_profile(pts_uv, c0)          # (nbins,)
                inp = torch.cat([hp.mean(0), hp.max(0)[0],
                                 torch.from_numpy(prof).to(device)])[None]
                r = model.refine(inp)
            else:
                r = model.refine(torch.cat([hp.mean(0), hp.max(0)[0]])[None])
            conf = float(torch.sigmoid(r["conf"][0]))
            if conf < thr:
                continue
            center = c0 + r["delta"][0].cpu().numpy()
            p = torch.softmax(r["spec"][0], -1).cpu().numpy()
            dets.append(np.concatenate([center, [conf], p]))
    else:
        for members in clusters:
            wm = s[members]
            center = (c[members] * wm[:, None]).sum(0) / wm.sum()
            p_mean = (probs[members] * wm[:, None]).sum(0) / wm.sum()
            dets.append(np.concatenate([center, [float(wm.mean())], p_mean]))
    # 检出端 NMS（P-048 配套）：投票散射把同一实例拆成多簇时去重，
    # 保最高 conf——反事实实测套筒精度 90.2%→98.0%、召回不变
    if len(dets) > 1:
        kept = []
        for a in np.argsort([float(d[2]) for d in dets])[::-1]:
            if all(np.linalg.norm(dets[a][:2] - dets[b][:2]) >= merge_r
                   for b in kept):
                kept.append(a)
        dets = [dets[a] for a in kept]
    if len(dets) == 0:
        return np.zeros((0, 3 + len(spec_table)), np.float32)
    return np.stack(dets).astype(np.float32)
