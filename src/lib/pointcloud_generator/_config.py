"""
生成器 v2.1 全局配置：标签体系、默认参数、随机状态。

v2.1（2026-10-17，sim2real 根因修复轮）在 v2 基础上：
  - 密度体系重建：基准 0.10~0.14（≈3mm 体素 cap），钢筋密度对齐真实（≈混凝土同级，
    非稀疏）+ 沿轴密度剖面（峰/末端衰减/随机凹陷）；端面与套筒环改绝对密度；
  - 地面修正：不规则外边界、铺到构件接触线（删等宽护城河）、构件-地面高度错开；
  - 周向可见性：中段仅 ~110°-240° 可见弧（实测 36 箱仅 ~10 箱有占），端部全可见；
  - SfM 低频翘曲：同一形变场作用于全云（σ12-40mm、波长 2.5-6m）。
管线侧配套（lib/utils/datasets.py、experiments/real_case_e2e.py）：
  训练 patch 生成与真实推理统一 3mm 体素细化（VOXEL_THIN_MM）+ 训练旋转增广。

v2 合成原理（2026-10-16 定调，参照 real_resource/scene_full_mm - Cloud.ply 实测）：
  - 布局 C 唯一：每个构件一端（x=L）外露纵筋束 + 端头垫片，另一端（x=0）端面套筒阵；
  - 套筒 = 端面上凸起 2~8 mm 的平面空心圆环（孔内无点），环点标【混凝土】
    （决策 A：套筒不设 DL 类，靠几何检测空心圆）；
  - 纵筋 = 部分圆弧的半圆柱采样（可见弧 150°~260°，朝外侧偏上）；
  - 结构化缺失：构件平躺于地面 z=0——底面及离地 <30 mm 的点全部缺失、
    30~80 mm 概率渐变；端面随机椭圆无数据斑；
  - 噪声三种：jitter 0.15~1.2mm / keep 90~98.5% / 多边形遮挡；
  - 地面背景 + 杂物块统一标【非预制构件点】。
"""

import random
import numpy as np
from typing import Optional

# ─── 标签体系 v2（四类：混凝土/纵筋/箍筋/非预制构件点） ─────
LABEL_NON_MEMBER: int = 0   # 非预制构件点（地面+背景杂物+噪声）
LABEL_CONCRETE: int = 1     # 混凝土（含套筒端面环——决策 A）
LABEL_REBAR: int = 2        # 纵筋（含端头垫片——决策 A）
LABEL_STIRRUP: int = 3      # 箍筋（叠合梁顶面外伸 U 形筋）
NUM_CLASSES: int = 4
CLASS_NAMES: list[str] = ["non_member", "concrete", "rebar", "stirrup"]

# ─── 随机种子管理 ──────────────────────────────────────────
_SEED: int | None = None


def set_seed(seed: int) -> None:
    global _SEED
    _SEED = seed
    random.seed(seed)
    np.random.seed(seed)


def get_rng():
    return random.Random(_SEED) if _SEED is not None else random


# ─── 数据目录 ──────────────────────────────────────────────
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MOCK_DIR: str = os.path.join(_PROJECT_ROOT, "mock")
ORIGIN_DIR: str = os.path.join(MOCK_DIR, "origin")
QA_DIR: str = os.path.join(MOCK_DIR, "qa")
INUSE_DIR: str = os.path.join(MOCK_DIR, "inuse")
TRAIN_DIR: str = os.path.join(INUSE_DIR, "train")
VAL_DIR: str = os.path.join(INUSE_DIR, "val")
TEST_DIR: str = os.path.join(INUSE_DIR, "test")

# ─── 采样密度基准 v2.1 ─────────────────────────────────────
# 2026-10-17 密度探查定稿（temp_scripts/script_20261017_220500）：
#   真实云（清理后）各表面 ~0.2-0.25 pts/mm²，钢筋 0.21 为最密区之一（非稀疏）；
#   管线统一 3mm 体素细化（cap≈0.111 pts/mm²）后两域密度统计一致
#   （细化后 NN p50：筋 2.02 / 柱 2.10mm，密度不再可分）。
#   合成基准取 cap 附近：high 略高于 cap（细化后≈cap）、low 略低（保留稀疏变化）。
DENSITY_HIGH: float = 0.14
DENSITY_LOW: float = 0.10

DENSITY_MAP: dict[str, float] = {
    "high": DENSITY_HIGH,
    "low": DENSITY_LOW,
}

# 管线体素细化阈值（密度均衡降采样）：训练 patch 生成与真实推理共用。
# s=3mm 依据：≥ 最密区 NN p50×2.5（削峰）且 ≤ 最小可见筋径/3（保截面）；
# 实测扫描表：截面保持 99%，细化后钢筋/混凝土 NN 收敛（2.02/2.10）。
VOXEL_THIN_MM: float = 3.0

# ─── 钢筋参数 (GB 50010, GB 1499.2) ────────────────────────
STANDARD_REBAR_DIAMETERS: list[float] = [16.0, 18.0, 20.0, 22.0, 25.0, 28.0, 32.0]
CONCRETE_COVER: float = 25.0                # 纵筋保护层 mm（钢筋中心→混凝土表面）
REBAR_PROTRUSION_RANGE: tuple[float, float] = (400.0, 1500.0)   # 外露长（实测 ~1450）
REBAR_ARC_SPAN_DEG: tuple[float, float] = (150.0, 260.0)        # 可见圆弧（半圆柱效果）
REBAR_ARC_UP_BIAS_DEG: tuple[float, float] = (10.0, 40.0)       # 弧心向上偏（扫描自上方）
# v2.1 密度对齐真实（探查：真实钢筋≈混凝土同级、0.21 pts/mm²、最密区之一）。
# 沿轴密度剖面实测：峰在距端面 10~35% 处、末端衰减至 0.68×、末段骤降、
# 单根内 max/min 波动 3.2×——凹陷段低于 cap，细化后仍稀疏（真实同款）。
REBAR_DENSITY_FACTOR: tuple[float, float] = (0.7, 1.5)      # 相对基准（峰段密度）
REBAR_VISIBLE_R_FACTOR: tuple[float, float] = (0.45, 1.0)   # 可见弧半径/名义半径（真实 5-7/10-12mm）
REBAR_DENS_START_FACTOR: float = 0.80                       # 面端密度/峰密度（实测 ~0.77）
REBAR_DENS_PEAK_T: tuple[float, float] = (0.10, 0.35)       # 峰位置（占外露长比例）
REBAR_DENS_TIP_FACTOR: tuple[float, float] = (0.45, 0.75)   # 末端密度/峰密度（实测 0.68）
REBAR_DENS_DIP_PROB: float = 0.25                           # 每段（1/8 外露长）稀疏凹陷概率
REBAR_DENS_DIP_FACTOR: tuple[float, float] = (0.12, 0.45)   # 凹陷密度乘子
REBAR_BOW_RANGE: tuple[float, float] = (0.0, 6.0)               # 轻微弯曲矢高 mm（直线度干扰）

# 端头垫片（实测 Ø40~45、长 50、整周可见、标纵筋；v2.1 密度=该筋峰段密度×系数）
WASHER_DIAM_FACTOR: tuple[float, float] = (2.0, 2.4)   # ×筋径
WASHER_LENGTH_RANGE: tuple[float, float] = (30.0, 60.0)
WASHER_DENSITY_FACTOR: tuple[float, float] = (0.8, 1.3)  # ×该筋峰段密度

# ─── 套筒参数 (JG/T 398-2019；实测孔 Ø25~48、环凸 3~10) ───
# {钢筋直径: (套筒外径, 套筒内腔直径, 套筒长度)}
SLEEVE_SPEC_MAP: dict[float, tuple[float, float, float]] = {
    16.0: (48.0, 36.0, 310.0),
    18.0: (50.0, 38.0, 350.0),
    20.0: (52.0, 40.0, 370.0),
    22.0: (54.0, 42.0, 410.0),
    25.0: (58.0, 46.0, 460.0),
    28.0: (62.0, 50.0, 505.0),
    32.0: (66.0, 54.0, 570.0),
}
SLEEVE_RING_PROUD_RANGE: tuple[float, float] = (2.0, 8.0)   # 环凸出端面高度 mm
SLEEVE_RING_HOLE_MARGIN: float = 1.0                        # 孔半径再外扩的挖孔余量 mm
# v2.1：环密度改绝对值（实测端面 0.78 pts/mm²；原 ×基准 的系数随基准抬高失真）
SLEEVE_RING_DENSITY: tuple[float, float] = (0.5, 1.0)       # pts/mm²

# ─── 端面密度 v2.1（实测端面 0.78 pts/mm²，MVS 正对拍摄效应；改绝对值） ──
END_FACE_DENSITY_RANGE: tuple[float, float] = (0.40, 0.90)

# ─── 端面泡沫残留 + 基础凹凸（2026-10-24 用户要求，拆模遗留泡沫建模；
#     实测校准=真实出筋端剔筋后：泡沫块面内 r95 19-49mm、凸出 p50 25/
#     p90 50/max 72mm、凸出≥15mm 约占端面面积 7-12%（其余为垫片区，
#     垫片已有独立机制）；基础凹凸 ~+5mm。凹不低于真实端面 → Δ≥0）──
END_FACE_ROUGH_SIGMA: tuple[float, float] = (3.0, 6.0)   # 基础凹凸半正态 σ mm
FOAM_PROB: float = 0.8                    # 每端面出现泡沫的概率
FOAM_N_RANGE: tuple[int, int] = (3, 10)   # 泡沫块数
FOAM_R_RANGE: tuple[float, float] = (15.0, 50.0)   # 块面内半径 mm
FOAM_H_RANGE: tuple[float, float] = (15.0, 70.0)   # 凸出高度 mm
FOAM_AREA_MAX: float = 0.15               # 泡沫累计面积占端面比例上限

# ─── 结构化缺失（贴地 + 端面空洞） ────────────────────────
GROUND_Z: float = 0.0
GRAZING_DROP_Z: float = 30.0        # 离地 <30mm 的表面点全部缺失
GRAZING_RAMP_Z: float = 80.0        # 30~80mm 概率渐变（0→1）
END_FACE_GAP_PROB: float = 0.5      # 每个端面出现无数据斑的概率（clean 版）
END_FACE_GAP_RADIUS: tuple[float, float] = (60.0, 200.0)   # 斑半径（实测 ~100-200）
END_FACE_MAX_GAPS: int = 1          # clean 版每端面最多 1 个

# ─── 地面背景 ──────────────────────────────────────────────
# v2.1 地面（用户观察修正 2026-10-17）：外边界不规则（径向起伏）、铺到构件
# 接触线（删等宽护城河）、构件贴地或仅高度错开 ±几 mm（扫描误差口径）。
GROUND_MARGIN_RANGE: tuple[float, float] = (250.0, 600.0)    # 构件 bbox 外扩基线
GROUND_DENSITY_FACTOR: tuple[float, float] = (0.6, 1.0)      # 相对基准表面密度
GROUND_ROUGHNESS_RANGE: tuple[float, float] = (0.2, 0.8)     # 地面起伏 σ mm
GROUND_FOOTPRINT_MARGIN: tuple[float, float] = (0.0, 5.0)    # 贴线遮挡（仅构件自身 footprint）
GROUND_HEIGHT_OFFSET: tuple[float, float] = (-4.0, 4.0)      # 构件-地面高度错开 mm
GROUND_EDGE_IRREG: tuple[float, float] = (80.0, 250.0)       # 外边界径向不规则幅度 mm
BG_BLOB_COUNT: tuple[int, int] = (1, 4)                     # 杂物块数
BG_BLOB_SIZE: tuple[float, float] = (80.0, 400.0)           # 杂物块边长 mm
BG_BLOB_DIST: tuple[float, float] = (200.0, 1200.0)         # 距构件距离 mm

# ─── 周向可见性 v2.1（实测真实柱中段周向 36 箱仅 ~10 箱有占） ──
MIDSPAN_VISIBLE_ARC_DEG: tuple[float, float] = (110.0, 240.0)  # 中段可见弧宽
MIDSPAN_ARC_CENTER_JITTER: float = 40.0                        # 可见窗中心自天顶偏移
MIDSPAN_FULL_END_ZONE: float = 400.0                           # 端部全可见区长度 mm

# ─── SfM 低频翘曲 v2.1（实测真实云 ~10cm 量级、波长米级、局部仍 σ<1mm） ──
WARP_SIGMA_RANGE: tuple[float, float] = (12.0, 40.0)
WARP_WAVELENGTH_RANGE: tuple[float, float] = (2500.0, 6000.0)
WARP_N_COMPONENTS: int = 3

# ─── 噪声参数 v2（2026-10-16 按真实云实测精简） ─────────────
# 真实扫描的退化以"缺失"为主（贴地/套筒内/端面斑/半弧——几何层已模拟，
# clean 版即含），噪声层只保留三种：
#   jitter  高斯抖动（实测顶面残差 σ≈0.4mm）
#   keep    随机丢失（扫描密度随机起伏）
#   occlusion 多边形遮挡（邻构件/杂物挡视线；真实两柱相贴即有）
# 均匀噪点/噪点簇/边缘噪点已移除（真实中几乎不存在）。
NOISE_LEVELS: dict[str, dict] = {
    "low": {
        "occlusion_pct_range": (0.0, 0.05),
        "jitter_sigma_range": (0.15, 0.5),
        "keep_pct_range": (0.94, 0.985),
    },
    "medium": {
        "occlusion_pct_range": (0.05, 0.18),
        "jitter_sigma_range": (0.4, 1.2),
        "keep_pct_range": (0.90, 0.96),
    },
}

# ─── 非均匀密度（MVS/TLS 距离效应，±25%） ─────────────────
DENSITY_VARIATION_N_ZONES: tuple[int, int] = (2, 4)
DENSITY_VARIATION_FACTOR_RANGE: tuple[float, float] = (0.75, 1.25)

# ─── 结构性噪点簇（施工粉尘/浮渣，空间聚集） ──────────────
CLUSTER_OUTLIER_N_CLUSTERS: tuple[int, int] = (2, 6)
CLUSTER_OUTLIER_SPREAD: tuple[float, float] = (10.0, 50.0)

# ─── 边缘噪点（混合像素） ─────────────────────────────────
EDGE_NOISE_SEARCH_RADIUS: float = 8.0
EDGE_NOISE_JITTER_SIGMA: float = 1.0
EDGE_NOISE_LABEL_FLIP_PROB: float = 0.08

# ─── 构件几何参数范围 ──────────────────────────────────────
# v2 统一：构件沿 X 平躺、截面在 YZ、底面贴地 z=0
COLUMN_WIDTH_RANGE: tuple[float, float] = (400.0, 800.0)    # 截面宽（y）
COLUMN_DEPTH_RANGE: tuple[float, float] = (400.0, 800.0)    # 截面高（z）
COLUMN_HEIGHT_RANGE: tuple[float, float] = (2000.0, 6000.0) # 长（x）
L_SHAPED_COLUMN_FLANGE_RANGE: tuple[float, float] = (400.0, 700.0)
L_SHAPED_COLUMN_WEB_RANGE: tuple[float, float] = (300.0, 500.0)
L_SHAPED_COLUMN_THICK_RANGE: tuple[float, float] = (150.0, 300.0)

BEAM_WIDTH_RANGE: tuple[float, float] = (300.0, 600.0)
BEAM_HEIGHT_RANGE: tuple[float, float] = (400.0, 800.0)
BEAM_LENGTH_RANGE: tuple[float, float] = (3000.0, 9000.0)

T_BEAM_FLANGE_WIDTH_RANGE: tuple[float, float] = (500.0, 1000.0)   # 底部翼缘宽（倒 T 平放）
T_BEAM_FLANGE_HEIGHT_RANGE: tuple[float, float] = (100.0, 200.0)
T_BEAM_WEB_WIDTH_RANGE: tuple[float, float] = (200.0, 400.0)
T_BEAM_TOTAL_HEIGHT_RANGE: tuple[float, float] = (500.0, 1000.0)

# ─── 箍筋（叠合梁顶面外伸 U 形发夹筋，标 stirrup） ─────────
# 直径=产品规格离散档（2026-10-23 用户拍板，供规格分类头）；height 随机化
# （原常数 110 学不到量，用户定 90–130）
HAIRPIN_DIAM_RANGE: tuple[float, float] = (8.0, 12.0)
HAIRPIN_DIAM_SPECS: tuple[float, ...] = (8.0, 10.0, 12.0)
HAIRPIN_SPACING_RANGE: tuple[float, float] = (150.0, 250.0)
HAIRPIN_HEIGHT_RANGE: tuple[float, float] = (90.0, 130.0)
HAIRPIN_DENSITY_FACTOR: tuple[float, float] = (0.3, 0.7)    # 相对基准（稀疏细筋）

# ─── 剪力墙板（v2.3，用户 2026-10-24：板+中间窗洞+洞口周围钢筋） ──
# 摆放：沿 x=墙宽；截面 yz = 墙高(y)×板厚(z)——大面 z=t 朝上、底面贴地；
# x=0 套筒端 / x=L 出筋端（布局 C 沿用）；窗洞顶面中央、沿厚度贯通。
WALL_HEIGHT_RANGE: tuple[float, float] = (2400.0, 3300.0)   # 截面 y=墙高
WALL_THICK_RANGE: tuple[float, float] = (160.0, 260.0)      # z=板厚
WALL_WIDTH_RANGE: tuple[float, float] = (2400.0, 4200.0)    # x=墙宽
WALL_WIN_X_RANGE: tuple[float, float] = (600.0, 1200.0)     # 窗洞宽（沿 x）
WALL_WIN_Y_RANGE: tuple[float, float] = (600.0, 1500.0)     # 窗洞高（沿 y）
WALL_WIN_EDGE_MIN: float = 500.0                            # 洞边距墙边最小
WALL_EDGE_BAR_DIAMS: tuple[float, ...] = (10.0, 12.0)       # 洞口边筋规格档

# ─── 采样配置 ──────────────────────────────────────────────
CONCRETE_ROUGHNESS: float = 0.4  # 混凝土表面粗糙度 σ mm（实测 0.41）

# ─── 数据集规模（8 类型 × 12 构型 × 2 密度 × 2 噪声 = 384） ──
FILE_PREFIX: str = "Mock"
NUM_VARIANTS_PER_TYPE: int = 12
TRAIN_VAL_TEST_SPLIT: tuple[int, int, int] = (6, 2, 2)  # 按 (type, vidx) 组 6:2:2
