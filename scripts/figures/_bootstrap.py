# -*- coding: utf-8 -*-
"""出图脚本公共引导：路径锚定 + sci_style 样式装载。各脚本 import 本模块
后再 import matplotlib，保证 apply_style 生效于全部面板。"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SKILL = os.path.join(os.environ.get("USERPROFILE", ""), ".zcode",
                     "skills", "scientific-plotting")
for p in (os.path.join(ROOT, "src"), SKILL and os.path.join(SKILL, "scripts")):
    if p and p not in sys.path:
        sys.path.insert(0, p)

OUT_ROOT = os.path.join(ROOT, "out", "figures")
os.makedirs(OUT_ROOT, exist_ok=True)

import sci_style as ss   # noqa: E402  （引导后统一装载样式）

ss.apply_style()         # 禁手写 rcParams；网格=短划线 0.5pt

# 项目定性图既定配色（Fig7-13 同款）
C_CONC = "#c8c8c8"   # 混凝土/扫描云
C_RB = "#d62728"     # 钢筋（材质染色）
C_ST = "#ff9f1c"     # 箍筋（材质染色）
C_GT = "#222222"     # 真值标记（黑）
C_DET = "#1f77b4"    # 检出标记（蓝）


def subsample(n, cap, seed=0):
    import numpy as np
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, cap, replace=False)
