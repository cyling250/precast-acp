# -*- coding: utf-8 -*-
"""Fig13 消融对照柱状图——英文重绘（数据图规范，离线无 GPU）。

数字来源＝draft_zh 表 4（用户已审 canonical）：
  级联融合/套筒召回    消融(未融合 BEV 头) 0.922 → 完整 0.962
  头选型/榀数全对率    消融(BEV 头)        0.125 → 完整(点级头) 0.958
  认证视图/箍筋召回    消融(默认顶面视图)   0.000 → 完整(认证)   0.915
保留采样项影响高度量测、不入图（题注口径同 draft）。

图内全英文；单图无编号；无大标题。交付：TIFF 600dpi LZW 原件
+ 同名 PNG 预览孪生（数据图规范）。

用法（项目根）：.venv/Scripts/python.exe -I temp/fig13/plot_fig13.py
产出：temp/fig13/Fig13.tiff + Fig13.png（72×65：排列一 72 宽 + 13mm 顶部图例带）
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SKILL = os.path.join(os.environ.get("USERPROFILE", ""), ".zcode",
                     "skills", "scientific-plotting")
sys.path.insert(0, os.path.join(SKILL, "scripts"))

import numpy as np
import sci_style as ss

ss.apply_style()   # 禁手写 rcParams，一律走样式模块

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Patch

OUT_ROOT = os.path.join(ROOT, "out", "figures")
OUT_DIR = os.path.join(OUT_ROOT, "fig13_ablation")
os.makedirs(OUT_DIR, exist_ok=True)

# 消融项目（英文标签两行堆叠；指标名移题注，72mm 宽 9pt 放不下双联标签）
GROUPS = [
    ("Cascade\nfusion", 0.922, 0.962),
    ("Head\nselection", 0.125, 0.958),
    ("Certified\nview", 0.000, 0.915),
]


def main():
    # 用户 2026-11-04 定稿：宽度保持排列一 72mm 不变，顶部加高 13mm
    # 放置图例（技能 top_mm 图例置顶模式），画幅 72×65
    fig, ax = ss.new_figure(72, 65)
    ax.set_position([ss.MARGIN_L_MM / 72, ss.MARGIN_B_MM / 65,
                     1 - (ss.MARGIN_L_MM + ss.MARGIN_R_MM) / 72,
                     1 - (ss.MARGIN_B_MM + 13.0) / 65])   # 顶部 13mm 图例带
    x = np.arange(len(GROUPS), dtype=float)
    w = 0.32
    for k, (dx, color) in enumerate([(-0.19, ss.GRAY), (0.19, ss.MAIN)]):
        vals = [g[1 + k] for g in GROUPS]
        bars = ax.bar(x + dx, vals, width=w, color=color,
                      edgecolor="none", zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [g[0] for g in GROUPS])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 1.14)
    ax.set_yticks((0.0, 0.2, 0.4, 0.6, 0.8, 1.0))   # 显式 0.2 步长
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(plt.NullLocator())    # 分类轴不挂次刻度
    ax.tick_params(axis="x", length=0)               # 分类轴无刻度线
    ax.set_ylabel("Metric value")
    ax.legend(handles=[Patch(fc=ss.GRAY, label="Ablated variant"),
                       Patch(fc=ss.MAIN, label="Full pipeline")],
              loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncols=2, frameon=False, fontsize=7, handlelength=1.4,
              columnspacing=1.2, borderaxespad=0.2)
    fig.savefig(os.path.join(OUT_DIR, "Fig.png"), dpi=ss.DPI,
                facecolor="white")
    ss.save_tiff(fig, os.path.join(OUT_DIR, "Fig.tiff"))
    plt.close(fig)
    ss.report_size_mm(os.path.join(OUT_DIR, "Fig.png"))


if __name__ == "__main__":
    main()
