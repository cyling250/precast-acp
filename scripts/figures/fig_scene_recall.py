# -*- coding: utf-8 -*-
"""堆放工况合并召回柱状图（Visio Figure9 页；draft 旧编号图 9）。

数据＝results/scene_pooled.json（canonical，每工况 3 场景合并池化；
箍筋仅评含叠合梁的 s4、s6）。与正文表 2 对账。

用法（项目根）：.venv/Scripts/python.exe -I scripts/figures/fig_scene_recall.py
产出：out/figures/fig_scene_recall/{Fig.png,Fig.tiff}（72×65）
"""
import json
import os

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from _bootstrap import OUT_ROOT, ROOT, ss

import matplotlib.pyplot as plt   # noqa: E402  （须在 _bootstrap 之后）
from matplotlib.patches import Patch

OUT_DIR = os.path.join(OUT_ROOT, "fig_scene_recall")
os.makedirs(OUT_DIR, exist_ok=True)

MODES = ["s3", "s4", "s5", "s6"]
SERIES = [("钢筋", "Rebar"), ("套筒", "Sleeve"), ("箍筋", "Stirrup")]


def main():
    with open(os.path.join(ROOT, "results", "scene_pooled.json"),
              encoding="utf-8") as f:
        pooled = json.load(f)["modes"]

    def get(entry, kind):
        # 箍筋仅 s4/s6 评测，缺类目返回 None（柱留空，图注说明）
        return (float(entry[kind]["召回"]) if kind in entry else None)

    vals = {kind: [get(pooled[m], kind) for m in MODES] for kind, _ in SERIES}
    for kind, v in vals.items():
        print(kind, v)

    fig, ax = ss.new_figure(72, 65)
    ax.set_position([ss.MARGIN_L_MM / 72, ss.MARGIN_B_MM / 65,
                     1 - (ss.MARGIN_L_MM + ss.MARGIN_R_MM) / 72,
                     1 - (ss.MARGIN_B_MM + 13.0) / 65])   # 顶部 13mm 图例带
    x = np.arange(4, dtype=float)
    w = 0.24
    for k, (kind, label) in enumerate(SERIES):
        dx = (k - 1) * (w + 0.02)
        v = [x_ if x_ is not None else np.nan for x_ in vals[kind]]
        ax.bar(x + dx, v, width=w, color=ss.colors(3)[k],
               edgecolor="none", zorder=3)
    ax.set_xticks(x, MODES)
    ax.tick_params(axis="x", length=0)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0, 1.05)
    ss.cap_axes(ax, ylim=(0, 1.05), cap_x=False, min_ticks=5)
    ax.set_xlabel("Stacking condition")
    ax.set_ylabel("Recall")
    ax.legend(handles=[Patch(fc=ss.colors(3)[k], label=lb)
                       for k, (_, lb) in enumerate(SERIES)],
              loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncols=3, frameon=False, fontsize=7, handlelength=1.4,
              columnspacing=1.2, borderaxespad=0.2)
    png = os.path.join(OUT_DIR, "Fig.png")
    fig.savefig(png, dpi=ss.DPI, facecolor="white")
    ss.save_tiff(fig, os.path.join(OUT_DIR, "Fig.tiff"))
    plt.close(fig)
    ss.report_size_mm(png)


if __name__ == "__main__":
    main()
