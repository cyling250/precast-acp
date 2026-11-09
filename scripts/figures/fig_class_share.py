# -*- coding: utf-8 -*-
"""逐类点数占比图（Visio Figure6 页 (a)；draft 旧编号图 5(a)）。

数据＝mock/inuse/test 64 件逐类点数占比。标签序按生成器
CLASS_NAMES = (non_member, concrete, rebar, stirrup)：
Background 42.9% / Concrete 55.5% / Rebar 1.3% / Stirrup 0.2%。
对数 y 轴，柱顶百分比标注。

用法（项目根）：.venv/Scripts/python.exe -I scripts/figures/fig_class_share.py
产出：out/figures/fig_class_share/{Fig.png,Fig.tiff}（72×52 排列一）
"""
import glob
import os

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from _bootstrap import OUT_ROOT, ROOT, ss

import matplotlib.pyplot as plt   # noqa: E402  （须在 _bootstrap 之后）

OUT_DIR = os.path.join(OUT_ROOT, "fig_class_share")
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ["Background", "Concrete", "Rebar", "Stirrup"]
COLORS = [ss.GRAY, ss.MAIN, ss.BLUE, ss.GREEN]


def main():
    cnt = np.zeros(4, np.int64)
    for f in sorted(glob.glob(os.path.join(ROOT, "mock", "inuse",
                                           "test", "*.npz"))):
        cnt += np.bincount(np.load(f)["labels"], minlength=4)
    pct = cnt / cnt.sum() * 100.0
    print("counts:", cnt.tolist(), "pct:", pct.round(2).tolist())

    fig, ax = ss.new_figure(72, 52)
    x = np.arange(4)
    ax.bar(x, cnt, width=0.62, color=COLORS, edgecolor="none", zorder=3)
    for xi, (c, p) in enumerate(zip(cnt, pct)):
        ax.text(xi, c * 1.35, f"{p:.1f}%", ha="center", va="bottom",
                fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(1e5, 1.5e9)
    ax.set_xticks(x, CLASSES)
    ax.tick_params(axis="x", length=0)               # 分类轴无刻度线
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel("Points per class")
    fig.savefig(os.path.join(OUT_DIR, "Fig.png"), dpi=ss.DPI,
                facecolor="white")
    ss.save_tiff(fig, os.path.join(OUT_DIR, "Fig.tiff"))
    plt.close(fig)
    ss.report_size_mm(os.path.join(OUT_DIR, "Fig.png"))


if __name__ == "__main__":
    main()
