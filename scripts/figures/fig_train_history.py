# -*- coding: utf-8 -*-
"""Fig8 检测头训练收敛过程（数据图，规范同 Fig9-15）。

数据＝out/exp1_train/{head_point,head_bev}/history.json 的验证指标快照
（每 5 轮记录一次）：
  (a) 钢筋点级头 val recall（60 轮）＋ 套筒柱化头 val recall（30 轮）
  (b) 箍筋点级头 val 腿位/高度误差 p50（mm，60 轮）
分割网络（PointMLP）训练时未落盘 history 日志，收敛曲线待评测分块集
重建后随重训补入（见 draft_zh 待办③），故本图暂不含分割曲线。

图内全英文；两面板排列一并排（总幅 14.4×5.2cm）；先子图后拼接。
用法（项目根）：.venv/Scripts/python.exe -I scripts/figures/fig_train_history.py
产出：out/figures/fig_train_history/{Fig8.tiff,Fig8.png}
      → 复制到 manuscript/figures/ 覆盖同名占位
"""
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import OUT_ROOT, ROOT, ss

import matplotlib.pyplot as plt   # noqa: E402  （须在 _bootstrap 之后）

OUT_DIR = os.path.join(OUT_ROOT, "fig_train_history")
PANEL_DIR = os.path.join(OUT_DIR, "panels")
os.makedirs(PANEL_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(ROOT, "out", "exp1_train")


def load_history(name):
    path = os.path.join(TRAIN_DIR, name, "history.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def panel_a(path):
    """验证召回：钢筋点级头（60 轮）＋套筒柱化头（30 轮）。"""
    pt = load_history("head_point")
    bv = load_history("head_bev")
    ss.apply_style()
    fig, ax = ss.new_figure()
    ax.plot([e["epoch"] for e in pt], [e["rebar"]["recall"] for e in pt],
            color=ss.MAIN, lw=ss.LINE_W, marker="o", markersize=3,
            label="Rebar point head")
    ax.plot([e["epoch"] for e in bv], [e["sleeve"]["recall"] for e in bv],
            color=ss.GRAY, lw=ss.LINE_W, marker="s", markersize=3,
            label="Sleeve columnized head")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val recall")
    ax.legend(loc="lower right")
    ss.cap_axes(ax, xlim=(0, 60), ylim=(0.8, 1.0))
    ss.label_panels([ax], ["(a)"])
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)


def panel_b(path):
    """验证量测误差：箍筋点级头腿位与高度 p50。"""
    pt = load_history("head_point")
    ss.apply_style()
    fig, ax = ss.new_figure()
    ax.plot([e["epoch"] for e in pt], [e["stirrup"]["leg_pos_p50"] for e in pt],
            color=ss.MAIN, lw=ss.LINE_W, marker="o", markersize=3,
            label="Leg position p50")
    ax.plot([e["epoch"] for e in pt], [e["stirrup"]["height_p50"] for e in pt],
            color=ss.GRAY, lw=ss.LINE_W, marker="s", markersize=3,
            label="Height p50")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val error (mm)")
    ax.legend(loc="upper right")
    ss.cap_axes(ax, xlim=(0, 60), ylim=(2, 6))
    ss.label_panels([ax], ["(b)"])
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)


def main():
    pa = os.path.join(PANEL_DIR, "panel_a.png")
    pb = os.path.join(PANEL_DIR, "panel_b.png")
    panel_a(pa)
    panel_b(pb)

    out_tiff = os.path.join(OUT_DIR, "Fig8.tiff")
    out_png = os.path.join(OUT_DIR, "Fig8.png")
    ss.stitch_rows([[pa, pb]], out_tiff)   # 排列一两图并排 14.4×5.2cm
    ss.stitch_rows([[pa, pb]], out_png)    # MD 预览孪生
    ss.report_size_mm(out_tiff)

    for ext in ("png", "tiff"):
        src = os.path.join(OUT_DIR, f"Fig8.{ext}")
        dst = os.path.join(ROOT, "manuscript", "figures", f"Fig8.{ext}")
        shutil.copyfile(src, dst)
        print("copied ->", dst)


if __name__ == "__main__":
    main()
