# -*- coding: utf-8 -*-
"""稠密融合云（COLMAP stereo_fusion fused.ply）三视图（原始 RGB，无标号无框）。

输入：out/sfm/prefab_scan/dense/fused.ply（binary_little_endian，x/y/z+法线+RGB）。
产出：out/sfm/prefab_scan/views/dense_views.png（排列二一行三张）
  + dense_panel_a_top / b_front / c_side 单张。
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
from _bootstrap import *   # noqa: F401  路径锚定 + sci_style 装载
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

WS = os.path.join(OUT_ROOT, "..", "sfm", "prefab_scan")
PLY = os.path.join(WS, "dense", "fused.ply")
OUT = os.path.join(WS, "views")
os.makedirs(OUT, exist_ok=True)

# ---- 二进制 PLY 解析（按头部属性表逐点读取）----
with open(PLY, "rb") as f:
    assert f.readline().strip() == b"ply"
    fmt = f.readline().strip()
    assert b"binary_little_endian" in fmt, fmt
    props, n_pts = [], None
    while True:
        line = f.readline().strip()
        if line.startswith(b"element vertex"):
            n_pts = int(line.split()[-1])
        elif line.startswith(b"property"):
            parts = line.split()
            props.append((parts[1].decode(), parts[2].decode()))
        elif line.startswith(b"end_header"):
            break
    dt = np.dtype([(f"{n}_{i}", np.float32 if t == "float" else np.uint8)
                   for i, (t, n) in enumerate(props)])
    data = np.frombuffer(f.read(n_pts * dt.itemsize), dtype=dt, count=n_pts)
names = [f"{n}_{i}" for i, (t, n) in enumerate(props)]
xyz = np.stack([data[names[0]], data[names[1]], data[names[2]]],
               axis=1).astype(np.float64)
rgb = np.stack([data[n] for n in names[-3:]], axis=1).astype(np.float64) / 255.0
print(f"points={n_pts}")
print("cols:", [n for _, n in props])

# 稳健边界：1–99% 分位
lo = np.percentile(xyz, 1.0, axis=0)
hi = np.percentile(xyz, 99.0, axis=0)
keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
xyz, rgb = xyz[keep], rgb[keep]
print(f"after clip: {len(xyz)} (dropped {n_pts - len(xyz)})")

# 渲染量上限（2D rasterized 直绘 200 万以内可接受，超出下采样）
if len(xyz) > 1_200_000:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xyz), size=1_200_000, replace=False)
    xyz, rgb = xyz[idx], rgb[idx]
    print(f"subsampled -> {len(xyz)}")

W_CELL, H_CELL, DPI = 48.0, 34.6, 600
views = [("dense_panel_a_top.png", 0, 1),
         ("dense_panel_b_front.png", 0, 2),
         ("dense_panel_c_side.png", 1, 2)]
panels = []
for name, ci, cj in views:
    fig = plt.figure(figsize=(W_CELL / 25.4, H_CELL / 25.4))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.scatter(xyz[:, ci], xyz[:, cj], s=0.3, c=rgb, marker="o",
               linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xyz[:, ci].min(), xyz[:, ci].max())
    ax.set_ylim(xyz[:, cj].min(), xyz[:, cj].max())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    panels.append(p)

stitched = ss.stitch_panels(panels, os.path.join(OUT, "dense_views.png"), cols=3)
ss.report_size_mm(stitched)
for p in panels:
    im = np.asarray(Image.open(p).convert("L"))
    print(f"{os.path.basename(p)}: {im.shape[1]}x{im.shape[0]}px nonwhite={int((im < 245).sum())}")
print("stitched ->", stitched)
