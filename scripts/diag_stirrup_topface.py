# -*- coding: utf-8 -*-
"""T-040 箍筋修复·单件解剖：pred 口径下复合梁箍筋整件归零断在哪层。

对失败件与健康对照件逐层打印：
①extract_faces(pred 标签) 的 top face 是否 None（run_one 无 top 直接
  跳过箍筋段——整件归零的头号嫌疑）；
②顶面带（GT top_z−60 以上）的分类混淆：GT 混凝土/箍筋点被预测成了什么；
③m3 带点量 pred vs GT（face_band_labels 同规则换标签源）；
④top 存在时其 d 与 GT top_z 的偏差。

用法（cwd=src）：
  python -m scripts.diag_stirrup_topface piece1.npz piece2.npz ...
"""
import json
import sys

import numpy as np
import torch

from lib.utils import anchor_project_path


def dissect(path):
    from lib.segmentation.inference import load_model, predict_labels
    from lib.e2e.face_pointgrow import extract_faces
    from lib.e2e.face_sem import face_band_labels
    d = np.load(anchor_project_path(path), allow_pickle=False)
    pts = d["points"].astype(np.float32)
    lbls = d["labels"].astype(np.int64)
    meta = json.loads(str(d["meta_json"]))
    c0 = pts.mean(0)
    p = pts - c0
    print(f"\n===== {path.rsplit('/', 1)[-1].rsplit(chr(92), 1)[-1]} =====")
    hps = meta.get("hairpin_gt") or []
    if not hps:
        print("无箍筋 GT → 跳过")
        return
    top_z_gt = float(hps[0]["top_z"])          # 世界系
    n_frame = len(hps)
    print(f"GT: {n_frame} 榀, top_z={top_z_gt:.0f}")

    seg_model, seg_dev = load_model()
    lbls_pred = predict_labels(pts, seg_model, seg_dev)

    # ① pred 顶面
    faces = extract_faces(p, lbls_pred)
    if faces is None:
        print("① extract_faces 整体失败（face_fail）")
        return
    top = faces.get("top")
    if top is None:
        print("① pred top face: None ← 整件归零直接原因")
    else:
        print(f"① pred top face: d={top['d']:.1f}  "
              f"偏差(top d−GT top_z+c0z)={top['d'] - (top_z_gt - c0[2]):+.1f} mm")
    # 端面健康度参照（rb/sl 能跑通的原因）
    for k in ("rebar", "sleeve"):
        f = faces.get(k)
        if f is not None:
            print(f"   {k} 面 d={f['d']:.0f} n3={np.round(f['n3'], 2)}")

    # ② 顶面带分类混淆（世界系 z > top_z−60；GT 视角看 pred 标签）
    band = pts[:, 2] > top_z_gt - 60.0
    print(f"② 顶面带 {int(band.sum())} 点，GT类→pred类混淆：")
    for gt_cls, name in ((1, "GT混凝土"), (3, "GT箍筋")):
        m = band & (lbls == gt_cls)
        if not m.sum():
            continue
        pc = np.bincount(lbls_pred[m], minlength=4)
        print(f"   {name} {int(m.sum())} → pred[bg {pc[0]} 混凝土 {pc[1]} "
              f"钢筋 {pc[2]} 箍筋 {pc[3]}]"
              f" 保留率 {((lbls_pred[m] == gt_cls).mean()):.2f}")

    # ③ m3 带点量（同规则 face_band_labels，换标签源）
    for tag, lb in (("GT", lbls), ("pred", lbls_pred)):
        lab = face_band_labels(pts, lb, meta)
        print(f"③ m3 带点[{tag}] = {int((lab == 3).sum())}")

    # ④ 混凝土点 z 分布（几何回退可行性：类无关顶面层是否存在）
    con_p = p[lbls_pred == 1]
    if len(con_p):
        z99 = float(np.percentile(con_p[:, 2], 99.5))
        skin = con_p[np.abs(con_p[:, 2] - z99) < 15.0]
        print(f"④ pred混凝土最高层: z99.5={z99:.0f} (GT top_z−c0z="
              f"{top_z_gt - c0[2]:.0f}), 顶皮带 {len(skin)} 点")
    all_p = p
    z99a = float(np.percentile(all_p[:, 2], 99.7))
    skina = all_p[np.abs(all_p[:, 2] - z99a) < 12.0]
    print(f"   类无关最高层: z99.7={z99a:.0f}, 顶皮带 {len(skina)} 点")


def main():
    seg_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = seg_dev
    for path in sys.argv[1:]:
        try:
            dissect(path)
        except Exception as e:
            print(f"{path}: 解剖失败 {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
