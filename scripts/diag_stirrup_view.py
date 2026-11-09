# -*- coding: utf-8 -*-
"""T-040 箍筋修复·视图级探针：同一件上 pred 带 vs GT 带跑真实箍筋头。

对每件打印：top_view 的窗参数、带点 z 分布、GT 脚 z 范围 vs 视图 z 门槛、
两种带下的检出数与 GT 脚匹配数——把故障定位到
（a）top_d 偏高切掉视图 （b）带点成分（分割洞）破坏检测
（c）检出存在但匹配失败。

用法（cwd=src）：
  python -m scripts.diag_stirrup_view piece1.npz ...
"""
import json
import sys

import numpy as np
import torch

from lib.utils import anchor_project_path
from lib.e2e.model_pillar import STIRRUP_DIAM_TABLE
from lib.e2e.point_head import detect, make_seeds
from lib.pipeline.single_chain import top_view
from lib.pipeline.real_chain import load_heads

SEED_VOX_TOP = 20.0


def match_frames(dets, hps, ox, oy, x0, yc):
    """eval_one 同款榀级匹配：腿配对（|Δx|<30）→ 榀中心贪心 ≤60mm。"""
    if not len(dets):
        return 0
    order = np.argsort(dets[:, 0])
    pairs, used = [], np.zeros(len(dets), bool)
    for a in range(len(order)):
        if used[order[a]]:
            continue
        for b in range(a + 1, len(order)):
            k = order[b]
            if not used[k] and abs(dets[k, 0] - dets[order[a], 0]) < 30.0:
                pairs.append((order[a], k))
                used[order[a]] = used[k] = True
                break
    gt_cx = [(float(g["p_foot_l"][0]) + float(g["p_foot_r"][0])) / 2
             - ox - x0 for g in hps]
    det_cx = [(dets[a, 0] + dets[k, 0]) / 2 for a, k in pairs]
    usedg = np.zeros(len(hps), bool)
    n = 0
    for pi in np.argsort(det_cx):
        if not len(gt_cx):
            break
        j = int(np.argmin(np.abs(np.asarray(gt_cx) - det_cx[pi])))
        if usedg[j] or abs(gt_cx[j] - det_cx[pi]) > 60.0:
            continue
        usedg[j] = True
        n += 1
    return n


def probe(path, model_st, device):
    from lib.segmentation.inference import load_model, predict_labels
    from lib.e2e.face_sem import face_band_labels
    d = np.load(anchor_project_path(path), allow_pickle=False)
    pts = d["points"].astype(np.float32)
    lbls = d["labels"].astype(np.int64)
    meta = json.loads(str(d["meta_json"]))
    hps = meta.get("hairpin_gt") or []
    if not hps:
        print(f"{path}: 无箍筋 GT")
        return
    c0 = pts.mean(0)
    top_z_gt = float(hps[0]["top_z"])
    feet_z = [float(f[2]) for g in hps for f in (g["p_foot_l"], g["p_foot_r"])]
    print(f"\n===== {path.rsplit('/', 1)[-1].rsplit(chr(92), 1)[-1]} "
          f"({len(hps)} 榀) =====")
    print(f"GT 脚 z 范围: [{min(feet_z):.0f}, {max(feet_z):.0f}] "
          f"(c0z={c0[2]:.0f})")

    seg_model, seg_dev = load_model()
    lbls_pred = predict_labels(pts, seg_model, seg_dev)
    rng = np.random.default_rng(0)

    for tag, lb, top_d in (("GT带+GT顶", lbls, top_z_gt - c0[2]),
                           ("pred带+GT顶", lbls_pred, top_z_gt - c0[2])):
        lab = face_band_labels(pts, lb, meta)
        pq = pts[lab == 3]
        tvw = top_view(pq, top_d, rng)
        if tvw is None:
            print(f"[{tag}] top_view None（带点<100）")
            continue
        tv, (x0, yc) = tvw
        m_in = pq[:, 2] - top_d > -50.0
        p_det = np.column_stack([tv[:, 2], tv[:, 0], tv[:, 1]])
        seeds, nb, extra, nb2 = make_seeds(p_det, seed_vox=SEED_VOX_TOP,
                                           rng=rng)
        dets = detect(model_st, seeds, nb, extra, nb2, STIRRUP_DIAM_TABLE,
                      thr=0.5, device=device, pts_uv=p_det[:, 1:3])
        n_frame = match_frames(dets[:, :3], hps, c0[0], c0[1], x0, yc)
        zq = np.percentile(pq[:, 2] - top_d, [0, 50, 99]) \
            if len(pq) else (0, 0, 0)
        print(f"[{tag}] 带点 {len(pq)}（入窗 {int(m_in.sum())}）z分位"
              f"[{zq[0]:.0f},{zq[1]:.0f},{zq[2]:.0f}] 窗x0={x0:.0f} "
              f"yc={yc:.0f} → 检出 {len(dets)}, 榀匹配 {n_frame}/{len(hps)}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_rb, model_sl, model_st = load_heads("out/exp1_train/head_bev/best.pth",
                                              "out/exp1_train/head_point/best.pth",
                                              device)
    for path in sys.argv[1:]:
        try:
            probe(path, model_st, device)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{path}: 探针失败 {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
