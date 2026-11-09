# -*- coding: utf-8 -*-
"""Fig11 底版评选：s4 三个场景（cfg01/02/03）端面配准 + 箍筋 + 滚转角统计。

管线段逐字复刻 scripts/render_qualitative.py::fig_scene（208-276 行语义），
只把画图换成统计；检出/GT 一律来自冻结管线现跑，不改管线本体。

用法（项目根）：  .venv/Scripts/python.exe -I temp/fig11/select_scene.py
产出：temp/fig11/s4_<name>_panels.npz（panels 对象数组，仿 fig10_s3_deploy.npz）
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lib.utils import anchor_project_path

TOL_RB = 30.0   # 出筋端"配准"门限 mm（钢筋直径 25，位置 p50≈3.6）
TOL_SL = 30.0   # 套筒端中心配准门限 mm（内径 46，中心误差 p50≈0.33）

SCENES = ["scene_s4_cfg01_0000", "scene_s4_cfg02_0001", "scene_s4_cfg03_0002"]


def load_models(device):
    from lib.pipeline.real_chain import load_heads
    from lib.segmentation.inference import load_model
    models = load_heads("out/exp1_train/head_bev/best.pth",
                        "out/exp1_train/head_point/best.pth", device)
    seg_model, seg_dev = load_model()
    return models, seg_model, seg_dev


def nn_stats(gt, det, tol):
    """最近邻配准统计：每个 GT 到最近检出的距离、每个检出是否落在门限内。"""
    gt = np.asarray(gt, np.float64).reshape(-1, 2)
    det = np.asarray(det, np.float64).reshape(-1, 2)
    if len(gt) == 0 or len(det) == 0:
        return dict(n_gt=len(gt), n_det=len(det), gt_nn=[],
                    det_far=[], matched=0)
    d = np.linalg.norm(gt[:, None, :] - det[None, :, :], axis=-1)
    gt_nn = d.min(axis=1)
    det_nn = d.min(axis=0)
    return dict(n_gt=len(gt), n_det=len(det),
                gt_nn=gt_nn.tolist(),
                det_far=det[det_nn > tol].tolist(),
                matched=int((gt_nn <= tol).sum()))


def roll_deg_of(pose_R, diag=False):
    """滚转角：世界竖直轴在构件截面系内的方位（相对构件 height 轴）。

    行向量约定下（p_world = p_local @ R.T + t，scene_chain.member_frame 同款），
    R 行 = 局部轴的世界系方向，世界→局部 = R @ v。滚转 =
    atan2(竖直轴局部 y 分量, 局部 z 分量)——0° 表示构件 height 轴正对天。
    diag=True 时打印两种约定与各局部轴的天顶分量，供人工核对。"""
    R = np.asarray(pose_R, np.float64)
    ez = np.array([0.0, 0.0, 1.0])
    up_row = R @ ez      # 假设 R 行=局部轴（scene_chain 约定）
    up_col = R.T @ ez    # 假设 R 列=局部轴
    if diag:
        print(f"    [diag] R 行·ez={np.round(R @ ez, 3)}  "
              f"R 列·ez={np.round(R.T @ ez, 3)}", flush=True)
    up_l = up_row
    return float(np.degrees(np.arctan2(up_l[1], up_l[2])))


def run_scene(name, models, device):
    from sklearn.cluster import DBSCAN
    from lib.pipeline.scene_chain import (crop_member_best, member_frame,
                                          member_meta, _fix_end_flip)
    from lib.e2e.face_pointgrow import extract_faces
    from lib.pipeline.real_chain import CLUSTER_EPS, CLUSTER_MIN, MIN_CLUSTER_N
    from lib.pipeline.single_chain import run_one

    scene_npz = anchor_project_path(f"mock/scene/v1/s4/{name}.npz")
    z = np.load(scene_npz, allow_pickle=True)
    vis = z["visible"].astype(bool)
    pts = z["points"][vis].astype(np.float32)
    member_id = z["member_id"][vis].astype(np.int64)
    mts = json.loads(str(z["member_table_json"]))
    lbls = np.load(anchor_project_path(
        f"out/cache/scene_batch/s4_{name}_labels.npz"))["lbls"]
    print(f"[{name}] {len(pts)} 点，{len(mts)} 构件 GT", flush=True)

    # ---- 分簇归组（与 fig_scene 同序）----
    rb_idx = np.flatnonzero(lbls == 2)
    rb = pts[rb_idx]
    cl = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN).fit(rb)
    by_mid = {}
    for k in np.unique(cl.labels_):
        if k >= 0 and (cl.labels_ == k).sum() >= MIN_CLUSTER_N:
            gidx = rb_idx[cl.labels_ == k]
            gv = member_id[gidx]
            gv = gv[gv >= 0]
            if not len(gv):
                continue
            mid = int(np.bincount(gv, minlength=len(mts)).argmax())
            by_mid.setdefault(mid, []).append(gidx)
    c_center = pts.mean(0)

    panels, report = [], []
    for mid in sorted(by_mid):
        gidx_list = by_mid[mid]
        got = crop_member_best(pts, lbls, c_center,
                               [pts[g] for g in gidx_list], gidx_list)
        if got is None:
            report.append(f"  m{mid}: crop 失败")
            continue
        (sub, lbls_sub, _widx, xa, _tf, _tt, _n), _faces0, info = got
        fr = member_frame(sub, xa)
        if fr is None:
            report.append(f"  m{mid}: member_frame 失败")
            continue
        p0, R, c0 = fr
        faces = extract_faces(p0, lbls_sub, axis_hint=(1.0, 0.0, 0.0))
        if faces is None:
            report.append(f"  m{mid}: extract_faces 失败")
            continue
        faces0 = _fix_end_flip(faces, p0, lbls_sub,
                               pts[np.unique(np.concatenate(gidx_list))],
                               c0, xa)
        # 箍筋面认证（逐字对齐 scene_chain.run_member 607-621，M4②）：
        # 侧面钢点 ≥500 且 >2.5×顶面才认证为箍筋面——滚转件判别视图
        stf = faces0.get("stface")
        cert = (stf is not None and stf["kind"] == "side"
                and stf["n_st"] >= 500.0
                and stf["n_st"] > 2.5 * float(stf.get("top_n_st") or 0.0))
        faces = faces0
        if cert:
            faces = dict(faces0)
            faces["top"] = {"n3": np.asarray(stf["n3"], np.float64),
                            "d": float(stf["d"])}
            print(f"  [认证] 箍筋面=侧面 n3={faces['top']['n3'].round(2)} "
                  f"钢点 {stf['n_st']}/顶 {stf.get('top_n_st')}", flush=True)
        mt = mts[mid]
        meta = member_meta(mt, R, c0)
        x_rb = float(faces["rebar"]["c3"][0])
        x_sl = float(faces["sleeve"]["c3"][0])
        m1 = (lbls_sub == 2) | ((lbls_sub == 1)
                                & (np.abs(p0[:, 0] - x_rb) < 60.0))
        m2 = (lbls_sub == 1) & (np.abs(p0[:, 0] - x_sl) < 60.0)
        top = faces.get("top")
        m3 = None
        if top is not None:
            n3t = np.asarray(top.get("n3", (0.0, 0.0, 1.0)), np.float64)
            nn = n3t[1:] / max(float(np.linalg.norm(n3t[1:])), 1e-9)
            s = p0[:, 1] * nn[0] + p0[:, 2] * nn[1] - float(top["d"])
            m3 = ((lbls_sub == 1) | (lbls_sub == 3)) & (s > -60.0)
        res = run_one(p0, None, faces, models, device, face_pts=(m1, m2, m3),
                      conc_mask=(lbls_sub == 1) | (lbls_sub == 2),
                      sl_fuse=True, rbst="point", st_steel=(lbls_sub == 3))
        if res is None or res.get("face_fail"):
            report.append(f"  m{mid}: run_one 失败")
            continue

        # ---- 统计 ----
        gt_yz = np.asarray(meta["rebar_positions_yz"], np.float32)
        det_rb = np.asarray(res["rebar"]["uv_world"], np.float32)
        rb_st = nn_stats(gt_yz, det_rb, TOL_RB)
        sls = meta.get("sleeve_gt") or []
        gt_sl = [g["center_yz"] for g in sls]
        det_sl = np.asarray(res["sleeve"]["uv_world"], np.float32)
        sl_st = nn_stats(gt_sl, det_sl, TOL_SL)

        # 箍筋：腿对配对 + 榀级贪心（语义同 eval_one 519-532 行）
        st = res.get("stirrup")
        hps = meta.get("hairpin_gt") or []
        n_pair, leg_lines = 0, []
        if st is not None and hps:
            dets = np.asarray(st["dets"], np.float32).reshape(-1, 3)
            x0, yc = float(st["x0"]), float(st["yc"])
            ev2 = st.get("ev2") or (1.0, 0.0)
            order = np.argsort(dets[:, 0]) if len(dets) else []
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
            n_pair = len(pairs)
            # member_meta 已把 GT 转到质心系（scene 链 st_off=(0,0) 同口径）
            gt_cx = [((float(g["p_foot_l"][0]) + float(g["p_foot_r"][0])) / 2
                      - x0) for g in hps]
            det_cx = [(dets[a, 0] + dets[k, 0]) / 2 for a, k in pairs]
            usedg = np.zeros(len(hps), bool)
            for pi in np.argsort(det_cx):
                c = det_cx[pi]
                if not len(gt_cx):
                    break
                j = int(np.argmin(np.abs(np.asarray(gt_cx) - c)))
                if usedg[j] or abs(gt_cx[j] - c) > 60.0:
                    continue
                usedg[j] = True
                g = hps[j]
                a, k = pairs[pi]

                def _leg_v(py, pz):
                    return py * ev2[0] + pz * ev2[1] - yc

                gl = sorted([(float(g["p_foot_l"][0]) - x0,
                              _leg_v(float(g["p_foot_l"][1]),
                                     float(g["p_foot_l"][2]))),
                             (float(g["p_foot_r"][0]) - x0,
                              _leg_v(float(g["p_foot_r"][1]),
                                     float(g["p_foot_r"][2])))])
                dl = sorted([(float(dets[a, 0]), float(dets[a, 1])),
                             (float(dets[k, 0]), float(dets[k, 1]))])
                dd = max(float(np.hypot(dx - gx, dy - gy))
                         for (dx, dy), (gx, gy) in zip(dl, gl))
                leg_lines.append(dd)

        roll = roll_deg_of(mt["pose_R"], diag=(mid == sorted(by_mid)[0]))
        panels.append({"mid": mid, "info": info, "p0": p0, "lbls_sub": lbls_sub,
                       "meta": meta, "res": res, "faces": faces,
                       "c0": np.asarray(c0, np.float32),
                       "roll_deg": np.float32(roll), "cert": bool(cert)})
        worst_rb = max(rb_st["gt_nn"]) if rb_st["gt_nn"] else float("nan")
        worst_sl = max(sl_st["gt_nn"]) if sl_st["gt_nn"] else float("nan")
        report.append(
            f"  m{mid} ({info['tag']}) 滚转{roll:+7.1f}° | "
            f"出筋端 检出{rb_st['n_det']:2d} vs GT{rb_st['n_gt']:2d} "
            f"配准{rb_st['matched']:2d} 最差GT距{worst_rb:6.1f}mm "
            f"游离检出{len(rb_st['det_far'])} | "
            f"套筒端 检出{sl_st['n_det']:2d} vs GT可见{sl_st['n_gt']:2d} "
            f"配准{sl_st['matched']:2d} 最差GT距{worst_sl:5.1f}mm "
            f"游离检出{len(sl_st['det_far'])} | "
            f"箍筋 检出榀{n_pair} vs GT{len(hps)} 腿距max"
            f"{max(leg_lines) if leg_lines else float('nan'):.1f}mm")
        print(report[-1], flush=True)

    out = os.path.join(ROOT, "out", "cache", f"fig11_s4_{name}_panels.npz")
    np.savez(out, panels=np.array(panels, dtype=object))
    print(f"[{name}] panels → {out}", flush=True)
    return panels, report


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    models, seg_model, seg_dev = load_models(torch.device(dev))
    for name in SCENES:
        run_scene(name, models, torch.device(dev))
        print("-", flush=True)


if __name__ == "__main__":
    main()
