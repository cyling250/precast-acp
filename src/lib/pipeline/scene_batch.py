# -*- coding: utf-8 -*-
"""场景级批量评估 + 工况池化指标（论文场景表与回归的权威入口）。

验证口径（用户 2026-10-29 定稿）：**不设任何单构件/单实例判据**——
每个工况（s3/s4/s5/s6）把全部构件的 GT 合并成一个大集合只算一次，
最终验证=真实场景池化数字。逐构件数字只作调试诊断，不进任何验收表。

用法（cwd=src）：论文入口=experiments.exp2_validate scene / pooled-only；
本模块保留同参 CLI 作调试直入口：
  # 批量跑（默认 s3/s4/s5/s6 每工况每 cfg 一件 ×3）+ 池化
  python -m lib.pipeline.scene_batch
  # 只对已有 json 重新池化（不跑推理）
  python -m lib.pipeline.scene_batch --pooled-only
  # 扩大样本
  python -m lib.pipeline.scene_batch --per-mode 8 --modes s3,s4,s6

确定性：分割标签缓存（--out 同名 _labels.npz）存在则不重跑分割；
检测头 GPU 推理存在非确定性（cuDNN 归约顺序），复跑会有毫级/个别
检出翻转——论文/交接引用的 canonical 数字以 results/ 入库 json 为准。
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

from lib.utils import anchor_project_path

MODES = ("s3", "s4", "s5", "s6")


# ── 池化聚合（口径的唯一权威实现）─────────────────────────────────────
def _p50(xs):
    return round(float(np.percentile(xs, 50)), 2) if xs else None


def _p90(xs):
    return round(float(np.percentile(xs, 90)), 2) if xs else None


def pool_evals(member_dicts, res_dicts):
    """把逐构件 eval dict（可跨场景）池化成一行指标。

    member_dicts: eval dict 列表（eval_one 产物，含 rb_*/sl_*/st_*）；
    res_dicts:    对应的 res dict 列表（rb_n/sl_n 检出数）。"""
    rb_tp = sum(d.get("rb_match", 0) for d in member_dicts)
    rb_gt = sum(d.get("rb_gt", 0) for d in member_dicts)
    rb_det = sum(r.get("rb_n", 0) for r in res_dicts)
    sl_tp = sum(d.get("sl_match", 0) for d in member_dicts)
    sl_gt = sum(d.get("sl_total_gt", 0) for d in member_dicts)
    sl_det = sum(r.get("sl_n", 0) for r in res_dicts)
    rb_pos = [e for d in member_dicts for e in d.get("rb_pos", [])]
    rb_diam = [abs(e) for d in member_dicts
               for e in d.get("rb_diam_err", [])]
    rb_expo = [e for d in member_dicts for e in d.get("rb_expo_err", [])]
    sl_pos = [e for d in member_dicts for e in d.get("sl_pos", [])]
    sl_id = [abs(e) for d in member_dicts
             for e in d.get("sl_diam_err", [])]
    st_tp = sum(d.get("st_match", 0) for d in member_dicts)
    st_gt = sum(d.get("st_total_gt", 0) for d in member_dicts)
    st_leg = [e for d in member_dicts for e in d.get("st_leg_err", [])]
    st_h = [e for d in member_dicts for e in d.get("st_h_err", [])]
    row = {
        "钢筋": {
            "召回": round(rb_tp / rb_gt, 3) if rb_gt else None,
            "误检率": round((rb_det - rb_tp) / rb_det, 3) if rb_det else None,
            "位置p50": _p50(rb_pos), "位置p90": _p90(rb_pos),
            "直径误差p50": _p50(rb_diam), "外露误差p50": _p50(rb_expo),
            "TP": rb_tp, "GT": rb_gt},
        "套筒": {
            "召回": round(sl_tp / sl_gt, 3) if sl_gt else None,
            "误检率": round((sl_det - sl_tp) / sl_det, 3) if sl_det else None,
            "位置p50": _p50(sl_pos), "ID误差p50": _p50(sl_id),
            "TP": sl_tp, "GT可见": sl_gt},
    }
    if st_gt:
        row["箍筋"] = {
            "召回": round(st_tp / st_gt, 3),
            "腿位p50": _p50(st_leg), "高度p50": _p50(st_h),
            "TP": st_tp, "GT": st_gt}
    return row


def pool_scene_json(paths):
    """读一批 scene_fullchain json，返回 (evals, res)。"""
    evals, ress = [], []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        for m in d.get("per_file", []):
            if m.get("skip") or "eval" not in m:
                continue
            evals.append(m["eval"])
            ress.append(m.get("res") or {})
    return evals, ress


# ── 批量执行 ─────────────────────────────────────────────────────────
def pick_scenes(modes, per_mode):
    """确定性选场：每个 cfg 取一件（cfg 内取排序最小者），取前 per_mode
    个 cfg。per_mode=3 时恰好复现 canonical 场景集
    （cfg01_0000 / cfg02_0001 / cfg03_0002）。"""
    out = []
    for mode in modes:
        files = sorted(glob.glob(anchor_project_path(
            f"mock/scene/v1/{mode}/*.npz")))
        cfgs = {}
        for f in files:
            base = os.path.basename(f).replace(".npz", "")
            cfg = base.split("_")[2]            # scene_<mode>_<cfgNN>_<seed>
            cfgs.setdefault(cfg, base)
        for cfg in sorted(cfgs)[:per_mode]:
            out.append((mode, cfgs[cfg]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--per-mode", type=int, default=3)
    ap.add_argument("--pooled-only", action="store_true",
                    help="只对已有 json 重新池化，不跑推理")
    ap.add_argument("--out-dir", default="results/scene_batch")
    ap.add_argument("--summary", default="results/scene_pooled.json")
    ap.add_argument("--label-cache-dir", default="out/cache/scene_batch",
                    help="分割标签缓存目录（透传 scene_chain）")
    args = ap.parse_args(argv)
    modes = [m for m in args.modes.split(",") if m]
    scenes = pick_scenes(modes, args.per_mode)
    out_dir = anchor_project_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not args.pooled_only:
        py = sys.executable
        for mode, name in scenes:
            out_json = os.path.join(out_dir, f"{mode}_{name}.json")
            cmd = [py, "-m", "lib.pipeline.scene_chain",
                   "--scene", f"mock/scene/v1/{mode}/{name}.npz",
                   "--out", os.path.relpath(out_json,
                                            anchor_project_path(".")),
                   "--label-cache-dir", args.label_cache_dir]
            t0 = time.time()
            r = subprocess.run(cmd, cwd=anchor_project_path("src"),
                               capture_output=True, text=True, timeout=3600)
            print(f"[{mode}] {name} rc={r.returncode} "
                  f"{time.time()-t0:.0f}s", flush=True)
            if r.returncode != 0:
                print(r.stderr[-600:], flush=True)

    # 池化（按工况 + 总体）
    summary = {"modes": {}, "overall": None, "scenes": {}}
    all_evals, all_ress = [], []
    for mode in modes:
        paths = sorted(glob.glob(os.path.join(out_dir, f"{mode}_*.json")))
        evals, ress = pool_scene_json(paths)
        summary["modes"][mode] = pool_evals(evals, ress)
        summary["scenes"][mode] = [os.path.basename(p) for p in paths]
        all_evals += evals
        all_ress += ress
    summary["overall"] = pool_evals(all_evals, all_ress)
    sp = anchor_project_path(args.summary)
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary["modes"], ensure_ascii=False, indent=1))
    print(f"→ {sp}")


if __name__ == "__main__":
    main()
