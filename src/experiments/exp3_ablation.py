# -*- coding: utf-8 -*-
"""exp3 —— 消融实验（论文消融表；数字已 canonical 入库 results/，重跑
仅作复现核验，允许 P-072 毫级漂移）。

消融轴（与主结果同配置，只变单一变量）：
  face-source  面来源四口径 oracle/pointgrow/pointgrow-pred/classic
               （=单件四口径表本身；canonical results/e2e_fullchain/）
  sleeve       套筒段 raw 裸头 vs fusion 级联融合（0.962 vs 修复后 raw 0.922）
  head         钢筋/箍筋段 point 点级头 vs bev BEV 头（T-039 榀数全对
               0.125→0.958）
  cert-view    箍筋认证视图（滚转构件 st 0→0.915，P-064）：场景链对比，
               用 lib.pipeline.scene_chain --no-roll vs 默认直跑单场景

用法（cwd=src）：
  python -m experiments.exp3_ablation face-source [--limit 20] [--out-dir DIR]
  python -m experiments.exp3_ablation sleeve [--limit 20] [--out-dir DIR]
  python -m experiments.exp3_ablation head [--limit 20] [--out-dir DIR]
  （--out-dir 缺省=只打印不落盘，canonical 防覆盖）
"""
import argparse
import os
import sys


def _run_one(face, sleeve, rbst, limit, out):
    from lib.pipeline import single_chain
    args = single_chain.build_parser().parse_args([])
    args.face, args.sleeve, args.rbst = face, sleeve, rbst
    args.limit, args.out = limit, out
    return single_chain.run_single(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("axis",
                    choices=("face-source", "sleeve", "head"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="",
                    help="结果落盘目录（缺省只打印；canonical 在 results/）")
    args = ap.parse_args()

    from lib.pipeline import single_chain
    parser = single_chain.build_parser()

    if args.axis == "face-source":
        faces = ("oracle", "pointgrow", "pointgrow-pred", "classic")
        for face in faces:
            out = (os.path.join(args.out_dir, f"e2e_fullchain_{face}.json")
                   if args.out_dir else "")
            print(f"\n===== face-source: {face} =====", flush=True)
            _run_one(face, "fusion", "point", args.limit, out)
    elif args.axis == "sleeve":
        for sleeve in ("fusion", "raw"):
            out = (os.path.join(args.out_dir,
                                f"e2e_fullchain_sleeve-{sleeve}.json")
                   if args.out_dir else "")
            print(f"\n===== sleeve: {sleeve} =====", flush=True)
            _run_one("pointgrow-pred", sleeve, "point", args.limit, out)
    elif args.axis == "head":
        for rbst in ("point", "bev"):
            out = (os.path.join(args.out_dir,
                                f"e2e_fullchain_head-{rbst}.json")
                   if args.out_dir else "")
            print(f"\n===== head: {rbst} =====", flush=True)
            _run_one("pointgrow-pred", "fusion", rbst, args.limit, out)


if __name__ == "__main__":
    main()
