# -*- coding: utf-8 -*-
"""exp6 —— 效率表（工程应用论文标配；纯计时/计参，非新验证线）。

产出 results/efficiency.json：
  params    三检测头（点级 rb/st、BEV sl）+ 场景分割 的参数量
  latency   单构件全链耗时（部署口径 pointgrow-pred，--limit 件均值，
            含分割/生长面/三头/融合）
硬件口径：RTX 4060 Ti 8GB（论文实验环境）。

用法（cwd=src）：
  python -m experiments.exp6_efficiency [--limit 20] [--out results/efficiency.json]
"""
import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20,
                    help="计时件数（0=不跑计时只出参数量）")
    ap.add_argument("--out", default="results/efficiency.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def n_params(m):
        return int(sum(p.numel() for p in m.parameters()))

    params = {}
    from lib.pipeline import real_chain, single_chain
    model_rb, model_sl, model_st = real_chain.load_heads(
        "out/exp1_train/head_bev/best.pth",
        "out/exp1_train/head_point/best.pth", device)
    params["head_point_rb"] = n_params(model_rb)
    params["head_point_st"] = n_params(model_st)
    params["head_bev_sleeve"] = n_params(model_sl)
    from lib.segmentation.inference import load_model
    seg_model, seg_dev = load_model()
    params["seg"] = n_params(seg_model)

    out = {"device": torch.cuda.get_device_name(0)
           if device.type == "cuda" else "cpu",
           "params": params}

    if args.limit:
        pargs = single_chain.build_parser().parse_args([])
        pargs.face = "pointgrow-pred"
        pargs.limit = args.limit
        pargs.out = ""
        agg = single_chain.run_single(pargs)["aggregate"]
        out["latency_per_piece_s"] = agg.get("耗时/件s")
        out["latency_pieces"] = args.limit

    from lib.utils import anchor_project_path
    op = anchor_project_path(args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"→ {op}")


if __name__ == "__main__":
    main()
