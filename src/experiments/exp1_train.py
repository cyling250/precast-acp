# -*- coding: utf-8 -*-
"""exp1 —— 模型训练 + train/val/test 指标（调度器）。

五个子模型（权重落盘 out/exp1_train/，训练缓存 out/exp1_train/cache/）：
  point      点级圆检测头（钢筋/箍筋；消费 bev 缓存）
  bev        BEV 柱化头（套筒 + rbst 消融臂）
  face       阶段A 连接面语义 MiniPointNet（--face sem 消融臂）
  seg        场景材质分割 PointMLP（--patches/--merge 前置）
  sceneface  超点面网络（M2 研究线工具，非论文主链）

注意：论文数字一律用现有已训权重（不重训，P-072 下重训也不可复现）；
本入口的价值=①固化训练配方口径 ②推理-only 出 train/val/test 指标表。

用法（cwd=src）：
  # 只评不训（推理-only 指标表 → results/exp1_*_split.json）
  python -m experiments.exp1_train point --eval-split
  python -m experiments.exp1_train bev   --eval-split
  python -m experiments.exp1_train seg   --eval-split   # 需先重建补丁池
  # 确需重训时透传原参数（写 out/exp1_train/<默认 tag>）
  python -m experiments.exp1_train point --epochs 60
  python -m experiments.exp1_train seg   --patches   # / --merge / --train
"""
import json
import os
import sys


def _dump(name, obj):
    from lib.utils import anchor_project_path
    p = anchor_project_path(f"results/{name}")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print(f"→ {p}")


def eval_point_split():
    """点级头 train/val/test 指标（缓存视图臂口径，train_point.eval_*）。"""
    import torch
    from lib.utils import anchor_project_path
    from lib.training import train_point as tp
    from lib.training.train_point import (eval_ends_pt, eval_neg_pt,
                                          eval_stirrup_pt)
    from lib.e2e.bev_dataset import load_bev_cache
    from lib.e2e.point_head import PointCircleNet
    from lib.e2e.model_pillar import (REBAR_DIAM_TABLE, SLEEVE_IDIAM_TABLE,
                                      STIRRUP_DIAM_TABLE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt = torch.load(anchor_project_path("out/exp1_train/head_point/best.pth"),
                    map_location=device)
    flags = dict(bn=True, coarse=True, refine=True, fg=True, in_extra=7,
                 ref_radial=int(pt.get("ref_radial", 0)))
    m_rb = PointCircleNet(len(REBAR_DIAM_TABLE), **flags).to(device)
    m_rb.load_state_dict(pt["rb"])
    m_sl = PointCircleNet(len(SLEEVE_IDIAM_TABLE), **flags).to(device)
    m_sl.load_state_dict(pt["sl"])
    m_st = PointCircleNet(len(STIRRUP_DIAM_TABLE), **flags).to(device)
    m_st.load_state_dict(pt["st"])
    out = {}
    for split in ("train", "val", "test"):
        z = load_bev_cache(anchor_project_path(tp.CACHE), split)
        row = {"rebar": eval_ends_pt(m_rb, z, "rebar", device)}
        row["rebar"].update(eval_neg_pt(m_rb, z, "rebar", device))
        row["sleeve"] = eval_ends_pt(m_sl, z, "sleeve", device)
        row["sleeve"].update(eval_neg_pt(m_sl, z, "sleeve", device))
        st = eval_stirrup_pt(m_st, z, device)
        if st:
            row["stirrup"] = st
        out[split] = row
        print(f"[{split}] {json.dumps(row, ensure_ascii=False)}", flush=True)
    _dump("exp1_point_split.json", out)


def eval_bev_split():
    """BEV 头 train/val/test 指标（train_bev.eval_* 口径）。"""
    import torch
    from lib.utils import anchor_project_path
    from lib.training import train_bev as tb
    from lib.training.train_bev import eval_ends, eval_neg_views, eval_stirrup
    from lib.e2e.bev_dataset import load_bev_cache
    from lib.e2e.model_pillar import (PillarBEVNet, REBAR_DIAM_TABLE,
                                      SLEEVE_IDIAM_TABLE, STIRRUP_DIAM_TABLE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(anchor_project_path("out/exp1_train/head_bev/best.pth"),
                    map_location=device)
    m_rb = PillarBEVNet(len(REBAR_DIAM_TABLE), in_ch=8).to(device)
    m_rb.load_state_dict(ck["rb"])
    m_sl = PillarBEVNet(len(SLEEVE_IDIAM_TABLE), in_ch=8).to(device)
    m_sl.load_state_dict(ck["sl"])
    m_st = PillarBEVNet(len(STIRRUP_DIAM_TABLE)).to(device)
    m_st.load_state_dict(ck["st"])
    out = {}
    for split in ("train", "val", "test"):
        z = load_bev_cache(anchor_project_path(tb.CACHE), split)
        row = {"rebar": eval_ends(m_rb, z, "rebar", device)}
        row["rebar"].update(eval_neg_views(m_rb, z, "rebar", device))
        row["sleeve"] = eval_ends(m_sl, z, "sleeve", device)
        row["sleeve"].update(eval_neg_views(m_sl, z, "sleeve", device))
        st = eval_stirrup(m_st, z, device)
        if st:
            row["stirrup"] = st
        out[split] = row
        print(f"[{split}] {json.dumps(row, ensure_ascii=False)}", flush=True)
    _dump("exp1_bev_split.json", out)


def eval_seg_split():
    """场景分割 train/val/test 逐类 IoU（需补丁池，缺失时报错提示重建）。"""
    import numpy as np
    import torch
    from lib.training import train_seg as ts
    from lib.segmentation import PointMLPSeg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ts.PROJ, "out/exp1_train/seg/best.pth"),
                    map_location=device)
    model = PointMLPSeg(num_classes=ts.N_CLASSES, in_channels=6).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    out = {}
    for split in ("train", "val", "test"):
        try:
            pts, lbl = ts.load_pool(split, device)
        except FileNotFoundError:
            print(f"[{split}] 补丁池缺失——先重建：exp1_train seg --patches "
                  f"&& --merge（训练数据缓存被清理后的重建路径）", flush=True)
            continue
        miou, iou = ts.eval_patches(model, pts, lbl, device, n=600,
                                    rng=np.random.default_rng(1))
        out[split] = {"mIoU": round(float(miou), 4),
                      "IoU": {name: round(float(v), 3) for name, v
                              in zip(ts.CLASS_NAMES, iou)}}
        print(f"[{split}] {json.dumps(out[split], ensure_ascii=False)}",
              flush=True)
    _dump("exp1_seg_split.json", out)


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return
    model, rest = argv[0], argv[1:]
    if model == "point":
        if "--eval-split" in rest:
            return eval_point_split()
        from lib.training import train_point
        return train_point.main(rest)
    if model == "bev":
        if "--eval-split" in rest:
            return eval_bev_split()
        from lib.training import train_bev
        return train_bev.main(rest)
    if model == "face":
        from lib.training import train_face
        return train_face.main(rest)
    if model == "seg":
        if "--eval-split" in rest:
            return eval_seg_split()
        from lib.training import train_seg
        return train_seg.main(rest)
    if model == "sceneface":
        from lib.training import train_sceneface
        return train_sceneface.main(rest)
    raise SystemExit(f"未知子模型 {model!r}（可选 point/bev/face/seg/sceneface）")


if __name__ == "__main__":
    main()
