# -*- coding: utf-8 -*-
"""论文表格生成器：results/ canonical json → markdown 表（初稿引用用）。

数字一律由本脚本从入库 json 生成，禁止手抄（防转写误差）。

用法（cwd=src）：
  python -m scripts.gen_paper_tables [--out ../manuscript/tables_auto.md]
"""
import argparse
import glob
import io
import json
import os

from lib.utils import anchor_project_path

FACE_LABELS = (("oracle", "oracle（真值面理想上限）"),
               ("pointgrow", "pointgrow（真值材质+生长面）"),
               ("pointgrow-pred", "pointgrow-pred（部署流程）"))

SPLIT_NAMES = {"train": "训练", "val": "验证", "test": "测试"}
KIND_NAMES = {"rebar": "钢筋", "sleeve": "套筒", "stirrup": "箍筋"}
PARAM_NAMES = {"seg": "分割网", "head_point_rb": "点级钢筋头",
               "head_point_st": "点级箍筋头", "head_bev_sleeve": "BEV 套筒头"}


def _fmt(v, unit=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}{unit}" if unit else (f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}")
    return f"{v}{unit}"


def table_single():
    """表1 单件四口径（deployment=pointgrow-pred 加粗）。"""
    rows = []
    for face, label in FACE_LABELS:
        p = anchor_project_path(
            f"results/e2e_fullchain/e2e_fullchain_{face}.json")
        if not os.path.exists(p):
            rows.append((label, None))
            continue
        agg = json.load(io.open(p, encoding="utf-8"))["aggregate"]
        rb, sl, st = agg.get("钢筋", {}), agg.get("套筒", {}), agg.get("箍筋", {})
        rows.append((label, {
            "rb_recall": _fmt(rb.get("召回")),
            "rb_pos": _fmt(rb.get("位置p50"), "mm"),
            "rb_expo": _fmt(rb.get("外露量测误差p50"), "mm"),
            "sl_recall": _fmt(sl.get("召回")),
            "sl_pos": _fmt(sl.get("位置p50"), "mm"),
            "st_recall": _fmt(st.get("召回")),
            "st_leg": _fmt(st.get("腿位p50"), "mm"),
            "st_h": _fmt(st.get("高度p50"), "mm"),
        }))
    out = ["### 表1 单件面来源设置（合成单件 test 64 件）", "",
           "| 输入面来源 | 钢筋召回 | 钢筋位置p50 | 外露误差p50 | 套筒召回 | 套筒位置p50 | 箍筋召回 | 腿位p50 | 高度p50 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for label, d in rows:
        if d is None:
            out.append(f"| {label} | — | — | — | — | — | — | — | — |")
            continue
        bold = label.startswith("pointgrow-pred")
        wrap = (lambda s: f"**{s}**") if bold else (lambda s: s)
        out.append(f"| {label} | {wrap(d['rb_recall'])} | {wrap(d['rb_pos'])} | "
                   f"{wrap(d['rb_expo'])} | {wrap(d['sl_recall'])} | "
                   f"{wrap(d['sl_pos'])} | {wrap(d['st_recall'])} | "
                   f"{wrap(d['st_leg'])} | {wrap(d['st_h'])} |")
    return "\n".join(out)


def table_scene():
    """表2 场景池化（每工况 3 场景 + overall）。"""
    p = anchor_project_path("results/scene_pooled.json")
    if not os.path.exists(p):
        return "（results/scene_pooled.json 缺失）"
    d = json.load(io.open(p, encoding="utf-8"))
    out = ["### 表2 场景合并统计（每工况 3 场景，真值合并后统一计算）", "",
           "| 工况 | 钢筋召回 | 钢筋误检率 | 位置p50 | 位置p90 | 外露p50 | "
           "套筒召回 | 套筒误检率 | 套筒位置p50 | 内径误差p50 | "
           "箍筋召回 | 腿位p50 | 高度p50 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode in list(d["modes"]) + ["overall"]:
        row = d["modes"].get(mode) if mode != "overall" else d["overall"]
        if row is None:
            continue
        rb, sl, st = row.get("钢筋", {}), row.get("套筒", {}), row.get("箍筋", {})
        name = {"overall": "**总体**"}.get(mode, mode)
        out.append(
            f"| {name} | {_fmt(rb.get('召回'))} | {_fmt(rb.get('误检率'))} | "
            f"{_fmt(rb.get('位置p50'),'mm')} | {_fmt(rb.get('位置p90'),'mm')} | "
            f"{_fmt(rb.get('外露误差p50'),'mm')} | {_fmt(sl.get('召回'))} | "
            f"{_fmt(sl.get('误检率'))} | {_fmt(sl.get('位置p50'),'mm')} | "
            f"{_fmt(sl.get('ID误差p50'),'mm')} | {_fmt(st.get('召回'))} | "
            f"{_fmt(st.get('腿位p50'),'mm')} | {_fmt(st.get('高度p50'),'mm')} |")
    return "\n".join(out)


def table_real():
    """表3 真实场景。"""
    p = anchor_project_path("results/real_fullchain.json")
    if not os.path.exists(p):
        return "（results/real_fullchain.json 缺失）"
    d = json.load(io.open(p, encoding="utf-8"))
    rb, sl = d.get("钢筋", {}), d.get("套筒", {})
    out = ["### 表3 真实场景（钢筋判定阈值 0.3 定稿工作点；含人工标注残差约 5.9mm 噪声本底）",
           "",
           "| 指标 | 钢筋 | 套筒 |", "|---|---|---|"]
    pairs = [("检出数/真值数", f"{rb.get('检出')}/{rb.get('GT根数')}",
              f"{sl.get('检出')}/{sl.get('GT环数')}"),
             ("匹配/配准内点", rb.get("匹配"), sl.get("配准内点")),
             ("位置/残差 p50 (mm)", rb.get("位置p50mm"),
              sl.get("内点残差p50mm")),
             ("外露量测误差 p50 (mm)", rb.get("外露量测误差p50mm"), "—"),
             ("内径误差 p50 (mm)", "—", sl.get("ID误差p50mm")),
             ("标注拟合残差 (mm)", rb.get("标注拟合残差mm"), "—")]
    for name, a, b in pairs:
        out.append(f"| {name} | {_fmt(a) if not isinstance(a,str) else a} | "
                   f"{_fmt(b) if not isinstance(b,str) else b} |")
    return "\n".join(out)


def table_split():
    """表4 头级 train/val/test 指标（exp1 --eval-split 产物）。"""
    out = []
    for name, f in (("点级头", "exp1_point_split.json"),
                    ("BEV 头", "exp1_bev_split.json")):
        p = anchor_project_path(f"results/{f}")
        if not os.path.exists(p):
            continue
        d = json.load(io.open(p, encoding="utf-8"))
        out.append(f"#### {name}（预存视图法，训练/验证/测试集）")
        out.append("")
        out.append("| 数据划分 | 检测头 | 根数全对 | 召回 | 误检率 | 位置p50 |")
        out.append("|---|---|---|---|---|---|")
        for split, row in d.items():
            for kind in ("rebar", "sleeve", "stirrup"):
                if kind not in row:
                    continue
                s = row[kind]
                if kind == "stirrup":
                    out.append(f"| {SPLIT_NAMES.get(split, split)} | 箍筋 | {_fmt(s.get('count_acc'))} | "
                               f"— | — | 腿位 {_fmt(s.get('leg_pos_p50'),'mm')} |")
                else:
                    prec = s.get("precision")
                    miss = (_fmt(1 - prec) if prec is not None else "—")
                    out.append(f"| {SPLIT_NAMES.get(split, split)} | {KIND_NAMES[kind]} | {_fmt(s.get('count_acc'))} | "
                               f"{_fmt(s.get('recall'))} | {miss} | "
                               f"{_fmt(s.get('pos_p50'),'mm')} |")
        out.append("")
    return "\n".join(out) if out else "（split 评估未运行）"


def table_eff():
    p = anchor_project_path("results/efficiency.json")
    if not os.path.exists(p):
        return "（效率表未运行）"
    d = json.load(io.open(p, encoding="utf-8"))
    params = d.get("params", {})
    param_str = "；".join(f"{PARAM_NAMES.get(k, k)} {v:,}"
                          for k, v in params.items()) if params else "—"
    lines = ["### 效率（RTX 4060 Ti 8GB）", "",
             f"- 设备：{d.get('device')}",
             f"- 参数量：{param_str}",
             f"- 单件全流程耗时：{d.get('latency_per_piece_s')} s/件"
             f"（{d.get('latency_pieces')} 件均值，含分割）"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manuscript/tables_auto.md")
    args = ap.parse_args()
    parts = [table_single(), "", table_scene(), "", table_real(), "",
             table_split(), "", table_eff()]
    text = "\n\n".join(parts)
    op = anchor_project_path(args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with io.open(op, "w", encoding="utf-8", newline="") as f:
        f.write("# 论文表格（自动生成，勿手改；源=results/ canonical json）\n\n")
        f.write(text)
    print(text)
    print(f"\n→ {op}")


if __name__ == "__main__":
    main()
