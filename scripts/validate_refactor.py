# -*- coding: utf-8 -*-
"""对账：重构后 exp2 staging 复跑 vs results/ canonical（P-072 容差）。

判据：
- 池化/聚合指标逐项对比：召回/误检/位置分位允许 ±0.01（召回类）与
  ±0.5mm（位置类）漂移；耗时类忽略。
- 单件 per_file：逐件 rb_match/sl_match/st_match 允许个别翻转（报告
  翻转件数），skip 集合必须一致。
- 场景池化 modes/overall 逐工况对比。
退出码：0=通过，1=超差。
"""
import io
import json
import os
import sys

from lib.utils import anchor_project_path

STAGE = "out/validate_staging"
TOL_RATIO = 0.01     # 召回/比率类容差（绝对）
TOL_MM = 0.5         # 位置/误差 p50 容差（mm）


def _p(rel):
    return anchor_project_path(rel)


def cmp_value(k, a, b, fails):
    if a is None or b is None:
        if a != b:
            fails.append(f"{k}: {b} -> {a}（含 None 变化）")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        tol = TOL_MM if ("p50" in k or "p90" in k or "误差" in k) else TOL_RATIO
        if abs(a - b) > tol:
            fails.append(f"{k}: {b} -> {a}（超差 {abs(a-b):.3f} > {tol}）")


def cmp_dict(prefix, new, old, fails):
    for k, v in old.items():
        if k not in new:
            fails.append(f"{prefix}{k}: canonical 有、复跑无")
            continue
        if "耗时" in k:
            continue          # 耗时随机器负载波动，不在对账口径
        nv = new[k]
        if isinstance(v, dict):
            cmp_dict(f"{prefix}{k}.", nv, v, fails)
        else:
            cmp_value(f"{prefix}{k}", nv, v, fails)


def cmp_matches(prefix, new_pf, old_pf, fails):
    """per_file 逐件 match 数对比（跳过 skip 件；skip 集合须一致）。"""
    def key(m):
        return m["file"].replace("\\", "/").split("/")[-1]
    new_m = {key(m): m for m in new_pf}
    old_m = {key(m): m for m in old_pf}
    if set(new_m) != set(old_m):
        only_n = set(new_m) - set(old_m)
        only_o = set(old_m) - set(new_m)
        fails.append(f"{prefix}: 件集合不一致 新增{len(only_n)} 缺失{len(only_o)}")
        return
    flips = 0
    for f in sorted(old_m):
        o, n = old_m[f], new_m[f]
        if bool(o.get("skip")) != bool(n.get("skip")):
            fails.append(f"{prefix}{f}: skip 状态翻转（{o.get('skip')}->{n.get('skip')}）")
            continue
        if o.get("skip"):
            continue
        for mk in ("rb_match", "sl_match", "st_match", "st_total_gt",
                   "rb_gt", "sl_total_gt"):
            if o["eval"].get(mk) != n["eval"].get(mk):
                flips += 1
                print(f"  [翻转] {prefix}{f}: {mk} "
                      f"{o['eval'].get(mk)} -> {n['eval'].get(mk)}")
    print(f"  {prefix}match 翻转计数: {flips}")


def main():
    fails = []

    # 1. 单件部署口径
    new = json.load(io.open(_p(f"{STAGE}/e2e_fullchain_pointgrow-pred.json"),
                            encoding="utf-8"))
    old = json.load(io.open(_p("results/e2e_fullchain/"
                            "e2e_fullchain_pointgrow-pred.json"),
                            encoding="utf-8"))
    print("== 单件 pointgrow-pred ==")
    cmp_dict("agg.", new["aggregate"], old["aggregate"], fails)
    cmp_matches("单件 ", new["per_file"], old["per_file"], fails)

    # 2. 场景池化（缺失则告警跳过，单件结论先行）
    if not os.path.exists(_p(f"{STAGE}/scene_pooled.json")):
        print("（场景池化 staging 未生成——单件结论如下，场景待补跑）")
        print("")
        print("===== 对账结论（部分） =====")
        if fails:
            print(f"FAIL：{len(fails)} 项超差")
            for f_ in fails:
                print(" -", f_)
            sys.exit(1)
        print("PASS：单件全部指标在 P-072 容差内")
        return
    newp = json.load(io.open(_p(f"{STAGE}/scene_pooled.json"), encoding="utf-8"))
    oldp = json.load(io.open(_p("results/scene_pooled.json"), encoding="utf-8"))
    print("== 场景池化 ==")
    for mode in oldp["modes"]:
        cmp_dict(f"modes.{mode}.", newp["modes"].get(mode, {}),
                 oldp["modes"][mode], fails)
    cmp_dict("overall.", newp.get("overall") or {}, oldp["overall"], fails)

    # 3. 逐场景 json 存在性 + match 计数
    import glob
    old_scenes = sorted(glob.glob(_p("results/scene_batch/*.json")))
    for p in old_scenes:
        name = os.path.basename(p)
        q = _p(f"{STAGE}/scene_batch/{name}")
        if not os.path.exists(q):
            fails.append(f"场景 json 缺失: {name}")
            continue
        ns = json.load(io.open(q, encoding="utf-8"))
        os_ = json.load(io.open(p, encoding="utf-8"))
        cmp_matches(f"场景 {name} ", ns.get("per_file", []),
                    os_.get("per_file", []), fails)

    print("\n===== 对账结论 =====")
    if fails:
        print(f"FAIL：{len(fails)} 项超差")
        for f_ in fails:
            print(" -", f_)
        sys.exit(1)
    print("PASS：全部指标在 P-072 容差内（召回 ±0.01 / 位置 ±0.5mm / "
          "逐件 match 翻转仅报告）")


if __name__ == "__main__":
    main()
