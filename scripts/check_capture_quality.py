# -*- coding: utf-8 -*-
"""
创建时间：2026-10-15
用途：预制混凝土柱"照片法"点云重建——现场拍摄质量快速验证工具。

设计目标：拍摄当天、撤离现场之前，在任意一台笔记本上对刚拍的照片做闭环验证，
把"回去才发现重建失败"的风险消灭在现场。证据有二：
  Stage A  单张质量：清晰度（Laplacian 方差）、曝光剪裁、疑似重复帧、EXIF 焦距一致性；
  Stage B  重建连通性：pycolmap（或 COLMAP 命令行）增量 SfM 稀疏重建，
           给出配准率、每分区配准情况、钢筋笼特写弧的方位角覆盖缺口、GSD 估计。
最终输出 PASS / WARN / FAIL / INCOMPLETE 结论 + 逐条补拍建议（中文，可直接转达拍摄者）。

使用约定：照片按以下 ASCII 分区目录存放（用 --init-zones 一键创建）：
  00_calib      内参标定板序列（不参与重建）
  01_overview   定场环绕
  02_side_a     长侧面 A 条带（两遍）
  03_side_b     长侧面 B 条带（两遍）
  04_end_plain  无钢筋端面
  05_rebar_end  钢筋笼特写弧（成败关键）
  06_top        顶面
  07_corners    四角缝合
若没有分区目录（照片混在一起），脚本会整体按 unsorted 处理并提示。

用法：
  一键创建分区目录骨架   python check_capture_quality.py --init-zones <目标目录>
  完整验证（默认）       python check_capture_quality.py <照片根目录>
  只查单张质量（快）     python check_capture_quality.py <照片根目录> --quick
  出发前笔记本演练       python check_capture_quality.py --selftest --fast --out <演练目录>

另一台笔记本快速就绪（Windows）：
  1) 安装 Python 3.10+（勾选 Add to PATH）；
  2) pip install numpy pillow pycolmap
     （若 pycolmap 安装失败，可改用官方 COLMAP：把 colmap.exe 所在目录加入 PATH）；
  3) 演练：python check_capture_quality.py --selftest --fast --out %TEMP%\\qc_drill
     打印 "[自检] 环境就绪" 即依赖与 SfM 后端可用（合成数据配准率不作为环境判据）；
  4) 现场验证：python check_capture_quality.py <读卡器盘符>:\\DCIM 或照片根目录
     笔记本 CPU 预期耗时：--quick 1-2 分钟；完整验证约 5-15 分钟
     （自动抽样 ≤150 张 + 自动降采样到 1600px 长边；验证只判"能否重建/覆盖是否完整"，
      不做稠密点云——稀疏配准是稠密重建的必要条件，稀疏都拼不上稠密必然失败）；
  输出在 <照片根目录>/_qc_report/run_<时间戳>/：report.txt（人读）、summary.json（机器判读）。

依赖：numpy、Pillow（必需）；pycolmap 或 COLMAP 命令行（Stage B，强烈建议）。
注意：本脚本自包含、无项目路径硬编码，可在任何机器独立运行；仅标准库 + 上述三方包。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    print("[X] 缺少 Pillow。请先执行: pip install numpy pillow")
    sys.exit(4)

# ---------------------------------------------------------------- 常量与分区定义

IMG_EXTS = {".jpg", ".jpeg"}
RAW_EXTS = {".nrw", ".nef"}

# name: (说明, 正式拍摄最少张数, 覆盖检查类型)
# 覆盖检查: "strict"=钢筋笼弧线(缺口>60°警告/>90°失败) "ring"=环绕(缺口>90°警告) None=只查数量
ZONES = {
    "00_calib":     ("内参标定板序列（不参与重建）", 25, None),
    "01_overview":  ("定场环绕（3-4m 一圈）", 12, "ring"),
    "02_side_a":    ("长侧面A条带（两遍：1.5-2m + 近距斜视）", 30, "ring"),
    "03_side_b":    ("长侧面B条带（两遍）", 30, "ring"),
    "04_end_plain": ("无钢筋端面（正视+四角斜视+俯视）", 8, None),
    "05_rebar_end": ("钢筋笼特写弧（成败关键，每15-20°一个机位）", 60, "strict"),
    "06_top":       ("顶面（两侧斜拍条带+俯拍）", 25, None),
    "07_corners":   ("四角缝合（各45°两三张）", 8, None),
}

SFM_ZONES_EXCLUDE = {"00_calib"}          # 不参与 SfM 的分区
DUP_DIFF_THR = 3.0                        # 相邻缩略图平均绝对差 < 该值判疑似重复
BLUR_REL_THR = 0.35                       # 清晰度 < 中位数×该值 → 疑似模糊
OVER_THR, OVER_FRAC = 250, 0.02           # 过曝像素灰度阈值与比例
UNDER_THR, UNDER_FRAC = 5, 0.01


# ---------------------------------------------------------------- 输出工具

class Report:
    """同时打印到控制台并落盘的日志（UTF-8）。"""

    def __init__(self):
        self.lines = []

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.lines.append(str(msg))

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines), encoding="utf-8")


def fix_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ---------------------------------------------------------------- 环境自检

def check_env(log):
    """检查依赖可用性，返回 (stage_a_ok, backend, backend_hint)。"""
    ok_a = True
    try:
        import numpy  # noqa: F401
    except ImportError:
        log("[X] 缺少 numpy: pip install numpy")
        ok_a = False
    try:
        import PIL  # noqa: F401
    except ImportError:
        log("[X] 缺少 Pillow: pip install pillow")
        ok_a = False

    backend, hint = None, ""
    try:
        import pycolmap  # noqa: F401
        backend = "pycolmap"
    except ImportError:
        exe = shutil.which("colmap")
        if exe:
            backend = "cli"
        else:
            hint = ("Stage B（SfM 重建验证）不可用：既没有 pycolmap 也没有 COLMAP 命令行。\n"
                    "    修复（二选一）:  pip install pycolmap\n"
                    "                    或下载 COLMAP 并把 colmap.exe 加入 PATH")
    return ok_a, backend, hint


# ---------------------------------------------------------------- 通用数学工具

def quat_to_rot(q):
    """四元数 (w,x,y,z) -> 3x3 旋转矩阵。"""
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def lookat_apex(cam_centers, view_dirs):
    """视线交汇点：最小化 Σ||(I-ddᵀ)(x-oᵢ)||²，即所有相机光轴的最小二乘交点。
    拍摄目标就是所有相机共同注视的物体，此估计与背景点云分布无关。"""
    D = np.asarray(view_dirs, float)
    C = np.asarray(cam_centers, float)
    n = len(D)
    if n < 3:
        return C.mean(axis=0) if n else np.zeros(3)
    M = n * np.eye(3) - D.T @ D
    dot = np.einsum("ij,ij->i", D, C)
    b = n * C.mean(axis=0) - (D * dot[:, None]).sum(axis=0)
    try:
        if abs(np.linalg.det(M)) < 1e-6 * (abs(M[0, 0] * M[1, 1] * M[2, 2]) + 1e-12):
            raise np.linalg.LinAlgError
        return np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        return C.mean(axis=0)


def cage_apex(pts, cam_centers):
    """钢筋笼中心估计：取离本分区机位最近的一小撮稀疏点的中位数。"""
    if len(pts) == 0 or len(cam_centers) == 0:
        return np.zeros(3)
    sub = pts[::max(1, len(pts) // 20000)]        # 限点数，控制内存
    min_d = np.full(len(sub), np.inf)
    for c in cam_centers:
        min_d = np.minimum(min_d, np.linalg.norm(sub - c[None, :], axis=1))
    k = max(50, len(sub) // 20)
    order = np.argsort(min_d)[:k]
    return np.median(sub[order], axis=0)


def azimuth_max_gap(cam_centers, apex):
    """机位相对 apex 的最佳拟合平面内做 12×30° 方位角分箱，返回 (最大空缺角度, 占用bin)。"""
    v = cam_centers - apex[None, :]
    if len(v) < 4:
        return 360.0, np.zeros(12, bool)
    X = v - v.mean(axis=0)
    cov = X.T @ X / len(X)
    _, vecs = np.linalg.eigh(cov)          # 特征值升序
    e1, e2 = vecs[:, 2], vecs[:, 1]
    az = np.degrees(np.arctan2(X @ e2, X @ e1)) % 360.0
    bins = (((az + 15.0) % 360.0) / 30.0).astype(int) % 12   # 半箱偏移，防边界跨箱幻影缺口
    occ = np.zeros(12, bool)
    occ[bins] = True
    if occ.all():
        return 0.0, occ
    ext = np.concatenate([occ, occ])
    max_run, run = 0, 0
    for o in ext:
        run = 0 if o else run + 1
        max_run = max(max_run, run)
    max_run = min(max_run, 12)             # 环形防重复计数
    return max_run * 30.0, occ


# ---------------------------------------------------------------- Stage A：单张质量

def qc_one_image(path):
    """对单张 JPEG 做质量指标。返回 dict 或 {"ok": False, "error": str}。"""
    out = {"path": str(path), "name": path.name, "zone": path.parent.name, "ok": False}
    try:
        exif_f = None
        with Image.open(path) as im:
            try:
                v = im.getexif().get(37386)     # FocalLength
                if v is not None:
                    exif_f = float(v)
            except Exception:
                pass
            im.draft("L", (1600, 1600))
            g = im.convert("L")
            arr = np.asarray(g, dtype=np.float32)
            if arr.shape[0] > 1600:
                s = 1600.0 / arr.shape[0]
                arr = np.asarray(g.resize((max(64, int(arr.shape[1] * s)), 1600),
                                          Image.BILINEAR), dtype=np.float32)
        lap = (4 * arr - np.roll(arr, 1, 0) - np.roll(arr, -1, 0)
               - np.roll(arr, 1, 1) - np.roll(arr, -1, 1))
        out.update(
            ok=True,
            h=int(arr.shape[0]), w=int(arr.shape[1]),
            sharp=float(lap.var()),
            over=float((arr > OVER_THR).mean()),
            under=float((arr < UNDER_THR).mean()),
            exif_f=exif_f,
            thumb=np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((64, 48)),
                             dtype=np.float32),
        )
    except Exception as e:
        out["error"] = repr(e)
    return out


def run_stage_a(zones, log, workers=8):
    jobs = [(zone, p) for zone, files in zones.items() for p in files]
    log(f"[Stage A] 单张质量检查：共 {len(jobs)} 张 JPEG（{workers} 线程）...")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(qc_one_image, p) for _, p in jobs]
        zone_of = {id(f): z for (z, _), f in zip(jobs, futs)}
        done = 0
        for f in futs:
            r = f.result()
            r["zone"] = zone_of[id(f)]
            results.append(r)
            done += 1
            if done % 50 == 0:
                log(f"    ... {done}/{len(jobs)}")
    log(f"[Stage A] 完成，用时 {time.time() - t0:.0f}s")
    return results


def summarize_stage_a(results, zones, log, issues, mode):
    """汇总 Stage A；把问题追加进 issues。返回指标 dict。"""
    ok_items = [r for r in results if r.get("ok")]
    bad_items = [r for r in results if not r.get("ok")]
    sharp_med = float(np.median([r["sharp"] for r in ok_items])) if ok_items else 0.0
    blur = [r for r in ok_items if r["sharp"] < sharp_med * BLUR_REL_THR]
    over = [r for r in ok_items if r["over"] > OVER_FRAC]
    under = [r for r in ok_items if r["under"] > UNDER_FRAC]

    log(f"    有效 {len(ok_items)} 张，读取失败 {len(bad_items)} 张；清晰度中位数 {sharp_med:.1f}")
    if bad_items:
        issues.append(f"[单张] {len(bad_items)} 张 JPEG 读取失败，例：{bad_items[0]['path']}")

    # 疑似模糊（列最差的几张）
    if len(ok_items) >= 20:
        worst = sorted(ok_items, key=lambda r: r["sharp"])[:8]
        if len(blur) > 0.05 * len(ok_items):
            issues.append(f"[单张] 疑似模糊 {len(blur)} 张（>5%），最差例："
                          + "；".join(f"{r['zone']}/{r['name']}(sharp={r['sharp']:.0f})" for r in worst[:3])
                          + " → 现场放大抽检，糊了就补拍对应机位")
            log("    最差 8 张：" + "、".join(f"{r['zone']}/{r['name']}" for r in worst))

    # 曝光
    if len(over) > 0.05 * len(ok_items) and ok_items:
        issues.append(f"[单张] 过曝(>{OVER_THR})占比>2% 的有 {len(over)} 张，例："
                      f"{over[0]['zone']}/{over[0]['name']} → 收一档曝光补拍对应机位")
    if len(under) > 0.05 * len(ok_items) and ok_items:
        issues.append(f"[单张] 死黑(<{UNDER_THR})占比>1% 的有 {len(under)} 张，例："
                      f"{under[0]['zone']}/{under[0]['name']} → 提一档曝光补拍对应机位")

    # 疑似重复帧（同分区按文件名排序的相邻帧）
    dup_total = 0
    for zone, files in zones.items():
        by_name = {r["name"]: r for r in ok_items if r["zone"] == zone}
        seq = [by_name[p.name] for p in files if p.name in by_name]
        for a, b in zip(seq, seq[1:]):
            if float(np.abs(a["thumb"] - b["thumb"]).mean()) < DUP_DIFF_THR:
                dup_total += 1
    if dup_total:
        log(f"    疑似重复帧 {dup_total} 对（原地停留过久，无碍重建，仅提示）")

    # EXIF 焦距一致性（变焦环是否被改动）
    focals = [r["exif_f"] for r in ok_items if r.get("exif_f")]
    if focals and len(ok_items):
        uniq = sorted({round(f, 1) for f in focals})
        if len(uniq) > 1 and min(uniq) / max(uniq) < 0.97:
            issues.append(f"[单张] EXIF 焦距不一致：发现 {len(uniq)} 档（{uniq[:4]}...）→ "
                          "变焦环可能被改动，确认后按当前焦距补齐各分区")
        elif len(focals) < 0.5 * len(ok_items):
            log("    提示：过半照片无 EXIF 焦距（可能转储软件剥离），跳过一致性检查")

    # RAW-only 分区（COLMAP 读不了尼康 RAW）
    for zone, files in zones.items():
        raws = [p for p in files if p.suffix.lower() in RAW_EXTS]
        jpgs = [p for p in files if p.suffix.lower() in IMG_EXTS]
        if raws and not jpgs:
            issues.append(f"[单张] 分区 {zone} 只有 RAW 没有 JPG → COLMAP 无法直接读取，"
                          "请用相机 JPEG 直出重导出该分区")

    return {
        "n_ok": len(ok_items), "n_fail": len(bad_items),
        "sharp_median": sharp_med, "n_blur_suspect": len(blur),
        "n_overexposed": len(over), "n_underexposed": len(under),
        "n_duplicate_pairs": dup_total,
    }


# ---------------------------------------------------------------- 照片收集与 staging

def collect_images(root, log):
    """按分区收集 JPEG；没有分区目录时整体按 unsorted 处理。返回 {zone: [Path]}。"""
    zone_dirs = [d for d in root.iterdir() if d.is_dir() and d.name in ZONES]
    zones = {}
    if zone_dirs:
        for d in sorted(zone_dirs):
            files = sorted(p for p in d.iterdir()
                           if p.is_file() and p.suffix.lower() in IMG_EXTS | RAW_EXTS)
            if files:
                zones[d.name] = files
    else:
        files = sorted(p for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMG_EXTS | RAW_EXTS)
        if files:
            zones["unsorted"] = files
        log("[!] 未发现分区目录，全部照片按 unsorted 处理；下次建议先 --init-zones 建骨架再导入")
    return zones


def ascii_report_dir(root):
    """报告目录首选 <root>/_qc_report；若路径含非 ASCII（COLMAP 会挂），退到系统临时目录。"""
    pref = root / "_qc_report"
    try:
        pref.resolve().as_posix().encode("ascii")
        return pref
    except UnicodeEncodeError:
        fb = Path(tempfile.gettempdir()) / "capture_qc_report"
        print(f"[!] 照片路径含非 ASCII 字符，报告目录改用 {fb}")
        return fb


def stage_images(zones, stage_dir, max_images, log, staging_max_px=1600):
    """把待重建照片（仅 JPG）复制/降采样进 ASCII 平铺目录（COLMAP 对非 ASCII 路径不稳定，
    且笔记本上全分辨率特征提取太慢——降采样到 1600px 是快速验证的关键）。
    返回 (staged_dir, {zone: [Path]} 子集, staged_name -> 原路径 映射)。"""
    sfm_zones = {z: [f for f in files if f.suffix.lower() in IMG_EXTS]
                 for z, files in zones.items() if z not in SFM_ZONES_EXCLUDE}
    sfm_zones = {z: f for z, f in sfm_zones.items() if f}
    total = sum(len(f) for f in sfm_zones.values())
    if total > max_images:
        factor = max_images / total
        subset = {}
        for z, files in sfm_zones.items():
            keep = max(min(8, len(files)), int(round(len(files) * factor)))
            idx = np.linspace(0, len(files) - 1, keep).astype(int)
            subset[z] = [files[i] for i in sorted(set(idx.tolist()))]
        log(f"[SfM] 照片共 {total} 张 > 上限 {max_images}，按分区比例抽样至 "
            f"{sum(len(f) for f in subset.values())} 张（抽样只影响本次验证，不影响回单位全量重建）")
    else:
        subset = sfm_zones

    stage_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    n_link = n_copy = n_down = 0
    for z, files in subset.items():
        for i, src in enumerate(files):
            if src.suffix.lower() not in IMG_EXTS:      # RAW 不进 SfM（COLMAP 读不了）
                continue
            dst = stage_dir / f"{z}__{i:04d}.jpg"
            mapping[dst.name] = str(src)
            if dst.exists():
                continue
            if staging_max_px and src.suffix.lower() in IMG_EXTS:
                try:
                    with Image.open(src) as im:
                        w, h = im.size
                        if max(w, h) > staging_max_px:  # 大图降采样：笔记本提速的关键
                            s = staging_max_px / max(w, h)
                            im.draft("RGB", (round(w * s), round(h * s)))
                            im = ImageOps.exif_transpose(im)
                            im = im.convert("RGB").resize(
                                (round(im.width * s), round(im.height * s)), Image.BILINEAR)
                            im.save(dst, "JPEG", quality=88)
                            n_down += 1
                            continue
                except Exception:
                    pass                                 # 处理失败退回归档路径
            try:
                os.link(src, dst)
                n_link += 1
            except OSError:
                shutil.copy2(src, dst)
                n_copy += 1
    log(f"[SfM] staging 就绪：{len(mapping)} 张（降采样 {n_down}，硬链接 {n_link}，复制 {n_copy}）"
        f"→ {stage_dir}")
    if n_down:
        log(f"[SfM] 已把长边 > {staging_max_px}px 的照片降到 {staging_max_px}px 再验证"
            f"（验证只判配准/覆盖，降采样不影响结论；全量原始分辨率重建回单位再做）")
    (stage_dir.parent / "manifest.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
    return stage_dir, {z: [f for f in fs if f.suffix.lower() in IMG_EXTS]
                       for z, fs in subset.items()}, mapping


# ---------------------------------------------------------------- Stage B：SfM 后端

def sfm_pycolmap(img_dir, work, log):
    import pycolmap
    log(f"[SfM] backend=pycolmap {getattr(pycolmap, '__version__', '?')}（CPU 亦可用，进度由库打印）")
    db = str(work / "database.db")
    sparse = str(work / "sparse")
    kw = {}
    try:
        kw["camera_mode"] = pycolmap.CameraMode.SINGLE
    except Exception:
        pass
    pycolmap.extract_features(database_path=db, image_path=str(img_dir), **kw)
    pycolmap.match_exhaustive(database_path=db)
    maps = pycolmap.incremental_mapping(database_path=db, image_path=str(img_dir),
                                        output_path=sparse)
    if isinstance(maps, dict):
        if not maps:
            raise RuntimeError("incremental_mapping 未得到任何模型")
        n_models = len(maps)
        n_reg_total = sum(len(getattr(r, "images", {})) for r in maps.values())
        rec = max(maps.values(), key=lambda r: len(getattr(r, "images", {})))
    else:
        n_models, n_reg_total = 1, len(getattr(maps, "images", {}))
    data = list(rec_to_data(rec))
    data[6]["n_models"] = n_models                    # 模型分裂 = 照片拼不到一个整体
    data[6]["n_reg_total"] = n_reg_total
    return tuple(data)


def sfm_cli(img_dir, work, log):
    exe = shutil.which("colmap")
    log(f"[SfM] backend=COLMAP CLI ({exe})")
    db = str(work / "database.db")
    sparse = work / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    steps = [
        ["feature_extractor", "--database_path", db, "--image_path", str(img_dir),
         "--ImageReader.single_camera", "1"],
        ["exhaustive_matcher", "--database_path", db],
        ["mapper", "--database_path", db, "--image_path", str(img_dir),
         "--output_path", str(sparse)],
    ]
    for step in steps:
        log(f"[SfM] colmap {step[0]} ...")
        proc = subprocess.run([exe] + step, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        tail = "\n".join((proc.stdout or "").splitlines()[-3:]
                         + (proc.stderr or "").splitlines()[-3:])
        log("    " + tail.replace("\n", "\n    "))
        if proc.returncode != 0:
            raise RuntimeError(f"colmap {step[0]} 失败 rc={proc.returncode}")
    models = [d for d in sparse.iterdir() if d.is_dir() and (d / "images.txt").exists()]
    if not models:
        raise RuntimeError("mapper 未产出任何模型")
    best, best_n, n_reg_total = None, -1, 0
    for d in models:
        n = sum(1 for ln in (d / "images.txt").read_text().splitlines()
                if ln.strip() and not ln.startswith("#")) // 2
        n_reg_total += n
        if n > best_n:
            best, best_n = d, n
    subprocess.run([exe, "model_converter", "--input_path", str(best),
                    "--output_path", str(best), "--output_type", "TXT"],
                   capture_output=True, text=True, check=True)
    data = list(parse_model_txt(best))
    data[6]["n_models"] = len(models)
    data[6]["n_reg_total"] = n_reg_total
    return tuple(data)


def rec_to_data(rec):
    """pycolmap Reconstruction -> (names, centers, fdirs, points, fx, width, extra)。"""
    names, centers, fdirs = [], [], []
    for img in rec.images.values():
        R, t = _pose(img)
        names.append(img.name)
        centers.append(-R.T @ t)
        fdirs.append(R.T @ np.array([0.0, 0.0, 1.0]))
    pts = (np.array([p.xyz for p in rec.points3D.values()], dtype=float)
           if len(rec.points3D) else np.zeros((0, 3)))
    cam = next(iter(rec.cameras.values()))
    extra = {}
    for attr, key in (("compute_mean_track_length", "mean_track_len"),
                      ("compute_mean_reprojection_error", "mean_reproj_err")):
        try:
            extra[key] = float(getattr(rec, attr)())
        except Exception:
            pass
    return names, np.array(centers), np.array(fdirs), pts, _focal_px(cam), _cam_width(cam), extra


def _pose(img):
    """兼容 pycolmap 各版本取相机位姿：4.x cam_from_world() 是方法，0.6 用 qvec/tvec。"""
    cfw_attr = getattr(img, "cam_from_world", None)
    if callable(cfw_attr):
        cfw = cfw_attr()
        rot = cfw.rotation
        if hasattr(rot, "matrix"):
            try:
                R = np.asarray(rot.matrix(), dtype=float)
            except TypeError:
                R = np.asarray(rot.matrix, dtype=float)
        else:
            R = quat_to_rot([rot.w, rot.x, rot.y, rot.z])
        tr = cfw.translation
        return R, np.asarray(tr() if callable(tr) else tr, dtype=float)
    if cfw_attr is not None:                       # 旧版属性式
        rot, t = cfw_attr.rotation, cfw_attr.translation
        R = np.asarray(rot.matrix(), dtype=float) if hasattr(rot, "matrix") \
            else quat_to_rot([rot.w, rot.x, rot.y, rot.z])
        return R, np.asarray(t, dtype=float)
    return (quat_to_rot(np.asarray(img.qvec, dtype=float)),
            np.asarray(img.tvec, dtype=float))


def _focal_px(cam):
    for attr in ("fx", "focal_length_x", "f"):
        v = getattr(cam, attr, None)
        if v:
            return float(v)
    try:
        return float(cam.params[0])
    except Exception:
        return None


def _cam_width(cam):
    v = getattr(cam, "width", None)
    if v:
        return int(v)
    try:
        return int(cam.params and 0) or None
    except Exception:
        return None


def parse_model_txt(d):
    """解析 COLMAP TXT 模型（CLI backend 用，不依赖 pycolmap）。"""
    cams = {}
    for ln in (d / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if ln.strip() and not ln.startswith("#"):
            p = ln.split()
            cams[p[0]] = p
    names, qts = [], {}
    lines = (d / "images.txt").read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip() and not ln.startswith("#"):
            p = ln.split()
            name = p[9]
            names.append(name)
            qts[name] = (np.array([float(x) for x in p[1:5]]),
                         np.array([float(x) for x in p[5:8]]))
            i += 2
        else:
            i += 1
    centers, fdirs = [], []
    cam0 = next(iter(cams.values())) if cams else None
    fx = float(cam0[4]) if cam0 and len(cam0) > 4 else None
    width = int(cam0[2]) if cam0 else None
    for name in names:
        q, t = qts[name]
        R = quat_to_rot(q)
        centers.append(-R.T @ t)
        fdirs.append(R.T @ np.array([0.0, 0.0, 1.0]))
    pts = []
    for ln in (d / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if ln.strip() and not ln.startswith("#"):
            p = ln.split()
            pts.append([float(p[1]), float(p[2]), float(p[3])])
    return (names, np.array(centers), np.array(fdirs),
            np.array(pts) if pts else np.zeros((0, 3)), fx, width, {})


# ---------------------------------------------------------------- 指标分析

def analyze_sfm(data, subset, zones_all, mode, log):
    """data = rec_to_data/parse_model_txt 的输出。返回指标 dict。"""
    names, centers, fdirs, pts, fx, width, extra = data
    n_reg = len(names)
    n_all = sum(len(f) for f in subset.values())
    ratio = n_reg / n_all if n_all else 0.0
    log(f"[SfM] 配准 {n_reg}/{n_all}（{ratio * 100:.1f}%），稀疏点 {len(pts)}"
        + (f"，平均轨迹长 {extra.get('mean_track_len', float('nan')):.1f}" if "mean_track_len" in extra else "")
        + (f"，重投影误差 {extra.get('mean_reproj_err', float('nan')):.2f}px" if "mean_reproj_err" in extra else ""))
    if extra.get("n_models", 1) > 1:
        log(f"[SfM] 警示：重建分裂为 {extra['n_models']} 个独立模型"
            f"（最大模型 {n_reg} / 各模型合计 {extra.get('n_reg_total', n_reg)} 张）——部分照片互相拼不上")

    apex = lookat_apex(centers, fdirs)
    dists = np.linalg.norm(centers - apex[None, :], axis=1) if n_reg else np.array([])
    gsds = dists * 1000 / fx if fx else np.array([])
    log("[SfM] 注意：SfM 尺度任意，以下距离/GSD 均为模型单位（未定标），"
        "只看相对关系；米制 GSD 回单位定标后再评估")
    if len(dists):
        log(f"[SfM] 机位到目标距离（模型单位）：中位 {np.median(dists):.2f}"
            f"，p10 {np.percentile(dists, 10):.2f} / p90 {np.percentile(dists, 90):.2f}")

    # 分区分组（staged 名 = zone__序号）
    zone_of_name = {nm: nm.split("__", 1)[0] for nm in names}
    reg_set = set(names)
    zone_stats = {}
    for z, files in subset.items():
        staged_names = [f"{z}__{i:04d}.jpg" for i in range(len(files))]
        reg = [nm for nm in staged_names if nm in reg_set]
        unreg = [files[int(nm.split("__", 1)[1][:4])].name
                 for nm in staged_names if nm not in reg_set]
        zdists = [d for nm, d in zip(names, dists) if zone_of_name.get(nm) == z]
        zgsd = [g for nm, g in zip(names, gsds) if zone_of_name.get(nm) == z]
        zone_stats[z] = {
            "desc": ZONES.get(z, ("未分区",))[0] if z in ZONES else "未分区照片",
            "n_input": len(files), "n_registered": len(reg),
            "ratio": round(len(reg) / len(files), 3) if files else 0.0,
            "dist_median_model": round(float(np.median(zdists)), 3) if zdists else None,
            "gsd_median_model": round(float(np.median(zgsd)), 3) if zgsd else None,
            "unregistered_examples": unreg[:15],
        }
        if unreg:
            log(f"[SfM] {z} 未配准 {len(unreg)} 张，例：{'、'.join(unreg[:5])}")

    # 方位角覆盖（只能用已配准机位近似——未配准机位不可见，缺口结论偏乐观，报告里注明）
    coverage = {}
    for z, files in subset.items():
        cov_type = ZONES.get(z, (None, None, None))[2] if z in ZONES else None
        if cov_type is None:
            continue
        zc = np.array([c for c, nm in zip(centers, names) if zone_of_name.get(nm) == z])
        if cov_type is None or len(zc) < 4:
            continue
        az_apex = cage_apex(pts, zc) if cov_type == "strict" else apex
        gap, occ = azimuth_max_gap(zc, az_apex)
        coverage[z] = {"type": cov_type, "max_gap_deg": gap,
                       "occupied_bins": int(occ.sum())}
        log(f"[覆盖] {z}: 方位角最大缺口 {gap:.0f}°（占用 {int(occ.sum())}/12 个 30° 扇区）")

    close = int((dists < 1.5).sum()) if len(dists) else 0
    mid = int(((dists >= 1.5) & (dists < 3.0)).sum()) if len(dists) else 0
    far = int((dists >= 3.0).sum()) if len(dists) else 0

    return {
        "n_all": n_all, "n_registered": n_reg, "ratio": round(ratio, 4),
        "n_sparse_points": int(len(pts)),
        "n_models": extra.get("n_models", 1),
        "n_registered_all_models": extra.get("n_reg_total", n_reg),
        "mean_track_length": extra.get("mean_track_len"),
        "mean_reproj_error_px": extra.get("mean_reproj_err"),
        "target_center_xyz": [round(float(v), 3) for v in apex],
        "focal_px": fx, "image_width_px": width,
        "distance_bands_model": {"lt1.5": close, "1.5-3": mid, "gt3": far},
        "zones": zone_stats, "coverage": coverage,
        "note_azimuth": "方位角覆盖按已配准机位统计，未配准机位不计入（结论偏乐观）",
        "note_scale": "SfM 尺度任意：距离/GSD 为模型单位，米制定标回单位用标定板/实测尺寸完成",
    }


# ---------------------------------------------------------------- 结论判定

def make_verdict(stage_a_metrics, sfm_metrics, zones, mode, sfm_ran, issues, log):
    advice = []

    if sfm_ran:
        ratio = sfm_metrics["ratio"]
        zs = sfm_metrics["zones"]
        cov = sfm_metrics["coverage"]
        npts = sfm_metrics["n_sparse_points"]

        if ratio < 0.60:
            issues.append(f"[SfM] 配准率仅 {ratio * 100:.0f}%（<60%）→ 重建大概率失败："
                          "检查是否运动模糊/变焦漂移/场景变动，重点分区整组重拍")
        elif ratio < 0.85:
            issues.append(f"[SfM] 配准率 {ratio * 100:.0f}%（<85%）→ 找到未配准机位集中的分区定点补拍")
        if npts < max(2000, 30 * sfm_metrics["n_registered"]):
            issues.append(f"[SfM] 稀疏点 {npts} 偏少 → 纹理不足或重叠不够，检查曝光与移动步长")

        # 整体性：照片必须拼进同一个模型，否则点云是碎的几坨
        if (sfm_metrics.get("n_models", 1) > 1
                and sfm_metrics["n_registered"] < 0.95 * sfm_metrics.get("n_registered_all_models", 0)):
            issues.append(f"[整体性] 重建分裂为 {sfm_metrics['n_models']} 个独立模型"
                          f"（最大 {sfm_metrics['n_registered']} / 合计 "
                          f"{sfm_metrics['n_registered_all_models']} 张配准）→ 照片组之间缺缝合："
                          "在分裂的分区交界处，双向 45° 各补 3-5 张过渡机位")
            advice.append("补缝合机位：各分区交界（尤其 07_corners 覆盖的四角），"
                          "让相邻分区同框出现，再重跑验证")

        # 分区数量下限（合成演练按实际张数放行，只验流程不验张数）
        for z, (desc, min_n, _) in ZONES.items():
            if z in ("00_calib", "unsorted"):
                continue
            st = zs.get(z)
            if st is None:
                if mode != "self":
                    issues.append(f"[分区] 缺少分区 {z}（{desc}）→ 按 {min_n}+ 张补拍整个分区")
                    advice.append(f"补拍分区 {z}：{desc}，至少 {min_n} 张")
                continue
            if mode != "self" and 0 < st["n_input"] < min_n:
                issues.append(f"[分区] {z} 仅 {st['n_input']} 张（建议≥{min_n}）→ 建议补拍")
            if st["ratio"] < 0.6:
                advice.append(f"重拍分区 {z}：该分区配准率只有 {st['ratio'] * 100:.0f}%，"
                              "注意对焦、稳定持机、相邻画面重叠≥70%")
            elif st["ratio"] < 0.85:
                advice.append(f"补拍分区 {z}：部分机位未配准（{st['n_input'] - st['n_registered']} 张），"
                              "对空白机位位置补 3-5 张")

        # 钢筋笼机位相对距离（尺度无关判据）：特写圈中位距离不应超过侧面条带的 0.8 倍
        d5 = zs.get("05_rebar_end", {}).get("dist_median_model")
        side_ds = [zs[z]["dist_median_model"] for z in ("02_side_a", "03_side_b")
                   if zs.get(z, {}).get("dist_median_model")]
        if d5 and side_ds and mode != "self" and d5 > 0.8 * float(np.median(side_ds)):
            issues.append("[相对距离] 钢筋笼特写机位偏远（中位距离 ≥ 侧面条带的 80%）→ "
                          "蹲近到 0.8-1.2m 补拍特写弧")

        # 钢筋笼覆盖缺口
        c5 = cov.get("05_rebar_end")
        if c5:
            if c5["max_gap_deg"] > 90:
                issues.append(f"[覆盖] 钢筋笼弧线有 {c5['max_gap_deg']:.0f}° 方位缺口（>90°）→ "
                              "该方向钢筋必丢，务必补拍对应扇区（每 15-20° 一个机位）")
                advice.append("补拍钢筋笼：按缺口方位补 4-8 个机位，半径 0.8-1.2m，蹲位与站位交替")
            elif c5["max_gap_deg"] > 60:
                issues.append(f"[覆盖] 钢筋笼弧线有 {c5['max_gap_deg']:.0f}° 方位缺口（>60°）→ "
                              "建议补拍对应扇区，防止细钢筋端头丢失")
        elif "05_rebar_end" in zs:
            issues.append("[覆盖] 钢筋笼分区机位太少，无法评估方位覆盖 → 至少 12 个机位环绕补拍")
    else:
        log("[!] Stage B 未运行，结论不完整：只能反映单张质量，不能反映重建连通性")

    # 汇总 verdict
    has_fail = any(s.startswith(("[SfM] 配准率仅", "[覆盖] 钢筋笼弧线有", "[分区] 缺少分区"))
                   or "只有 RAW" in s or "重建大概率失败" in s for s in issues)
    if not sfm_ran:
        verdict = "INCOMPLETE"
    elif has_fail:
        verdict = "FAIL"
    elif issues:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return verdict, advice


# ---------------------------------------------------------------- 主检查流程

def run_check(root, report_root, log, mode="real", quick=False,
              max_images=150, backend_pref="auto", staging_max_px=1600):
    zones = collect_images(root, log)
    if not zones:
        log("[X] 未找到任何 JPEG/RAW 照片，检查路径与盘符")
        return "ERROR", {}, {}, {}

    log(f"[输入] 分区概况：")
    for z in sorted(zones):
        desc, min_n, _ = ZONES.get(z, ("未分区", 0, None))
        n_jpg = sum(1 for p in zones[z] if p.suffix.lower() in IMG_EXTS)
        log(f"    {z:12s} {len(zones[z]):4d} 张（JPG {n_jpg}）  {desc}"
            + ("" if mode == "self" or z == "unsorted" or len(zones[z]) >= min_n
               else f"  [!] 少于建议下限 {min_n}"))

    run_dir = report_root / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    report_txt = run_dir / "report.txt"

    issues = []
    results = run_stage_a(zones, log)
    stage_a_metrics = summarize_stage_a(results, zones, log, issues, mode)

    sfm_metrics, sfm_ran = None, False
    if not quick:
        _, backend, hint = check_env(lambda m: None)
        if backend_pref != "auto":
            backend = backend_pref if backend_pref in ("pycolmap", "cli") else None
        if backend is None:
            log("[!] " + hint)
            issues.append("[环境] " + hint.split("\n")[0])
        else:
            stage_dir, subset, _ = stage_images(
                zones, run_dir / "images_staged", max_images, log,
                staging_max_px=staging_max_px)
            work = run_dir / ("colmap_" + backend)
            work.mkdir(exist_ok=True)
            try:
                t0 = time.time()
                data = (sfm_pycolmap(stage_dir, work, log) if backend == "pycolmap"
                        else sfm_cli(stage_dir, work, log))
                log(f"[SfM] 完成，用时 {time.time() - t0:.0f}s")
                sfm_metrics = analyze_sfm(data, subset, zones, mode, log)
                sfm_ran = True
            except Exception as e:
                log(f"[X] SfM 失败：{e}")
                traceback.print_exc()
                issues.append(f"[环境] SfM 运行失败：{e}（换 backend 或减 --max-images 重试）")

    verdict, advice = make_verdict(stage_a_metrics, sfm_metrics, zones, mode, sfm_ran,
                                   issues, log)

    banner = {"PASS": "[PASS] 可以撤离现场", "WARN": "[WARN] 建议定点补拍后再撤",
              "FAIL": "[FAIL] 重建会失败，必须重拍", "INCOMPLETE": "[INCOMPLETE] 环境不全，结论不完整",
              "ERROR": "[ERROR] 输入有误"}[verdict]
    log("")
    log("=" * 60)
    log(f"  结论：{banner}")
    log("=" * 60)
    if issues:
        log("问题清单（按严重度排列）：")
        for i, s in enumerate(issues, 1):
            log(f"  {i}. {s}")
    if advice:
        log("补拍行动项（转达拍摄者）：")
        for i, s in enumerate(advice, 1):
            log(f"  {i}. {s}")
    if verdict == "PASS":
        log("未发现问题。请：① 双备份照片；② 记录焦距/分区表/实测尺寸，然后收工。")
    log(f"详细报告：{report_txt}")

    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": mode, "photos_root": str(root), "verdict": verdict,
        "issues": issues, "advice": advice,
        "stage_a": stage_a_metrics, "sfm": sfm_metrics,
        "zone_inputs": {z: len(f) for z, f in zones.items()},
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log.save(report_txt)
    return verdict, summary, issues, advice


# ---------------------------------------------------------------- --init-zones

def init_zones(target, log):
    target.mkdir(parents=True, exist_ok=True)
    for z, (desc, min_n, _) in ZONES.items():
        d = target / z
        d.mkdir(exist_ok=True)
        note = d / "_拍摄说明.txt"
        note.write_text(f"{z}：{desc}\n建议张数：≥ {min_n}\n"
                        f"（文件名不限，JPG 直出；本分区{'参与' if z not in SFM_ZONES_EXCLUDE else '不参与'}重建验证）\n",
                        encoding="utf-8")
    log(f"[OK] 已创建分区骨架 → {target}")
    for z, (desc, min_n, _) in ZONES.items():
        log(f"    {z:12s} ≥{min_n:3d} 张  {desc}")


# ---------------------------------------------------------------- --selftest 合成演练

def _scene_texture(rng, size=256):
    """随机灰格（NEAREST 放大保持锐利）+ 矩形斑 + 椒盐点：全尺度都有 SIFT 角点。"""
    cells = rng.randint(40, 220, (10, 10)).astype(np.uint8)
    im = Image.fromarray(cells).resize((size, size), Image.NEAREST)
    arr = np.asarray(im).astype(np.int16)
    for _ in range(30):                                       # 小矩形斑 → 强角点
        x0, y0 = rng.randint(0, size - 9, 2)
        arr[y0:y0 + rng.randint(2, 7), x0:x0 + rng.randint(2, 7)] = rng.randint(0, 255)
    ys, xs = rng.randint(0, size, 200), rng.randint(0, size, 200)
    arr[ys, xs] += rng.choice((-80, 80), 200)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    rgba = np.dstack([arr, arr, arr, np.full_like(arr, 255)])
    return Image.fromarray(rgba, "RGBA")


def _quad(x0, y0, z0, x1, y1, z1, axis, pos):
    """在 axis='x'/'y'/'z'、坐标 pos 处取盒子的一个面。"""
    c = {(0, 1, 2): [(x0, y0, pos), (x1, y0, pos), (x1, y1, pos), (x0, y1, pos)],
         (1, 0, 2): [(pos, y0, z0), (pos, y1, z0), (pos, y1, z1), (pos, y0, z1)],
         (2, 0, 1): [(x0, pos, z0), (x1, pos, z0), (x1, pos, z1), (x0, pos, z1)]}
    key = {"z": (0, 1, 2), "x": (1, 0, 2), "y": (2, 0, 1)}[axis]
    return [np.array(p, float) for p in c[key]]


def _find_coeffs(pa, pb):
    A = []
    for (x, y), (u, v) in zip(pa, pb):
        A.append([x, y, 1, 0, 0, 0, u * x, u * y])
        A.append([0, 0, 0, x, y, 1, v * x, v * y])
    try:
        coeffs = np.linalg.solve(np.array(A, float), np.array([c for p in pb for c in p], float))
        return tuple(coeffs)
    except np.linalg.LinAlgError:
        return None


def build_selftest_quads():
    """极简合成场景：中央立方体 + 8 面立牌（无地面——画家算法在贴地视角下排序会乱）。
    各面深度互不交错，排序恒正确；特征密度高，SfM 必然可配准。
    演练只验证环境与流水线本身，分区/覆盖逻辑由正式拍摄数据检验。"""
    quads = []
    x0, x1, y0, y1 = -0.5, 0.5, -0.5, 0.5             # 中央立方体 1m
    for axis, pos in (("z", 1.0), ("z", 0.0), ("y", y1), ("y", y0), ("x", x1), ("x", x0)):
        quads.append(_quad(x0, y0, 0, x1, y1, 1, axis, pos))
    for k in range(8):                                # 8 面立牌 2×2m，径向朝内
        a = 2 * np.pi * k / 8 + np.pi / 8
        c = np.array([3.5 * np.cos(a), 3.5 * np.sin(a), 1.2])
        t = np.array([-np.sin(a), np.cos(a), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        quads.append([c - t - up, c + t - up, c + t + up, c - t + up])
    return quads


def render_selftest_images(out_dir, fast, log):
    """渲染演练照片（无分区，平铺在 out_dir 根下，24 个环拍机位）。"""
    W = 1920 if fast else 2400     # 大于默认降采样阈值 1600，演练即覆盖"大图→降采样→SfM"路径
    H = int(W * 2 / 3)
    f = W / (2 * np.tan(np.radians(27.5)))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
    quads = build_selftest_quads()
    texs = [_scene_texture(np.random.RandomState(i)) for i in range(64)]
    tw = th = texs[0].size[0]
    src_corners = [(0, 0), (tw, 0), (tw, th), (0, th)]
    up_world = np.array([0.0, 0.0, 1.0])
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(24):
        a = 2 * np.pi * i / 24
        eye = np.array([2.3 * np.cos(a), 2.3 * np.sin(a), 0.6 + 1.2 * (i % 3) / 2])
        tgt = np.array([0.0, 0.0, 0.5])
        fwd = tgt - eye
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, up_world)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0, 0])
        right = right / np.linalg.norm(right)
        upc = np.cross(right, fwd)
        R = np.stack([right, -upc, fwd], axis=0)
        canvas = Image.new("RGBA", (W, H), (128, 128, 128, 255))
        items = []
        for qi, q in enumerate(quads):
            Xc = (q - eye[None, :]) @ R.T
            if Xc[:, 2].min() < 0.05:
                continue
            uv = (Xc @ K.T)
            uv = uv[:, :2] / uv[:, 2:3]
            if uv[:, 0].max() < 0 or uv[:, 0].min() > W or uv[:, 1].max() < 0 or uv[:, 1].min() > H:
                continue
            items.append((float(Xc[:, 2].mean()), qi, uv))
        items.sort(key=lambda it: -it[0])
        for _, qi, uv in items:
            coeffs = _find_coeffs([tuple(p) for p in uv], src_corners)
            if coeffs is None:
                continue
            warped = texs[qi % len(texs)].transform((W, H), Image.PERSPECTIVE,
                                                    coeffs, Image.BICUBIC)
            canvas = Image.alpha_composite(canvas, warped)
        canvas.convert("RGB").save(out_dir / f"DSC_{i:04d}.jpg", quality=92)
    log(f"    渲染演练照片 24 张 → {out_dir}")
    return out_dir


def run_selftest(out_dir, fast, log, backend_pref="auto"):
    log(f"[自检] 合成数据演练（fast={fast}）：渲染极简合成场景并跑完整验证流水线"
        "（验证环境与流水线；分区/覆盖逻辑由正式拍摄数据检验）")
    _, backend, hint = check_env(lambda m: None)
    if backend_pref in ("pycolmap", "cli"):
        backend = backend_pref if backend_pref in ("pycolmap", "cli") else None
    if backend is None:
        log("[X] " + hint)
        log("    演练必须在 Stage B 可用的环境下进行（这正是演练的目的）。")
        return "INCOMPLETE"
    if out_dir.exists() and any(out_dir.iterdir()):
        out_dir = out_dir.with_name(out_dir.name + "_" + datetime.now().strftime("%H%M%S"))
    img_root = out_dir / "selftest_images"
    t0 = time.time()
    render_selftest_images(img_root, fast, log)
    log(f"[自检] 渲染完成，用时 {time.time() - t0:.0f}s → {img_root}")
    verdict, summary, _, _ = run_check(img_root, out_dir, log, mode="self", quick=False,
                                       max_images=400, backend_pref=backend_pref)
    # 演练只验证环境（依赖齐全、SfM 后端能跑、报告能写），不判合成数据质量
    log("")
    if summary.get("sfm"):
        log("[自检] 环境就绪：依赖齐全、SfM 后端正常、报告落盘。"
            "（合成配准率仅供建模参考，不作数据质量结论）")
        return "PASS"
    log("[自检] 环境不完整：SfM 后端未能运行，按上方提示修复后再演练一次")
    return "INCOMPLETE"


# ---------------------------------------------------------------- 入口

def main():
    fix_console()
    ap = argparse.ArgumentParser(
        description="预制构件现场拍摄质量快速验证（Stage A 单张质量 + Stage B SfM 连通性）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python %(prog)s --init-zones D:/shoot/site01      # 建分区骨架\n"
               "  python %(prog)s D:/shoot/site01                   # 完整验证\n"
               "  python %(prog)s E: --quick                        # 只查单张质量\n"
               "  python %(prog)s --selftest --fast --out %%TEMP%%/qc_drill  # 笔记本演练\n")
    ap.add_argument("photos_root", nargs="?", help="照片根目录（读卡器盘符或拷贝目录），含分区子目录")
    ap.add_argument("--init-zones", metavar="DIR", help="在指定目录创建分区骨架后退出")
    ap.add_argument("--selftest", action="store_true", help="合成数据演练（验证本机环境全流程）")
    ap.add_argument("--drill", action="store_true",
                    help="把 photos_root 当演练数据判定（放行张数/GSD 下限，只看流程是否通过）")
    ap.add_argument("--fast", action="store_true", help="selftest 用小图快档")
    ap.add_argument("--quick", action="store_true", help="跳过 SfM，只做 Stage A")
    ap.add_argument("--out", metavar="DIR", help="selftest 输出目录（默认系统临时目录下）")
    ap.add_argument("--max-images", type=int, default=150,
                    help="参与 SfM 验证的最大张数（多则按分区比例抽样，默认 150）")
    ap.add_argument("--staging-max-px", type=int, default=1600,
                    help="SfM 验证前把照片长边降到该像素数（笔记本提速关键，0=原始分辨率）")
    ap.add_argument("--backend", choices=("auto", "pycolmap", "cli"), default="auto",
                    help="Stage B 后端（默认 auto：优先 pycolmap，退回 COLMAP CLI）")
    args = ap.parse_args()

    log = Report()
    stage_a_ok, _, _ = check_env(log)
    if not stage_a_ok:
        return 4
    if args.init_zones:
        init_zones(Path(args.init_zones), log)
        return 0
    if args.selftest:
        out = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "qc_selftest"
        v = run_selftest(out, args.fast, log, args.backend)
        if args.out is None:
            log(f"（未指定 --out，演练产物在系统临时目录，可手动删除）")
        return {"PASS": 0, "WARN": 2, "FAIL": 3, "INCOMPLETE": 4, "ERROR": 5}[v]
    if not args.photos_root:
        ap.error("需要照片根目录，或使用 --init-zones / --selftest")
    root = Path(args.photos_root)
    if not root.exists():
        print(f"[X] 路径不存在: {root}")
        return 5
    v, *_ = run_check(root, ascii_report_dir(root), log,
                      mode="self" if args.drill else "real", quick=args.quick,
                      max_images=args.max_images, backend_pref=args.backend,
                      staging_max_px=args.staging_max_px)
    return {"PASS": 0, "WARN": 2, "FAIL": 3, "INCOMPLETE": 4, "ERROR": 5}[v]


if __name__ == "__main__":
    sys.exit(main())
