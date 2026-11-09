# -*- coding: utf-8 -*-
"""
创建时间：2026-09-28 17:45:00
用途：补充实验总队列执行器（适配 src/ 与 out/ 结构）。
      顺序执行剩余实验，全部幂等：完成标记存在则跳过，训练类实验自动断点续训，
      失败步骤在末尾统一重试（最多 2 轮）。状态写入 out/train_logs/_runner_status.json，
      关键节点追加 out/train_logs/progress.md。
      运行方式: 由项目 .venv python 启动本脚本; 各实验步以 cwd=<root>/src
      执行 `python -m experiments.<模块>`（模块搜索路径由 cwd 提供）。
依赖：仅标准库
步骤链（顺序沿用此前会话拍板: 训练队列 → E4 → E3 → E1 → S3/S4 新旧口径 → E5-deep）：
  13 train_a5_wce        续训 a5_m5_wce (断点 ep38/120) → out/checkpoints/ablation_A5_wce/summary.json
  14 E4_b2_geo_valtest   几何基线 val 标定 + test 报告   → out/E4/b2_test_eval.json
  15 E3_fps_survival     FPS 逐级存活率 (3 骨干×3种子)   → out/E3/e3_fps_survival.json
  16 E1_tables           统一 cmIoU 口径重算 Table5/6/8  → out/E1/E1_tables.json (依赖 A5 权重)
  17 S3_density          密度敏感性 (仅测试构型, 新口径) → out/sensitivity/S3_summary.json
  18 S3_density_allkeys  同上, 旧口径泄漏对照            → out/sensitivity/S3_summary_allkeys.json
  19 S4_noise            噪声敏感性 (新口径)             → out/sensitivity/S4_summary.json
  20 S4_noise_allkeys    同上, 旧口径泄漏对照            → out/sensitivity/S4_summary_allkeys.json
  21 E5_measurement_deep 分割→测量闭环 (gt/geo/级联/M3)  → out/E5/e5_summary.json
"""
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # 项目根
SRC = os.path.join(ROOT, "src")
PY = sys.executable
TRAIN_LOGS = os.path.join(ROOT, "out", "train_logs")
CKPT = os.path.join(ROOT, "out", "checkpoints")
SENS = os.path.join(ROOT, "out", "sensitivity")
STATUS_PATH = os.path.join(TRAIN_LOGS, "_runner_status.json")
PROGRESS_PATH = os.path.join(TRAIN_LOGS, "progress.md")
MAX_RETRY_PASSES = 2

STEPS = [
    ("train_a5_wce",
     os.path.join(CKPT, "ablation_A5_wce", "summary.json"),
     ["-m", "experiments.experiment_a5_weighted_ce"]),
    ("E4_b2_geo_valtest",
     os.path.join(ROOT, "out", "E4", "b2_test_eval.json"),
     ["-m", "experiments.experiment_b2_geo_valtest", "--out", "out/E4"]),
    ("E3_fps_survival",
     os.path.join(ROOT, "out", "E3", "e3_fps_survival.json"),
     ["-m", "experiments.experiment_e3_fps_survival", "--out", "out/E3"]),
    ("E1_tables",
     os.path.join(ROOT, "out", "E1", "E1_tables.json"),
     ["-m", "experiments.experiment_e1_tables", "--out", "out/E1"]),
    ("S3_density",
     os.path.join(SENS, "S3_summary.json"),
     ["-m", "experiments.experiment_s3_density"]),
    ("S3_density_allkeys",
     os.path.join(SENS, "S3_summary_allkeys.json"),
     ["-m", "experiments.experiment_s3_density", "--all-keys"]),
    ("S4_noise",
     os.path.join(SENS, "S4_summary.json"),
     ["-m", "experiments.experiment_s4_noise"]),
    ("S4_noise_allkeys",
     os.path.join(SENS, "S4_summary_allkeys.json"),
     ["-m", "experiments.experiment_s4_noise", "--all-keys"]),
    ("E5_measurement_deep",
     os.path.join(ROOT, "out", "E5", "e5_summary.json"),
     ["-m", "experiments.experiment_e5_measurement", "--out", "out/E5", "--deep"]),
]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_progress(text):
    line = f"{now()} | {text}\n"
    with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"[progress] {text}", flush=True)


def load_status():
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"steps": {}}


def save_status(status):
    status["updated_at"] = now()
    status["runner_pid"] = os.getpid()
    status["runner_script"] = "scripts/queue_runner.py (v2, src/+out/ 结构)"
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


def log_path_for(name):
    numbering = {
        "train_a5_wce": "13_a5_m5_wce_resume",
        "E4_b2_geo_valtest": "14_E4_b2_geo_valtest",
        "E3_fps_survival": "15_E3_fps_survival",
        "E1_tables": "16_E1_tables",
        "S3_density": "17_S3_density",
        "S3_density_allkeys": "18_S3_density_allkeys",
        "S4_noise": "19_S4_noise",
        "S4_noise_allkeys": "20_S4_noise_allkeys",
        "E5_measurement_deep": "21_E5_measurement_deep",
    }
    return os.path.join(TRAIN_LOGS, numbering[name] + ".log")


def run_step(name, marker, args, status):
    entry = status["steps"].setdefault(name, {})
    if os.path.exists(marker):
        entry["state"] = "SKIPPED_DONE"
        save_status(status)
        print(f"[skip] {name}: 完成标记已存在 -> {os.path.relpath(marker, ROOT)}", flush=True)
        return True

    log_file = log_path_for(name)
    entry.update({"state": "RUNNING", "log": log_file,
                  "started_at": now(), "cmd": "python " + " ".join(args) + "  (cwd=src)"})
    save_status(status)
    print(f"[run ] {name}: python {' '.join(args)} (cwd=src)", flush=True)

    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    t0 = datetime.now()
    with open(log_file, "a", encoding="utf-8", errors="replace") as lf:
        lf.write(f"\n==== START {name} {now()} (v2 runner pid={os.getpid()}) ====\n")
        lf.flush()
        proc = subprocess.run([PY] + args, cwd=SRC, env=env,
                              stdout=lf, stderr=subprocess.STDOUT)
        lf.write(f"\n==== END {name} {now()} rc={proc.returncode} ====\n")

    entry["ended_at"] = now()
    entry["elapsed_s"] = round((datetime.now() - t0).total_seconds(), 1)
    entry["rc"] = proc.returncode
    ok = proc.returncode == 0 and os.path.exists(marker)
    entry["state"] = "OK" if ok else "FAIL"
    save_status(status)

    if ok:
        log_progress(f"{name} OK ({entry['elapsed_s']:.0f}s) | 输出 {os.path.relpath(marker, ROOT)}")
    else:
        log_progress(f"⚠ {name} FAIL (rc={proc.returncode}, 标记{'已生成' if os.path.exists(marker) else '未生成'}) | 日志 {os.path.basename(log_file)}")
    return ok


def main():
    os.makedirs(TRAIN_LOGS, exist_ok=True)
    status = load_status()
    status["runner_started_at"] = now()
    save_status(status)
    log_progress(
        "(队列恢复v2) 目录重组(src/+out/)后执行器v2启动 (pid={}) | a5 断点 ep38/120 (best.pth);"
        "步骤链: a5续训 → E4 → E3 → E1 → S3/S4新旧口径 → E5-deep, 全程幂等可断点"
        .format(os.getpid()))

    try:
        for pass_no in range(1 + MAX_RETRY_PASSES):
            pending = [s for s in STEPS
                       if status["steps"].get(s[0], {}).get("state") not in ("OK", "SKIPPED_DONE")]
            if not pending:
                break
            if pass_no > 0:
                log_progress(f"(重试第 {pass_no}/{MAX_RETRY_PASSES} 轮) 待重试: {[s[0] for s in pending]}")
            for name, marker, args in pending:
                run_step(name, marker, args, status)

        final = all(status["steps"].get(s[0], {}).get("state") in ("OK", "SKIPPED_DONE")
                    for s in STEPS)
        states = {s[0]: status["steps"].get(s[0], {}).get("state") for s in STEPS}
        if final:
            log_progress(f"(ALL DONE) 补充实验队列全部完成 | {states}")
        else:
            log_progress(f"(队列结束但有失败) {states} | 可直接重启本执行器重试失败步")
        save_status(status)
        sys.exit(0 if final else 1)
    except Exception:
        traceback.print_exc()
        log_progress(f"⚠ 队列执行器异常退出, 请查看执行器输出 | pid={os.getpid()}")
        save_status(status)
        sys.exit(2)


if __name__ == "__main__":
    main()
