# -*- coding: utf-8 -*-
"""
真实套筒端面人工标注工具（自包含 HTML）。

把真实套筒端面点云渲染成 2D 俯视图并配圆形画笔：在浏览器里对着孔画圆，
导出 JSON（圆心 h2/h3 + 内径半径），作为套筒识别的 GT。

用法（项目根）：
  .venv/Scripts/python.exe scripts/sleeve_gt_tool.py       [--cache temp_scripts/out/real_labels_cache.npz]       [--input "real_resource/scene_full_mm - Cloud.ply"]       [--out-dir out/sleeve_gt]
然后：
  .venv/Scripts/python.exe -m http.server 8138 --directory out/sleeve_gt

依赖：项目 .venv（numpy/scipy/open3d；无缓存时需 torch 推理）
"""
import json
import argparse
import os
import sys
import traceback

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
OUT_DIR = os.path.join(PROJECT_ROOT, "out", "sleeve_gt")

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>真实套筒端面 GT 标注</title>
<style>
  body { font-family: "Microsoft YaHei", sans-serif; margin: 0; background: #14161a; color: #d8dce2; }
  .wrap { display: flex; height: 100vh; }
  #left { flex: 0 0 auto; padding: 10px; }
  canvas { background: #0c0e11; cursor: crosshair; border: 1px solid #333; display: block; }
  #side { flex: 1; padding: 12px 16px; overflow-y: auto; border-left: 1px solid #2a2e35; }
  h2 { font-size: 15px; margin: 4px 0 8px; }
  .hint { font-size: 12px; color: #8b93a1; line-height: 1.7; }
  .hint b { color: #d8dce2; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
  th, td { border: 1px solid #2a2e35; padding: 4px 6px; text-align: right; }
  th { background: #1e2228; color: #9aa3b2; font-weight: normal; }
  tr:hover td { background: #1a1e24; }
  .del { color: #e06060; cursor: pointer; text-align: center; }
  button { background: #2b6cb0; color: #fff; border: 0; padding: 7px 12px;
           border-radius: 4px; cursor: pointer; font-size: 12px; margin: 2px; }
  button.gray { background: #3a3f47; }
  textarea { width: 100%; height: 130px; background: #0c0e11; color: #9fd0a0;
             border: 1px solid #2a2e35; font-size: 11px; margin-top: 8px; }
  .legend { font-size: 11px; color: #8b93a1; margin-top: 6px; }
  .sw { display: inline-block; width: 10px; height: 10px; margin: 0 4px 0 10px; vertical-align: middle; }
</style>
</head>
<body>
<div class="wrap">
  <div id="left"><canvas id="cv"></canvas></div>
  <div id="side">
    <h2>真实套筒端面 GT 标注</h2>
    <div class="hint">
      <b>操作</b>：在点云上<b>按下鼠标</b>定圆心，<b>拖拽</b>调半径，<b>松开</b>完成一个圆。<br/>
      <b>删除</b>：点表格最右列的 ×。<br/>
      <b>导出</b>：点"导出 JSON"，文件下载到浏览器下载目录，同时显示在下方文本框（可直接复制给我）。<br/>
      画的时候按套筒<b>内孔</b>边缘画（内径圆），尽量对齐可见环的内侧。
    </div>
    <div class="legend">
      <span class="sw" style="background:#ffd23f"></span>自动检出（参考，黄色虚线）
      <span class="sw" style="background:#e06060"></span>你画的 GT（红色）
      <span class="sw" style="background:#7f8fa6"></span>端面点云（亮度=凸起高度）
    </div>
    <div style="margin-top:10px">
      <button onclick="exportJson()">导出 JSON</button>
      <button class="gray" onclick="clearAll()">清空全部</button>
      <button class="gray" onclick="undo()">撤销上一个</button>
    </div>
    <table id="tbl">
      <thead><tr><th>#</th><th>h2 (mm)</th><th>h3 (mm)</th><th>半径 (mm)</th><th>删</th></tr></thead>
      <tbody></tbody>
    </table>
    <textarea id="out" placeholder="导出的 JSON 会显示在这里"></textarea>
  </div>
</div>
<script>
const DATA = __DATA__;
const BBOX = __BBOX__;
const DETECTED = __DETECTED__;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const DPR = window.devicePixelRatio || 1;
// 窗口高度兜底：后台/嵌入场景 window.innerHeight 可能为 0，导致画布尺寸为负
const MAXPX = Math.min(940, window.innerHeight > 100 ? window.innerHeight - 24 : 700);
const spanX = BBOX.xmax - BBOX.xmin, spanY = BBOX.ymax - BBOX.ymin;
const scale = MAXPX / Math.max(spanX, spanY);
const W = Math.round(spanX * scale), H = Math.round(spanY * scale);
cv.width = W * DPR; cv.height = H * DPR;
cv.style.width = W + 'px'; cv.style.height = H + 'px';
ctx.scale(DPR, DPR);

function toPx(h2, h3) { return [ (h2 - BBOX.xmin) * scale, H - (h3 - BBOX.ymin) * scale ]; }
function toMm(px, py) { return [ px / scale + BBOX.xmin, (H - py) / scale + BBOX.ymin ]; }

let circles = [], drawing = null, mouse = null;

function hColor(h) {
  // h: 高度 mm，范围约 [-15, 15]；凸起 2~8mm 用暖色高亮
  const t = Math.max(-1, Math.min(1, h / 10));
  if (t >= 0) {
    const k = Math.min(1, t * 1.6);
    return `rgb(${Math.round(120 + 135 * k)},${Math.round(120 + 30 * k)},${Math.round(120 - 90 * k)})`;
  }
  const k = Math.min(1, -t);
  return `rgb(${Math.round(110 - 50 * k)},${Math.round(115 - 40 * k)},${Math.round(125 + 10 * k)})`;
}

function draw() {
  ctx.clearRect(0, 0, W, H);
  // 网格（50mm）
  ctx.strokeStyle = '#1a1f26'; ctx.lineWidth = 1;
  const step = 50;
  for (let x = Math.ceil(BBOX.xmin / step) * step; x <= BBOX.xmax; x += step) {
    const [px] = toPx(x, 0); ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
  }
  for (let y = Math.ceil(BBOX.ymin / step) * step; y <= BBOX.ymax; y += step) {
    const [, py] = toPx(0, y); ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(W, py); ctx.stroke();
  }
  // 点云
  for (const p of DATA) {
    const [px, py] = toPx(p[0], p[1]);
    ctx.fillStyle = hColor(p[2]);
    ctx.fillRect(px, py, 1.4, 1.4);
  }
  // 自动检出（黄色虚线）
  ctx.setLineDash([5, 4]); ctx.strokeStyle = '#ffd23f'; ctx.lineWidth = 1.4;
  for (const d of DETECTED) {
    const [cx, cy] = toPx(d[0], d[1]);
    ctx.beginPath(); ctx.arc(cx, cy, d[2] * scale, 0, 2 * Math.PI); ctx.stroke();
  }
  ctx.setLineDash([]);
  // 已画 GT（红）
  ctx.strokeStyle = '#e06060'; ctx.lineWidth = 1.8;
  for (const c of circles) {
    const [cx, cy] = toPx(c.h2, c.h3);
    ctx.beginPath(); ctx.arc(cx, cy, c.r * scale, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - 4, cy); ctx.lineTo(cx + 4, cy);
    ctx.moveTo(cx, cy - 4); ctx.lineTo(cx, cy + 4); ctx.stroke();
  }
  if (drawing && mouse) {
    const [cx, cy] = toPx(drawing.h2, drawing.h3);
    const r = Math.hypot(mouse[0] - cx, mouse[1] - cy);
    ctx.strokeStyle = '#ffffff'; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.stroke();
    ctx.setLineDash([]);
  }
}

function renderTable() {
  const tb = document.querySelector('#tbl tbody');
  tb.innerHTML = '';
  circles.forEach((c, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${i + 1}</td><td>${c.h2.toFixed(1)}</td>
      <td>${c.h3.toFixed(1)}</td><td>${c.r.toFixed(1)}</td>
      <td class="del" onclick="del(${i})">×</td>`;
    tb.appendChild(tr);
  });
}

cv.addEventListener('mousedown', e => {
  const r = cv.getBoundingClientRect();
  const [h2, h3] = toMm(e.clientX - r.left, e.clientY - r.top);
  drawing = { h2, h3 };
  mouse = [e.clientX - r.left, e.clientY - r.top];
  draw();
});
cv.addEventListener('mousemove', e => {
  const r = cv.getBoundingClientRect();
  mouse = [e.clientX - r.left, e.clientY - r.top];
  if (drawing) draw();
});
window.addEventListener('mouseup', e => {
  if (!drawing) return;
  const r = cv.getBoundingClientRect();
  const [cx, cy] = toPx(drawing.h2, drawing.h3);
  const rad = Math.hypot(mouse[0] - cx, mouse[1] - cy) / scale;
  if (rad > 3) { circles.push({ h2: drawing.h2, h3: drawing.h3, r: rad }); }
  drawing = null; mouse = null; renderTable(); draw();
});
function del(i) { circles.splice(i, 1); renderTable(); draw(); }
function clearAll() { if (confirm('清空全部 GT 圆？')) { circles = []; renderTable(); draw(); } }
function undo() { circles.pop(); renderTable(); draw(); }
function exportJson() {
  const payload = {
    meta: { face_x: BBOX.face_x, sleeve_end: BBOX.sleeve_end, n: circles.length },
    circles: circles.map(c => ({ h2: +c.h2.toFixed(2), h3: +c.h3.toFixed(2), r_mm: +c.r.toFixed(2) }))
  };
  const txt = JSON.stringify(payload, null, 1);
  document.getElementById('out').value = txt;
  const blob = new Blob([txt], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'sleeve_gt_manual.json';
  a.click();
}
draw(); renderTable();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="真实套筒端面人工标注工具（生成 HTML + 圆形画笔）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache", type=str, default=None,
                        help="推理缓存 npz（pts/labels）；缺省或不存在时自动推理")
    parser.add_argument("--input", type=str,
                        default="real_resource/scene_full_mm - Cloud.ply")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="分割权重（缺省用 out/checkpoints/seg_v21_pointmlp_ce/best.pth）")
    parser.add_argument("--out-dir", type=str, default="out/sleeve_gt")
    args = parser.parse_args()

    try:
        from lib.utils import anchor_project_path, CHECKPOINT_BASE
        from lib.fitting.pipeline import run_deploy_pipeline, load_points

        cache = anchor_project_path(args.cache) if args.cache else None
        if cache and os.path.exists(cache):
            d = np.load(cache)
            pts = d["pts"].astype(np.float64)
            labels = d["labels"].astype(np.int64)
        else:
            import torch
            from lib.segmentation.inference import predict_labels_voxel
            from lib.segmentation import PointMLPSeg
            ckpt_path = anchor_project_path(args.ckpt) if args.ckpt else                 os.path.join(CHECKPOINT_BASE, "seg_v21_pointmlp_ce", "best.pth")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = PointMLPSeg(num_classes=4, in_channels=3).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            pts = load_points(anchor_project_path(args.input))
            labels, _ = predict_labels_voxel(pts, model)
            if cache:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                np.savez_compressed(cache, pts=pts.astype(np.float32),
                                    labels=labels.astype(np.int64))
        res = run_deploy_pipeline(pts, labels, rng=np.random.default_rng(0),
                                  crop=True, with_sleeves=True, verbose=False)
        frame = res["frame"]
        se = frame["sleeve_end"]
        pl = frame[f"end_{se}_plane"]
        c0, u = frame["axis_point"], frame["axis_dir"]
        e2, e3 = frame["e2"], frame["e3"]
        end_x = frame["end_neg_x"] if se == "neg" else frame["end_pos_x"]
        sign = -1.0 if se == "neg" else 1.0
        rel = pts - c0
        t = rel @ u
        band = np.abs((t - end_x)) < 80.0
        conc = band & (labels == 1)
        bp = pts[conc]
        h = (bp - pl["centroid"]) @ pl["normal"] * sign
        h = h - np.median(h)      # 端面主体归零，环凸（2~8mm）用暖色突出
        h2 = (bp - c0) @ e2
        h3 = (bp - c0) @ e3
        # 只保留端面附近 ±25mm 高度内的点（排除侧面）
        keep = (h > -25) & (h < 25)
        h2, h3, h = h2[keep], h3[keep], h[keep]
        print(f"端带混凝土点 {len(h2)}（保留 {len(h2)}）")
        # 降采样到 ≤ 120k
        rng = np.random.default_rng(0)
        if len(h2) > 120000:
            idx = rng.choice(len(h2), 120000, replace=False)
            h2, h3, h = h2[idx], h3[idx], h[idx]
        data = np.column_stack([np.round(h2, 1), np.round(h3, 1), np.round(h, 1)])
        bbox = {"xmin": float(h2.min()), "xmax": float(h2.max()),
                "ymin": float(h3.min()), "ymax": float(h3.max()),
                "face_x": float(end_x), "sleeve_end": se}
        # 自动检出参考环
        det = [[round(s["center_h2"], 1), round(s["center_h3"], 1),
                round(s["inner_r_mm"], 1)] for s in res["sleeves"]["sleeves"]
               if not s.get("filled")]
        print(f"参考环 {len(det)} 个")
        html = (HTML.replace("__DATA__", json.dumps(data.tolist()))
                    .replace("__BBOX__", json.dumps(bbox))
                    .replace("__DETECTED__", json.dumps(det)))
        OUT_DIR = anchor_project_path(args.out_dir)
        os.makedirs(OUT_DIR, exist_ok=True)
        out = os.path.join(OUT_DIR, "sleeve_gt_tool.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"工具 → {out}")
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
