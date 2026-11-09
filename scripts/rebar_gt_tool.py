# -*- coding: utf-8 -*-
"""
真实钢筋插入位置人工标注工具（自包含 HTML，双模式）。

绘制模式：端面 2D 俯视图，对着每根外露钢筋的可见弧画圆，圆心=插入位置
真值，导出 JSON（h2/h3 + 半径）。与 scripts/sleeve_gt_tool.py 同款交互。
三维视图：WebGL 点云（端带三层点 + 12 根识别骨架线 + 端面圆环），拖拽
旋转、滚轮缩放，用于核对钢筋空间位置后再回绘制模式画圆。

用法（项目根）：
  .venv/Scripts/python.exe scripts/rebar_gt_tool.py \
      [--cache temp_scripts/out/real_labels_cache.npz] \
      [--input "real_resource/scene_full_mm - Cloud.ply"] \
      [--gt real_resource/rebar_only.ply] \
      [--out-dir out/rebar_gt]
然后：
  .venv/Scripts/python -m http.server 8139 --directory out/rebar_gt

导出 rebar_gt_manual.json：{meta:{face_x, sleeve_end, n},
rebars:[{h2, h3, r_mm}]}——h2/h3 为拟合系端面内坐标，与识别报告
face_pos_h2h3_mm 同系，可直接做位置误差对照。

依赖：项目 .venv（numpy/scipy/open3d；无缓存时需 torch 推理）
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>真实钢筋插入位置 GT 标注</title>
<style>
  body { font-family: "Microsoft YaHei", sans-serif; margin: 0; background: #14161a; color: #d8dce2; }
  .wrap { display: flex; height: 100vh; }
  #left { flex: 0 0 auto; padding: 10px; }
  canvas { background: #0c0e11; border: 1px solid #333; display: block; }
  #cv { cursor: crosshair; }
  #cv3 { cursor: grab; display: none; }
  #cv3:active { cursor: grabbing; }
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
  button.mode { background: #3a3f47; }
  button.mode.active { background: #2b6cb0; }
  textarea { width: 100%; height: 130px; background: #0c0e11; color: #9fd0a0;
             border: 1px solid #2a2e35; font-size: 11px; margin-top: 8px; }
  .legend { font-size: 11px; color: #8b93a1; margin-top: 6px; line-height: 1.8; }
  .sw { display: inline-block; width: 10px; height: 10px; margin: 0 4px 0 10px; vertical-align: middle; }
</style>
</head>
<body>
<div class="wrap">
  <div id="left">
    <div style="margin-bottom:6px">
      <button class="mode active" id="m2d" onclick="setMode('2d')">绘制模式（端面）</button>
      <button class="mode" id="m3d" onclick="setMode('3d')">三维视图（拖拽旋转）</button>
    </div>
    <canvas id="cv"></canvas>
    <canvas id="cv3"></canvas>
  </div>
  <div id="side">
    <h2>真实钢筋插入位置 GT 标注</h2>
    <div class="hint" id="hint2d">
      <b>操作</b>：对着每根外露钢筋在端面附近的<b>可见弧</b>画圆——按下鼠标定圆心，
      拖拽调半径，松开完成。圆心 = 插入位置真值，半径 ≈ 可见筋半径。<br/>
      <b>删除</b>：点表格最右列的 ×；<b>撤销</b>：按钮；共 12 根。<br/>
      <b>三维视图</b>：不确定钢筋位置时先切到三维视图旋转核对，再回来画。<br/>
      <b>导出</b>：点"导出 JSON"，文件下载到浏览器下载目录，同时显示在下方文本框（可直接复制给我）。
    </div>
    <div class="hint" id="hint3d" style="display:none">
      <b>操作</b>：按住左键拖拽旋转，滚轮缩放，<b>复位视角</b>按钮回到初始角度。<br/>
      本视图只用于核对钢筋空间位置；<b>画圆请切回绘制模式</b>。<br/>
      可先取消勾选"端面点"只看骨架与钢筋点，认筋更清楚。
    </div>
    <div class="legend">
      <span class="sw" style="background:#57d98a"></span>钢筋预测点（对着绿色弧画）
      <span class="sw" style="background:#6a8fd8"></span>手筛钢筋 GT 点（rebar_only.ply）
      <span class="sw" style="background:#7f8fa6"></span>混凝土端面点（亮度=凸起）
      <span class="sw" style="background:#ffd23f"></span>自动识别插入位置（黄色虚线，参考）
      <span class="sw" style="background:#e06060"></span>你画的 GT（红色）
      <span class="sw" style="background:#ff8844"></span>识别骨架线（三维视图）
    </div>
    <div style="margin-top:10px">
      <button onclick="exportJson()">导出 JSON</button>
      <button class="gray" onclick="clearAll()">清空全部</button>
      <button class="gray" onclick="undo()">撤销上一个</button>
      <button class="gray" onclick="toggleLayers()" id="layerBtn">隐藏 GT 蓝点</button>
      <button class="gray" id="resetBtn" style="display:none" onclick="resetView()">复位视角</button>
    </div>
    <table id="tbl">
      <thead><tr><th>#</th><th>h2 (mm)</th><th>h3 (mm)</th><th>半径 (mm)</th><th>直径 (mm)</th><th>删</th></tr></thead>
      <tbody></tbody>
    </table>
    <textarea id="out" placeholder="导出的 JSON 会显示在这里"></textarea>
  </div>
</div>
<script>
const CONC = __CONC__;      // [h2, t_rel, h3, h] 端带混凝土
const REBAR = __REBAR__;    // [h2, t_rel, h3] 端带钢筋预测
const GTPTS = __GTPTS__;    // [h2, t_rel, h3] 手筛 GT 端带投影
const SKELETONS = __SKELETONS__; // 识别骨架折线（每根 [h2,t,h3]×24）
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
for (const c of [cv, document.getElementById('cv3')]) {
  c.width = W * DPR; c.height = H * DPR;
  c.style.width = W + 'px'; c.style.height = H + 'px';
}
ctx.scale(DPR, DPR);

function toPx(h2, h3) { return [ (h2 - BBOX.xmin) * scale, H - (h3 - BBOX.ymin) * scale ]; }
function toMm(px, py) { return [ px / scale + BBOX.xmin, (H - py) / scale + BBOX.ymin ]; }

let circles = [], drawing = null, mouse = null, showGt = true, mode = '2d';

function hColor(h) {
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
  ctx.strokeStyle = '#1a1f26'; ctx.lineWidth = 1;
  const step = 50;
  for (let x = Math.ceil(BBOX.xmin / step) * step; x <= BBOX.xmax; x += step) {
    const [px] = toPx(x, 0); ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
  }
  for (let y = Math.ceil(BBOX.ymin / step) * step; y <= BBOX.ymax; y += step) {
    const [, py] = toPx(0, y); ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(W, py); ctx.stroke();
  }
  for (const p of CONC) {
    const [px, py] = toPx(p[0], p[2]);
    ctx.fillStyle = hColor(p[3]);
    ctx.fillRect(px, py, 1.4, 1.4);
  }
  if (showGt) {
    ctx.fillStyle = '#6a8fd8';
    for (const p of GTPTS) {
      const [px, py] = toPx(p[0], p[2]);
      ctx.fillRect(px, py, 1.8, 1.8);
    }
  }
  ctx.fillStyle = '#57d98a';
  for (const p of REBAR) {
    const [px, py] = toPx(p[0], p[2]);
    ctx.fillRect(px, py, 1.6, 1.6);
  }
  ctx.setLineDash([5, 4]); ctx.strokeStyle = '#ffd23f'; ctx.lineWidth = 1.4;
  for (const d of DETECTED) {
    const [cx, cy] = toPx(d[0], d[1]);
    ctx.beginPath(); ctx.arc(cx, cy, d[2] * scale, 0, 2 * Math.PI); ctx.stroke();
  }
  ctx.setLineDash([]);
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

// ── 三维视图（WebGL 点云 + 骨架线 + 端面圆环）──
const cv3 = document.getElementById('cv3');
let gl = null, prog = null, nVerts = 0, bufPts = null, bufLine = null, nLine = 0,
    bufRing = null, nRing = 0, glReady = false;
const cam = { yaw: -0.85, pitch: 0.5, dist: 2600, cx: 0, cy: 550, cz: 0 };
let dragging = null;

const VS = `attribute vec3 aPos; attribute vec3 aCol; attribute float aSize;
uniform mat4 uMVP; uniform float uPtH;
varying vec3 vCol; varying float vFog;
void main(){
  gl_Position = uMVP * vec4(aPos, 1.0);
  float w = max(gl_Position.w, 1.0);
  gl_PointSize = clamp(aSize * uPtH / w, 1.0, 12.0);
  vCol = aCol; vFog = clamp((w - 800.0) / 4200.0, 0.0, 0.65);
}`;
const FS = `precision mediump float; varying vec3 vCol; varying float vFog;
void main(){
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  gl_FragColor = vec4(mix(vCol, vec3(0.05, 0.06, 0.07), vFog), 1.0);
}`;

function hex2rgb(h) {
  return [parseInt(h.slice(1, 3), 16) / 255,
          parseInt(h.slice(3, 5), 16) / 255,
          parseInt(h.slice(5, 7), 16) / 255];
}

function initGL() {
  if (glReady) return;
  gl = cv3.getContext('webgl', { antialias: true });
  if (!gl) { alert('浏览器不支持 WebGL，无法使用三维视图'); return; }
  function sh(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
      throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog); gl.useProgram(prog);
  // 顶点：pos3 + col3 + size1
  const verts = [];
  for (const p of CONC) {
    const c = hColor(p[3]).match(/\d+/g).map(Number).map(v => v / 255);
    verts.push(p[0], p[1], p[2], c[0], c[1], c[2], 3.0);
  }
  const cg = hex2rgb('#57d98a'), cb = hex2rgb('#6a8fd8');
  for (const p of REBAR) verts.push(p[0], p[1], p[2], cg[0], cg[1], cg[2], 3.6);
  if (showGt) for (const p of GTPTS) verts.push(p[0], p[1], p[2], cb[0], cb[1], cb[2], 3.6);
  nVerts = verts.length / 7;
  bufPts = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bufPts);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
  // 骨架线（橙）
  const lines = [];
  const co = hex2rgb('#ff8844');
  for (const sk of SKELETONS)
    for (let i = 0; i < sk.length - 1; i++) {
      lines.push(...sk[i], ...co, ...sk[i + 1], ...co);
    }
  nLine = lines.length / 6;
  bufLine = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bufLine);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.STATIC_DRAW);
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.047, 0.055, 0.067, 1);
  glReady = true;
  syncRings();
  render3d();
}

function bindAttrs() {
  const stride = 28;
  const aPos = gl.getAttribLocation(prog, 'aPos');
  const aCol = gl.getAttribLocation(prog, 'aCol');
  const aSize = gl.getAttribLocation(prog, 'aSize');
  gl.enableVertexAttribArray(aPos);
  gl.enableVertexAttribArray(aCol);
  gl.enableVertexAttribArray(aSize);
  if (nRing > 0 && bufRing) gl.bindBuffer(gl.ARRAY_BUFFER, bufRing);
  else if (nLine > 0 && bufLine) gl.bindBuffer(gl.ARRAY_BUFFER, bufLine);
  else gl.bindBuffer(gl.ARRAY_BUFFER, bufPts);
  const hasSize = gl.getParameter(gl.ARRAY_BUFFER_BINDING) === bufPts;
  gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, hasSize ? stride : 24, 0);
  gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, hasSize ? stride : 24, 12);
  if (hasSize) {
    gl.enableVertexAttribArray(aSize);
    gl.vertexAttribPointer(aSize, 1, gl.FLOAT, false, stride, 24);
  } else {
    gl.disableVertexAttribArray(aSize);
    gl.vertexAttrib1f(aSize, 2.0);
  }
}

function syncRings() {
  if (!glReady) return;
  const pts = [];
  const cy_ = hex2rgb('#ffd23f'), cr = hex2rgb('#e06060');
  function ring(h2, h3, r, col) {
    const N = 56;
    for (let i = 0; i < N; i++) {
      const a0 = i / N * 2 * Math.PI, a1 = (i + 1) / N * 2 * Math.PI;
      pts.push(h2 + r * Math.cos(a0), 0, h3 + r * Math.sin(a0), ...col,
               h2 + r * Math.cos(a1), 0, h3 + r * Math.sin(a1), ...col);
    }
  }
  for (const d of DETECTED) ring(d[0], d[1], d[2], cy_);
  for (const c of circles) ring(c.h2, c.h3, c.r, cr);
  nRing = pts.length / 6;
  if (!bufRing) bufRing = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bufRing);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pts), gl.STATIC_DRAW);
  render3d();
}

function mPersp(fov, asp, n, f) {
  const t = 1 / Math.tan(fov / 2), d = 1 / (n - f);
  return [t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (n + f) * d, -1, 0, 0, 2 * n * f * d, 0];
}
function mLookAt(e, c, up) {
  let zx = e[0] - c[0], zy = e[1] - c[1], zz = e[2] - c[2];
  let l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  l = Math.hypot(xx, xy, xz); xx /= l; xy /= l; xz /= l;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  return [xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
          -(xx * e[0] + xy * e[1] + xz * e[2]),
          -(yx * e[0] + yy * e[1] + yz * e[2]),
          -(zx * e[0] + zy * e[1] + zz * e[2]), 1];
}
function mMul(a, b) {
  const o = new Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
    o[i * 4 + j] = s;
  }
  return o;
}

function render3d() {
  if (!glReady) return;
  gl.viewport(0, 0, cv3.width, cv3.height);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const cyw = Math.cos(cam.yaw), syw = Math.sin(cam.yaw);
  const eye = [cam.cx + cam.dist * cp * syw, cam.cy + cam.dist * sp, cam.cz + cam.dist * cp * cyw];
  const asp = cv3.width / cv3.height;
  const mvp = mMul(mPersp(40 * Math.PI / 180, asp, 50, 9000), mLookAt(eye, [cam.cx, cam.cy, cam.cz], [0, 0, 1]));
  gl.uniformMatrix4fv(gl.getUniformLocation(prog, 'uMVP'), false, new Float32Array(mvp));
  gl.uniform1f(gl.getUniformLocation(prog, 'uPtH'), cv3.height / (2 * Math.tan(20 * Math.PI / 180)) * 0.0042);
  // 点
  gl.bindBuffer(gl.ARRAY_BUFFER, bufPts);
  const stride = 28;
  const aPos = gl.getAttribLocation(prog, 'aPos');
  const aCol = gl.getAttribLocation(prog, 'aCol');
  const aSize = gl.getAttribLocation(prog, 'aSize');
  gl.enableVertexAttribArray(aPos); gl.enableVertexAttribArray(aCol); gl.enableVertexAttribArray(aSize);
  gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, stride, 0);
  gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, stride, 12);
  gl.vertexAttribPointer(aSize, 1, gl.FLOAT, false, stride, 24);
  gl.drawArrays(gl.POINTS, 0, nVerts);
  // 线（骨架 + 圆环）：pos3+col3
  gl.disableVertexAttribArray(aSize);
  gl.vertexAttrib1f(aSize, 2.0);
  gl.lineWidth(2);
  if (nLine > 0) {
    gl.bindBuffer(gl.ARRAY_BUFFER, bufLine);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 24, 0);
    gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 24, 12);
    gl.drawArrays(gl.LINES, 0, nLine);
  }
  if (nRing > 0) {
    gl.bindBuffer(gl.ARRAY_BUFFER, bufRing);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 24, 0);
    gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 24, 12);
    gl.drawArrays(gl.LINES, 0, nRing);
  }
}

function setMode(m) {
  mode = m;
  document.getElementById('m2d').classList.toggle('active', m === '2d');
  document.getElementById('m3d').classList.toggle('active', m === '3d');
  cv.style.display = m === '2d' ? 'block' : 'none';
  cv3.style.display = m === '3d' ? 'block' : 'none';
  document.getElementById('hint2d').style.display = m === '2d' ? '' : 'none';
  document.getElementById('hint3d').style.display = m === '3d' ? '' : 'none';
  document.getElementById('resetBtn').style.display = m === '3d' ? '' : 'none';
  if (m === '3d') { initGL(); render3d(); }
  else draw();
}
function resetView() {
  cam.yaw = -0.85; cam.pitch = 0.5; cam.dist = 2600; cam.cx = 0; cam.cy = 550; cam.cz = 0;
  render3d();
}
cv3.addEventListener('mousedown', e => {
  dragging = [e.clientX, e.clientY];
});
window.addEventListener('mousemove', e => {
  if (!dragging || mode !== '3d') return;
  const dx = e.clientX - dragging[0], dy = e.clientY - dragging[1];
  dragging = [e.clientX, e.clientY];
  cam.yaw -= dx * 0.006;
  cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + dy * 0.006));
  render3d();
});
window.addEventListener('mouseup', () => { dragging = null; });
cv3.addEventListener('wheel', e => {
  e.preventDefault();
  cam.dist = Math.max(200, Math.min(9000, cam.dist * (e.deltaY > 0 ? 1.12 : 0.89)));
  render3d();
}, { passive: false });

function renderTable() {
  const tb = document.querySelector('#tbl tbody');
  tb.innerHTML = '';
  circles.forEach((c, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${i + 1}</td><td>${c.h2.toFixed(1)}</td>
      <td>${c.h3.toFixed(1)}</td><td>${c.r.toFixed(1)}</td>
      <td>${(2 * c.r).toFixed(1)}</td>
      <td class="del" onclick="del(${i})">×</td>`;
    tb.appendChild(tr);
  });
}

cv.addEventListener('mousedown', e => {
  if (mode !== '2d') return;
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
  if (!drawing || mode !== '2d') return;
  const r = cv.getBoundingClientRect();
  const [cx, cy] = toPx(drawing.h2, drawing.h3);
  const rad = Math.hypot(mouse[0] - cx, mouse[1] - cy) / scale;
  if (rad > 3) { circles.push({ h2: drawing.h2, h3: drawing.h3, r: rad }); }
  drawing = null; mouse = null; renderTable(); draw(); syncRings();
});
function del(i) { circles.splice(i, 1); renderTable(); draw(); syncRings(); }
function clearAll() { if (confirm('清空全部 GT 圆？')) { circles = []; renderTable(); draw(); syncRings(); } }
function undo() { circles.pop(); renderTable(); draw(); syncRings(); }
function toggleLayers() {
  showGt = !showGt;
  document.getElementById('layerBtn').textContent = showGt ? '隐藏 GT 蓝点' : '显示 GT 蓝点';
  if (glReady) rebuildPoints();  // 3D 点 VBO 建好后固化，开关需重建
  draw();
}
// GT 蓝点开关后重建点 VBO（点数据在建 VBO 时固化；旧的 program/buffer
// 随 context 覆盖，切换次数少不清理）
function rebuildPoints() {
  glReady = false; bufPts = null; bufRing = null; nRing = 0;
  initGL();
}
function exportJson() {
  const payload = {
    meta: { face_x: BBOX.face_x, sleeve_end: BBOX.sleeve_end, n: circles.length },
    rebars: circles.map(c => ({ h2: +c.h2.toFixed(2), h3: +c.h3.toFixed(2), r_mm: +c.r.toFixed(2) }))
  };
  const txt = JSON.stringify(payload, null, 1);
  document.getElementById('out').value = txt;
  const blob = new Blob([txt], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'rebar_gt_manual.json';
  a.click();
}
draw(); renderTable();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="真实钢筋插入位置人工标注工具（2D 画圆 + WebGL 三维核对）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache", type=str, default=None,
                        help="推理缓存 npz（pts/labels）；缺省或不存在时自动推理")
    parser.add_argument("--input", type=str,
                        default="real_resource/scene_full_mm - Cloud.ply",
                        help="全场景点云（缓存缺失时使用）")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="分割权重（缺省 out/checkpoints/seg_v21_pointmlp_ce/best.pth）")
    parser.add_argument("--gt", type=str, default="real_resource/rebar_only.ply",
                        help="手筛钢筋 GT ply（端带投影作蓝色参考层；传空串关闭）")
    parser.add_argument("--band", type=float, default=120.0,
                        help="端带半宽 mm（离端面多远内的点参与渲染）")
    parser.add_argument("--out-dir", type=str, default="out/rebar_gt")
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
            ckpt_path = anchor_project_path(args.ckpt) if args.ckpt else \
                os.path.join(CHECKPOINT_BASE, "seg_v21_pointmlp_ce", "best.pth")
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
        # 钢筋端 = 套筒端对面（外露纵筋束所在端）
        se = frame["sleeve_end"]
        re_end = "neg" if se == "pos" else "pos"
        pl = frame[f"end_{re_end}_plane"]
        c0, u = frame["axis_point"], frame["axis_dir"]
        e2, e3 = frame["e2"], frame["e3"]
        end_x = frame["end_neg_x"] if re_end == "neg" else frame["end_pos_x"]
        sign = -1.0 if re_end == "neg" else 1.0
        rel = pts - c0
        t = rel @ u
        band = np.abs(t - end_x) < args.band

        def _proj3d(mask_pts):
            """端带点 → (h2, t_rel, h3, h)；t_rel 相对端面、外露方向为正。"""
            h2 = (mask_pts - c0) @ e2
            h3 = (mask_pts - c0) @ e3
            tr = (mask_pts - c0) @ u - end_x
            h = (mask_pts - pl["centroid"]) @ pl["normal"] * sign
            return h2, tr, h3, h - np.median(h)

        conc_m = band & (labels == 1)
        rebar_m = band & (labels == 2)
        h2c, tc, h3c, hc = _proj3d(pts[conc_m])
        # 只保留端面 ±25mm 高度内的混凝土点（剔除端带内的侧面点）
        _keep = (hc > -25.0) & (hc < 25.0)
        h2c, tc, h3c, hc = h2c[_keep], tc[_keep], h3c[_keep], hc[_keep]
        h2r, tr, h3r, _hr = _proj3d(pts[rebar_m])
        print(f"钢筋端 {re_end}（套筒端 {se}）| 端带 ±{args.band:.0f}mm："
              f"混凝土 {conc_m.sum()} 点，钢筋预测 {rebar_m.sum()} 点")

        # 手筛钢筋 GT（全云精确子集）投影到端带
        gtpts = []
        if args.gt:
            try:
                g = load_points(anchor_project_path(args.gt))
                tg = (g - c0) @ u
                gm = np.abs(tg - end_x) < args.band
                h2g, tg2, h3g, _ = _proj3d(g[gm])
                gtpts = np.column_stack([np.round(h2g, 1), np.round(tg2, 1),
                                         np.round(h3g, 1)])
                print(f"钢筋 GT 端带投影 {gm.sum()} 点")
            except Exception as e:
                print(f"（钢筋 GT 读取失败，跳过蓝点层：{e}）")

        # 降采样到各 ≤120k（保持渲染流畅）
        rng = np.random.default_rng(0)

        def _thin(cols, cap=120000):
            n = len(cols[0])
            if n <= cap:
                return [np.round(c, 1) for c in cols]
            idx = rng.choice(n, cap, replace=False)
            return [np.round(c[idx], 1) for c in cols]

        h2c, tc, h3c, hc = _thin([h2c, tc, h3c, hc])
        h2r, tr, h3r = _thin([h2r, tr, h3r])
        if len(gtpts) > 120000:
            idx = rng.choice(len(gtpts), 120000, replace=False)
            gtpts = gtpts[idx]
        conc_data = np.column_stack([h2c, tc, h3c, hc])
        rebar_data = np.column_stack([h2r, tr, h3r])
        all_h2 = np.concatenate([h2c, h2r] + ([gtpts[:, 0]] if len(gtpts) else []))
        all_h3 = np.concatenate([h3c, h3r] + ([gtpts[:, 2]] if len(gtpts) else []))
        bbox = {"xmin": float(all_h2.min()), "xmax": float(all_h2.max()),
                "ymin": float(all_h3.min()), "ymax": float(all_h3.max()),
                "face_x": float(end_x), "sleeve_end": se}
        # 自动识别插入位置参考（外露筋端面位置 + 估计直径）
        det = [[round(r["face_h2"], 1), round(r["face_h3"], 1),
                round(r["diam_est_mm"] / 2, 1)]
               for r in res["rebars"]["rebars"]]
        # 识别骨架线（帧坐标 [h2, t_rel, h3]，三维视图用）
        sks = []
        for r in res["rebars"]["rebars"]:
            line = np.asarray(r["centerline_mm"], dtype=np.float64)
            rl = line - c0
            sks.append(np.column_stack([rl @ e2, rl @ u - end_x, rl @ e3])
                       .round(1).tolist())
        print(f"识别插入位置参考 {len(det)} 个，骨架线 {len(sks)} 条")
        html = (HTML.replace("__CONC__", json.dumps(conc_data.tolist()))
                    .replace("__REBAR__", json.dumps(rebar_data.tolist()))
                    .replace("__GTPTS__", json.dumps(gtpts.tolist()))
                    .replace("__SKELETONS__", json.dumps(sks))
                    .replace("__BBOX__", json.dumps(bbox))
                    .replace("__DETECTED__", json.dumps(det)))
        out_dir = anchor_project_path(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "rebar_gt_tool.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"工具 → {out}")
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
