"""
Renders a self-contained HTML dashboard showing what a trained SudokuGridNet
has learned: per-stage activation heatmaps (what each downsampling stage
responds to on a real board image), the learned first-layer filters, and the
final layer's per-cell confidence grid.

Can be run standalone:
    python visualize.py --checkpoint checkpoints/best.pt \
        --data ../dataset_gen/output --file sample_0000.png --out dashboard.html

train.py calls build_dashboard() directly after training so the dashboard is
produced automatically at the end of a run.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import SudokuGridNet

# =====================================================================
# Sequential colormap: single hue (blue), light -> dark. 13 stops, 100-700.
# =====================================================================
_SEQ_HEX = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


_SEQ_RGB = np.array([_hex_to_rgb(h) for h in _SEQ_HEX], dtype=np.float32)


def sequential_colormap(t):
    """t: ndarray in [0,1] of any shape -> uint8 RGB array shape (...,3)."""
    t = np.clip(t, 0.0, 1.0)
    n = len(_SEQ_RGB) - 1
    pos = t * n
    i0 = np.clip(np.floor(pos).astype(int), 0, n - 1)
    i1 = i0 + 1
    frac = (pos - i0)[..., None]
    return (_SEQ_RGB[i0] * (1 - frac) + _SEQ_RGB[i1] * frac).astype(np.uint8)


def _png_data_uri(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _activation_overlay(base_uint8, act_hw, alpha_max=0.78):
    """Blends a per-pixel heat color into the base image, with alpha driven by
    the (normalized) activation strength itself -- low-response regions stay
    close to the original photo, high-response regions glow."""
    t = act_hw.astype(np.float32)
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    size = (base_uint8.shape[1], base_uint8.shape[0])

    color = sequential_colormap(t)
    color_img = np.array(Image.fromarray(color).resize(size, Image.BILINEAR)).astype(np.float32)

    alpha_img = Image.fromarray((t * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    alpha = (np.array(alpha_img).astype(np.float32) / 255.0)[..., None] * alpha_max

    out = base_uint8.astype(np.float32) * (1 - alpha) + color_img * alpha
    return Image.fromarray(out.astype(np.uint8))


def _filter_thumb(weight_3ch, upscale=16):
    """weight_3ch: (in_ch=3, kh, kw) conv kernel -> upscaled RGB thumbnail."""
    patch = weight_3ch.transpose(1, 2, 0)  # (kh, kw, RGB)
    lo, hi = patch.min(), patch.max()
    norm = (patch - lo) / (hi - lo + 1e-6)
    img = Image.fromarray((norm * 255).astype(np.uint8), mode="RGB")
    side = img.width * upscale
    return img.resize((side, side), Image.NEAREST)


# =====================================================================
# Forward pass with activation hooks
# =====================================================================
def _run_with_hooks(model, tensor):
    activations = {}

    def hook(name):
        def fn(_module, _inp, out):
            activations[name] = out.detach()
        return fn

    handles = [block.register_forward_hook(hook(f"stage{i + 1}"))
               for i, block in enumerate(model.backbone)]
    with torch.no_grad():
        logits = model(tensor)
    for h in handles:
        h.remove()
    return activations, logits


def _cells_from_probs(probs, label=None):
    """probs: (10,9,9) softmax output -> list of 81 {r,c,pred,conf[,truth,correct]}."""
    pred = probs.argmax(axis=0)
    conf = probs.max(axis=0)
    cells = []
    for r in range(9):
        for c in range(9):
            entry = {"r": r, "c": c, "pred": int(pred[r, c]), "conf": float(conf[r, c])}
            if label is not None:
                truth = int(label[r, c])
                entry["truth"] = truth
                entry["correct"] = truth == entry["pred"]
            cells.append(entry)
    return cells


def predict_grid(model, pil_image, image_size):
    """Runs the model on an arbitrary PIL image and returns the 81-cell prediction list.
    Used by server.py's /recognize and /sample endpoints (no activation hooks needed)."""
    model.eval()
    device = next(model.parameters()).device
    img = pil_image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    tensor = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    return _cells_from_probs(probs)


def _load_sample(data_root, record, image_size):
    path = Path(data_root) / "images" / record["file"]
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    base_uint8 = np.array(img)
    tensor = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])(img).unsqueeze(0)
    label = None
    if "grid" in record:
        label = torch.tensor([int(c) for c in record["grid"]], dtype=torch.long).view(9, 9)
    return base_uint8, tensor, label


# =====================================================================
# Dashboard build
# =====================================================================
def build_dashboard(model, data_root, record, out_html, image_size=288,
                     checkpoint_meta=None, history=None):
    model.eval()
    device = next(model.parameters()).device
    base_uint8, tensor, label = _load_sample(data_root, record, image_size)
    activations, logits = _run_with_hooks(model, tensor.to(device))

    stage_cards = []
    for i in range(1, len(model.backbone) + 1):
        act_t = activations[f"stage{i}"][0]  # (C,H,W)
        act_map = act_t.abs().mean(dim=0).cpu().numpy()
        overlay = _activation_overlay(base_uint8, act_map)
        c, h, w = act_t.shape
        stage_cards.append({
            "index": i,
            "uri": _png_data_uri(overlay),
            "channels": int(c), "h": int(h), "w": int(w),
            "mean_act": float(act_t.abs().mean()),
            "peak_act": float(act_t.abs().max()),
        })

    conv1 = model.backbone[0][0]
    weights = conv1.weight.detach().cpu().numpy()  # (32,3,3,3)
    norms = np.linalg.norm(weights.reshape(len(weights), -1), axis=1)
    order = np.argsort(-norms)
    filter_cards = [{
        "index": int(f),
        "uri": _png_data_uri(_filter_thumb(weights[f])),
        "norm": float(norms[f]),
    } for f in order]

    probs = F.softmax(logits, dim=1)[0].cpu().numpy()  # (10,9,9)
    cells = _cells_from_probs(probs, label)
    cell_accuracy = sum(c["correct"] for c in cells) / 81 if label is not None else None

    meta = dict(checkpoint_meta or {})
    meta["image_file"] = record["file"]
    meta["sample_cell_accuracy"] = cell_accuracy

    html = _render_html(meta, stage_cards, filter_cards, cells, history)
    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return str(out_path)


def _digit(v):
    return "&middot;" if v == 0 else str(v)


def _line_chart_svg(history):
    if not history or len(history) < 2:
        return ""
    W, H = 640, 220
    pad_l, pad_r, pad_t, pad_b = 36, 46, 16, 28
    n = len(history)

    def xs(i):
        return pad_l + (W - pad_l - pad_r) * (i / (n - 1))

    def ys(v):
        return pad_t + (H - pad_t - pad_b) * (1 - v)

    def path_for(key):
        pts = [(xs(i), ys(h[key])) for i, h in enumerate(history)]
        return "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    grid = []
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        grid.append(f'<line x1="{pad_l}" y1="{ys(g):.1f}" x2="{W - pad_r}" y2="{ys(g):.1f}" class="gridline" />')
        grid.append(f'<text x="{pad_l - 8}" y="{ys(g) + 4:.1f}" class="axis-label" text-anchor="end">{int(g * 100)}</text>')

    last = history[-1]
    return f'''<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Validation accuracy across training epochs">
{"".join(grid)}
<path d="{path_for("cell_acc")}" class="line line-accent" />
<path d="{path_for("board_acc")}" class="line line-series2" />
<circle cx="{xs(n - 1):.1f}" cy="{ys(last['cell_acc']):.1f}" r="3.5" class="dot-accent" />
<circle cx="{xs(n - 1):.1f}" cy="{ys(last['board_acc']):.1f}" r="3.5" class="dot-series2" />
<text x="{xs(n - 1) + 7:.1f}" y="{ys(last['cell_acc']) + 4:.1f}" class="end-label end-accent">{last['cell_acc'] * 100:.1f}%</text>
<text x="{xs(n - 1) + 7:.1f}" y="{ys(last['board_acc']) + 4:.1f}" class="end-label end-series2">{last['board_acc'] * 100:.1f}%</text>
<text x="{pad_l}" y="{H - 6}" class="axis-label">epoch 1</text>
<text x="{W - pad_r}" y="{H - 6}" class="axis-label" text-anchor="end">epoch {n}</text>
</svg>'''


_CSS = """
:root, :root[data-theme="light"] {
  --surface: #fcfcfb;
  --page: #f9f9f7;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid-line: #e1e0d9;
  --border: rgba(11,11,11,0.10);
  --accent: #2a78d6;
  --accent-rgb: 42,120,214;
  --series2: #1baf7a;
  --series2-rgb: 27,175,122;
  --good: #0ca30c;
  --critical: #d03b3b;
  --shadow: 0 1px 2px rgba(11,11,11,0.04), 0 8px 24px rgba(11,11,11,0.05);
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid-line: #2c2c2a; --border: rgba(255,255,255,0.10);
    --accent: #3987e5; --accent-rgb: 59,135,229; --series2: #199e70; --series2-rgb: 25,158,112;
    --good: #0ca30c; --critical: #e66767;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.35);
  }
}
:root[data-theme="dark"] {
  --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid-line: #2c2c2a; --border: rgba(255,255,255,0.10);
  --accent: #3987e5; --accent-rgb: 59,135,229; --series2: #199e70; --series2-rgb: 25,158,112;
  --good: #0ca30c; --critical: #e66767;
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.35);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
.mono { font-family: ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", monospace; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 40px 24px 80px; }

.eyebrow {
  font-size: 12px; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--accent); margin: 0 0 8px;
}
h1 { font-size: 28px; font-weight: 700; margin: 0 0 6px; text-wrap: balance; }
.subtitle { color: var(--ink-2); font-size: 14px; margin: 0; }
.subtitle .mono { color: var(--muted); }

section { margin-top: 40px; }
h2 { font-size: 15px; font-weight: 700; margin: 0 0 4px; }
.section-note { color: var(--ink-2); font-size: 13px; margin: 0 0 16px; max-width: 62ch; }

.stat-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.stat-tile {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 18px; box-shadow: var(--shadow);
}
.stat-tile .value { font-size: 26px; font-weight: 700; line-height: 1.1; }
.stat-tile .label { font-size: 12px; color: var(--ink-2); margin-top: 4px; }

.chart-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 18px 20px; box-shadow: var(--shadow);
}
.legend { display: flex; gap: 18px; font-size: 12px; color: var(--ink-2); margin-bottom: 10px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.chart { width: 100%; height: auto; display: block; }
.gridline { stroke: var(--grid-line); stroke-width: 1; }
.axis-label { fill: var(--muted); font-size: 10px; font-family: ui-monospace, monospace; }
.line { fill: none; stroke-width: 2; }
.line-accent { stroke: var(--accent); }
.line-series2 { stroke: var(--series2); }
.dot-accent { fill: var(--accent); }
.dot-series2 { fill: var(--series2); }
.end-label { font-size: 11px; font-weight: 600; font-family: ui-monospace, monospace; }
.end-accent { fill: var(--accent); }
.end-series2 { fill: var(--series2); }

.filmstrip { display: flex; gap: 14px; overflow-x: auto; padding-bottom: 6px; }
.stage-card {
  flex: 0 0 200px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; overflow: hidden; box-shadow: var(--shadow);
}
.stage-card img { width: 100%; display: block; background: var(--page); }
.stage-card .stage-body { padding: 10px 12px 12px; }
.stage-card .stage-num { font-size: 11px; font-weight: 700; color: var(--accent); letter-spacing: 0.04em; }
.stage-card .stage-shape { font-size: 12px; margin-top: 2px; }
.stage-card .stage-stats { font-size: 11px; color: var(--muted); margin-top: 4px; }

.filters-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(52px, 1fr)); gap: 8px;
}
[data-tip] { position: relative; }
[data-tip]:hover::after {
  content: attr(data-tip);
  position: absolute; bottom: calc(100% + 6px); left: 50%; transform: translateX(-50%);
  background: var(--ink); color: var(--page); padding: 4px 8px; border-radius: 6px;
  font-size: 11px; white-space: nowrap; pointer-events: none; z-index: 30;
  box-shadow: var(--shadow); font-family: ui-monospace, monospace;
}
.filter-thumb {
  aspect-ratio: 1; border-radius: 6px; overflow: hidden; border: 1px solid var(--border);
}
.filter-thumb img { width: 100%; height: 100%; display: block; image-rendering: pixelated; }

.board-grid-wrap { display: flex; gap: 28px; flex-wrap: wrap; align-items: flex-start; }
.board-grid {
  display: grid; grid-template-columns: repeat(9, 1fr); width: min(100%, 420px);
  aspect-ratio: 1; border: 2px solid var(--ink); border-radius: 8px; overflow: hidden;
  background: var(--surface);
}
.board-cell {
  display: flex; align-items: center; justify-content: center;
  font-size: clamp(12px, 2.6vw, 18px); font-weight: 600; position: relative;
  border: 1px solid var(--grid-line);
}
.board-cell:nth-child(3n) { border-right: 2px solid var(--ink-2); }
.board-cell:nth-child(9n+1) { border-left: 2px solid var(--ink-2); }
.board-cell:nth-child(n+19):nth-child(-n+27),
.board-cell:nth-child(n+46):nth-child(-n+54) { border-bottom: 2px solid var(--ink-2); }
.board-cell .dot {
  position: absolute; top: 3px; right: 3px; width: 6px; height: 6px; border-radius: 50%;
}
.board-legend { display: flex; flex-direction: column; gap: 14px; font-size: 12px; color: var(--ink-2); min-width: 200px; }
.confidence-bar {
  height: 10px; border-radius: 5px;
  background: linear-gradient(90deg, rgba(var(--accent-rgb),0.08), rgba(var(--accent-rgb),0.9));
}
.confidence-scale { display: flex; justify-content: space-between; font-size: 11px; margin-top: 4px; font-family: ui-monospace, monospace; }
.dot-legend { display: flex; align-items: center; gap: 6px; }
.dot-legend .dot { position: static; width: 8px; height: 8px; border-radius: 50%; }

footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--border); font-size: 12px; color: var(--muted); }

.train-bar {
  position: sticky; top: 0; z-index: 100;
  background: var(--surface); border-bottom: 1px solid var(--border); box-shadow: var(--shadow);
  padding: 10px 24px; display: flex; align-items: center; gap: 12px;
}
.train-bar .spacer { flex: 1; }
.train-bar label { font-size: 12px; color: var(--ink-2); display: flex; align-items: center; gap: 6px; }
.train-bar input[type="number"] {
  width: 60px; font-family: ui-monospace, monospace; font-size: 12px;
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 6px;
  background: var(--page); color: var(--ink);
}
.train-btn {
  background: var(--accent); color: #fff; border: none; border-radius: 8px;
  padding: 8px 16px; font-weight: 600; font-size: 13px; cursor: pointer; font-family: inherit;
}
.train-btn:disabled { opacity: 0.55; cursor: default; }
.train-status { font-size: 12px; color: var(--ink-2); font-family: ui-monospace, monospace; }
.train-status.is-error { color: var(--critical); }

.tabs { display: flex; gap: 20px; margin-top: 28px; border-bottom: 1px solid var(--border); }
.tab-btn {
  background: none; border: none; padding: 10px 2px; font: inherit;
  font-size: 13px; font-weight: 600; color: var(--ink-2); cursor: pointer;
  border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.tab-btn.active { color: var(--ink); border-bottom-color: var(--accent); }
.tab-panel[hidden] { display: none; }

.solve-row { display: flex; gap: 28px; flex-wrap: wrap; align-items: flex-start; margin-top: 16px; }
.dropzone {
  flex: 1 1 260px; min-height: 220px; border: 2px dashed var(--border); border-radius: 10px;
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px;
  padding: 20px; text-align: center; color: var(--ink-2); font-size: 13px; cursor: pointer;
  background: var(--surface);
}
.dropzone.drag-over { border-color: var(--accent); background: rgba(var(--accent-rgb), 0.06); }
.dropzone img { max-width: 100%; max-height: 240px; border-radius: 6px; }
.solve-col { flex: 1 1 320px; }
.solve-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
.train-btn.primary { background: var(--accent); }
.train-btn.secondary { background: var(--page); color: var(--ink); border: 1px solid var(--border); }
.board-grid input.board-cell {
  width: 100%; height: 100%; border: none; background: transparent; outline: none;
  text-align: center; font: inherit; color: inherit; padding: 0;
}
.board-grid input.board-cell:focus { outline: 2px solid var(--accent); outline-offset: -2px; z-index: 2; }
.board-cell.given { font-weight: 700; }
.board-grid input.board-cell.filled-by-solver { background-color: rgba(var(--series2-rgb), 0.18); color: var(--series2); font-weight: 700; }
"""


def _control_bar_html(default_epochs=30):
    return f'''<div class="train-bar">
  <button id="train-btn" class="train-btn" type="button">Train model</button>
  <label>epochs <input id="train-epochs" type="number" min="1" value="{default_epochs}" /></label>
  <div class="spacer"></div>
  <span id="train-status" class="train-status">idle</span>
</div>
<script>
(function () {{
  var btn = document.getElementById("train-btn");
  var epochsInput = document.getElementById("train-epochs");
  var statusEl = document.getElementById("train-status");
  var polling = null;

  function render(s) {{
    statusEl.classList.remove("is-error");
    if (s.status === "running") {{
      statusEl.textContent = "training \\u2026 epoch " + s.epoch + "/" + s.epochs;
      btn.disabled = true;
    }} else if (s.status === "done") {{
      statusEl.textContent = "done \\u2014 reloading \\u2026";
      btn.disabled = true;
      clearInterval(polling);
      setTimeout(function () {{ location.reload(); }}, 800);
    }} else if (s.status === "error") {{
      statusEl.textContent = "error: " + (s.error || "unknown");
      statusEl.classList.add("is-error");
      btn.disabled = false;
      clearInterval(polling);
    }} else {{
      statusEl.textContent = "idle";
      btn.disabled = false;
    }}
  }}

  function poll() {{
    fetch("/status").then(function (r) {{ return r.json(); }}).then(render).catch(function () {{
      statusEl.textContent = "no training server running (static preview)";
    }});
  }}

  btn.addEventListener("click", function () {{
    btn.disabled = true;
    statusEl.textContent = "starting \\u2026";
    fetch("/train", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ epochs: parseInt(epochsInput.value, 10) || 30 }})
    }}).then(function (r) {{
      if (!r.ok) return r.json().then(function (e) {{ throw new Error(e.error || "failed to start"); }});
      polling = setInterval(poll, 1500);
      poll();
    }}).catch(function (e) {{
      statusEl.textContent = "error: " + e.message;
      statusEl.classList.add("is-error");
      btn.disabled = false;
    }});
  }});

  poll();
}})();
</script>'''


_TAB_SWITCH_JS = '''<script>
document.querySelectorAll(".tab-btn").forEach(function (btn) {
  btn.addEventListener("click", function () {
    document.querySelectorAll(".tab-btn").forEach(function (b) { b.classList.remove("active"); });
    document.querySelectorAll(".tab-panel").forEach(function (p) { p.hidden = true; });
    btn.classList.add("active");
    document.querySelector('.tab-panel[data-panel="' + btn.dataset.tab + '"]').hidden = false;
  });
});
</script>'''


def _tabs_html():
    return '''<div class="tabs" role="tablist">
  <button type="button" class="tab-btn active" data-tab="intro" role="tab">Model introspection</button>
  <button type="button" class="tab-btn" data-tab="solve" role="tab">Upload &amp; solve</button>
</div>'''


def _solve_tab_html():
    return '''<div class="tab-panel" data-panel="solve" hidden>
  <section>
    <h2>Upload a board, get the solved answer</h2>
    <p class="section-note">Drop in a photo of a Sudoku board (or try a sample from the training set).
    &ldquo;Solve from image&rdquo; reads all 81 cells with the model above and solves the puzzle in one step &mdash;
    the grid stays editable if a cell was misread, so you can fix it and solve again.</p>
    <div class="solve-row">
      <div class="dropzone" id="dropzone" tabindex="0" role="button" aria-label="Choose or drop a board image">
        <span id="dropzone-label">Drop an image here, or click to choose</span>
        <img id="preview-img" hidden alt="Selected board" />
        <input type="file" id="file-input" accept="image/*" hidden />
      </div>
      <div class="solve-col">
        <div class="board-grid" id="solve-grid"></div>
        <div class="solve-actions">
          <button type="button" id="sample-btn" class="train-btn secondary">Try a sample</button>
          <button type="button" id="upload-solve-btn" class="train-btn primary" disabled>Solve from image</button>
          <button type="button" id="solve-btn" class="train-btn secondary">Solve grid</button>
          <button type="button" id="clear-btn" class="train-btn secondary">Clear</button>
        </div>
        <div id="solve-status" class="train-status" style="margin-top:8px">no image selected</div>
        <div class="board-legend" style="flex-direction:row;gap:18px;margin-top:12px">
          <div class="dot-legend"><span class="dot" style="background:var(--accent)"></span>read from the image</div>
          <div class="dot-legend"><span class="dot" style="background:var(--series2)"></span>filled in by the solver</div>
        </div>
      </div>
    </div>
  </section>
</div>
<script>
(function () {
  var grid = document.getElementById("solve-grid");
  var inputs = [];
  for (var i = 0; i < 81; i++) {
    var inp = document.createElement("input");
    inp.className = "board-cell mono";
    inp.maxLength = 1;
    inp.inputMode = "numeric";
    inp.autocomplete = "off";
    (function (el) {
      el.addEventListener("input", function () {
        el.value = el.value.replace(/[^1-9]/g, "").slice(0, 1);
        el.classList.remove("given", "filled-by-solver");
        el.style.backgroundColor = "";
      });
    })(inp);
    grid.appendChild(inp);
    inputs.push(inp);
  }

  function setCells(cells) {
    cells.forEach(function (cell) {
      var inp = inputs[cell.r * 9 + cell.c];
      inp.classList.remove("given", "filled-by-solver");
      if (!cell.pred) {
        inp.value = "";
        inp.style.backgroundColor = "";
      } else {
        inp.value = String(cell.pred);
        inp.classList.add("given");
        var alpha = 0.12 + 0.80 * (cell.conf != null ? cell.conf : 1);
        inp.style.backgroundColor = "rgba(var(--accent-rgb), " + alpha.toFixed(3) + ")";
      }
    });
  }

  function currentGrid() {
    return inputs.map(function (inp) { return parseInt(inp.value, 10) || 0; });
  }

  function clearGrid() {
    inputs.forEach(function (inp) {
      inp.value = "";
      inp.className = "board-cell mono";
      inp.style.backgroundColor = "";
    });
  }

  var dropzone = document.getElementById("dropzone");
  var fileInput = document.getElementById("file-input");
  var previewImg = document.getElementById("preview-img");
  var dropzoneLabel = document.getElementById("dropzone-label");
  var uploadSolveBtn = document.getElementById("upload-solve-btn");
  var sampleBtn = document.getElementById("sample-btn");
  var solveBtn = document.getElementById("solve-btn");
  var clearBtn = document.getElementById("clear-btn");
  var statusEl = document.getElementById("solve-status");
  var currentFile = null;

  function setStatus(msg, isError) {
    statusEl.textContent = msg;
    statusEl.classList.toggle("is-error", !!isError);
  }

  function showPreview(src) {
    previewImg.src = src;
    previewImg.hidden = false;
    dropzoneLabel.hidden = true;
  }

  function pickFile(file) {
    if (!file) return;
    currentFile = file;
    uploadSolveBtn.disabled = false;
    var reader = new FileReader();
    reader.onload = function () { showPreview(reader.result); };
    reader.readAsDataURL(file);
    setStatus("image selected \\u2014 click \\u201cSolve from image\\u201d");
  }

  dropzone.addEventListener("click", function () { fileInput.click(); });
  dropzone.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
  dropzone.addEventListener("dragover", function (e) { e.preventDefault(); dropzone.classList.add("drag-over"); });
  dropzone.addEventListener("dragleave", function () { dropzone.classList.remove("drag-over"); });
  dropzone.addEventListener("drop", function (e) {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) pickFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", function () { pickFile(fileInput.files[0]); });

  function recognize(file) {
    var form = new FormData();
    form.append("image", file);
    return fetch("/recognize", { method: "POST", body: form }).then(function (r) {
      return r.json().then(function (d) { return { ok: r.ok, d: d }; });
    }).then(function (res) {
      if (!res.ok || !res.d.ok) throw new Error(res.d.error || "recognition failed");
      return res.d.cells;
    });
  }

  function solveGrid(flatGrid) {
    return fetch("/solve", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ grid: flatGrid })
    }).then(function (r) {
      return r.json().then(function (d) { return { ok: r.ok, d: d }; });
    }).then(function (res) {
      if (!res.ok || !res.d.ok) throw new Error(res.d.error || "no solution");
      return res.d.solved;
    });
  }

  function applySolution(before, solved) {
    solved.forEach(function (v, i) {
      var inp = inputs[i];
      inp.value = String(v);
      inp.style.backgroundColor = "";
      if (before[i] === 0) {
        inp.classList.remove("given");
        inp.classList.add("filled-by-solver");
      }
    });
  }

  uploadSolveBtn.addEventListener("click", function () {
    if (!currentFile) return;
    uploadSolveBtn.disabled = true;
    setStatus("reading grid \\u2026");
    recognize(currentFile).then(function (cells) {
      clearGrid();
      setCells(cells);
      setStatus("solving \\u2026");
      var before = currentGrid();
      return solveGrid(before).then(function (solved) {
        applySolution(before, solved);
        setStatus("solved");
      });
    }).catch(function (e) {
      setStatus("error: " + e.message, true);
    }).finally(function () { uploadSolveBtn.disabled = false; });
  });

  sampleBtn.addEventListener("click", function () {
    setStatus("loading sample \\u2026");
    fetch("/sample").then(function (r) {
      return r.json().then(function (d) { return { ok: r.ok, d: d }; });
    }).then(function (res) {
      if (!res.ok || !res.d.ok) throw new Error(res.d.error || "failed to load sample");
      var d = res.d;
      showPreview(d.image);
      currentFile = null;
      uploadSolveBtn.disabled = true;
      clearGrid();
      if (d.cells) {
        setCells(d.cells);
        if (d.truth) {
          var correct = 0;
          d.cells.forEach(function (cell) { if (cell.pred === d.truth[cell.r * 9 + cell.c]) correct++; });
          setStatus("sample loaded \\u2014 model matched " + correct + "/81 ground-truth cells");
        } else {
          setStatus("sample loaded \\u2014 grid read from image");
        }
      } else {
        setStatus("sample loaded (no trained model to read it yet)");
      }
    }).catch(function (e) { setStatus("error: " + e.message, true); });
  });

  solveBtn.addEventListener("click", function () {
    var before = currentGrid();
    setStatus("solving \\u2026");
    solveGrid(before).then(function (solved) {
      applySolution(before, solved);
      setStatus("solved");
    }).catch(function (e) { setStatus("error: " + e.message, true); });
  });

  clearBtn.addEventListener("click", function () {
    clearGrid();
    previewImg.hidden = true;
    dropzoneLabel.hidden = false;
    currentFile = null;
    uploadSolveBtn.disabled = true;
    setStatus("no image selected");
  });
})();
</script>'''


def render_empty_state():
    """Page shown by server.py before any checkpoint/dashboard exists yet."""
    return f'''<title>SudokuGridNet &middot; training dashboard</title>
<style>{_CSS}</style>
{_control_bar_html()}
<div class="wrap">
  <p class="eyebrow">Model introspection</p>
  <h1>No trained model yet</h1>
  {_tabs_html()}
  <div class="tab-panel" data-panel="intro">
    <p class="subtitle" style="margin-top:24px">Click &ldquo;Train model&rdquo; above to run a training pass over
    the generated dataset. This page will fill in with per-layer activations, learned filters, and the final
    prediction grid once it finishes.</p>
  </div>
  {_solve_tab_html()}
</div>
{_TAB_SWITCH_JS}'''


def _render_html(meta, stage_cards, filter_cards, cells, history):
    title = "SudokuGridNet &middot; training dashboard"

    stat_tiles = []
    if meta.get("best_board_acc") is not None:
        stat_tiles.append(("Best board accuracy", f"{meta['best_board_acc'] * 100:.1f}%"))
    if meta.get("best_cell_acc") is not None:
        stat_tiles.append(("Best cell accuracy", f"{meta['best_cell_acc'] * 100:.1f}%"))
    if meta.get("epochs") is not None:
        stat_tiles.append(("Epochs trained", str(meta["epochs"])))
    if meta.get("sample_cell_accuracy") is not None:
        stat_tiles.append(("This sample&rsquo;s cell accuracy", f"{meta['sample_cell_accuracy'] * 100:.1f}%"))
    stat_html = "".join(
        f'<div class="stat-tile"><div class="value">{v}</div><div class="label">{k}</div></div>'
        for k, v in stat_tiles
    )

    chart_section = ""
    svg = _line_chart_svg(history)
    if svg:
        chart_section = f'''
<section>
  <h2>Learning curve</h2>
  <p class="section-note">Validation accuracy per epoch. Board accuracy (every one of the 81 cells correct) is
  the harder, stricter measure of the two.</p>
  <div class="chart-card">
    <div class="legend">
      <span><i class="swatch" style="background:var(--accent)"></i>Cell accuracy</span>
      <span><i class="swatch" style="background:var(--series2)"></i>Board accuracy</span>
    </div>
    {svg}
  </div>
</section>'''

    stage_html = "".join(f'''
<div class="stage-card">
  <img src="{s['uri']}" alt="Stage {s['index']} activation heatmap" />
  <div class="stage-body">
    <div class="stage-num">STAGE {s['index']} / {len(stage_cards)}</div>
    <div class="stage-shape mono">{s['w']}&times;{s['h']} &middot; {s['channels']}ch</div>
    <div class="stage-stats mono">mean {s['mean_act']:.2f} &middot; peak {s['peak_act']:.2f}</div>
  </div>
</div>''' for s in stage_cards)

    filter_html = "".join(
        f'<div class="filter-thumb" data-tip="filter {f["index"]} &middot; norm {f["norm"]:.2f}">'
        f'<img src="{f["uri"]}" alt="Conv filter {f["index"]}" /></div>'
        for f in filter_cards
    )

    cell_html = []
    for cell in cells:
        alpha = 0.12 + 0.80 * cell["conf"]
        style = f'background-color: rgba(var(--accent-rgb), {alpha:.3f});'
        tip = f'row {cell["r"] + 1}, col {cell["c"] + 1} &middot; predicted {_digit(cell["pred"])} &middot; confidence {cell["conf"] * 100:.1f}%'
        dot = ""
        if "correct" in cell:
            tip += f' &middot; truth {_digit(cell["truth"])} &middot; {"correct" if cell["correct"] else "wrong"}'
            color = "var(--good)" if cell["correct"] else "var(--critical)"
            dot = f'<span class="dot" style="background:{color}"></span>'
        cell_html.append(
            f'<div class="board-cell mono" style="{style}" data-tip="{tip}">{_digit(cell["pred"])}{dot}</div>'
        )
    cell_html = "".join(cell_html)

    has_truth = any("correct" in c for c in cells)
    dot_legend = ""
    if has_truth:
        dot_legend = f'''
    <div>
      <div class="dot-legend"><span class="dot" style="background:var(--good)"></span>matches ground truth</div>
      <div class="dot-legend"><span class="dot" style="background:var(--critical)"></span>disagrees with ground truth</div>
    </div>'''

    meta_line = f'checkpoint epoch {meta.get("epoch", "?")} &middot; sample <span class="mono">{meta["image_file"]}</span>'

    return f'''<title>{title}</title>
<style>{_CSS}</style>
{_control_bar_html()}
<div class="wrap">
  <p class="eyebrow">Model introspection</p>
  <h1>What SudokuGridNet is focusing on</h1>
  <p class="subtitle">{meta_line}</p>

  {_tabs_html()}

  <div class="tab-panel" data-panel="intro">
    <section>
      <div class="stat-row">{stat_html}</div>
    </section>

    {chart_section}

    <section>
      <h2>Per-stage activations</h2>
      <p class="section-note">Each stage halves the spatial resolution and doubles (roughly) the channel depth.
      The heat overlay shows where that stage's activation magnitude is highest on this board &mdash;
      color intensity is scaled to the response itself, so faint regions are close to the raw photo.</p>
      <div class="filmstrip">{stage_html}</div>
    </section>

    <section>
      <h2>What the first layer learned</h2>
      <p class="section-note">The 32 kernels of the first convolution, each 3&times;3 over R, G, B &mdash;
      the only layer whose weights are directly viewable as color. Sorted by weight norm.</p>
      <div class="filters-grid">{filter_html}</div>
    </section>

    <section>
      <h2>Final layer: prediction per cell</h2>
      <p class="section-note">The output of the last 1&times;1 conv, one 10-way softmax per board cell.
      Background intensity encodes confidence in the predicted digit.</p>
      <div class="board-grid-wrap">
        <div class="board-grid">{cell_html}</div>
        <div class="board-legend">
          <div>
            <div>Confidence</div>
            <div class="confidence-bar"></div>
            <div class="confidence-scale mono"><span>low</span><span>high</span></div>
          </div>
          {dot_legend}
        </div>
      </div>
    </section>
  </div>

  {_solve_tab_html()}

  <footer>Generated by dataset_gen &rarr; train.py &rarr; visualize.py. Re-run <span class="mono">predict.py</span> or
  <span class="mono">visualize.py --checkpoint ... --file ...</span> for a different sample.</footer>
</div>
{_TAB_SWITCH_JS}
'''


# =====================================================================
# CLI
# =====================================================================
def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default=str(Path(__file__).parent / ".." / "dataset_gen" / "output"))
    p.add_argument("--file", default=None, help="Image filename inside --data/images (default: first in labels.jsonl)")
    p.add_argument("--out", default="dashboard.html")
    return p.parse_args()


def main():
    args = _cli()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    image_size = ckpt["args"]["image_size"]

    model = SudokuGridNet().to(device)
    model.load_state_dict(ckpt["model"])

    with open(Path(args.data) / "labels.jsonl") as f:
        records = [json.loads(line) for line in f if line.strip()]
    record = next((r for r in records if r["file"] == args.file), records[0]) if args.file else records[0]

    out = build_dashboard(model, args.data, record, args.out, image_size=image_size,
                           checkpoint_meta={"epoch": ckpt.get("epoch")})
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
