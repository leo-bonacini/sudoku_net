#!/usr/bin/env python3
"""
Small local web app around the training pipeline: serves the layer-introspection
dashboard and exposes a "Train model" button that kicks off train.py as a
background subprocess and reports live progress.

Run directly:
    python server.py
Or via Docker (see Dockerfile / docker-compose.yml in this folder).
"""
import base64
import json
import os
import random
import re
import subprocess
import sys
import threading
from pathlib import Path

import torch
from flask import Flask, Response, jsonify, request
from PIL import Image

from model import SudokuGridNet
from solver import find_conflicts, solve as solve_grid
from visualize import predict_grid, render_empty_state

HERE = Path(__file__).parent
DATA_DIR = Path(os.environ.get("SUDOKU_DATA_DIR", HERE / ".." / "dataset_gen" / "output"))
CHECKPOINT_DIR = Path(os.environ.get("SUDOKU_CHECKPOINT_DIR", HERE / "checkpoints"))
DASHBOARD_PATH = CHECKPOINT_DIR / "dashboard.html"

app = Flask(__name__)

_model_cache = {"model": None, "image_size": None, "mtime": None}


def _get_model():
    """Lazily loads best.pt, reloading whenever the checkpoint file changes
    (in particular, right after a training run finishes)."""
    ckpt_path = CHECKPOINT_DIR / "best.pt"
    if not ckpt_path.exists():
        return None, None
    mtime = ckpt_path.stat().st_mtime
    if _model_cache["model"] is None or _model_cache["mtime"] != mtime:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = SudokuGridNet().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        _model_cache.update(model=model, image_size=ckpt["args"]["image_size"], mtime=mtime)
    return _model_cache["model"], _model_cache["image_size"]

_lock = threading.Lock()
_state = {"status": "idle", "epoch": 0, "epochs": 0, "log": [], "error": None}
_EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+)")


def _run_training(epochs, batch_size):
    with _lock:
        _state.update(status="running", epoch=0, epochs=epochs, log=[], error=None)

    cmd = [sys.executable, "train.py", "--epochs", str(epochs), "--batch-size", str(batch_size)]
    proc = subprocess.Popen(cmd, cwd=str(HERE), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        with _lock:
            _state["log"].append(line)
            _state["log"] = _state["log"][-200:]
            m = _EPOCH_RE.search(line)
            if m:
                _state["epoch"] = int(m.group(1))
                _state["epochs"] = int(m.group(2))
    proc.wait()
    with _lock:
        _state["status"] = "done" if proc.returncode == 0 else "error"
        if proc.returncode != 0:
            _state["error"] = f"train.py exited with code {proc.returncode}"


@app.post("/train")
def start_training():
    with _lock:
        if _state["status"] == "running":
            return jsonify({"ok": False, "error": "Training already running"}), 409
    payload = request.get_json(silent=True) or {}
    epochs = int(payload.get("epochs", 30))
    batch_size = int(payload.get("batch_size", 32))
    threading.Thread(target=_run_training, args=(epochs, batch_size), daemon=True).start()
    return jsonify({"ok": True})


@app.get("/status")
def status():
    with _lock:
        return jsonify(dict(_state))


@app.post("/recognize")
def recognize():
    model, image_size = _get_model()
    if model is None:
        return jsonify({"ok": False, "error": "No trained checkpoint yet — train a model first."}), 400
    file = request.files.get("image")
    if not file:
        return jsonify({"ok": False, "error": "No image uploaded"}), 400
    try:
        img = Image.open(file.stream)
    except Exception:
        return jsonify({"ok": False, "error": "Could not read that file as an image"}), 400
    cells = predict_grid(model, img, image_size)
    return jsonify({"ok": True, "cells": cells})


@app.get("/sample")
def sample():
    labels_path = DATA_DIR / "labels.jsonl"
    if not labels_path.exists():
        return jsonify({"ok": False, "error": f"No dataset found at {DATA_DIR}"}), 400
    records = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    record = random.choice(records)
    img_path = DATA_DIR / "images" / record["file"]

    data_uri = "data:image/png;base64," + base64.b64encode(img_path.read_bytes()).decode("ascii")
    truth = [int(ch) for ch in record["grid"]]

    model, image_size = _get_model()
    cells = predict_grid(model, Image.open(img_path), image_size) if model is not None else None

    return jsonify({"ok": True, "image": data_uri, "truth": truth, "cells": cells})


@app.post("/solve")
def solve_endpoint():
    payload = request.get_json(silent=True) or {}
    flat = payload.get("grid")
    if not isinstance(flat, list) or len(flat) != 81:
        return jsonify({"ok": False, "error": "Expected a `grid` of 81 values"}), 400
    try:
        grid = [[int(flat[r * 9 + c]) for c in range(9)] for r in range(9)]
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Grid values must be integers 0-9"}), 400

    conflicts = find_conflicts(grid)
    if conflicts:
        return jsonify({"ok": False, "error": "That grid has conflicting digits", "conflicts": conflicts}), 400

    solved = [row[:] for row in grid]
    if not solve_grid(solved):
        return jsonify({"ok": False, "error": "No solution exists for this grid"}), 400
    return jsonify({"ok": True, "solved": [v for row in solved for v in row]})


@app.get("/")
def index():
    if DASHBOARD_PATH.exists():
        html = DASHBOARD_PATH.read_text()
    else:
        html = render_empty_state()
    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
