# Training pipeline

Trains a CNN to read a rendered Sudoku board image and output all 81 cell values,
using the images/labels produced by `dataset_gen/generate.js`.

## Topology

The model (`model.py`) is a fully-convolutional backbone that downsamples the
288x288 input by exactly 32x (5 stride-2 stages), landing on a 9x9 feature map
that is spatially aligned with the board's 9x9 cells. A final 1x1 conv maps
each of those 81 positions to a 10-way class score (0 = blank, 1-9 = digit),
giving an `(N, 10, 9, 9)` output.

This is the "81 outputs" idea, just implemented as one shared conv head over a
9x9 grid of features instead of 81 separate dense heads: same effective output,
far fewer parameters, and it doesn't need each cell hard-registered to a fixed
pixel offset the way a flatten-to-dense head would.

## Usage

```
pip install -r requirements.txt

# generate more data first if needed (see ../dataset_gen/run.sh)
python train.py --data ../dataset_gen/output --epochs 30 --out ./checkpoints

python predict.py --checkpoint checkpoints/best.pt --image ../dataset_gen/output/images/sample_0000.png
```

`train.py` reports, per epoch: training loss, validation loss, per-cell
accuracy, and per-board accuracy (all 81 cells correct). It checkpoints the
last epoch and the best-by-board-accuracy epoch to `--out`, then (unless
`--no-dashboard`) renders `dashboard.html` next to the checkpoints: per-stage
activation heatmaps on a real board image, the learned first-layer filters,
the final 81-cell confidence grid, and the training curves.

## Dashboard / "Train model" button

`dashboard.html` also works as a live control panel when served by `server.py`:
it has a "Train model" button that starts a real training run as a background
process and polls progress until it's done, then reloads to show the new
result. Opened as a plain file (or via a static preview) the button is inert
&mdash; it just has no server to talk to.

```
python server.py            # serves the dashboard at http://localhost:5000
```

## Upload & solve

The dashboard has a second tab, "Upload & solve": drop in (or drag-and-drop) a
board image, or click "Try a sample" to pull a random one from the dataset.
"Solve from image" reads all 81 cells with the trained model and solves the
puzzle in one step; the grid stays editable, so a misread cell can be fixed
by hand before re-solving with "Solve grid". Backed by three endpoints on
`server.py`:

- `POST /recognize` (multipart `image` file) &rarr; `{cells: [...]}`
- `GET /sample` &rarr; a random dataset image + its recognized/ground-truth grids
- `POST /solve` (JSON `{grid: [81 ints]}`) &rarr; the solved grid, or an error
  if the grid has conflicting digits or no solution (`solver.py` is a plain
  backtracking solver, independent of the model)

Like the "Train model" button, this only works when the dashboard is served
by `server.py` (or Docker) &mdash; on a static file or a static Artifact
preview there's no backend to call, so those buttons just report that.

### Docker

```
docker compose up --build
```

Builds `Dockerfile` (PyTorch + CUDA base image) and starts the dashboard at
`http://localhost:5000`, with `dataset_gen/output` mounted read-only as
training data and `train/checkpoints` mounted so checkpoints/the dashboard
persist on the host. Requires the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
for GPU access &mdash; without it, drop the `deploy.resources` block in
`docker-compose.yml` and it trains on CPU.

## Note on dataset size

`dataset_gen/output` currently has ~1000 samples. That's enough to confirm the
pipeline trains end-to-end, but a CNN trained from scratch will likely overfit
at that scale. Generate more (`../dataset_gen/run.sh --count 20000 --out
./output --resume`) before training a model you intend to actually use.
