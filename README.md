# sudoku_net

An end-to-end pipeline that teaches a CNN to read a Sudoku board from a screenshot: play the game, render thousands of labeled board images from it, train a model to recognize all 81 cells, then solve.

## Pieces

| Folder | What it is |
|---|---|
| [`sudoku/`](https://github.com/leo-bonacini/sudoku) | The Sudoku game itself (vanilla HTML/CSS/JS, no build step). Pulled in as a git submodule. |
| `dataset_gen/` | Drives the game in a headless browser (Playwright) to render labeled board images: random theme, font, rotation, and opacity per sample. |
| `train/` | A fully-convolutional CNN (PyTorch) that reads a rendered board image and predicts all 81 cells, plus a Flask dashboard/server for training, live introspection, and image-to-solve. |

## Quickstart

Clone with submodules (needed to pull in the game code):

```
git clone --recursive https://github.com/leo-bonacini/sudoku_net.git
cd sudoku_net
```

(Already cloned without `--recursive`? Run `git submodule update --init`.)

Generate a dataset, then train:

```
cd dataset_gen && npm install && ./run.sh --count 2000 --out ./output
cd ../train && pip install -r requirements.txt
python train.py --data ../dataset_gen/output --epochs 30 --out ./checkpoints
python server.py   # dashboard + upload-and-solve at http://localhost:5000
```

Or run the training dashboard in Docker (see `docker-compose.yml`; requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU, otherwise it falls back to CPU):

```
docker compose up --build
```

See `dataset_gen/generate.js` (`--help`-style comment block at the top) and [`train/README.md`](train/README.md) for full options, the model architecture, and the dashboard/API details.

## License

MIT, see [LICENSE](LICENSE).
