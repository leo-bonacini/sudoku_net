# GPU-enabled image for the SudokuGridNet training dashboard.
# Requires the NVIDIA Container Toolkit on the host to actually use the GPU
# (see docker-compose.yml); falls back to CPU training otherwise.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV SUDOKU_DATA_DIR=/app/data \
    SUDOKU_CHECKPOINT_DIR=/app/checkpoints \
    PORT=5000 \
    MKL_THREADING_LAYER=GNU

EXPOSE 5000

CMD ["python", "server.py"]
