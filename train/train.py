#!/usr/bin/env python3
"""
Train SudokuGridNet on the images/labels produced by dataset_gen/generate.js.

Usage:
    python train.py --data ../dataset_gen/output --epochs 30 --out ./checkpoints
"""
import argparse
import os
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import SudokuImageDataset
from model import SudokuGridNet
from visualize import build_dashboard


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.environ.get(
        "SUDOKU_DATA_DIR", str(Path(__file__).parent / ".." / "dataset_gen" / "output")))
    p.add_argument("--out", default=os.environ.get(
        "SUDOKU_CHECKPOINT_DIR", str(Path(__file__).parent / "checkpoints")))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=288)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dashboard", action="store_true", default=True,
                    help="Render the layer-introspection dashboard after training (default: on)")
    p.add_argument("--no-dashboard", dest="dashboard", action="store_false")
    return p.parse_args()


def split_indices(n, val_split, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(n * val_split))
    return idx[n_val:], idx[:n_val]


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_cells, correct_cells, correct_boards, total_boards = 0.0, 0, 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)  # (N, 9, 9)
        correct = preds == labels
        correct_cells += correct.sum().item()
        total_cells += correct.numel()
        correct_boards += correct.all(dim=-1).all(dim=-1).sum().item()
        total_boards += images.size(0)

    return (
        total_loss / total_boards,
        correct_cells / total_cells,
        correct_boards / total_boards,
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full = SudokuImageDataset(args.data, image_size=args.image_size)
    train_idx, val_idx = split_indices(len(full), args.val_split, args.seed)
    print(f"Dataset: {len(full)} samples -> {len(train_idx)} train / {len(val_idx)} val")

    train_ds = SudokuImageDataset(args.data, indices=train_idx, image_size=args.image_size, augment=args.augment)
    val_ds = SudokuImageDataset(args.data, indices=val_idx, image_size=args.image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = SudokuGridNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_board_acc, best_cell_acc, best_epoch = -1.0, -1.0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_ds)
        val_loss, cell_acc, board_acc = evaluate(model, val_loader, device, criterion)
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  cell_acc={cell_acc:.4%}  board_acc={board_acc:.4%}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "cell_acc": cell_acc, "board_acc": board_acc})

        torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args)},
                   out_dir / "last.pt")
        if board_acc > best_board_acc:
            best_board_acc, best_cell_acc, best_epoch = board_acc, cell_acc, epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args)},
                       out_dir / "best.pt")

    print(f"Done. Best val board accuracy: {best_board_acc:.4%}. Checkpoints in {out_dir}")

    if args.dashboard:
        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model"])
        sample_record = val_ds.records[0]
        dashboard_path = build_dashboard(
            model, args.data, sample_record, out_dir / "dashboard.html",
            image_size=args.image_size,
            checkpoint_meta={"epoch": best_epoch, "epochs": args.epochs,
                              "best_board_acc": best_board_acc, "best_cell_acc": best_cell_acc},
            history=history,
        )
        print(f"Dashboard: {dashboard_path}")


if __name__ == "__main__":
    main()
