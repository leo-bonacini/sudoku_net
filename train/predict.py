#!/usr/bin/env python3
"""
Run a trained checkpoint on a single board image and print the predicted grid.

Usage:
    python predict.py --checkpoint checkpoints/best.pt --image ../dataset_gen/output/images/sample_0000.png
"""
import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import SudokuGridNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    return p.parse_args()


def format_grid(flat):
    rows = []
    for r in range(9):
        row = flat[r * 9:(r + 1) * 9]
        rows.append(" ".join(str(v) if v else "." for v in row))
    return "\n".join(rows)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    image_size = ckpt["args"]["image_size"]

    model = SudokuGridNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    img = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        preds = logits.argmax(dim=1).squeeze(0).flatten().tolist()

    print(format_grid(preds))


if __name__ == "__main__":
    main()
