"""Dataset for the labeled Sudoku board images produced by dataset_gen/generate.js."""
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

NUM_CLASSES = 10  # 0 = blank cell, 1-9 = digit


class SudokuImageDataset(Dataset):
    def __init__(self, root, indices=None, image_size=288, augment=False):
        self.root = Path(root)
        with open(self.root / "labels.jsonl") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        if indices is not None:
            self.records = [self.records[i] for i in indices]

        ops = [transforms.Resize((image_size, image_size))]
        if augment:
            # Only photometric jitter: the model relies on each cell landing in a
            # fixed spatial location, so geometric augmentation would fight that.
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        ops += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(self.root / "images" / rec["file"]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor([int(c) for c in rec["grid"]], dtype=torch.long).view(9, 9)
        return img, label
