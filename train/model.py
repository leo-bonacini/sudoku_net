"""
Fully-convolutional Sudoku board reader.

Rather than flattening to 81 independent dense heads, the backbone downsamples
the input by exactly 32x (5 stride-2 stages) so a 288x288 input lands on a 9x9
feature map -- one feature vector per board cell, in the same spatial layout
as the board itself. A final 1x1 conv turns each of those into a 10-way
class score (blank + digits 1-9), giving an (N, 10, 9, 9) output that is
equivalent to "81 outputs" but shares weights across cells and stays robust to
the image not being pixel-perfectly aligned.
"""
import torch.nn as nn

from dataset import NUM_CLASSES


def _down_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SudokuGridNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = nn.Sequential(
            _down_block(3, 32),    # 288 -> 144
            _down_block(32, 64),   # 144 -> 72
            _down_block(64, 128),  # 72  -> 36
            _down_block(128, 256), # 36  -> 18
            _down_block(256, 256), # 18  -> 9
        )
        self.head = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)  # (N, num_classes, 9, 9)
