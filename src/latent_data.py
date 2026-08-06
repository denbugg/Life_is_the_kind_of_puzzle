"""Clean-canvas data loaders for the latent image-prior experiment."""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from canvas_data import clean_canvas_and_patches
from config import TRAIN_TGT
from imgio import load


class CleanCanvasDataset(Dataset):
    """Clean 96x96/120x120 area-pooled targets, without needless corruption IO."""

    def __init__(self, names: Sequence[str], patch: int = 4) -> None:
        self.names = list(names)
        self.patch = int(patch)
        if not self.names:
            raise ValueError("CleanCanvasDataset needs image names")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = load(os.path.join(TRAIN_TGT, self.names[index]))
        canvas, _ = clean_canvas_and_patches(image, self.patch)
        return torch.from_numpy(np.ascontiguousarray(canvas)).permute(2, 0, 1).float()
