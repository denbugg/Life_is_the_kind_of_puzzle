"""Data for the instance-conditioned Canvas-first jigsaw experiment.

The important distinction from the old assembly datasets is that this loader never
needs a recovered permutation for a *real* input.  A real input/target pair still
supervises the ordered low-frequency target canvas.  Exact permutation labels are
used only for freshly generated synthetic examples, where they are known without
noise.
"""
from __future__ import annotations

import os
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from config import FS, GRID, NFRAG, TRAIN_INP, TRAIN_TGT
from distort import distort_frags
from imgio import load, to_frags


def clean_canvas_and_patches(image: np.ndarray, patch: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return an aligned low-frequency canvas and its per-grid-cell patches.

    ``patch=4`` maps each original 20x20 tile to a 4x4 clean descriptor by exact
    5x5 area pooling.  Unlike a global resize, this never averages pixels across
    a puzzle seam; a predicted cell is therefore directly comparable to a tile.
    Arrays are float32 in [0, 1].
    """
    if FS % patch:
        raise ValueError(f"patch ({patch}) must divide fragment size ({FS})")
    x = image.astype(np.float32) / 255.0
    stride = FS // patch
    # (row, patch-y, within-bin-y, col, patch-x, within-bin-x, rgb)
    x = x.reshape(GRID, patch, stride, GRID, patch, stride, 3).mean(axis=(2, 5))
    # (row, col, patch-y, patch-x, rgb)
    p = x.transpose(0, 2, 1, 3, 4)
    canvas = p.reshape(GRID * patch, GRID * patch, 3)
    patches = p.reshape(NFRAG, patch, patch, 3)
    return np.ascontiguousarray(canvas), np.ascontiguousarray(patches)


class CanvasDataset(Dataset):
    """Unordered corrupted tile bags with an ordered clean coarse canvas.

    A synthetic sample has a perfect ``perm`` mapping input tile index to its clean
    grid cell.  A real sample deliberately exposes no such mapping: the loader
    returns ``has_perm=False`` rather than leaking the noisy recovered cache into
    the new approach.
    """

    def __init__(
        self,
        names: Sequence[str],
        *,
        patch: int = 4,
        real_prob: float = 0.5,
        seed: int = 1234,
    ) -> None:
        if not names:
            raise ValueError("CanvasDataset needs at least one image name")
        self.names = list(names)
        self.patch = int(patch)
        self.real_prob = float(real_prob)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod
    def _to_tensor(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # The worker-local entropy makes every exposure a new shuffle/distortion.
        rng = np.random.default_rng(self.seed + index * 1_000_003 + np.random.randint(2**31))
        name = self.names[index % len(self.names)]
        clean = load(os.path.join(TRAIN_TGT, name))
        canvas, target_patches = clean_canvas_and_patches(clean, self.patch)
        use_real = self.real_prob > 0.0 and rng.random() < self.real_prob

        if use_real:
            frags = to_frags(load(os.path.join(TRAIN_INP, name)))
        else:
            frags = distort_frags(to_frags(clean), rng)

        # Input-grid locations are arbitrary and must never become a model cue.
        perm = rng.permutation(NFRAG).astype(np.int64)
        frags = frags[perm]
        out: Dict[str, torch.Tensor] = {
            "tiles": torch.from_numpy(np.ascontiguousarray(frags)).permute(0, 3, 1, 2).float() / 255.0,
            "canvas": self._to_tensor(canvas),
            "target_patches": torch.from_numpy(target_patches).float(),
            # Kept on CPU by the trainer except during optional visual metrics.
            "clean": self._to_tensor(clean.astype(np.float32) / 255.0),
            "perm": torch.from_numpy(perm),
            "has_perm": torch.tensor(not use_real, dtype=torch.bool),
        }
        return out
