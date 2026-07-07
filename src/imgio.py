"""Image IO + fragment <-> grid conversions."""
import os
import numpy as np
from PIL import Image
from config import GRID, FS, IMG, NFRAG, TRAIN_TGT, TRAIN_INP, TEST_DIR, VAL_COUNT


def load(path):
    return np.array(Image.open(path).convert("RGB"))


def save(path, arr):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def to_frags(img):
    """(g*FS, g*FS, 3) -> (g*g, FS, FS, 3), row-major (idx = r*g + c)."""
    H, W = img.shape[:2]
    gh, gw = H // FS, W // FS
    img = img.reshape(gh, FS, gw, FS, 3).transpose(0, 2, 1, 3, 4)
    return img.reshape(gh * gw, FS, FS, 3)


def from_frags(frags):
    """(g*g, FS, FS, 3) -> (g*FS, g*FS, 3), row-major (square grid)."""
    n = frags.shape[0]
    g = int(round(n ** 0.5))
    grid = frags.reshape(g, g, FS, FS, 3).transpose(0, 2, 1, 3, 4)
    return grid.reshape(g * FS, g * FS, 3)


def assemble(frags, order):
    """Place fragment `order[p]` at grid position p. order: len-576 array of frag indices."""
    return from_frags(frags[np.asarray(order)])


def list_train():
    names = sorted(os.listdir(TRAIN_TGT))
    return names


def train_val_split():
    names = list_train()
    val = names[-VAL_COUNT:]
    trn = names[:-VAL_COUNT]
    return trn, val


def list_test():
    return sorted(os.listdir(TEST_DIR))
