"""Kaggle entrypoint for training the global jigsaw model on all 7000 targets."""
import os
from pathlib import Path


def find_targets():
    candidates = []
    for root in (Path("/kaggle/input"), Path(".")):
        if root.exists():
            candidates.extend(root.rglob("train/targets"))
    candidates = [p for p in candidates if len(list(p.glob("*.png"))) >= 1000]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one full train/targets directory, got {candidates}")
    return candidates[0]


os.environ.setdefault("IMAGE_DIR", str(find_targets()))
os.environ.setdefault("OUT_DIR", "/kaggle/working")
os.environ.setdefault("EPOCHS", "20")
os.environ.setdefault("SAMPLES_PER_EPOCH", "7000")
os.environ.setdefault("VAL_SAMPLES", "512")
os.environ.setdefault("VAL_SOURCES", "512")
os.environ.setdefault("BATCH_SIZE", "4")
os.environ.setdefault("DIM", "192")
os.environ.setdefault("LAYERS", "6")
os.environ.setdefault("LR", "2e-4")

from train_global_jigsaw_transformer import main

if __name__ == "__main__":
    main()
