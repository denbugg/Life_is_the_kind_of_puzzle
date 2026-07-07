"""Global configuration: paths and puzzle constants."""
import os

# ---- Puzzle geometry ----
GRID = 24            # 24x24 fragments
FS = 20              # fragment size (px)
IMG = GRID * FS      # 480
NFRAG = GRID * GRID  # 576

# ---- Paths (big data on E:, code on C:) ----
DATA_ROOT = os.environ.get("PAZZLE_DATA", r"E:/pazzle_data")
WORK_ROOT = os.environ.get("PAZZLE_WORK", r"E:/pazzle_work")

TRAIN_INP = os.path.join(DATA_ROOT, "train", "inputs")
TRAIN_TGT = os.path.join(DATA_ROOT, "train", "targets")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

CKPT_DIR  = os.path.join(WORK_ROOT, "ckpt")
CACHE_DIR = os.path.join(WORK_ROOT, "cache")
SUB_DIR   = os.path.join(WORK_ROOT, "submissions")
for d in (CKPT_DIR, CACHE_DIR, SUB_DIR):
    os.makedirs(d, exist_ok=True)

# ---- Distortion params (from Task.txt) ----
BRIGHT = 30.0        # +/- additive
CONTRAST = (0.70, 1.30)
NOISE_SIGMA = (40.0, 55.0)
JPEG_Q = (35, 50)
BLUR_KSIZE = 3       # Gaussian 3x3

# ---- Held-out split for local validation ----
VAL_COUNT = 300      # last N train names reserved for local eval
SEED = 1234
