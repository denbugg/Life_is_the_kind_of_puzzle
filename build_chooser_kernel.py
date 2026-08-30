"""Build the Kaggle kernel that runs M412's experiment at the scale it asked for.

M412 rejected the five-candidate chooser and said why in its own words: 136
boards is not enough for a transformer over five candidates, and "the honest
reading is that the experiment has not yet been RUN at the scale it needs
rather than that it failed". The dump kernel has since produced 3141 boards,
which is twenty-three times more, and its own control says they may all be
used: the matcher reads top-1 0.3055 on the region it was trained on against
0.3001 on the held-out region, a difference of 0.005, so there is no
distribution shift to protect against.

Two design points come from M412 and are kept because both were measured: a
zero-initialised score head so the model STARTS at the matcher's own top-1
exactly, and a discounted NONE class, since NONE is the right answer for 47% of
fragments and plain cross-entropy drives a model that starts at 347.3 correct
bonds down to 275.9 in a single epoch.

The number to watch is CORRECT BONDS on held-out boards against the matcher's
own top-1 recomputed on the same boards, never accuracy and never a quoted
baseline.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "src"
OUT = REPO / "kaggle_chooser"
EMBED = ["config.py", "restore_tile.py", "choose5.py", "train_choose5.py"]

SETUP = '''# Everything the chooser needs, unpacked where the trainer expects it.
# Kaggle mounts a dataset at /kaggle/input/datasets/<owner>/<slug> rather than
# under the slug directly, so both searches walk instead of listing one level.
import os, sys, json, shutil
from pathlib import Path

BASE = Path("/kaggle/input")
print("mounted:", [q.name for q in BASE.iterdir()])

W = Path("/kaggle/working")
WORK = W / "pazzle_work"
(WORK / "cache").mkdir(parents=True, exist_ok=True)
(WORK / "ckpt").mkdir(parents=True, exist_ok=True)
SRC = W / "src"
SRC.mkdir(exist_ok=True)
for name, text in FILES.items():
    (SRC / name).write_text(text, encoding="utf-8")
sys.path.insert(0, str(SRC))


def find_dir(pred):
    for q in BASE.rglob("*"):
        if q.is_dir() and pred(q):
            return q
    return None


DATA = find_dir(lambda q: (q / "train" / "inputs").is_dir())
assert DATA is not None, "puzzle images not found"
DUMPS = find_dir(lambda q: q.name == "top5_new" and any(q.glob("*.npz")))
assert DUMPS is not None, "top5 dumps not found"
LABELS = next(BASE.rglob("restore_labels.npz"), None)
assert LABELS is not None, "restore_labels.npz not found"
shutil.copy(LABELS, WORK / "cache" / "restore_labels.npz")

# config.py reads these at IMPORT time and the names are PAZZLE_DATA and
# PAZZLE_WORK, not the _ROOT spellings the variables inside it use
os.environ["PAZZLE_DATA"] = str(DATA)
os.environ["PAZZLE_WORK"] = str(WORK)
print("data:", DATA)
print("dumps:", DUMPS, len(list(DUMPS.glob("*.npz"))), "boards")
'''

TRAIN = '''# M412 at twenty-three times its data. Everything else is held at the values
# M412 measured, so the only variable is the scale it named.
import numpy as np, torch, time
from pathlib import Path

sys.argv = ["train_choose5.py",
            "--dumps", str(DUMPS),
            "--held", "240",
            "--epochs", "%(epochs)d",
            "--ch", "%(ch)d", "--dim", "%(dim)d", "--layers", "%(layers)d",
            "--strip", "4",
            "--none-weight", "0.3",
            "--lr", "%(lr)g",
            "--eval-every", "1",
            "--seed", "0",
            "--out", "/kaggle/working/choose5_big.pt"]
import train_choose5
t0 = time.time()
train_choose5.main()
print(f"done in {(time.time() - t0) / 3600:.2f} h")
'''


def cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def main():
    OUT.mkdir(exist_ok=True)
    files = {n: (SRC / n).read_text(encoding="utf-8") for n in EMBED}
    head = "FILES = " + json.dumps(files, ensure_ascii=False) + "\n\n"
    nb = {"cells": [cell(head + SETUP),
                    cell(TRAIN % dict(epochs=12, ch=64, dim=192, layers=3,
                                      lr=3e-4))],
          "metadata": {"kernelspec": {"language": "python",
                                      "display_name": "Python 3",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (OUT / "vsos-pazzle-chooser-big.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    (OUT / "kernel-metadata.json").write_text(json.dumps({
        "id": "pasha883/vsos-pazzle-chooser-big",
        "title": "VsOS pazzle chooser big",
        "code_file": "vsos-pazzle-chooser-big.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["pasha883/vsos-ai-initiative-pazzle",
                            "pasha883/vsos-pazzle-chooser-inputs"],
        "kernel_sources": ["pasha883/vsos-pazzle-top5-dump"],
        "competition_sources": [],
        "model_sources": []}, indent=1), encoding="utf-8")
    print("built", OUT)


if __name__ == "__main__":
    main()
