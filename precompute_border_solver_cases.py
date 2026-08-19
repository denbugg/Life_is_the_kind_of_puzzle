"""Freeze border-ranker score matrices for fast global-solver optimization."""
import os
from pathlib import Path
import numpy as np
import torch
import evaluate_submit_border_pipeline as pipeline

COUNT = int(os.getenv("CASE_COUNT", "8"))
OUTPUT = Path(os.getenv("CASE_FILE", "border_solver_cases.npz"))

def main():
    device = torch.device("cuda"); models = pipeline.load_models(device)
    z = np.load(pipeline.MAP_FILE); stems, maps = z["stems"], z["maps"]
    order = np.arange(len(stems)); np.random.default_rng(pipeline.SEED).shuffle(order)
    val = order[-max(100, len(order) // 10):]
    chosen = val[np.linspace(7, len(val) - 8, min(COUNT, len(val)), dtype=int)]
    rights, downs, positions, truths, names = [], [], [], [], []
    for k, j in enumerate(chosen):
        stem = str(stems[j]); tiles = pipeline.split(pipeline.DATA_ROOT / "inputs" / f"{stem}.png")
        restored = pipeline.restore(models[0], tiles, device)
        rights.append(pipeline.ranker_matrix(models[1], restored, 0, device))
        downs.append(pipeline.ranker_matrix(models[1], restored, 1, device))
        positions.append(pipeline.position_matrix(models[2], restored, device))
        truths.append(maps[j].astype(np.int32)); names.append(stem)
        print({"done": k + 1, "total": len(chosen), "stem": stem}, flush=True)
    np.savez_compressed(OUTPUT, right=np.asarray(rights, np.float32), down=np.asarray(downs, np.float32),
                        pos=np.asarray(positions, np.float32), truth=np.asarray(truths, np.int32), stems=np.asarray(names))
    print({"output": str(OUTPUT), "cases": len(names)}, flush=True)

if __name__ == "__main__": main()
