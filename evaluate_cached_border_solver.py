"""Fixed evaluator: candidate receives score matrices but never ground truth."""
import os
from pathlib import Path
import numpy as np
from global_solver_candidate import solve_layout

GRID, N = 24, 576
CASE_FILE = Path(os.getenv("CASE_FILE", "/home/kva/pazzle_directional_transformer/border_solver_cases.npz"))

def adjacency(layout, true_layout):
    target_of = np.empty(N, np.int32); target_of[true_layout] = np.arange(N)
    x = target_of[np.asarray(layout)].reshape(GRID, GRID)
    right = (x[:, 1:] == x[:, :-1] + 1) & (x[:, 1:] // GRID == x[:, :-1] // GRID)
    down = x[1:] == x[:-1] + GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))

def main():
    z = np.load(CASE_FILE); scores = []
    for i, (right, down, pos) in enumerate(zip(z["right"], z["down"], z["pos"])):
        layout = np.asarray(solve_layout(right, down, pos, 20260818 + i * 100), np.int64)
        if layout.shape != (N,) or len(np.unique(layout)) != N or layout.min() != 0 or layout.max() != N - 1:
            raise ValueError("solve_layout must return a permutation of 0..575")
        scores.append(adjacency(layout, z["truth"][i]))
    print(f"adjacency_score: {np.mean(scores):.9f}")

if __name__ == "__main__": main()
