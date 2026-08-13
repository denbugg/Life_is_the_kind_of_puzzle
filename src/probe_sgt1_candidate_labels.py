from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

GRID = 24
OFF = np.array([-GRID, GRID, -1, 1], dtype=np.int64)

p = argparse.ArgumentParser()
p.add_argument('paths', nargs='+', type=Path)
p.add_argument('--k', type=int, default=128)
a = p.parse_args()
if a.k < 1:
    raise SystemExit('--k must be positive')
all_coverage = []
all_top1 = []
all_top1_covered = []
for path in a.paths:
    with np.load(path, allow_pickle=False) as z:
        perm = z['permutation'].astype(np.int64)
        ids = z['candidate_ids'].astype(np.int64)
        scores = z['candidate_scores'].astype(np.float32)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    covered = []
    top1 = []
    all_valid = 0
    for tile in range(len(perm)):
        pos = int(perm[tile])
        r, c = divmod(pos, GRID)
        for d, (dr, dc) in enumerate(((-1,0),(1,0),(0,-1),(0,1))):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < GRID and 0 <= cc < GRID):
                continue
            all_valid += 1
            truth = int(inv[rr * GRID + cc])
            q = tile * 4 + d
            candidate_order = np.argsort(-scores[q], kind='stable')[: a.k]
            row = ids[tile][candidate_order]
            loc = np.flatnonzero(row == truth)
            if len(loc):
                covered.append(1)
                top1.append(int(row[0] == truth))
            else:
                covered.append(0)
    coverage_value = float(np.mean(covered))
    top1_value = float(np.sum(top1) / all_valid)
    covered_top1_value = float(np.mean(top1)) if top1 else float('nan')
    all_coverage.append(coverage_value)
    all_top1.append(top1_value)
    all_top1_covered.append(covered_top1_value)
    print(path.name, 'valid=', all_valid, 'coverage=', coverage_value, 'score_top1_all=', top1_value, 'score_top1_covered=', covered_top1_value)
print('AGGREGATE', {'k': a.k, 'n': len(all_coverage), 'coverage_mean': float(np.mean(all_coverage)), 'coverage_min': float(np.min(all_coverage)), 'score_top1_all_mean': float(np.mean(all_top1)), 'score_top1_covered_mean': float(np.mean(all_top1_covered))})
