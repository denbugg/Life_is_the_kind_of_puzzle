"""P36 CSRP-24: confidence-weighted 2x2 support; preserves all rank96 candidates."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p13_component_pose as p13
from solve_buddies import solve_buddies_from_scores

N = 576
RIGHT, DOWN = 0, 1
ALPHA = 0.20
NEG = -1.0e9


def row_standard(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64)
    for row in range(values.shape[0]):
        present = mask[row]
        x = values[row, present]
        if x.size:
            output[row, present] = (x - x.mean()) / max(x.std(), 1e-6)
    return output


def soft_support(right: np.ndarray, down: np.ndarray):
    return down @ right @ down.T, right @ down @ right.T


def dense_direction(candidates, valid, values, direction: int):
    matrix = np.full((N, N), NEG, dtype=np.float64)
    mask = np.zeros((N, N), dtype=bool)
    for tile in range(N):
        keep = valid[tile]
        destinations = candidates[tile, keep]
        matrix[tile, destinations] = values[direction, tile, keep]
        mask[tile, destinations] = True
    np.fill_diagonal(matrix, NEG)
    np.fill_diagonal(mask, False)
    return matrix, mask


def rerank(candidates, valid, scores):
    raw_r, mask_r = dense_direction(candidates, valid, scores, RIGHT)
    raw_d, mask_d = dense_direction(candidates, valid, scores, DOWN)
    z_r = row_standard(raw_r, mask_r)
    z_d = row_standard(raw_d, mask_d)
    weight_r = np.where(mask_r, 1.0 / (1.0 + np.exp(-np.clip(z_r, -30, 30))), 0.0)
    weight_d = np.where(mask_d, 1.0 / (1.0 + np.exp(-np.clip(z_d, -30, 30))), 0.0)
    support_r, support_d = soft_support(weight_r, weight_d)
    new_r = raw_r.copy()
    new_d = raw_d.copy()
    new_r[mask_r] += ALPHA * row_standard(support_r, mask_r)[mask_r]
    new_d[mask_d] += ALPHA * row_standard(support_d, mask_d)[mask_d]
    if not (np.isfinite(new_r[mask_r]).all() and np.isfinite(new_d[mask_d]).all()):
        raise RuntimeError("non-finite candidate score")
    if not (np.array_equal(mask_r, new_r > NEG / 2) and np.array_equal(mask_d, new_d > NEG / 2)):
        raise RuntimeError("candidate preservation failure")
    return raw_r, raw_d, new_r, new_d, support_r, support_d, mask_r, mask_d


def load(cache: Path, source: str):
    candidates, valid, scores = p13.load_score_cache(cache, source)
    if candidates.shape != (N, 128) or valid.shape != (N, 128) or scores.shape != (4, N, 128):
        raise RuntimeError("unexpected frozen cache schema")
    return candidates, valid, scores


def gt_board(label_dir: Path, source: str):
    label = p13.load_cached_label(label_dir, source)
    board = np.empty(N, np.int64)
    board[label] = np.arange(N, dtype=np.int64)
    return board.reshape(24, 24)


def solve_accuracy(right, down, truth):
    board, objective = solve_buddies_from_scores(right, down, max_edges=96, min_margin=0.0, repair_passes=2)
    board = np.asarray(board)
    valid = board.shape == (24, 24) and np.unique(board).size == N and board.min() >= 0 and board.max() < N
    return float(np.mean(board == truth)) if valid else float("nan"), valid, float(objective)


def gate_g0(args):
    right = np.zeros((4, 4), np.float64)
    down = np.zeros((4, 4), np.float64)
    down[0, 2] = down[1, 3] = 1.0
    right[2, 3] = 1.0
    sr, sd = soft_support(right, down)
    ok = bool(sr[0, 1] > 0.99 and sr[0, 2] == 0.0 and np.isfinite(sd).all())
    return {"experiment":"P36_CSRP24","gate":"G0","right_support":float(sr[0,1]),"absent_support":float(sr[0,2]),"invalid":0,"targets_opened":False,"p8_imported":False,"passes_G0":ok}


def gate_g1(args):
    rows = []
    for source in args.sources:
        candidates, valid, scores = load(args.scores, source)
        _, _, new_r, new_d, sr, sd, mr, md = rerank(candidates, valid, scores)
        rows.append({"source":source,"finite":bool(np.isfinite(new_r[mr]).all() and np.isfinite(new_d[md]).all()),"preserved":bool(mr.sum()==valid.sum() and md.sum()==valid.sum()),"right_support_mean":float(sr[mr].mean()),"down_support_mean":float(sd[md].mean())})
    invalid = sum(not (r["finite"] and r["preserved"]) for r in rows)
    return {"experiment":"P36_CSRP24","gate":"G1","sources":len(rows),"invalid":invalid,"targets_opened":False,"p8_imported":False,"passes_G1":bool(invalid==0),"rows":rows}


def gate_g2(args):
    train, held = p13.source_lists(args.manifest)
    if len(args.sources) != 96 or set(args.sources) != set(train[:96]):
        raise RuntimeError("G2 must use exactly locked 96 FIT-train sources")
    rows=[]; started=time.perf_counter()
    for index, source in enumerate(args.sources, 1):
        candidates, valid, scores = load(args.scores, source)
        base_r, base_d, new_r, new_d, _, _, _, _ = rerank(candidates, valid, scores)
        truth = gt_board(args.labels, source)
        before, good0, _ = solve_accuracy(base_r, base_d, truth)
        after, good1, _ = solve_accuracy(new_r, new_d, truth)
        rows.append({"source":source,"baseline":before,"soft":after,"gain":after-before,"valid":bool(good0 and good1)})
        if index % 4 == 0: print(json.dumps({"stage":"solve","done":index,"total":len(args.sources)}),flush=True)
    gain=float(np.mean([r["gain"] for r in rows])); invalid=sum(not r["valid"] for r in rows); seconds=time.perf_counter()-started
    return {"experiment":"P36_CSRP24","gate":"G2","sources":len(rows),"baseline_placement":float(np.mean([r["baseline"] for r in rows])),"soft_placement":float(np.mean([r["soft"] for r in rows])),"gain_pp":100.0*gain,"invalid":invalid,"seconds":seconds,"targets_opened":False,"p8_imported":False,"selection_opened":False,"held_opened":False,"passes_G2":bool(gain>=0.005 and invalid==0 and seconds<900),"rows":rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("g0","g1","g2"),required=True)
    parser.add_argument("--scores",type=Path)
    parser.add_argument("--labels",type=Path)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--sources",nargs="*",default=[])
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    report={"g0":gate_g0,"g1":gate_g1,"g2":gate_g2}[args.mode](args)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
    if not report[f"passes_{args.mode.upper()}"]:
        raise RuntimeError(f"P36 {args.mode.upper()} rejected")

if __name__=="__main__":
    main()
