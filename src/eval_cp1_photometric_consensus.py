"""CP1: deterministic candidate-conditioned photometric consensus evaluation.

CP1 estimates per-tile diagonal affine color corrections only from mutual top-1
frozen candidate edges, then reranks *the same* K candidate rows with corrected
boundary continuity.  The known permutation is never consulted before the
correction/scoring step; it is used only for post-hoc covered-neighbour metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

GRID = 24
N = GRID * GRID
DIRECTIONS = 4
OPPOSITE = (1, 0, 3, 2)
DEFAULT_CACHE = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\SGT2_visual_graph\visual_cache")
DEFAULT_WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\CP1_photometric_consensus")


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            d.update(block)
    return d.hexdigest()


def zscore(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    safe = np.where(valid, values, 0.0)
    count = valid.sum(axis=1, keepdims=True).clip(min=1)
    mean = safe.sum(axis=1, keepdims=True) / count
    var = np.where(valid, (values - mean) ** 2, 0.0).sum(axis=1, keepdims=True) / count
    return np.where(valid, (values - mean) / np.sqrt(var + 1e-6), -20.0).astype(np.float32)


def topk_cache(candidate_ids: np.ndarray, scores: np.ndarray, anchors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    raw = np.take_along_axis(scores, order, axis=1)
    valid = np.isfinite(raw)
    ids = np.take_along_axis(candidate_ids[anchors], order, axis=1)
    return ids, valid, zscore(raw, valid)


def facing(tile: np.ndarray, direction: int, band: int) -> np.ndarray:
    if direction == 0:  # source up
        return tile[:band, :, :]
    if direction == 1:  # source down
        return tile[-band:, :, :]
    if direction == 2:  # source left
        return tile[:, :band, :]
    if direction == 3:  # source right
        return tile[:, -band:, :]
    raise ValueError(direction)


def mutual_seed_edges(tiles: np.ndarray, candidate_ids: np.ndarray, scores: np.ndarray, anchors: np.ndarray) -> list[tuple[int, int, int, float]]:
    """Input-only mutual first-choice directional edges, deduplicated canonically."""
    best = np.full((N, DIRECTIONS), -1, dtype=np.int64)
    best_score = np.full((N, DIRECTIONS), -np.inf, dtype=np.float32)
    for q, (source, direction) in enumerate(zip(anchors, np.arange(N * DIRECTIONS) % DIRECTIONS, strict=True)):
        row = scores[q]
        index = int(np.argmax(row))
        if np.isfinite(row[index]):
            best[source, direction] = int(candidate_ids[source, index])
            best_score[source, direction] = float(row[index])
    seen: set[tuple[int, int, int]] = set()
    edges: list[tuple[int, int, int, float]] = []
    for source in range(N):
        for direction in range(DIRECTIONS):
            target = int(best[source, direction])
            if target < 0 or target == source:
                continue
            if best[target, OPPOSITE[direction]] != source:
                continue
            # preserve directional relation, but do not add both reciprocal forms
            key = (min(source, target), max(source, target), min(direction, OPPOSITE[direction]))
            if key in seen:
                continue
            seen.add(key)
            weight = float(np.exp(np.clip(min(best_score[source, direction], best_score[target, OPPOSITE[direction]]), -8.0, 8.0) / 8.0))
            edges.append((source, target, direction, weight))
    return edges


def solve_affine(tiles: np.ndarray, edges: list[tuple[int, int, int, float]], band: int, regularization: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve per-tile gain/offset from provisional seam correspondences."""
    lap = np.eye(N, dtype=np.float64) * regularization
    offset_rhs = np.zeros((N, 3), dtype=np.float64)
    gain_rhs = np.zeros((N, 3), dtype=np.float64)
    relation_count = 0
    for source, target, direction, weight in edges:
        x = facing(tiles[source].astype(np.float64), direction, band).reshape(-1, 3)
        y = facing(tiles[target].astype(np.float64), OPPOSITE[direction], band).reshape(-1, 3)
        mx, my = x.mean(axis=0), y.mean(axis=0)
        sx = x.std(axis=0).clip(2.0, None)
        sy = y.std(axis=0).clip(2.0, None)
        # b_source - b_target = mean(target) - mean(source)
        delta_b = my - mx
        # log(a_source)-log(a_target) = log(std(target)/std(source))
        delta_g = np.log(sy / sx)
        lap[source, source] += weight
        lap[target, target] += weight
        lap[source, target] -= weight
        lap[target, source] -= weight
        offset_rhs[source] += weight * delta_b
        offset_rhs[target] -= weight * delta_b
        gain_rhs[source] += weight * delta_g
        gain_rhs[target] -= weight * delta_g
        relation_count += 1
    offsets = np.linalg.solve(lap, offset_rhs)
    log_gains = np.linalg.solve(lap, gain_rhs)
    gains_raw = np.exp(np.clip(log_gains, -2.0, 2.0))
    gains = np.clip(gains_raw, 0.65, 1.50)
    offsets = np.clip(offsets, -75.0, 75.0)
    diagnostic = {
        "mutual_seed_edges": relation_count,
        "gain_min": float(gains.min()), "gain_max": float(gains.max()),
        "offset_min": float(offsets.min()), "offset_max": float(offsets.max()),
        "finite": bool(np.isfinite(gains).all() and np.isfinite(offsets).all()),
    }
    return gains.astype(np.float32), offsets.astype(np.float32), diagnostic


def corrected_tiles(tiles: np.ndarray, gains: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    value = tiles.astype(np.float32) * gains[:, None, None, :] + offsets[:, None, None, :]
    return np.clip(value, 0.0, 255.0).astype(np.float32)


def seam_scores(tiles: np.ndarray, candidate_by_query: np.ndarray, valid: np.ndarray, anchors: np.ndarray, band: int) -> np.ndarray:
    queries, k = candidate_by_query.shape
    raw_cost = np.full((queries, k), np.inf, dtype=np.float32)
    for q in range(queries):
        source = int(anchors[q])
        direction = q % DIRECTIONS
        source_band = facing(tiles[source], direction, band)
        # compare aligned boundary strips; opposite-side reversal is not needed for square tile adjacency
        for index, target in enumerate(candidate_by_query[q]):
            if not valid[q, index]:
                continue
            target_band = facing(tiles[int(target)], OPPOSITE[direction], band)
            color = np.mean(np.abs(source_band - target_band))
            if band >= 2:
                deriv = np.mean(np.abs(np.diff(source_band, axis=0 if direction < 2 else 1) - np.diff(target_band, axis=0 if direction < 2 else 1)))
            else:
                deriv = 0.0
            raw_cost[q, index] = color + 0.30 * deriv
    return -raw_cost


def true_index(permutation: np.ndarray, candidate_by_query: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(N, dtype=np.int64)
    target = np.full(N * DIRECTIONS, -1, dtype=np.int64)
    valid_query = np.zeros(N * DIRECTIONS, dtype=bool)
    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for source in range(N):
        row, col = divmod(int(permutation[source]), GRID)
        for direction, (dr, dc) in enumerate(deltas):
            q = source * DIRECTIONS + direction
            rr, cc = row + dr, col + dc
            if not (0 <= rr < GRID and 0 <= cc < GRID):
                continue
            valid_query[q] = True
            truth = int(inv[rr * GRID + cc])
            found = np.flatnonzero((candidate_by_query[q] == truth) & valid[q])
            if found.size:
                target[q] = int(found[0])
    return target, valid_query


def evaluate_one(path: Path, k: int, band: int, regularization: float, alpha: float | None = None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        tiles = np.asarray(z["tiles_rgb"], dtype=np.uint8)
        perm = np.asarray(z["permutation"], dtype=np.int64)
        ids = np.asarray(z["candidate_ids"], dtype=np.int64)
        scores = np.asarray(z["candidate_scores"], dtype=np.float32)
        anchors = np.asarray(z["anchors"], dtype=np.int64)
        split = str(z["split_name"].item())
    candidate, valid, raw_z = topk_cache(ids, scores, anchors, k)
    edges = mutual_seed_edges(tiles, ids, scores, anchors)
    gains, offsets, diagnostics = solve_affine(tiles, edges, band, regularization)
    if not diagnostics["finite"]:
        raise FloatingPointError(f"non-finite CP1 correction: {path}")
    corrected = corrected_tiles(tiles, gains, offsets)
    seam_z = zscore(seam_scores(corrected, candidate, valid, anchors, band), valid)
    truth, valid_query = true_index(perm, candidate, valid)
    covered = truth >= 0
    coverage = float(covered.sum() / max(1, valid_query.sum()))
    frozen_top1 = float((raw_z[covered].argmax(axis=1) == truth[covered]).mean())
    result = {"name": path.name, "split": split, "coverage": coverage, "covered_queries": int(covered.sum()), "frozen_top1": frozen_top1, "diagnostics": diagnostics}
    if alpha is not None:
        fused = raw_z + float(alpha) * seam_z
        cp1_top1 = float((fused[covered].argmax(axis=1) == truth[covered]).mean())
        result |= {"alpha": float(alpha), "cp1_top1": cp1_top1, "delta": cp1_top1 - frozen_top1}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--cal", default="image_0051_k64.npz")
    ap.add_argument("--dev", default="image_0014_k64.npz,image_0020_k64.npz")
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--band", type=int, default=3)
    ap.add_argument("--regularization", type=float, default=2.0)
    ap.add_argument("--alpha-grid", default="0,0.125,0.25,0.5,0.75,1,1.5,2,3,4")
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()
    cal_path = args.cache_dir / args.cal
    dev_paths = [args.cache_dir / name.strip() for name in args.dev.split(",") if name.strip()]
    if not cal_path.is_file() or not all(path.is_file() for path in dev_paths):
        raise FileNotFoundError("requested CP1 calibration/dev visual cache missing")
    if args.k != 96:
        raise ValueError("CP1 local gate is pre-registered at frozen K=96")
    grid = [float(x) for x in args.alpha_grid.split(",")]
    cal_rows = [evaluate_one(cal_path, args.k, args.band, args.regularization, alpha) for alpha in grid]
    best = max(cal_rows, key=lambda row: (row["cp1_top1"], -row["alpha"]))
    alpha = float(best["alpha"])
    dev = [evaluate_one(path, args.k, args.band, args.regularization, alpha) for path in dev_paths]
    summary = {key: float(np.mean([row[key] for row in dev])) for key in ("frozen_top1", "cp1_top1", "delta", "coverage")}
    gate = {"condition": "source-disjoint covered top-1 delta > +0.01 at unchanged K=96; correction finite", "passed": bool(summary["delta"] > 0.01 and all(row["diagnostics"]["finite"] for row in dev)), "decision": "advance_to_shared_layout_ssim" if summary["delta"] > 0.01 else "reject_CP1_before_solver"}
    report = {"experiment": "CP1_candidate_conditioned_photometric_consensus", "scope": "input-only mutual provisional candidates estimate deterministic per-tile corrections; permutation is evaluation-only; K=96 fixed", "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "cache_manifest_sha256": sha256(args.cache_dir / "visual_cache_manifest.json"), "calibration": {"rows": cal_rows, "selected": best}, "dev": {"rows": dev, "summary": summary}, "gate": gate}
    args.work.mkdir(parents=True, exist_ok=True)
    output = args.report or args.work / "cp1_g1_local_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
