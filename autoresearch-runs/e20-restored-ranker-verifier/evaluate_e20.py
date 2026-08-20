"""Paired smoke evaluator for E14 versus the coverage-gated E20 critic."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity
import torch

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(HERE)]

import kaggle_e14_solver as e14
from e20_common import (BONUS_WEIGHT, EXPECTED_RANKER_SHA256, TOP_K, Z_CLIP,
                        sha256, validate_inputs)
from e20_verifier import verified_scores
from train_restored_border_ranker import BorderRanker

EXPECTED_RANKER_PARAMETERS = 153_745


def device_from_name(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def assemble(tiles: np.ndarray, layout: np.ndarray) -> np.ndarray:
    return (tiles[layout].reshape(e14.GRID, e14.GRID, e14.TILE, e14.TILE, 3)
            .transpose(0, 2, 1, 3, 4).reshape(480, 480, 3))


def adjacency(layout: np.ndarray, truth: np.ndarray) -> float:
    target_of = np.empty(e14.N, np.int32)
    target_of[truth] = np.arange(e14.N)
    board = target_of[layout].reshape(e14.GRID, e14.GRID)
    right = ((board[:, 1:] == board[:, :-1] + 1)
             & (board[:, 1:] // e14.GRID == board[:, :-1] // e14.GRID))
    down = board[1:] == board[:-1] + e14.GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


def summarize(values: list[float]):
    values = np.asarray(values, np.float64)
    folds = np.asarray([values[offset::4].mean() for offset in range(4)])
    return {"mean": float(values.mean()),
            "robust": float(values.mean() - 0.5 * folds.std()),
            "folds": folds.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    coverage = json.loads(args.coverage_report.read_text())
    if not coverage.get("predeclared_coverage_gate"):
        raise RuntimeError("E20 coverage gate failed; layout smoke is forbidden")
    if coverage.get("start") != args.start or coverage.get("cases") != args.limit:
        raise ValueError("coverage report does not match requested smoke slice")
    data, sidecar, provenance, cache_hash, sidecar_hash = validate_inputs(
        args.cache, args.sidecar
    )
    ranker_hash = sha256(args.ranker)
    if ranker_hash != EXPECTED_RANKER_SHA256:
        raise ValueError(f"ranker hash mismatch: {ranker_hash}")
    checkpoint = torch.load(args.ranker, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != 12 or checkpoint.get("config") != {
        "grid": 24, "tile": 20, "border_width": 6, "candidates": 32
    }:
        raise ValueError("ranker checkpoint contract mismatch")
    ranker = BorderRanker(base=48)
    ranker.load_state_dict(checkpoint["model"])
    parameters = sum(parameter.numel() for parameter in ranker.parameters())
    if parameters != EXPECTED_RANKER_PARAMETERS:
        raise ValueError(f"ranker parameter mismatch: {parameters}")
    device = device_from_name(args.device)
    ranker = ranker.to(device).eval()

    stop = min(args.start + args.limit, len(data["stems"]))
    methods = ("e14", "e20")
    ssims = {name: [] for name in methods}
    adjacencies = {name: [] for name in methods}
    solver_times = {name: [] for name in methods}
    raw_preprocessing, verifier_times = [], []
    rows, failures = [], []
    for index in range(args.start, stop):
        seed = 20260818 + index * 100 + args.seed_offset
        row = {"index": index, "stem": str(data["stems"][index])}
        try:
            raw = np.asarray(data["tiles"][index], np.uint8)
            restored = np.asarray(sidecar["restored"][index], np.uint8)
            good = ~np.asarray(sidecar["bad_mask"][index], np.bool_)
            started = time.perf_counter()
            classical_right, classical_down = e14.classical_mgc_ssd_scores(raw)
            e14_right = e14.fuse_scores(data["right"][index], classical_right)
            e14_down = e14.fuse_scores(data["down"][index], classical_down)
            raw_prep = time.perf_counter() - started
            raw_preprocessing.append(raw_prep)

            started = time.perf_counter()
            e20_right, right_stats = verified_scores(
                ranker, restored, good, e14_right, 0, device
            )
            e20_down, down_stats = verified_scores(
                ranker, restored, good, e14_down, 1, device
            )
            verifier_time = time.perf_counter() - started
            verifier_times.append(verifier_time)
            row.update({"good_tiles": int(good.sum()),
                        "raw_preprocessing_seconds": raw_prep,
                        "verifier_seconds": verifier_time,
                        "right_verifier": right_stats,
                        "down_verifier": down_stats})

            for method, right, down in (
                ("e14", e14_right, e14_down),
                ("e20", e20_right, e20_down),
            ):
                started = time.perf_counter()
                layout = np.asarray(
                    e14.solve_layout(right, down, data["pos"][index], seed),
                    np.int32,
                )
                elapsed = time.perf_counter() - started
                if not e14.is_valid_layout(layout):
                    raise ValueError(f"{method} returned invalid permutation")
                score = float(structural_similarity(
                    data["target"][index], assemble(raw, layout),
                    channel_axis=2, data_range=255,
                ))
                adj = adjacency(layout, data["truth"][index])
                if not all(np.isfinite(value) for value in (elapsed, score, adj)):
                    raise FloatingPointError(f"non-finite result from {method}")
                ssims[method].append(score)
                adjacencies[method].append(adj)
                solver_times[method].append(elapsed)
                row[method] = {"raw_ssim": score, "adjacency": adj,
                               "solver_seconds": elapsed,
                               "valid_permutation": True}
        except Exception as exc:
            failures.append({"index": index, "error": repr(exc),
                             "traceback": traceback.format_exc()})
            row["failure"] = repr(exc)
        rows.append(row)
        print(json.dumps({"done": index - args.start + 1,
                          "total": stop - args.start, "stem": row["stem"]}), flush=True)

    cases = stop - args.start
    report = {
        "experiment": "E20 locked sparse restored-ranker verifier",
        "cache_sha256": cache_hash,
        "sidecar_sha256": sidecar_hash,
        "sidecar_provenance": provenance,
        "ranker_sha256": ranker_hash,
        "ranker_epoch": int(checkpoint["epoch"]),
        "ranker_parameters": parameters,
        "coverage_report": str(args.coverage_report.resolve()),
        "coverage": coverage["coverage"],
        "cases": cases, "start": args.start, "seed_offset": args.seed_offset,
        "locked_spec": {"topk_each": TOP_K, "bonus_weight": BONUS_WEIGHT,
                        "robust_z_clip": [-Z_CLIP, Z_CLIP],
                        "ranker_input": "unguarded restored",
                        "good_mask": "exact no-gray guard"},
        "selection_inputs": ["raw tiles", "unguarded restored tiles", "guard mask",
                             "right", "down", "pos"],
        "selection_excludes": ["target", "truth", "SSIM", "adjacency"],
        "methods": {}, "failures": failures, "images": rows,
        "promotable": False,
        "promotion_blocker": "ranker/restorer training-stem overlap cannot be disproved locally",
    }
    for method in methods:
        if len(ssims[method]) != cases:
            report["methods"][method] = {"completed": len(ssims[method])}
            continue
        preprocessing = np.sum(raw_preprocessing)
        if method == "e20":
            preprocessing += np.sum(verifier_times)
        report["methods"][method] = {
            "raw_ssim": summarize(ssims[method]),
            "mean_adjacency": float(np.mean(adjacencies[method])),
            "solver_runtime_seconds": float(np.sum(solver_times[method])),
            "preprocessing_seconds": float(preprocessing),
            "end_to_end_runtime_seconds": float(preprocessing + np.sum(solver_times[method])),
            "valid_permutations": cases,
        }
    if not failures:
        baseline, candidate = report["methods"]["e14"], report["methods"]["e20"]
        comparison = {
            "raw_robust_ssim_delta": candidate["raw_ssim"]["robust"] - baseline["raw_ssim"]["robust"],
            "raw_mean_ssim_delta": candidate["raw_ssim"]["mean"] - baseline["raw_ssim"]["mean"],
            "mean_adjacency_delta": candidate["mean_adjacency"] - baseline["mean_adjacency"],
            "raw_ssim_wins": int((np.asarray(ssims["e20"]) > np.asarray(ssims["e14"])).sum()),
            "adjacency_wins": int((np.asarray(adjacencies["e20"]) > np.asarray(adjacencies["e14"])).sum()),
            "runtime_ratio": candidate["end_to_end_runtime_seconds"] / baseline["end_to_end_runtime_seconds"],
        }
        report["comparison"] = comparison
        report["predeclared_smoke_gate"] = bool(
            comparison["raw_robust_ssim_delta"] >= 0.0015
            and comparison["raw_mean_ssim_delta"] >= 0.0015
            and comparison["mean_adjacency_delta"] >= 0.005
            and comparison["raw_ssim_wins"] >= 10
            and comparison["runtime_ratio"] <= 4.0
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "images"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
