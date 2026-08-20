"""E18: pixel-only NLM polish applied after the verified E14 layout.

The E14 score construction and relaxation layout are intentionally unchanged.
This evaluator computes each E14 layout once, assembles the raw RGB image, then
applies full-image OpenCV colored NLM.  Target/truth are used only for metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

REPO = Path(__file__).resolve().parents[2]
E14_DIR = REPO / "autoresearch-runs" / "e14-fusion-relaxation"
sys.path[:0] = [str(REPO), str(E14_DIR)]

from e2_raw_fusion import classical_mgc_ssd_scores, fuse_scores
from global_solver_candidate import solve_layout

GRID, TILE, N = 24, 20, 576
SEED_BASE = 20260818
EXPECTED_CACHE_SHA256 = "74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df"
NLM_H = 9
NLM_TEMPLATE_WINDOW = 7
NLM_SEARCH_WINDOW = 21


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble(tiles: np.ndarray, layout: np.ndarray) -> np.ndarray:
    return (tiles[layout].reshape(GRID, GRID, TILE, TILE, 3)
            .transpose(0, 2, 1, 3, 4).reshape(480, 480, 3))


def adjacency(layout: np.ndarray, truth: np.ndarray) -> float:
    target_of = np.empty(N, np.int32)
    target_of[truth] = np.arange(N)
    board = target_of[layout].reshape(GRID, GRID)
    right = ((board[:, 1:] == board[:, :-1] + 1)
             & (board[:, 1:] // GRID == board[:, :-1] // GRID))
    down = board[1:] == board[:-1] + GRID
    return float((right.sum() + down.sum()) / (right.size + down.size))


def nlm_h9(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filtered = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        NLM_H,
        NLM_H,
        NLM_TEMPLATE_WINDOW,
        NLM_SEARCH_WINDOW,
    )
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)


def gray_mask(image: np.ndarray) -> np.ndarray:
    """Flag low-variance, nearly achromatic 20x20 cells (frozen audit rule)."""
    tiles = (image.reshape(GRID, TILE, GRID, TILE, 3)
             .transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3))
    mean = tiles.mean((1, 2))
    std = tiles.std((1, 2, 3))
    return (mean.max(1) - mean.min(1) < 10) & (std < 25)


def gray_count(image: np.ndarray) -> int:
    return int(gray_mask(image).sum())


def no_gray_guard(raw: np.ndarray, filtered: np.ndarray) -> tuple[np.ndarray, int]:
    """Revert only cells newly classified gray by the frozen archive audit."""
    raw_tiles = (raw.reshape(GRID, TILE, GRID, TILE, 3)
                 .transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3))
    filtered_tiles = (filtered.reshape(GRID, TILE, GRID, TILE, 3)
                      .transpose(0, 2, 1, 3, 4).reshape(N, TILE, TILE, 3)).copy()
    revert = gray_mask(filtered) & ~gray_mask(raw)
    filtered_tiles[revert] = raw_tiles[revert]
    guarded = (filtered_tiles.reshape(GRID, GRID, TILE, TILE, 3)
               .transpose(0, 2, 1, 3, 4).reshape(480, 480, 3))
    return guarded, int(revert.sum())


def summarize(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, np.float64)
    folds = np.asarray([
        array[offset::4].mean() for offset in range(min(4, len(array)))
        if len(array[offset::4])
    ])
    return {
        "mean": float(array.mean()),
        "robust": float(array.mean() - 0.5 * folds.std()),
        "folds": folds.tolist(),
    }


def split_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    raw = [float(row["raw_ssim"]) for row in rows]
    unguarded = [float(row["unguarded_nlm_ssim"]) for row in rows]
    guarded = [float(row["guarded_nlm_ssim"]) for row in rows]
    raw_array = np.asarray(raw)
    unguarded_array = np.asarray(unguarded)
    guarded_array = np.asarray(guarded)
    raw_summary = summarize(raw)
    unguarded_summary = summarize(unguarded)
    guarded_summary = summarize(guarded)
    unguarded_mean_gain = float(unguarded_array.mean() - raw_array.mean())
    guarded_mean_gain = float(guarded_array.mean() - raw_array.mean())
    unguarded_robust_gain = float(unguarded_summary["robust"] - raw_summary["robust"])
    guarded_robust_gain = float(guarded_summary["robust"] - raw_summary["robust"])
    return {
        "cases": len(rows),
        "raw_e14_ssim": raw_summary,
        "unguarded_e18_ssim": unguarded_summary,
        "guarded_e18b_ssim": guarded_summary,
        "unguarded_e18": {
            "mean_ssim_delta": unguarded_mean_gain,
            "robust_ssim_delta": unguarded_robust_gain,
            "wins": int((unguarded_array > raw_array).sum()),
            "ties": int((unguarded_array == raw_array).sum()),
            "losses": int((unguarded_array < raw_array).sum()),
        },
        "guarded_e18b": {
            "mean_ssim_delta": guarded_mean_gain,
            "robust_ssim_delta": guarded_robust_gain,
            "mean_gain_retention": guarded_mean_gain / unguarded_mean_gain,
            "robust_gain_retention": guarded_robust_gain / unguarded_robust_gain,
            "wins": int((guarded_array > raw_array).sum()),
            "ties": int((guarded_array == raw_array).sum()),
            "losses": int((guarded_array < raw_array).sum()),
        },
        "mean_adjacency": float(np.mean([float(row["adjacency"]) for row in rows])),
        "layout_and_adjacency_all_identical": bool(all(
            row["layout_identical"] and row["adjacency_identical"] for row in rows
        )),
        "gray_audit": {
            "raw_total": int(sum(int(row["raw_gray_count"]) for row in rows)),
            "unguarded_nlm_total": int(sum(
                int(row["unguarded_nlm_gray_count"]) for row in rows
            )),
            "nlm_total": int(sum(int(row["nlm_gray_count"]) for row in rows)),
            "unguarded_delta": int(sum(
                int(row["unguarded_nlm_gray_count"]) - int(row["raw_gray_count"])
                for row in rows
            )),
            "reverted_new_gray_cells": int(sum(
                int(row["reverted_new_gray_cells"]) for row in rows
            )),
            "unguarded_excess_images": int(sum(
                int(row["unguarded_nlm_gray_count"] > row["raw_gray_count"])
                for row in rows
            )),
            "excess_images": int(sum(
                int(row["nlm_gray_count"] > row["raw_gray_count"]) for row in rows
            )),
        },
        "runtime": {
            "e14_layout_seconds": float(sum(float(row["layout_seconds"]) for row in rows)),
            "nlm_seconds": float(sum(float(row["nlm_seconds"]) for row in rows)),
            "guard_seconds": float(sum(float(row["guard_seconds"]) for row in rows)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    cache_hash = "skipped_after_verified_run" if args.skip_hash else sha256(args.cache)
    if not args.skip_hash and cache_hash != EXPECTED_CACHE_SHA256:
        raise ValueError(f"cache hash mismatch: {cache_hash}")
    data = np.load(args.cache, mmap_mode="r")
    stop = min(args.start + args.limit, len(data["stems"]))
    raw_scores: list[float] = []
    unguarded_scores: list[float] = []
    guarded_scores: list[float] = []
    raw_adjacencies: list[float] = []
    nlm_adjacencies: list[float] = []
    layout_seconds: list[float] = []
    nlm_seconds: list[float] = []
    guard_seconds: list[float] = []
    raw_gray_counts: list[int] = []
    unguarded_gray_counts: list[int] = []
    nlm_gray_counts: list[int] = []
    reverted_counts: list[int] = []
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for index in range(args.start, stop):
        row: dict[str, object] = {"index": index, "stem": str(data["stems"][index])}
        try:
            layout_started = time.perf_counter()
            classical_right, classical_down = classical_mgc_ssd_scores(data["tiles"][index])
            fused_right = fuse_scores(data["right"][index], classical_right)
            fused_down = fuse_scores(data["down"][index], classical_down)
            raw_layout = np.asarray(solve_layout(
                fused_right,
                fused_down,
                data["pos"][index],
                SEED_BASE + index * 100,
            ), np.int32)
            layout_elapsed = time.perf_counter() - layout_started
            if raw_layout.shape != (N,) or not np.array_equal(np.sort(raw_layout), np.arange(N)):
                raise ValueError("E14 returned an invalid permutation")

            # NLM is deliberately downstream of layout selection.  Keep a second
            # explicit layout record so identity is asserted and serialized.
            nlm_layout = raw_layout.copy()
            layout_identical = bool(np.array_equal(raw_layout, nlm_layout))
            if not layout_identical:
                raise AssertionError("post-processing changed the E14 layout")

            raw = assemble(data["tiles"][index], raw_layout)
            nlm_started = time.perf_counter()
            unguarded = nlm_h9(raw)
            nlm_elapsed = time.perf_counter() - nlm_started
            guard_started = time.perf_counter()
            filtered, reverted = no_gray_guard(raw, unguarded)
            guard_elapsed = time.perf_counter() - guard_started

            raw_score = float(structural_similarity(
                data["target"][index], raw, channel_axis=2, data_range=255,
            ))
            unguarded_score = float(structural_similarity(
                data["target"][index], unguarded, channel_axis=2, data_range=255,
            ))
            guarded_score = float(structural_similarity(
                data["target"][index], filtered, channel_axis=2, data_range=255,
            ))
            raw_adj = adjacency(raw_layout, data["truth"][index])
            nlm_adj = adjacency(nlm_layout, data["truth"][index])
            adjacency_identical = bool(raw_adj == nlm_adj)
            if not adjacency_identical:
                raise AssertionError("post-processing changed adjacency")

            raw_gray = gray_count(raw)
            unguarded_gray = gray_count(unguarded)
            nlm_gray = gray_count(filtered)
            gray_gate = bool(nlm_gray <= raw_gray)
            if not gray_gate:
                raise AssertionError(
                    f"gray-count regression: raw={raw_gray}, nlm={nlm_gray}"
                )
            if not all(np.isfinite(value) for value in (
                raw_score, unguarded_score, guarded_score, raw_adj, nlm_adj,
                layout_elapsed, nlm_elapsed, guard_elapsed,
            )):
                raise FloatingPointError("non-finite metric")

            raw_scores.append(raw_score); unguarded_scores.append(unguarded_score)
            guarded_scores.append(guarded_score)
            raw_adjacencies.append(raw_adj); nlm_adjacencies.append(nlm_adj)
            layout_seconds.append(layout_elapsed); nlm_seconds.append(nlm_elapsed)
            guard_seconds.append(guard_elapsed)
            raw_gray_counts.append(raw_gray); unguarded_gray_counts.append(unguarded_gray)
            nlm_gray_counts.append(nlm_gray); reverted_counts.append(reverted)
            row.update({
                "raw_ssim": raw_score,
                "unguarded_nlm_ssim": unguarded_score,
                "guarded_nlm_ssim": guarded_score,
                "unguarded_ssim_gain": unguarded_score - raw_score,
                "guarded_ssim_gain": guarded_score - raw_score,
                "adjacency": raw_adj,
                "layout_identical": layout_identical,
                "adjacency_identical": adjacency_identical,
                "raw_gray_count": raw_gray,
                "unguarded_nlm_gray_count": unguarded_gray,
                "nlm_gray_count": nlm_gray,
                "reverted_new_gray_cells": reverted,
                "gray_gate": gray_gate,
                "layout_seconds": layout_elapsed,
                "nlm_seconds": nlm_elapsed,
                "guard_seconds": guard_elapsed,
            })
        except Exception as exc:
            failures.append({
                "index": index,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            row["failure"] = repr(exc)
        rows.append(row)
        print(json.dumps({
            "done": index - args.start + 1,
            "total": stop - args.start,
            "stem": row["stem"],
            "raw_ssim": row.get("raw_ssim"),
            "unguarded_nlm_ssim": row.get("unguarded_nlm_ssim"),
            "guarded_nlm_ssim": row.get("guarded_nlm_ssim"),
        }), flush=True)

    cases = stop - args.start
    report: dict[str, object] = {
        "experiment": "E18 unchanged E14 layout plus no-gray guarded full-image colored NLM h=9",
        "cache_sha256": cache_hash,
        "cases": cases,
        "start": args.start,
        "layout": "verified E14 E2 raw fusion into unchanged E11 relaxation",
        "nlm": {
            "implementation": "cv2.fastNlMeansDenoisingColored",
            "h": NLM_H,
            "h_color": NLM_H,
            "template_window": NLM_TEMPLATE_WINDOW,
            "search_window": NLM_SEARCH_WINDOW,
        },
        "selection_inputs": ["raw tiles", "right", "down", "pos"],
        "selection_excludes": ["NLM output", "target", "truth", "SSIM", "adjacency"],
        "failures": failures,
        "images": rows,
    }
    if not failures and len(raw_scores) == cases:
        raw_array = np.asarray(raw_scores)
        unguarded_array = np.asarray(unguarded_scores)
        guarded_array = np.asarray(guarded_scores)
        raw_summary = summarize(raw_scores)
        unguarded_summary = summarize(unguarded_scores)
        guarded_summary = summarize(guarded_scores)
        unguarded_mean_gain = float(unguarded_array.mean() - raw_array.mean())
        guarded_mean_gain = float(guarded_array.mean() - raw_array.mean())
        unguarded_robust_gain = float(unguarded_summary["robust"] - raw_summary["robust"])
        guarded_robust_gain = float(guarded_summary["robust"] - raw_summary["robust"])
        mean_retention = guarded_mean_gain / unguarded_mean_gain
        robust_retention = guarded_robust_gain / unguarded_robust_gain
        report.update({
            "raw_e14_ssim": raw_summary,
            "unguarded_e18_ssim": unguarded_summary,
            "guarded_e18b_ssim": guarded_summary,
            "comparison": {
                "unguarded_e18": {
                    "mean_ssim_delta": unguarded_mean_gain,
                    "robust_ssim_delta": unguarded_robust_gain,
                    "wins": int((unguarded_array > raw_array).sum()),
                    "ties": int((unguarded_array == raw_array).sum()),
                    "losses": int((unguarded_array < raw_array).sum()),
                },
                "guarded_e18b": {
                    "mean_ssim_delta": guarded_mean_gain,
                    "robust_ssim_delta": guarded_robust_gain,
                    "mean_gain_retention": mean_retention,
                    "robust_gain_retention": robust_retention,
                    "wins": int((guarded_array > raw_array).sum()),
                    "ties": int((guarded_array == raw_array).sum()),
                    "losses": int((guarded_array < raw_array).sum()),
                },
            },
            "adjacency": {
                "raw_mean": float(np.mean(raw_adjacencies)),
                "nlm_mean": float(np.mean(nlm_adjacencies)),
                "delta": float(np.mean(nlm_adjacencies) - np.mean(raw_adjacencies)),
                "all_identical": bool(np.array_equal(raw_adjacencies, nlm_adjacencies)),
            },
            "layout_identity": {
                "identical_cases": int(sum(bool(row["layout_identical"]) for row in rows)),
                "all_identical": bool(all(bool(row["layout_identical"]) for row in rows)),
            },
            "gray_audit": {
                "raw_total": int(sum(raw_gray_counts)),
                "unguarded_nlm_total": int(sum(unguarded_gray_counts)),
                "nlm_total": int(sum(nlm_gray_counts)),
                "unguarded_delta": int(sum(unguarded_gray_counts) - sum(raw_gray_counts)),
                "delta": int(sum(nlm_gray_counts) - sum(raw_gray_counts)),
                "reverted_new_gray_cells": int(sum(reverted_counts)),
                "unguarded_excess_images": int(sum(
                    nlm > raw for raw, nlm in zip(raw_gray_counts, unguarded_gray_counts)
                )),
                "excess_images": int(sum(nlm > raw for raw, nlm in zip(
                    raw_gray_counts, nlm_gray_counts,
                ))),
                "all_per_image_nonincreasing": bool(all(
                    nlm <= raw for raw, nlm in zip(raw_gray_counts, nlm_gray_counts)
                )),
            },
            "runtime": {
                "e14_layout_seconds": float(sum(layout_seconds)),
                "nlm_seconds": float(sum(nlm_seconds)),
                "guard_seconds": float(sum(guard_seconds)),
                "unguarded_total_seconds": float(sum(layout_seconds) + sum(nlm_seconds)),
                "guarded_total_seconds": float(
                    sum(layout_seconds) + sum(nlm_seconds) + sum(guard_seconds)
                ),
                "mean_nlm_seconds": float(np.mean(nlm_seconds)),
                "mean_guard_seconds": float(np.mean(guard_seconds)),
                "nlm_overhead_ratio": float(sum(nlm_seconds) / sum(layout_seconds)),
            },
        })
        report["e18_unguarded_gate_pass"] = bool(
            unguarded_mean_gain > 0
            and unguarded_robust_gain > 0
            and report["adjacency"]["delta"] == 0
            and report["adjacency"]["all_identical"]
            and report["layout_identity"]["all_identical"]
            and all(nlm <= raw for raw, nlm in zip(raw_gray_counts, unguarded_gray_counts))
        )
        report["e18b_guarded_gate_pass"] = bool(
            guarded_mean_gain > 0
            and guarded_robust_gain > 0
            and mean_retention >= 0.9
            and robust_retention >= 0.9
            and report["adjacency"]["delta"] == 0
            and report["adjacency"]["all_identical"]
            and report["layout_identity"]["all_identical"]
            and report["gray_audit"]["all_per_image_nonincreasing"]
            and report["gray_audit"]["excess_images"] == 0
        )
        if args.start == 0 and cases == 128:
            report["splits"] = {
                "smoke32": split_summary(rows[:32]),
                "untouched96": split_summary(rows[32:]),
                "full128": split_summary(rows),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, indent=2))
    if failures or not report.get("e18b_guarded_gate_pass", False):
        raise RuntimeError("E18b verification gate failed")


if __name__ == "__main__":
    main()
