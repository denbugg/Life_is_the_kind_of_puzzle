#!/usr/bin/env python3
"""Audit the archived Pasha883 C64 pair model on full 576-candidate retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.legacy_upgrade import directional_scores, solve_buddies
from aiijc_puzzle.pasha883_pairwise import (
    load_pasha883_pairwise,
    pasha883_directional_retrieval_metrics,
    pasha883_full_pair_scores,
)
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    assemble_tiles,
    contest_ssim,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts" / "prior-pasha883"
DEFAULT_INPUTS = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
DEFAULT_TARGETS = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
GRID = 24
TILE_COUNT = GRID * GRID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARTIFACTS / "pair_best.pt")
    parser.add_argument("--permutations", type=Path, default=DEFAULT_ARTIFACTS / "perms.npz")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--indices",
        default="6996,6997,6998,6999",
        help="comma-separated row indices in the archived 7000-board permutation cache",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--buddy-edges", type=int, default=96)
    parser.add_argument(
        "--d64-pooled-ot-r1",
        type=float,
        default=0.17764945652173914,
    )
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def _parse_indices(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("indices must be comma-separated integers") from error
    if len(values) < 4 or len(set(values)) != len(values) or min(values) < 6700:
        raise ValueError("provide at least four distinct archived validation indices >= 6700")
    if max(values) >= 7000:
        raise ValueError("archived cache indices must be below 7000")
    return values


def _mean_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    metrics = rows[0][key]
    return {
        name: float(np.mean([float(row[key][name]) for row in rows]))
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _freeze_scores(
    *,
    index: int,
    filename: str,
    model: torch.nn.Module,
    inputs: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    path = output_dir / f"scores-{index:04d}.npz"
    if path.exists():
        with np.load(path) as archive:
            if str(archive["filename"]) != filename or int(archive["index"]) != index:
                raise ValueError(f"frozen score identity mismatch: {path}")
            right = archive["right"]
            down = archive["down"]
            elapsed = float(archive["inference_seconds"])
        if right.shape != (TILE_COUNT, TILE_COUNT) or down.shape != right.shape:
            raise ValueError(f"frozen score shape mismatch: {path}")
        print(f"reused {path.name} {filename}", flush=True)
        return {
            "index": index,
            "filename": filename,
            "path": str(path),
            "sha256": sha256_file(path),
            "inference_seconds": elapsed,
            "reused": True,
        }

    tiles = split_tiles(_load_rgb(inputs / filename))
    right, down, elapsed = pasha883_full_pair_scores(
        model,
        tiles,
        device=device,
        batch_size=batch_size,
    )
    # This artifact is dirty-only: neither recovered permutation nor target is
    # accepted by the scoring function or serialized here.
    np.savez_compressed(
        path,
        index=np.asarray(index),
        filename=np.asarray(filename),
        right=right,
        down=down,
        inference_seconds=np.asarray(elapsed),
    )
    print(f"froze {path.name} {filename} in {elapsed:.2f}s", flush=True)
    return {
        "index": index,
        "filename": filename,
        "path": str(path),
        "sha256": sha256_file(path),
        "inference_seconds": elapsed,
        "reused": False,
    }


def main() -> None:
    args = parse_args()
    indices = _parse_indices(args.indices)
    if args.batch_size <= 0 or not 1 <= args.buddy_edges <= TILE_COUNT:
        raise ValueError("batch-size and buddy-edges must be positive/in range")
    if not np.isfinite(args.d64_pooled_ot_r1) or not 0 <= args.d64_pooled_ot_r1 <= 1:
        raise ValueError("d64-pooled-ot-r1 must be in [0, 1]")
    device = _device(args.device)
    checkpoint = load_pasha883_pairwise(args.checkpoint, device=device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read only the public identity roster before freezing dirty-only scores.
    with np.load(args.permutations) as identity_archive:
        names = identity_archive["names"]
    if names.shape != (7000,):
        raise ValueError("archived name roster must contain 7000 entries")
    selected = tuple((index, str(names[index])) for index in indices)
    if any(not (args.inputs / filename).is_file() for _, filename in selected):
        raise FileNotFoundError("one or more selected dirty inputs are missing")

    freeze_started = perf_counter()
    frozen = [
        _freeze_scores(
            index=index,
            filename=filename,
            model=checkpoint.model,
            inputs=args.inputs,
            output_dir=output_dir,
            device=device,
            batch_size=args.batch_size,
        )
        for index, filename in selected
    ]
    freeze_seconds = perf_counter() - freeze_started

    # Reference-assisted scoring begins only after every dirty-only matrix is frozen.
    evaluation_started = perf_counter()
    with np.load(args.permutations) as reference_archive:
        reference_names = reference_archive["names"]
        inverse = reference_archive["inv"]
        confidence = reference_archive["conf"]
        rows: list[dict[str, Any]] = []
        for item in frozen:
            index = int(item["index"])
            filename = str(item["filename"])
            if str(reference_names[index]) != filename:
                raise ValueError("reference cache identity changed after score freeze")
            with np.load(item["path"]) as score_archive:
                right = score_archive["right"]
                down = score_archive["down"]
            dirty = split_tiles(_load_rgb(args.inputs / filename))
            target = _load_rgb(args.targets / filename)
            reference = inverse[index].astype(np.int64)
            pasha_local = pasha883_directional_retrieval_metrics(
                right,
                down,
                reference,
            )
            bilateral_right, bilateral_down = directional_scores(
                dirty, views=("bilateral",)
            )["bilateral"]
            bilateral_local = pasha883_directional_retrieval_metrics(
                bilateral_right,
                bilateral_down,
                reference,
            )
            pasha_solve = solve_buddies(right, down, max_edges=args.buddy_edges)
            bilateral_solve = solve_buddies(
                bilateral_right,
                bilateral_down,
                max_edges=args.buddy_edges,
            )
            global_metrics: dict[str, Any] = {}
            for name, solve in (
                ("pasha883_buddies", pasha_solve),
                ("bilateral_buddies", bilateral_solve),
            ):
                geometry = evaluate_layout(
                    solve.layout,
                    reference,
                    reference_is_exact=False,
                ).as_dict()
                reconstruction = assemble_tiles(dirty[solve.layout])
                global_metrics[name] = geometry | {
                    "raw_ssim": contest_ssim(target, reconstruction),
                    "solver": solve.solver,
                    "objective": solve.objective,
                }
            rows.append(
                {
                    "index": index,
                    "filename": filename,
                    "reference_confidence_mean": float(confidence[index].mean()),
                    "reference_confidence_below_0_5_fraction": float(
                        np.mean(confidence[index] < 0.5)
                    ),
                    "pasha883_local": pasha_local,
                    "bilateral_local": bilateral_local,
                    "global": global_metrics,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "evaluated",
                        "index": index,
                        "filename": filename,
                        "pasha_pooled_r1": pasha_local["pooled_r1"],
                        "pasha_pooled_r25": pasha_local["pooled_r25"],
                        "pasha_direct": global_metrics["pasha883_buddies"][
                            "direct_placement"
                        ],
                    }
                ),
                flush=True,
            )
    evaluation_seconds = perf_counter() - evaluation_started

    pasha_mean = _mean_metrics(rows, "pasha883_local")
    bilateral_mean = _mean_metrics(rows, "bilateral_local")
    global_mean = {
        variant: {
            metric: float(np.mean([row["global"][variant][metric] for row in rows]))
            for metric in (
                "direct_placement",
                "row_accuracy",
                "column_accuracy",
                "translation_aligned_placement",
                "adjacency",
                "raw_ssim",
            )
        }
        for variant in ("pasha883_buddies", "bilateral_buddies")
    }
    report = {
        "experiment": "pasha883-archived-c64-full-pair-audit-v1",
        "status": "source-exposed-historical-diagnostic",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "architecture": "checkpoint-exact PairwiseNet C64 global-average-pooling",
            "parameters": sum(parameter.numel() for parameter in checkpoint.model.parameters()),
            "step": checkpoint.step,
            "sampled_validation_accuracy_at_32_mislabeled_acc_at_48": (
                checkpoint.sampled_validation_accuracy_at_32_mislabeled_acc_at_48
            ),
        },
        "reference_cache": {
            "path": str(args.permutations.resolve()),
            "sha256": sha256_file(args.permutations),
            "contract": "target-assisted recovered permutation; inv[position] = input tile",
        },
        "protocol": {
            "selected_indices": list(indices),
            "selected_filenames": [filename for _, filename in selected],
            "historical_validation_range": [6700, 6999],
            "model_selection_exposure": (
                "checkpoint was selected by repeated sampled evaluation over the same last-300 "
                "source roster; code used 32 candidates and 48 anchors although logs said acc@48"
            ),
            "historical_sampled_metric_caveat": (
                "32 random candidates per anchor; random draw could include self, duplicate or "
                "additional true candidate, so checkpoint val is not full-pool R@1"
            ),
            "candidate_pool": "all 576 input tiles; self masked; 552 queries per direction",
            "rank_ties": "one plus strictly-greater score count; exact ties share best rank",
            "score_matrices_frozen_before_reference_or_target_access": True,
            "calibration_split_claim": False,
            "holdout_split_claim": False,
            "competition_test_opened": False,
        },
        "frozen_scores": frozen,
        "runtime_seconds": {
            "dirty_only_freeze_wall": freeze_seconds,
            "reference_assisted_evaluation": evaluation_seconds,
            "dirty_only_model_sum": float(sum(item["inference_seconds"] for item in frozen)),
        },
        "boards": rows,
        "aggregate": {
            "pasha883_local": pasha_mean,
            "bilateral_local": bilateral_mean,
            "global": global_mean,
            "pasha_minus_bilateral_pooled_r1": (
                pasha_mean["pooled_r1"] - bilateral_mean["pooled_r1"]
            ),
            "d64_external_pooled_ot_r1": args.d64_pooled_ot_r1,
            "pasha_minus_d64_external_pooled_r1": (
                pasha_mean["pooled_r1"] - args.d64_pooled_ot_r1
            ),
            "d64_comparison_caveat": (
                "different checkpoint, source panel and exact-synthetic/recovered-label protocol"
            ),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "pasha883_local": pasha_mean,
                "bilateral_local": bilateral_mean,
                "global": global_mean,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
