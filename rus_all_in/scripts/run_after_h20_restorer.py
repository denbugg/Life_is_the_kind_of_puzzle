#!/usr/bin/env python3
"""Train and preregister a legal tile-local residual after frozen NLM h20.

The workflow is intentionally split into commands:

1. ``train`` may read only the fixed train fit/diagnostic records;
2. ``preregister`` freezes checkpoint/code/config hashes and calibration panels;
3. ``prepare-primary`` has no target-directory argument and commits predictions;
4. ``score-primary`` verifies the commitment before opening primary targets.

Confirmation is deliberately not implemented here: it remains unopened unless
the primary numerical gates pass and a human supplies a zero-severe-artifact
manual review in a separately frozen authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.stats import t as student_t

from aiijc_puzzle.after_h20_restorer import (
    AfterH20ModelConfig,
    AfterH20TileRestorer,
    blend_around_h20,
    infer_frozen_inputs,
    model_config_dict,
    paired_clean_tiles,
    restore_tiles,
)
from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.edge_ranker_final_tail import layout_metrics, paired_bootstrap_ci
from aiijc_puzzle.frozen_final_evaluator import load_rgb_verified
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    IMAGE_SIZE,
    TILE_SIZE,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
INPUTS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
TARGETS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
TRAIN_ROOT = PROJECT_ROOT / "outputs" / "after-h20-restorer" / "train96-diagnostic24-v1"
CONFIG_PATH = PROJECT_ROOT / "configs" / "after_h20_restorer_reused_calibration_v1.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "after-h20-restorer" / "evaluation"

FIT_COUNT = 96
DIAGNOSTIC_COUNT = 24
TRAIN_SELECTION_COUNT = FIT_COUNT + DIAGNOSTIC_COUNT
TRAIN_SEED = 20260830
TRAIN_STEPS = 2_500
TRAIN_BATCH_SIZE = 256
TRUSTED_MARGIN_MIN = 1.0
TRUSTED_RGB_RMSE_MAX = 80.0
MODEL_CONFIG = AfterH20ModelConfig(width=32, blocks=8, residual_limit=64.0 / 255.0)
DIAGNOSTIC_ALPHAS = (0.125, 0.25, 0.5, 1.0)
PRIMARY_OFFSET = 192
PRIMARY_COUNT = 24
CONFIRMATION_OFFSET = 216
CONFIRMATION_COUNT = 24
MANIFEST_SHA256 = "4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da"
PROTOCOL_DIGEST = "2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4"
BOOTSTRAP_REPLICATES = 20_000

SOURCE_PATHS = {
    "after_h20_restorer": PROJECT_ROOT / "src" / "aiijc_puzzle" / "after_h20_restorer.py",
    "runner": PROJECT_ROOT / "scripts" / "run_after_h20_restorer.py",
    "candidate_supply": PROJECT_ROOT / "src" / "aiijc_puzzle" / "candidate_supply.py",
    "legacy_upgrade": PROJECT_ROOT / "src" / "aiijc_puzzle" / "legacy_upgrade.py",
    "postassembly_harmonizer": (
        PROJECT_ROOT / "src" / "aiijc_puzzle" / "postassembly_harmonizer.py"
    ),
    "pixel_tails": PROJECT_ROOT / "src" / "aiijc_puzzle" / "pixel_tails.py",
    "protocol": PROJECT_ROOT / "src" / "aiijc_puzzle" / "protocol.py",
    "permutation_audit": (PROJECT_ROOT / "src" / "aiijc_puzzle" / "compliant_atlas_decoder.py"),
    "frozen_method_config": (PROJECT_ROOT / "src" / "aiijc_puzzle" / "frozen_final_evaluator.py"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--output-dir", type=Path, default=TRAIN_ROOT)
    train.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    train.add_argument("--steps", type=int, default=TRAIN_STEPS)
    train.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)

    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--train-report", type=Path, default=TRAIN_ROOT / "report.json")
    preregister.add_argument("--checkpoint", type=Path, default=TRAIN_ROOT / "checkpoint.pt")
    preregister.add_argument("--config", type=Path, default=CONFIG_PATH)

    prepare = subparsers.add_parser("prepare-primary")
    prepare.add_argument("--config", type=Path, default=CONFIG_PATH)
    prepare.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    prepare.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    prepare.add_argument("--tile-batch-size", type=int, default=576)

    score = subparsers.add_parser("score-primary")
    score.add_argument("--config", type=Path, default=CONFIG_PATH)
    score.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    score.add_argument("--targets", type=Path, default=TARGETS_DIR)
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in SOURCE_PATHS.items()}


def names_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = b"\0".join(str(record["filename"]).encode() for record in records)
    return hashlib.sha256(payload).hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{','.join(map(str, array.shape))}|".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def choose_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def load_manifest() -> dict[str, Any]:
    if sha256_file(MANIFEST_PATH) != MANIFEST_SHA256:
        raise ValueError("validation manifest hash drifted")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != PROTOCOL_DIGEST:
        raise ValueError("protocol digest drifted")
    return manifest


def write_json_exclusive(path: Path, payload: Mapping[str, Any], *, readonly: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444 if readonly else 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    if readonly:
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def save_checkpoint_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    finally:
        temporary.unlink(missing_ok=True)


def png_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB").save(
        buffer, format="PNG", optimize=False
    )
    return buffer.getvalue()


def write_bytes_exclusive(path: Path, payload: bytes, *, readonly: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444 if readonly else 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    if readonly:
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def tensor_batch(tiles: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    selected = np.ascontiguousarray(tiles[indices])
    return (
        torch.from_numpy(selected).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
        / 255.0
    )


def restoration_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    h20: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    error = prediction - target
    pixel = torch.sqrt(error.square() + 1e-6).mean()
    pred_h = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_h = target[..., :, 1:] - target[..., :, :-1]
    pred_v = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_v = target[..., 1:, :] - target[..., :-1, :]
    gradient = 0.5 * (
        torch.sqrt((pred_h - target_h).square() + 1e-6).mean()
        + torch.sqrt((pred_v - target_v).square() + 1e-6).mean()
    )
    residual_penalty = (prediction - h20).square().mean()
    loss = pixel + 0.15 * gradient + 0.02 * residual_penalty
    return loss, {
        "loss": float(loss.detach()),
        "pixel": float(pixel.detach()),
        "gradient": float(gradient.detach()),
        "residual_penalty": float(residual_penalty.detach()),
    }


def build_fit_arrays(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, np.ndarray], Any]:
    pre_parts: list[np.ndarray] = []
    h20_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    for index, record in enumerate(records):
        filename = str(record["filename"])
        dirty = load_rgb_verified(INPUTS_DIR / filename, str(record["input_sha256"]))
        clean = load_rgb_verified(TARGETS_DIR / filename, str(record["target_sha256"]))
        inferred = infer_frozen_inputs(dirty, include_h28=False)
        clean_tiles, margins, alignment = paired_clean_tiles(dirty, clean, inferred["layout"])
        ordered_dirty = split_tiles(dirty)[inferred["layout"]]
        paired_rmse = np.sqrt(
            np.mean(
                np.square(ordered_dirty.astype(np.float32) - clean_tiles.astype(np.float32)),
                axis=(1, 2, 3),
            )
        )
        trusted = (margins >= TRUSTED_MARGIN_MIN) & (paired_rmse <= TRUSTED_RGB_RMSE_MAX)
        pre_parts.append(split_tiles(inferred["pre_h20"])[trusted])
        h20_parts.append(split_tiles(inferred["h20"])[trusted])
        target_parts.append(clean_tiles[trusted])
        rows.append(
            {
                "filename": filename,
                "trusted_tiles": int(trusted.sum()),
                "trusted_fraction": float(trusted.mean()),
                "paired_rmse_quantiles": {
                    str(q): float(np.quantile(paired_rmse, q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)
                },
                "alignment": alignment,
                "layout_objective": inferred["layout_objective"],
                "permutation_audit_passed": bool(inferred["audit"]["passed"]),
            }
        )
        print(
            json.dumps(
                {
                    "phase": "fit-preprocess",
                    "completed": index + 1,
                    "total": len(records),
                    "filename": filename,
                    "trusted": int(trusted.sum()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    arrays = {
        "pre_h20": np.ascontiguousarray(np.concatenate(pre_parts)),
        "h20": np.ascontiguousarray(np.concatenate(h20_parts)),
        "target": np.ascontiguousarray(np.concatenate(target_parts)),
    }
    return arrays, {
        "seconds": perf_counter() - started,
        "total_trusted_tiles": int(len(arrays["target"])),
        "per_board": rows,
    }


def safety_metrics(rgb: np.ndarray) -> dict[str, float]:
    image = np.asarray(rgb)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError("safety metrics require uint8 RGB 480x480")
    luminance = (
        0.299 * image[..., 0].astype(np.float64)
        + 0.587 * image[..., 1].astype(np.float64)
        + 0.114 * image[..., 2].astype(np.float64)
    )
    horizontal = np.abs(np.diff(luminance, axis=1))
    vertical = np.abs(np.diff(luminance, axis=0))
    positions = np.arange(1, IMAGE_SIZE)
    h_interior = horizontal[:, positions % TILE_SIZE != 0]
    v_interior = vertical[positions % TILE_SIZE != 0, :]
    h_grid = horizontal[:, positions % TILE_SIZE == 0]
    v_grid = vertical[positions % TILE_SIZE == 0, :]
    within = float((h_interior.sum() + v_interior.sum()) / (h_interior.size + v_interior.size))
    grid = float((h_grid.sum() + v_grid.sum()) / (h_grid.size + v_grid.size))
    laplacian = float(np.mean(np.abs(cv2.Laplacian(luminance, cv2.CV_64F, ksize=3))))
    return {
        "within_tile_gradient": within,
        "laplacian_energy": laplacian,
        "grid_ratio": grid / max(within, 1e-12),
    }


def safety_ratios(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    numerator = safety_metrics(candidate)
    denominator = safety_metrics(baseline)
    return {
        "within_tile_gradient_retention": (
            numerator["within_tile_gradient"] / denominator["within_tile_gradient"]
        ),
        "laplacian_retention": (numerator["laplacian_energy"] / denominator["laplacian_energy"]),
        "grid_ratio_relative": numerator["grid_ratio"] / denominator["grid_ratio"],
    }


def summarize_ratios(values: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "within_tile_gradient_retention",
        "laplacian_retention",
        "grid_ratio_relative",
    ):
        array = np.asarray([row[name] for row in values], dtype=np.float64)
        result[name] = {
            "mean": float(array.mean()),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return result


def strict_safety_pass(summary: Mapping[str, Any], *, pure: bool = False) -> bool:
    thresholds = (
        (0.85, 0.75, 0.78, 0.65, 1.03, 1.08) if pure else (0.80, 0.70, 0.72, 0.60, 1.05, 1.12)
    )
    return bool(
        summary["within_tile_gradient_retention"]["mean"] >= thresholds[0]
        and summary["within_tile_gradient_retention"]["min"] >= thresholds[1]
        and summary["laplacian_retention"]["mean"] >= thresholds[2]
        and summary["laplacian_retention"]["min"] >= thresholds[3]
        and summary["grid_ratio_relative"]["mean"] <= thresholds[4]
        and summary["grid_ratio_relative"]["max"] <= thresholds[5]
    )


def load_checkpoint(path: Path, device: torch.device, expected_sha: str | None = None) -> Any:
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise ValueError("checkpoint hash drifted")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = AfterH20ModelConfig(**payload["model_config"])
    model = AfterH20TileRestorer(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def diagnose(
    records: Sequence[Mapping[str, Any]],
    model: AfterH20TileRestorer,
    device: torch.device,
) -> dict[str, Any]:
    arms = ["h20", "h28", *(f"alpha_{alpha:g}" for alpha in DIAGNOSTIC_ALPHAS)]
    scores = {arm: [] for arm in arms}
    safety = {arm: [] for arm in arms[2:]}
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        filename = str(record["filename"])
        dirty = load_rgb_verified(INPUTS_DIR / filename, str(record["input_sha256"]))
        target = load_rgb_verified(TARGETS_DIR / filename, str(record["target_sha256"]))
        inferred = infer_frozen_inputs(dirty)
        restored = restore_tiles(model, inferred["pre_h20"], inferred["h20"], device=device)
        predictions = {"h20": inferred["h20"], "h28": inferred["h28"]}
        predictions.update(
            {
                f"alpha_{alpha:g}": blend_around_h20(inferred["h20"], restored, alpha)
                for alpha in DIAGNOSTIC_ALPHAS
            }
        )
        board_scores = {arm: contest_ssim(target, value) for arm, value in predictions.items()}
        for arm, value in board_scores.items():
            scores[arm].append(value)
        for arm in arms[2:]:
            safety[arm].append(safety_ratios(predictions[arm], predictions["h20"]))
        rows.append(
            {
                "filename": filename,
                "scores": board_scores,
                "layout_sha256": array_digest(inferred["layout"].astype("<i4")),
                "permutation_audit_passed": bool(inferred["audit"]["passed"]),
            }
        )
        print(
            json.dumps(
                {
                    "phase": "train-diagnostic",
                    "completed": index + 1,
                    "total": len(records),
                    "filename": filename,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summaries: dict[str, Any] = {}
    baseline = np.asarray(scores["h20"])
    for arm in arms:
        values = np.asarray(scores[arm])
        summaries[arm] = {
            "mean_ssim": float(values.mean()),
            "mean_gain_vs_h20": float((values - baseline).mean()),
            "wins_vs_h20": int(np.sum(values > baseline)),
        }
        if arm in safety:
            summaries[arm]["safety"] = summarize_ratios(safety[arm])
            summaries[arm]["safety_pass"] = strict_safety_pass(
                summaries[arm]["safety"], pure=arm == "alpha_1"
            )

    conventional = ["alpha_0.125", "alpha_0.25", "alpha_0.5"]
    eligible = [arm for arm in conventional if summaries[arm]["safety_pass"]]
    pure = "alpha_1"
    if summaries[pure]["safety_pass"]:
        eligible.append(pure)
    if not eligible:
        selected = "alpha_0.125"
        selection_note = "no candidate passed train-only safety; least alpha frozen"
    else:
        selected = sorted(
            eligible,
            key=lambda arm: (-summaries[arm]["mean_ssim"], DIAGNOSTIC_ALPHAS[arms[2:].index(arm)]),
        )[0]
        selection_note = "highest train-diagnostic mean among fixed safety-pass candidates"
    return {
        "arms": arms,
        "summaries": summaries,
        "selected_arm": selected,
        "selected_alpha": float(selected.removeprefix("alpha_")),
        "selection_note": selection_note,
        "per_board": rows,
    }


def train_command(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"training output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = choose_device(args.device)
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    manifest = load_manifest()
    selected = select_manifest_records(manifest, "train", limit=TRAIN_SELECTION_COUNT)
    fit_records = selected[:FIT_COUNT]
    diagnostic_records = selected[FIT_COUNT:]
    arrays, preprocessing = build_fit_arrays(fit_records)

    model = AfterH20TileRestorer(MODEL_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=2e-5
    )
    generator = np.random.default_rng(TRAIN_SEED)
    history: list[dict[str, Any]] = []
    started = perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        indices = generator.integers(0, len(arrays["target"]), size=args.batch_size)
        pre = tensor_batch(arrays["pre_h20"], indices, device)
        h20 = tensor_batch(arrays["h20"], indices, device)
        target = tensor_batch(arrays["target"], indices, device)
        prediction = model(pre, h20)
        loss, components = restoration_loss(prediction, target, h20)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            synchronize(device)
            record = {
                "step": step,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **components,
            }
            history.append(record)
            print(json.dumps({"phase": "train", **record}, sort_keys=True), flush=True)
    training_seconds = perf_counter() - started
    checkpoint_path = args.output_dir / "checkpoint.pt"
    checkpoint_payload = {
        "schema": "aiijc-after-h20-tile-restorer-checkpoint-v1",
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_config": model_config_dict(model),
        "training_contract": {
            "seed": TRAIN_SEED,
            "fit_count": FIT_COUNT,
            "diagnostic_count": DIAGNOSTIC_COUNT,
            "fit_names_digest": names_digest(fit_records),
            "diagnostic_names_digest": names_digest(diagnostic_records),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "trusted_margin_min": TRUSTED_MARGIN_MIN,
            "trusted_rgb_rmse_max": TRUSTED_RGB_RMSE_MAX,
            "objective": "paired clean identity; Charbonnier + gradient + residual penalty",
            "inference_geometry": "shared independent upright 20x20 tiles",
            "semantic_source_sha256": source_hashes(),
            "manifest_sha256": MANIFEST_SHA256,
            "protocol_digest": PROTOCOL_DIGEST,
        },
    }
    save_checkpoint_exclusive(checkpoint_path, checkpoint_payload)
    checkpoint_sha = sha256_file(checkpoint_path)
    model.eval()
    diagnostic = diagnose(diagnostic_records, model, device)
    report = {
        "schema": "aiijc-after-h20-tile-restorer-train-report-v1",
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_config": model_config_dict(model),
        "device": str(device),
        "training_seconds": training_seconds,
        "training_history": history,
        "preprocessing": preprocessing,
        "fit": {
            "count": FIT_COUNT,
            "names_digest": names_digest(fit_records),
            "filenames": [str(record["filename"]) for record in fit_records],
        },
        "diagnostic": {
            "count": DIAGNOSTIC_COUNT,
            "names_digest": names_digest(diagnostic_records),
            "filenames": [str(record["filename"]) for record in diagnostic_records],
            **diagnostic,
        },
        "source_sha256_at_training": source_hashes(),
        "legality": {
            "targets_used_only_for_train_identity_pairing_and_train_diagnostic": True,
            "calibration_targets_opened": False,
            "model_has_no_cross_tile_or_cross_board_context": True,
            "tile_geometry": "upright unchanged 20x20",
            "inference_reference_or_substitution": False,
        },
    }
    write_json_exclusive(args.output_dir / "report.json", report, readonly=True)
    print(
        json.dumps(
            {
                "checkpoint_sha256": checkpoint_sha,
                "diagnostic": diagnostic["summaries"],
                "selected_alpha": diagnostic["selected_alpha"],
            },
            indent=2,
        ),
        flush=True,
    )


def exact_panel(manifest: Mapping[str, Any], offset: int, count: int) -> Any:
    prefix = select_manifest_records(
        manifest,
        "calibration",
        limit=offset + count,
        seed=EXPERIMENT_SUBSET_SEED,
        namespace=EXPERIMENT_SUBSET_NAMESPACE,
    )
    return tuple(prefix[offset:])


def preregister_command(args: argparse.Namespace) -> None:
    report = json.loads(args.train_report.read_text(encoding="utf-8"))
    if sha256_file(args.checkpoint) != report["checkpoint_sha256"]:
        raise ValueError("training checkpoint/report mismatch")
    manifest = load_manifest()
    primary = exact_panel(manifest, PRIMARY_OFFSET, PRIMARY_COUNT)
    confirmation = exact_panel(manifest, CONFIRMATION_OFFSET, CONFIRMATION_COUNT)
    if {row["filename"] for row in primary} & {row["filename"] for row in confirmation}:
        raise ValueError("primary and confirmation panels overlap")
    selected_alpha = float(report["diagnostic"]["selected_alpha"])
    config = {
        "schema": "aiijc-after-h20-restorer-reused-calibration-preregistration-v1",
        "status_at_freeze": "immutable_before_this_run_decodes_any_evaluation_target_pixels",
        "scientific_scope": (
            "reused calibration; all 700 records were historically exposed in legacy report; "
            "primary and confirmation are only mutually disjoint"
        ),
        "historical_exposure": {
            "freshness_claim": False,
            "historically_exposed_records": "all calibration records",
            "legacy_report": "outputs/legacy-upgrade/calibration700-champion/report.json",
            "legacy_report_sha256": sha256_file(
                PROJECT_ROOT
                / "outputs"
                / "legacy-upgrade"
                / "calibration700-champion"
                / "report.json"
            ),
        },
        "protocol": {
            "manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "manifest_sha256": MANIFEST_SHA256,
            "protocol_digest": PROTOCOL_DIGEST,
            "selector": {
                "namespace": EXPERIMENT_SUBSET_NAMESPACE,
                "seed": EXPERIMENT_SUBSET_SEED,
            },
            "primary": {
                "split": "calibration",
                "offset": PRIMARY_OFFSET,
                "count": PRIMARY_COUNT,
                "filename_nul_sha256": names_digest(primary),
            },
            "confirmation_if_and_only_if_primary_and_manual_pass": {
                "split": "calibration",
                "offset": CONFIRMATION_OFFSET,
                "count": CONFIRMATION_COUNT,
                "filename_nul_sha256": names_digest(confirmation),
            },
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve().relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(args.checkpoint),
            "train_report_path": str(args.train_report.resolve().relative_to(PROJECT_ROOT)),
            "train_report_sha256": sha256_file(args.train_report),
            "fit_count": FIT_COUNT,
            "diagnostic_count": DIAGNOSTIC_COUNT,
            "fit_names_digest": report["fit"]["names_digest"],
            "diagnostic_names_digest": report["diagnostic"]["names_digest"],
        },
        "source_sha256": source_hashes(),
        "inference": {
            "layout": "dirty-only bilateral scores -> solve_buddies(max_edges=96)",
            "raw_permutation_audit": "strict before any restoration",
            "harmonization": "exact frozen RGB offsets then bounded luminance gains",
            "conditioning": "same-position upright 20x20 pre-h20 and post-h20 RGB tiles",
            "model_operation": "one shared tile-local bounded residual around h20",
            "model_cross_tile_context": False,
            "reference_template_substitution_generation": False,
            "selected_alpha_from_train_diagnostic": selected_alpha,
            "arms": {
                "A_h20": "harmonized full-canvas NLM h20",
                "B_h28": "same harmonized full-canvas NLM h28",
                "C_after_h20": f"convex blend alpha={selected_alpha:g} around A using model",
            },
        },
        "primary_gate": {
            "candidate": "C_after_h20",
            "mean_ssim_min": 0.27,
            "paired_ci95_lower_vs_A_strict_gt": 0.0,
            "paired_ci95_lower_vs_B_strict_gt": 0.0,
            "wins_vs_A_min": 18,
            "wins_vs_B_min": 15,
            "layout_adjacency_ci95_lower_vs_A_min": 0.0,
            "translation_aligned_placement_non_decrease": True,
            "mean_within_tile_gradient_retention_min": 0.80,
            "minimum_board_within_tile_gradient_retention_min": 0.70,
            "mean_laplacian_retention_min": 0.72,
            "minimum_board_laplacian_retention_min": 0.60,
            "mean_grid_ratio_relative_max": 1.05,
            "maximum_board_grid_ratio_relative_max": 1.12,
            "manual_severe_artifact_count_required": 0,
            "all_numeric_gates_and_manual_gate_required": True,
        },
        "confirmation_policy": (
            "forbidden unless every primary numerical gate passes and a frozen manual sheet "
            "authorization records zero severe artifacts; no retuning"
        ),
        "test_holdout_production_policy": "forbidden in this experiment",
    }
    write_json_exclusive(args.config, config, readonly=True)
    print(
        json.dumps(
            {
                "config": str(args.config),
                "sha256": sha256_file(args.config),
                "primary_digest": names_digest(primary),
                "confirmation_digest": names_digest(confirmation),
            },
            indent=2,
        )
    )


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        config.get("status_at_freeze")
        != "immutable_before_this_run_decodes_any_evaluation_target_pixels"
    ):
        raise ValueError("invalid preregistration status")
    if config["protocol"]["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("manifest contract drifted")
    for name, expected in config["source_sha256"].items():
        if source_hashes().get(name) != expected:
            raise ValueError(f"semantic source drifted after preregistration: {name}")
    return config


def primary_records(config: Mapping[str, Any]) -> Any:
    panel = config["protocol"]["primary"]
    records = exact_panel(load_manifest(), int(panel["offset"]), int(panel["count"]))
    if names_digest(records) != panel["filename_nul_sha256"]:
        raise ValueError("primary filename digest drifted")
    return records


def artifact_record(root: Path, board_stem: str, name: str, value: np.ndarray) -> Any:
    relative = Path("artifacts") / board_stem / f"{name}.png"
    path = root / relative
    write_bytes_exclusive(path, png_bytes(value))
    return {
        "path": relative.as_posix(),
        "file_sha256": sha256_file(path),
        "array_sha256": array_digest(value),
    }


def prepare_primary_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    config_sha = sha256_file(args.config)
    root = args.output_root / config_sha / "primary"
    if root.exists():
        raise FileExistsError(f"primary output already exists: {root}")
    root.mkdir(parents=True)
    checkpoint_path = PROJECT_ROOT / config["checkpoint"]["path"]
    device = choose_device(args.device)
    model, _ = load_checkpoint(checkpoint_path, device, config["checkpoint"]["sha256"])
    records = primary_records(config)
    alpha = float(config["inference"]["selected_alpha_from_train_diagnostic"])
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        filename = str(record["filename"])
        dirty = load_rgb_verified(INPUTS_DIR / filename, str(record["input_sha256"]))
        inferred = infer_frozen_inputs(dirty)
        model_image = restore_tiles(
            model,
            inferred["pre_h20"],
            inferred["h20"],
            device=device,
            batch_size=args.tile_batch_size,
        )
        predictions = {
            "raw": inferred["raw"],
            "pre_h20": inferred["pre_h20"],
            "A_h20": inferred["h20"],
            "B_h28": inferred["h28"],
            "model_full": model_image,
            "C_after_h20": blend_around_h20(inferred["h20"], model_image, alpha),
        }
        artifacts = {
            name: artifact_record(root, Path(filename).stem, name, value)
            for name, value in predictions.items()
        }
        layout_relative = Path("artifacts") / Path(filename).stem / "layout.npy"
        layout_buffer = io.BytesIO()
        np.save(layout_buffer, inferred["layout"], allow_pickle=False)
        write_bytes_exclusive(root / layout_relative, layout_buffer.getvalue())
        rows.append(
            {
                "index": index,
                "filename": filename,
                "input_sha256": str(record["input_sha256"]),
                "target_sha256_committed_but_not_opened": str(record["target_sha256"]),
                "layout_path": layout_relative.as_posix(),
                "layout_file_sha256": sha256_file(root / layout_relative),
                "layout_array_sha256": array_digest(inferred["layout"].astype("<i4")),
                "permutation_audit": inferred["audit"],
                "artifacts": artifacts,
                "safety": {
                    arm: safety_metrics(predictions[arm])
                    for arm in ("A_h20", "B_h28", "C_after_h20")
                },
            }
        )
        print(
            json.dumps(
                {
                    "phase": "prepare-primary",
                    "completed": index + 1,
                    "total": len(records),
                    "filename": filename,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    commitment = {
        "schema": "aiijc-after-h20-restorer-prediction-commitment-v1",
        "config_path": str(args.config.resolve().relative_to(PROJECT_ROOT)),
        "config_sha256": config_sha,
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "source_sha256": config["source_sha256"],
        "panel": "primary",
        "count": len(records),
        "filename_nul_sha256": names_digest(records),
        "selected_alpha": alpha,
        "target_access_contract": {
            "prepare_command_has_no_target_directory_argument": True,
            "no_evaluation_target_was_opened": True,
            "raw_layout_pre_h20_h20_h28_model_and_blend_frozen": True,
            "all_permutation_audits_passed": all(
                row["permutation_audit"]["passed"] for row in rows
            ),
        },
        "aggregate_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "per_board": rows,
    }
    write_json_exclusive(root / "prediction-commitment.json", commitment, readonly=True)
    print(
        json.dumps(
            {
                "commitment": str(root / "prediction-commitment.json"),
                "sha256": sha256_file(root / "prediction-commitment.json"),
            },
            indent=2,
        )
    )


def load_artifact(root: Path, record: Mapping[str, Any], arm: str) -> np.ndarray:
    metadata = record["artifacts"][arm]
    path = root / metadata["path"]
    if sha256_file(path) != metadata["file_sha256"]:
        raise ValueError(f"artifact file drifted: {path}")
    with Image.open(path) as image:
        image.load()
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array_digest(value) != metadata["array_sha256"]:
        raise ValueError(f"artifact array drifted: {path}")
    return value


def paired_t_ci(differences: np.ndarray) -> list[float]:
    values = np.asarray(differences, dtype=np.float64)
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    critical = float(student_t.ppf(0.975, len(values) - 1))
    mean = float(values.mean())
    return [mean - critical * standard_error, mean + critical * standard_error]


def create_contact_sheet(root: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    thumb = 160
    label = 24
    sheet = Image.new("RGB", (thumb * 4, (thumb + label) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        for column, arm in enumerate(("input", "A_h20", "B_h28", "C_after_h20")):
            if arm == "input":
                value = load_rgb_verified(INPUTS_DIR / row["filename"], row["input_sha256"])
            else:
                value = load_artifact(root, row, arm)
            image = Image.fromarray(value).resize((thumb, thumb), Image.Resampling.LANCZOS)
            sheet.paste(image, (column * thumb, row_index * (thumb + label)))
            draw.text(
                (column * thumb + 2, row_index * (thumb + label) + thumb + 2), arm, fill="black"
            )
    path = root / "manual-contact-sheet.png"
    write_bytes_exclusive(path, png_bytes(np.asarray(sheet)))
    return path


def score_primary_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    config_sha = sha256_file(args.config)
    root = args.output_root / config_sha / "primary"
    commitment_path = root / "prediction-commitment.json"
    commitment_sha_before_targets = sha256_file(commitment_path)
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    if commitment["config_sha256"] != config_sha:
        raise ValueError("commitment/config mismatch")
    records = primary_records(config)
    if commitment["filename_nul_sha256"] != names_digest(records):
        raise ValueError("commitment panel drifted")
    if len(commitment["per_board"]) != len(records):
        raise ValueError("commitment board count drifted")

    arms = ("A_h20", "B_h28", "C_after_h20")
    scores = {arm: [] for arm in arms}
    ratios: list[dict[str, float]] = []
    adjacency_differences: list[float] = []
    translation_differences: list[float] = []
    rows: list[dict[str, Any]] = []
    for record, committed in zip(records, commitment["per_board"], strict=True):
        if committed["filename"] != record["filename"]:
            raise ValueError("commitment row order drifted")
        target = load_rgb_verified(args.targets / record["filename"], record["target_sha256"])
        predictions = {arm: load_artifact(root, committed, arm) for arm in arms}
        board_scores = {arm: contest_ssim(target, value) for arm, value in predictions.items()}
        for arm in arms:
            scores[arm].append(board_scores[arm])
        ratios.append(safety_ratios(predictions["C_after_h20"], predictions["A_h20"]))
        dirty = load_rgb_verified(INPUTS_DIR / record["filename"], record["input_sha256"])
        recovered = recover_layout(split_tiles(dirty), split_tiles(target))
        layout_path = root / committed["layout_path"]
        if sha256_file(layout_path) != committed["layout_file_sha256"]:
            raise ValueError("layout file drifted")
        layout = np.load(layout_path, allow_pickle=False)
        geometry_a = layout_metrics(layout, recovered)
        geometry_c = layout_metrics(layout, recovered)
        adjacency_differences.append(geometry_c["adjacency"] - geometry_a["adjacency"])
        translation_differences.append(
            geometry_c["translation_aligned_placement"]
            - geometry_a["translation_aligned_placement"]
        )
        rows.append(
            {
                "filename": record["filename"],
                "input_sha256": record["input_sha256"],
                "scores": board_scores,
                "geometry": geometry_a,
            }
        )

    score_summary: dict[str, Any] = {}
    for arm in arms:
        values = np.asarray(scores[arm])
        score_summary[arm] = {
            "mean_ssim": float(values.mean()),
            "std_ssim": float(values.std()),
            "min_ssim": float(values.min()),
            "max_ssim": float(values.max()),
        }
    candidate = np.asarray(scores["C_after_h20"])
    baseline_a = np.asarray(scores["A_h20"])
    baseline_b = np.asarray(scores["B_h28"])
    difference_a = candidate - baseline_a
    difference_b = candidate - baseline_b
    ci_a = paired_bootstrap_ci(difference_a, seed=TRAIN_SEED + 1, replicates=BOOTSTRAP_REPLICATES)
    ci_b = paired_bootstrap_ci(difference_b, seed=TRAIN_SEED + 2, replicates=BOOTSTRAP_REPLICATES)
    adjacency_ci = paired_bootstrap_ci(
        adjacency_differences, seed=TRAIN_SEED + 3, replicates=BOOTSTRAP_REPLICATES
    )
    safety = summarize_ratios(ratios)
    gate = config["primary_gate"]
    checks = {
        "mean_ssim_min": score_summary["C_after_h20"]["mean_ssim"] >= gate["mean_ssim_min"],
        "paired_ci95_lower_vs_A_strict_gt": ci_a["ci95_lower"]
        > gate["paired_ci95_lower_vs_A_strict_gt"],
        "paired_ci95_lower_vs_B_strict_gt": ci_b["ci95_lower"]
        > gate["paired_ci95_lower_vs_B_strict_gt"],
        "wins_vs_A_min": int(np.sum(difference_a > 0)) >= gate["wins_vs_A_min"],
        "wins_vs_B_min": int(np.sum(difference_b > 0)) >= gate["wins_vs_B_min"],
        "layout_adjacency_ci95_lower_vs_A_min": adjacency_ci["ci95_lower"]
        >= gate["layout_adjacency_ci95_lower_vs_A_min"],
        "translation_aligned_placement_non_decrease": float(np.min(translation_differences)) >= 0.0,
        "mean_within_tile_gradient_retention_min": safety["within_tile_gradient_retention"]["mean"]
        >= gate["mean_within_tile_gradient_retention_min"],
        "minimum_board_within_tile_gradient_retention_min": safety[
            "within_tile_gradient_retention"
        ]["min"]
        >= gate["minimum_board_within_tile_gradient_retention_min"],
        "mean_laplacian_retention_min": safety["laplacian_retention"]["mean"]
        >= gate["mean_laplacian_retention_min"],
        "minimum_board_laplacian_retention_min": safety["laplacian_retention"]["min"]
        >= gate["minimum_board_laplacian_retention_min"],
        "mean_grid_ratio_relative_max": safety["grid_ratio_relative"]["mean"]
        <= gate["mean_grid_ratio_relative_max"],
        "maximum_board_grid_ratio_relative_max": safety["grid_ratio_relative"]["max"]
        <= gate["maximum_board_grid_ratio_relative_max"],
    }
    numerical_pass = all(checks.values())
    contact_sheet = None
    if numerical_pass:
        contact_sheet = str(
            create_contact_sheet(root, commitment["per_board"]).relative_to(PROJECT_ROOT)
        )
    report = {
        "schema": "aiijc-after-h20-restorer-primary-report-v1",
        "scientific_scope": config["scientific_scope"],
        "config_sha256": config_sha,
        "commitment_sha256_verified_before_target_decode": commitment_sha_before_targets,
        "score_summary": score_summary,
        "candidate_comparisons": {
            "vs_A_h20": {
                "mean_gain": float(difference_a.mean()),
                "bootstrap_ci95": ci_a,
                "paired_t_ci95": paired_t_ci(difference_a),
                "wins_ties_losses": [
                    int(np.sum(difference_a > 0)),
                    int(np.sum(difference_a == 0)),
                    int(np.sum(difference_a < 0)),
                ],
            },
            "vs_B_h28": {
                "mean_gain": float(difference_b.mean()),
                "bootstrap_ci95": ci_b,
                "paired_t_ci95": paired_t_ci(difference_b),
                "wins_ties_losses": [
                    int(np.sum(difference_b > 0)),
                    int(np.sum(difference_b == 0)),
                    int(np.sum(difference_b < 0)),
                ],
            },
        },
        "geometry": {
            "same_frozen_layout_for_every_arm": True,
            "adjacency_difference_bootstrap_ci95": adjacency_ci,
            "translation_difference_min": float(np.min(translation_differences)),
        },
        "target_free_safety": safety,
        "primary_gate": {
            "checks": checks,
            "all_numerical_gates_passed": numerical_pass,
            "manual_severe_artifact_count": None,
            "manual_gate_authorized": False,
            "all_gates_passed": False,
            "contact_sheet_if_numerically_passed": contact_sheet,
        },
        "confirmation_status": "forbidden_and_unopened"
        if not numerical_pass
        else "awaiting_manual_zero_severe_artifact_authorization",
        "holdout_test_production_status": "forbidden_and_unopened",
        "per_board": rows,
    }
    write_json_exclusive(root / "report.json", report, readonly=True)
    print(
        json.dumps(
            {
                "score_summary": score_summary,
                "comparisons": report["candidate_comparisons"],
                "checks": checks,
                "numerical_pass": numerical_pass,
                "confirmation_status": report["confirmation_status"],
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "preregister":
        preregister_command(args)
    elif args.command == "prepare-primary":
        prepare_primary_command(args)
    elif args.command == "score-primary":
        score_primary_command(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
