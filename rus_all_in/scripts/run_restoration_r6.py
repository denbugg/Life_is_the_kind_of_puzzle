#!/usr/bin/env python3
"""Train and evaluate broad post-layout restoration candidates on frozen splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

from aiijc_puzzle.content_substitution import recover_dirty_tile_alignment, render_tiles
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
)
from aiijc_puzzle.restoration_r6 import (
    HistoricalRestoreNet,
    TileAwareDualNAFNet,
    distort_canvas,
    image_tensor,
    nlm_color,
    restoration_loss,
    restore_image,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
INPUTS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
TARGETS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "restoration-r6" / "run.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--architecture", choices=("historical", "dual_naf"), required=True)
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--real-limit", type=int, default=32)
    parser.add_argument("--eval-limit", type=int, default=12)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--eval-split", choices=("calibration", "holdout"), default="calibration")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--crop", type=int, default=160)
    parser.add_argument("--real-probability", type=float, default=0.5)
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument(
        "--nlm-h",
        type=int,
        default=9,
        help="colored NLM strength used as the conditioning and comparison tail",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser


def _selection_digest(records: list[dict[str, str]]) -> str:
    names = "\n".join(record["filename"] for record in records)
    return hashlib.sha256(names.encode()).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError(f"manifest digest mismatch: {path}")
    return manifest


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {path}")
    return np.ascontiguousarray(image[..., ::-1])


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return device


def _model(args: argparse.Namespace) -> torch.nn.Module:
    if args.architecture == "historical":
        return HistoricalRestoreNet(base=args.base, depth=args.depth)
    return TileAwareDualNAFNet(base=args.base, depth=args.depth, blocks=args.blocks)


def _crop(image: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    return np.ascontiguousarray(image[top : top + size, left : left + size])


def _aligned_real(input_image: np.ndarray, target_image: np.ndarray) -> np.ndarray:
    alignment = recover_dirty_tile_alignment(input_image, target_image)
    return render_tiles(alignment.aligned_tiles)


def _contest_ssim(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(structural_similarity(target, prediction, channel_axis=2, data_range=255))


def _normal_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, mean
    error = float(array.std(ddof=1) / np.sqrt(len(array)))
    return mean - 1.96 * error, mean + 1.96 * error


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _train(
    model: torch.nn.Module,
    records: list[dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float | int]], float]:
    clean_cache: dict[str, np.ndarray] = {}
    real_cache: dict[str, np.ndarray] = {}
    prep_started = time.perf_counter()
    for index, record in enumerate(records):
        name = record["filename"]
        target_path = TARGETS_DIR / name
        if sha256_file(target_path) != record["target_sha256"]:
            raise ValueError(f"target hash mismatch: {name}")
        clean_cache[name] = _load_rgb(target_path)
        if index < args.real_limit:
            input_path = INPUTS_DIR / name
            if sha256_file(input_path) != record["input_sha256"]:
                raise ValueError(f"input hash mismatch: {name}")
            real_cache[name] = _aligned_real(_load_rgb(input_path), clean_cache[name])
        if (index + 1) % 16 == 0 or index + 1 == len(records):
            print(
                f"prepare train {index + 1:03d}/{len(records):03d} real={len(real_cache)}",
                flush=True,
            )
    prepare_seconds = time.perf_counter() - prep_started

    rng = np.random.default_rng(args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    history: list[dict[str, float | int]] = []
    names = [record["filename"] for record in records]
    real_names = list(real_cache)
    model.train()
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        use_real = bool(real_names) and rng.random() < args.real_probability
        name = str(rng.choice(real_names if use_real else names))
        clean_full = clean_cache[name]
        cells = clean_full.shape[0] // 20
        crop_cells = args.crop // 20
        top = int(rng.integers(0, cells - crop_cells + 1)) * 20
        left = int(rng.integers(0, cells - crop_cells + 1)) * 20
        clean = _crop(clean_full, top, left, args.crop)
        if use_real:
            dirty = _crop(real_cache[name], top, left, args.crop)
        else:
            dirty = distort_canvas(clean, rng)
        conditioned = nlm_color(dirty, args.nlm_h)
        source = image_tensor(dirty, device)
        nlm = image_tensor(conditioned, device)
        target = image_tensor(clean, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(source, nlm)
        loss = restoration_loss(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % max(args.steps // 12, 1) == 0 or step == args.steps:
            elapsed = time.perf_counter() - started
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "seconds": elapsed,
                "real_example": int(use_real),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    return history, prepare_seconds


def _evaluate(
    model: torch.nn.Module,
    records: list[dict[str, str]],
    device: torch.device,
    nlm_h: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_key = f"nlm_h{nlm_h}"
    double_key = f"nlm_h{nlm_h}_twice"
    restored_nlm_key = f"restored_then_nlm_h{nlm_h}"
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        name = record["filename"]
        input_path, target_path = INPUTS_DIR / name, TARGETS_DIR / name
        if sha256_file(input_path) != record["input_sha256"]:
            raise ValueError(f"input hash mismatch: {name}")
        if sha256_file(target_path) != record["target_sha256"]:
            raise ValueError(f"target hash mismatch: {name}")
        target = _load_rgb(target_path)
        raw = _aligned_real(_load_rgb(input_path), target)
        baseline = nlm_color(raw, nlm_h)
        double_nlm = nlm_color(baseline, nlm_h)
        started = time.perf_counter()
        restored = restore_image(model, raw, device)
        restored_nlm = nlm_color(restored, nlm_h)
        row = {
            "filename": name,
            "raw": _contest_ssim(target, raw),
            baseline_key: _contest_ssim(target, baseline),
            double_key: _contest_ssim(target, double_nlm),
            "restored": _contest_ssim(target, restored),
            restored_nlm_key: _contest_ssim(target, restored_nlm),
            "seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(
            f"eval {index:03d}/{len(records):03d} {name} "
            f"nlm={row[baseline_key]:.5f} restored={row['restored']:.5f} "
            f"r6nlm={row[restored_nlm_key]:.5f}",
            flush=True,
        )
    methods = ("raw", baseline_key, double_key, "restored", restored_nlm_key)
    aggregate: dict[str, Any] = {
        method: float(np.mean([row[method] for row in rows])) for method in methods
    }
    for method in (double_key, "restored", restored_nlm_key):
        deltas = [float(row[method]) - float(row[baseline_key]) for row in rows]
        aggregate[f"{method}_minus_{baseline_key}"] = float(np.mean(deltas))
        aggregate[f"{method}_minus_{baseline_key}_normal_95"] = list(_normal_ci(deltas))
        aggregate[f"{method}_wins_vs_{baseline_key}"] = sum(delta > 0 for delta in deltas)
    aggregate["passed_pixel_gate"] = bool(
        max(
            aggregate[f"{double_key}_minus_{baseline_key}"],
            aggregate[f"restored_minus_{baseline_key}"],
            aggregate[f"{restored_nlm_key}_minus_{baseline_key}"],
        )
        >= 0.005
    )
    return rows, aggregate


def main() -> None:
    args = _parser().parse_args()
    if args.train_limit < 1 or args.eval_limit < 1 or args.steps < 1:
        raise ValueError("train-limit, eval-limit and steps must be positive")
    if args.real_limit < 0 or args.real_limit > args.train_limit:
        raise ValueError("real-limit must be between zero and train-limit")
    if args.crop < 160 or args.crop % 20:
        raise ValueError("crop must be a multiple of 20 and at least 160")
    if args.nlm_h < 1:
        raise ValueError("nlm-h must be positive")
    if not 0 <= args.real_probability <= 1:
        raise ValueError("real-probability must be in [0, 1]")
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    train_records = [
        dict(record)
        for record in select_manifest_records(manifest, "train", limit=args.train_limit)
    ]
    eval_panel = [
        dict(record)
        for record in select_manifest_records(
            manifest,
            args.eval_split,
            limit=args.eval_offset + args.eval_limit,
        )
    ]
    eval_records = eval_panel[args.eval_offset :]
    configuration = {
        "manifest_path": str(manifest_path),
        "protocol_digest": manifest["protocol_digest"],
        "subset_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "subset_seed": EXPERIMENT_SUBSET_SEED,
        "architecture": args.architecture,
        "train_limit": args.train_limit,
        "real_limit": args.real_limit,
        "train_selection_digest": _selection_digest(train_records),
        "eval_split": args.eval_split,
        "eval_limit": args.eval_limit,
        "eval_offset": args.eval_offset,
        "eval_selection_digest": _selection_digest(eval_records),
        "train_filenames": [record["filename"] for record in train_records],
        "eval_filenames": [record["filename"] for record in eval_records],
        "steps": args.steps,
        "crop": args.crop,
        "real_probability": args.real_probability,
        "base": args.base,
        "depth": args.depth,
        "blocks": args.blocks,
        "nlm_h": args.nlm_h,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "checkpoint_in": str(args.checkpoint_in.resolve()) if args.checkpoint_in else None,
    }
    if args.dry_run:
        print(json.dumps(configuration, indent=2))
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    model = _model(args).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    started = time.perf_counter()
    if args.checkpoint_in:
        checkpoint = torch.load(args.checkpoint_in.resolve(), map_location="cpu", weights_only=True)
        expected = {key: configuration[key] for key in ("architecture", "base", "depth", "blocks")}
        if checkpoint.get("model_configuration") != expected:
            raise ValueError("checkpoint model configuration mismatch")
        model.load_state_dict(checkpoint["model"])
        history, prepare_seconds = [], 0.0
    else:
        history, prepare_seconds = _train(model, train_records, args, device)

    rows, aggregate = _evaluate(model, eval_records, device, args.nlm_h)
    output_path = args.output.resolve()
    if args.checkpoint_in:
        checkpoint_path = args.checkpoint_in.resolve()
    else:
        checkpoint_path = output_path.with_suffix(".pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "model_configuration": {
                    key: configuration[key] for key in ("architecture", "base", "depth", "blocks")
                },
                "training_configuration": configuration,
            },
            checkpoint_path,
        )
    payload = {
        "schema_version": 1,
        "experiment": "broad-post-layout-restoration-r6",
        "status": "completed",
        "scope": (
            "target-assisted fixed-layout pixel gate; no solver, test inputs, or submission score"
        ),
        "configuration": configuration,
        "model": {"parameters": parameter_count, "device": str(device)},
        "training": {"history": history, "prepare_seconds": prepare_seconds},
        "per_board": rows,
        "aggregate": aggregate,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "code_sha256": sha256_file(PROJECT_ROOT / "src" / "aiijc_puzzle" / "restoration_r6.py"),
        },
    }
    _write_json(payload, output_path)
    print(json.dumps({"aggregate": aggregate, "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
