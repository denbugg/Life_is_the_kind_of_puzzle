#!/usr/bin/env python3
"""Recover the historical full-board HBT matcher under the frozen protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.compliant_atlas_decoder import (
    audit_raw_permutation,
    population_position_scores,
    solve_buddies_with_position,
)
from aiijc_puzzle.hbt_recovery import (
    HISTORICAL_COMMIT,
    HISTORICAL_LEARNED_BLOB,
    HISTORICAL_TRAIN_BLOB,
    SideEmbeddingNet,
    dense_scores,
    direction_labels,
    embedding_hard_triplet_loss,
    exact_retrieval_counts,
    make_synthetic_panel,
    tiles_tensor,
    view_tiles,
)
from aiijc_puzzle.legacy_upgrade import directional_scores, layout_digest
from aiijc_puzzle.low_frequency_prior import FrozenLowFrequencyPrior
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    GRID_SIZE,
    IMAGE_SIZE,
    TILE_COUNT,
    assemble_tiles,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.restoration_r6 import nlm_color

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_ATLAS = PROJECT_ROOT / "artifacts" / "low-frequency-prior" / "train5600-v1.npz"
MODEL_VIEWS = ("raw", "bilateral")
MODEL_CONFIGURATION = {
    "channels": 64,
    "embedding_dim": 320,
    "side_band": 4,
    "tangent_bins": 10,
    "temperature": 0.07,
    "input_mode": "rgb_sobel",
    "edge_threshold": 0.12,
}
LOSS_CONFIGURATION = {
    "margin": 0.2,
    "cross_entropy_weight": 0.25,
    "embedding_l2_weight": 1e-4,
    "outside_weight": 0.2,
}
ATLAS_WEIGHT = 0.03
EDGE_BUDGET = 96
NLM_PASSES = 5
NLM_H = 10
GATE_FIELDS = (
    "raw_ssim",
    "nlm5_ssim",
    "adjacency",
    "right_adjacency",
    "down_adjacency",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inputs", type=Path, default=Path("data/raw/train/inputs"))
    parser.add_argument("--targets", type=Path, default=Path("data/raw/train/targets"))
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eval-offset", type=int, default=72)
    parser.add_argument("--eval-limit", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--views",
        nargs="+",
        choices=MODEL_VIEWS,
        default=list(MODEL_VIEWS),
        help="paired arms to train; the preregistered pilot uses both",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="load bound <view>.pt checkpoints instead of training",
    )
    return parser.parse_args()


def load_rgb(path: Path, expected_sha256: str | None = None) -> np.ndarray:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}")
    with Image.open(path) as image:
        image.load()
        if (
            image.format != "PNG"
            or image.mode != "RGB"
            or image.size
            != (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest_names(records: tuple[Mapping[str, Any], ...]) -> str:
    value = "\n".join(str(record["filename"]) for record in records).encode()
    return hashlib.sha256(value).hexdigest()


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def panel_seed(seed: int, name: str, epoch: int) -> int:
    value = f"{seed}\0hbt-organizer-panel-v1\0{name}\0{epoch}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def choose_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def code_hashes() -> dict[str, str]:
    paths = {
        "hbt_recovery": PROJECT_ROOT / "src" / "aiijc_puzzle" / "hbt_recovery.py",
        "restoration_r6": PROJECT_ROOT / "src" / "aiijc_puzzle" / "restoration_r6.py",
        "protocol": PROJECT_ROOT / "src" / "aiijc_puzzle" / "protocol.py",
        "candidate_supply": PROJECT_ROOT / "src" / "aiijc_puzzle" / "candidate_supply.py",
        "legacy_upgrade": PROJECT_ROOT / "src" / "aiijc_puzzle" / "legacy_upgrade.py",
        "compliant_atlas_decoder": (
            PROJECT_ROOT / "src" / "aiijc_puzzle" / "compliant_atlas_decoder.py"
        ),
        "runner": Path(__file__),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def checkpoint_contract(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    train_records: tuple[Mapping[str, Any], ...],
    *,
    view: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "architecture": "historical-side-embedding-net-pooled",
        "historical_commit": HISTORICAL_COMMIT,
        "historical_learned_blob": HISTORICAL_LEARNED_BLOB,
        "historical_train_blob": HISTORICAL_TRAIN_BLOB,
        "model_configuration": MODEL_CONFIGURATION,
        "loss_configuration": LOSS_CONFIGURATION,
        "view": view,
        "corruption": (
            "restoration_r6.distort_tiles defaults: contrast .70-1.30, brightness +/-30, "
            "noise sigma 40-55, separable blur3, per-tile JPEG quality 35-50"
        ),
        "replicas": args.epochs,
        "protocol_digest": compute_protocol_digest(manifest),
        "selector_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "selector_seed": EXPERIMENT_SUBSET_SEED,
        "train_limit": len(train_records),
        "train_selection_digest": digest_names(train_records),
        "train_filenames": [record["filename"] for record in train_records],
        "training_semantic_code_sha256": code_hashes(),
    }


def new_models(views: tuple[str, ...], device: torch.device) -> dict[str, SideEmbeddingNet]:
    first = SideEmbeddingNet(**MODEL_CONFIGURATION).to(device)
    initial_state = {name: value.detach().clone() for name, value in first.state_dict().items()}
    models = {views[0]: first}
    for view in views[1:]:
        model = SideEmbeddingNet(**MODEL_CONFIGURATION).to(device)
        model.load_state_dict(initial_state)
        models[view] = model
    return models


def train_models(
    models: dict[str, SideEmbeddingNet],
    records: tuple[Mapping[str, Any], ...],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    optimizers = {
        view: torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        for view, model in models.items()
    }
    history: list[dict[str, Any]] = []
    started = perf_counter()
    for epoch in range(args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(len(records))
        totals = {view: defaultdict(float) for view in models}
        for step, record_index in enumerate(order, start=1):
            record = records[int(record_index)]
            name = str(record["filename"])
            target = load_rgb(args.targets / name, str(record["target_sha256"]))
            seed = panel_seed(args.seed, name, epoch)
            panel = make_synthetic_panel(target, seed=seed)
            del target
            for view, model in models.items():
                model.train()
                tensor = tiles_tensor(view_tiles(panel.slot_tiles, view=view), device)
                optimizer = optimizers[view]
                optimizer.zero_grad(set_to_none=True)
                outputs = model(tensor)
                loss, metrics = embedding_hard_triplet_loss(
                    outputs,
                    panel.labels,
                    temperature=model.temperature,
                    **LOSS_CONFIGURATION,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                for key, value in metrics.items():
                    totals[view][key] += value
                del tensor, outputs, loss
            if step == 1 or step % 16 == 0 or step == len(records):
                status = {
                    view: {
                        "loss": totals[view]["loss"] / step,
                        "r1": totals[view]["recall_at_1"] / step,
                    }
                    for view in models
                }
                print(
                    json.dumps(
                        {
                            "event": "hbt_train",
                            "epoch": epoch + 1,
                            "step": step,
                            "boards": len(records),
                            "elapsed_seconds": perf_counter() - started,
                            "arms": status,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        for view in models:
            record = {key: value / len(records) for key, value in sorted(totals[view].items())}
            history.append({"epoch": epoch + 1, "view": view, **record})
    return history, perf_counter() - started


def save_checkpoints(
    models: dict[str, SideEmbeddingNet],
    output_dir: Path,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    train_records: tuple[Mapping[str, Any], ...],
    history: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for view, model in models.items():
        path = output_dir / f"{view}.pt"
        contract = checkpoint_contract(args, manifest, train_records, view=view)
        payload = {
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "contract": contract,
            "training_configuration": {
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
                "epochs": args.epochs,
                "device": str(next(model.parameters()).device),
            },
            "training_history": [row for row in history if row["view"] == view],
        }
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        result[view] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "contract": contract,
        }
    return result


def load_checkpoints(
    checkpoint_dir: Path,
    views: tuple[str, ...],
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    train_records: tuple[Mapping[str, Any], ...],
    device: torch.device,
) -> tuple[dict[str, SideEmbeddingNet], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    models: dict[str, SideEmbeddingNet] = {}
    metadata: dict[str, dict[str, Any]] = {}
    histories: list[dict[str, Any]] = []
    for view in views:
        path = checkpoint_dir / f"{view}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = checkpoint_contract(args, manifest, train_records, view=view)
        contract = payload.get("contract")
        if not isinstance(contract, dict):
            raise ValueError(f"checkpoint {path} has no contract")
        for field, expected_value in expected.items():
            if field == "training_semantic_code_sha256":
                continue
            if contract.get(field) != expected_value:
                raise ValueError(f"checkpoint {path} contract mismatch for {field}")
        model = SideEmbeddingNet(**MODEL_CONFIGURATION).to(device)
        model.load_state_dict(payload["state_dict"])
        models[view] = model
        metadata[view] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "contract": contract,
            "training_configuration": payload.get("training_configuration", {}),
        }
        histories.extend(payload.get("training_history", []))
    return models, metadata, histories


def repeated_nlm(image: np.ndarray) -> np.ndarray:
    result = image
    for _ in range(NLM_PASSES):
        result = nlm_color(result, h=NLM_H)
    return result


def variant_scores(
    tiles: np.ndarray,
    models: dict[str, SideEmbeddingNet],
    device: torch.device,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    variants = {
        "bilateral_atlas_baseline": directional_scores(tiles, views=("bilateral",))["bilateral"]
    }
    outside: dict[str, np.ndarray] = {}
    learned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for view, model in models.items():
        right, down, outside_logits = dense_scores(
            model, view_tiles(tiles, view=view), device=device
        )
        learned[view] = (right, down)
        variants[f"hbt_{view}_atlas"] = (right, down)
        outside[view] = outside_logits
    if set(MODEL_VIEWS).issubset(learned):
        raw, bilateral = learned["raw"], learned["bilateral"]
        pair_mean = (
            np.mean([raw[0], bilateral[0]], axis=0, dtype=np.float64).astype(np.float32),
            np.mean([raw[1], bilateral[1]], axis=0, dtype=np.float64).astype(np.float32),
        )
        variants["hbt_pair_mean_atlas"] = pair_mean
        baseline = variants["bilateral_atlas_baseline"]
        variants["hbt_pair_plus_classical_atlas"] = (
            np.mean([pair_mean[0], baseline[0]], axis=0, dtype=np.float64).astype(np.float32),
            np.mean([pair_mean[1], baseline[1]], axis=0, dtype=np.float64).astype(np.float32),
        )
    return variants, outside


def freeze_predictions(
    models: dict[str, SideEmbeddingNet],
    records: tuple[Mapping[str, Any], ...],
    atlas: FrozenLowFrequencyPrior,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    """Create every dirty-only score/layout/image before any calibration target access."""

    frozen: list[dict[str, Any]] = []
    started = perf_counter()
    for index, record in enumerate(records, start=1):
        name = str(record["filename"])
        dirty = load_rgb(args.inputs / name, str(record["input_sha256"]))
        tiles = split_tiles(dirty)
        scores, outside = variant_scores(tiles, models, device)
        position = population_position_scores(tiles, atlas.generic_tile_template)
        variants: dict[str, Any] = {}
        for variant, (right, down) in scores.items():
            solved = solve_buddies_with_position(
                right,
                down,
                position,
                position_weight=ATLAS_WEIGHT,
                max_edges=EDGE_BUDGET,
            )
            layout = solved.layout
            raw = assemble_tiles(tiles[layout])
            audit = audit_raw_permutation(dirty, raw, layout, restoration_applied_after_audit=True)
            if not audit.passed:
                raise RuntimeError(f"strict permutation audit failed for {name}/{variant}")
            restored = repeated_nlm(raw)
            variants[variant] = {
                "right": right,
                "down": down,
                "layout": layout,
                "raw": raw,
                "nlm5": restored,
                "inference": {
                    "layout_sha256": layout_digest(layout),
                    "raw_sha256": image_digest(raw),
                    "nlm5_sha256": image_digest(restored),
                    "unique_tiles": int(len(np.unique(layout))),
                    "permutation_audit": audit.as_dict(),
                    "solver": solved.solver,
                    "solver_seconds": solved.runtime_seconds,
                },
            }
        frozen.append(
            {
                "filename": name,
                "target_sha256": str(record["target_sha256"]),
                "dirty": dirty,
                "tiles": tiles,
                "outside": outside,
                "variants": variants,
            }
        )
        print(
            json.dumps(
                {
                    "event": "hbt_freeze",
                    "board": index,
                    "boards": len(records),
                    "filename": name,
                    "variants": list(variants),
                    "elapsed_seconds": perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return frozen, perf_counter() - started


def save_frozen_manifest(path: Path, frozen: list[dict[str, Any]]) -> str:
    payload = {
        "schema_version": 1,
        "prediction_boundary": "dirty-input-only; written before target access",
        "boards": [
            {
                "filename": item["filename"],
                "variants": {name: value["inference"] for name, value in item["variants"].items()},
            }
            for item in frozen
        ],
    }
    atomic_json(path, payload)
    return sha256_file(path)


def layout_metrics(layout: np.ndarray, recovered: Any) -> dict[str, float]:
    truth = recovered.dirty_at_position
    position_of_dirty = recovered.position_of_dirty
    predicted_positions = np.empty_like(layout)
    predicted_positions[layout] = np.arange(len(layout))
    shifts: dict[tuple[int, int], int] = {}
    for tile, predicted in enumerate(predicted_positions):
        true = int(position_of_dirty[tile])
        predicted_row, predicted_column = divmod(int(predicted), GRID_SIZE)
        true_row, true_column = divmod(true, GRID_SIZE)
        shift = (true_row - predicted_row, true_column - predicted_column)
        shifts[shift] = shifts.get(shift, 0) + 1
    grid = layout.reshape(GRID_SIZE, GRID_SIZE)
    left = position_of_dirty[grid[:, :-1]]
    right = position_of_dirty[grid[:, 1:]]
    top = position_of_dirty[grid[:-1]]
    bottom = position_of_dirty[grid[1:]]
    right_accuracy = np.mean((right - left == 1) & (right // GRID_SIZE == left // GRID_SIZE))
    down_accuracy = np.mean(bottom - top == GRID_SIZE)
    return {
        "direct_placement": float(np.mean(layout == truth)),
        "translation_aligned_placement": float(max(shifts.values()) / len(layout)),
        "right_adjacency": float(right_accuracy),
        "down_adjacency": float(down_accuracy),
        "adjacency": float(0.5 * (right_accuracy + down_accuracy)),
    }


def merge_retrieval(
    rows: list[list[dict[str, int | str]]],
) -> dict[str, dict[str, float | int]]:
    totals: defaultdict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    for group in rows:
        for record in group:
            key = (str(record["direction"]), int(record["k"]))
            totals[key][0] += int(record["edges"])
            totals[key][1] += int(record["hits"])
    result: dict[str, dict[str, float | int]] = {}
    for (direction, k), (edges, hits) in sorted(totals.items()):
        result[f"{direction}.r{k}"] = {
            "edges": edges,
            "hits": hits,
            "recall": hits / edges,
        }
    for k in (1, 5, 32):
        right = result[f"right.r{k}"]
        down = result[f"down.r{k}"]
        edges = int(right["edges"]) + int(down["edges"])
        hits = int(right["hits"]) + int(down["hits"])
        result[f"pooled.r{k}"] = {"edges": edges, "hits": hits, "recall": hits / edges}
    return result


def evaluate_frozen(
    frozen: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load calibration targets only after the complete inference roster is frozen."""

    retrieval_rows: defaultdict[str, list[list[dict[str, int | str]]]] = defaultdict(list)
    boards: list[dict[str, Any]] = []
    for item in frozen:
        name = item["filename"]
        target = load_rgb(args.targets / name, item["target_sha256"])
        recovered = recover_layout(item["tiles"], split_tiles(target))
        labels = direction_labels(recovered.position_of_dirty)
        board: dict[str, Any] = {"filename": name, "variants": {}}
        for variant, value in item["variants"].items():
            retrieval_rows[variant].append(
                exact_retrieval_counts(value["right"], value["down"], labels)
            )
            board["variants"][variant] = {
                "raw_ssim": contest_ssim(target, value["raw"]),
                "nlm5_ssim": contest_ssim(target, value["nlm5"]),
                **layout_metrics(value["layout"], recovered),
            }
        boards.append(board)
    retrieval = {variant: merge_retrieval(rows) for variant, rows in sorted(retrieval_rows.items())}
    summary: dict[str, Any] = {"boards": boards, "variants": {}}
    for variant in boards[0]["variants"]:
        fields = boards[0]["variants"][variant]
        summary["variants"][variant] = {
            field: float(np.mean([board["variants"][variant][field] for board in boards]))
            for field in fields
        }
    return retrieval, summary


def scale_gate(summary: dict[str, Any]) -> dict[str, Any]:
    baseline_name = "bilateral_atlas_baseline"
    baseline = summary["variants"][baseline_name]
    candidates: dict[str, Any] = {}
    for variant, metrics in summary["variants"].items():
        if variant == baseline_name:
            continue
        deltas = {field: metrics[field] - baseline[field] for field in GATE_FIELDS}
        conditions = {
            "raw_ssim_positive": deltas["raw_ssim"] > 0.0,
            "nlm5_ssim_positive": deltas["nlm5_ssim"] > 0.0,
            "adjacency_positive": deltas["adjacency"] > 0.0,
            "right_adjacency_nonnegative": deltas["right_adjacency"] >= 0.0,
            "down_adjacency_nonnegative": deltas["down_adjacency"] >= 0.0,
        }
        candidates[variant] = {
            "deltas": deltas,
            "quantitative_passed": all(conditions.values()),
            "conditions": conditions,
        }
    passing = [name for name, row in candidates.items() if row["quantitative_passed"]]
    winner = (
        max(passing, key=lambda name: candidates[name]["deltas"]["nlm5_ssim"]) if passing else None
    )
    return {
        "passed": bool(winner),
        "requires_manual_visual_coherence_check": bool(winner),
        "manual_visual_check_passed": None,
        "eligible_for_2048_scale": False,
        "winner": winner,
        "baseline": baseline_name,
        "candidates": candidates,
    }


def make_contact_sheet(
    frozen: list[dict[str, Any]], args: argparse.Namespace, winner: str, path: Path
) -> None:
    selected = frozen[: min(6, len(frozen))]
    cell = 240
    header = 28
    columns = ("baseline raw", "winner raw", "baseline NLM5", "winner NLM5", "target")
    canvas = Image.new("RGB", (cell * len(columns), (cell + header) * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(columns):
        draw.text((column * cell + 4, 4), title, fill="black")
    for row, item in enumerate(selected):
        target = load_rgb(args.targets / item["filename"], item["target_sha256"])
        baseline = item["variants"]["bilateral_atlas_baseline"]
        learned = item["variants"][winner]
        images = (baseline["raw"], learned["raw"], baseline["nlm5"], learned["nlm5"], target)
        y = row * (cell + header) + header
        for column, values in enumerate(images):
            image = Image.fromarray(values).resize((cell, cell), Image.Resampling.BILINEAR)
            canvas.paste(image, (column * cell, y))
        draw.text(
            (4, y + cell - 16), item["filename"], fill="white", stroke_width=2, stroke_fill="black"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = parse_args()
    if min(args.train_limit, args.epochs, args.eval_limit) <= 0:
        raise ValueError("train-limit, epochs and eval-limit must be positive")
    if args.eval_offset < 0:
        raise ValueError("eval-offset must be non-negative")
    views = tuple(dict.fromkeys(args.views))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest is invalid")
    train_records = select_manifest_records(manifest, "train", limit=args.train_limit)
    calibration_prefix = select_manifest_records(
        manifest, "calibration", limit=args.eval_offset + args.eval_limit
    )
    eval_records = tuple(calibration_prefix[args.eval_offset :])
    if set(record["filename"] for record in train_records) & set(
        record["filename"] for record in eval_records
    ):
        raise RuntimeError("train/calibration overlap")
    atlas = FrozenLowFrequencyPrior.load(args.atlas)
    if atlas.metadata.get("protocol_digest") != manifest["protocol_digest"]:
        raise ValueError("atlas/manifest protocol mismatch")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint_dir:
        models, checkpoints, history = load_checkpoints(
            args.checkpoint_dir,
            views,
            args,
            manifest,
            train_records,
            device,
        )
        training_seconds = 0.0
    else:
        models = new_models(views, device)
        history, training_seconds = train_models(models, train_records, args, device)
        checkpoints = save_checkpoints(models, output_dir, args, manifest, train_records, history)
    frozen, inference_seconds = freeze_predictions(models, eval_records, atlas, args, device)
    frozen_manifest_path = output_dir / "frozen-inference.json"
    frozen_manifest_sha256 = save_frozen_manifest(frozen_manifest_path, frozen)

    target_started = perf_counter()
    retrieval, downstream = evaluate_frozen(frozen, args)
    target_seconds = perf_counter() - target_started
    gate = scale_gate(downstream)
    contact_sheet = None
    if gate["winner"] is not None:
        contact_path = output_dir / "gate-contact-sheet.png"
        make_contact_sheet(frozen, args, str(gate["winner"]), contact_path)
        contact_sheet = {"path": str(contact_path), "sha256": sha256_file(contact_path)}

    report = {
        "experiment": "historical-hbt-recovery-current-protocol-v1",
        "verdict": (
            "quantitative-gate-passed-manual-review-required"
            if gate["passed"]
            else "pilot-gate-failed-do-not-scale"
        ),
        "leakage_boundary": {
            "training": "manifest-train clean targets -> synthetic corruption and labels",
            "prediction": "actual dirty calibration tiles only",
            "target_access": "after every score/layout/raw/NLM5 prediction was frozen",
            "frozen_manifest": str(frozen_manifest_path),
            "frozen_manifest_sha256": frozen_manifest_sha256,
            "holdout_opened": False,
            "test_opened": False,
        },
        "compliance": {
            "strict_bijection": True,
            "decoder": "buddies96 + train-only population atlas w=0.03",
            "input_fragments": TILE_COUNT,
            "constant_or_template_substitution": False,
            "restoration": "five sequential colored NLM h=10 passes after raw audit",
        },
        "historical_provenance": {
            "commit": HISTORICAL_COMMIT,
            "learned_source_blob": HISTORICAL_LEARNED_BLOB,
            "training_script_blob": HISTORICAL_TRAIN_BLOB,
            "port": "architecture and hard-triplet loss mathematically exact; protocol glue new",
            "historical_selected_configuration": {
                **MODEL_CONFIGURATION,
                **LOSS_CONFIGURATION,
                "optimizer": "AdamW(lr=3e-4, weight_decay=1e-4), grad_clip=1",
            },
            "historical_reference": {
                "train_sources": 2048,
                "epochs": 2,
                "raw_validation_r1": 0.1790081551298499,
                "tile-naf-denoised_validation_r1": 0.22384511260315776,
                "checkpoints_available_here": False,
            },
        },
        "r6_tile_view_decision": (
            "not run: available R6 model is a post-layout full-canvas restorer, not a frozen "
            "per-tile denoiser; using it before layout would be semantically invalid"
        ),
        "configuration": {
            "train_limit": args.train_limit,
            "epochs": args.epochs,
            "eval_offset": args.eval_offset,
            "eval_limit": args.eval_limit,
            "views": list(views),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "device": str(device),
            "model": MODEL_CONFIGURATION,
            "loss": LOSS_CONFIGURATION,
        },
        "selection": {
            "protocol_digest": manifest["protocol_digest"],
            "selector_namespace": EXPERIMENT_SUBSET_NAMESPACE,
            "selector_seed": EXPERIMENT_SUBSET_SEED,
            "train_filenames": [record["filename"] for record in train_records],
            "train_digest": digest_names(train_records),
            "calibration_filenames": [record["filename"] for record in eval_records],
            "calibration_digest": digest_names(eval_records),
            "source_disjoint": True,
        },
        "runtime_seconds": {
            "training": training_seconds,
            "dirty_only_inference_freeze": inference_seconds,
            "target_assisted_evaluation": target_seconds,
        },
        "checkpoints": checkpoints,
        "training_history": history,
        "local_exact_retrieval": retrieval,
        "downstream": downstream,
        "scale_gate": gate,
        "contact_sheet": contact_sheet,
        "runtime_code_sha256": code_hashes(),
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "gate": gate,
                "runtime_seconds": report["runtime_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
