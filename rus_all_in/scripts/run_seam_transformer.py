#!/usr/bin/env python3
"""Precompute, train, and gate the deep ordered-seam Transformer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from aiijc_puzzle.candidate_supply import DEFAULT_VIEWS, split_tiles
from aiijc_puzzle.legacy_upgrade import layout_digest, solve_buddies
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    assemble_tiles,
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
)
from aiijc_puzzle.seam_transformer import (
    OrderedSeamTransformer,
    SeamCandidateBoard,
    SeamCandidateRow,
    attach_exact_training_labels,
    augment_ordered_pairs,
    build_inference_board,
    listwise_hard_negative_loss,
    rerank_score_matrices,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
INPUTS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
TARGETS_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "seam-transformer" / "pilot.json"
CACHE_SCHEMA = "aiijc-ordered-seam-transformer-cache-v1"
CHECKPOINT_SCHEMA = "aiijc-ordered-seam-transformer-checkpoint-v1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--train-limit", type=int, default=16)
    result.add_argument("--eval-limit", type=int, default=8)
    result.add_argument("--eval-offset", type=int, default=96)
    result.add_argument("--candidate-k", type=int, default=4)
    result.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    result.add_argument("--epochs", type=int, default=3)
    result.add_argument("--rows-per-board", type=int, default=128)
    result.add_argument("--batch-rows", type=int, default=4)
    result.add_argument("--inference-batch-rows", type=int, default=12)
    result.add_argument("--learning-rate", type=float, default=2e-4)
    result.add_argument("--weight-decay", type=float, default=2e-3)
    result.add_argument("--dim", type=int, default=256)
    result.add_argument("--heads", type=int, default=8)
    result.add_argument("--layers", type=int, default=10)
    result.add_argument("--mlp-ratio", type=int, default=4)
    result.add_argument("--dropout", type=float, default=0.1)
    result.add_argument("--seed", type=int, default=20260830)
    result.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    result.add_argument("--cache", type=Path)
    result.add_argument("--checkpoint-in", type=Path)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--disable-augmentation", action="store_true")
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if compute_protocol_digest(manifest) != manifest.get("protocol_digest"):
        raise ValueError("validation manifest digest mismatch")
    return manifest


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (480, 480, 3):
        raise ValueError(f"failed to load a 480x480 RGB image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def selection_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(record["filename"]) for record in records).encode()
    ).hexdigest()


def code_hashes() -> dict[str, str]:
    paths = {
        "seam_transformer.py": PROJECT_ROOT / "src" / "aiijc_puzzle" / "seam_transformer.py",
        "candidate_supply.py": PROJECT_ROOT / "src" / "aiijc_puzzle" / "candidate_supply.py",
        "legacy_upgrade.py": PROJECT_ROOT / "src" / "aiijc_puzzle" / "legacy_upgrade.py",
        "pixel_tails.py": PROJECT_ROOT / "src" / "aiijc_puzzle" / "pixel_tails.py",
        "protocol.py": PROJECT_ROOT / "src" / "aiijc_puzzle" / "protocol.py",
        "run_seam_transformer.py": Path(__file__).resolve(),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def record_panels(
    manifest: dict[str, Any], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [
        dict(record)
        for record in select_manifest_records(manifest, "train", limit=args.train_limit)
    ]
    panel = [
        dict(record)
        for record in select_manifest_records(
            manifest, "calibration", limit=args.eval_offset + args.eval_limit
        )
    ]
    return train, panel[args.eval_offset :]


def prepare_train_board(
    record: dict[str, Any], *, views: list[str], candidate_k: int
) -> SeamCandidateBoard:
    filename = str(record["filename"])
    input_path = INPUTS_DIR / filename
    target_path = TARGETS_DIR / filename
    if sha256_file(input_path) != record["input_sha256"]:
        raise ValueError(f"input hash mismatch: {filename}")
    dirty = split_tiles(load_rgb(input_path))
    inference_board = build_inference_board(
        dirty, filename=filename, views=views, candidate_k=candidate_k
    )
    if sha256_file(target_path) != record["target_sha256"]:
        raise ValueError(f"target hash mismatch: {filename}")
    clean = split_tiles(load_rgb(target_path))
    return attach_exact_training_labels(inference_board, clean)


def prepare_eval_board(
    record: dict[str, Any], *, views: list[str], candidate_k: int
) -> SeamCandidateBoard:
    filename = str(record["filename"])
    input_path = INPUTS_DIR / filename
    if sha256_file(input_path) != record["input_sha256"]:
        raise ValueError(f"input hash mismatch: {filename}")
    # The evaluation target path is intentionally never constructed here.
    return build_inference_board(
        split_tiles(load_rgb(input_path)),
        filename=filename,
        views=views,
        candidate_k=candidate_k,
    )


def write_pickle_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_cache(
    path: Path,
    manifest: dict[str, Any],
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    train_boards = []
    for index, record in enumerate(train_records, start=1):
        board_started = time.perf_counter()
        train_boards.append(
            prepare_train_board(record, views=args.views, candidate_k=args.candidate_k)
        )
        eligible = sum(row.trusted and row.exact_choice >= 0 for row in train_boards[-1].rows)
        print(
            f"precompute train {index:03d}/{len(train_records):03d} "
            f"{record['filename']} eligible={eligible} {time.perf_counter() - board_started:.2f}s",
            flush=True,
        )
    eval_boards = []
    for index, record in enumerate(eval_records, start=1):
        board_started = time.perf_counter()
        eval_boards.append(
            prepare_eval_board(record, views=args.views, candidate_k=args.candidate_k)
        )
        print(
            f"precompute eval-dirty-only {index:03d}/{len(eval_records):03d} "
            f"{record['filename']} {time.perf_counter() - board_started:.2f}s",
            flush=True,
        )
    metadata = {
        "schema": CACHE_SCHEMA,
        "protocol_digest": manifest["protocol_digest"],
        "train_selection_digest": selection_digest(train_records),
        "eval_selection_digest": selection_digest(eval_records),
        "train_filenames": [record["filename"] for record in train_records],
        "eval_filenames": [record["filename"] for record in eval_records],
        "eval_split": "calibration",
        "eval_offset": args.eval_offset,
        "views": args.views,
        "candidate_k": args.candidate_k,
        "eval_cache_target_access": False,
        "code_hashes": code_hashes(),
        "runtime_seconds": time.perf_counter() - started,
    }
    payload = {"metadata": metadata, "train_boards": train_boards, "eval_boards": eval_boards}
    write_pickle_atomic(path, payload)
    return payload


def validate_cache(
    payload: dict[str, Any],
    manifest: dict[str, Any],
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    expected = {
        "schema": CACHE_SCHEMA,
        "protocol_digest": manifest["protocol_digest"],
        "train_selection_digest": selection_digest(train_records),
        "eval_selection_digest": selection_digest(eval_records),
        "train_filenames": [record["filename"] for record in train_records],
        "eval_filenames": [record["filename"] for record in eval_records],
        "eval_split": "calibration",
        "eval_offset": args.eval_offset,
        "views": args.views,
        "candidate_k": args.candidate_k,
        "eval_cache_target_access": False,
        "code_hashes": code_hashes(),
    }
    metadata = payload.get("metadata", {})
    failed = [name for name, value in expected.items() if metadata.get(name) != value]
    if failed:
        raise ValueError(f"precompute cache contract mismatch: {failed}")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return torch.device(requested)


def eligible_epoch_rows(
    boards: list[SeamCandidateBoard], *, rows_per_board: int, seed: int
) -> list[tuple[SeamCandidateBoard, SeamCandidateRow]]:
    rng = np.random.default_rng(seed)
    refs: list[tuple[SeamCandidateBoard, SeamCandidateRow]] = []
    for board in boards:
        eligible = [row for row in board.rows if row.trusted and row.exact_choice >= 0]
        if len(eligible) > rows_per_board:
            selected = rng.choice(len(eligible), rows_per_board, replace=False)
            eligible = [eligible[int(index)] for index in selected]
        refs.extend((board, row) for row in eligible)
    rng.shuffle(refs)
    return refs


def collate_rows(
    refs: list[tuple[SeamCandidateBoard, SeamCandidateRow]],
    *,
    rng: np.random.Generator | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    anchors = np.stack([board.tiles[row.anchor] for board, row in refs])
    candidates = np.concatenate([board.tiles[row.candidates] for board, row in refs])
    features = np.concatenate([row.features for _, row in refs])
    row_ids = np.repeat(np.arange(len(refs)), [len(row.candidates) for _, row in refs])
    directions = np.repeat(
        np.asarray([row.direction for _, row in refs]),
        [len(row.candidates) for _, row in refs],
    )
    offsets = np.cumsum([0, *[len(row.candidates) for _, row in refs]])
    exact = np.asarray([offsets[index] + row.exact_choice for index, (_, row) in enumerate(refs)])
    if rng is not None:
        anchors, candidates = augment_ordered_pairs(anchors, candidates, row_ids, rng=rng)
    pair_anchors = anchors[row_ids]
    return {
        "anchors": torch.from_numpy(pair_anchors).permute(0, 3, 1, 2).to(device),
        "candidates": torch.from_numpy(candidates).permute(0, 3, 1, 2).to(device),
        "features": torch.from_numpy(features).to(device),
        "directions": torch.from_numpy(directions).to(device),
        "row_ids": torch.from_numpy(row_ids).to(device),
        "exact": torch.from_numpy(exact).to(device),
    }


def train_model(
    model: OrderedSeamTransformer,
    boards: list[SeamCandidateBoard],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    for epoch in range(args.epochs):
        model.train()
        refs = eligible_epoch_rows(
            boards, rows_per_board=args.rows_per_board, seed=args.seed + epoch
        )
        rng = (
            None if args.disable_augmentation else np.random.default_rng(args.seed + 10_000 + epoch)
        )
        losses = []
        correct = 0
        total = 0
        started = time.perf_counter()
        for start in range(0, len(refs), args.batch_rows):
            batch_refs = refs[start : start + args.batch_rows]
            batch = collate_rows(batch_refs, rng=rng, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["anchors"], batch["candidates"], batch["features"], batch["directions"]
            )
            loss = listwise_hard_negative_loss(logits, batch["row_ids"], batch["exact"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for local_row, exact in enumerate(batch["exact"]):
                indices = torch.nonzero(batch["row_ids"] == local_row, as_tuple=False).squeeze(1)
                correct += int(indices[torch.argmax(logits[indices])] == exact)
                total += 1
            if (start // args.batch_rows + 1) % 50 == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} step {start // args.batch_rows + 1}/"
                    f"{(len(refs) + args.batch_rows - 1) // args.batch_rows} "
                    f"loss={np.mean(losses[-50:]):.4f}",
                    flush=True,
                )
        scheduler.step()
        epoch_result = {
            "epoch": epoch + 1,
            "rows": len(refs),
            "mean_loss": float(np.mean(losses)),
            "augmented_train_exact": correct / max(total, 1),
            "learning_rate_after_epoch": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)
    return history


def predict_board_rows(
    model: OrderedSeamTransformer,
    board: SeamCandidateBoard,
    *,
    device: torch.device,
    batch_rows: int,
) -> list[np.ndarray]:
    model.eval()
    result: list[np.ndarray] = []
    refs = [(board, row) for row in board.rows]
    with torch.inference_mode():
        for start in range(0, len(refs), batch_rows):
            batch_refs = refs[start : start + batch_rows]
            batch = collate_rows(batch_refs, rng=None, device=device)
            logits = (
                model(batch["anchors"], batch["candidates"], batch["features"], batch["directions"])
                .cpu()
                .numpy()
            )
            offsets = np.cumsum([0, *[len(row.candidates) for _, row in batch_refs]])
            result.extend(
                logits[offsets[index] : offsets[index + 1]] for index in range(len(batch_refs))
            )
    return result


def prediction_hash(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def local_exact_metrics(
    labelled: SeamCandidateBoard,
    learned_choice_by_key: dict[tuple[int, int], int],
) -> dict[str, Any]:
    def summarize(rows: list[SeamCandidateRow]) -> dict[str, Any]:
        pool = sum(row.exact_choice >= 0 for row in rows)
        baseline = sum(row.exact_choice == row.baseline_choice for row in rows)
        learned = sum(
            row.exact_choice >= 0
            and learned_choice_by_key[(row.direction, row.anchor)] == row.exact_choice
            for row in rows
        )
        return {
            "rows": len(rows),
            "candidate_recall": pool / len(rows),
            "classical_exact_r1": baseline / len(rows),
            "transformer_exact_r1": learned / len(rows),
            "delta_exact_r1": (learned - baseline) / len(rows),
            "conditional_classical_exact_r1": baseline / max(pool, 1),
            "conditional_transformer_exact_r1": learned / max(pool, 1),
        }

    return {
        "all": summarize(list(labelled.rows)),
        "right": summarize([row for row in labelled.rows if row.direction == 0]),
        "down": summarize([row for row in labelled.rows if row.direction == 1]),
    }


def evaluate(
    model: OrderedSeamTransformer,
    eval_boards: list[SeamCandidateBoard],
    eval_records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    rows = []
    for index, (board, record) in enumerate(zip(eval_boards, eval_records, strict=True), start=1):
        started = time.perf_counter()
        row_logits = predict_board_rows(
            model, board, device=device, batch_rows=args.inference_batch_rows
        )
        learned_right, learned_down = rerank_score_matrices(board, row_logits)
        baseline_layout = solve_buddies(board.right_scores, board.down_scores, max_edges=96).layout
        learned_layout = solve_buddies(learned_right, learned_down, max_edges=96).layout
        baseline_raw = assemble_tiles(board.tiles[baseline_layout])
        learned_raw = assemble_tiles(board.tiles[learned_layout])
        baseline_nlm = apply_nlm_color(baseline_raw, h=10).image
        learned_nlm = apply_nlm_color(learned_raw, h=10).image
        # Freeze all deployable artifacts before target validation/decode.
        frozen = {
            "baseline_layout_sha256": layout_digest(baseline_layout),
            "transformer_layout_sha256": layout_digest(learned_layout),
            "baseline_raw_sha256": prediction_hash(baseline_raw),
            "transformer_raw_sha256": prediction_hash(learned_raw),
            "baseline_nlm_h10_sha256": prediction_hash(baseline_nlm),
            "transformer_nlm_h10_sha256": prediction_hash(learned_nlm),
        }
        learned_choices = {
            (candidate_row.direction, candidate_row.anchor): int(np.argmax(logits))
            for candidate_row, logits in zip(board.rows, row_logits, strict=True)
        }

        target_path = TARGETS_DIR / str(record["filename"])
        if sha256_file(target_path) != record["target_sha256"]:
            raise ValueError(f"evaluation target hash mismatch: {record['filename']}")
        target = load_rgb(target_path)
        labelled = attach_exact_training_labels(board, split_tiles(target))
        local = local_exact_metrics(labelled, learned_choices)
        board_result = {
            "filename": board.filename,
            "predictions_frozen_before_target_decode": True,
            "frozen_artifacts": frozen,
            "local_exact": local,
            "ssim": {
                "classical_raw": contest_ssim(target, baseline_raw),
                "transformer_raw": contest_ssim(target, learned_raw),
                "classical_nlm_h10": contest_ssim(target, baseline_nlm),
                "transformer_nlm_h10": contest_ssim(target, learned_nlm),
            },
            "runtime_seconds": time.perf_counter() - started,
        }
        rows.append(board_result)
        print(
            json.dumps(
                {
                    "eval": index,
                    "total": len(eval_boards),
                    "filename": board.filename,
                    "exact_delta": local["all"]["delta_exact_r1"],
                    "ssim_delta": board_result["ssim"]["transformer_nlm_h10"]
                    - board_result["ssim"]["classical_nlm_h10"],
                }
            ),
            flush=True,
        )
    aggregate: dict[str, Any] = {}
    for key in rows[0]["ssim"]:
        values = np.asarray([row["ssim"][key] for row in rows])
        aggregate[key] = {"mean": float(values.mean()), "std": float(values.std())}
    local_baseline = np.asarray([row["local_exact"]["all"]["classical_exact_r1"] for row in rows])
    local_learned = np.asarray([row["local_exact"]["all"]["transformer_exact_r1"] for row in rows])
    ssim_baseline = np.asarray([row["ssim"]["classical_nlm_h10"] for row in rows])
    ssim_learned = np.asarray([row["ssim"]["transformer_nlm_h10"] for row in rows])
    gate = {
        "requirements": {
            "mean_exact_r1_strictly_better": True,
            "mean_end_to_end_nlm_h10_ssim_strictly_better": True,
        },
        "mean_exact_r1_delta": float(np.mean(local_learned - local_baseline)),
        "mean_end_to_end_nlm_h10_ssim_delta": float(np.mean(ssim_learned - ssim_baseline)),
        "exact_board_wins": int(np.sum(local_learned > local_baseline)),
        "ssim_board_wins": int(np.sum(ssim_learned > ssim_baseline)),
    }
    gate["passed"] = bool(
        gate["mean_exact_r1_delta"] > 0.0 and gate["mean_end_to_end_nlm_h10_ssim_delta"] > 0.0
    )
    return {"aggregate": aggregate, "pilot_gate": gate, "per_board": rows}


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


def main() -> None:
    args = parser().parse_args()
    for name in (
        "train_limit",
        "eval_limit",
        "candidate_k",
        "epochs",
        "rows_per_board",
        "batch_rows",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.eval_offset < 96:
        raise ValueError("this track is locked to a fresh calibration offset of at least 96")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = load_manifest(args.manifest.resolve())
    train_records, eval_records = record_panels(manifest, args)
    output_path = args.output.resolve()
    cache_path = (
        args.cache.resolve()
        if args.cache
        else output_path.with_name(
            f"precompute-train{args.train_limit}-eval{args.eval_offset}-{args.eval_limit}.pkl"
        )
    )
    if cache_path.exists():
        with cache_path.open("rb") as stream:
            cache = pickle.load(stream)  # noqa: S301 - locally generated, hash-bound cache
        validate_cache(cache, manifest, train_records, eval_records, args)
    else:
        cache = prepare_cache(cache_path, manifest, train_records, eval_records, args)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "cache": str(cache_path),
                    "sha256": sha256_file(cache_path),
                    "metadata": cache["metadata"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    device = resolve_device(args.device)
    architecture = {
        "feature_dim": len(args.views) * 3,
        "dim": args.dim,
        "heads": args.heads,
        "layers": args.layers,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "patch_tokens": "4x4 patches over canonical 20x40 join",
        "seam_tokens": "20 row tokens over two pixels from each seam side",
    }
    model = OrderedSeamTransformer(
        feature_dim=architecture["feature_dim"],
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} parameters={parameter_count:,}", flush=True)
    configuration = {
        "protocol_digest": manifest["protocol_digest"],
        "train_limit": args.train_limit,
        "eval_split": "calibration",
        "eval_offset": args.eval_offset,
        "eval_limit": args.eval_limit,
        "subset_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "subset_seed": EXPERIMENT_SUBSET_SEED,
        "train_selection_digest": selection_digest(train_records),
        "eval_selection_digest": selection_digest(eval_records),
        "views": args.views,
        "candidate_k": args.candidate_k,
        "epochs": args.epochs,
        "rows_per_board": args.rows_per_board,
        "batch_rows": args.batch_rows,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "augmentation": {
            "enabled": not args.disable_augmentation,
            "brightness": "uniform[-18,18] per tile",
            "contrast": "uniform[0.75,1.25] per tile",
            "channel_gain": "uniform[0.88,1.12] per tile/channel",
            "gaussian_noise": "p=.75 sigma uniform[2,14]",
            "gaussian_blur": "p=.35 sigma uniform[.25,1.15]",
            "jpeg": "p=.30 quality integer[35,90]",
            "geometry": "shared vertical flip per ordered query",
        },
    }
    checkpoint_path = output_path.with_suffix(".pt")
    started = time.perf_counter()
    if args.checkpoint_in:
        checkpoint = torch.load(args.checkpoint_in.resolve(), map_location="cpu", weights_only=True)
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint schema mismatch")
        if checkpoint.get("architecture") != architecture:
            raise ValueError("checkpoint architecture mismatch")
        model.load_state_dict(checkpoint["model"])
        history = []
        checkpoint_path = args.checkpoint_in.resolve()
    else:
        history = train_model(model, cache["train_boards"], args, device)
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        atomic_torch_save(
            checkpoint_path,
            {
                "schema": CHECKPOINT_SCHEMA,
                "model": state,
                "architecture": architecture,
                "configuration": configuration,
                "code_hashes": code_hashes(),
                "cache_sha256": sha256_file(cache_path),
            },
        )
    evaluation = evaluate(
        model,
        cache["eval_boards"],
        eval_records,
        args=args,
        device=device,
    )
    report = {
        "schema": "aiijc-ordered-seam-transformer-pilot-v1",
        "experiment": "deep ordered-pair seam Transformer reranker",
        "status": "promote" if evaluation["pilot_gate"]["passed"] else "reject_as_tested",
        "architecture": {**architecture, "parameter_count": parameter_count},
        "configuration": configuration,
        "training": {"history": history},
        "evaluation": evaluation,
        "compliance_audit": {
            "inference_inputs": [
                "dirty shuffled RGB tiles",
                "dirty-only raw/tile_z/bilateral/gray candidates and costs",
                "direction token",
            ],
            "training_only_inputs": ["manifest-train clean targets for recovered exact labels"],
            "evaluation_only_inputs": ["calibration targets decoded after predictions freeze"],
            "holdout_access": False,
            "test_access": False,
            "eval_cache_target_access": False,
            "candidate_order_equivariance": "pair scores independent; covered by unit test",
            "render_contract": "strict tile permutation -> unchanged assembly -> RGB NLM h10",
            "forbidden_operations": (
                "no template, substitution, source lookup, warp, or target feature"
            ),
        },
        "runtime": {
            "seconds_excluding_precompute": time.perf_counter() - started,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "artifacts": {
            "cache": str(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "code_hashes": code_hashes(),
        },
    }
    write_json_atomic(output_path, report)
    print(json.dumps({"report": str(output_path), **evaluation["pilot_gate"]}), flush=True)


if __name__ == "__main__":
    main()
