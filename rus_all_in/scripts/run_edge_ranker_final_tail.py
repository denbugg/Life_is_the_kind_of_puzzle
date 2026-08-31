#!/usr/bin/env python3
"""Audit the frozen raw edge ranker under the compliant h20x1 final tail."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.edge_ranker import (
    attach_target_labels,
    build_inference_board,
    exact_edge_counts,
    score_board,
)
from aiijc_puzzle.edge_ranker_final_tail import (
    RAW_CHECKPOINT_EVAL_COUNT,
    RAW_CHECKPOINT_EVAL_OFFSET,
    RAW_CHECKPOINT_SHA256,
    dual_manual_gate,
    layout_metrics,
    load_verified_raw_checkpoint,
    names_digest,
    paired_bootstrap_ci,
)
from aiijc_puzzle.frozen_final_evaluator import _validate_method_configs
from aiijc_puzzle.legacy_upgrade import layout_digest, solve_buddies, validate_layout
from aiijc_puzzle.pixel_tails import apply_nlm_color
from aiijc_puzzle.postassembly_harmonizer import (
    apply_luminance_gains,
    apply_rgb_offsets,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from aiijc_puzzle.protocol import (
    EXPERIMENT_SUBSET_NAMESPACE,
    EXPERIMENT_SUBSET_SEED,
    IMAGE_SIZE,
    assemble_tiles,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
DEFAULT_INPUTS = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
DEFAULT_TARGETS = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "edge-ranker" / "scale-raw-train64-cal12" / "edge_ranker.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "edge-ranker" / "manual-tail-raw-checkpoint-cal24-offset204"
)
VIEWS = ("raw", "tile_z", "bilateral", "gray")
BOOTSTRAP_REPLICATES = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offset", type=int, default=204)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--pair-batch", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_verified_rgb(path: Path, expected_sha256: str) -> np.ndarray:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"manifest content hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def harmonized_tail(ordered_tiles: np.ndarray, rgb_config: Any, luma_config: Any) -> dict[str, Any]:
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered_tiles, rgb_config)
    rgb_tiles = apply_rgb_offsets(ordered_tiles, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(rgb_tiles, luma_config)
    harmonized = assemble_tiles(apply_luminance_gains(rgb_tiles, gains))
    final = apply_nlm_color(harmonized, h=20).image
    return {
        "harmonized": harmonized,
        "final": final,
        "diagnostics": {
            "rgb_seam_offsets": rgb_diagnostics,
            "bounded_luminance_gains": luma_diagnostics,
        },
    }


def merge_edge_counts(groups: list[list[dict[str, int | str]]]) -> dict[str, Any]:
    totals: defaultdict[tuple[str, str, int], list[int]] = defaultdict(lambda: [0, 0])
    for group in groups:
        for record in group:
            key = (str(record["scope"]), str(record["direction"]), int(record["k"]))
            totals[key][0] += int(record["edges"])
            totals[key][1] += int(record["hits"])
    output: dict[str, Any] = {}
    for (scope, direction, k), (edges, hits) in sorted(totals.items()):
        output[f"{scope}.{direction}.r{k}"] = {
            "edges": edges,
            "hits": hits,
            "recall": hits / edges if edges else 0.0,
        }
    for scope in ("all", "trusted_query"):
        for k in (1, 5):
            right = output[f"{scope}.right.r{k}"]
            down = output[f"{scope}.down.r{k}"]
            edges = right["edges"] + down["edges"]
            hits = right["hits"] + down["hits"]
            output[f"{scope}.pooled.r{k}"] = {
                "edges": edges,
                "hits": hits,
                "recall": hits / edges,
            }
    return output


def freeze_predictions(
    records: tuple[Any, ...],
    *,
    inputs: Path,
    model: Any,
    device: torch.device,
    pair_batch: int,
    rgb_config: Any,
    luma_config: Any,
) -> list[dict[str, Any]]:
    frozen: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        started = perf_counter()
        name = record["filename"]
        dirty = load_verified_rgb(inputs / name, record["input_sha256"])
        tiles = split_tiles(dirty)
        board = build_inference_board(tiles, filename=name, views=VIEWS, candidate_k=5)
        learned_right, learned_down, delta = score_board(
            model, board, device=device, pair_batch=pair_batch
        )
        baseline_solved = solve_buddies(board.right_baseline, board.down_baseline, max_edges=96)
        learned_solved = solve_buddies(learned_right, learned_down, max_edges=96)
        variants: dict[str, Any] = {}
        for variant, solved in (("baseline", baseline_solved), ("learned", learned_solved)):
            layout = validate_layout(solved.layout)
            ordered = np.ascontiguousarray(tiles[layout])
            raw = assemble_tiles(ordered)
            audit = audit_raw_permutation(
                dirty,
                raw,
                layout,
                restoration_applied_after_audit=True,
            )
            if not audit.passed:
                raise RuntimeError(
                    f"strict raw audit failed for {name}/{variant}: {audit.as_dict()}"
                )
            tail = harmonized_tail(ordered, rgb_config, luma_config)
            variants[variant] = {
                "layout": layout,
                "raw": raw,
                "harmonized": tail["harmonized"],
                "final": tail["final"],
                "audit": audit.as_dict(),
                "objective": float(solved.objective),
                "solver": solved.solver,
                "tail_diagnostics": tail["diagnostics"],
            }
            for field in ("layout", "raw", "harmonized", "final"):
                variants[variant][field].flags.writeable = False
        frozen.append(
            {
                "record": record,
                "dirty": dirty,
                "board": board,
                "baseline_right": board.right_baseline,
                "baseline_down": board.down_baseline,
                "learned_right": learned_right,
                "learned_down": learned_down,
                "variants": variants,
                "delta_diagnostics": delta,
                "runtime_seconds": perf_counter() - started,
            }
        )
        print(f"froze {index}/{len(records)} {name}", flush=True)
    return frozen


def commitment_payload(
    frozen: list[dict[str, Any]],
    *,
    checkpoint: Path,
    records: tuple[Any, ...],
    offset: int,
) -> dict[str, Any]:
    boards: list[dict[str, Any]] = []
    for item in frozen:
        variants: dict[str, Any] = {}
        for name, value in item["variants"].items():
            variants[name] = {
                "layout_sha256": layout_digest(value["layout"]),
                "raw_sha256": array_digest(value["raw"]),
                "harmonized_sha256": array_digest(value["harmonized"]),
                "final_sha256": array_digest(value["final"]),
                "audit": value["audit"],
                "objective": value["objective"],
                "solver": value["solver"],
            }
        boards.append(
            {
                "filename": item["record"]["filename"],
                "input_sha256": item["record"]["input_sha256"],
                "score_delta": item["delta_diagnostics"],
                "variants": variants,
                "runtime_seconds": item["runtime_seconds"],
            }
        )
    payload: dict[str, Any] = {
        "schema": "aiijc-edge-ranker-manual-tail-prediction-commitment-v1",
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_expected_sha256": RAW_CHECKPOINT_SHA256,
        "selector_namespace": EXPERIMENT_SUBSET_NAMESPACE,
        "selector_seed": EXPERIMENT_SUBSET_SEED,
        "split": "calibration",
        "offset": offset,
        "count": len(records),
        "filenames": [record["filename"] for record in records],
        "filenames_sha256": names_digest(records),
        "all_predictions_frozen_before_target_access": True,
        "target_paths_present_in_phase_one": False,
        "pipeline": {
            "layout_control": "bilateral scores -> buddies96",
            "layout_learned": "frozen raw edge-ranker residual -> buddies96",
            "strict_assembly": "all 576 original upright dirty tiles exactly once",
            "tail": "RGB seam offsets -> bounded luminance -> colored NLM h20x1",
        },
        "boards": boards,
    }
    payload["commitment_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def make_manual_sheet(
    frozen: list[dict[str, Any]],
    targets: dict[str, np.ndarray],
    output: Path,
) -> None:
    thumb = 144
    header = 34
    label_width = 150
    columns = (
        "dirty input",
        "clean target",
        "baseline raw",
        "ranker raw",
        "baseline final",
        "ranker final",
    )
    canvas = Image.new(
        "RGB",
        (label_width + thumb * len(columns), header * 2 + thumb * len(frozen)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(columns):
        draw.text((label_width + column * thumb + 4, 8), label, fill="black")
    for row, item in enumerate(frozen):
        y = header + row * thumb
        name = item["record"]["filename"]
        draw.text((4, y + 6), name, fill="black")
        images = (
            item["dirty"],
            targets[name],
            item["variants"]["baseline"]["raw"],
            item["variants"]["learned"]["raw"],
            item["variants"]["baseline"]["final"],
            item["variants"]["learned"]["final"],
        )
        for column, array in enumerate(images):
            image = Image.fromarray(array).resize((thumb, thumb), Image.Resampling.LANCZOS)
            canvas.paste(image, (label_width + column * thumb, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def evaluate_after_commitment(
    frozen: list[dict[str, Any]],
    *,
    targets_dir: Path,
    commitment_sha256: str,
    sheet_path: Path,
) -> dict[str, Any]:
    boards: list[dict[str, Any]] = []
    targets: dict[str, np.ndarray] = {}
    baseline_counts: list[list[dict[str, int | str]]] = []
    learned_counts: list[list[dict[str, int | str]]] = []
    for item in frozen:
        record = item["record"]
        target = load_verified_rgb(targets_dir / record["filename"], record["target_sha256"])
        targets[record["filename"]] = target
        clean_tiles = split_tiles(target)
        recovered = recover_layout(item["board"].tiles, clean_tiles)
        labelled = attach_target_labels(item["board"], clean_tiles)
        baseline_counts.append(
            exact_edge_counts(labelled, item["baseline_right"], item["baseline_down"])
        )
        learned_counts.append(
            exact_edge_counts(labelled, item["learned_right"], item["learned_down"])
        )
        board_record: dict[str, Any] = {"filename": record["filename"], "variants": {}}
        for name, variant in item["variants"].items():
            board_record["variants"][name] = {
                **layout_metrics(variant["layout"], recovered),
                "raw_ssim": contest_ssim(target, variant["raw"]),
                "harmonized_ssim": contest_ssim(target, variant["harmonized"]),
                "final_ssim": contest_ssim(target, variant["final"]),
            }
        boards.append(board_record)
    make_manual_sheet(frozen, targets, sheet_path)

    fields = tuple(boards[0]["variants"]["baseline"])
    means: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    for variant in ("baseline", "learned"):
        means[variant] = {
            field: float(np.mean([board["variants"][variant][field] for board in boards]))
            for field in fields
        }
    for index, field in enumerate(fields):
        differences = np.asarray(
            [
                board["variants"]["learned"][field] - board["variants"]["baseline"][field]
                for board in boards
            ]
        )
        deltas[field] = paired_bootstrap_ci(
            differences,
            seed=20260830 + 10 + index,
            replicates=BOOTSTRAP_REPLICATES,
        )
        deltas[field]["wins"] = int(np.sum(differences > 0))
        deltas[field]["ties"] = int(np.sum(differences == 0))
        deltas[field]["losses"] = int(np.sum(differences < 0))
    gate = dual_manual_gate(
        [
            board["variants"]["learned"]["adjacency"] - board["variants"]["baseline"]["adjacency"]
            for board in boards
        ],
        [
            board["variants"]["learned"]["final_ssim"] - board["variants"]["baseline"]["final_ssim"]
            for board in boards
        ],
        seed=20260830,
        replicates=BOOTSTRAP_REPLICATES,
    )
    return {
        "commitment_sha256": commitment_sha256,
        "target_access_started_only_after_commitment": True,
        "board_count": len(boards),
        "means": means,
        "paired_deltas": deltas,
        "primary_dual_gate": gate,
        "local_edge_metrics": {
            "baseline_bilateral": merge_edge_counts(baseline_counts),
            "learned_raw_checkpoint": merge_edge_counts(learned_counts),
        },
        "boards": boards,
        "manual_sheet": str(sheet_path),
    }


def main() -> None:
    args = parse_args()
    if args.offset < 0 or args.count <= 0 or args.pair_batch <= 0:
        raise ValueError("offset must be non-negative and count/pair-batch positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ranked = select_manifest_records(
        manifest,
        "calibration",
        limit=args.offset + args.count,
    )
    records = tuple(ranked[args.offset :])
    prior_edge = select_manifest_records(
        manifest,
        "calibration",
        limit=RAW_CHECKPOINT_EVAL_OFFSET + RAW_CHECKPOINT_EVAL_COUNT,
    )[RAW_CHECKPOINT_EVAL_OFFSET:]
    if set(record["filename"] for record in records) & set(
        record["filename"] for record in prior_edge
    ):
        raise RuntimeError("fresh panel overlaps the checkpoint's old evaluation panel")
    device = choose_device(args.device)
    model, payload = load_verified_raw_checkpoint(
        args.checkpoint,
        manifest=manifest,
        project_root=PROJECT_ROOT,
        device=device,
    )
    rgb_config, luma_config, method_hashes = _validate_method_configs()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    frozen = freeze_predictions(
        records,
        inputs=args.inputs,
        model=model,
        device=device,
        pair_batch=args.pair_batch,
        rgb_config=rgb_config,
        luma_config=luma_config,
    )
    freeze_seconds = perf_counter() - started
    commitment = commitment_payload(
        frozen,
        checkpoint=args.checkpoint,
        records=records,
        offset=args.offset,
    )
    commitment_path = output_dir / "prediction-commitment.json"
    atomic_json(commitment_path, commitment)
    on_disk = json.loads(commitment_path.read_text(encoding="utf-8"))
    if on_disk.get("commitment_sha256") != commitment["commitment_sha256"]:
        raise RuntimeError("prediction commitment readback failed")

    target_started = perf_counter()
    evaluation = evaluate_after_commitment(
        frozen,
        targets_dir=args.targets,
        commitment_sha256=commitment["commitment_sha256"],
        sheet_path=output_dir / "manual-layout-sheet.png",
    )
    target_seconds = perf_counter() - target_started
    gate_passed = bool(evaluation["primary_dual_gate"]["passed"])
    report = {
        "schema": "aiijc-edge-ranker-manual-tail-audit-v1",
        "experiment": "frozen-raw-edge-ranker-vs-bilateral-under-rgb-luma-nlm-h20x1",
        "verdict": (
            "dual-gate-passed-requires-fresh-confirmation"
            if gate_passed
            else "dual-gate-failed-do-not-integrate"
        ),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "expected_sha256": RAW_CHECKPOINT_SHA256,
            "training_configuration": payload.get("training_configuration"),
            "contract": payload.get("contract"),
            "retrained_this_run": False,
        },
        "selection": {
            "split": "calibration",
            "offset": args.offset,
            "count": args.count,
            "filenames": [record["filename"] for record in records],
            "filenames_sha256": names_digest(records),
            "overlap_with_checkpoint_train": 0,
            "overlap_with_checkpoint_eval_52_64": 0,
            "shared_selector_predecessor": "learned matcher panels end at offset 204",
            "freshness_scope": (
                "new to this checkpoint and the contiguous legal learned-matcher sequence; "
                "quarantined historical calibration700 sweeps are excluded from selection claims"
            ),
        },
        "leakage_boundary": {
            "phase_one": "dirty input only; scores, layouts, raw audits and complete tails frozen",
            "commitment_path": str(commitment_path),
            "all_predictions_frozen_before_target_access": True,
            "phase_two": "paired target only for metrics and manual comparison sheet",
            "holdout_opened": False,
            "test_opened": False,
        },
        "compliance": {
            "baseline": "no-atlas bilateral buddies96",
            "learned": "raw checkpoint scores with the same buddies96 decoder",
            "input_tiles": 576,
            "required_unique_tiles": 576,
            "raw_assembly": "original upright input tile pixels exactly once",
            "tail": "frozen RGB offsets -> bounded luma -> colored NLM h20x1",
            "method_config_sha256": method_hashes,
            "constant_template_or_cross_board_substitution": False,
        },
        "runtime_seconds": {
            "prediction_freeze": freeze_seconds,
            "target_assisted_evaluation_and_sheet": target_seconds,
        },
        "evaluation": evaluation,
    }
    report_path = output_dir / "report.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "verdict": report["verdict"],
                "means": evaluation["means"],
                "primary_dual_gate": evaluation["primary_dual_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
