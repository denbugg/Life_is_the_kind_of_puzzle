#!/usr/bin/env python3
"""Evaluate the preregistered k16 edge ranker under the compliant h20x1 tail."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.edge_ranker import (
    PairwiseEdgeRanker,
    attach_target_labels,
    build_inference_board,
    exact_edge_counts,
    score_board,
)
from aiijc_puzzle.edge_ranker_final_tail import (
    layout_metrics,
    names_digest,
    paired_bootstrap_ci,
)
from aiijc_puzzle.edge_ranker_scale_gate import candidate_coverage, scale_promotion_gate
from aiijc_puzzle.frozen_final_evaluator import _validate_method_configs, array_digest, atomic_json
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
    compute_protocol_digest,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
INPUTS = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
TARGETS = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
RUN_ROOT = PROJECT_ROOT / "outputs" / "edge-ranker" / "scale-raw-k16-train256-cal24-offset228"
CHECKPOINT = RUN_ROOT / "edge_ranker.pt"
TRAINING_REPORT = RUN_ROOT / "report.json"
PREREGISTRATION = PROJECT_ROOT / "configs" / "edge_ranker_k16_scale_preregistered_v1.json"
PREREGISTRATION_SHA256 = "22d81c542b5cd598fde6cdd6fadb7847ea974ef68c7f5774e336e9fc5b5ab422"
VIEWS = ("raw", "tile_z", "bilateral", "gray")
BOOTSTRAP_REPLICATES = 20_000
EXPECTED_CONTRACT = {
    "architecture": "joint-seam-context-cnn-v1",
    "views": list(VIEWS),
    "candidate_k": 16,
    "view_mode": "raw",
    "feature_dim": 12,
    "width": 32,
    "hidden": 64,
    "label_policy": "exact recovered neighbour; trusted-query training only",
    "teacher_policy": "trusted candidate clean symmetric extrapolation listwise CE",
    "teacher_weight": 0.15,
    "selector_namespace": EXPERIMENT_SUBSET_NAMESPACE,
    "selector_seed": EXPERIMENT_SUBSET_SEED,
    "train_limit": 256,
}
EXPECTED_TRAINING = {
    "epochs": 4,
    "rows_per_board": 128,
    "batch_rows": 24,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "teacher_weight": 0.15,
    "seed": 20260830,
    "device": "mps",
}
PANELS = {
    "primary": (228, 24, "d36f91a3f83718a28b295700f7f0bb7e1a8374f1e820f774e51d132ee793b103"),
    "confirmation": (
        252,
        24,
        "1bd59c6db73fa4af59e4304949c87512a3a4bb700845ad2bbe0ef677d48a27f1",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(PANELS), default="primary")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--inputs", type=Path, default=INPUTS)
    parser.add_argument("--targets", type=Path, default=TARGETS)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--training-report", type=Path, default=TRAINING_REPORT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pair-batch", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def load_verified_rgb(path: Path, expected_sha256: str) -> np.ndarray:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"manifest content hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected strict RGB 480x480 PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def load_checkpoint(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    device: torch.device,
) -> tuple[PairwiseEdgeRanker, Mapping[str, Any]]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("preregistration hash drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("checkpoint has no mapping contract")
    for key, expected in EXPECTED_CONTRACT.items():
        if contract.get(key) != expected:
            raise ValueError(f"checkpoint contract mismatch for {key}")
    if contract.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("checkpoint protocol digest mismatch")
    train_records = select_manifest_records(manifest, "train", limit=256)
    if contract.get("train_filenames") != [record["filename"] for record in train_records]:
        raise ValueError("checkpoint train roster mismatch")
    if contract.get("train_selection_digest") != names_digest(train_records):
        raise ValueError("checkpoint train digest mismatch")
    training = payload.get("training_configuration")
    if not isinstance(training, Mapping):
        raise ValueError("checkpoint has no training configuration")
    for key, expected in EXPECTED_TRAINING.items():
        if training.get(key) != expected:
            raise ValueError(f"checkpoint training mismatch for {key}")
    semantic = contract.get("semantic_code_sha256")
    if not isinstance(semantic, Mapping):
        raise ValueError("checkpoint has no semantic source hashes")
    semantic_paths = {
        "edge_ranker": PROJECT_ROOT / "src" / "aiijc_puzzle" / "edge_ranker.py",
        "candidate_supply": PROJECT_ROOT / "src" / "aiijc_puzzle" / "candidate_supply.py",
        "legacy_upgrade": PROJECT_ROOT / "src" / "aiijc_puzzle" / "legacy_upgrade.py",
        "protocol": PROJECT_ROOT / "src" / "aiijc_puzzle" / "protocol.py",
        "runner": PROJECT_ROOT / "scripts" / "run_edge_ranker.py",
    }
    for name, source in semantic_paths.items():
        if semantic.get(name) != sha256_file(source):
            raise ValueError(f"checkpoint semantic source drift: {name}")
    model = PairwiseEdgeRanker(
        feature_dim=int(contract["feature_dim"]),
        view_mode=str(contract["view_mode"]),
        width=int(contract["width"]),
        hidden=int(contract["hidden"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def harmonized_tail(ordered_tiles: np.ndarray, rgb_config: Any, luma_config: Any) -> dict[str, Any]:
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(ordered_tiles, rgb_config)
    rgb_tiles = apply_rgb_offsets(ordered_tiles, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(rgb_tiles, luma_config)
    harmonized = assemble_tiles(apply_luminance_gains(rgb_tiles, gains))
    return {
        "harmonized": harmonized,
        "final": apply_nlm_color(harmonized, h=20).image,
        "diagnostics": {
            "rgb_seam_offsets": rgb_diagnostics,
            "bounded_luminance_gains": luma_diagnostics,
        },
    }


def freeze_predictions(
    records: tuple[Mapping[str, Any], ...],
    *,
    inputs: Path,
    model: PairwiseEdgeRanker,
    device: torch.device,
    pair_batch: int,
    rgb_config: Any,
    luma_config: Any,
) -> list[dict[str, Any]]:
    frozen: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        started = perf_counter()
        dirty = load_verified_rgb(inputs / str(record["filename"]), str(record["input_sha256"]))
        tiles = split_tiles(dirty)
        board = build_inference_board(
            tiles,
            filename=str(record["filename"]),
            views=VIEWS,
            candidate_k=16,
        )
        narrow_board = build_inference_board(
            tiles,
            filename=str(record["filename"]),
            views=VIEWS,
            candidate_k=5,
        )
        learned_right, learned_down, delta = score_board(
            model,
            board,
            device=device,
            pair_batch=pair_batch,
        )
        solved_variants = {
            "baseline": solve_buddies(board.right_baseline, board.down_baseline, max_edges=96),
            "learned": solve_buddies(learned_right, learned_down, max_edges=96),
        }
        variants: dict[str, Any] = {}
        for name, solved in solved_variants.items():
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
                raise RuntimeError(f"strict raw audit failed for {record['filename']}/{name}")
            tail = harmonized_tail(ordered, rgb_config, luma_config)
            variants[name] = {
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
                variants[name][field].flags.writeable = False
        frozen.append(
            {
                "record": record,
                "dirty": dirty,
                "board": board,
                "narrow_board": narrow_board,
                "baseline_right": board.right_baseline,
                "baseline_down": board.down_baseline,
                "learned_right": learned_right,
                "learned_down": learned_down,
                "variants": variants,
                "delta_diagnostics": delta,
                "runtime_seconds": perf_counter() - started,
            }
        )
        print(f"froze {index}/{len(records)} {record['filename']}", flush=True)
    return frozen


def commitment_payload(
    frozen: list[dict[str, Any]],
    *,
    checkpoint: Path,
    records: tuple[Mapping[str, Any], ...],
    mode: str,
) -> dict[str, Any]:
    offset, count, selection_sha256 = PANELS[mode]
    boards = []
    for item in frozen:
        variants = {}
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
        "schema": "aiijc-edge-ranker-k16-tail-prediction-commitment-v1",
        "mode": mode,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "checkpoint_sha256": sha256_file(checkpoint),
        "split": "calibration",
        "offset": offset,
        "count": count,
        "filenames": [record["filename"] for record in records],
        "filenames_sha256": names_digest(records),
        "expected_filenames_sha256": selection_sha256,
        "all_predictions_frozen_before_target_access": True,
        "target_paths_present_in_phase_one": False,
        "pipeline": {
            "layout_control": "bilateral scores -> buddies96",
            "layout_learned": "frozen k16 raw edge-ranker residual -> buddies96",
            "strict_assembly": "all 576 original upright dirty tiles exactly once",
            "tail": "RGB seam offsets -> bounded luminance -> coloured NLM h20x1",
        },
        "boards": boards,
    }
    payload["commitment_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


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


def merge_coverage(groups: list[dict[str, dict[str, float | int]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scope in ("all", "right", "down", "trusted_query"):
        rows = sum(int(group[scope]["rows"]) for group in groups)
        present = sum(int(group[scope]["exact_in_candidate_union"]) for group in groups)
        output[scope] = {
            "rows": rows,
            "exact_in_candidate_union": present,
            "coverage": present / rows if rows else 0.0,
        }
    return output


def make_manual_sheet(
    frozen: list[dict[str, Any]],
    targets: Mapping[str, np.ndarray],
    output: Path,
) -> None:
    thumb, header, label_width = 144, 34, 150
    columns = (
        "dirty input",
        "clean target",
        "baseline raw",
        "k16 raw",
        "baseline final",
        "k16 final",
    )
    canvas = Image.new(
        "RGB",
        (label_width + thumb * len(columns), header + thumb * len(frozen)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(columns):
        draw.text((label_width + column * thumb + 4, 8), label, fill="black")
    for row, item in enumerate(frozen):
        y = header + row * thumb
        name = str(item["record"]["filename"])
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
    broad_coverage: list[dict[str, dict[str, float | int]]] = []
    narrow_coverage: list[dict[str, dict[str, float | int]]] = []
    for item in frozen:
        record = item["record"]
        target = load_verified_rgb(
            targets_dir / str(record["filename"]), str(record["target_sha256"])
        )
        targets[str(record["filename"])] = target
        clean_tiles = split_tiles(target)
        recovered = recover_layout(item["board"].tiles, clean_tiles)
        labelled = attach_target_labels(item["board"], clean_tiles)
        narrow_labelled = attach_target_labels(item["narrow_board"], clean_tiles)
        broad_coverage.append(candidate_coverage(labelled.rows))
        narrow_coverage.append(candidate_coverage(narrow_labelled.rows))
        baseline_counts.append(
            exact_edge_counts(labelled, item["baseline_right"], item["baseline_down"])
        )
        learned_counts.append(
            exact_edge_counts(labelled, item["learned_right"], item["learned_down"])
        )
        record_metrics: dict[str, Any] = {
            "filename": record["filename"],
            "variants": {},
        }
        for name, variant in item["variants"].items():
            record_metrics["variants"][name] = {
                **layout_metrics(variant["layout"], recovered),
                "raw_ssim": contest_ssim(target, variant["raw"]),
                "harmonized_ssim": contest_ssim(target, variant["harmonized"]),
                "final_ssim": contest_ssim(target, variant["final"]),
            }
        boards.append(record_metrics)
    make_manual_sheet(frozen, targets, sheet_path)

    fields = tuple(boards[0]["variants"]["baseline"])
    means = {
        variant: {
            field: float(np.mean([board["variants"][variant][field] for board in boards]))
            for field in fields
        }
        for variant in ("baseline", "learned")
    }
    paired_deltas: dict[str, Any] = {}
    difference_vectors: dict[str, np.ndarray] = {}
    for index, field in enumerate(fields):
        differences = np.asarray(
            [
                board["variants"]["learned"][field] - board["variants"]["baseline"][field]
                for board in boards
            ],
            dtype=np.float64,
        )
        difference_vectors[field] = differences
        paired_deltas[field] = paired_bootstrap_ci(
            differences,
            seed=20260840 + index,
            replicates=BOOTSTRAP_REPLICATES,
        )
        paired_deltas[field].update(
            {
                "wins": int(np.sum(differences > 0)),
                "ties": int(np.sum(differences == 0)),
                "losses": int(np.sum(differences < 0)),
            }
        )
    gate = scale_promotion_gate(
        [board["variants"]["learned"]["adjacency"] for board in boards],
        difference_vectors["adjacency"],
        difference_vectors["final_ssim"],
        difference_vectors["translation_aligned_placement"],
        replicates=BOOTSTRAP_REPLICATES,
    )
    local = {
        "baseline_bilateral": merge_edge_counts(baseline_counts),
        "learned_k16": merge_edge_counts(learned_counts),
        "candidate_coverage_k5_counterfactual": merge_coverage(narrow_coverage),
        "candidate_coverage_k16": merge_coverage(broad_coverage),
    }
    local["collapse_diagnostics"] = {
        "k16_coverage_below_same_panel_k5": bool(
            local["candidate_coverage_k16"]["all"]["coverage"]
            < local["candidate_coverage_k5_counterfactual"]["all"]["coverage"]
        ),
        "learned_all_pooled_r1_below_bilateral": bool(
            local["learned_k16"]["all.pooled.r1"]["recall"]
            < local["baseline_bilateral"]["all.pooled.r1"]["recall"]
        ),
    }
    return {
        "commitment_sha256": commitment_sha256,
        "target_access_started_only_after_commitment": True,
        "board_count": len(boards),
        "means": means,
        "paired_deltas": paired_deltas,
        "primary_gate": gate,
        "local_edge_metrics": local,
        "boards": boards,
        "manual_sheet": str(sheet_path),
    }


def require_confirmation_authorisation(checkpoint: Path) -> None:
    report_path = RUN_ROOT / "final-tail-primary" / "report.json"
    if not report_path.is_file():
        raise RuntimeError("confirmation requires the completed primary report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("evaluation", {}).get("primary_gate", {}).get("passed"):
        raise RuntimeError("primary gate failed; confirmation is forbidden")
    if report.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint):
        raise RuntimeError("checkpoint differs from the primary run")


def main() -> None:
    args = parse_args()
    if args.pair_batch <= 0:
        raise ValueError("pair-batch must be positive")
    if args.mode == "confirmation":
        require_confirmation_authorisation(args.checkpoint)
    offset, count, expected_selection = PANELS[args.mode]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ranked = select_manifest_records(manifest, "calibration", limit=offset + count)
    records = tuple(ranked[offset:])
    if len(records) != count or names_digest(records) != expected_selection:
        raise RuntimeError("panel roster differs from the preregistration")
    output_dir = (
        args.output_dir if args.output_dir is not None else RUN_ROOT / f"final-tail-{args.mode}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, checkpoint_payload = load_checkpoint(
        args.checkpoint,
        manifest=manifest,
        device=device,
    )
    rgb_config, luma_config, method_hashes = _validate_method_configs()

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
        mode=args.mode,
    )
    commitment_path = output_dir / "prediction-commitment.json"
    atomic_json(commitment_path, commitment)
    if (
        json.loads(commitment_path.read_text(encoding="utf-8")).get("commitment_sha256")
        != commitment["commitment_sha256"]
    ):
        raise RuntimeError("prediction commitment readback failed")

    target_started = perf_counter()
    evaluation = evaluate_after_commitment(
        frozen,
        targets_dir=args.targets,
        commitment_sha256=commitment["commitment_sha256"],
        sheet_path=output_dir / "manual-layout-sheet.png",
    )
    target_seconds = perf_counter() - target_started
    gate_passed = bool(evaluation["primary_gate"]["passed"])
    report = {
        "schema": "aiijc-edge-ranker-k16-tail-audit-v1",
        "experiment": "preregistered-k16-train256-raw-ranker-vs-bilateral-h20x1",
        "mode": args.mode,
        "verdict": (
            "gate-passed-confirmation-required"
            if args.mode == "primary" and gate_passed
            else "confirmation-passed-eligible-for-review"
            if args.mode == "confirmation" and gate_passed
            else "gate-failed-do-not-confirm-or-integrate"
        ),
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": sha256_file(PREREGISTRATION),
            "expected_sha256": PREREGISTRATION_SHA256,
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "contract": checkpoint_payload.get("contract"),
            "training_configuration": checkpoint_payload.get("training_configuration"),
            "training_history": checkpoint_payload.get("training_history"),
            "training_report_path": str(args.training_report.resolve()),
            "training_report_sha256": sha256_file(args.training_report),
        },
        "selection": {
            "split": "calibration",
            "offset": offset,
            "count": count,
            "filenames": [record["filename"] for record in records],
            "filenames_sha256": names_digest(records),
            "expected_filenames_sha256": expected_selection,
        },
        "leakage_boundary": {
            "phase_one": (
                "dirty input only; k5/k16 candidates, scores, strict layouts and full tails frozen"
            ),
            "commitment_path": str(commitment_path),
            "all_predictions_frozen_before_target_access": True,
            "phase_two": "paired target only for labels, metrics and manual sheet",
            "holdout_opened": False,
            "test_opened": False,
        },
        "compliance": {
            "decoder": "no-atlas buddies96 for both arms",
            "input_tiles": 576,
            "required_unique_tiles": 576,
            "raw_assembly": "original upright input tile pixels exactly once",
            "tail": "frozen RGB offsets -> bounded luma -> coloured NLM h20x1",
            "method_config_sha256": method_hashes,
            "constant_template_cross_board_external_substitution": False,
        },
        "source_sha256": {
            "evaluator": sha256_file(Path(__file__)),
            "scale_gate": sha256_file(
                PROJECT_ROOT / "src" / "aiijc_puzzle" / "edge_ranker_scale_gate.py"
            ),
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
                "primary_gate": evaluation["primary_gate"],
                "local_edge_metrics": evaluation["local_edge_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
