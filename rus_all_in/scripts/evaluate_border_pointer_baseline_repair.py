#!/usr/bin/env python3
"""One preregistered same-panel baseline-guided BorderPointer rescue."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.border_pointer_repair import (
    BaselineRepairConfig,
    baseline_guided_pointer_repair,
)
from aiijc_puzzle.border_pointer_sorter import BorderPointerSorter
from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    compute_protocol_digest,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.socket_sorter_production import load_socket_checkpoint
from aiijc_puzzle.synthetic_socket_evaluation import (
    ExactSyntheticReference,
    SyntheticSocketInput,
    make_exact_synthetic_case,
    names_digest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INITIAL_REPORT = (
    PROJECT_ROOT
    / "outputs/border-pointer/pilot-d64-train128-s400-exact16-mps/report.json"
)
DEFAULT_PREREGISTRATION = (
    PROJECT_ROOT / "configs/border_pointer_baseline_repair_preregistered_v1.json"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
GRID = 24
COUNT = GRID * GRID
TRACE_K = 5


@dataclass(frozen=True)
class CleanBoard:
    filename: str
    target_sha256: str
    tiles: np.ndarray


@dataclass(frozen=True)
class TrainingCase:
    synthetic_input: SyntheticSocketInput
    reference: ExactSyntheticReference


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected RGB 480x480 image: {path}")
        return np.asarray(image, dtype=np.uint8)


def load_clean_boards(
    records: tuple[dict[str, Any], ...],
    *,
    targets: Path,
) -> list[CleanBoard]:
    boards: list[CleanBoard] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        path = targets / filename
        observed = sha256_file(path)
        if observed != record.get("target_sha256"):
            raise ValueError(f"manifest target hash mismatch for {filename}")
        boards.append(CleanBoard(filename, observed, split_tiles(_load_rgb(path))))
        if index == 1 or index == len(records):
            print(f"loaded clean source {index}/{len(records)} {filename}", flush=True)
    return boards


def build_cases(boards: list[CleanBoard], *, seed: int) -> list[TrainingCase]:
    cases: list[TrainingCase] = []
    for board in boards:
        synthetic_input, reference = make_exact_synthetic_case(
            board.tiles,
            source_filename=board.filename,
            draw_index=0,
            seed=seed,
        )
        cases.append(TrainingCase(synthetic_input, reference))
    return cases


def _tensor_tiles(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).permute(0, 3, 1, 2).unsqueeze(0).to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-report", type=Path, default=DEFAULT_INITIAL_REPORT)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def _records_by_names(
    manifest: dict[str, Any],
    names: list[str],
) -> tuple[dict[str, Any], ...]:
    train = manifest.get("splits", {}).get("train")
    if not isinstance(train, list):
        raise ValueError("manifest has no train split")
    by_name = {str(record.get("filename")): record for record in train}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"evaluation filenames are absent from manifest train: {missing}")
    return tuple(by_name[name] for name in names)


def _load_model(
    initial: dict[str, Any],
    *,
    socket: Any,
    device: torch.device,
) -> tuple[BorderPointerSorter, dict[str, Any], Path]:
    architecture = initial.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("initial report has no architecture mapping")
    model = BorderPointerSorter(
        socket_backbone=socket.model,
        feature_width=int(architecture["feature_width"]),
        feature_blocks=int(architecture["feature_blocks"]),
        dimension=int(architecture["dimension"]),
        heads=int(architecture["heads"]),
        board_layers=int(architecture["board_layers"]),
        pointer_layers=int(architecture["pointer_layers"]),
        max_grid=GRID,
        freeze_socket=True,
    ).to(device)
    checkpoint_record = initial.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise ValueError("initial report has no checkpoint record")
    checkpoint_path = Path(str(checkpoint_record["path"])).resolve()
    if sha256_file(checkpoint_path) != checkpoint_record.get("sha256"):
        raise ValueError("initial BorderPointer checkpoint digest changed")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "aiijc-border-pointer-checkpoint-v1":
        raise ValueError("unsupported BorderPointer checkpoint schema")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload, checkpoint_path


def _load_initial_baselines(
    initial: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    record = initial.get("frozen_predictions")
    if not isinstance(record, dict):
        raise ValueError("initial report has no frozen predictions")
    arrays_path = Path(str(record["arrays_path"])).resolve()
    metadata_path = Path(str(record["metadata_path"])).resolve()
    if sha256_file(arrays_path) != record.get("arrays_sha256") or sha256_file(
        metadata_path
    ) != record.get("metadata_sha256"):
        raise ValueError("initial frozen prediction digest changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("contains_exact_references") is not False:
        raise ValueError("initial prediction metadata is not label-free")
    cases = metadata.get("cases")
    if not isinstance(cases, list) or len(cases) != 16:
        raise ValueError("initial frozen metadata must contain 16 cases")
    loaded = np.load(arrays_path)
    baselines = {
        str(case["case_id"]): np.ascontiguousarray(
            loaded[f"{case['array_prefix']}__socket_decoder144"].astype(np.int32)
        )
        for case in cases
    }
    return cases, baselines


def _write_frozen(
    frozen: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    arrays: dict[str, np.ndarray] = {}
    metadata_cases: list[dict[str, Any]] = []
    for index, row in enumerate(frozen):
        prefix = f"case_{index:04d}"
        arrays[f"{prefix}__baseline"] = row["baseline"]
        arrays[f"{prefix}__repair_budget4"] = row["repair_budget4"]
        arrays[f"{prefix}__repair_budget16"] = row["repair_budget16"]
        arrays[f"{prefix}__prefix_topk"] = row["prefix_topk"]
        arrays[f"{prefix}__no_prefix_topk"] = row["no_prefix_topk"]
        metadata_cases.append(
            {
                "array_prefix": prefix,
                "case_id": row["case_id"],
                "source_filename": row["source_filename"],
                "draw_index": row["draw_index"],
                "corrupted_tiles_sha256": row["corrupted_tiles_sha256"],
                "repair_proposals": row["repair_proposals"],
                "runtime_seconds": row["runtime_seconds"],
            }
        )
    arrays_path = output_dir / "frozen_predictions.npz"
    np.savez_compressed(arrays_path, **arrays)
    metadata_path = output_dir / "frozen_predictions.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "aiijc-border-pointer-baseline-repair-frozen-v1",
                "contains_exact_references": False,
                "contains_clean_pixels": False,
                "cases": metadata_cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return arrays_path, metadata_path


def _numeric_mean(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def _conditional_metrics(
    topk: np.ndarray,
    baseline: np.ndarray,
    reference: np.ndarray,
) -> dict[str, int | float]:
    baseline_position = np.empty(COUNT, dtype=np.int32)
    baseline_position[baseline] = np.arange(COUNT)
    positions = np.arange(COUNT)
    eligible = baseline_position[reference] >= positions
    eligible_count = int(eligible.sum())
    hits1 = int(np.count_nonzero((topk[:, 0] == reference) & eligible))
    hits5 = int(
        np.count_nonzero(np.any(topk[:, :TRACE_K] == reference[:, None], axis=1) & eligible)
    )
    return {
        "eligible": eligible_count,
        "total": COUNT,
        "coverage": eligible_count / COUNT,
        "hits_at_1": hits1,
        "r1": hits1 / eligible_count,
        "hits_at_5": hits5,
        "r5": hits5 / eligible_count,
    }


def _aggregate_conditional(rows: list[dict[str, int | float]]) -> dict[str, int | float]:
    eligible = sum(int(row["eligible"]) for row in rows)
    total = sum(int(row["total"]) for row in rows)
    hits1 = sum(int(row["hits_at_1"]) for row in rows)
    hits5 = sum(int(row["hits_at_5"]) for row in rows)
    return {
        "eligible": eligible,
        "total": total,
        "coverage": eligible / total,
        "hits_at_1": hits1,
        "r1": hits1 / eligible,
        "hits_at_5": hits5,
        "r5": hits5 / eligible,
    }


def _score_frozen(
    frozen: list[dict[str, Any]],
    cases: list[TrainingCase],
) -> dict[str, Any]:
    references = {case.reference.case_id: case.reference.tile_at_position for case in cases}
    boards: list[dict[str, Any]] = []
    for row in frozen:
        reference = references[row["case_id"]]
        global_metrics = {
            name: evaluate_layout(layout, reference, reference_is_exact=True).as_dict()
            for name, layout in (
                ("baseline", row["baseline"]),
                ("repair_budget4", row["repair_budget4"]),
                ("repair_budget16", row["repair_budget16"]),
            )
        }
        conditional = {
            "baseline_prefix": _conditional_metrics(
                row["prefix_topk"], row["baseline"], reference
            ),
            "no_prefix": _conditional_metrics(
                row["no_prefix_topk"], row["baseline"], reference
            ),
        }
        boards.append(
            {
                "case_id": row["case_id"],
                "source_filename": row["source_filename"],
                "global": global_metrics,
                "conditional": conditional,
            }
        )
    variants = ("baseline", "repair_budget4", "repair_budget16")
    means = {
        variant: _numeric_mean([board["global"][variant] for board in boards])
        for variant in variants
    }
    totals = {
        variant: {
            "correct_tile_count": int(
                sum(board["global"][variant]["correct_tile_count"] for board in boards)
            ),
            "adjacency_correct": int(
                sum(board["global"][variant]["adjacency_correct"] for board in boards)
            ),
        }
        for variant in variants
    }
    delta_keys = (
        "correct_tile_count",
        "correct_row_count",
        "correct_column_count",
        "adjacency",
    )
    deltas = {
        variant: {
            key: means[variant][key] - means["baseline"][key] for key in delta_keys
        }
        for variant in variants[1:]
    }
    conditional = {
        name: _aggregate_conditional(
            [board["conditional"][name] for board in boards]
        )
        for name in ("baseline_prefix", "no_prefix")
    }
    conditional["delta_prefix_minus_no_prefix"] = {
        "r1": float(conditional["baseline_prefix"]["r1"])
        - float(conditional["no_prefix"]["r1"]),
        "r5": float(conditional["baseline_prefix"]["r5"])
        - float(conditional["no_prefix"]["r5"]),
    }
    low_signal: dict[str, Any] = {}
    for variant in variants[1:]:
        exact_total_delta = (
            totals[variant]["correct_tile_count"] - totals["baseline"]["correct_tile_count"]
        )
        adjacency_delta = deltas[variant]["adjacency"]
        low_signal[variant] = {
            "exact_total_delta": exact_total_delta,
            "adjacency_delta": adjacency_delta,
            "exact_signal": exact_total_delta >= 1 and adjacency_delta >= -0.005,
            "geometry_signal": adjacency_delta >= 0 and exact_total_delta >= -1,
        }
    return {
        "reference": "same already-opened exact16 inverse shuffles",
        "case_count": len(boards),
        "global_mean": means,
        "global_totals": totals,
        "delta_vs_baseline": deltas,
        "conditional": conditional,
        "low_discovery_signal": low_signal,
        "promotion_authorized": False,
        "boards": boards,
    }


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preregistration_hash = sha256_file(args.preregistration)
    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    initial_hash = sha256_file(args.initial_report)
    if initial_hash != preregistration["initial_pilot"]["report_sha256"]:
        raise ValueError("initial report differs from the preregistered rescue input")
    initial = json.loads(args.initial_report.read_text(encoding="utf-8"))
    if initial.get("experiment") != "border-pointer-24-bounded-v1":
        raise ValueError("unsupported initial BorderPointer report")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise ValueError("manifest protocol digest is invalid")
    socket_record = initial["socket_checkpoint"]
    socket = load_socket_checkpoint(Path(socket_record["path"]), device=device)
    if socket.sha256 != socket_record["sha256"]:
        raise ValueError("frozen Socket checkpoint differs from the initial pilot")
    model, model_payload, checkpoint_path = _load_model(
        initial,
        socket=socket,
        device=device,
    )
    metadata_cases, baselines = _load_initial_baselines(initial)
    eval_names = list(initial["selection"]["evaluation_source_filenames"])
    if names_digest(eval_names) != initial["selection"]["evaluation_source_digest"]:
        raise ValueError("initial evaluation source digest is invalid")
    if eval_names != [str(row["source_filename"]) for row in metadata_cases]:
        raise ValueError("initial metadata order differs from evaluation selection")
    records = _records_by_names(manifest, eval_names)
    selection_commitment = json.loads(
        Path(initial["selection"]["selection_commitment_path"]).read_text(encoding="utf-8")
    )
    seed = int(selection_commitment["seed"])
    clean = load_clean_boards(
        records,
        targets=args.targets.resolve(),
    )
    cases = build_cases(clean, seed=seed + 100_000)
    if [case.synthetic_input.case_id for case in cases] != [
        str(row["case_id"]) for row in metadata_cases
    ]:
        raise ValueError("reconstructed case identifiers differ from the initial panel")

    fixed = preregistration["fixed_algorithm"]
    config = BaselineRepairConfig(
        logit_margin=float(fixed["acceptance_logit_margin"]),
        budgets=tuple(int(value) for value in fixed["accepted_repair_budgets"]),
        socket_support_topk=int(fixed["socket_guard"]["supported_edge_topk"]),
    )
    frozen: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        tiles = _tensor_tiles(case.synthetic_input.tiles, device)
        baseline = baselines[case.synthetic_input.case_id]
        started = perf_counter()
        with torch.no_grad():
            socket_output = socket.model(tiles, grid=GRID)
            result = baseline_guided_pointer_repair(
                model,
                tiles,
                baseline,
                socket_output.right_log_assignment,
                socket_output.down_log_assignment,
                grid=GRID,
                config=config,
                trace_topk=TRACE_K,
            )
        runtime = perf_counter() - started
        frozen.append(
            {
                "case_id": case.synthetic_input.case_id,
                "source_filename": case.synthetic_input.source_filename,
                "draw_index": case.synthetic_input.draw_index,
                "corrupted_tiles_sha256": hashlib.sha256(
                    np.ascontiguousarray(case.synthetic_input.tiles).tobytes()
                ).hexdigest(),
                "baseline": baseline,
                "repair_budget4": result.layouts[4],
                "repair_budget16": result.layouts[16],
                "prefix_topk": result.trace.prefix_topk,
                "no_prefix_topk": result.trace.no_prefix_topk,
                "repair_proposals": list(result.proposals),
                "runtime_seconds": runtime,
            }
        )
        accepted = sum(bool(row["accepted"]) for row in result.proposals)
        print(
            f"froze rescue {index}/{len(cases)} {case.synthetic_input.case_id} "
            f"accepted={accepted} runtime={runtime:.2f}s",
            flush=True,
        )
    arrays_path, metadata_path = _write_frozen(frozen, output_dir=output_dir)
    frozen_record = {
        "arrays_path": str(arrays_path),
        "arrays_sha256": sha256_file(arrays_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
    print(f"frozen artifact committed: {arrays_path}", flush=True)
    evaluation = _score_frozen(frozen, cases)
    proposal_counts = {
        "margin_passed": sum(len(row["repair_proposals"]) for row in frozen),
        "guard_accepted": sum(
            sum(bool(proposal["accepted"]) for proposal in row["repair_proposals"])
            for row in frozen
        ),
        "guard_vetoed": sum(
            sum(not bool(proposal["accepted"]) for proposal in row["repair_proposals"])
            for row in frozen
        ),
    }
    report = {
        "experiment": "border-pointer-baseline-guided-repair-v1",
        "status": "same-opened-panel-development-only",
        "preregistration": {
            "path": str(args.preregistration.resolve()),
            "sha256": preregistration_hash,
            "payload": preregistration,
        },
        "initial_pilot": {
            "path": str(args.initial_report.resolve()),
            "sha256": initial_hash,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "protocol": {
            "same_already_opened_exact16_panel": True,
            "dirty_only_rescue_predictions_frozen_before_rescue_scoring": True,
            "truth_used_during_prediction": False,
            "strict_original_upright_tile_permutation": True,
            "fresh_source64_draw2_opened": False,
            "competition_test_opened": False,
            "promotion_authorized": False,
        },
        "selection": {
            "evaluation_source_filenames": eval_names,
            "evaluation_source_digest": names_digest(eval_names),
        },
        "fixed_config": {
            "logit_margin": config.logit_margin,
            "budgets": list(config.budgets),
            "socket_support_topk": config.socket_support_topk,
        },
        "model_checkpoint_schema": model_payload["schema"],
        "proposal_counts": proposal_counts,
        "frozen_predictions": frozen_record,
        "evaluation": evaluation,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "report": str(report_path),
                "proposal_counts": proposal_counts,
                "global_mean": evaluation["global_mean"],
                "delta": evaluation["delta_vs_baseline"],
                "conditional": evaluation["conditional"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
