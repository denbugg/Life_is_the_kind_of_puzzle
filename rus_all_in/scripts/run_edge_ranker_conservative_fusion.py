#!/usr/bin/env python3
"""Preregistered reused-calibration audit of conservative k16 edge fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from run_edge_ranker_k16_tail import (
    CHECKPOINT,
    INPUTS,
    MANIFEST,
    TARGETS,
    VIEWS,
    choose_device,
    harmonized_tail,
    load_checkpoint,
    load_verified_rgb,
)

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.edge_ranker import build_inference_board, score_board
from aiijc_puzzle.edge_ranker_conservative_fusion import FusionArm, apply_conservative_fusion
from aiijc_puzzle.edge_ranker_final_tail import (
    layout_metrics,
    names_digest,
    paired_bootstrap_ci,
)
from aiijc_puzzle.frozen_final_evaluator import (
    _validate_method_configs,
    array_digest,
    atomic_json,
)
from aiijc_puzzle.legacy_upgrade import layout_digest, solve_buddies, validate_layout
from aiijc_puzzle.protocol import (
    assemble_tiles,
    contest_ssim,
    select_manifest_records,
    sha256_file,
    split_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = PROJECT_ROOT / "configs" / "edge_ranker_conservative_fusion_preregistered_v1.json"
PREREGISTRATION_SHA256 = "6acf7d813cc1609b10bbff8651cef447bbe9f8bb1771551d966bdf895690614f"
RUN_ROOT = PROJECT_ROOT / "outputs" / "edge-ranker" / "conservative-fusion-k16"
BOOTSTRAP_REPLICATES = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("primary", "confirmation"), default="primary")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--pair-batch", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_preregistration() -> Mapping[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("conservative-fusion preregistration hash drift")
    config = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    diagnostic = config["train_only_arm_selection"]
    pinned_paths = {
        "report_sha256": PROJECT_ROOT / diagnostic["report_path"],
        "diagnostic_runner_sha256": (
            PROJECT_ROOT / "scripts" / "run_edge_ranker_conservative_fusion_train_diagnostic.py"
        ),
        "fusion_source_sha256": (
            PROJECT_ROOT / "src" / "aiijc_puzzle" / "edge_ranker_conservative_fusion.py"
        ),
    }
    for field, path in pinned_paths.items():
        if sha256_file(path) != diagnostic[field]:
            raise RuntimeError(f"preregistered dependency hash drift: {path}")
    if sha256_file(CHECKPOINT) != config["frozen_checkpoint"]["sha256"]:
        raise RuntimeError("preregistered checkpoint hash drift")
    return config


def arms_from_config(config: Mapping[str, Any]) -> tuple[FusionArm, ...]:
    arms = tuple(
        FusionArm(
            str(item["name"]),
            int(item["max_new_edges"]),
            int(item["min_top4_view_votes"]),
            float(item["min_confidence"]),
        )
        for item in config["arms_in_preregistered_priority_order"]
    )
    if not 1 <= len(arms) <= 5 or len({arm.name for arm in arms}) != len(arms):
        raise RuntimeError("preregistration must contain one to five uniquely named arms")
    return arms


def panel_records(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    mode: str,
) -> tuple[Mapping[str, Any], ...]:
    panel = config["panels"][mode]
    offset, count = int(panel["offset"]), int(panel["count"])
    ranked = select_manifest_records(
        manifest,
        str(panel["split"]),
        limit=offset + count,
    )
    records = tuple(ranked[offset:])
    if len(records) != count:
        raise RuntimeError("selected panel is incomplete")
    if names_digest(records) != panel["newline_filenames_sha256"]:
        raise RuntimeError("selected panel roster differs from preregistration")
    return records


def require_confirmation_authorisation(
    config: Mapping[str, Any],
) -> FusionArm:
    report_path = RUN_ROOT / "primary-cal360-count24" / "report.json"
    if not report_path.is_file():
        raise RuntimeError("confirmation requires a completed primary report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise RuntimeError("primary report belongs to a different preregistration")
    winner = report.get("evaluation", {}).get("winner")
    if not isinstance(winner, str):
        raise RuntimeError("primary has no passing winner; confirmation is forbidden")
    arms = {arm.name: arm for arm in arms_from_config(config)}
    if winner not in arms:
        raise RuntimeError("primary winner is not a preregistered arm")
    return arms[winner]


def freeze_predictions(
    records: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    device: Any,
    pair_batch: int,
    arms: Sequence[FusionArm],
    rgb_config: Any,
    luma_config: Any,
) -> list[dict[str, Any]]:
    frozen = []
    for board_index, record in enumerate(records, start=1):
        started = perf_counter()
        dirty = load_verified_rgb(INPUTS / str(record["filename"]), str(record["input_sha256"]))
        tiles = split_tiles(dirty)
        board = build_inference_board(
            tiles,
            filename=str(record["filename"]),
            views=VIEWS,
            candidate_k=16,
        )
        learned_right, learned_down, delta = score_board(
            model,
            board,
            device=device,
            pair_batch=pair_batch,
        )
        scores: dict[str, tuple[np.ndarray, np.ndarray, Mapping[str, Any]]] = {
            "baseline": (
                board.right_baseline.copy(),
                board.down_baseline.copy(),
                {"selected_count": 0, "selected": []},
            )
        }
        for arm in arms:
            scores[arm.name] = apply_conservative_fusion(
                board,
                learned_right,
                learned_down,
                arm,
            )
        variants = {}
        tail_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, (right, down, diagnostics) in scores.items():
            solved = solve_buddies(right, down, max_edges=96)
            layout = validate_layout(solved.layout)
            digest = layout_digest(layout)
            if digest not in tail_cache:
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
                tail_cache[digest] = (raw, tail["harmonized"], tail["final"])
            raw, harmonized, final = tail_cache[digest]
            variant = {
                "right": np.ascontiguousarray(right),
                "down": np.ascontiguousarray(down),
                "layout": layout,
                "raw": raw,
                "harmonized": harmonized,
                "final": final,
                "audit": audit_raw_permutation(
                    dirty,
                    raw,
                    layout,
                    restoration_applied_after_audit=True,
                ).as_dict(),
                "objective": float(solved.objective),
                "solver": solved.solver,
                "fusion_diagnostics": diagnostics,
            }
            for field in ("right", "down", "layout", "raw", "harmonized", "final"):
                variant[field].flags.writeable = False
            variants[name] = variant
        frozen.append(
            {
                "record": record,
                "dirty": dirty,
                "board": board,
                "learned_delta": delta,
                "variants": variants,
                "runtime_seconds": perf_counter() - started,
            }
        )
        print(f"froze {board_index}/{len(records)} {record['filename']}", flush=True)
    return frozen


def _commitment_self_hash(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "commitment_sha256"}
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _numeric_array_digest(value: np.ndarray) -> str:
    """Content-address an arbitrary finite numeric array with its exact contract."""

    array = np.ascontiguousarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("numeric commitment arrays must be finite")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def commitment_payload(
    frozen: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    config: Mapping[str, Any],
    arms: Sequence[FusionArm],
) -> dict[str, Any]:
    boards = []
    for item in frozen:
        variants = {}
        for name, value in item["variants"].items():
            variants[name] = {
                "right_scores_sha256": _numeric_array_digest(value["right"]),
                "down_scores_sha256": _numeric_array_digest(value["down"]),
                "layout_sha256": layout_digest(value["layout"]),
                "raw_sha256": array_digest(value["raw"]),
                "harmonized_sha256": array_digest(value["harmonized"]),
                "final_sha256": array_digest(value["final"]),
                "audit": value["audit"],
                "objective": value["objective"],
                "solver": value["solver"],
                "fusion_diagnostics": value["fusion_diagnostics"],
            }
        boards.append(
            {
                "filename": item["record"]["filename"],
                "input_sha256": item["record"]["input_sha256"],
                "learned_delta": item["learned_delta"],
                "variants": variants,
                "runtime_seconds": item["runtime_seconds"],
            }
        )
    panel = config["panels"][mode]
    payload: dict[str, Any] = {
        "schema": "aiijc-edge-ranker-conservative-fusion-commitment-v1",
        "mode": mode,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "historical_calibration_exposure_acknowledged": True,
        "current_run_target_bytes_read_before_commitment": False,
        "split": panel["split"],
        "offset": panel["offset"],
        "count": panel["count"],
        "filenames": [item["record"]["filename"] for item in frozen],
        "filenames_sha256": names_digest([item["record"] for item in frozen]),
        "expected_filenames_sha256": panel["newline_filenames_sha256"],
        "evaluated_arms": [arm.__dict__ for arm in arms],
        "all_score_matrices_layouts_and_images_frozen": True,
        "boards": boards,
    }
    payload["commitment_sha256"] = _commitment_self_hash(payload)
    return payload


def make_manual_sheet(
    frozen: Sequence[Mapping[str, Any]],
    targets: Mapping[str, np.ndarray],
    arm_names: Sequence[str],
    output: Path,
) -> None:
    thumb, header, label_width = 116, 34, 150
    names = ("baseline", *arm_names)
    columns = (
        "dirty",
        "target",
        *(f"{name} raw" for name in names),
        *(f"{name} final" for name in names),
    )
    canvas = Image.new(
        "RGB",
        (label_width + thumb * len(columns), header + thumb * len(frozen)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(columns):
        draw.text((label_width + column * thumb + 3, 8), label, fill="black")
    for row_index, item in enumerate(frozen):
        y = header + row_index * thumb
        filename = str(item["record"]["filename"])
        draw.text((4, y + 5), filename, fill="black")
        images = (
            item["dirty"],
            targets[filename],
            *(item["variants"][name]["raw"] for name in names),
            *(item["variants"][name]["final"] for name in names),
        )
        for column, array in enumerate(images):
            image = Image.fromarray(array).resize((thumb, thumb), Image.Resampling.LANCZOS)
            canvas.paste(image, (label_width + column * thumb, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _gate(
    boards: Sequence[Mapping[str, Any]],
    arm_name: str,
    *,
    arm_index: int,
) -> dict[str, Any]:
    final = np.asarray(
        [
            board["variants"][arm_name]["final_ssim"] - board["variants"]["baseline"]["final_ssim"]
            for board in boards
        ],
        dtype=np.float64,
    )
    adjacency = np.asarray(
        [
            board["variants"][arm_name]["adjacency"] - board["variants"]["baseline"]["adjacency"]
            for board in boards
        ],
        dtype=np.float64,
    )
    translation = np.asarray(
        [
            board["variants"][arm_name]["translation_aligned_placement"]
            - board["variants"]["baseline"]["translation_aligned_placement"]
            for board in boards
        ],
        dtype=np.float64,
    )
    final_ci = paired_bootstrap_ci(
        final,
        seed=20260910 + arm_index * 10,
        replicates=BOOTSTRAP_REPLICATES,
    )
    adjacency_ci = paired_bootstrap_ci(
        adjacency,
        seed=20260911 + arm_index * 10,
        replicates=BOOTSTRAP_REPLICATES,
    )
    final_mean = float(np.mean([board["variants"][arm_name]["final_ssim"] for board in boards]))
    baseline_mean = float(
        np.mean([board["variants"]["baseline"]["final_ssim"] for board in boards])
    )
    observed = {
        "final_ssim_mean": final_mean,
        "final_ssim_delta_mean": final_mean - baseline_mean,
        "final_ssim_delta_ci95_lower": float(final_ci["ci95_lower"]),
        "adjacency_delta_ci95_lower": float(adjacency_ci["ci95_lower"]),
        "translation_aligned_placement_delta_mean": float(translation.mean()),
    }
    required = (
        ("final_ssim_delta_mean", ">= 0.004", observed["final_ssim_delta_mean"] >= 0.004),
        ("final_ssim_mean", ">= 0.27", observed["final_ssim_mean"] >= 0.27),
        (
            "final_ssim_delta_ci95_lower",
            "> 0",
            observed["final_ssim_delta_ci95_lower"] > 0.0,
        ),
        (
            "adjacency_delta_ci95_lower",
            ">= 0",
            observed["adjacency_delta_ci95_lower"] >= 0.0,
        ),
        (
            "translation_aligned_placement_delta_mean",
            ">= 0",
            observed["translation_aligned_placement_delta_mean"] >= 0.0,
        ),
    )
    conditions = [
        {"metric": metric, "observed": observed[metric], "required": rule, "passed": bool(ok)}
        for metric, rule, ok in required
    ]
    return {
        "passed": all(condition["passed"] for condition in conditions),
        "conditions": conditions,
        "final_ssim_delta": final_ci,
        "adjacency_delta": adjacency_ci,
        "translation_delta_mean": float(translation.mean()),
        "wins_ties_losses_final": {
            "wins": int(np.sum(final > 0)),
            "ties": int(np.sum(final == 0)),
            "losses": int(np.sum(final < 0)),
        },
    }


def evaluate_after_commitment(
    frozen: Sequence[Mapping[str, Any]],
    *,
    commitment_sha256: str,
    arms: Sequence[FusionArm],
    sheet_path: Path,
) -> dict[str, Any]:
    targets = {}
    boards = []
    for item in frozen:
        record = item["record"]
        target = load_verified_rgb(
            TARGETS / str(record["filename"]),
            str(record["target_sha256"]),
        )
        targets[str(record["filename"])] = target
        recovered = recover_layout(item["board"].tiles, split_tiles(target))
        variants = {}
        for name, prediction in item["variants"].items():
            variants[name] = {
                **layout_metrics(prediction["layout"], recovered),
                "raw_ssim": contest_ssim(target, prediction["raw"]),
                "harmonized_ssim": contest_ssim(target, prediction["harmonized"]),
                "final_ssim": contest_ssim(target, prediction["final"]),
                "selected_count": int(prediction["fusion_diagnostics"]["selected_count"]),
            }
        boards.append({"filename": record["filename"], "variants": variants})
    arm_names = [arm.name for arm in arms]
    make_manual_sheet(frozen, targets, arm_names, sheet_path)
    fields = (
        "adjacency",
        "right_adjacency",
        "down_adjacency",
        "direct_placement",
        "translation_aligned_placement",
        "raw_ssim",
        "harmonized_ssim",
        "final_ssim",
        "selected_count",
    )
    means = {
        name: {
            field: float(np.mean([board["variants"][name][field] for board in boards]))
            for field in fields
        }
        for name in ("baseline", *arm_names)
    }
    gates = {arm.name: _gate(boards, arm.name, arm_index=index) for index, arm in enumerate(arms)}
    passing = [arm for arm in arms if gates[arm.name]["passed"]]
    priority = {arm.name: index for index, arm in enumerate(arms)}
    winner = None
    if passing:
        winner = max(
            passing,
            key=lambda arm: (
                means[arm.name]["final_ssim"],
                means[arm.name]["adjacency"],
                -priority[arm.name],
            ),
        ).name
    return {
        "commitment_sha256": commitment_sha256,
        "target_access_started_only_after_commitment": True,
        "historical_calibration_exposure_acknowledged": True,
        "board_count": len(boards),
        "means": means,
        "gates": gates,
        "winner": winner,
        "boards": boards,
        "manual_sheet": str(sheet_path),
    }


def main() -> None:
    args = parse_args()
    if args.pair_batch <= 0:
        raise ValueError("pair-batch must be positive")
    config = load_preregistration()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = panel_records(manifest, config, args.mode)
    all_arms = arms_from_config(config)
    arms = all_arms if args.mode == "primary" else (require_confirmation_authorisation(config),)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else RUN_ROOT
        / ("primary-cal360-count24" if args.mode == "primary" else "confirmation-cal444-count24")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, _ = load_checkpoint(CHECKPOINT, manifest=manifest, device=device)
    rgb_config, luma_config, method_hashes = _validate_method_configs()
    started = perf_counter()
    frozen = freeze_predictions(
        records,
        model=model,
        device=device,
        pair_batch=args.pair_batch,
        arms=arms,
        rgb_config=rgb_config,
        luma_config=luma_config,
    )
    freeze_seconds = perf_counter() - started
    commitment = commitment_payload(
        frozen,
        mode=args.mode,
        config=config,
        arms=arms,
    )
    commitment_path = output_dir / "prediction-commitment.json"
    atomic_json(commitment_path, commitment)
    readback = json.loads(commitment_path.read_text(encoding="utf-8"))
    if readback.get("commitment_sha256") != _commitment_self_hash(readback):
        raise RuntimeError("prediction commitment readback/self-hash failed")
    target_started = perf_counter()
    evaluation = evaluate_after_commitment(
        frozen,
        commitment_sha256=commitment["commitment_sha256"],
        arms=arms,
        sheet_path=output_dir / "manual-layout-sheet.png",
    )
    target_seconds = perf_counter() - target_started
    winner = evaluation["winner"]
    report = {
        "schema": "aiijc-edge-ranker-conservative-fusion-audit-v1",
        "mode": args.mode,
        "verdict": (
            "primary-gate-passed-confirmation-required"
            if args.mode == "primary" and winner is not None
            else "confirmation-gate-passed-eligible-for-review"
            if args.mode == "confirmation" and winner is not None
            else "gate-failed-do-not-confirm-or-integrate"
        ),
        "preregistration_path": str(PREREGISTRATION),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "fusion_source_sha256": sha256_file(
            PROJECT_ROOT / "src" / "aiijc_puzzle" / "edge_ranker_conservative_fusion.py"
        ),
        "historical_exposure": config["historical_exposure"],
        "method_hashes": method_hashes,
        "commitment_path": str(commitment_path),
        "commitment_sha256": commitment["commitment_sha256"],
        "evaluated_arms": [arm.__dict__ for arm in arms],
        "evaluation": evaluation,
        "runtime_seconds": {
            "freeze_before_target_access": freeze_seconds,
            "target_assisted_evaluation": target_seconds,
            "total": perf_counter() - started,
        },
        "prohibited_followup": (
            "confirmation, holdout, test, and production integration"
            if args.mode == "primary" and winner is None
            else "holdout, test, and production integration"
            if args.mode == "confirmation" and winner is None
            else None
        ),
    }
    atomic_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "means": evaluation["means"],
                "gates": evaluation["gates"],
                "winner": winner,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
