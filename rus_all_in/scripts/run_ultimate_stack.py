#!/usr/bin/env python3
"""Freeze, score, and manually gate the fixed four-arm ultimate legal stack."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from run_edge_ranker_k16_tail import (
    CHECKPOINT as EDGE_CHECKPOINT,
)
from run_edge_ranker_k16_tail import (
    VIEWS,
    choose_device,
    load_verified_rgb,
)
from run_edge_ranker_k16_tail import (
    load_checkpoint as load_edge_checkpoint,
)

from aiijc_puzzle.candidate_supply import recover_layout
from aiijc_puzzle.compliant_atlas_decoder import audit_raw_permutation
from aiijc_puzzle.edge_ranker import build_inference_board, score_board
from aiijc_puzzle.edge_ranker_conservative_fusion import FusionArm, apply_conservative_fusion
from aiijc_puzzle.edge_ranker_final_tail import layout_metrics, names_digest
from aiijc_puzzle.frozen_final_evaluator import _validate_method_configs, array_digest
from aiijc_puzzle.legacy_upgrade import atomic_write_png, layout_digest, solve_buddies
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
from aiijc_puzzle.restoration_r6 import TileAwareDualNAFNet
from aiijc_puzzle.tilewise_renderer import render_tiles_independently
from aiijc_puzzle.ultimate_stack import (
    ARM_A,
    ARM_B,
    ARM_D,
    ARMS,
    quantitative_gate,
    render_arms,
    safety_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = PROJECT_ROOT / "configs" / "ultimate_stack_preregistered_v1.json"
PREREGISTRATION_SHA256 = "4857fe1e67be9c56cad06f0eb651215250e5bbd8e6e80c91a6d095a7cdd1de63"
MANIFEST = PROJECT_ROOT / "data" / "interim" / "validation_manifest.json"
INPUTS = PROJECT_ROOT / "data" / "raw" / "train" / "inputs"
TARGETS = PROJECT_ROOT / "data" / "raw" / "train" / "targets"
DUALNAF_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "restoration-r6" / "compliant-r6-medium-train256-step2000-h10.pt"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ultimate-stack"
FUSION_ARM = FusionArm("cap08-v0-c050", 8, 0, 0.5)
EDGE_BUDGET = 96
PAIR_BATCH = 1024
DUALNAF_CONDITIONING_H = 10
DUALNAF_BATCH_SIZE = 144


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "record-manual"))
    parser.add_argument("--stage", choices=("primary", "confirmation"), default="primary")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--severe-artifacts", type=int)
    parser.add_argument("--review-note", default="")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def self_hash(value: Mapping[str, Any], field: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != field}
    return hashlib.sha256(canonical_json(unsigned)).hexdigest()


def numeric_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("numeric digest requires a finite numeric array")
    header = f"{array.dtype.str}|{','.join(map(str, array.shape))}|".encode("ascii")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def write_json_exclusive(
    path: Path,
    payload: Mapping[str, Any],
    *,
    readonly: bool = False,
) -> None:
    contents = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444 if readonly else 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        if readonly:
            os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except BaseException:
        # A partial commitment is intentionally not removed: uncertain freezes
        # must fail closed rather than silently become repeatable.
        raise


def load_preregistration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("ultimate-stack preregistration hash drift")
    config = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if tuple(item["name"] for item in config["arms"]) != ARMS:
        raise RuntimeError("ultimate-stack arm roster drift")
    if config["historical_exposure"]["globally_fresh"] is not False:
        raise RuntimeError("historical calibration exposure must remain explicit")
    dependencies = config["frozen_dependencies"]
    for relative, expected in dependencies["source_sha256"].items():
        if sha256_file(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"frozen source hash drift: {relative}")
    pinned_files = {
        dependencies["edge_checkpoint"]["path"]: dependencies["edge_checkpoint"]["sha256"],
        dependencies["dualnaf_checkpoint"]["path"]: dependencies["dualnaf_checkpoint"]["sha256"],
        dependencies["rgb_config"]["path"]: dependencies["rgb_config"]["sha256"],
        dependencies["luma_config"]["path"]: dependencies["luma_config"]["sha256"],
    }
    for relative, expected in pinned_files.items():
        if sha256_file(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"frozen artifact hash drift: {relative}")
    return config


def input_roster_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(f"{record['filename']} {record['input_sha256']}" for record in records).encode(
            "utf-8"
        )
    ).hexdigest()


def load_context(
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any], tuple[Mapping[str, Any], ...]]:
    config = load_preregistration()
    if sha256_file(MANIFEST) != config["manifest"]["sha256"]:
        raise RuntimeError("manifest file hash drift")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if compute_protocol_digest(manifest) != config["manifest"]["protocol_digest"]:
        raise RuntimeError("manifest protocol digest drift")
    panel = config["panels"][stage]
    offset, count = int(panel["offset"]), int(panel["count"])
    ranked = select_manifest_records(
        manifest,
        "calibration",
        limit=offset + count,
        seed=EXPERIMENT_SUBSET_SEED,
        namespace=EXPERIMENT_SUBSET_NAMESPACE,
    )
    records = tuple(ranked[offset:])
    if len(records) != count:
        raise RuntimeError("panel record count drift")
    if names_digest(records) != panel["newline_filenames_sha256"]:
        raise RuntimeError("panel filename digest drift")
    if input_roster_digest(records) != panel["input_roster_sha256"]:
        raise RuntimeError("panel input roster digest drift")
    return config, manifest, records


def stage_root(stage: str) -> Path:
    return OUTPUT_ROOT / PREREGISTRATION_SHA256 / stage


def require_confirmation_authorized() -> None:
    primary = stage_root("primary")
    report_path = primary / "report.json"
    manual_path = primary / "manual-review.json"
    if not report_path.is_file() or not manual_path.is_file():
        raise RuntimeError("primary quantitative and manual reports are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manual = json.loads(manual_path.read_text(encoding="utf-8"))
    if report.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise RuntimeError("primary report preregistration mismatch")
    if not report.get("quantitative_gate", {}).get("passed"):
        raise RuntimeError("primary quantitative gate failed; confirmation forbidden")
    if not manual.get("passed") or manual.get("severe_new_artifacts") != 0:
        raise RuntimeError("primary manual gate failed; confirmation forbidden")


def load_dualnaf_model(device: torch.device, config: Mapping[str, Any]) -> TileAwareDualNAFNet:
    dependency = config["frozen_dependencies"]["dualnaf_checkpoint"]
    if sha256_file(DUALNAF_CHECKPOINT) != dependency["sha256"]:
        raise RuntimeError("DualNAF checkpoint hash drift")
    checkpoint = torch.load(DUALNAF_CHECKPOINT, map_location="cpu", weights_only=True)
    model_configuration = checkpoint.get("model_configuration")
    training = checkpoint.get("training_configuration")
    if not isinstance(model_configuration, Mapping) or not isinstance(training, Mapping):
        raise RuntimeError("DualNAF checkpoint contract missing")
    if model_configuration.get("architecture") != "dual_naf":
        raise RuntimeError("DualNAF architecture drift")
    if training.get("protocol_digest") != config["manifest"]["protocol_digest"]:
        raise RuntimeError("DualNAF protocol drift")
    if training.get("nlm_h") != DUALNAF_CONDITIONING_H:
        raise RuntimeError("DualNAF conditioning drift")
    model = TileAwareDualNAFNet(
        base=int(model_configuration["base"]),
        depth=int(model_configuration["depth"]),
        blocks=int(model_configuration["blocks"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def freeze(args: argparse.Namespace) -> Path:
    config, manifest, records = load_context(args.stage)
    if args.stage == "confirmation":
        require_confirmation_authorized()
    root = stage_root(args.stage)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite frozen stage: {root}")
    root.mkdir(parents=True)
    device = choose_device(args.device)
    edge_model, _ = load_edge_checkpoint(EDGE_CHECKPOINT, manifest=manifest, device=device)
    dualnaf_model = load_dualnaf_model(device, config)
    rgb_config, luma_config, method_hashes = _validate_method_configs()
    started = perf_counter()
    boards = []
    for index, record in enumerate(records, start=1):
        filename = str(record["filename"])
        dirty = load_verified_rgb(INPUTS / filename, str(record["input_sha256"]))
        input_tiles = split_tiles(dirty)
        board = build_inference_board(
            input_tiles,
            filename=filename,
            views=VIEWS,
            candidate_k=16,
        )
        learned_right, learned_down, learned_diagnostics = score_board(
            edge_model,
            board,
            device=device,
            pair_batch=PAIR_BATCH,
        )
        fused_right, fused_down, fusion_diagnostics = apply_conservative_fusion(
            board,
            learned_right,
            learned_down,
            FUSION_ARM,
        )
        bilateral_solved = solve_buddies(
            board.right_baseline,
            board.down_baseline,
            max_edges=EDGE_BUDGET,
        )
        fused_solved = solve_buddies(fused_right, fused_down, max_edges=EDGE_BUDGET)
        layouts = {
            "bilateral": np.asarray(bilateral_solved.layout, dtype=np.int64),
            "fused": np.asarray(fused_solved.layout, dtype=np.int64),
        }
        ordered_bilateral = np.ascontiguousarray(input_tiles[layouts["bilateral"]])
        ordered_fused = np.ascontiguousarray(input_tiles[layouts["fused"]])
        raw_bilateral = assemble_tiles(ordered_bilateral)
        raw_fused = assemble_tiles(ordered_fused)
        audits = {
            "bilateral": audit_raw_permutation(
                dirty,
                raw_bilateral,
                layouts["bilateral"],
                restoration_applied_after_audit=True,
            ),
            "fused": audit_raw_permutation(
                dirty,
                raw_fused,
                layouts["fused"],
                restoration_applied_after_audit=True,
            ),
        }
        if not all(audit.passed for audit in audits.values()):
            raise RuntimeError(f"strict pre-restoration permutation audit failed: {filename}")
        rendered_unordered, renderer_diagnostics = render_tiles_independently(
            dualnaf_model,
            input_tiles,
            device,
            nlm_h=DUALNAF_CONDITIONING_H,
            batch_size=DUALNAF_BATCH_SIZE,
        )
        ordered_rendered_fused = np.ascontiguousarray(rendered_unordered[layouts["fused"]])
        predictions, arm_diagnostics = render_arms(
            ordered_bilateral,
            ordered_fused,
            ordered_rendered_fused,
            rgb_config=rgb_config,
            luma_config=luma_config,
        )
        prediction_records = {}
        board_directory = root / "predictions" / Path(filename).stem
        for arm, prediction in predictions.items():
            output = board_directory / f"{arm}.png"
            prediction_records[arm] = {
                "relative_path": str(output.relative_to(root)),
                "png_sha256": atomic_write_png(output, prediction),
                "pixel_sha256": array_digest(prediction),
                "safety": arm_diagnostics[arm]["safety"],
            }
        boards.append(
            {
                "filename": filename,
                "input_sha256": record["input_sha256"],
                "layouts": {
                    name: {
                        "indices": layout.tolist(),
                        "sha256": layout_digest(layout),
                        "raw_sha256": array_digest(
                            raw_bilateral if name == "bilateral" else raw_fused
                        ),
                        "permutation_audit": audits[name].as_dict(),
                    }
                    for name, layout in layouts.items()
                },
                "score_sha256": {
                    "bilateral_right": numeric_digest(board.right_baseline),
                    "bilateral_down": numeric_digest(board.down_baseline),
                    "fused_right": numeric_digest(fused_right),
                    "fused_down": numeric_digest(fused_down),
                },
                "learned_diagnostics": learned_diagnostics,
                "fusion_diagnostics": fusion_diagnostics,
                "renderer_diagnostics": renderer_diagnostics.as_dict(),
                "arm_diagnostics": arm_diagnostics,
                "predictions": prediction_records,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "freeze-before-target-decode",
                    "stage": args.stage,
                    "done": index,
                    "total": len(records),
                    "filename": filename,
                }
            ),
            flush=True,
        )
    commitment: dict[str, Any] = {
        "schema": "aiijc-ultimate-stack-prediction-commitment-v1",
        "status": "all_predictions_frozen_before_current_stage_target_decode",
        "stage": args.stage,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "historical_exposure": config["historical_exposure"],
        "targets_decoded_during_freeze": False,
        "holdout_access": False,
        "competition_test_access": False,
        "selection": config["panels"][args.stage],
        "manifest_sha256": sha256_file(MANIFEST),
        "edge_checkpoint_sha256": sha256_file(EDGE_CHECKPOINT),
        "dualnaf_checkpoint_sha256": sha256_file(DUALNAF_CHECKPOINT),
        "method_hashes": method_hashes,
        "arm_names": list(ARMS),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                PROJECT_ROOT / "src/aiijc_puzzle/ultimate_stack.py",
                PROJECT_ROOT / "src/aiijc_puzzle/edge_ranker_conservative_fusion.py",
                PROJECT_ROOT / "src/aiijc_puzzle/dualnaf_bounded_residual.py",
                PROJECT_ROOT / "src/aiijc_puzzle/tilewise_renderer.py",
            )
        },
        "runtime_seconds": perf_counter() - started,
        "boards": boards,
    }
    commitment["commitment_self_sha256"] = self_hash(commitment, "commitment_self_sha256")
    path = root / "prediction-commitment.json"
    write_json_exclusive(path, commitment, readonly=True)
    print(
        json.dumps(
            {
                "commitment": str(path),
                "file_sha256": sha256_file(path),
                "self_sha256": commitment["commitment_self_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    return path


def load_frozen_predictions(
    root: Path,
    commitment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    frozen = []
    for board in commitment["boards"]:
        if tuple(board["predictions"]) != ARMS:
            raise RuntimeError("committed prediction roster drift")
        predictions = {}
        for arm in ARMS:
            metadata = board["predictions"][arm]
            path = root / metadata["relative_path"]
            if sha256_file(path) != metadata["png_sha256"]:
                raise RuntimeError(f"frozen PNG hash drift: {path}")
            image = load_verified_png(path, metadata["png_sha256"])
            if array_digest(image) != metadata["pixel_sha256"]:
                raise RuntimeError(f"frozen pixel hash drift: {path}")
            predictions[arm] = image
        frozen.append({"board": board, "predictions": predictions})
    return frozen


def load_verified_png(path: Path, expected_sha256: str) -> np.ndarray:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError(f"PNG file hash mismatch: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"invalid frozen PNG: {path}")
        return np.asarray(image, dtype=np.uint8)


def make_manual_sheets(
    root: Path,
    frozen: Sequence[Mapping[str, Any]],
    targets: Mapping[str, np.ndarray],
    dirty_images: Mapping[str, np.ndarray],
) -> list[str]:
    directory = root / "manual-review-sheets"
    directory.mkdir(exist_ok=True)
    columns = ("dirty", "target", *ARMS)
    paths = []
    thumb, label_width, header = 200, 150, 34
    for page_start in range(0, len(frozen), 6):
        page = frozen[page_start : page_start + 6]
        canvas = Image.new(
            "RGB",
            (label_width + thumb * len(columns), header + thumb * len(page)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for column, name in enumerate(columns):
            draw.text((label_width + column * thumb + 4, 8), name, fill="black")
        for row, item in enumerate(page):
            filename = str(item["board"]["filename"])
            y = header + row * thumb
            draw.text((4, y + 5), filename, fill="black")
            images = (
                dirty_images[filename],
                targets[filename],
                *(item["predictions"][arm] for arm in ARMS),
            )
            for column, value in enumerate(images):
                image = Image.fromarray(value).resize((thumb, thumb), Image.Resampling.NEAREST)
                canvas.paste(image, (label_width + column * thumb, y))
        path = directory / f"page-{page_start // 6 + 1}.png"
        Image.fromarray(np.asarray(canvas)).save(path, format="PNG", compress_level=6)
        paths.append(str(path.relative_to(root)))
    return paths


def score(args: argparse.Namespace) -> Path:
    config, _, records = load_context(args.stage)
    root = stage_root(args.stage)
    report_path = root / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite score report: {report_path}")
    commitment_path = root / "prediction-commitment.json"
    if not commitment_path.is_file():
        raise RuntimeError("prediction commitment required before target decode")
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    if commitment.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise RuntimeError("commitment preregistration mismatch")
    if commitment.get("commitment_self_sha256") != self_hash(commitment, "commitment_self_sha256"):
        raise RuntimeError("commitment self-hash mismatch")
    if commitment.get("selection") != config["panels"][args.stage]:
        raise RuntimeError("commitment selection mismatch")
    frozen = load_frozen_predictions(root, commitment)
    if len(frozen) != len(records):
        raise RuntimeError("frozen prediction count mismatch")

    rows = []
    targets = {}
    dirty_images = {}
    for item, record in zip(frozen, records, strict=True):
        filename = str(record["filename"])
        if item["board"]["filename"] != filename:
            raise RuntimeError("commitment and manifest order differ")
        dirty = load_verified_rgb(INPUTS / filename, str(record["input_sha256"]))
        target = load_verified_rgb(TARGETS / filename, str(record["target_sha256"]))
        dirty_images[filename] = dirty
        targets[filename] = target
        recovered = recover_layout(split_tiles(dirty), split_tiles(target))
        layout_results = {
            name: layout_metrics(
                np.asarray(item["board"]["layouts"][name]["indices"], dtype=np.int64),
                recovered,
            )
            for name in ("bilateral", "fused")
        }
        rows.append(
            {
                "filename": filename,
                "ssim": {arm: contest_ssim(target, item["predictions"][arm]) for arm in ARMS},
                "layout_metrics": layout_results,
            }
        )
    scores = {arm: np.asarray([row["ssim"][arm] for row in rows]) for arm in ARMS}
    adjacency_delta = np.asarray(
        [
            row["layout_metrics"]["fused"]["adjacency"]
            - row["layout_metrics"]["bilateral"]["adjacency"]
            for row in rows
        ]
    )
    translation_delta = np.asarray(
        [
            row["layout_metrics"]["fused"]["translation_aligned_placement"]
            - row["layout_metrics"]["bilateral"]["translation_aligned_placement"]
            for row in rows
        ]
    )
    safety = safety_summary(
        [item["board"]["predictions"][ARM_A]["safety"] for item in frozen],
        [item["board"]["predictions"][ARM_D]["safety"] for item in frozen],
    )
    gate = quantitative_gate(
        scores[ARM_A],
        scores[ARM_B],
        scores[ARM_D],
        adjacency_delta,
        translation_delta,
        safety,
    )
    sheets = make_manual_sheets(root, frozen, targets, dirty_images)
    means = {arm: float(scores[arm].mean()) for arm in ARMS}
    report = {
        "schema": "aiijc-ultimate-stack-score-v1",
        "status": "scored_after_verified_prediction_commitment",
        "stage": args.stage,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "prediction_commitment_file_sha256": sha256_file(commitment_path),
        "all_prediction_files_verified_before_first_target_decode": True,
        "historical_exposure": config["historical_exposure"],
        "holdout_access": False,
        "competition_test_access": False,
        "means": means,
        "quantitative_gate": gate,
        "manual_review": {
            "status": "pending independent visual inspection",
            "required_severe_new_artifacts": 0,
            "sheets": sheets,
        },
        "per_board": rows,
    }
    write_json_exclusive(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "means": means,
                "quantitative_gate": gate,
            },
            indent=2,
        ),
        flush=True,
    )
    return report_path


def record_manual(args: argparse.Namespace) -> Path:
    if args.severe_artifacts is None or args.severe_artifacts < 0:
        raise ValueError("record-manual requires --severe-artifacts >= 0")
    root = stage_root(args.stage)
    report_path = root / "report.json"
    if not report_path.is_file():
        raise RuntimeError("score report is required before manual review")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sheets = report.get("manual_review", {}).get("sheets")
    if not isinstance(sheets, list) or not sheets:
        raise RuntimeError("manual review sheets are missing")
    for relative in sheets:
        if not (root / relative).is_file():
            raise RuntimeError(f"manual sheet missing: {relative}")
    payload = {
        "schema": "aiijc-ultimate-stack-manual-review-v1",
        "stage": args.stage,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "score_report_sha256": sha256_file(report_path),
        "reviewed_sheets": sheets,
        "severe_new_artifacts": args.severe_artifacts,
        "passed": args.severe_artifacts == 0,
        "note": args.review_note,
        "scope": "relative artifact safety only; does not prove correct hidden layout",
    }
    path = root / "manual-review.json"
    write_json_exclusive(path, payload, readonly=True)
    print(json.dumps(payload, indent=2), flush=True)
    return path


def main() -> None:
    args = parse_args()
    if not args.run:
        raise SystemExit("dry-run only; pass --run to execute")
    if args.phase == "freeze":
        freeze(args)
    elif args.phase == "score":
        score(args)
    else:
        record_manual(args)


if __name__ == "__main__":
    main()
