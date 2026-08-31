#!/usr/bin/env python3
"""Protocol-only scoring repair for the immutable joint-native FIT64 freeze.

The v1 scoring attempt verified its freeze, materialised ``target_slots`` for
one cache, and then failed before producing a usable exact reference, any case
metric, or a score artifact because sparse candidate coverage cannot reconstruct
the full grid.  This separately signed binding never changes the construction,
layouts, controls, head, gate, or seeds.  It proves the frozen layout bytes are
identical first, then reconstructs the exact FIT cases with the already existing
synthetic generator and original organizer-train target images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aiijc_puzzle.protocol import IMAGE_SIZE, sha256_file, split_tiles
from aiijc_puzzle.structured_decoder_fit_oracle import layout_metrics, strict_layout
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_joint_native_head_arm_fit as v1  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs/joint_native_head_arm_fit_score_v2.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/joint-native-head-arm-fit/fixed-v1"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
CONFIG_SCHEMA = "aiijc-joint-native-head-arm-fit-score-binding-v2"
REPORT_SCHEMA = "aiijc-joint-native-head-arm-fit-score-v2"
SCORE_NAME = "score-v2.json"
FIT_CASE_SEED = 20260914
EXPECTED_LAYOUT_PAIR_DIGEST = "34183d3acea165f1924e772ae0cccd333461d9ae5485a9647db360c48188ad91"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("validate", "score"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    return parser.parse_args(argv)


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _path(value: str) -> Path:
    result = Path(value)
    return result.resolve() if result.is_absolute() else (PROJECT_ROOT / result).resolve()


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": _project_path(resolved), "sha256": sha256_file(resolved)}


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _load_signed_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed v2 score binding is unavailable")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise RuntimeError("v2 score binding sidecar mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA or config.get("status") != "signed-fixed-protocol":
        raise RuntimeError("v2 score binding is not signed/fixed")
    if config.get("config_path") != _project_path(resolved):
        raise RuntimeError("runtime v2 config path differs from signed path")
    if config.get("repair_only", {}).get("fit_case_seed") != FIT_CASE_SEED:
        raise RuntimeError("v2 exact reconstruction seed changed")
    if config.get("repair_only", {}).get("layout_pair_digest") != EXPECTED_LAYOUT_PAIR_DIGEST:
        raise RuntimeError("v2 frozen layout digest changed")
    immutable = config.get("immutable_v1_commitments", {})
    expected = {
        "construction_changed": False,
        "layout_or_control_changed": False,
        "head_or_order_changed": False,
        "gate_changed": False,
        "seed_changed": False,
        "tail_added": False,
        "whole_arm_selection_added": False,
    }
    if immutable != expected:
        raise RuntimeError("v2 binding changes a v1 commitment")
    for artifact in config.get("frozen_inputs", {}).values():
        target = _path(str(artifact["path"]))
        if not target.is_file() or sha256_file(target) != artifact["sha256"]:
            raise RuntimeError(f"v2 frozen input changed: {target}")
    return config, digest


def _load_and_prove_frozen_layouts(
    config: Mapping[str, Any],
    output_dir: Path,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Verify and copy every candidate/control before any exact-source access."""

    v1_config_path = _path(config["frozen_inputs"]["v1_construction_config"]["path"])
    v1_config, v1_sha = v1._load_signed_config(v1_config_path)
    if v1_sha != config["repair_only"]["v1_construction_config_sha256"]:
        raise RuntimeError("v2 binding points to another construction config")
    if config.get("frozen_inputs_gate_copy") != v1_config["gate"]:
        raise RuntimeError("v2 gate copy differs from the signed v1 construction gate")
    candidate_path, frozen_rows = v1._verify_own_freeze(v1_sha, output_dir)
    _, _, head_rows, control_path, control_rows, _ = v1._inventory(v1_config)
    prefixes = [str(row["prefix"]) for row in head_rows]
    if [row["prefix"] for row in frozen_rows] != prefixes or [
        row["prefix"] for row in control_rows
    ] != prefixes:
        raise RuntimeError("v2 sibling layout order changed")
    candidates: dict[str, np.ndarray] = {}
    controls: dict[str, np.ndarray] = {}
    digest = hashlib.sha256()
    with (
        np.load(candidate_path, allow_pickle=False) as candidate_archive,
        np.load(control_path, allow_pickle=False) as control_archive,
    ):
        for row in frozen_rows:
            prefix = str(row["prefix"])
            candidate = strict_layout(
                candidate_archive[f"{prefix}__candidate_layout"],
                grid=v1.GRID,
                name="candidate_layout",
            )
            control = strict_layout(
                control_archive[f"{prefix}__control_layout"],
                grid=v1.GRID,
                name="control_layout",
            )
            if hashlib.sha256(candidate.tobytes()).hexdigest() != row["candidate_layout_sha256"]:
                raise RuntimeError("candidate layout differs from target-free metadata")
            if hashlib.sha256(control.tobytes()).hexdigest() != row["control_layout_sha256"]:
                raise RuntimeError("control layout differs from target-free metadata")
            digest.update(prefix.encode("ascii"))
            digest.update(b"\0")
            digest.update(candidate.tobytes())
            digest.update(control.tobytes())
            candidates[prefix] = candidate.copy()
            controls[prefix] = control.copy()
    if digest.hexdigest() != config["repair_only"]["layout_pair_digest"]:
        raise RuntimeError("all-layout bit identity digest differs from v2 commitment")
    return frozen_rows, head_rows, candidates, controls


def _manifest_records(
    config: Mapping[str, Any],
    manifest_path: Path,
) -> tuple[dict[str, Any], ...]:
    if _record(manifest_path) != config["frozen_inputs"]["manifest"]:
        raise RuntimeError("runtime manifest differs from v2 binding")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {str(row["filename"]): row for row in manifest["splits"]["train"]}
    signed = tuple(config["repair_only"]["source_roster"])
    names = [str(row["filename"]) for row in signed]
    if len(names) != 32 or len(set(names)) != 32:
        raise RuntimeError("v2 FIT source roster must contain 32 unique names")
    records: list[dict[str, Any]] = []
    for expected in signed:
        observed = by_name.get(str(expected["filename"]))
        if observed is None or observed.get("target_sha256") != expected["target_sha256"]:
            raise RuntimeError("v2 FIT source target hash declaration changed")
        records.append(observed)
    return tuple(records)


def _load_exact_boards(
    records: Sequence[Mapping[str, Any]],
    targets_dir: Path,
) -> dict[str, np.ndarray]:
    """First organizer-target byte access in the v2 runner."""

    boards: dict[str, np.ndarray] = {}
    for record in records:
        filename = str(record["filename"])
        path = targets_dir / filename
        if sha256_file(path) != record["target_sha256"]:
            raise RuntimeError(f"organizer target hash mismatch: {filename}")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(f"organizer target shape/mode changed: {filename}")
            board = np.asarray(image, dtype=np.uint8)
        boards[filename] = np.ascontiguousarray(split_tiles(board))
    return boards


def run_validate(
    config: Mapping[str, Any],
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    frozen_rows, head_rows, candidates, controls = _load_and_prove_frozen_layouts(
        config, output_dir
    )
    records = _manifest_records(config, manifest_path)
    return {
        "schema": "aiijc-joint-native-head-arm-fit-score-v2-validation-v1",
        "status": "ready-layouts-bit-identical-before-exact-reconstruction",
        "case_count": len(frozen_rows),
        "head_case_count": len(head_rows),
        "source_count": len(records),
        "candidate_layout_count": len(candidates),
        "control_layout_count": len(controls),
        "layout_pair_digest": config["repair_only"]["layout_pair_digest"],
        "make_exact_synthetic_case_called": False,
        "organizer_target_images_opened": False,
    }


def run_score(
    config: Mapping[str, Any],
    config_sha: str,
    output_dir: Path,
    manifest_path: Path,
    targets_dir: Path,
) -> dict[str, Any]:
    # This complete target-free proof must finish before exact target image access.
    frozen_rows, head_rows, candidates, controls = _load_and_prove_frozen_layouts(
        config, output_dir
    )
    records = _manifest_records(config, manifest_path)
    boards = _load_exact_boards(records, targets_dir)
    result_rows: list[dict[str, Any]] = []
    for frozen, _head_row in zip(frozen_rows, head_rows, strict=True):
        prefix = str(frozen["prefix"])
        filename = str(frozen["source_filename"])
        item, exact = make_exact_synthetic_case(
            boards[filename],
            source_filename=filename,
            draw_index=int(frozen["draw_index"]),
            seed=FIT_CASE_SEED,
        )
        observed = {
            "case_id": item.case_id,
            "source_filename": item.source_filename,
            "draw_index": item.draw_index,
            "dirty_sha256": hashlib.sha256(item.tiles.tobytes()).hexdigest(),
        }
        if any(observed[key] != frozen[key] for key in observed):
            raise RuntimeError("v2 exact synthetic reconstruction differs from frozen case")
        reference = exact.tile_at_position
        candidate_metrics = layout_metrics(candidates[prefix], reference, grid=v1.GRID)
        control_metrics = layout_metrics(controls[prefix], reference, grid=v1.GRID)
        benefit = {
            "satisfied_pairs": (
                candidate_metrics.satisfied_pairs - control_metrics.satisfied_pairs
            ),
            "exact_tiles": candidate_metrics.exact_tiles - control_metrics.exact_tiles,
            "manhattan": (
                control_metrics.mean_absolute_manhattan - candidate_metrics.mean_absolute_manhattan
            ),
            "radius2_recall": (candidate_metrics.radius2_recall - control_metrics.radius2_recall),
        }
        result_rows.append(
            {
                "prefix": prefix,
                **observed,
                "control": control_metrics.as_dict(),
                "candidate": candidate_metrics.as_dict(),
                "benefit_delta": benefit,
            }
        )
    metric_names = ("satisfied_pairs", "exact_tiles", "manhattan", "radius2_recall")
    robust = {
        name: v1._robust_metric(result_rows, name, metric_index=index)
        for index, name in enumerate(metric_names)
    }
    metric_fields = (
        "satisfied_pairs",
        "exact_tiles",
        "mean_absolute_manhattan",
        "radius2_recall",
    )
    control_mean = {
        name: float(np.mean([row["control"][name] for row in result_rows]))
        for name in metric_fields
    }
    candidate_mean = {
        name: float(np.mean([row["candidate"][name] for row in result_rows]))
        for name in metric_fields
    }
    checks = {
        "pair_mean_strictly_positive": robust["satisfied_pairs"]["case_distribution"]["mean"] > 0.0,
        "pair_source_bootstrap_95pct_lower_nonnegative": robust["satisfied_pairs"][
            "source_bootstrap_mean_95pct_ci"
        ][0]
        >= 0.0,
        "exact_mean_nonnegative": robust["exact_tiles"]["case_distribution"]["mean"] >= 0.0,
        "manhattan_benefit_mean_nonnegative": robust["manhattan"]["case_distribution"]["mean"]
        >= 0.0,
        "radius2_mean_nonnegative": robust["radius2_recall"]["case_distribution"]["mean"] >= 0.0,
    }
    if set(checks) != set(config["frozen_inputs_gate_copy"]):
        raise RuntimeError("v2 scoring checks differ from the immutable v1 gate copy")
    report = {
        "schema": REPORT_SCHEMA,
        "status": (
            "pass-new-joint-native-arm" if all(checks.values()) else "fail-stop-do-not-promote"
        ),
        "config_sha256": config_sha,
        "case_count": 64,
        "source_count": 32,
        "draws_per_source": 2,
        "aggregate": {
            "control_mean": control_mean,
            "candidate_mean": candidate_mean,
        },
        "robust_metrics": robust,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "raw_arm_comparator": {
            "available": False,
            "reason": config["repair_only"]["raw_arm_unavailable_reason"],
        },
        "rows": result_rows,
        "protocol_repair": {
            "v1_score_artifact_created": False,
            "v1_partial_target_slots_read_count": 1,
            "v1_usable_exact_reference_created": False,
            "v1_case_metric_count": 0,
            "v1_failure_reason": config["repair_only"]["v1_failure_reason"],
            "v2_signed_before_first_make_exact_synthetic_case": True,
            "frozen_layout_pair_digest_verified_before_target_images": (
                config["repair_only"]["layout_pair_digest"]
            ),
            "construction_layout_control_head_gate_seed_or_tail_changed": False,
        },
        "freeze": {
            "archive_sha256": config["frozen_inputs"]["candidate_archive"]["sha256"],
            "metadata_sha256": config["frozen_inputs"]["candidate_metadata"]["sha256"],
            "pre_score_sha256": config["frozen_inputs"]["candidate_pre_score_freeze"]["sha256"],
            "verified_before_exact_reconstruction": True,
        },
        "all_outputs_strict_576_original_upright_permutations": True,
        "fit_only": True,
        "dev_local_terminal_test_or_competition_accessed": False,
        "model_training_or_inference_run": False,
        "whole_arm_reselection_run": False,
        "weco_logged": False,
    }
    _write_json_exclusive(output_dir.resolve() / SCORE_NAME, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_sha = _load_signed_config(args.config)
    if args.mode == "validate":
        report = run_validate(config, args.output_dir, args.manifest)
    else:
        report = run_score(
            config,
            config_sha,
            args.output_dir,
            args.manifest,
            args.targets,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if args.mode == "score" and not report["gate"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
