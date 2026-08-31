#!/usr/bin/env python3
"""Strictly freeze FIT heads while never materialising ``target_slots``.

The runner is roster-size agnostic and is intentionally separate from the
historical v2-bound freezer.  It verifies each complete NPZ member inventory
and whole-file SHA first, then indexes only the explicit inference allowlist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.joint_reciprocal_tri_emitter_verifier import (
    RECIPROCAL_HEAD_FRACTION,
)
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.tri_emitter_edge_verifier import (
    AUXILIARY_DIM,
    DINO_PROJECTION_DIM,
    EMITTERS,
    TOP_K,
)

try:
    from scripts import run_joint_reciprocal_scale256_real as scale
    from scripts import run_joint_reciprocal_tri_emitter_real as base
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import run_joint_reciprocal_scale256_real as scale
    import run_joint_reciprocal_tri_emitter_real as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/joint_reciprocal_scale256_real_preregistered_v1.json"
)
DEFAULT_EXPERIMENT = scale.DEFAULT_EXPERIMENT_DIR
TARGET_FREE_CACHE_KEYS = frozenset(
    {
        "raw_sides",
        "dino_sides",
        "candidates",
        "valid",
        "auxiliary",
        "raw_baseline",
        "emitter_topk",
    }
)
LABEL_CACHE_KEYS = frozenset({"target_slots"})
FULL_CACHE_MEMBER_NAMES = TARGET_FREE_CACHE_KEYS | LABEL_CACHE_KEYS
HEAD_METADATA_SCHEMA = "aiijc-joint-reciprocal-target-free-fit-heads-v1"
HEAD_FREEZE_SCHEMA = "aiijc-joint-reciprocal-fit-heads-pre-score-freeze-v1"
LOADER_SCHEMA = "aiijc-joint-reciprocal-strict-target-free-fit-cache-loader-v2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.device == "mps" and not args.allow_nondeterministic_mps:
        raise ValueError("MPS inference requires explicit nondeterminism consent")
    if args.device == "cpu" and args.allow_nondeterministic_mps:
        raise ValueError("MPS consent is incompatible with CPU inference")
    if not args.experiment_dir.resolve().is_dir():
        raise FileNotFoundError("completed FIT experiment directory is missing")


def _project_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": _project_label(resolved), "sha256": sha256_file(resolved)}


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def validate_target_free_fit_cache_arrays(
    arrays: Mapping[str, np.ndarray], *, expected_tile_count: int = 576
) -> None:
    """Fail closed on every inference-visible member, shape and identity."""

    if set(arrays) != TARGET_FREE_CACHE_KEYS:
        missing = sorted(TARGET_FREE_CACHE_KEYS - set(arrays))
        extra = sorted(set(arrays) - TARGET_FREE_CACHE_KEYS)
        raise RuntimeError(
            f"target-free FIT cache keys changed; missing={missing}, extra={extra}"
        )
    if expected_tile_count < 2:
        raise ValueError("expected_tile_count must be at least two")
    count = expected_tile_count
    candidates = np.asarray(arrays["candidates"])
    valid = np.asarray(arrays["valid"])
    if candidates.ndim != 3 or candidates.shape[:2] != (2, count):
        raise RuntimeError("FIT cache candidates must be 2 x N x K")
    width = candidates.shape[2]
    expected_shapes = {
        "raw_sides": (4, count, 20, 6),
        "dino_sides": (4, count, 14, DINO_PROJECTION_DIM),
        "valid": (2, count, width),
        "auxiliary": (2, count, width, AUXILIARY_DIM),
        "raw_baseline": (2, count, width),
        "emitter_topk": (len(EMITTERS), 2, count, min(TOP_K, count - 1)),
    }
    for key, shape in expected_shapes.items():
        if np.asarray(arrays[key]).shape != shape:
            raise RuntimeError(f"target-free FIT cache {key} shape changed")
    if candidates.dtype not in (np.int32, np.int64):
        raise RuntimeError("target-free FIT candidates must be int32/int64")
    if valid.dtype != np.bool_:
        raise RuntimeError("target-free FIT valid mask must be boolean")
    emitter = np.asarray(arrays["emitter_topk"])
    if emitter.dtype not in (np.int32, np.int64):
        raise RuntimeError("target-free FIT emitter identities must be int32/int64")
    for key in ("raw_sides", "dino_sides", "auxiliary", "raw_baseline"):
        value = np.asarray(arrays[key])
        if value.dtype not in (np.float16, np.float32) or not np.isfinite(value).all():
            raise RuntimeError(
                f"target-free FIT cache {key} must be finite float16/float32"
            )
    if np.any(valid & ((candidates < 0) | (candidates >= count))):
        raise RuntimeError("target-free FIT candidate identity is out of range")
    if np.any((emitter < 0) | (emitter >= count)):
        raise RuntimeError("target-free FIT emitter identity is out of range")
    for axis in range(2):
        for source in range(count):
            row = candidates[axis, source, valid[axis, source]]
            if len(row) != len(np.unique(row)):
                raise RuntimeError("target-free FIT cache has duplicate candidates")
            if np.any(row == source):
                raise RuntimeError("target-free FIT cache contains a self candidate")


def load_target_free_fit_cache(
    path: Path, *, expected_sha256: str, expected_tile_count: int = 576
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Hash and inventory one NPZ, then decode only non-label members."""

    resolved = path.resolve()
    if not resolved.is_file() or sha256_file(resolved) != expected_sha256:
        raise RuntimeError(f"immutable FIT cache bytes changed: {resolved}")
    materialised: list[str] = []
    with np.load(resolved, allow_pickle=False) as archive:
        if set(archive.files) != FULL_CACHE_MEMBER_NAMES:
            missing = sorted(FULL_CACHE_MEMBER_NAMES - set(archive.files))
            extra = sorted(set(archive.files) - FULL_CACHE_MEMBER_NAMES)
            raise RuntimeError(
                "FIT cache member-name inventory changed; "
                f"missing={missing}, extra={extra}"
            )
        arrays: dict[str, np.ndarray] = {}
        for key in sorted(TARGET_FREE_CACHE_KEYS):
            arrays[key] = np.ascontiguousarray(archive[key])
            materialised.append(key)
    validate_target_free_fit_cache_arrays(
        arrays, expected_tile_count=expected_tile_count
    )
    if set(materialised) & LABEL_CACHE_KEYS:
        raise RuntimeError("label cache member was materialised")
    return arrays, tuple(materialised)


def _target_free_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(TARGET_FREE_CACHE_KEYS):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _verify_completed_fit(
    experiment: Path, *, config_sha256: str, expected_steps: int
) -> tuple[Path, Path, dict[str, Any]]:
    report_path = experiment / base.FIT_REPORT
    endpoint = experiment / base.FIT_ENDPOINT
    if not report_path.is_file() or not endpoint.is_file():
        raise FileNotFoundError("FIT endpoint/report is not complete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "aiijc-joint-reciprocal-real-fit-report-v1":
        raise RuntimeError("FIT report schema changed")
    if report.get("status") != "complete-single-endpoint-ready-for-target-free-dev-freeze":
        raise RuntimeError("FIT did not reach its complete endpoint state")
    if report.get("config_sha256") != config_sha256:
        raise RuntimeError("FIT report belongs to another config")
    if report.get("endpoint") != _record(endpoint):
        raise RuntimeError("FIT endpoint differs from its report")
    training = report.get("training", {})
    if training.get("from_scratch") is not True or training.get("steps") != expected_steps:
        raise RuntimeError("FIT training contract changed")
    return endpoint, report_path, report


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    config, config_sha, rosters = scale.load_signed_runtime_config(args.config)
    _fit_cache_report, rows = base._fit_cache_manifest(config, rosters)
    if len(rows) != len(rosters["fit"]) * len(
        config["source_protocol"]["fit_draw_indices"]
    ):
        raise RuntimeError("strict target-free FIT case count changed")
    experiment = args.experiment_dir.resolve()
    endpoint, fit_report_path, _fit_report = _verify_completed_fit(
        experiment,
        config_sha256=config_sha,
        expected_steps=int(config["training"]["optimizer_updates"]),
    )
    device = base._device(args.device)
    torch.use_deterministic_algorithms(
        True, warn_only=bool(args.allow_nondeterministic_mps)
    )
    model = base._load_real_checkpoint(
        endpoint, config, config_sha, device=device
    )

    all_arrays: dict[str, np.ndarray] = {}
    metadata_rows: list[dict[str, Any]] = []
    materialised_contract: tuple[str, ...] | None = None
    for index, row in enumerate(rows):
        values, materialised = load_target_free_fit_cache(
            base._project_path(row["path"]),
            expected_sha256=str(row["sha256"]),
        )
        if materialised_contract is None:
            materialised_contract = materialised
        elif materialised != materialised_contract:
            raise RuntimeError("target-free cache materialisation roster changed")
        frozen = base._freeze_fit_head_case(model, values, device=device)
        prefix = f"case_{index:04d}"
        all_arrays.update(
            {f"{prefix}__{key}": value for key, value in frozen.items()}
        )
        metadata_rows.append(
            {
                "prefix": prefix,
                "case_id": row["case_id"],
                "source_filename": row["source_filename"],
                "draw_index": int(row["draw_index"]),
                "dirty_sha256": row["dirty_sha256"],
                "fit_cache": {"path": row["path"], "sha256": row["sha256"]},
                "union_identity_digest": bytes(
                    frozen["union_identity_digest_ascii"]
                ).decode("ascii"),
                "target_free_input_digest": _target_free_digest(values),
            }
        )
        print(
            json.dumps(
                {
                    "event": "strict_target_free_joint_fit_head",
                    "case": index + 1,
                    "count": len(rows),
                    "source": row["source_filename"],
                    "draw": row["draw_index"],
                }
            ),
            flush=True,
        )

    archive = experiment / base.FIT_HEAD_ARCHIVE
    metadata = experiment / base.FIT_HEAD_METADATA
    freeze = experiment / base.FIT_HEAD_PRE_SCORE_FREEZE
    _write_npz_exclusive(archive, all_arrays)
    _write_json_exclusive(
        metadata,
        {
            "schema": HEAD_METADATA_SCHEMA,
            "config_sha256": config_sha,
            "contains_target_slots_truth_or_reference_labels": False,
            "contains_pixels": False,
            "tile_id_space": "immutable-shuffled-tile-bag-identity",
            "candidate_identities_immutable": True,
            "fixed_fraction_per_axis_per_board": RECIPROCAL_HEAD_FRACTION,
            "expected_requested_count_for_576_tiles": 29,
            "strict_target_free_loader_schema": LOADER_SCHEMA,
            "npz_member_names_inspected_only": sorted(FULL_CACHE_MEMBER_NAMES),
            "npz_members_materialised": list(materialised_contract or ()),
            "label_members_materialised": [],
            "rows": metadata_rows,
        },
    )
    _write_json_exclusive(
        freeze,
        {
            "schema": HEAD_FREEZE_SCHEMA,
            "created_before_fit_head_label_scoring": True,
            "contains_target_slots_truth_or_reference_labels": False,
            "label_cache_members_materialised": False,
            "strict_target_free_loader_schema": LOADER_SCHEMA,
            "config_sha256": config_sha,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "fit_endpoint": _record(endpoint),
                "fit_report": _record(fit_report_path),
                "config": _record(args.config),
                "runner": _record(Path(__file__)),
                "fit_wrapper": _record(Path(scale.__file__).resolve()),
                "fit_runner": _record(Path(base.__file__).resolve()),
                "module": _record(
                    PROJECT_ROOT
                    / "src/aiijc_puzzle/joint_reciprocal_tri_emitter_verifier.py"
                ),
            },
        },
    )
    return {
        "schema": (
            "aiijc-joint-reciprocal-strict-target-free-fit-head-freeze-result-v2"
        ),
        "status": "target-free-fit-heads-frozen-label-payloads-never-materialised",
        "case_count": len(rows),
        "archive": _record(archive),
        "metadata": _record(metadata),
        "pre_score_freeze": _record(freeze),
        "fit_head_labels_scored": False,
        "fit_cache_label_members_materialised": False,
        "dev_pixels_or_labels_opened": False,
        "terminal16_opened": False,
        "competition_test_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
