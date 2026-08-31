#!/usr/bin/env python3
"""Freeze and score one CPU-only Wiener candidate emitter on FIT32 x draw2.

``freeze-fit`` never opens a label archive and writes only candidate identities.
``score-fit`` is a separate, hash-bound stage that refuses to run without a
signed binding to the completed pre-label freeze and the physically separate
historical FIT label archive.  No DEV, local, terminal, competition-test or
submission mode exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.wiener_matcher_view import (
    WINDOW,
    fixed_top32,
    fixed_wiener_directional_scores,
)

if __package__:
    from scripts import run_guided_fourth_emitter_fit_capacity as base
else:
    import run_guided_fourth_emitter_fit_capacity as base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/wiener_candidate_emitter_fit_preregistered_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/wiener-candidate-emitter/fit32-draw2-v1"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"

CONFIG_SCHEMA = "aiijc-wiener-candidate-emitter-fit-protocol-v1"
CONFIG_STATUS = "signed-target-free-freeze-only"
BINDING_SCHEMA = "aiijc-wiener-candidate-emitter-fit-score-binding-v1"
BINDING_STATUS = "signed-post-freeze-fit-coverage-only"
METADATA_SCHEMA = "aiijc-wiener-candidate-emitter-target-free-cache-v1"
FREEZE_SCHEMA = "aiijc-wiener-candidate-emitter-pre-label-freeze-v1"
REPORT_SCHEMA = "aiijc-wiener-candidate-emitter-fit-capacity-report-v1"
ARCHIVE_KEYS = frozenset({"emitter_topk", "legacy_identity_digest_ascii"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("freeze-fit", "score-fit"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    return parser.parse_args(argv)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        label = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        label = str(resolved)
    return {"path": label, "sha256": sha256_file(resolved)}


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _load_signed_json(
    path: Path,
    *,
    schema: str,
    status: str,
) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"signed JSON or sidecar is missing: {resolved}")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise RuntimeError(f"signed JSON sidecar mismatch: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") != schema or payload.get("status") != status:
        raise RuntimeError(f"signed JSON schema/status mismatch: {resolved}")
    return payload, digest


def _verify_frozen_artifact(artifact: Mapping[str, str]) -> Path:
    path = _project_path(artifact["path"])
    if not path.is_file() or sha256_file(path) != artifact["sha256"]:
        raise RuntimeError(f"frozen artifact changed: {path}")
    return path


def _load_config(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    config, digest = _load_signed_json(
        path,
        schema=CONFIG_SCHEMA,
        status=CONFIG_STATUS,
    )
    if config.get("labels_opened_by_freeze_stage") is not False:
        raise RuntimeError("freeze config must forbid labels")
    if config.get("dev_local_terminal_test_or_submission_modes") != []:
        raise RuntimeError("non-FIT modes must be absent")
    recipe = config.get("fixed_recipe", {})
    if recipe != {
        "name": "per_tile_per_channel_local_wiener",
        "window": 3,
        "padding": "reflect",
        "noise_variance": "mean_local_variance_per_tile_channel",
        "top_k": 32,
        "matcher_view_only": True,
    } or WINDOW != 3:
        raise RuntimeError("fixed Wiener recipe changed")
    for artifact in config.get("frozen_inputs", {}).values():
        _verify_frozen_artifact(artifact)
    base_path = _verify_frozen_artifact(config["frozen_inputs"]["guided_fit_config"])
    base_config, base_sha = base._load_config(base_path)
    if base_sha != config["frozen_inputs"]["guided_fit_config"]["sha256"]:
        raise RuntimeError("base FIT config hash changed")
    source = config.get("source_protocol", {})
    if source != {
        "fit_digest": base_config["source_protocol"]["fit_digest"],
        "fit_draw_indices": base_config["source_protocol"]["fit_draw_indices"],
        "case_seed": base_config["source_protocol"]["case_seed"],
        "case_count": 64,
    }:
        raise RuntimeError("Wiener FIT roster differs from immutable base FIT roster")
    return config, digest, base_config


def _load_base_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _verify_frozen_artifact(config["frozen_inputs"]["guided_target_free_metadata"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != base.METADATA_SCHEMA or payload.get("case_count") != 64:
        raise RuntimeError("base target-free metadata schema changed")
    if payload.get("contains_target_slots_truth_or_reference_labels") is not False:
        raise RuntimeError("base target-free metadata unexpectedly contains labels")
    return payload


def _archive_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != ARCHIVE_KEYS:
            raise RuntimeError("Wiener target-free archive keys changed")
        arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    topk = arrays["emitter_topk"]
    if topk.shape != (4, 2, 576, 32) or topk.dtype not in (np.int32, np.int64):
        raise RuntimeError("Wiener target-free emitter roster changed")
    if np.any((topk < 0) | (topk >= 576)):
        raise RuntimeError("Wiener target-free archive contains invalid identities")
    for emitter in range(4):
        for axis in range(2):
            for source in range(576):
                row = topk[emitter, axis, source]
                if source in row or len(np.unique(row)) != 32:
                    raise RuntimeError("Wiener target-free top-k row is invalid")
    return arrays


def _topk_digest(topk: np.ndarray) -> str:
    value = np.ascontiguousarray(topk, dtype=np.int32)
    return hashlib.sha256(value.tobytes()).hexdigest()


def run_freeze_fit(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    base_rows = base._legacy_rows(base_config)
    base_metadata = _load_base_metadata(config)
    names = tuple(base_config["source_protocol"]["fit_filenames"])
    records = base._manifest_records(args.manifest, names)
    cache_dir = output / "target-free-cache"
    cache_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    row_index = 0
    for source_index, record in enumerate(records):
        clean = base._load_clean_tiles(record, args.targets)
        for draw_index in base.FIT_DRAWS:
            legacy_row = base_rows[row_index]
            base_row = base_metadata["rows"][row_index]
            case_id, dirty = base.make_target_free_fit_case(
                clean,
                source_filename=str(record["filename"]),
                draw_index=draw_index,
            )
            dirty_sha = base._array_sha256(dirty)
            identity = (str(record["filename"]), draw_index, case_id, dirty_sha)
            expected = (
                base_row["source_filename"],
                int(base_row["draw_index"]),
                base_row["case_id"],
                base_row["dirty_sha256"],
            )
            if identity != expected:
                raise RuntimeError("target-free Wiener replay differs from frozen FIT input")
            legacy_path = _project_path(legacy_row["path"])
            if sha256_file(legacy_path) != legacy_row["sha256"]:
                raise RuntimeError("immutable legacy cache changed")
            legacy = base._load_legacy_pool_target_free(legacy_path)
            case_started = perf_counter()
            wiener_topk = fixed_top32(fixed_wiener_directional_scores(dirty))
            emitter_topk = np.concatenate(
                (legacy.emitter_topk.astype(np.int32), wiener_topk[None]), axis=0
            )
            path = cache_dir / f"source_{source_index:02d}_draw_{draw_index}.npz"
            _write_npz_exclusive(
                path,
                {
                    "emitter_topk": emitter_topk.astype(np.int32),
                    "legacy_identity_digest_ascii": np.frombuffer(
                        legacy.identity_digest.encode(), dtype=np.uint8
                    ),
                },
            )
            _archive_arrays(path)
            rows.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_file(path),
                    "source_filename": record["filename"],
                    "draw_index": draw_index,
                    "case_id": case_id,
                    "dirty_sha256": dirty_sha,
                    "legacy_cache": {"path": legacy_row["path"], "sha256": legacy_row["sha256"]},
                    "legacy_identity_digest": legacy.identity_digest,
                    "emitter_topk_digest": _topk_digest(emitter_topk),
                    "wiener_cpu_seconds": perf_counter() - case_started,
                }
            )
            row_index += 1
            print(
                json.dumps(
                    {"event": "wiener_target_free", "case": row_index, "count": 64}
                ),
                flush=True,
            )
    metadata_path = output / "target-free-cache.json"
    _write_json_exclusive(
        metadata_path,
        {
            "schema": METADATA_SCHEMA,
            "config_sha256": config_sha,
            "created_before_fit_label_archive_opened": True,
            "contains_target_slots_truth_or_reference_labels": False,
            "contains_dirty_clean_or_output_pixels": False,
            "candidate_identities_target_blind": True,
            "case_count": len(rows),
            "rows": rows,
        },
    )
    freeze_path = output / "pre-label-freeze.json"
    freeze = {
        "schema": FREEZE_SCHEMA,
        "status": "target-free-identities-frozen-score-stage-not-run",
        "config_sha256": config_sha,
        "created_before_fit_label_archive_opened": True,
        "contains_target_slots_truth_or_reference_labels": False,
        "case_count": len(rows),
        "artifacts": {
            "config": _record(args.config),
            "metadata": _record(metadata_path),
            "runner": _record(Path(__file__)),
            "module": _record(PROJECT_ROOT / "src/aiijc_puzzle/wiener_matcher_view.py"),
        },
        "case_files": [{"path": row["path"], "sha256": row["sha256"]} for row in rows],
        "runtime_seconds": perf_counter() - started,
        "dev_local_terminal_test_or_submission_accessed": False,
    }
    _write_json_exclusive(freeze_path, freeze)
    return freeze


def _verify_freeze(output: Path, config_sha: str) -> tuple[dict[str, Any], Path]:
    freeze_path = output / "pre-label-freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("config_sha256") != config_sha:
        raise RuntimeError("pre-label freeze does not match the signed experiment")
    if freeze.get("created_before_fit_label_archive_opened") is not True:
        raise RuntimeError("target-free identities were not frozen before labels")
    if freeze.get("contains_target_slots_truth_or_reference_labels") is not False:
        raise RuntimeError("pre-label freeze unexpectedly contains labels")
    metadata_path = output / "target-free-cache.json"
    if sha256_file(metadata_path) != freeze["artifacts"]["metadata"]["sha256"]:
        raise RuntimeError("target-free metadata changed after freeze")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != METADATA_SCHEMA or metadata.get("case_count") != 64:
        raise RuntimeError("target-free metadata contract changed")
    for row, frozen in zip(metadata["rows"], freeze["case_files"], strict=True):
        path = _project_path(row["path"])
        if row["path"] != frozen["path"] or row["sha256"] != frozen["sha256"]:
            raise RuntimeError("target-free cache roster changed")
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError("target-free cache changed after freeze")
    return metadata, freeze_path


def _coverage(topk: np.ndarray, truth: np.ndarray) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for axis, name in enumerate(("right", "down")):
        eligible = truth[axis] >= 0
        raw = np.any(topk[0, axis] == truth[axis, :, None], axis=1) & eligible
        legacy = np.any(topk[:3, axis] == truth[axis][None, :, None], axis=(0, 2)) & eligible
        wiener = np.any(topk[3, axis] == truth[axis, :, None], axis=1) & eligible
        extended = np.any(topk[:, axis] == truth[axis][None, :, None], axis=(0, 2)) & eligible
        result[name] = {
            "eligible": int(np.count_nonzero(eligible)),
            "raw_top32": int(np.count_nonzero(raw)),
            "wiener_top32": int(np.count_nonzero(wiener)),
            "legacy_union": int(np.count_nonzero(legacy)),
            "extended_union": int(np.count_nonzero(extended)),
            "wiener_unique_recovered": int(np.count_nonzero(extended & ~legacy)),
        }
    return result


def _bootstrap_source_mean(values: np.ndarray, *, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(20_000, len(values)))
    samples = values[draws].mean(axis=1)
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def run_score_fit(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
) -> dict[str, Any]:
    if args.binding is None:
        raise RuntimeError("score-fit requires a separately signed --binding")
    binding, binding_sha = _load_signed_json(
        args.binding,
        schema=BINDING_SCHEMA,
        status=BINDING_STATUS,
    )
    output = args.output_dir.resolve()
    metadata, freeze_path = _verify_freeze(output, config_sha)
    if binding.get("experiment_config_sha256") != config_sha:
        raise RuntimeError("score binding targets another experiment config")
    if binding.get("pre_label_freeze_sha256") != sha256_file(freeze_path):
        raise RuntimeError("score binding targets another pre-label freeze")
    if binding.get("scope") != {
        "fit_coverage_counts_only": True,
        "model_fit": False,
        "parameter_selection": False,
        "dev_local_terminal_test_submission": False,
    }:
        raise RuntimeError("score binding scope changed")
    labels_path = _verify_frozen_artifact(binding["separated_fit_labels_metadata"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if labels.get("schema") != base.LABEL_SCHEMA or labels.get("case_count") != 64:
        raise RuntimeError("separated FIT labels metadata changed")
    if labels.get("labels_physically_separate_from_target_free_cache") is not True:
        raise RuntimeError("FIT labels are not physically separate")
    totals = {
        axis: {
            key: 0
            for key in (
                "eligible",
                "raw_top32",
                "wiener_top32",
                "legacy_union",
                "extended_union",
                "wiener_unique_recovered",
            )
        }
        for axis in ("right", "down")
    }
    per_case: list[dict[str, Any]] = []
    source_gains = np.zeros(32, dtype=np.int32)
    for index, (row, label_row) in enumerate(zip(metadata["rows"], labels["rows"], strict=True)):
        identity = (row["source_filename"], int(row["draw_index"]), row["case_id"])
        label_identity = (
            label_row["source_filename"],
            int(label_row["draw_index"]),
            label_row["case_id"],
        )
        if identity != label_identity:
            raise RuntimeError("target-free and label rosters are misaligned")
        arrays = _archive_arrays(_project_path(row["path"]))
        label_file = _project_path(label_row["path"])
        if sha256_file(label_file) != label_row["sha256"]:
            raise RuntimeError("separated FIT label file changed")
        with np.load(label_file, allow_pickle=False) as archive:
            if set(archive.files) != {"truth_by_source", "target_slots"}:
                raise RuntimeError("separated FIT label keys changed")
            truth = np.ascontiguousarray(archive["truth_by_source"], dtype=np.int32)
        if truth.shape != (2, 576):
            raise RuntimeError("FIT truth shape changed")
        coverage = _coverage(arrays["emitter_topk"], truth)
        gain = sum(coverage[axis]["wiener_unique_recovered"] for axis in coverage)
        source_gains[index // 2] += gain
        for axis in totals:
            for key, value in coverage[axis].items():
                totals[axis][key] += value
        per_case.append(
            {
                **{
                    key: value
                    for key, value in zip(
                        ("source_filename", "draw_index", "case_id"),
                        identity,
                        strict=True,
                    )
                },
                "coverage": coverage,
            }
        )
    pooled = {key: totals["right"][key] + totals["down"][key] for key in totals["right"]}
    rates = {
        key: pooled[key] / pooled["eligible"]
        for key in ("raw_top32", "wiener_top32", "legacy_union", "extended_union")
    }
    gain = rates["extended_union"] - rates["legacy_union"]
    gate = config["coverage_gate"]
    positive_sources = int(np.count_nonzero(source_gains > 0))
    passed = bool(
        gain >= float(gate["minimum_absolute_gain"])
        and positive_sources >= int(gate["minimum_positive_source_groups"])
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "retain-as-candidate-supply" if passed else "stop-as-tested",
        "claim": "FIT-only candidate coverage; not real evaluation or promotion evidence",
        "config_sha256": config_sha,
        "binding_sha256": binding_sha,
        "pre_label_freeze_verified_before_labels": True,
        "coverage_counts": {**totals, "pooled": pooled},
        "pooled_coverage_rates": rates,
        "legacy_to_extended": {
            "additional_true_neighbours": pooled["wiener_unique_recovered"],
            "absolute_rate_gain": gain,
            "positive_source_groups": positive_sources,
            "source_group_count": 32,
            "mean_unique_per_case": float(source_gains.sum() / 64),
            "source_group_bootstrap_ci95_unique_per_case": [
                value / 2
                for value in _bootstrap_source_mean(
                    source_gains.astype(np.float64), seed=20260918
                )
            ],
        },
        "gate": {**gate, "passed": passed},
        "legality": {
            "organizer_train_fit_only": True,
            "wiener_pixels_matcher_only": True,
            "original_upright_tiles_unchanged": True,
            "labels_physically_separate": True,
            "dev_local_terminal_test_or_submission_accessed": False,
        },
        "artifacts": {
            "config": _record(args.config),
            "binding": _record(args.binding),
            "target_free_metadata": _record(output / "target-free-cache.json"),
            "pre_label_freeze": _record(freeze_path),
            "separated_fit_labels_metadata": _record(labels_path),
            "runner": _record(Path(__file__)),
            "module": _record(PROJECT_ROOT / "src/aiijc_puzzle/wiener_matcher_view.py"),
        },
        "cases": per_case,
    }
    _write_json_exclusive(output / "capacity-report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config, config_sha, base_config = _load_config(args.config)
    if args.mode == "freeze-fit":
        result = run_freeze_fit(args, config, config_sha, base_config)
    else:
        result = run_score_fit(args, config, config_sha)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
