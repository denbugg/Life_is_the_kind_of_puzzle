#!/usr/bin/env python3
"""Freeze then separately score one fixed Haar-BayesShrink seventh FIT emitter.

``freeze-fit`` has no label input and writes only target-blind candidate
identities. ``score-fit`` requires a later signed binding to the immutable
freeze and physically separate FIT labels. No DEV, local, terminal,
competition-test, submission, fitting, or parameter-selection mode exists.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.wavelet_shrink_matcher_view import fixed_wavelet_top32

if __package__:
    from scripts import run_local_rank_candidate_fit_capacity as rbase
else:
    import run_local_rank_candidate_fit_capacity as rbase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/wavelet_candidate_fit_preregistered_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/wavelet-candidate-emitter/fit32-draw2-v1"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"

CONFIG_SCHEMA = "aiijc-wavelet-candidate-fit-protocol-v1"
CONFIG_STATUS = "signed-target-free-freeze-only"
BINDING_SCHEMA = "aiijc-wavelet-candidate-fit-score-binding-v1"
BINDING_STATUS = "signed-post-freeze-fit-coverage-only"
METADATA_SCHEMA = "aiijc-wavelet-candidate-target-free-cache-v1"
FREEZE_SCHEMA = "aiijc-wavelet-candidate-pre-label-freeze-v1"
REPORT_SCHEMA = "aiijc-wavelet-candidate-fit-capacity-report-v1"
EMITTER_ORDER = (
    "raw",
    "adapter1600",
    "dinov2",
    "guided",
    "wiener",
    "local_rank",
    "haar_bayesshrink",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("freeze-fit", "score-fit"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    return parser.parse_args(argv)


def _load_config(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    config, digest = rbase.wbase._load_signed_json(
        path,
        schema=CONFIG_SCHEMA,
        status=CONFIG_STATUS,
    )
    if config.get("labels_opened_by_freeze_stage") is not False:
        raise RuntimeError("freeze stage must forbid labels")
    if config.get("dev_local_terminal_test_or_submission_modes") != []:
        raise RuntimeError("non-FIT modes must be absent")
    expected_recipe = {
        "name": "per_tile_per_channel_haar_bayesshrink",
        "levels": 1,
        "wavelet": "orthonormal_haar",
        "block_phase": "tile_top_left_nonoverlapping_2x2",
        "noise_estimator": "diagonal_detail_mad_div_0.67448975",
        "threshold": "per_detail_bayes_sigma_noise_squared_div_sigma_signal",
        "thresholding": "soft",
        "lowpass": "unchanged",
        "top_k": 32,
        "matcher_view_only": True,
    }
    if config.get("fixed_recipe") != expected_recipe:
        raise RuntimeError("fixed Haar-BayesShrink recipe changed")
    for artifact in config.get("frozen_inputs", {}).values():
        rbase.wbase._verify_frozen_artifact(artifact)
    rank_path = rbase.wbase._verify_frozen_artifact(
        config["frozen_inputs"]["local_rank_fit_config"]
    )
    rank_config, rank_sha, base_config = rbase._load_config(rank_path)
    if rank_sha != config["frozen_inputs"]["local_rank_fit_config"]["sha256"]:
        raise RuntimeError("local-rank base config changed")
    if config.get("source_protocol") != rank_config["source_protocol"]:
        raise RuntimeError("wavelet FIT roster differs from local-rank FIT roster")
    return config, digest, base_config


def _load_rank_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    path = rbase.wbase._verify_frozen_artifact(
        config["frozen_inputs"]["local_rank_target_free_metadata"]
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != rbase.METADATA_SCHEMA or payload.get("case_count") != 64:
        raise RuntimeError("local-rank target-free metadata contract changed")
    if payload.get("contains_target_slots_truth_or_reference_labels") is not False:
        raise RuntimeError("local-rank target-free metadata unexpectedly contains labels")
    if payload.get("emitter_order") != list(EMITTER_ORDER[:6]):
        raise RuntimeError("frozen all6 emitter order changed")
    return payload


def _write_archive(path: Path, topk: np.ndarray) -> None:
    rbase.wbase._write_npz_exclusive(path, {"emitter_topk": topk.astype(np.int32)})
    rbase._load_topk(path, emitters=7)


def run_freeze_fit(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    rank_metadata = _load_rank_metadata(config)
    names = tuple(base_config["source_protocol"]["fit_filenames"])
    records = rbase.wbase.base._manifest_records(args.manifest, names)
    cache_dir = output / "target-free-cache"
    cache_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    index = 0
    for source_index, record in enumerate(records):
        clean = rbase.wbase.base._load_clean_tiles(record, args.targets)
        for draw_index in rbase.wbase.base.FIT_DRAWS:
            rank_row = rank_metadata["rows"][index]
            case_id, dirty = rbase.wbase.base.make_target_free_fit_case(
                clean,
                source_filename=str(record["filename"]),
                draw_index=draw_index,
            )
            dirty_sha = rbase.wbase.base._array_sha256(dirty)
            identity = (record["filename"], draw_index, case_id, dirty_sha)
            expected = (
                rank_row["source_filename"],
                int(rank_row["draw_index"]),
                rank_row["case_id"],
                rank_row["dirty_sha256"],
            )
            if identity != expected:
                raise RuntimeError("wavelet replay differs from frozen FIT input")
            rank_path = rbase.wbase._project_path(rank_row["path"])
            if sha256_file(rank_path) != rank_row["sha256"]:
                raise RuntimeError("frozen all6 target-free archive changed")
            all6 = rbase._load_topk(rank_path, emitters=6)
            case_started = perf_counter()
            wavelet_top = fixed_wavelet_top32(dirty)
            all7 = np.concatenate((all6, wavelet_top[None]), axis=0)
            path = cache_dir / f"source_{source_index:02d}_draw_{draw_index}.npz"
            _write_archive(path, all7)
            rows.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_file(path),
                    "source_filename": record["filename"],
                    "draw_index": draw_index,
                    "case_id": case_id,
                    "dirty_sha256": dirty_sha,
                    "wavelet_cpu_seconds": perf_counter() - case_started,
                }
            )
            index += 1
            print(
                json.dumps({"event": "wavelet_target_free", "case": index, "count": 64}),
                flush=True,
            )

    metadata_path = output / "target-free-cache.json"
    rbase.wbase._write_json_exclusive(
        metadata_path,
        {
            "schema": METADATA_SCHEMA,
            "config_sha256": config_sha,
            "created_before_fit_label_archive_opened": True,
            "contains_target_slots_truth_or_reference_labels": False,
            "contains_pixels": False,
            "candidate_identities_target_blind": True,
            "emitter_order": list(EMITTER_ORDER),
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
            "config": rbase.wbase._record(args.config),
            "metadata": rbase.wbase._record(metadata_path),
            "runner": rbase.wbase._record(Path(__file__)),
            "module": rbase.wbase._record(
                PROJECT_ROOT / "src/aiijc_puzzle/wavelet_shrink_matcher_view.py"
            ),
        },
        "case_files": [{"path": row["path"], "sha256": row["sha256"]} for row in rows],
        "runtime_seconds": perf_counter() - started,
        "dev_local_terminal_test_or_submission_accessed": False,
    }
    rbase.wbase._write_json_exclusive(freeze_path, freeze)
    return freeze


def _verify_freeze(output: Path, config_sha: str) -> tuple[dict[str, Any], Path]:
    freeze_path = output / "pre-label-freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("config_sha256") != config_sha:
        raise RuntimeError("wavelet pre-label freeze mismatch")
    if freeze.get("created_before_fit_label_archive_opened") is not True:
        raise RuntimeError("wavelet identities were not frozen before labels")
    if freeze.get("contains_target_slots_truth_or_reference_labels") is not False:
        raise RuntimeError("wavelet freeze unexpectedly contains labels")
    metadata_path = output / "target-free-cache.json"
    if sha256_file(metadata_path) != freeze["artifacts"]["metadata"]["sha256"]:
        raise RuntimeError("wavelet target-free metadata changed after freeze")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != METADATA_SCHEMA or metadata.get("case_count") != 64:
        raise RuntimeError("wavelet target-free metadata contract changed")
    if metadata.get("emitter_order") != list(EMITTER_ORDER):
        raise RuntimeError("wavelet emitter order changed")
    for row, frozen in zip(metadata["rows"], freeze["case_files"], strict=True):
        path = rbase.wbase._project_path(row["path"])
        if row["path"] != frozen["path"] or row["sha256"] != frozen["sha256"]:
            raise RuntimeError("wavelet target-free roster changed")
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError("wavelet target-free archive changed after freeze")
        rbase._load_topk(path, emitters=7)
    return metadata, freeze_path


def coverage_counts(topk: np.ndarray, truth: np.ndarray) -> dict[str, dict[str, int]]:
    """Count exact-neighbour supply for frozen all6 and all6+wavelet."""

    result: dict[str, dict[str, int]] = {}
    for axis, name in enumerate(("right", "down")):
        eligible = truth[axis] >= 0
        hit6 = np.any(topk[:6, axis] == truth[axis][None, :, None], axis=(0, 2)) & eligible
        wavelet = np.any(topk[6, axis] == truth[axis, :, None], axis=1) & eligible
        hit7 = hit6 | wavelet
        result[name] = {
            "eligible": int(np.count_nonzero(eligible)),
            "all6_union": int(np.count_nonzero(hit6)),
            "wavelet_top32": int(np.count_nonzero(wavelet)),
            "all7_union": int(np.count_nonzero(hit7)),
            "wavelet_unique_over_all6": int(np.count_nonzero(wavelet & ~hit6)),
        }
    return result


def volume_matched_null(
    topk: np.ndarray,
    truth: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Return the exact uniform-null expectation at matched new-identity volume.

    The volume ``m`` is target-blind: it is the number of wavelet identities in
    one row that are absent from the frozen all6 union. Conditional on an
    eligible truth being absent from all6, a uniform draw without replacement
    from the remaining ``575 - |all6 union|`` identities hits it with
    probability ``m / available``. No random seed or Monte Carlo draw is used.
    """

    result: dict[str, dict[str, float | int]] = {}
    for axis, name in enumerate(("right", "down")):
        eligible_misses = 0
        actual = 0
        expected = 0.0
        proposed_unique = 0
        for source in range(576):
            target = int(truth[axis, source])
            if target < 0:
                continue
            base = np.unique(topk[:6, axis, source].reshape(-1))
            wavelet_unique = np.setdiff1d(
                topk[6, axis, source],
                base,
                assume_unique=True,
            )
            if target in base:
                continue
            available = 575 - len(base)
            if available <= 0 or len(wavelet_unique) > available:
                raise RuntimeError("invalid all6 complement for volume-matched null")
            eligible_misses += 1
            proposed_unique += len(wavelet_unique)
            expected += len(wavelet_unique) / available
            actual += int(target in wavelet_unique)
        hit_rate = actual / eligible_misses if eligible_misses else 0.0
        null_rate = expected / eligible_misses if eligible_misses else 0.0
        result[name] = {
            "eligible_all6_misses": eligible_misses,
            "new_unique_proposals_on_misses": proposed_unique,
            "actual_unique_hits": actual,
            "uniform_volume_matched_expected_hits": expected,
            "specific_excess_hits": actual - expected,
            "actual_available_miss_hit_rate": hit_rate,
            "uniform_null_available_miss_hit_rate": null_rate,
            "specific_excess_available_miss_hit_rate": hit_rate - null_rate,
        }
    return result


def _sign_counts(values: np.ndarray) -> dict[str, int]:
    return {
        "positive": int(np.count_nonzero(values > 0)),
        "zero": int(np.count_nonzero(values == 0)),
        "negative": int(np.count_nonzero(values < 0)),
    }


def run_score_fit(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
) -> dict[str, Any]:
    if args.binding is None:
        raise RuntimeError("score-fit requires a separately signed --binding")
    binding, binding_sha = rbase.wbase._load_signed_json(
        args.binding,
        schema=BINDING_SCHEMA,
        status=BINDING_STATUS,
    )
    output = args.output_dir.resolve()
    metadata, freeze_path = _verify_freeze(output, config_sha)
    if binding.get("experiment_config_sha256") != config_sha:
        raise RuntimeError("binding config mismatch")
    if binding.get("pre_label_freeze_sha256") != sha256_file(freeze_path):
        raise RuntimeError("binding freeze mismatch")
    if binding.get("scope") != {
        "fit_coverage_counts_only": True,
        "model_fit_or_parameter_selection": False,
        "dev_local_terminal_test_submission": False,
    }:
        raise RuntimeError("binding scope changed")
    labels_path = rbase.wbase._verify_frozen_artifact(
        binding["separated_fit_labels_metadata"]
    )
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if (
        labels.get("case_count") != 64
        or labels.get("labels_physically_separate_from_target_free_cache") is not True
    ):
        raise RuntimeError("separated FIT label metadata changed")

    metric_keys = (
        "eligible",
        "all6_union",
        "wavelet_top32",
        "all7_union",
        "wavelet_unique_over_all6",
    )
    totals = {
        axis: {key: 0 for key in metric_keys}
        for axis in ("right", "down")
    }
    case_gains = np.zeros(64, dtype=np.int32)
    source_gains = np.zeros(32, dtype=np.int32)
    case_null_expectation = np.zeros(64, dtype=np.float64)
    source_null_expectation = np.zeros(32, dtype=np.float64)
    null_totals = {
        axis: {
            key: 0.0
            for key in (
                "eligible_all6_misses",
                "new_unique_proposals_on_misses",
                "actual_unique_hits",
                "uniform_volume_matched_expected_hits",
                "specific_excess_hits",
            )
        }
        for axis in ("right", "down")
    }
    for index, (row, label_row) in enumerate(
        zip(metadata["rows"], labels["rows"], strict=True)
    ):
        if (row["source_filename"], row["draw_index"], row["case_id"]) != (
            label_row["source_filename"],
            label_row["draw_index"],
            label_row["case_id"],
        ):
            raise RuntimeError("target-free and label rosters are misaligned")
        topk = rbase._load_topk(rbase.wbase._project_path(row["path"]), emitters=7)
        label_file = rbase.wbase._project_path(label_row["path"])
        if sha256_file(label_file) != label_row["sha256"]:
            raise RuntimeError("separated FIT label file changed")
        with np.load(label_file, allow_pickle=False) as archive:
            truth = np.ascontiguousarray(archive["truth_by_source"], dtype=np.int32)
        coverage = coverage_counts(topk, truth)
        null = volume_matched_null(topk, truth)
        unique = sum(coverage[axis]["wavelet_unique_over_all6"] for axis in coverage)
        expected = sum(
            float(null[axis]["uniform_volume_matched_expected_hits"])
            for axis in null
        )
        case_gains[index] = unique
        source_gains[index // 2] += unique
        case_null_expectation[index] = expected
        source_null_expectation[index // 2] += expected
        for axis in totals:
            for key, value in coverage[axis].items():
                totals[axis][key] += value
            for key in null_totals[axis]:
                null_totals[axis][key] += float(null[axis][key])

    pooled = {key: totals["right"][key] + totals["down"][key] for key in metric_keys}
    rates = {
        key: pooled[key] / pooled["eligible"]
        for key in ("all6_union", "wavelet_top32", "all7_union")
    }
    gain = rates["all7_union"] - rates["all6_union"]
    rng = np.random.default_rng(20260921)
    source_draws = rng.integers(0, 32, size=(20_000, 32))
    source_bootstrap_per_case = source_gains[source_draws].mean(axis=1) / 2.0
    source_excess = source_gains.astype(np.float64) - source_null_expectation
    source_bootstrap_excess_per_case = source_excess[source_draws].mean(axis=1) / 2.0
    case_draws = rng.integers(0, 64, size=(20_000, 64))
    case_bootstrap = case_gains[case_draws].mean(axis=1)
    case_excess = case_gains.astype(np.float64) - case_null_expectation
    pooled_null = {
        key: null_totals["right"][key] + null_totals["down"][key]
        for key in null_totals["right"]
    }
    all6_misses = int(pooled_null["eligible_all6_misses"])
    null_expected = float(pooled_null["uniform_volume_matched_expected_hits"])
    specific_excess_hits = float(pooled_null["specific_excess_hits"])
    specific_excess_rate = specific_excess_hits / pooled["eligible"]
    actual_miss_hit_rate = pooled["wavelet_unique_over_all6"] / all6_misses
    null_miss_hit_rate = null_expected / all6_misses
    source_excess_ci = np.quantile(
        source_bootstrap_excess_per_case,
        (0.025, 0.975),
    )
    gate = config["coverage_gate"]
    null_gate = config["volume_matched_null_gate"]
    raw_gate_passed = gain >= float(gate["minimum_incremental_gain_over_all6"])
    excess_size_passed = specific_excess_rate >= float(
        null_gate["minimum_specific_excess_absolute_gain"]
    )
    excess_ci_passed = float(source_excess_ci[0]) > float(
        null_gate["source_bootstrap_lower_bound_must_exceed"]
    )
    passed = raw_gate_passed and excess_size_passed and excess_ci_passed
    report = {
        "schema": REPORT_SCHEMA,
        "status": "retain-as-seventh-candidate-supply" if passed else "compact-negative-stop",
        "claim": (
            "FIT-only append-only candidate coverage; not ranking, solver, DEV, "
            "or promotion evidence"
        ),
        "config_sha256": config_sha,
        "binding_sha256": binding_sha,
        "coverage_counts": {**totals, "pooled": pooled},
        "pooled_coverage_rates": rates,
        "volume_matched_uniform_null": {
            "definition": (
                "conditional on an eligible all6 miss, uniformly sample the same "
                "target-blind count of identities newly proposed by wavelet from "
                "the identities absent from all6"
            ),
            "direction_counts": null_totals,
            "pooled_counts": pooled_null,
            "actual_available_miss_hit_rate": actual_miss_hit_rate,
            "uniform_null_available_miss_hit_rate": null_miss_hit_rate,
            "specific_excess_available_miss_hit_rate": (
                actual_miss_hit_rate - null_miss_hit_rate
            ),
            "specific_excess_hits": specific_excess_hits,
            "specific_excess_absolute_gain": specific_excess_rate,
            "case_excess_direction_counts_64": _sign_counts(case_excess),
            "source_excess_direction_counts_32": _sign_counts(source_excess),
            "mean_specific_excess_per_case": float(case_excess.mean()),
            "source_bootstrap_ci95_specific_excess_per_case": source_excess_ci.tolist(),
            "source_bootstrap_ci95_specific_excess_absolute_gain": (
                source_excess_ci / 1104.0
            ).tolist(),
        },
        "incremental_over_all6": {
            "additional_true_neighbours": pooled["wavelet_unique_over_all6"],
            "absolute_rate_gain": gain,
            "direction_additional": {
                axis: totals[axis]["wavelet_unique_over_all6"] for axis in totals
            },
            "case_direction_counts_64": _sign_counts(case_gains),
            "source_direction_counts_32": _sign_counts(source_gains),
            "mean_unique_per_case": float(case_gains.mean()),
            "median_unique_per_case": float(np.median(case_gains)),
            "min_unique_per_case": int(case_gains.min()),
            "max_unique_per_case": int(case_gains.max()),
            "source_bootstrap_ci95_unique_per_case": np.quantile(
                source_bootstrap_per_case, (0.025, 0.975)
            ).tolist(),
            "source_bootstrap_ci95_absolute_rate_gain": (
                np.quantile(source_bootstrap_per_case, (0.025, 0.975)) / 1104.0
            ).tolist(),
            "case_bootstrap_ci95_unique_per_case": np.quantile(
                case_bootstrap, (0.025, 0.975)
            ).tolist(),
        },
        "gate": {
            "raw_incremental": {**gate, "passed": raw_gate_passed},
            "volume_matched_null": {
                **null_gate,
                "specific_excess_size_passed": excess_size_passed,
                "source_bootstrap_lower_passed": excess_ci_passed,
            },
            "passed": passed,
        },
        "legality": {
            "organizer_train_fit_only": True,
            "wavelet_pixels_matcher_only": True,
            "original_upright_tiles_unchanged": True,
            "identities_frozen_before_labels": True,
            "direct_fusion_or_replacement_authorized": False,
            "dev_local_terminal_test_or_submission_accessed": False,
        },
        "artifacts": {
            "config": rbase.wbase._record(args.config),
            "binding": rbase.wbase._record(args.binding),
            "metadata": rbase.wbase._record(output / "target-free-cache.json"),
            "freeze": rbase.wbase._record(freeze_path),
            "labels": rbase.wbase._record(labels_path),
            "runner": rbase.wbase._record(Path(__file__)),
            "module": rbase.wbase._record(
                PROJECT_ROOT / "src/aiijc_puzzle/wavelet_shrink_matcher_view.py"
            ),
        },
    }
    rbase.wbase._write_json_exclusive(output / "capacity-report.json", report)
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
