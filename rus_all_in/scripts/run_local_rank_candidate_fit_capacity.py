#!/usr/bin/env python3
"""Freeze then separately score a fixed local-rank sixth FIT emitter."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.local_rank_matcher_view import WINDOW, fixed_local_rank_top32
from aiijc_puzzle.protocol import sha256_file

if __package__:
    from scripts import run_wiener_candidate_emitter_fit_capacity as wbase
else:
    import run_wiener_candidate_emitter_fit_capacity as wbase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/local_rank_candidate_fit_preregistered_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/local-rank-candidate-emitter/fit32-draw2-v1"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"

CONFIG_SCHEMA = "aiijc-local-rank-candidate-fit-protocol-v1"
CONFIG_STATUS = "signed-target-free-freeze-only"
BINDING_SCHEMA = "aiijc-local-rank-candidate-fit-score-binding-v1"
BINDING_STATUS = "signed-post-freeze-fit-coverage-only"
METADATA_SCHEMA = "aiijc-local-rank-candidate-target-free-cache-v1"
FREEZE_SCHEMA = "aiijc-local-rank-candidate-pre-label-freeze-v1"
REPORT_SCHEMA = "aiijc-local-rank-candidate-fit-capacity-report-v1"


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
    config, digest = wbase._load_signed_json(
        path, schema=CONFIG_SCHEMA, status=CONFIG_STATUS
    )
    if config.get("labels_opened_by_freeze_stage") is not False:
        raise RuntimeError("freeze stage must forbid labels")
    if config.get("fixed_recipe") != {
        "name": "per_tile_per_channel_local_midrank",
        "window": 3,
        "padding": "reflect",
        "comparison": "eight_neighbours_lower_plus_half_equal",
        "top_k": 32,
        "matcher_view_only": True,
    } or WINDOW != 3:
        raise RuntimeError("fixed local-rank recipe changed")
    for artifact in config.get("frozen_inputs", {}).values():
        wbase._verify_frozen_artifact(artifact)
    wiener_path = wbase._verify_frozen_artifact(
        config["frozen_inputs"]["wiener_fit_config"]
    )
    wiener_config, wiener_sha, base_config = wbase._load_config(wiener_path)
    if wiener_sha != config["frozen_inputs"]["wiener_fit_config"]["sha256"]:
        raise RuntimeError("Wiener base config changed")
    source = config.get("source_protocol")
    if source != wiener_config["source_protocol"]:
        raise RuntimeError("local-rank FIT roster differs from Wiener FIT roster")
    return config, digest, base_config


def _load_json_artifact(artifact: Mapping[str, str]) -> dict[str, Any]:
    path = wbase._verify_frozen_artifact(artifact)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_topk(path: Path, *, emitters: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "emitter_topk" not in archive.files:
            raise RuntimeError("target-free archive lacks emitter_topk")
        topk = np.ascontiguousarray(archive["emitter_topk"], dtype=np.int32)
    if topk.shape != (emitters, 2, 576, 32):
        raise RuntimeError("target-free emitter roster shape changed")
    if np.any((topk < 0) | (topk >= 576)):
        raise RuntimeError("target-free emitter roster contains invalid identities")
    for emitter in range(emitters):
        for axis in range(2):
            for source in range(576):
                row = topk[emitter, axis, source]
                if source in row or len(np.unique(row)) != 32:
                    raise RuntimeError("target-free top-k row is invalid")
    return topk


def _write_archive(path: Path, topk: np.ndarray) -> None:
    wbase._write_npz_exclusive(path, {"emitter_topk": topk.astype(np.int32)})
    _load_topk(path, emitters=6)


def run_freeze_fit(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    wmeta = _load_json_artifact(config["frozen_inputs"]["wiener_target_free_metadata"])
    gmeta = _load_json_artifact(config["frozen_inputs"]["guided_target_free_metadata"])
    if wmeta.get("case_count") != 64 or gmeta.get("case_count") != 64:
        raise RuntimeError("base target-free roster must contain 64 cases")
    if wmeta.get("contains_target_slots_truth_or_reference_labels") is not False:
        raise RuntimeError("Wiener target-free metadata unexpectedly contains labels")
    if gmeta.get("contains_target_slots_truth_or_reference_labels") is not False:
        raise RuntimeError("guided target-free metadata unexpectedly contains labels")
    names = tuple(base_config["source_protocol"]["fit_filenames"])
    records = wbase.base._manifest_records(args.manifest, names)
    cache_dir = output / "target-free-cache"
    cache_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    index = 0
    for source_index, record in enumerate(records):
        clean = wbase.base._load_clean_tiles(record, args.targets)
        for draw_index in wbase.base.FIT_DRAWS:
            wrow, grow = wmeta["rows"][index], gmeta["rows"][index]
            case_id, dirty = wbase.base.make_target_free_fit_case(
                clean,
                source_filename=str(record["filename"]),
                draw_index=draw_index,
            )
            dirty_sha = wbase.base._array_sha256(dirty)
            identity = (record["filename"], draw_index, case_id, dirty_sha)
            if identity != (
                wrow["source_filename"],
                int(wrow["draw_index"]),
                wrow["case_id"],
                wrow["dirty_sha256"],
            ) or identity != (
                grow["source_filename"],
                int(grow["draw_index"]),
                grow["case_id"],
                grow["dirty_sha256"],
            ):
                raise RuntimeError("local-rank replay differs from frozen FIT input")
            wpath, gpath = wbase._project_path(wrow["path"]), wbase._project_path(grow["path"])
            if sha256_file(wpath) != wrow["sha256"] or sha256_file(gpath) != grow["sha256"]:
                raise RuntimeError("base target-free archive changed")
            wtop = _load_topk(wpath, emitters=4)
            gtop = _load_topk(gpath, emitters=4)
            if not np.array_equal(wtop[:3], gtop[:3]):
                raise RuntimeError("legacy emitter identities differ across frozen bases")
            case_started = perf_counter()
            rank_top = fixed_local_rank_top32(dirty)
            all_top = np.concatenate((wtop[:3], gtop[3:4], wtop[3:4], rank_top[None]))
            path = cache_dir / f"source_{source_index:02d}_draw_{draw_index}.npz"
            _write_archive(path, all_top)
            rows.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_file(path),
                    "source_filename": record["filename"],
                    "draw_index": draw_index,
                    "case_id": case_id,
                    "dirty_sha256": dirty_sha,
                    "rank_cpu_seconds": perf_counter() - case_started,
                }
            )
            index += 1
            print(json.dumps({"event": "rank_target_free", "case": index, "count": 64}), flush=True)
    metadata_path = output / "target-free-cache.json"
    wbase._write_json_exclusive(
        metadata_path,
        {
            "schema": METADATA_SCHEMA,
            "config_sha256": config_sha,
            "created_before_fit_label_archive_opened": True,
            "contains_target_slots_truth_or_reference_labels": False,
            "contains_pixels": False,
            "emitter_order": ["raw", "adapter1600", "dinov2", "guided", "wiener", "local_rank"],
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
            "config": wbase._record(args.config),
            "metadata": wbase._record(metadata_path),
            "runner": wbase._record(Path(__file__)),
            "module": wbase._record(PROJECT_ROOT / "src/aiijc_puzzle/local_rank_matcher_view.py"),
        },
        "case_files": [{"path": row["path"], "sha256": row["sha256"]} for row in rows],
        "runtime_seconds": perf_counter() - started,
        "dev_local_terminal_test_or_submission_accessed": False,
    }
    wbase._write_json_exclusive(freeze_path, freeze)
    return freeze


def _verify_freeze(output: Path, config_sha: str) -> tuple[dict[str, Any], Path]:
    freeze_path = output / "pre-label-freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("config_sha256") != config_sha:
        raise RuntimeError("pre-label freeze mismatch")
    if freeze.get("created_before_fit_label_archive_opened") is not True:
        raise RuntimeError("identities were not frozen before labels")
    metadata_path = output / "target-free-cache.json"
    if sha256_file(metadata_path) != freeze["artifacts"]["metadata"]["sha256"]:
        raise RuntimeError("target-free metadata changed after freeze")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for row, frozen in zip(metadata["rows"], freeze["case_files"], strict=True):
        path = wbase._project_path(row["path"])
        if row["path"] != frozen["path"] or row["sha256"] != frozen["sha256"]:
            raise RuntimeError("target-free roster changed")
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError("target-free archive changed after freeze")
        _load_topk(path, emitters=6)
    return metadata, freeze_path


def coverage_counts(topk: np.ndarray, truth: np.ndarray) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for axis, name in enumerate(("right", "down")):
        eligible = truth[axis] >= 0
        hit5 = np.any(topk[:5, axis] == truth[axis][None, :, None], axis=(0, 2)) & eligible
        rank = np.any(topk[5, axis] == truth[axis, :, None], axis=1) & eligible
        hit6 = hit5 | rank
        result[name] = {
            "eligible": int(np.count_nonzero(eligible)),
            "all5_union": int(np.count_nonzero(hit5)),
            "rank_top32": int(np.count_nonzero(rank)),
            "all6_union": int(np.count_nonzero(hit6)),
            "rank_unique_over_all5": int(np.count_nonzero(rank & ~hit5)),
        }
    return result


def run_score_fit(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
) -> dict[str, Any]:
    if args.binding is None:
        raise RuntimeError("score-fit requires a separately signed --binding")
    binding, binding_sha = wbase._load_signed_json(
        args.binding, schema=BINDING_SCHEMA, status=BINDING_STATUS
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
    labels_path = wbase._verify_frozen_artifact(binding["separated_fit_labels_metadata"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if (
        labels.get("case_count") != 64
        or labels.get("labels_physically_separate_from_target_free_cache") is not True
    ):
        raise RuntimeError("separated label metadata changed")
    totals = {
        axis: {
            key: 0
            for key in (
                "eligible",
                "all5_union",
                "rank_top32",
                "all6_union",
                "rank_unique_over_all5",
            )
        }
        for axis in ("right", "down")
    }
    source_gains = np.zeros(32, dtype=np.int32)
    for index, (row, label_row) in enumerate(zip(metadata["rows"], labels["rows"], strict=True)):
        if (row["source_filename"], row["draw_index"], row["case_id"]) != (
            label_row["source_filename"], label_row["draw_index"], label_row["case_id"]
        ):
            raise RuntimeError("target-free and label rosters are misaligned")
        topk = _load_topk(wbase._project_path(row["path"]), emitters=6)
        label_file = wbase._project_path(label_row["path"])
        if sha256_file(label_file) != label_row["sha256"]:
            raise RuntimeError("separated label file changed")
        with np.load(label_file, allow_pickle=False) as archive:
            truth = np.ascontiguousarray(archive["truth_by_source"], dtype=np.int32)
        coverage = coverage_counts(topk, truth)
        source_gains[index // 2] += sum(
            coverage[axis]["rank_unique_over_all5"] for axis in coverage
        )
        for axis in totals:
            for key, value in coverage[axis].items():
                totals[axis][key] += value
    pooled = {key: totals["right"][key] + totals["down"][key] for key in totals["right"]}
    rates = {
        key: pooled[key] / pooled["eligible"]
        for key in ("all5_union", "rank_top32", "all6_union")
    }
    gain = rates["all6_union"] - rates["all5_union"]
    gate = config["coverage_gate"]
    passed = gain >= float(gate["minimum_incremental_gain_over_all5"])
    rng = np.random.default_rng(20260920)
    draws = rng.integers(0, 32, size=(20_000, 32))
    bootstrap = source_gains[draws].mean(axis=1) / 2
    report = {
        "schema": REPORT_SCHEMA,
        "status": "retain-as-sixth-candidate-supply" if passed else "compact-negative-stop",
        "claim": "FIT-only candidate coverage; not real evaluation or promotion evidence",
        "config_sha256": config_sha,
        "binding_sha256": binding_sha,
        "coverage_counts": {**totals, "pooled": pooled},
        "pooled_coverage_rates": rates,
        "incremental_over_all5": {
            "additional_true_neighbours": pooled["rank_unique_over_all5"],
            "absolute_rate_gain": gain,
            "positive_source_groups": int(np.count_nonzero(source_gains > 0)),
            "mean_unique_per_case": float(source_gains.sum() / 64),
            "bootstrap_ci95_unique_per_case": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
        },
        "gate": {**gate, "passed": passed},
        "legality": {
            "organizer_train_fit_only": True,
            "rank_pixels_matcher_only": True,
            "original_upright_tiles_unchanged": True,
            "identities_frozen_before_labels": True,
            "dev_local_terminal_test_or_submission_accessed": False,
        },
        "artifacts": {
            "config": wbase._record(args.config),
            "binding": wbase._record(args.binding),
            "metadata": wbase._record(output / "target-free-cache.json"),
            "freeze": wbase._record(freeze_path),
            "labels": wbase._record(labels_path),
            "runner": wbase._record(Path(__file__)),
            "module": wbase._record(PROJECT_ROOT / "src/aiijc_puzzle/local_rank_matcher_view.py"),
        },
    }
    wbase._write_json_exclusive(output / "capacity-report.json", report)
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
