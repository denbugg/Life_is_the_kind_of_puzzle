from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_union_hard_edge_priority_pilot.py"
    specification = importlib.util.spec_from_file_location(
        "run_union_hard_edge_priority_pilot_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _manifest(names: list[str]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": {"test": True},
        "splits": {
            "train": [
                {
                    "filename": name,
                    "input_sha256": "0" * 64,
                    "target_sha256": "1" * 64,
                }
                for name in names
            ]
        },
    }
    payload["protocol_digest"] = compute_protocol_digest(payload)
    return payload


def _config(
    fit: list[str],
    evaluation: list[str],
    frozen: dict[str, dict[str, str]],
    *,
    manifest_digest: str,
    registry_digest: str | None = None,
) -> dict[str, object]:
    registry_digest = registry_digest or "2" * 64
    effective_namespace = runner._effective_namespace(manifest_digest, registry_digest)
    return {
        "schema": runner.CONFIG_SCHEMA,
        "experiment": runner.EXPERIMENT,
        "registered_before_target_access": True,
        "protocol": {
            "commitment_written_before_target_access": True,
            "eval_predictions_frozen_before_reference": True,
            "single_eval_no_tuning": True,
            "calibration_holdout_test_forbidden": True,
        },
        "model": {
            "architecture": "union-hard-edge-deepsets-bounded-residual-v1",
            "feature_dimension": len(runner.FEATURE_NAMES),
            "feature_names": list(runner.FEATURE_NAMES),
            "hidden_dimension": runner.HIDDEN_DIMENSION,
            "residual_limit": runner.RESIDUAL_LIMIT,
            "hard_edge_count": runner.HARD_EDGE_COUNT,
            "edge_budget_per_axis": runner.DECODER_EDGE_BUDGET,
            "zero_initialised_residual": True,
        },
        "fullres_priority": {
            "query_cap": runner.FUSION_QUERY_CAP,
            "candidate_rank_cap": runner.FUSION_CANDIDATE_RANK_CAP,
            "boost_scale": runner.FUSION_BOOST_SCALE,
        },
        "training": {
            "steps": runner.TRAINING_STEPS,
            "learning_rate": runner.LEARNING_RATE,
            "weight_decay": runner.WEIGHT_DECAY,
            "pairwise_weight": runner.PAIRWISE_WEIGHT,
            "residual_weight": runner.RESIDUAL_WEIGHT,
            "gradient_clip": runner.GRADIENT_CLIP,
            "hyperparameter_sweep": False,
        },
        "runtime": {
            "pilot_device": "mps",
            "mps_requires_explicit_nondeterminism": True,
            "cpu_benchmark_allowed": True,
            "inference_batch": runner.INFERENCE_BATCH,
            "feature_cache_dtype": "float32",
        },
        "selection": {
            "manifest_split": "train",
            "namespace": runner.SELECTION_NAMESPACE,
            "effective_namespace": effective_namespace,
            "global_exclusion_digest_in_namespace": False,
            "registry_digest": registry_digest,
            "selection_seed": runner.SELECTION_SEED,
            "synthetic_seed": runner.SYNTHETIC_SEED,
            "bootstrap_seed": runner.BOOTSTRAP_SEED,
            "draw_indices": list(runner.DRAWS),
            "organizer_train_count": runner.ORGANIZER_TRAIN_COUNT,
            "eligible_count": runner.ELIGIBLE_TRAIN_COUNT,
            "excluded_train_count": runner.EXCLUDED_TRAIN_COUNT,
            "excluded_train_digest": runner.EXCLUDED_TRAIN_DIGEST,
            "global_exclusion_count": runner.GLOBAL_EXCLUSION_COUNT,
            "global_exclusion_digest": runner.GLOBAL_EXCLUSION_DIGEST,
            "fit_source_count": runner.FIT_SOURCE_COUNT,
            "eval_source_count": runner.EVAL_SOURCE_COUNT,
            "fit_case_count": runner.FIT_CASE_COUNT,
            "eval_case_count": runner.EVAL_CASE_COUNT,
            "fit_source_filenames": fit,
            "fit_source_order_digest": runner._names_digest(fit),
            "fit_source_set_digest": runner._names_digest(tuple(sorted(fit))),
            "fit_cases_digest": runner.FIT_CASES_DIGEST,
            "eval_source_filenames": evaluation,
            "eval_source_order_digest": runner._names_digest(evaluation),
            "eval_source_set_digest": runner._names_digest(tuple(sorted(evaluation))),
            "eval_cases_digest": runner.EVAL_CASES_DIGEST,
            "overlaps": {"fit_eval": 0, "global_exclusion": 0},
        },
        "frozen_inputs": frozen,
    }


def _write_signed_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.with_name(f"{path.name}.sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n",
        encoding="utf-8",
    )


def _selection_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Namespace:
    names = [f"image_{index:04d}.png" for index in range(80)]
    fit = names[:64]
    evaluation = names[64:]
    monkeypatch.setattr(runner, "ORGANIZER_TRAIN_COUNT", len(names))
    monkeypatch.setattr(runner, "FIT_SOURCE_ORDER_DIGEST", runner._names_digest(fit))
    monkeypatch.setattr(
        runner,
        "EVAL_SOURCE_ORDER_DIGEST",
        runner._names_digest(evaluation),
    )
    monkeypatch.setattr(runner, "FIT_CASES_DIGEST", runner._cases_digest(fit, runner.DRAWS))
    monkeypatch.setattr(
        runner,
        "EVAL_CASES_DIGEST",
        runner._cases_digest(evaluation, runner.DRAWS),
    )
    frozen: dict[str, dict[str, str]] = {}
    for key in runner.DEFAULT_FROZEN_INPUTS:
        path = tmp_path / f"{key}.bin"
        path.write_bytes(key.encode())
        frozen[key] = {"path": str(path), "sha256": sha256_file(path)}
    manifest = _manifest(names)
    audit_path = tmp_path / "roster-audit.json"
    audit_path.write_text(json.dumps({"schema": runner.ROSTER_AUDIT_SCHEMA}), encoding="utf-8")
    config = _config(
        fit,
        evaluation,
        frozen,
        manifest_digest=str(manifest["protocol_digest"]),
    )
    config["roster_audit"] = {
        "path": str(audit_path),
        "sha256": sha256_file(audit_path),
    }
    config_path = tmp_path / "config.json"
    _write_signed_config(config_path, config)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return Namespace(
        config=config_path,
        manifest=manifest_path,
        output_dir=tmp_path / "output",
        benchmark_one_case=False,
    )


def test_selection_freezes_explicit_rosters_draws_and_lineage_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _selection_fixture(tmp_path, monkeypatch)
    path = runner.create_selection_commitment(args)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == runner.CONFIG_SCHEMA
    assert payload["artifact"] == "selection-commitment"
    assert payload["created_before_target_access"] is True
    assert payload["namespace"] == runner.SELECTION_NAMESPACE
    assert payload["fit"]["case_count"] == 128
    assert payload["eval"]["case_count"] == 32
    assert payload["draw_indices"] == [0, 1]
    assert set(payload["frozen_inputs"]) == set(runner.DEFAULT_FROZEN_INPUTS)
    assert set(payload["runtime_sources"]) == set(runner.RUNTIME_SOURCE_PATHS)
    assert runner._load_commitment(args.output_dir, args.config, args.manifest) == payload
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.create_selection_commitment(args)


def test_write_config_from_canonical_roster_audit_is_exclusive_and_target_blind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = [f"image_{index:04d}.png" for index in range(80)]
    monkeypatch.setattr(runner, "ORGANIZER_TRAIN_COUNT", len(names))
    monkeypatch.setattr(runner, "ELIGIBLE_TRAIN_COUNT", len(names))
    monkeypatch.setattr(runner, "EXCLUDED_TRAIN_COUNT", 0)
    monkeypatch.setattr(runner, "EXCLUDED_TRAIN_DIGEST", runner._names_digest(()))
    monkeypatch.setattr(runner, "GLOBAL_EXCLUSION_COUNT", 0)
    monkeypatch.setattr(runner, "GLOBAL_EXCLUSION_DIGEST", runner._names_digest(()))
    monkeypatch.setattr(runner, "ACTIVE_GLOBAL_REGISTRY_INDICES", (0,))
    monkeypatch.setattr(runner, "LINEAGE_ONLY_REGISTRY_INDICES", ())
    frozen: dict[str, dict[str, str]] = {}
    defaults: dict[str, Path] = {}
    for key in runner.DEFAULT_FROZEN_INPUTS:
        path = tmp_path / f"{key}.bin"
        path.write_bytes(key.encode())
        defaults[key] = path
        frozen[key] = {"path": str(path), "sha256": sha256_file(path)}
    monkeypatch.setattr(runner, "DEFAULT_FROZEN_INPUTS", defaults)
    manifest = _manifest(names)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    registry_artifact = tmp_path / "registry-source.json"
    registry_artifact.write_text('{"names": []}\n', encoding="utf-8")
    registry = [
        {
            "path": str(registry_artifact),
            "sha256": sha256_file(registry_artifact),
            "fields": ["names"],
        }
    ]
    registry_digest = runner._canonical_json_digest(registry)
    effective_namespace = runner._effective_namespace(
        str(manifest["protocol_digest"]),
        registry_digest,
    )
    ranked = runner.select_manifest_records(
        manifest,
        "train",
        limit=len(names),
        seed=runner.SELECTION_SEED,
        namespace=effective_namespace,
    )
    selected = [str(row["filename"]) for row in ranked]
    fit = selected[:64]
    evaluation = selected[64:]
    monkeypatch.setattr(runner, "FIT_SOURCE_ORDER_DIGEST", runner._names_digest(fit))
    monkeypatch.setattr(
        runner,
        "EVAL_SOURCE_ORDER_DIGEST",
        runner._names_digest(evaluation),
    )
    monkeypatch.setattr(runner, "FIT_CASES_DIGEST", runner._cases_digest(fit, runner.DRAWS))
    monkeypatch.setattr(
        runner,
        "EVAL_CASES_DIGEST",
        runner._cases_digest(evaluation, runner.DRAWS),
    )
    selection = _config(
        fit,
        evaluation,
        frozen,
        manifest_digest=str(manifest["protocol_digest"]),
        registry_digest=registry_digest,
    )["selection"]
    selection["effective_namespace"] = effective_namespace
    audit_path = tmp_path / "roster-audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema": runner.ROSTER_AUDIT_SCHEMA,
                "created_before_target_access": True,
                "target_images_accessed": False,
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                    "protocol_digest": manifest["protocol_digest"],
                },
                "recipes": {
                    "effective_namespace": {
                        "global_exclusion_digest_in_namespace": False,
                        "exact_value": effective_namespace,
                    }
                },
                "selection": selection,
                "exclusion": {
                    "registry": registry,
                    "registry_digest": registry_digest,
                    "membership_recipe": {
                        "active_registry_indices": [0],
                        "lineage_pin_only_registry_indices": [],
                    },
                    "registry_resolution": [
                        {
                            "registry_index": 0,
                            "active_for_membership": True,
                            "resolved_filename_count": 0,
                            "resolved_filename_digest": runner._names_digest(()),
                        }
                    ],
                    "excluded_train_filenames": [],
                    "global_exclusion_filenames": [],
                },
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "pilot.json"
    args = Namespace(
        mode="selection",
        write_config=True,
        benchmark_one_case=False,
        roster_audit=audit_path,
        config=config_path,
        manifest=manifest_path,
    )
    assert runner.write_preregistered_config(args) == config_path
    config, observed_sha = runner._load_config(config_path)
    assert config["selection"]["fit_source_filenames"] == fit
    assert config["selection"]["eval_source_filenames"] == evaluation
    assert config["roster_audit"]["sha256"] == sha256_file(audit_path)
    assert config_path.with_name(f"{config_path.name}.sha256").read_text().split()[0] == (
        observed_sha
    )
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar mismatch"):
        runner._load_config(config_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.write_preregistered_config(args)


def test_selection_rejects_nonzero_overlap_and_unpinned_draws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _selection_fixture(tmp_path, monkeypatch)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["selection"]["overlaps"]["fit_eval"] = 1
    _write_signed_config(args.config, config)
    with pytest.raises(ValueError, match="all-zero"):
        runner.create_selection_commitment(args)
    config["selection"]["overlaps"]["fit_eval"] = 0
    config["selection"]["draw_indices"] = [0]
    _write_signed_config(args.config, config)
    with pytest.raises(ValueError, match="draw_indices"):
        runner.create_selection_commitment(args)


def test_run_artifact_contract_refuses_any_existing_output(tmp_path: Path) -> None:
    paths = runner._prepare_run_paths(tmp_path / "run")
    paths.checkpoint.touch()
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._prepare_run_paths(tmp_path / "run")


def test_target_free_case_view_cannot_expose_exact_position_mapping() -> None:
    source = Namespace(
        case_id="case",
        source_filename="image.png",
        dirty_tiles=np.zeros((runner.COUNT, 20, 20, 3), dtype=np.uint8),
        input_tile_to_position=np.arange(runner.COUNT, dtype=np.int32),
    )
    view = runner._target_free_case(source)
    assert view.case_id == "case"
    assert view.dirty_tiles is source.dirty_tiles
    assert not hasattr(view, "input_tile_to_position")


def test_fit_fixed_budget_counter_matches_top144_per_axis() -> None:
    axis = np.repeat(np.arange(2, dtype=np.int8), runner.HARD_EDGES_PER_AXIS)
    scores = np.tile(
        np.linspace(2.0, 1.0, runner.HARD_EDGES_PER_AXIS),
        2,
    )
    labels = np.zeros(runner.HARD_EDGE_COUNT, dtype=bool)
    labels[:10] = True
    labels[runner.HARD_EDGES_PER_AXIS : runner.HARD_EDGES_PER_AXIS + 7] = True
    assert runner._fixed_budget_label_correct(scores, labels, axis) == 17


def test_pre_score_hashes_fail_closed_before_reference_access(tmp_path: Path) -> None:
    paths = runner._prepare_run_paths(tmp_path / "run")
    paths.cache_dir.mkdir()
    frozen = {
        "target_free_cache_metadata": paths.cache_dir / "metadata.json",
        "checkpoint": paths.checkpoint,
        "frozen_eval": paths.frozen_eval,
        "frozen_eval_metadata": paths.frozen_eval_metadata,
    }
    for name, path in frozen.items():
        path.write_text(name, encoding="utf-8")
    expected = {name: sha256_file(path) for name, path in frozen.items()}
    runner._validate_pre_score_hashes(paths, expected)
    paths.frozen_eval.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen_eval changed"):
        runner._validate_pre_score_hashes(paths, expected)


def test_device_contract_requires_explicit_mps_nondeterminism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runner._select_device("cpu", allow_nondeterministic_mps=False).type == "cpu"
    with pytest.raises(ValueError, match="requires MPS"):
        runner._select_device("cpu", allow_nondeterministic_mps=True)
    with pytest.raises(ValueError, match="requires --allow"):
        runner._select_device("mps", allow_nondeterministic_mps=False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        runner._select_device("mps", allow_nondeterministic_mps=True)


def _true_edges(axis: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(runner.COUNT, dtype=np.int32)
    if axis == 0:
        valid = positions % runner.GRID != runner.GRID - 1
        return positions[valid], positions[valid] + 1
    valid = positions < runner.COUNT - runner.GRID
    return positions[valid], positions[valid] + runner.GRID


def test_fixed_top288_uses_frozen_priorities_without_redecoding() -> None:
    prefix = "case_0000"
    archive: dict[str, np.ndarray] = {}
    for axis in (0, 1):
        source, target = _true_edges(axis)
        priority = np.linspace(2.0, 1.0, len(source))
        archive[f"{prefix}__axis_{axis}_source"] = source
        archive[f"{prefix}__axis_{axis}_target"] = target
        archive[f"{prefix}__axis_{axis}_baseline_priority"] = priority
        archive[f"{prefix}__axis_{axis}_learned_priority"] = priority
    observed = runner._fixed_top288_correct(
        archive,
        prefix,
        np.arange(runner.COUNT, dtype=np.int32),
        arm="learned",
    )
    assert observed == 2 * runner.DECODER_EDGE_BUDGET == 288


def test_source_clustered_ci_is_deterministic_over_two_draws() -> None:
    sources = [f"source-{index}" for index in range(16) for _ in runner.DRAWS]
    values = np.linspace(-1.0, 2.0, len(sources)).tolist()
    first = runner.source_clustered_delta_ci(values, sources, seed=7, resamples=1_000)
    second = runner.source_clustered_delta_ci(values, sources, seed=7, resamples=1_000)
    assert first == second
    assert first["source_count"] == 16
    assert first["case_count"] == 32
    assert first["ci95_lower"] < first["mean"] < first["ci95_upper"]
    with pytest.raises(ValueError, match="both registered draws"):
        runner.source_clustered_delta_ci(values[:-1], sources[:-1], seed=7)


def test_cli_has_only_selection_or_run_and_benchmark_is_opt_in() -> None:
    selection = runner.parse_args(["selection"])
    run = runner.parse_args(["run", "--benchmark-one-case"])
    assert selection.mode == "selection"
    assert selection.device == "mps"
    assert not selection.allow_nondeterministic_mps
    assert not selection.benchmark_one_case
    assert run.mode == "run" and run.benchmark_one_case
    with pytest.raises(SystemExit):
        runner.parse_args(["sweep"])
