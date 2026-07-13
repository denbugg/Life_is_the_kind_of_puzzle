from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "runs"
    / "assembly_v1"
    / "kaggle"
    / "qap_weight_confirmation_job"
    / "run_qap_weight_confirmation.py"
)
METADATA_PATH = RUNNER_PATH.with_name("kernel-metadata.json")


def load_runner():
    spec = importlib.util.spec_from_file_location("qap_weight_confirmation_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_evaluator():
    path = ROOT / "scripts" / "evaluate_qap_weight_confirmation.py"
    spec = importlib.util.spec_from_file_location(
        "qap_weight_confirmation_runner_integration_evaluator", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_envelope(runner, path: Path, payload: dict) -> str:
    envelope = {
        "payload": payload,
        "payload_sha256": runner.sha256_bytes(runner.canonical_bytes(payload)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.canonical_bytes(envelope) + b"\n")
    return runner.sha256(path)


def write_canonical(runner, path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.canonical_bytes(payload) + b"\n")


def phase_manifest(runner, rank: int, names: list[str]) -> dict:
    return {
        "schema_version": 1,
        "kind": "qap_weight_confirmation_phase_a_shard",
        "rank": rank,
        "world_size": 2,
        "target_paths_constructed": False,
        "target_files_opened": False,
        "records": [
            {"source_index": index, "name": names[index]}
            for index in range(rank, len(names), 2)
        ],
    }


def write_phase_dirs(
    tmp_path: Path, runner, names: list[str]
) -> tuple[list[Path], list[dict], list[str]]:
    directories = [tmp_path / "phase0", tmp_path / "phase1"]
    manifests = []
    anchors = []
    for rank, directory in enumerate(directories):
        payload = phase_manifest(runner, rank, names)
        for record in payload["records"]:
            stem = Path(record["name"]).stem
            variants = {}
            for key in ("baseline", "candidate"):
                layout_relative = f"artifacts/{stem}.{key}.layout.npy"
                render_relative = f"artifacts/{stem}.{key}.png"
                for relative in (layout_relative, render_relative):
                    artifact = directory / relative
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    artifact.write_bytes(f"{rank}:{relative}".encode("utf-8"))
                variants[key] = {
                    "layout_path": layout_relative,
                    "render_path": render_relative,
                }
            record["variants"] = variants
        anchors.append(
            write_envelope(runner, directory / runner.SHARD_MANIFEST_NAME, payload)
        )
        manifests.append(payload)
    return directories, manifests, anchors


def full_config(runner, names: list[str]) -> dict:
    return {
        "assets": {
            "denoiser": "denoiser.pt",
            "denoiser_sha256": "a" * 64,
            "hbt": "hbt.pt",
            "hbt_sha256": "b" * 64,
            "manifest": "manifest.json",
            "manifest_sha256": "c" * 64,
            "quarantine": "quarantine.json",
            "quarantine_sha256": "d" * 64,
        },
        "common_solver": {"qap_iterations": 25},
        "baseline": {"label": "base", "score": "w4", "hbt_weight": 4.0},
        "candidate": {"label": "candidate", "score": "w1", "hbt_weight": 1.0},
        "original_real_confirmation": {
            "split": "assembly_incremental_gate",
            "offset": 128,
            "count": 64,
            "names_sha256": runner.names_sha256(names),
            "metric": {
                "tie_tolerance": 1e-12,
                "bootstrap_seed": 20260711,
                "bootstrap_resamples": 20000,
                "bootstrap_quantiles": [0.025, 0.975],
            },
            "gate": {
                "logic": "all_of",
                "mean_ssim_delta_min": 0.005,
                "ssim_bootstrap_95_lower_gt": 0.0,
                "wins_min": 40,
                "large_regressions_max": 6,
                "valid_permutation_count": 64,
            },
            "post_phase_b_mutation_policy": "no_retuning",
        },
    }


def asset_records(config: dict, root: Path) -> dict:
    return {
        label: {
            "path": str((root / f"{label}.asset").resolve()),
            "sha256": config["assets"][f"{label}_sha256"],
            "configured_path": config["assets"][label],
        }
        for label in ("denoiser", "hbt", "manifest", "quarantine")
    }


def valid_report(
    runner,
    config_path: Path,
    config: dict,
    evaluator: Path,
    assets: dict,
    names: list[str],
    finalized_manifest: Path,
    target_marker: Path,
) -> dict:
    # Forty clear wins, six small losses, and eighteen ties pass every fixed gate.
    deltas = [0.01] * 40 + [-0.001] * 6 + [0.0] * 17 + [5e-13]
    values = runner.np.asarray(deltas, dtype=runner.np.float64)
    metric = config["original_real_confirmation"]["metric"]
    rng = runner.np.random.default_rng(metric["bootstrap_seed"])
    indices = rng.integers(0, 64, size=(metric["bootstrap_resamples"], 64))
    interval = runner.np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()
    finalized_payload, finalized_sha = runner.load_exact_envelope(finalized_manifest)
    finalized_payload_sha = runner.sha256_bytes(runner.canonical_bytes(finalized_payload))
    marker_payload = {
        "schema_version": 1,
        "kind": "qap_weight_confirmation_target_access_event",
        "phase_a_manifest_sha256": finalized_sha,
        "phase_a_payload_sha256": finalized_payload_sha,
        "config_sha256": runner.sha256(config_path),
        "code_sha256": runner.sha256(evaluator),
        "source_names_sha256": runner.names_sha256(names),
        "target_access_started": True,
        "target_files_may_have_been_opened": True,
    }
    marker_sha = write_envelope(runner, target_marker, marker_payload)
    baseline_scores = [0.2] * 64
    candidate_scores = [base + delta for base, delta in zip(baseline_scores, deltas)]
    records = [
        {
            "source_index": index,
            "name": names[index],
            "baseline_ssim": baseline_scores[index],
            "candidate_ssim": candidate_scores[index],
            "delta_ssim": deltas[index],
            "baseline_layout_sha256": finalized_payload["records"][index]["variants"][
                "baseline"
            ]["layout_value_sha256"],
            "candidate_layout_sha256": finalized_payload["records"][index]["variants"][
                "candidate"
            ]["layout_value_sha256"],
        }
        for index in range(64)
    ]
    return {
        "schema_version": 1,
        "kind": "qap_weight_confirmation_report",
        "status": "promotion_gate_passed",
        "safe_for_submission": False,
        "eligible_for_final_audit": True,
        "protocol": {
            "config_path": str(config_path.resolve()),
            "config_sha256": runner.sha256(config_path),
            "split": "assembly_incremental_gate",
            "offset": 128,
            "count": 64,
            "source_names_sha256": runner.names_sha256(names),
        },
        "config": {"path": str(config_path.resolve()), "sha256": runner.sha256(config_path)},
        "code": {"path": str(evaluator.resolve()), "sha256": runner.sha256(evaluator)},
        "assets": assets,
        "split": "assembly_incremental_gate[128:192]",
        "source_names": names,
        "source_names_sha256": runner.names_sha256(names),
        "baseline": config["baseline"],
        "candidate": config["candidate"],
        "common_solver": config["common_solver"],
        "solver": {
            "common": config["common_solver"],
            "baseline": config["baseline"],
            "candidate": config["candidate"],
        },
        "phase_a": {
            "manifest": str(finalized_manifest.resolve()),
            "manifest_sha256": finalized_sha,
            "payload_sha256": finalized_payload_sha,
            "integrity_before_sha256": "e" * 64,
            "integrity_after_sha256": "e" * 64,
            "source_count": 64,
            "source_names_sha256": runner.names_sha256(names),
            "shards": finalized_payload.get("shards"),
        },
        "target_access": {
            "marker": str(target_marker.resolve()),
            "marker_sha256": marker_sha,
            "marker_payload_sha256": runner.sha256_bytes(
                runner.canonical_bytes(marker_payload)
            ),
            "marker_preceded_first_target_path_construction": True,
            "target_access_count": 64,
        },
        "phase_b": {
            "marker": str(target_marker.resolve()),
            "marker_sha256": marker_sha,
            "marker_payload_sha256": runner.sha256_bytes(
                runner.canonical_bytes(marker_payload)
            ),
            "marker_preceded_first_target_path_construction": True,
            "target_access_count": 64,
            "integrity_before_sha256": "e" * 64,
            "integrity_after_sha256": "e" * 64,
            "post_score_rehash_matched": True,
        },
        "metric": metric,
        "records": records,
        "aggregate": {
            "source_count": 64,
            "bootstrap_unit": "paired_whole_source_delta_candidate_minus_baseline",
            "mean_baseline_ssim": float(runner.np.mean(baseline_scores)),
            "mean_candidate_ssim": float(runner.np.mean(candidate_scores)),
            "mean_ssim_delta": float(values.mean()),
            "median_ssim_delta": float(runner.np.median(values)),
            "wins": 40,
            "losses": 6,
            "ties": 18,
            "large_regressions": 0,
            "valid_permutation_count": 64,
            "bootstrap_95_ci": interval,
        },
        "paired_metrics": {
            "source_count": 64,
            "bootstrap_unit": "paired_whole_source_delta_candidate_minus_baseline",
            "mean_baseline_ssim": float(runner.np.mean(baseline_scores)),
            "mean_candidate_ssim": float(runner.np.mean(candidate_scores)),
            "mean_ssim_delta": float(values.mean()),
            "median_ssim_delta": float(runner.np.median(values)),
            "wins": 40,
            "losses": 6,
            "ties": 18,
            "large_regressions": 0,
            "valid_permutation_count": 64,
            "bootstrap_95_ci": interval,
        },
        "gate": {
            "logic": "all_of",
            "checks": {
                "mean_ssim_delta_ge_0.005": True,
                "bootstrap_95_lower_gt_0": True,
                "wins_ge_40": True,
                "large_regressions_le_6": True,
                "valid_permutation_count_eq_64": True,
            },
            "passed": True,
        },
        "sealed_sets": {
            "final_audit_opened": False,
            "confirmation_audit_opened": False,
            "must_remain_unopened": True,
        },
        "post_phase_b_mutation_policy": "no_retuning",
    }


def test_kernel_metadata_is_private_offline_exact_t4x2_job() -> None:
    runner = load_runner()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert runner.sha256(METADATA_PATH) == runner.EXPECTED_KERNEL_METADATA_SHA256
    assert metadata["id"] == runner.KERNEL_ID
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == runner.DATASET_SOURCES == [
        "pasha883/vsos-ai-initiative-pazzle",
        "pasha883/vsos-assembly-v1-runtime",
        "pasha883/vsos-solver-rework-night-code",
        "pasha883/vsos-qap-weight-confirmation-code",
    ]


def test_exact_overlay_zip_and_base_archive_stage_with_full_hash_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner()
    base_mount = tmp_path / "base_mount"
    overlay_mount = tmp_path / "overlay_mount"
    base_mount.mkdir()
    overlay_mount.mkdir()
    shutil.copy2(
        ROOT
        / "runs"
        / "assembly_v1"
        / "kaggle"
        / "solver_rework_night_code_dataset"
        / "solver_rework_code.zip",
        base_mount / "solver_rework_code.zip",
    )
    overlay_archive = (
        ROOT
        / "runs"
        / "assembly_v1"
        / "kaggle"
        / "qap_weight_confirmation_code_dataset"
        / "qap_weight_confirmation_code.zip"
    )
    assert runner.sha256(overlay_archive) == runner.EXPECTED_OVERLAY_ARCHIVE_SHA256
    shutil.copy2(overlay_archive, overlay_mount / overlay_archive.name)
    monkeypatch.setattr(runner, "BASE_INPUT", base_mount)
    monkeypatch.setattr(runner, "OVERLAY_INPUT", overlay_mount)
    monkeypatch.setattr(runner, "STAGING", tmp_path / "staging")
    code_root, provenance = runner.stage_code()
    assert provenance["overlay_source"]["mode"] == "pinned_archive"
    assert provenance["overlay_source"]["sha256"] == runner.EXPECTED_OVERLAY_ARCHIVE_SHA256
    assert set(provenance["staged_overlay_hashes"]) == set(runner.EXPECTED_OVERLAY_HASHES)
    runner.exact_hashes(code_root, runner.EXPECTED_BASE_HASHES, "test staged base")
    runner.exact_hashes(code_root, runner.EXPECTED_OVERLAY_HASHES, "test staged overlay")


def test_exact_direct_kaggle_overlay_files_stage_with_same_hash_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner()
    base_mount = tmp_path / "base_mount"
    overlay_mount = tmp_path / "overlay_mount"
    base_mount.mkdir()
    overlay_mount.mkdir()
    shutil.copy2(
        ROOT
        / "runs"
        / "assembly_v1"
        / "kaggle"
        / "solver_rework_night_code_dataset"
        / "solver_rework_code.zip",
        base_mount / "solver_rework_code.zip",
    )
    overlay_archive = (
        ROOT
        / "runs"
        / "assembly_v1"
        / "kaggle"
        / "qap_weight_confirmation_code_dataset"
        / "qap_weight_confirmation_code.zip"
    )
    with zipfile.ZipFile(overlay_archive) as handle:
        handle.extractall(overlay_mount)
    monkeypatch.setattr(runner, "BASE_INPUT", base_mount)
    monkeypatch.setattr(runner, "OVERLAY_INPUT", overlay_mount)
    monkeypatch.setattr(runner, "STAGING", tmp_path / "staging")
    code_root, provenance = runner.stage_code()
    assert provenance["overlay_source"] == {"mode": "direct_dataset_files"}
    assert set(provenance["staged_overlay_hashes"]) == set(runner.EXPECTED_OVERLAY_HASHES)
    runner.exact_hashes(code_root, runner.EXPECTED_BASE_HASHES, "test staged base")
    runner.exact_hashes(code_root, runner.EXPECTED_OVERLAY_HASHES, "test staged overlay")


def test_interleaved_phase_a_is_reconstructed_by_canonical_index(tmp_path: Path) -> None:
    runner = load_runner()
    names = [f"img_{index:06d}.png" for index in range(64)]
    directories, manifests, anchors = write_phase_dirs(tmp_path, runner, names)
    config = full_config(runner, names)
    actual_manifests, actual_names, actual_anchors = runner.validate_phase_a_manifests(
        directories, config
    )
    assert actual_manifests == manifests
    assert actual_names == names
    assert actual_anchors == anchors


@pytest.mark.parametrize("mutation", ["gap", "duplicate", "wrong_parity"])
def test_phase_a_rejects_index_partition_attacks(tmp_path: Path, mutation: str) -> None:
    runner = load_runner()
    names = [f"img_{index:06d}.png" for index in range(64)]
    directories, _, _ = write_phase_dirs(tmp_path, runner, names)
    path = directories[1] / runner.SHARD_MANIFEST_NAME
    payload, _ = runner.load_exact_envelope(path)
    if mutation == "gap":
        payload["records"].pop()
    elif mutation == "duplicate":
        payload["records"][-1]["source_index"] = payload["records"][0]["source_index"]
    else:
        payload["records"][0]["source_index"] = 2
    write_envelope(runner, path, payload)
    with pytest.raises(RuntimeError):
        runner.validate_phase_a_manifests(directories, full_config(runner, names))


def test_finalize_manifest_requires_canonical_order_and_shard_anchors(tmp_path: Path) -> None:
    runner = load_runner()
    names = [f"img_{index:06d}.png" for index in range(64)]
    phase_dirs, phase_payloads, anchors = write_phase_dirs(tmp_path, runner, names)
    config = full_config(runner, names)
    config_path = tmp_path / "config.json"
    atomic_json(config_path, config)
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    assets = {
        label: {
            "sha256": record["sha256"],
            "configured_path": record["configured_path"],
        }
        for label, record in asset_records(config, tmp_path).items()
    }
    finalized = tmp_path / "finalized"
    records = sorted(
        [record for payload in phase_payloads for record in payload["records"]],
        key=lambda record: record["source_index"],
    )
    payload = {
        "schema_version": 1,
        "kind": "qap_weight_confirmation_finalized_phase_a",
        "config_path": str(config_path.resolve()),
        "config_sha256": runner.sha256(config_path),
        "code_path": str(evaluator.resolve()),
        "code_sha256": runner.sha256(evaluator),
        "assets": assets,
        "split": "assembly_incremental_gate[128:192]",
        "source_names": names,
        "source_names_sha256": runner.names_sha256(names),
        "source_count": 64,
        "artifact_root": "artifacts",
        "common_solver": config["common_solver"],
        "baseline": config["baseline"],
        "candidate": config["candidate"],
        "shards": [
            {
                "rank": rank,
                "world_size": 2,
                "manifest_path": str((phase_dirs[rank] / runner.SHARD_MANIFEST_NAME).resolve()),
                "manifest_sha256": anchors[rank],
                "payload_sha256": runner.sha256_bytes(
                    runner.canonical_bytes(phase_payloads[rank])
                ),
                "artifact_snapshot_sha256": runner.sha256_bytes(
                    runner.canonical_bytes(
                        {
                            Path(variant[path_key]).as_posix(): runner.sha256(
                                phase_dirs[rank] / Path(variant[path_key])
                            )
                            for record in phase_payloads[rank]["records"]
                            for variant in record["variants"].values()
                            for path_key in ("layout_path", "render_path")
                        }
                    )
                ),
            }
            for rank in range(2)
        ],
        "target_paths_constructed": False,
        "target_files_opened": False,
        "final_audit_opened": False,
        "confirmation_audit_opened": False,
        "records": records,
    }
    envelope = write_envelope(runner, finalized / runner.GLOBAL_MANIFEST_NAME, payload)
    actual, envelope = runner.validate_finalized_manifest(
        finalized,
        config=config,
        config_path=config_path,
        evaluator=evaluator,
        asset_records=assets,
        phase_dirs=phase_dirs,
        phase_payloads=phase_payloads,
        canonical_names=names,
        shard_envelopes=anchors,
    )
    assert actual == payload
    assert envelope == runner.sha256(finalized / runner.GLOBAL_MANIFEST_NAME)

    attacked = json.loads(json.dumps(payload))
    attacked["records"][0], attacked["records"][1] = (
        attacked["records"][1],
        attacked["records"][0],
    )
    write_envelope(runner, finalized / runner.GLOBAL_MANIFEST_NAME, attacked)
    with pytest.raises(RuntimeError):
        runner.validate_finalized_manifest(
            finalized,
            config=config,
            config_path=config_path,
            evaluator=evaluator,
            asset_records=assets,
            phase_dirs=phase_dirs,
            phase_payloads=phase_payloads,
            canonical_names=names,
            shard_envelopes=anchors,
        )


def test_report_gate_recomputes_counts_ties_and_bootstrap(tmp_path: Path) -> None:
    runner = load_runner()
    names = [f"img_{index:06d}.png" for index in range(64)]
    config = full_config(runner, names)
    config_path = tmp_path / "config.json"
    atomic_json(config_path, config)
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    assets = asset_records(config, tmp_path)
    finalized_manifest = tmp_path / "finalized" / runner.GLOBAL_MANIFEST_NAME
    finalized_payload = {
        "records": [
            {
                "source_index": index,
                "name": name,
                "variants": {
                    "baseline": {
                        "layout_value_sha256": f"{index + 1:064x}",
                        "valid_permutation": True,
                    },
                    "candidate": {
                        "layout_value_sha256": f"{index + 101:064x}",
                        "valid_permutation": True,
                    },
                },
            }
            for index, name in enumerate(names)
        ]
    }
    write_envelope(runner, finalized_manifest, finalized_payload)
    target_marker = finalized_manifest.with_name(runner.TARGET_MARKER_NAME)
    report = valid_report(
        runner,
        config_path,
        config,
        evaluator,
        assets,
        names,
        finalized_manifest,
        target_marker,
    )
    report_path = tmp_path / runner.REPORT_NAME
    write_canonical(runner, report_path, report)
    summary = runner.validate_final_report(
        report_path,
        config_path=config_path,
        evaluator=evaluator,
        asset_records=assets,
        finalized_manifest_path=finalized_manifest,
        target_marker_path=target_marker,
        combined_names=names,
    )
    assert summary["gate_passed"] is True

    for mutator in (
        lambda payload: payload["aggregate"].__setitem__("wins", 41),
        lambda payload: payload["aggregate"]["bootstrap_95_ci"].__setitem__(0, -0.5),
        lambda payload: payload["gate"]["checks"].__setitem__("large_regressions_le_6", False),
    ):
        attacked = json.loads(json.dumps(report))
        mutator(attacked)
        write_canonical(runner, report_path, attacked)
        with pytest.raises(RuntimeError):
            runner.validate_final_report(
                report_path,
                config_path=config_path,
                evaluator=evaluator,
                asset_records=assets,
                finalized_manifest_path=finalized_manifest,
                target_marker_path=target_marker,
                combined_names=names,
            )


def test_actual_evaluator_three_phase_output_passes_every_runner_validator(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    evaluator = load_evaluator()
    config_path = ROOT / "configs" / "qap_weight_confirmation_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    probe_args = evaluator.parse_args(
        ["--action", "phase-a", "--config", str(config_path)]
    )
    protocol, full_assets = evaluator._validated_protocol_and_assets(probe_args)
    names = evaluator._expected_names(protocol, full_assets)
    assert len(names) == 64
    portable_assets = {
        label: {
            "sha256": record["sha256"],
            "configured_path": record["configured_path"],
        }
        for label, record in full_assets.items()
    }

    data_root = tmp_path / "puzzle"
    input_dir = data_root / "train" / "inputs"
    input_dir.mkdir(parents=True)
    zero_image = runner.np.zeros((480, 480, 3), dtype=runner.np.uint8)
    png = evaluator._png_bytes(zero_image)
    for name in names:
        (input_dir / name).write_bytes(png)
    assert sorted(path.name for path in (data_root / "train").iterdir()) == ["inputs"]
    identity = runner.np.arange(576, dtype=runner.np.int32)

    def fake_predictor(name: str, image):
        return evaluator.PhaseAPrediction(
            layouts={"baseline": identity.copy(), "candidate": identity.copy()},
            renders={"baseline": image.copy(), "candidate": image.copy()},
            initial_layout=identity.copy(),
            qap_seed=evaluator._filename_qap_seed(name),
            denoised_tiles_sha256=evaluator._bytes_sha256(image.tobytes()),
            diagnostics={"baseline": {}, "candidate": {}},
        )

    phase_dirs = [tmp_path / "phase0", tmp_path / "phase1"]
    anchors = []
    for rank, phase_dir in enumerate(phase_dirs):
        args = evaluator.parse_args(
            [
                "--action",
                "phase-a",
                "--config",
                str(config_path),
                "--rank",
                str(rank),
                "--world-size",
                "2",
                "--phase-a-dir",
                str(phase_dir),
                "--data-root",
                str(data_root),
            ]
        )
        result = evaluator.run_phase_a(args, predictor=fake_predictor)
        anchors.append(result["phase_a_envelope_sha256"])

    phase_payloads, canonical_names, validated_anchors = runner.validate_phase_a_manifests(
        phase_dirs, config
    )
    assert canonical_names == names
    assert validated_anchors == anchors
    runner.validate_phase_a_artifacts(
        payloads=phase_payloads,
        phase_dirs=phase_dirs,
        config=config,
        config_path=config_path,
        evaluator=ROOT / "scripts" / "evaluate_qap_weight_confirmation.py",
        data_root=data_root,
        asset_records=portable_assets,
    )

    finalized_dir = tmp_path / "finalized"
    finalize_args = evaluator.parse_args(
        [
            "--action",
            "finalize-phase-a",
            "--config",
            str(config_path),
            "--phase-a-dirs",
            *(str(path) for path in phase_dirs),
            "--phase-a-envelope-sha256s",
            *anchors,
            "--finalized-phase-a-dir",
            str(finalized_dir),
        ]
    )
    finalized_result = evaluator.run_finalize_phase_a(finalize_args)
    finalized_payload, finalized_anchor = runner.validate_finalized_manifest(
        finalized_dir,
        config=config,
        config_path=config_path,
        evaluator=ROOT / "scripts" / "evaluate_qap_weight_confirmation.py",
        asset_records=portable_assets,
        phase_dirs=phase_dirs,
        phase_payloads=phase_payloads,
        canonical_names=names,
        shard_envelopes=anchors,
    )
    assert finalized_anchor == finalized_result["phase_a_envelope_sha256"]
    runner.validate_finalized_artifacts(
        finalized_dir=finalized_dir,
        payload=finalized_payload,
        phase_dirs=phase_dirs,
    )

    target_dir = data_root / "train" / "targets"
    target_dir.mkdir()
    for name in names:
        (target_dir / name).write_bytes(png)
    report_path = tmp_path / runner.REPORT_NAME
    phase_b_args = evaluator.parse_args(
        [
            "--action",
            "phase-b",
            "--config",
            str(config_path),
            "--data-root",
            str(data_root),
            "--finalized-phase-a-dir",
            str(finalized_dir),
            "--phase-a-envelope-sha256",
            finalized_anchor,
            "--output",
            str(report_path),
        ]
    )
    evaluator.run_phase_b(phase_b_args)
    summary = runner.validate_final_report(
        report_path,
        config_path=config_path,
        evaluator=ROOT / "scripts" / "evaluate_qap_weight_confirmation.py",
        asset_records=full_assets,
        finalized_manifest_path=finalized_dir / runner.GLOBAL_MANIFEST_NAME,
        target_marker_path=finalized_dir / runner.TARGET_MARKER_NAME,
        combined_names=names,
    )
    assert summary["status"] == "promotion_gate_failed"
    assert summary["gate_passed"] is False
    assert summary["safe_for_submission"] is False


def test_frozen_phase_a_archive_is_deterministic_and_byte_verifiable(tmp_path: Path) -> None:
    runner = load_runner()
    names = [f"img_{index:06d}.png" for index in range(64)]
    directories, _, _ = write_phase_dirs(tmp_path, runner, names)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    first_record = runner.write_deterministic_phase_a_archive(
        first, [("shard0", directories[0]), ("shard1", directories[1])]
    )
    second_record = runner.write_deterministic_phase_a_archive(
        second, [("shard0", directories[0]), ("shard1", directories[1])]
    )
    assert first_record["sha256"] == second_record["sha256"]
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [record["path"] for record in first_record["members"]]
        assert all(info.date_time == runner.ARCHIVE_TIMESTAMP for info in archive.infolist())


def test_finalized_archive_revalidates_after_relocation_from_extracted_bytes_only(
    tmp_path: Path,
) -> None:
    runner = load_runner()
    finalized = tmp_path / "original_absolute_location" / "finalized"
    artifacts = finalized / "artifacts"
    artifacts.mkdir(parents=True)
    names = [f"img_{index:06d}.png" for index in range(64)]
    layout = runner.np.arange(576, dtype=runner.np.int32)
    layout_buffer = runner.BytesIO()
    runner.np.save(layout_buffer, layout, allow_pickle=False)
    layout_bytes = layout_buffer.getvalue()
    image_buffer = runner.BytesIO()
    runner.Image.fromarray(
        runner.np.zeros((480, 480, 3), dtype=runner.np.uint8), mode="RGB"
    ).save(image_buffer, format="PNG", compress_level=6)
    image_bytes = image_buffer.getvalue()
    records = []
    for source_index, name in enumerate(names):
        stem = Path(name).stem
        variants = {}
        for key in ("baseline", "candidate"):
            layout_relative = f"artifacts/{stem}.{key}.layout.npy"
            render_relative = f"artifacts/{stem}.{key}.png"
            (finalized / layout_relative).write_bytes(layout_bytes)
            (finalized / render_relative).write_bytes(image_bytes)
            variants[key] = {
                "layout_path": layout_relative,
                "layout_sha256": runner.sha256(finalized / layout_relative),
                "layout_value_sha256": runner.sha256_bytes(layout.tobytes()),
                "render_path": render_relative,
                "render_sha256": runner.sha256(finalized / render_relative),
            }
        records.append(
            {
                "source_index": source_index,
                "name": name,
                "input_path": f"train/inputs/{name}",
                "variants": variants,
            }
        )
    payload = {
        "artifact_root": "artifacts",
        "source_names_sha256": runner.names_sha256(names),
        "records": records,
    }
    manifest = finalized / runner.GLOBAL_MANIFEST_NAME
    manifest_sha = write_envelope(runner, manifest, payload)
    archive = tmp_path / "frozen.zip"
    runner.write_deterministic_phase_a_archive(archive, [("finalized", finalized)])
    relocated_root = runner.safe_extract(archive, tmp_path / "different_absolute_location")
    actual, actual_sha = runner.validate_relocated_finalized_tree(
        relocated_root / "finalized", expected_manifest_sha256=manifest_sha
    )
    assert actual == payload
    assert actual_sha == manifest_sha


def test_phase_b_marker_is_the_only_allowed_finalized_tree_addition(tmp_path: Path) -> None:
    runner = load_runner()
    finalized = tmp_path / "finalized"
    atomic_json(finalized / runner.GLOBAL_MANIFEST_NAME, {"frozen": True})
    before = runner.freeze_tree(finalized, keep_root_writable=True)
    atomic_json(finalized / runner.TARGET_MARKER_NAME, {"started": True})
    assert runner.tree_sha256(
        finalized, ignore_names={runner.TARGET_MARKER_NAME}
    ) == before
    (finalized / "unexpected.txt").write_text("mutation", encoding="utf-8")
    assert runner.tree_sha256(
        finalized, ignore_names={runner.TARGET_MARKER_NAME}
    ) != before


def test_input_only_root_physically_omits_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner()
    data_root = tmp_path / "puzzle"
    (data_root / "train" / "inputs").mkdir(parents=True)
    (data_root / "train" / "targets").mkdir(parents=True)
    monkeypatch.setattr(runner, "STAGING", tmp_path / "staging")
    root = runner.make_input_only_data_root(data_root)
    assert (root / "train" / "inputs").is_symlink()
    assert not (root / "train" / "targets").exists()
