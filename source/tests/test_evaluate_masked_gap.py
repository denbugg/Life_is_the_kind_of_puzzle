from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.masked_gap import (
    MaskedGapGenerator,
    PairListwiseRanker,
    module_state_sha256,
    state_dict_payload,
)
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_evaluate_masked_gap as gate


def test_frozen_protocol_names_hashes_and_exposure_disclosure() -> None:
    audit = gate.protocol_audit(
        manifest=REPO_ROOT / "configs/denoise_splits_seed20260710.json",
        quarantine=REPO_ROOT / "configs/denoise_validation_quarantine_v1.json",
    )
    assert audit["exact_basename_audit"] is True
    assert audit["all_splits_pairwise_disjoint"] is True
    assert audit["all_112_vs_quarantine_audit_assembly_hbt_intersections_empty"] is True
    assert all(not values for values in audit["exposure_intersections"].values())
    assert audit["hbt_exposure_names"] == 2080
    assert "not fully source-unseen" in audit["upstream_exposure_disclosure"]
    for key, (_, _, count, expected_hash) in gate.FROZEN_SPLITS.items():
        assert audit["splits"][key]["count"] == count
        assert audit["splits"][key]["names_sha256"] == expected_hash
    assert audit["forbidden_historical_artifacts"] == ["spatial-prior", "l1-real-pseudo"]


def test_phase_a_cli_has_no_label_or_data_target_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "masked-gap",
            "phase-a",
            "--input-manifest",
            "inputs.json",
            "--checkpoint",
            "model.pt",
            "--output-dir",
            "phase-a",
        ],
    )
    args = gate.parse_args()
    assert args.command == "phase-a"
    for forbidden in ("label_manifest", "data_root", "manifest", "quarantine"):
        assert not hasattr(args, forbidden)
    assert gate.INPUT_KEYS.isdisjoint(gate.LABEL_KEYS)
    assert not any(
        token in key
        for key in gate.INPUT_KEYS
        for token in ("target", "label", "clean", "truth", "slot_to_target")
    )


def test_authorization_is_hash_bound_and_rejects_changed_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "scores.npz"
    matrix = np.zeros((576, 576), dtype=np.float32)
    np.fill_diagonal(matrix, np.inf)
    arrays = {f"m{index}": matrix for index in range(10)}
    gate.atomic_npz(artifact, **arrays)
    matrix_manifest = {key: gate.matrix_fingerprint(value) for key, value in arrays.items()}
    report_path = tmp_path / "phase_a_report.json"
    report = {
        "kind": gate.PHASE_A_KIND,
        "split": "calibration_b",
        "target_metrics_opened": False,
        "labels_or_targets_loaded": False,
        "qap_run": False,
        "artifact_sha256": gate.sha256(artifact),
        "input_manifest_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "records": [{"index": 0}],
        "panels": list(gate.PANELS),
        "source_names_sha256": "3" * 64,
        "upstream_asset_sha256": {"restorer": "4" * 64, "embedding": "5" * 64},
        "evaluator_code_sha256": gate.sha256(Path(gate.__file__)),
        "core_code_sha256": gate.sha256(REPO_ROOT / "src/puzzle_assembly/masked_gap.py"),
        "scientific_config_sha256": "6" * 64,
        "generator_state_sha256": "7" * 64,
        "training_ledger_sha256": "8" * 64,
        "matrix_manifest": matrix_manifest,
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "authorization.json"
    args = type(
        "Args",
        (),
        {
            "phase_a_report": str(report_path),
            "phase_a_artifact": str(artifact),
            "output": str(output),
        },
    )()
    gate.authorize_command(args)
    authorization = json.loads(output.read_text(encoding="utf-8"))
    assert authorization["phase_b_authorized"] is True
    assert authorization["phase_a_artifact_sha256"] == gate.sha256(artifact)
    artifact.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        gate.authorize_command(args)


def _passing_record(source_index: int, panel: str) -> dict:
    return {
        "name": f"img_{source_index:06d}.png",
        "panel": panel,
        "metrics": {
            "w4": {"mrr": 0.10, "recall_at_1": 0.10, "recall_at_5": 0.20},
            "direct_only": {"mrr": 0.09, "recall_at_1": 0.08, "recall_at_5": 0.18},
            "inpaint_only": {"mrr": 0.10, "recall_at_1": 0.09, "recall_at_5": 0.19},
            "direct_control": {"mrr": 0.110, "recall_at_1": 0.105, "recall_at_5": 0.21},
            "candidate": {"mrr": 0.115, "recall_at_1": 0.110, "recall_at_5": 0.220},
        },
        "reconstruction": {
            "generator_charbonnier": 0.095,
            "copy_charbonnier": 0.100,
            "interpolation_charbonnier": 0.110,
            "generator_mae": 0.095,
            "copy_mae": 0.100,
            "interpolation_mae": 0.110,
        },
    }


def test_gate_applies_all_frozen_thresholds_per_panel_and_control() -> None:
    records = [
        _passing_record(source, panel)
        for source in range(8)
        for panel in gate.PANELS
    ]
    decision = gate.gate_decision(records, final_holdout=True)
    assert decision["passed"] is True
    assert decision["source_mean_over_two_panels_mrr_wins"] == 8
    assert all(decision["conditions"].values())

    records[0]["reconstruction"]["generator_mae"] = 0.096
    failed = gate.gate_decision(records, final_holdout=True)
    assert failed["passed"] is False
    assert failed["conditions"][
        "reconstruction_charbonnier_and_mae_5pct_better_each_control_each_panel"
    ] is False
    with pytest.raises(RuntimeError, match="exactly 16"):
        gate.gate_decision(records[:7], final_holdout=True)

    five_win_records = [
        _passing_record(source, panel)
        for source in range(8)
        for panel in gate.PANELS
    ]
    for record in five_win_records:
        source = int(Path(record["name"]).stem.split("_")[1])
        record["metrics"]["direct_control"]["mrr"] = 0.090
        record["metrics"]["candidate"]["mrr"] = 0.099 if source < 3 else 0.130
    five_win = gate.gate_decision(five_win_records, final_holdout=True)
    assert five_win["source_mean_over_two_panels_mrr_wins"] == 5
    assert five_win["conditions"]["source_mean_over_two_panels_mrr_wins_ge_6_of_8"] is False


def test_dense_score_sign_coverage_dtype_and_chunk_equivalence(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_logits(
        ranker, generator, raw, denoised, first, second, direction, *, inpaint
    ):
        values = (
            first.astype(np.float32) * 1e-3
            + second.astype(np.float32) * 1e-4
            + direction.astype(np.float32) * 1e-2
            + (1.0 if inpaint else 2.0)
        )
        return torch.from_numpy(values).to(raw.device)

    monkeypatch.setattr(gate, "_ranker_logits", fake_logits)
    tiles = np.zeros((576, 20, 20, 3), dtype=np.uint8)
    model = nn.Identity()
    first = gate.dense_scores(
        model, model, model, tiles, tiles,
        device=torch.device("cpu"), query_chunk=64, pair_chunk=100_000, amp=False,
    )
    second = gate.dense_scores(
        model, model, model, tiles, tiles,
        device=torch.device("cpu"), query_chunk=37, pair_chunk=17_777, amp=False,
    )
    for first_score, second_score in zip(first, second):
        for side in ("right", "down"):
            left = getattr(first_score, side)
            right = getattr(second_score, side)
            assert left.dtype == np.float32 and left.shape == (576, 576)
            assert np.array_equal(left, right)
            assert np.all(np.isposinf(np.diag(left)))
            assert np.all(np.isfinite(left[~np.eye(576, dtype=bool)]))
    assert first[0].right[3, 4] == pytest.approx(-(1.0 + 3e-3 + 4e-4))
    assert first[1].down[3, 4] == pytest.approx(-(2.0 + 3e-3 + 4e-4 + 1e-2))


def test_dense_scores_forward_exactly_575_nonself_candidates_per_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forwarded = {"inpaint": 0, "direct": 0}

    def fake_logits(
        ranker, generator, raw, denoised, first, second, direction, *, inpaint
    ):
        assert np.all(first != second)
        forwarded["inpaint" if inpaint else "direct"] += len(first)
        return torch.zeros(len(first), device=raw.device)

    monkeypatch.setattr(gate, "_ranker_logits", fake_logits)
    tiles = np.zeros((576, 20, 20, 3), dtype=np.uint8)
    gate.dense_scores(
        nn.Identity(), nn.Identity(), nn.Identity(), tiles, tiles,
        device=torch.device("cpu"), query_chunk=37, pair_chunk=17_777, amp=False,
    )
    assert forwarded == {
        "inpaint": 2 * 576 * 575,
        "direct": 2 * 576 * 575,
    }


def test_phase_a_open_spy_never_reads_label_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    names = gate.frozen_names(
        "calibration_b",
        manifest=REPO_ROOT / "configs/denoise_splits_seed20260710.json",
        quarantine=REPO_ROOT / "configs/denoise_validation_quarantine_v1.json",
    )
    input_dir = tmp_path / "input_only"
    label_dir = tmp_path / "labels_forbidden"
    input_dir.mkdir()
    label_dir.mkdir()
    (label_dir / "never-open.labels.npz").write_bytes(b"sentinel")
    matrix = np.zeros((576, 576), dtype=np.float32)
    np.fill_diagonal(matrix, np.inf)
    records = []
    for name in names:
        for panel in gate.PANELS:
            path = input_dir / f"{Path(name).stem}__{panel}.input.npz"
            gate.atomic_npz(
                path,
                raw_tiles=np.zeros((576, 20, 20, 3), dtype=np.uint8),
                denoised_tiles=np.zeros((576, 20, 20, 3), dtype=np.uint8),
                w4_right=matrix,
                w4_down=matrix,
            )
            records.append({"name": name, "panel": panel, "file": path.name, "sha256": gate.sha256(path)})
    manifest = {
        "kind": gate.INPUT_MANIFEST_KIND,
        "split": "calibration_b",
        "names": names,
        "names_sha256": gate.names_sha256(names),
        "panels": list(gate.PANELS),
        "records": records,
        "allowed_npz_keys": sorted(gate.INPUT_KEYS),
        "target_or_label_fields_attached": False,
        "panel_seed_attached": False,
        "panel_seed_derivation_available": False,
        "upstream_asset_sha256": {"restorer": "a" * 64, "embedding": "b" * 64},
    }
    manifest_path = input_dir / "input_manifest.json"
    gate.atomic_json(manifest_path, manifest)

    generator = MaskedGapGenerator(width=16, blocks=1)
    inpaint = PairListwiseRanker(width=16, blocks=1)
    direct = PairListwiseRanker(width=16, blocks=1)
    config = gate.frozen_scientific_config(width=16, generator_blocks=1, ranker_blocks=1)
    generator_hash = module_state_sha256(generator)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        state_dict_payload(
            generator, inpaint, direct,
            metadata={
                "safe_for_submission": False,
                "scientific_config": config,
                "scientific_config_sha256": gate.canonical_hash(config),
                "generator_frozen_sha256_after_ranker": generator_hash,
                "generator_frozen_sha256_before_ranker": generator_hash,
                "ranker_initial_state_sha256": "c" * 64,
                "shared_group_ledger_sha256": "d" * 64,
                "inpaint_optimizer_steps": 1,
                "direct_optimizer_steps": 1,
                "generator_optimizer_steps": 1,
                "training_precision": {"autocast": "float16", "grad_scaler": True},
                "capacity_selection_binding": {"report_sha256": "e" * 64},
                "per_rank_group_ledger_sha256": ["f" * 64, "0" * 64],
                "per_rank_final_model_state_sha256": [
                    {
                        "generator": generator_hash,
                        "inpaint_ranker": "1" * 64,
                        "direct_ranker": "2" * 64,
                    },
                    {
                        "generator": generator_hash,
                        "inpaint_ranker": "1" * 64,
                        "direct_ranker": "2" * 64,
                    },
                ],
                "synchronized_model_state_sha256": {
                    "generator": generator_hash,
                    "inpaint_ranker": "1" * 64,
                    "direct_ranker": "2" * 64,
                },
                "distributed_execution": {
                    "kind": "torch_DistributedDataParallel",
                    "backend": "nccl",
                    "world_size": 2,
                    "sources_per_rank": 48,
                    "allreduce_every_optimizer_step": True,
                    "generator_batch_per_rank": 128,
                    "ranker_groups_per_rank": 4,
                },
                "optimizer": {"name": "AdamW", "learning_rate": 3e-4, "weight_decay": 1e-4},
                "scheduler": None,
            },
        ),
        checkpoint,
    )
    fake = CompatibilityMatrices("fake", matrix.copy(), matrix.copy())
    monkeypatch.setattr(gate, "dense_scores", lambda *args, **kwargs: (fake, fake))
    real_np_load = np.load
    opened: list[Path] = []

    def spying_load(path, *args, **kwargs):
        opened.append(Path(path).resolve())
        assert label_dir.resolve() not in Path(path).resolve().parents
        return real_np_load(path, *args, **kwargs)

    monkeypatch.setattr(gate.np, "load", spying_load)
    args = type("Args", (), {
        "input_manifest": str(manifest_path), "checkpoint": str(checkpoint),
        "output_dir": str(tmp_path / "phase_a"), "device": "cpu",
        "query_chunk": 4, "score_pair_chunk": 512,
    })()
    gate.phase_a_command(args)
    assert opened and all(label_dir.resolve() not in path.parents for path in opened)


def test_external_capacity_report_is_exactly_bound_and_dynamic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capacity = {"width": 32, "generator_blocks": 3, "ranker_blocks": 3}
    key = "w32_g3_r3"
    capacities = [
        {"width": 64, "generator_blocks": 6, "ranker_blocks": 5},
        {"width": 48, "generator_blocks": 4, "ranker_blocks": 4},
        capacity,
        {"width": 24, "generator_blocks": 2, "ranker_blocks": 2},
        {"width": 16, "generator_blocks": 2, "ranker_blocks": 2},
    ]
    contract = {
        "capacities_largest_first": capacities,
        "workload": {
            "generator_train_true_pairs": 96 * gate.TRUE_GROUPS * 2 * gate.TRAIN_EPOCHS,
            "ranker_train_pair_candidates_per_arm": 96 * gate.TRUE_GROUPS * 2 * 32 * 2 * gate.TRAIN_EPOCHS,
            "all_source_panel_preparations_tilenaf": 808,
            "all_source_panel_preparations_w4": 424,
        },
        "batches": {
            "ddp_generator_per_gpu": 128,
            "ddp_ranker_groups_per_gpu": 4,
            "ddp_ranker_pairs_per_arm_per_gpu": 256,
            "ddp_dense_pairs_per_gpu": 512,
        },
        "optimizers": {
            "generator": "separate AdamW(lr=3e-4, weight_decay=1e-4)",
            "inpaint_ranker": "separate AdamW(lr=3e-4, weight_decay=1e-4)",
            "direct_ranker": "separate AdamW(lr=3e-4, weight_decay=1e-4)",
            "ranker_grad_scaler": "one shared CUDA GradScaler",
            "ranker_microsteps": "view0 no_sync backward plus view1 synchronized backward",
            "ranker_update": "separate unscale, max_norm=1.0 clip, and step per ranker",
        },
        "timing": {
            "amp_dtype": "float16",
            "two_processes_one_per_gpu": True,
            "measured_ddp_all_reduce_during_training": True,
            "ddp_gradient_buckets_in_peak_memory": True,
            "data_parallel_route": "not executed by protocol v2",
        },
    }
    contract_hash = gate.canonical_hash(contract)
    source_hash = "a" * 64
    bundle_hash = "b" * 64
    monkeypatch.setattr(gate, "EXPECTED_BENCHMARK_SOURCE_SHA256", source_hash)
    monkeypatch.setattr(gate, "EXPECTED_BENCHMARK_BUNDLE_SHA256", bundle_hash)
    monkeypatch.setattr(gate, "EXPECTED_BENCHMARK_CONTRACT_SHA256", contract_hash)
    selected = {
        "capacity_key": key,
        "capacity": capacity,
        "projected_seconds_with_1p35_safety": 12_000.0,
        "projected_hours_with_1p35_safety": 12_000.0 / 3600.0,
        "max_peak_reserved_bytes": 9_000_000_000,
        "execution_route": "DDP_T4x2_AMP_v2",
    }
    candidates = []
    for index, value in enumerate(capacities):
        capacity_key = f"w{value['width']}_g{value['generator_blocks']}_r{value['ranker_blocks']}"
        candidates.append({
            "capacity_key": capacity_key,
            "capacity": value,
            "status": "complete",
            "feasible": index >= 2,
            "throughput_aggregation": "2*minimum_per_rank_rate",
            "ddp_all_reduce_cost_measured_in_training_rates": True,
            "ddp_buckets_in_peak_memory": True,
            "isolated_fresh_process_pair": True,
            "allocator_cleared_before_capacity": True,
        })
    report = {
        "kind": gate.EXTERNAL_BENCHMARK_KIND,
        "status": "complete",
        "safe_for_submission": False,
        "launches_scientific_training": False,
        "scientific_images_labels_targets_opened": False,
        "synthetic_only": True,
        "synthetic_optimizer_steps": True,
        "weights_discarded": True,
        "selection_is_engineering_only": True,
        "scientific_hypothesis_or_threshold_changed": False,
        "benchmark_source_sha256": source_hash,
        "contract_sha256": contract_hash,
        "contract": contract,
        "hardware": {
            "devices": [
                {"index": 0, "name": "Tesla T4", "capability": [7, 5]},
                {"index": 1, "name": "Tesla T4", "capability": [7, 5]},
            ]
        },
        "candidates": candidates,
        "selected_capacity": selected,
    }
    report_path = tmp_path / "selection.json"
    gate.atomic_json(report_path, report)
    wrapper = {
        "kind": "masked_gap_t4x2_ddp_benchmark_wrapper_v2",
        "status": "complete",
        "safe_for_submission": False,
        "launches_scientific_training": False,
        "synthetic_optimizer_steps": True,
        "weights_discarded": True,
        "synthetic_only": True,
        "scientific_images_labels_targets_opened": False,
        "selection_report_sha256": gate.sha256(report_path),
        "code_bundle_sha256": bundle_hash,
        "benchmark_source_sha256": source_hash,
        "selected_capacity": selected,
    }
    wrapper_path = tmp_path / "wrapper.json"
    gate.atomic_json(wrapper_path, wrapper)
    monkeypatch.setattr(gate, "EXPECTED_CAPACITY_REPORT_SHA256", gate.sha256(report_path))
    monkeypatch.setattr(gate, "EXPECTED_CAPACITY_WRAPPER_REPORT_SHA256", gate.sha256(wrapper_path))
    monkeypatch.setattr(gate, "EXPECTED_SELECTED_CAPACITY", selected)
    chosen, binding = gate.validate_external_capacity_selection(
        report_path,
        wrapper_path,
        expected_report_sha256=gate.sha256(report_path),
        expected_wrapper_sha256=gate.sha256(wrapper_path),
    )
    assert chosen == capacity
    assert binding["report_sha256"] == gate.sha256(report_path)
    assert binding["ddp_selection_evidence_sha256"] == gate.canonical_hash({
        "candidates": report["candidates"],
        "selected_capacity": report["selected_capacity"],
    })

    report["candidates"][2]["throughput_aggregation"] = "sum_of_unequal_rates"
    gate.atomic_json(report_path, report)
    wrapper["selection_report_sha256"] = gate.sha256(report_path)
    gate.atomic_json(wrapper_path, wrapper)
    monkeypatch.setattr(gate, "EXPECTED_CAPACITY_REPORT_SHA256", gate.sha256(report_path))
    monkeypatch.setattr(gate, "EXPECTED_CAPACITY_WRAPPER_REPORT_SHA256", gate.sha256(wrapper_path))
    with pytest.raises(RuntimeError, match="DDP candidate measurement"):
        gate.validate_external_capacity_selection(
            report_path,
            wrapper_path,
            expected_report_sha256=gate.sha256(report_path),
            expected_wrapper_sha256=gate.sha256(wrapper_path),
        )


def test_exact_training_workload_amp_and_no_subsampling_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = gate.frozen_scientific_config(
        width=40,
        generator_blocks=3,
        ranker_blocks=4,
        capacity_selection_binding={"report_sha256": "a" * 64},
    )
    assert config["training_workload"] == {
        "train_sources": 96,
        "train_sources_per_rank": 48,
        "panels_per_source_per_epoch": 2,
        "epochs": 2,
        "generator_true_pairs_per_source_panel_epoch": 1104,
        "ranker_groups_per_source_panel_epoch": 1104,
        "ranker_candidates_per_group": 32,
        "ranker_views": ["outgoing", "incoming"],
        "subsampling": False,
    }
    assert config["batches"] == {
        "generator_per_rank": 128,
        "ranker_groups_per_rank": 4,
        "dense_pairs_per_rank": 512,
    }
    assert config["precision"] == {
        "training_autocast": "float16",
        "grad_scaler": True,
        "execution": "torch DistributedDataParallel",
        "world_size": 2,
        "backend": "nccl",
        "allreduce_every_optimizer_step": True,
        "separate_ranker_optimizers": True,
        "ranker_microsteps": "view0 no_sync backward plus view1 synchronized backward",
        "ranker_update": "separate unscale, max_norm=1.0 clip, and step per ranker",
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "masked-gap", "train", "--output-dir", "out",
            "--capacity-report", "selection.json",
            "--capacity-wrapper-report", "wrapper.json",
            "--capacity-report-sha256", "a" * 64,
            "--capacity-wrapper-report-sha256", "b" * 64,
        ],
    )
    args = gate.parse_args()
    for forbidden in (
        "max_train_sources", "generator_pairs_per_source", "ranker_groups_per_source",
        "generator_batch_size", "ranker_group_batch_size", "score_pair_chunk",
        "learning_rate",
    ):
        assert not hasattr(args, forbidden)
    source = Path(gate.__file__).read_text(encoding="utf-8")
    assert "torch.autocast(device_type=\"cuda\", dtype=torch.float16)" in source
    assert "generator_scaler.scale(loss).backward()" in source
    assert "ranker_scaler.step(inpaint_optimizer)" in source
    assert "ranker_scaler.step(direct_optimizer)" in source
    assert "DistributedDataParallel(" in source
    assert "dist.broadcast(selection_tensor, src=0)" in source


def test_distributed_shards_cover_records_exactly_and_merge_in_order() -> None:
    assert gate.shard_indices(8, 0, 2) == [0, 2, 4, 6]
    assert gate.shard_indices(8, 1, 2) == [1, 3, 5, 7]
    merged = gate.merge_indexed_shards(
        [
            [{"index": index, "value": f"r0-{index}"} for index in (0, 2, 4, 6)],
            [{"index": index, "value": f"r1-{index}"} for index in (1, 3, 5, 7)],
        ],
        8,
    )
    assert [record["index"] for record in merged] == list(range(8))
    with pytest.raises(RuntimeError, match="coverage"):
        gate.merge_indexed_shards([[{"index": 0}], [{"index": 0}]], 2)


def test_manifest_recomputes_names_hash_before_opening_records(tmp_path: Path) -> None:
    names = gate.frozen_names(
        "calibration_b",
        manifest=REPO_ROOT / "configs/denoise_splits_seed20260710.json",
        quarantine=REPO_ROOT / "configs/denoise_validation_quarantine_v1.json",
    )
    tampered = list(names)
    tampered[0] = "img_999999.png"
    manifest = {
        "kind": gate.INPUT_MANIFEST_KIND,
        "split": "calibration_b",
        "names": tampered,
        "names_sha256": gate.FROZEN_SPLITS["calibration_b"][3],
        "records": [],
    }
    path = tmp_path / "input_manifest.json"
    gate.atomic_json(path, manifest)
    with pytest.raises(RuntimeError, match="frozen names/hash"):
        gate._verify_manifest_records(path, gate.INPUT_MANIFEST_KIND)


def test_phase_b_rejects_invalid_authorization_before_label_path_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_manifest = tmp_path / "input_manifest.json"
    phase_a_report = tmp_path / "phase_a_report.json"
    artifact = tmp_path / "phase_a_scores.npz"
    authorization = tmp_path / "authorization.json"
    checkpoint = tmp_path / "checkpoint.pt"
    gate.atomic_json(input_manifest, {})
    gate.atomic_json(phase_a_report, {})
    artifact.write_bytes(b"scores")
    gate.atomic_json(authorization, {"kind": "invalid", "phase_b_authorized": False})
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        gate,
        "_verify_manifest_records",
        lambda path, kind: (
            {"split": "calibration_b", "names_sha256": "a" * 64, "upstream_asset_sha256": {}},
            [],
        ),
    )
    label_manifest = tmp_path / "must_not_be_resolved" / "label_manifest.json"
    args = type("Args", (), {
        "input_manifest": str(input_manifest),
        "label_manifest": str(label_manifest),
        "phase_a_report": str(phase_a_report),
        "phase_a_artifact": str(artifact),
        "authorization": str(authorization),
        "checkpoint": str(checkpoint),
        "output": str(tmp_path / "phase_b.json"),
        "calibration_b_report": None,
        "device": "cpu",
    })()
    with pytest.raises(RuntimeError, match="missing global Phase B authorization"):
        gate.phase_b_command(args)
    assert not label_manifest.parent.exists()


def _secret_mapping_payload(names: list[str]) -> dict:
    return {
        "kind": gate.SECRET_SEED_MAPPING_KIND,
        "split": "calibration_b",
        "records": [
            {"name": name, "panel": panel, "seed": 2**63 + index}
            for index, (name, panel) in enumerate(
                (identity for name in names for identity in ((name, gate.PANELS[0]), (name, gate.PANELS[1])))
            )
        ],
    }


def test_secret_gate_seed_breaks_public_per_source_permutation_reconstruction(tmp_path: Path) -> None:
    names = gate.frozen_names(
        "calibration_b",
        manifest=REPO_ROOT / "configs/denoise_splits_seed20260710.json",
        quarantine=REPO_ROOT / "configs/denoise_validation_quarantine_v1.json",
    )
    mapping_path = tmp_path / "labels_only" / "secret_panel_seeds.json"
    mapping_path.parent.mkdir()
    gate.atomic_json(mapping_path, _secret_mapping_payload(names))
    mapping = gate.load_secret_panel_seeds(
        mapping_path, split="calibration_b", names=names
    )
    name = names[0]
    panel = "independent_libjpeg"
    secret_seed = mapping[(name, panel)]
    public_seed = per_source_seed(
        gate.MASTER_SEED, f"masked-gap-calibration_b-{panel}", name
    )
    assert secret_seed != public_seed
    clean = np.zeros((480, 480, 3), dtype=np.uint8)
    actual = make_exact_panel(clean, panel="clean_shuffle", seed=secret_seed)
    reconstructed = make_exact_panel(clean, panel="clean_shuffle", seed=public_seed)
    assert not np.array_equal(actual.slot_to_target, reconstructed.slot_to_target)


@pytest.mark.parametrize("failure", ["missing", "extra", "reused", "negative", "too_large"])
def test_secret_seed_mapping_rejects_incomplete_reused_or_non_uint64_values(
    tmp_path: Path, failure: str
) -> None:
    names = gate.frozen_names(
        "calibration_b",
        manifest=REPO_ROOT / "configs/denoise_splits_seed20260710.json",
        quarantine=REPO_ROOT / "configs/denoise_validation_quarantine_v1.json",
    )
    payload = _secret_mapping_payload(names)
    if failure == "missing":
        payload["records"].pop()
    elif failure == "extra":
        payload["records"].append({"name": "img_999999.png", "panel": gate.PANELS[0], "seed": 7})
    elif failure == "reused":
        payload["records"][1]["seed"] = payload["records"][0]["seed"]
    elif failure == "negative":
        payload["records"][0]["seed"] = -1
    elif failure == "too_large":
        payload["records"][0]["seed"] = 2**64
    path = tmp_path / f"{failure}.json"
    gate.atomic_json(path, payload)
    with pytest.raises(RuntimeError, match="secret panel seed"):
        gate.load_secret_panel_seeds(path, split="calibration_b", names=names)


def test_prepare_source_requires_secret_seeds_only_for_sealed_gate_splits() -> None:
    common = {
        "data_root": "unused",
        "restorer": nn.Identity(),
        "embedding": nn.Identity(),
        "device": torch.device("cpu"),
        "denoise_batch_size": 1,
        "seed": gate.MASTER_SEED,
        "require_w4": False,
    }
    with pytest.raises(RuntimeError, match="requires an explicit secret panel seed"):
        gate.prepare_source(
            "img_000001.png", gate.PANELS[0], "calibration_b", **common
        )
    with pytest.raises(RuntimeError, match="reserved for sealed gate fixtures"):
        gate.prepare_source(
            "img_000001.png", gate.PANELS[0], "train", panel_seed=1, **common
        )


def test_torch_generator_accepts_full_uint64_secret_seed_domain() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2**64 - 1)
    assert generator.initial_seed() == 2**64 - 1


def test_input_and_phase_a_surfaces_have_no_secret_seed_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(gate.__file__).read_text(encoding="utf-8")
    input_block = source.split("input_manifest = {", 1)[1].split("label_manifest = {", 1)[0]
    assert '"panel_seed_attached": False' in input_block
    assert '"panel_seed_derivation_available": False' in input_block
    assert '"seed":' not in input_block
    assert "secret_seed_mapping" not in input_block
    checkpoint_block = source.split("payload = state_dict_payload(", 1)[1].split("def prepare_command", 1)[0]
    assert "secret_panel_seed" not in checkpoint_block
    monkeypatch.setattr(
        sys,
        "argv",
        ["masked-gap", "phase-a", "--input-manifest", "input.json", "--checkpoint", "model.pt", "--output-dir", "out"],
    )
    args = gate.parse_args()
    assert not hasattr(args, "secret_seed_mapping")
