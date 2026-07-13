from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest
import torch

from scripts import benchmark_masked_gap_t4 as benchmark
from puzzle_assembly.masked_gap import (
    MaskedGapGenerator,
    PairListwiseRanker,
    listwise_pair_loss,
)


ROOT = Path(__file__).resolve().parents[1]


def _measurement(
    capacity: benchmark.Capacity,
    *,
    rate: float = 1.0e12,
    peak: int = 1_000_000,
) -> dict:
    return {
        "capacity_key": capacity.key,
        "capacity": {
            "width": capacity.width,
            "generator_blocks": capacity.generator_blocks,
            "ranker_blocks": capacity.ranker_blocks,
        },
        "status": "complete",
        "throughput_2gpu": {
            "generator_train_pairs_per_second": rate,
            "joint_ranker_train_pairs_per_arm_per_second": rate,
            "dense_pipeline_pairs_per_second": rate,
        },
        "peak_reserved_bytes_per_gpu": [peak, peak],
    }


def test_frozen_workload_arithmetic_is_exact() -> None:
    workload = benchmark.frozen_workload()
    assert benchmark.ADJACENCY_GROUPS == 2 * 24 * 23 == 1104
    assert workload == {
        "generator_train_true_pairs": 423_936,
        "ranker_train_pair_candidates_per_arm": 27_131_904,
        "ranker_train_pair_candidates_two_arms": 54_263_808,
        "development_dense_pairs_per_model_per_pass": 5_299_200,
        "checkpoint_selection_dense_pairs_per_model_two_epochs": 10_598_400,
        "calibration_b_dense_pairs_per_model": 5_299_200,
        "final_dense_pairs_per_model": 10_598_400,
        "all_dense_pairs_per_model": 26_496_000,
        "all_dense_component_forwards_generator_plus_two_rankers": 79_488_000,
        "generator_source_panel_preparations_tilenaf": 384,
        "ranker_source_panel_preparations_tilenaf_plus_w4": 384,
        "checkpoint_selection_source_panel_preparations_tilenaf_plus_w4": 16,
        "calibration_b_source_panel_preparations_tilenaf_plus_w4": 8,
        "final_source_panel_preparations_tilenaf_plus_w4": 16,
        "all_source_panel_preparations_tilenaf": 808,
        "all_source_panel_preparations_w4": 424,
    }
    contract = benchmark.frozen_contract()
    assert contract["workload"] == workload
    assert contract["timing"]["two_processes_one_per_gpu"] is True
    assert contract["timing"]["fresh_process_pair_per_capacity"] is True
    assert contract["timing"]["measured_ddp_all_reduce_during_training"] is True
    assert contract["timing"]["ddp_gradient_buckets_in_peak_memory"] is True
    assert contract["timing"]["cuda_synchronize"] is True
    assert contract["timing"]["amp_dtype"] == "float16"
    assert contract["timing"]["final_decision"] == (
        "largest DDP-feasible capacity in precommitted order"
    )
    assert contract["timing"]["data_parallel_route"] == (
        "not executed by protocol v2"
    )
    assert contract["batches"] == {
        "ddp_generator_per_gpu": 128,
        "ddp_ranker_groups_per_gpu": 4,
        "ddp_ranker_pairs_per_arm_per_gpu": 256,
        "ddp_dense_pairs_per_gpu": 512,
    }
    assert contract["optimizers"] == {
        "generator": "separate AdamW(lr=3e-4, weight_decay=1e-4)",
        "inpaint_ranker": "separate AdamW(lr=3e-4, weight_decay=1e-4)",
        "direct_ranker": "separate AdamW(lr=3e-4, weight_decay=1e-4)",
        "ranker_grad_scaler": "one shared CUDA GradScaler",
        "ranker_microsteps": "view0 no_sync backward plus view1 synchronized backward",
        "ranker_update": "separate unscale, max_norm=1.0 clip, and step per ranker",
    }
    assert contract["selection"]["fixed_source_preparation_reserve_seconds_before_safety"] == 3600


def test_fixed_capacity_order_and_largest_feasible_selection() -> None:
    assert [
        (value.width, value.generator_blocks, value.ranker_blocks)
        for value in benchmark.CAPACITIES
    ] == [(64, 6, 5), (48, 4, 4), (32, 3, 3), (24, 2, 2), (16, 2, 2)]
    measurements = [
        _measurement(benchmark.CAPACITIES[0], rate=1.0),
        _measurement(
            benchmark.CAPACITIES[1],
            peak=benchmark.MAX_PEAK_BYTES_PER_GPU + 1,
        ),
        _measurement(
            benchmark.CAPACITIES[2],
            peak=benchmark.MAX_PEAK_BYTES_PER_GPU,
        ),
        _measurement(benchmark.CAPACITIES[3]),
        _measurement(benchmark.CAPACITIES[4]),
    ]
    selected, projected = benchmark.select_largest_feasible(measurements)
    assert selected is not None
    assert selected["capacity_key"] == "w32_g3_r3"
    assert [value["feasible"] for value in projected] == [False, False, True, True, True]

    with pytest.raises(ValueError, match="capacity order/config drift"):
        benchmark.select_largest_feasible(list(reversed(measurements)))

    none_selected, all_rejected = benchmark.select_largest_feasible(
        [_measurement(capacity, rate=1.0) for capacity in benchmark.CAPACITIES]
    )
    assert none_selected is None
    assert all(not value["feasible"] for value in all_rejected)


def test_selection_thresholds_are_inclusive_and_fail_one_unit_above() -> None:
    assert benchmark.inclusive_feasibility(
        benchmark.MAX_PROJECTED_SECONDS,
        benchmark.MAX_PEAK_BYTES_PER_GPU,
    ) == (True, True, True)
    assert benchmark.inclusive_feasibility(
        benchmark.MAX_PROJECTED_SECONDS + 1.0e-9,
        benchmark.MAX_PEAK_BYTES_PER_GPU,
    ) == (False, True, False)
    assert benchmark.inclusive_feasibility(
        benchmark.MAX_PROJECTED_SECONDS,
        benchmark.MAX_PEAK_BYTES_PER_GPU + 1,
    ) == (True, False, False)
    with pytest.raises(ValueError, match="finite/non-negative"):
        benchmark.inclusive_feasibility(float("nan"), 0)


def test_projection_uses_per_arm_joint_ranker_and_full_dense_pipeline() -> None:
    measurement = _measurement(benchmark.CAPACITIES[-1], rate=1000.0, peak=1234)
    projected = benchmark.project_candidate(measurement)
    workload = benchmark.frozen_workload()
    components = projected["projection_components_seconds_before_safety"]
    assert components["generator_train_seconds"] == pytest.approx(
        workload["generator_train_true_pairs"] / 1000.0
    )
    assert components["two_rankers_train_seconds"] == pytest.approx(
        workload["ranker_train_pair_candidates_per_arm"] / 1000.0
    )
    assert components["all_dense_generator_plus_two_rankers_seconds"] == pytest.approx(
        workload["all_dense_pairs_per_model"] / 1000.0
    )
    assert components["fixed_source_preparation_reserve_seconds"] == 3600.0
    assert projected["projected_seconds_with_1p35_safety"] == pytest.approx(
        sum(components.values()) * 1.35
    )
    assert projected["max_peak_reserved_bytes"] == 1234


def _rank_report(
    capacity: benchmark.Capacity,
    rank: int,
    *,
    generator_rate: float,
    ranker_rate: float,
    dense_rate: float,
    peak: int,
) -> dict:
    return {
        "status": "complete",
        "rank": rank,
        "capacity_key": capacity.key,
        "device": {
            "index": rank,
            "name": "Tesla T4",
            "capability": [7, 5],
            "actual_tensor_op": 1.0,
        },
        "measurement": {
            "capacity_key": capacity.key,
            "parameter_counts": {"generator": 1, "ranker_per_arm": 2, "pipeline": 5},
            "throughput": {
                "generator_train_pairs_per_second": generator_rate,
                "joint_ranker_train_pairs_per_arm_per_second": ranker_rate,
                "dense_pipeline_pairs_per_second": dense_rate,
            },
            "peak_reserved_bytes": peak,
            "peak_allocated_bytes": peak - 1,
            "ddp_buckets_in_peak_memory": True,
            "allocator_cleared_before_capacity": True,
        },
    }


def test_rank_imbalance_uses_twice_slower_rank_not_optimistic_sum() -> None:
    capacity = benchmark.CAPACITIES[0]
    result = benchmark.aggregate_capacity(
        capacity,
        [
            _rank_report(
                capacity, 0, generator_rate=100.0, ranker_rate=80.0,
                dense_rate=120.0, peak=1000,
            ),
            _rank_report(
                capacity, 1, generator_rate=60.0, ranker_rate=70.0,
                dense_rate=90.0, peak=2000,
            ),
        ],
    )
    assert result["throughput_2gpu"] == {
        "generator_train_pairs_per_second": 120.0,
        "joint_ranker_train_pairs_per_arm_per_second": 140.0,
        "dense_pipeline_pairs_per_second": 180.0,
    }
    assert result["throughput_aggregation"] == "2*minimum_per_rank_rate"
    assert result["rank_imbalance_ratio"]["generator_train_pairs_per_second"] == pytest.approx(100 / 60)
    assert result["ddp_all_reduce_cost_measured_in_training_rates"] is True


def test_isolated_oom_capacity_is_rejected_and_smaller_capacity_can_win() -> None:
    largest = benchmark.CAPACITIES[0]
    oom = benchmark.aggregate_capacity(
        largest,
        [
            {"status": "oom", "rank": 0, "capacity_key": largest.key, "device": None},
            {"status": "oom", "rank": 1, "capacity_key": largest.key, "device": None},
        ],
    )
    measurements = [oom] + [
        _measurement(capacity) for capacity in benchmark.CAPACITIES[1:]
    ]
    selected, projected = benchmark.select_largest_feasible(measurements)
    assert projected[0]["feasible"] is False
    assert projected[0]["rejection_reason"] == "isolated_capacity_out_of_memory"
    assert selected is not None
    assert selected["capacity_key"] == benchmark.CAPACITIES[1].key


def test_capacity_reports_prove_clean_isolated_peaks_and_ddp_buckets() -> None:
    capacity = benchmark.CAPACITIES[-1]
    result = benchmark.aggregate_capacity(
        capacity,
        [
            _rank_report(
                capacity, 0, generator_rate=1, ranker_rate=1, dense_rate=1, peak=10,
            ),
            _rank_report(
                capacity, 1, generator_rate=1, ranker_rate=1, dense_rate=1, peak=20,
            ),
        ],
    )
    assert result["isolated_fresh_process_pair"] is True
    assert result["allocator_cleared_before_capacity"] is True
    assert result["ddp_buckets_in_peak_memory"] is True
    assert result["peak_reserved_bytes_per_gpu"] == [10, 20]


def test_synthetic_models_have_exact_20x40_shapes_and_finite_outputs() -> None:
    capacity = benchmark.CAPACITIES[-1]
    generator = benchmark.BenchmarkGenerator(
        capacity.width, capacity.generator_blocks
    )
    ranker = benchmark.BenchmarkRanker(capacity.width, capacity.ranker_blocks)
    gap = generator(torch.full((2, 7, 20, 40), 0.25))
    scores = ranker(torch.full((2, 10, 20, 40), 0.25))
    assert gap.shape == (2, 3, 20, 4)
    assert scores.shape == (2,)
    assert torch.isfinite(gap).all()
    assert torch.isfinite(scores).all()


def test_benchmark_architectures_match_scientific_core_for_every_capacity() -> None:
    for capacity in benchmark.CAPACITIES:
        benchmark_generator = benchmark.BenchmarkGenerator(
            capacity.width, capacity.generator_blocks
        )
        core_generator = MaskedGapGenerator(
            width=capacity.width, blocks=capacity.generator_blocks
        )
        assert {
            key: tuple(value.shape)
            for key, value in benchmark_generator.state_dict().items()
        } == {key: tuple(value.shape) for key, value in core_generator.state_dict().items()}

        benchmark_ranker = benchmark.BenchmarkRanker(
            capacity.width, capacity.ranker_blocks
        )
        core_ranker = PairListwiseRanker(
            width=capacity.width, blocks=capacity.ranker_blocks
        )
        assert {
            key: tuple(value.shape)
            for key, value in benchmark_ranker.state_dict().items()
        } == {key: tuple(value.shape) for key, value in core_ranker.state_dict().items()}


def test_two_separate_ranker_views_sum_to_exact_scientific_loss() -> None:
    outgoing = torch.linspace(-1.0, 1.0, 4 * 32).reshape(4, 32)
    incoming = torch.linspace(1.0, -1.0, 4 * 32).reshape(4, 32)
    expected, _ = listwise_pair_loss(outgoing, incoming)
    actual = benchmark._ranker_view_loss(outgoing.reshape(-1))
    actual = actual + benchmark._ranker_view_loss(incoming.reshape(-1))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def test_benchmark_source_has_no_input_tree_or_scientific_artifact_access() -> None:
    source = (ROOT / "scripts" / "benchmark_masked_gap_t4.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "/kaggle/input",
        "puzzle/train",
        "puzzle/test",
        ".rglob(",
        ".glob(",
        "np.load(",
        "torch.load(",
    ):
        assert forbidden not in source
    assert '"safe_for_submission": False' in source
    assert '"launches_scientific_training": False' in source
    assert '"synthetic_optimizer_steps": True' in source
    assert '"weights_discarded": True' in source
    assert '"scientific_images_labels_targets_opened": False' in source
    assert benchmark.REPORT_KIND == "masked_gap_t4x2_amp_ddp_capacity_selection_v2"
    assert "nn.DataParallel(" not in source
    assert "_data_parallel_confirmation" not in source
    assert '"data_parallel_confirmation"' not in source
    assert "DistributedDataParallel as DDP" in source
    assert "generator_optimizer = torch.optim.AdamW" in source
    assert "inpaint_optimizer = torch.optim.AdamW" in source
    assert "direct_optimizer = torch.optim.AdamW" in source
    assert "ranker_optimizer = torch.optim.AdamW" not in source


def test_kaggle_job_is_private_t4_target_free_and_hash_pinned() -> None:
    job = ROOT / "runs" / "assembly_v1" / "kaggle" / "masked_gap_benchmark_job"
    metadata = json.loads((job / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata == {
        "id": "pasha883/vsos-masked-gap-target-free-ddp-t4x2-benchmark-v2",
        "title": "VSOS Masked Gap Target-Free DDP T4x2 Benchmark V2",
        "code_file": "run_masked_gap_benchmark.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": False,
        "dataset_sources": ["pasha883/vsos-masked-gap-ddp-benchmark-code-v2"],
    }
    source_path = ROOT / "scripts" / "benchmark_masked_gap_t4.py"
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    contract_hash = benchmark.canonical_json_sha256(benchmark.frozen_contract())
    wrapper = (job / "run_masked_gap_benchmark.py").read_text(encoding="utf-8")
    assert source_hash in wrapper
    assert contract_hash in wrapper
    assert 'INPUT_ROOT.rglob("masked_gap_benchmark_code.bin")' in wrapper
    assert '"safe_for_submission": False' in wrapper
    assert '"launches_scientific_training": False' in wrapper
    dataset_metadata = json.loads(
        (
            ROOT
            / "runs"
            / "assembly_v1"
            / "kaggle"
            / "masked_gap_ddp_benchmark_code_dataset_v2"
            / "dataset-metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert dataset_metadata["id"] == metadata["dataset_sources"][0]
    assert dataset_metadata["isPrivate"] is True

    upload = (
        ROOT
        / "runs"
        / "assembly_v1"
        / "kaggle"
        / "masked_gap_ddp_benchmark_code_dataset_v2"
        / "upload_v1"
    )
    assert json.loads((upload / "dataset-metadata.json").read_text(encoding="utf-8")) == dataset_metadata
    archive = upload / "masked_gap_benchmark_code.bin"
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive_hash in wrapper
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["benchmark_masked_gap_t4.py"]
        assert hashlib.sha256(bundle.read("benchmark_masked_gap_t4.py")).hexdigest() == source_hash


def test_wrapper_roundtrip_extracts_only_hash_pinned_staged_script(tmp_path: Path) -> None:
    wrapper_path = (
        ROOT
        / "runs"
        / "assembly_v1"
        / "kaggle"
        / "masked_gap_benchmark_job"
        / "run_masked_gap_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("masked_gap_benchmark_wrapper", wrapper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.INPUT_ROOT = (
        ROOT
        / "runs"
        / "assembly_v1"
        / "kaggle"
        / "masked_gap_ddp_benchmark_code_dataset_v2"
        / "upload_v1"
    )
    module.WORKING = tmp_path
    extracted, archive = module.extract_benchmark()
    assert hashlib.sha256(extracted.read_bytes()).hexdigest() == module.EXPECTED_BENCHMARK_SHA256
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == module.EXPECTED_BUNDLE_SHA256


def test_wrapper_accepts_ddp_v2_schema_and_rejects_legacy_confirmation() -> None:
    wrapper_path = (
        ROOT
        / "runs"
        / "assembly_v1"
        / "kaggle"
        / "masked_gap_benchmark_job"
        / "run_masked_gap_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "masked_gap_benchmark_wrapper_schema", wrapper_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    capacities = [
        {
            "width": value.width,
            "generator_blocks": value.generator_blocks,
            "ranker_blocks": value.ranker_blocks,
        }
        for value in benchmark.CAPACITIES
    ]
    candidates = [
        {
            "capacity_key": value.key,
            "capacity": capacity,
            "status": "complete",
            "feasible": True,
            "projected_seconds_with_1p35_safety": 3600.0 * (1.0 + index),
            "projected_hours_with_1p35_safety": 1.0 + index,
            "max_peak_reserved_bytes": 1_000_000 + index,
            "throughput_aggregation": "2*minimum_per_rank_rate",
            "ddp_all_reduce_cost_measured_in_training_rates": True,
            "ddp_buckets_in_peak_memory": True,
            "isolated_fresh_process_pair": True,
            "allocator_cleared_before_capacity": True,
        }
        for index, (value, capacity) in enumerate(zip(benchmark.CAPACITIES, capacities))
    ]
    contract = benchmark.frozen_contract()
    report = {
        "kind": benchmark.REPORT_KIND,
        "status": "complete",
        "safe_for_submission": False,
        "launches_scientific_training": False,
        "scientific_images_labels_targets_opened": False,
        "synthetic_only": True,
        "synthetic_optimizer_steps": True,
        "weights_discarded": True,
        "benchmark_source_sha256": module.EXPECTED_BENCHMARK_SHA256,
        "contract_sha256": module.EXPECTED_CONTRACT_SHA256,
        "contract": contract,
        "candidates": candidates,
        "selected_capacity": {
            "capacity_key": benchmark.CAPACITIES[0].key,
            "capacity": capacities[0],
            "projected_seconds_with_1p35_safety": 3600.0,
            "projected_hours_with_1p35_safety": 1.0,
            "max_peak_reserved_bytes": 1_000_000,
            "execution_route": "DDP_T4x2_AMP_v2",
        },
        "hardware": {
            "devices": [
                {
                    "index": index,
                    "name": "Tesla T4",
                    "capability": [7, 5],
                    "actual_tensor_op": 1.0,
                }
                for index in range(2)
            ]
        },
    }
    module.validate_selection(report)
    with pytest.raises(RuntimeError, match="legacy DataParallel confirmation"):
        module.validate_selection({**report, "data_parallel_confirmation": {}})
    without_contract_hash = dict(report)
    without_contract_hash.pop("contract_sha256")
    with pytest.raises(RuntimeError, match="contract hash mismatch"):
        module.validate_selection(without_contract_hash)
    with pytest.raises(RuntimeError, match="contract hash mismatch"):
        module.validate_selection({**report, "contract_sha256": "0" * 64})
    mutated_contract = {
        **contract,
        "workload": {
            **contract["workload"],
            "all_dense_pairs_per_model": 26_496_001,
        },
    }
    with pytest.raises(RuntimeError, match="contract payload mismatch"):
        module.validate_selection({**report, "contract": mutated_contract})


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() != 2,
    reason="T4x2 CUDA integration only runs on the remote benchmark worker",
)
def test_t4x2_hardware_contract_and_real_tensor_op_integration() -> None:
    for index in range(2):
        assert "T4" in torch.cuda.get_device_name(index).upper()
        assert torch.cuda.get_device_capability(index) == (7, 5)
        values = torch.randn(32, 32, device=f"cuda:{index}")
        assert torch.isfinite(values @ values).all()
