from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.denoiser_adaptation import (
    MultiViewSideEmbeddingNet,
    load_multiview_checkpoint,
    names_sha256,
    retrieval_diagnostics,
    save_multiview_checkpoint,
    sha256_file,
    successor_truth,
    validate_protocol_safety,
)
from puzzle_assembly.learned import SideEmbeddingNet
from puzzle_assembly.protocol import source_names_for_split
from prepare_solver_denoiser_adaptation import prepare


CONFIG = ROOT / "configs" / "solver_denoiser_adaptation_v1.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_protocol_is_safe_and_bound_to_upstream_stop() -> None:
    config = load_config()
    validate_protocol_safety(config)
    upstream = config["upstream_new_denoiser_result"]
    assert upstream["decision"] == "stop_no_development_signal"
    assert upstream["selected_checkpoint"] is None
    assert upstream["permits_adaptation"] is False
    assert config["launch_interlock"]["permanently_closed_for_this_5x5_run"] is True
    artifact = ROOT / upstream["artifact"]
    assert sha256_file(artifact) == upstream["artifact_sha256"]


def test_protocol_rejects_oracle_label_reference() -> None:
    config = deepcopy(load_config())
    config["unsafe"] = "opaque/fixture_label/secret"
    with pytest.raises(ValueError, match="must not reference oracle labels"):
        validate_protocol_safety(config)


def test_frozen_whole_source_slices_and_hashes() -> None:
    config = load_config()
    assets = config["authoritative_inputs"]
    manifest = ROOT / assets["manifest"]["path"]
    quarantine = ROOT / assets["quarantine"]["path"]
    for key in ("manifest", "quarantine", "old_denoiser", "production_hbt"):
        record = assets[key]
        assert sha256_file(ROOT / record["path"]) == record["sha256"]
    resolved = {}
    for key in ("scorer_training", "exact_selection"):
        spec = config["source_partitions"][key]
        names = source_names_for_split(
            spec["split"], manifest_path=manifest, quarantine_path=quarantine
        )[spec["offset"] : spec["offset"] + spec["count"]]
        assert names_sha256(names) == spec["names_sha256"]
        if key == "exact_selection":
            assert names == spec["names"]
        resolved[key] = set(names)
    assert not (resolved["scorer_training"] & resolved["exact_selection"])


def perfect_compatibility() -> tuple[CompatibilityMatrices, np.ndarray]:
    slot_to_target = np.arange(576, dtype=np.int32)
    truth = successor_truth(slot_to_target)
    right = np.full((576, 576), 10.0, dtype=np.float32)
    down = np.full((576, 576), 10.0, dtype=np.float32)
    np.fill_diagonal(right, np.inf)
    np.fill_diagonal(down, np.inf)
    for matrix, targets in ((right, truth.right), (down, truth.down)):
        for query, target in enumerate(targets.tolist()):
            if target >= 0:
                matrix[query, target] = 0.0
    return CompatibilityMatrices("perfect", right, down), slot_to_target


def test_retrieval_diagnostics_perfect_truth() -> None:
    scores, truth = perfect_compatibility()
    metrics = retrieval_diagnostics(scores, truth)
    assert metrics["top1"] == 1.0
    assert metrics["top5"] == 1.0
    assert metrics["top10"] == 1.0
    assert metrics["mutual_correct_coverage"] == 1.0
    assert metrics["mutual_precision"] > 0.95
    assert metrics["candidate_recall_at_32"] == 1.0
    assert metrics["candidate_lcc_at_32"] == 576.0
    assert metrics["true_directed_edges"] == 1104.0


def test_retrieval_diagnostics_detects_bad_ranking() -> None:
    scores, truth = perfect_compatibility()
    right = scores.right.copy()
    down = scores.down.copy()
    true = successor_truth(truth)
    for matrix, targets in ((right, true.right), (down, true.down)):
        for query, target in enumerate(targets.tolist()):
            if target >= 0:
                matrix[query, target] = 20.0
    bad = retrieval_diagnostics(CompatibilityMatrices("bad", right, down), truth)
    assert bad["top1"] == 0.0
    assert bad["top10"] < 0.05


def test_multiview_model_zero_mask_reproduces_anchor() -> None:
    torch.manual_seed(7)
    encoder = SideEmbeddingNet(
        channels=8,
        embedding_dim=16,
        side_band=2,
        tangent_bins=4,
        input_mode="rgb_sobel",
    )
    model = MultiViewSideEmbeddingNet.from_production_encoder(encoder)
    views = {
        "dirty": torch.rand(4, 3, 20, 20),
        "old_denoised": torch.rand(4, 3, 20, 20),
        "new_denoised": torch.rand(4, 3, 20, 20),
    }
    encoder.eval()
    model.eval()
    with torch.inference_mode():
        anchor = encoder(views["old_denoised"])
        output = model(views, view_mask=torch.zeros(2))
    for key in ("q_right", "k_left", "q_down", "k_up", "outside_logits"):
        torch.testing.assert_close(output[key], anchor[key], atol=1e-6, rtol=1e-6)


def test_multiview_checkpoint_roundtrip(tmp_path: Path) -> None:
    encoder = SideEmbeddingNet(
        channels=8,
        embedding_dim=16,
        side_band=2,
        tangent_bins=4,
        input_mode="rgb_norm",
    )
    model = MultiViewSideEmbeddingNet.from_production_encoder(encoder)
    checkpoint = tmp_path / "model.pt"
    save_multiview_checkpoint(checkpoint, model, metadata={"seed": 17})
    loaded, metadata = load_multiview_checkpoint(checkpoint)
    assert metadata == {"seed": 17}
    assert loaded.config() == model.config()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_runtime_pin_refuses_rejected_upstream(tmp_path: Path) -> None:
    args = argparse.Namespace(
        template=str(CONFIG),
        new_denoiser=str(tmp_path / "unused.pt"),
        expected_new_denoiser_sha256="0" * 64,
        oracle_verdict="inconclusive",
        root_launch_signal="ROOT_AUTHORIZED",
        output=str(tmp_path / "runtime.json"),
        receipt=str(tmp_path / "receipt.json"),
    )
    with pytest.raises(ValueError, match="selected no checkpoint"):
        prepare(args)


def test_no_launch_receipt_and_kaggle_entrypoint_are_bound(tmp_path: Path) -> None:
    receipt = json.loads(
        (ROOT / "runs/assembly_v1/solver_denoiser_adaptation/NO_LAUNCH.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["protocol_sha256"] == sha256_file(CONFIG)
    assert receipt["upstream_selection_sha256"] == load_config()[
        "upstream_new_denoiser_result"
    ]["artifact_sha256"]
    assert receipt["gpu_training_launched"] is False
    runner_path = (
        ROOT
        / "runs/assembly_v1/kaggle/solver_denoiser_adaptation_job"
        / "run_solver_denoiser_adaptation.py"
    )
    spec = importlib.util.spec_from_file_location("solver_adaptation_staged_runner", runner_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.WORKING = tmp_path
    module.main()
    payload = json.loads(
        (tmp_path / "solver_denoiser_adaptation_NO_LAUNCH.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["protocol_sha256"] == sha256_file(CONFIG)
    assert payload["training_launched"] is False
    assert payload["gpu_probed"] is False
