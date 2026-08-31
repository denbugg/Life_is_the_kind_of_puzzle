from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.union_hard_edge_cutoff_loss import (
    union_hard_edge_cutoff_exchange_loss,
)
from aiijc_puzzle.union_hard_edge_priority import UnionHardEdgePriority

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = PROJECT_ROOT / "outputs/union-hard-edge-priority/pilot-v1-final"
SCRIPTS = PROJECT_ROOT / "scripts"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_union_hard_edge_cutoff_continuation_opened32.py"
    specification = importlib.util.spec_from_file_location(
        "run_union_hard_edge_cutoff_continuation_opened32_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _complete_recipe() -> dict[str, object]:
    payload = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    commitment = json.loads((PARENT_ROOT / "selection-commitment.json").read_text(encoding="utf-8"))
    payload["protocol"]["target_free_feature_cache_reused_without_feature_pass"] = True
    payload["panel"] = {
        "source_filenames": commitment["eval"]["source_filenames"],
        "source_order_digest": runner.EVAL_SOURCE_ORDER_DIGEST,
        "draws": list(runner.DRAWS),
        "case_count": runner.EVAL_CASE_COUNT,
        "cases_digest": runner.EVAL_CASES_DIGEST,
        "previously_opened": True,
    }
    return payload


def test_fixed_recipe_and_signed_sidecar_are_required(tmp_path: Path) -> None:
    payload = _complete_recipe()
    assert runner._validate_recipe(payload) == tuple(payload["panel"]["source_filenames"])

    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar"):
        runner._load_preregistration(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    observed, observed_digest, names = runner._load_preregistration(path)
    assert observed == payload
    assert observed_digest == digest
    assert names == tuple(payload["panel"]["source_filenames"])

    payload["continuation"]["additional_steps"] = 201
    path.write_text(json.dumps(payload), encoding="utf-8")
    changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256").write_text(f"{changed_digest}  {path.name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="additional_steps"):
        runner._load_preregistration(path)


def test_case_order_is_the_single_fixed_reproducible_200_step_stream() -> None:
    seed = 1_267_233_517 + runner.CASE_ORDER_SEED_OFFSET
    first = runner._case_order(seed)
    second = runner._case_order(seed)
    assert first == second
    assert len(first) == runner.ADDITIONAL_STEPS
    assert first[:5] == [56, 19, 76, 117, 8]
    assert sorted(first[: runner.FIT_CASE_COUNT]) == list(range(runner.FIT_CASE_COUNT))
    assert runner._integer_sequence_digest(first) == (
        "aa67690afae7cdf2ece540f037c885ac7df049372201adcadf81795361f9c771"
    )


def test_parent_checkpoint_strict_load_and_cached_cutoff_loss_are_compatible() -> None:
    checkpoint = runner._validate_parent_checkpoint(PARENT_ROOT / "union-hard-edge-priority.pt")
    model = UnionHardEdgePriority(
        hidden_dimension=runner.HIDDEN_DIMENSION,
        residual_limit=runner.RESIDUAL_LIMIT,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    board = runner._cached_board(
        PARENT_ROOT / "target-free-cache",
        "fit",
        0,
        device=torch.device("cpu"),
    )
    labels = np.load(PARENT_ROOT / "fit-labels.npy", mmap_mode="r")
    truth = torch.from_numpy(np.asarray(labels[0], dtype=bool).copy())
    output = model(board)
    loss, diagnostics = union_hard_edge_cutoff_exchange_loss(output, board, truth)
    assert torch.isfinite(loss)
    assert diagnostics["edge_budget_per_axis"] == runner.EDGE_BUDGET_PER_AXIS
    assert diagnostics["selected_edges"] == 2 * runner.EDGE_BUDGET_PER_AXIS


def test_pair_metric_uses_integer_adjacency_count_with_recall_cross_check() -> None:
    reference = np.arange(runner.COUNT, dtype=np.int32)
    evaluation = evaluate_layout(reference, reference, reference_is_exact=True)
    metrics = runner._layout_metrics(evaluation, fixed_top288_correct=288)
    assert metrics == {
        "satisfied_adjacent_pairs": 1104,
        "adjacency_recall": 1.0,
        "fixed_top288_correct": 288,
        "exact_tiles": 576,
    }


def test_scoring_cannot_construct_target_cache_before_freeze_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target_cache_constructed = False

    def construct_target_cache(_path: Path) -> None:
        nonlocal target_cache_constructed
        target_cache_constructed = True
        raise AssertionError("target cache must not be constructed")

    def reject_freeze(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("frozen roster changed")

    monkeypatch.setattr(runner, "CleanTileCache", construct_target_cache)
    monkeypatch.setattr(runner, "_validate_pre_score_freeze", reject_freeze)
    artifacts = SimpleNamespace(frozen_target_free_eval_metadata=tmp_path / "absent")
    paths = SimpleNamespace(pre_score_freeze=tmp_path / "absent")
    with pytest.raises(RuntimeError, match="frozen roster"):
        runner._score_frozen_eval(
            artifacts,
            paths,
            {},
            {},
            targets=tmp_path / "targets",
        )
    assert target_cache_constructed is False


def test_pre_score_roster_distinguishes_fit_labels_from_evaluation_labels(
    tmp_path: Path,
) -> None:
    def artifact(name: str) -> Path:
        path = tmp_path / name
        path.write_bytes(name.encode())
        return path

    artifacts = runner.ParentArtifacts(
        config=artifact("parent-config.json"),
        selection_commitment=artifact("parent-selection.json"),
        checkpoint=artifact("parent-checkpoint.pt"),
        fit_labels=artifact("fit-labels.npy"),
        fit_labels_metadata=artifact("fit-labels.json"),
        target_free_cache_metadata=artifact("cache-metadata.json"),
        frozen_target_free_eval=artifact("parent-eval.npz"),
        frozen_target_free_eval_metadata=artifact("parent-eval.json"),
        report=artifact("parent-report.json"),
    )
    paths = runner.RunPaths(
        checkpoint=artifact("candidate-checkpoint.pt"),
        frozen_eval=artifact("candidate-eval.npz"),
        frozen_eval_metadata=artifact("candidate-eval.json"),
        pre_score_freeze=tmp_path / "pre-score-freeze.json",
        report=tmp_path / "report.json",
    )
    config = artifact("config.json")
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    frozen = runner._freeze_pre_score_roster(
        paths,
        artifacts,
        config_path=config,
        config_sha256=config_sha,
    )
    payload = json.loads(paths.pre_score_freeze.read_text(encoding="utf-8"))
    assert payload["contains_evaluation_references_or_labels"] is False
    assert "contains_exact_references_or_labels" not in payload
    assert "fit_labels" in payload["artifacts"]
    runner._validate_pre_score_freeze(paths, frozen)


def test_run_orders_training_freeze_hash_roster_then_scoring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    paths = runner.RunPaths(
        checkpoint=tmp_path / "checkpoint.pt",
        frozen_eval=tmp_path / "frozen.npz",
        frozen_eval_metadata=tmp_path / "frozen.json",
        pre_score_freeze=tmp_path / "freeze.json",
        report=tmp_path / "report.json",
    )
    artifacts = SimpleNamespace(checkpoint=tmp_path / "parent.pt")
    monkeypatch.setattr(
        runner,
        "_load_preregistration",
        lambda _path: ({}, "config-sha", tuple(f"img_{i}.png" for i in range(16))),
    )
    monkeypatch.setattr(runner, "_validate_parent_artifacts", lambda _config: artifacts)
    monkeypatch.setattr(
        runner,
        "_validate_parent_commitment",
        lambda _artifacts, _names: {"synthetic_seed": 1_267_233_517},
    )
    monkeypatch.setattr(runner, "_validate_cache", lambda *_args: {})
    monkeypatch.setattr(
        runner,
        "_validate_fit_labels",
        lambda *_args: np.zeros((128, 1104), dtype=bool),
    )
    monkeypatch.setattr(runner, "_validate_parent_frozen_eval", lambda *_args: None)
    monkeypatch.setattr(runner, "_validate_parent_checkpoint", lambda *_args: {})
    monkeypatch.setattr(runner, "_prepare_run_paths", lambda _path: paths)
    monkeypatch.setattr(runner, "_select_device", lambda *_args, **_kwargs: torch.device("cpu"))

    def train(
        *_args: object, **_kwargs: object
    ) -> tuple[object, list[dict[str, float]], float, list[int]]:
        events.append("train")
        return object(), [{"loss": 1.0}] * 20, 1.0, list(range(200))

    def freeze(*_args: object, **_kwargs: object) -> tuple[str, str]:
        events.append("freeze")
        return "frozen-sha", "metadata-sha"

    def roster(*_args: object, **_kwargs: object) -> dict[str, dict[str, str]]:
        events.append("roster")
        return {}

    def score(
        *_args: object, **_kwargs: object
    ) -> tuple[list[object], dict[str, object], dict[str, bool]]:
        events.append("score")
        return [], {}, {"passed": False}

    def write(_path: Path, _payload: object) -> None:
        events.append("report")

    monkeypatch.setattr(runner, "_train_continuation", train)
    monkeypatch.setattr(runner, "_freeze_eval", freeze)
    monkeypatch.setattr(runner, "_freeze_pre_score_roster", roster)
    monkeypatch.setattr(runner, "_score_frozen_eval", score)
    monkeypatch.setattr(runner, "_write_json_exclusive", write)
    monkeypatch.setattr(runner, "sha256_file", lambda _path: "sha")
    monkeypatch.setattr(runner, "_record", lambda _path: {"path": "x", "sha256": "sha"})
    args = Namespace(
        config=tmp_path / "config.json",
        targets=tmp_path / "targets",
        output_dir=tmp_path / "out",
        device="cpu",
        allow_nondeterministic_mps=False,
        log_every=20,
    )
    runner.run(args)
    assert events == ["train", "freeze", "roster", "score", "report"]


def test_run_paths_refuse_to_overwrite_any_durable_artifact(tmp_path: Path) -> None:
    first = runner._prepare_run_paths(tmp_path / "run")
    first.pre_score_freeze.write_text("already exists", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner._prepare_run_paths(tmp_path / "run")


def test_recipe_rejects_any_arm_sweep() -> None:
    payload = _complete_recipe()
    changed = copy.deepcopy(payload)
    changed["protocol"]["hyperparameter_or_arm_sweep"] = True
    with pytest.raises(ValueError, match="hyperparameter_or_arm_sweep"):
        runner._validate_recipe(changed)
