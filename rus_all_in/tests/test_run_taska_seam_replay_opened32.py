from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "run_taska_seam_replay_opened32.py"
    specification = importlib.util.spec_from_file_location(
        "run_taska_seam_replay_opened32_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _complete_recipe() -> dict[str, object]:
    return json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_signed_fixed_recipe_rejects_historical_leaky_quad(tmp_path: Path) -> None:
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

    changed = copy.deepcopy(payload)
    changed["harvest"]["quad_weight"] = 0.4
    path.write_text(json.dumps(changed), encoding="utf-8")
    changed_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256").write_text(
        f"{changed_digest}  {path.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="quad_weight"):
        runner._load_preregistration(path)


def test_recipe_fixes_legal_historical_transplant_and_pair_metric() -> None:
    payload = _complete_recipe()
    assert payload["matcher"]["kinds"] == ["v3", "local"]
    assert payload["matcher"]["views"] == ["raw", "median", "bilateral"]
    assert payload["harvest"] == {
        "depth": 1,
        "votes_fallback": 10,
        "vote_target": 350,
        "weighted": False,
        "margin": 0.0,
        "order": "raw_fused_score",
        "quad_weight": 0.0,
        "historical_quad_0_4_excluded_as_target_id_dependent": True,
    }
    assert payload["solver"]["border_unary"] is False
    assert payload["evaluation"]["pair_denominator"] == 1104


def test_pair_metric_uses_integer_adjacency_count_with_recall_cross_check() -> None:
    reference = np.arange(runner.COUNT, dtype=np.int32)
    evaluation = evaluate_layout(reference, reference, reference_is_exact=True)
    assert runner._layout_metrics(evaluation) == {
        "satisfied_adjacent_pairs": 1104,
        "adjacency_recall": 1.0,
        "exact_tiles": 576,
        "strict_permutation": True,
    }


def test_dirty_only_freeze_persists_scores_harvest_and_strict_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dirty_tiles = np.zeros((runner.COUNT, 20, 20, 3), dtype=np.uint8)
    edge = RawTailEdge(7, 11, "right")
    matrix = np.ones((runner.COUNT, runner.COUNT), dtype=np.float64)
    matched = SimpleNamespace(
        cost_right=matrix,
        cost_down=matrix + 1,
        right_log=-matrix,
        down_log=-(matrix + 1),
        candidate_edges=(edge,),
        vote_records=(
            SimpleNamespace(edge=edge, minimum_margin=2.5, vote_count=10),
        ),
        chosen_vote_threshold=10,
        scorer_count=12,
        checkpoint_sha256=(runner.V3_CHECKPOINT_SHA256, runner.LOCAL_CHECKPOINT_SHA256),
        diagnostics={"selected_vote_threshold": 10},
    )
    solver_diagnostics = SimpleNamespace(
        as_dict=lambda: {"strict_permutation": True, "candidate_edges": 1}
    )
    solver = SimpleNamespace(
        layout=np.arange(runner.COUNT, dtype=np.int32),
        diagnostics=solver_diagnostics,
    )
    dirty = runner.DirtyCase("case-id", "img_000001.png", dirty_tiles)
    dirty_sha = "d" * 64
    parent_row = {
        "prefix": "case_0000",
        "case_id": dirty.case_id,
        "source_filename": dirty.source_filename,
        "draw_index": 0,
        "dirty_sha256": dirty_sha,
    }
    paths = runner.RunPaths(
        frozen_eval=tmp_path / "frozen.npz",
        frozen_eval_metadata=tmp_path / "frozen.json",
        pre_score_freeze=tmp_path / "freeze.json",
        report=tmp_path / "report.json",
    )
    artifacts = runner.Artifacts(
        selection_commitment=tmp_path / "selection.json",
        parent_frozen_eval=tmp_path / "parent.npz",
        parent_frozen_eval_metadata=tmp_path / "parent.json",
        matcher_v3=tmp_path / "v3.pt",
        matcher_local=tmp_path / "local.pt",
    )

    monkeypatch.setattr(runner, "_load_matchers", lambda *_args, **_kwargs: (object(),))
    monkeypatch.setattr(runner, "CleanTileCache", lambda _path: object())
    monkeypatch.setattr(runner, "_dirty_case", lambda *_args, **_kwargs: dirty)
    monkeypatch.setattr(runner, "_dirty_sha256", lambda _tiles: dirty_sha)
    monkeypatch.setattr(runner, "_match_dirty_tiles", lambda *_args, **_kwargs: matched)
    monkeypatch.setattr(runner, "solve_raw_tail_global", lambda *_args, **_kwargs: solver)

    _seconds, count = runner._freeze_target_free_eval(
        paths,
        [({}, dirty.source_filename, 0)],
        [parent_row],
        artifacts,
        targets=tmp_path / "targets",
        device=runner.torch.device("cpu"),
    )
    assert count == 1
    with np.load(paths.frozen_eval) as archive:
        assert set(archive.files) == {
            "case_0000__cost_right",
            "case_0000__cost_down",
            "case_0000__right_log",
            "case_0000__down_log",
            "case_0000__edge_source",
            "case_0000__edge_target",
            "case_0000__edge_axis",
            "case_0000__edge_weight",
            "case_0000__edge_vote_count",
            "case_0000__taska_layout",
        }
        assert archive["case_0000__edge_source"].tolist() == [7]
        assert archive["case_0000__edge_target"].tolist() == [11]
        assert archive["case_0000__edge_axis"].tolist() == [0]
        assert archive["case_0000__edge_vote_count"].tolist() == [10]
        assert np.array_equal(
            archive["case_0000__taska_layout"],
            np.arange(runner.COUNT),
        )
    metadata = json.loads(paths.frozen_eval_metadata.read_text(encoding="utf-8"))
    assert metadata["contains_exact_references_or_labels"] is False
    assert metadata["contains_target_ids_or_source_grid_coordinates"] is False
    assert metadata["contains_dirty_derived_scores"] is True
    assert metadata["quad_weight"] == 0.0
    assert all("reference" not in key for key in metadata["rows"][0])


def test_scoring_validates_freeze_before_constructing_target_cache(
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
    artifacts = SimpleNamespace(parent_frozen_eval_metadata=tmp_path / "absent")
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


def test_summary_reports_pairs_recall_exact_and_smoke_has_no_fake_ci() -> None:
    row = {
        "source_filename": "img.png",
        "union_v2": {
            "satisfied_adjacent_pairs": 100,
            "adjacency_recall": 100 / 1104,
            "exact_tiles": 1,
        },
        "learned_priority": {
            "satisfied_adjacent_pairs": 110,
            "adjacency_recall": 110 / 1104,
            "exact_tiles": 2,
        },
        "taska_legal_raw_tail": {
            "satisfied_adjacent_pairs": 150,
            "adjacency_recall": 150 / 1104,
            "exact_tiles": 3,
        },
    }
    summary = runner._summarize_rows([row], full_panel=False)
    assert summary["arms"]["taska_legal_raw_tail"]["satisfied_adjacent_pairs"] == 150
    delta = summary["candidate_deltas"]["learned_priority"]
    assert delta["satisfied_adjacent_pairs"]["mean"] == 40
    assert delta["exact_tiles"]["mean"] == 1
    assert delta["satisfied_adjacent_pairs"]["ci95_lower"] is None
    assert delta["satisfied_adjacent_pairs"]["smoke_only"] is True


def test_strict_layout_rejects_duplicates_and_wrong_shape() -> None:
    duplicate = np.arange(runner.COUNT, dtype=np.int32)
    duplicate[-1] = duplicate[-2]
    with pytest.raises(ValueError, match="strict"):
        runner._strict_layout(duplicate)
    with pytest.raises(ValueError, match="strict"):
        runner._strict_layout(np.arange(4))
