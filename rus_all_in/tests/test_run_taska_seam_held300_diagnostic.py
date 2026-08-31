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
    path = SCRIPTS / "run_taska_seam_held300_diagnostic.py"
    specification = importlib.util.spec_from_file_location(
        "run_taska_seam_held300_diagnostic_test",
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


def test_signed_roster_is_deterministic_last300_and_opened32_disjoint() -> None:
    payload, digest, names = runner._load_preregistration(runner.DEFAULT_CONFIG)
    assert digest == hashlib.sha256(runner.DEFAULT_CONFIG.read_bytes()).hexdigest()
    assert names == runner._deterministic_source_roster()
    assert len(names) == 16
    assert all(6_700 <= int(name[4:10]) <= 6_999 for name in names)
    assert runner._names_digest(runner._eligible_names()) == runner.UNIVERSE_DIGEST

    artifacts = runner._validate_artifacts(payload)
    opened32 = runner._validate_reference_recipe(artifacts, payload)
    assert not (set(names) & opened32)


def test_held300_recipe_exactly_reuses_opened32_matcher_harvest_solver() -> None:
    payload = _complete_recipe()
    opened = json.loads(
        (PROJECT_ROOT / "configs/taska_seam_replay_opened32_v1.json").read_text(encoding="utf-8")
    )
    for section in ("matcher", "harvest", "solver"):
        assert payload[section] == opened[section]
    assert payload["protocol"]["historical_matcher_training_disjoint"] is True
    assert payload["protocol"]["historical_model_selection_exposed"] is True
    assert payload["protocol"]["fresh_promotion_claimed"] is False


def test_signed_recipe_rejects_hyperparameter_change(tmp_path: Path) -> None:
    changed = copy.deepcopy(_complete_recipe())
    changed["harvest"]["vote_target"] = 349
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="harvest"):
        runner._load_preregistration(path)


def test_pair_metric_and_source_clustered_intervals() -> None:
    reference = np.arange(runner.COUNT, dtype=np.int32)
    evaluation = evaluate_layout(reference, reference, reference_is_exact=True)
    assert runner._layout_metrics(evaluation) == {
        "satisfied_adjacent_pairs": 1104,
        "adjacency_recall": 1.0,
        "exact_tiles": 576,
        "strict_permutation": True,
    }

    summary = runner.source_clustered_mean_ci(
        [100.0, 120.0, 200.0, 220.0],
        ["a", "a", "b", "b"],
        seed=7,
        resamples=1_000,
    )
    assert summary["mean"] == 160.0
    assert summary["source_cluster_mean"] == 160.0
    assert summary["ci95_lower"] <= 160.0 <= summary["ci95_upper"]
    assert summary["source_count"] == 2
    assert summary["case_count"] == 4

    report = runner._summarize_rows(
        [
            {
                "source_filename": "a",
                "taska_legal_raw_tail": {
                    "satisfied_adjacent_pairs": 100,
                    "adjacency_recall": 100 / 1104,
                    "exact_tiles": 2,
                },
            }
        ],
        full_panel=False,
    )
    assert report["pair_denominator"] == 1104
    assert report["satisfied_adjacent_pairs_per_board"]["mean"] == 100
    assert report["exact_tiles_per_board"]["mean"] == 2


def test_dirty_only_generator_matches_later_exact_recreation() -> None:
    clean = np.random.default_rng(9).integers(
        0,
        256,
        size=(runner.COUNT, 20, 20, 3),
        dtype=np.uint8,
    )
    cache = SimpleNamespace(load=lambda _record: clean)
    source = "img_006700.png"
    dirty_only = runner._dirty_case(cache, {"filename": source}, source, 1)
    dirty, reference = runner.make_exact_synthetic_case(
        clean,
        source_filename=source,
        draw_index=1,
        seed=runner.SYNTHETIC_SEED,
    )
    assert dirty_only.case_id == dirty.case_id == reference.case_id
    assert np.array_equal(dirty_only.dirty_tiles, dirty.tiles)
    assert np.array_equal(np.sort(reference.tile_at_position), np.arange(runner.COUNT))


def test_dirty_only_freeze_persists_scores_harvest_and_strict_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dirty_tiles = np.zeros((runner.COUNT, 20, 20, 3), dtype=np.uint8)
    edge = RawTailEdge(7, 11, "right")
    matrix = np.ones((runner.COUNT, runner.COUNT), dtype=np.float64)
    vote = SimpleNamespace(edge=edge, minimum_margin=2.5, vote_count=10)
    matched = SimpleNamespace(
        cost_right=matrix,
        cost_down=matrix + 1,
        right_log=-matrix,
        down_log=-(matrix + 1),
        candidate_edges=(edge,),
        vote_records=(vote,),
        chosen_vote_threshold=10,
        scorer_count=12,
        checkpoint_sha256=(runner.V3_CHECKPOINT_SHA256, runner.LOCAL_CHECKPOINT_SHA256),
    )
    solved = SimpleNamespace(
        layout=np.arange(runner.COUNT, dtype=np.int32),
        diagnostics=SimpleNamespace(as_dict=lambda: {"strict_permutation": True}),
    )
    dirty = runner.DirtyCase("case-id", "img_006700.png", 0, dirty_tiles)
    paths = runner.RunPaths(
        frozen_eval=tmp_path / "frozen.npz",
        frozen_eval_metadata=tmp_path / "frozen.json",
        pre_score_freeze=tmp_path / "freeze.json",
        report=tmp_path / "report.json",
    )
    artifacts = runner.Artifacts(
        manifest=tmp_path / "manifest.json",
        opened32_recipe=tmp_path / "opened.json",
        matcher_v3=tmp_path / "v3.pt",
        matcher_local=tmp_path / "local.pt",
    )

    monkeypatch.setattr(runner, "load_default_taska_ensemble", lambda *_a, **_k: (1, 2))
    monkeypatch.setattr(runner, "CleanTileCache", lambda *_a, **_k: object())
    monkeypatch.setattr(runner, "_dirty_case", lambda *_a, **_k: dirty)
    monkeypatch.setattr(runner, "match_taska_tiles", lambda *_a, **_k: matched)
    monkeypatch.setattr(runner, "solve_raw_tail_global", lambda *_a, **_k: solved)

    _seconds, count = runner._freeze_target_free_eval(
        paths,
        [({}, dirty.source_filename, 0)],
        artifacts,
        targets=tmp_path / "targets",
        device=runner.torch.device("cpu"),
    )
    assert count == 1
    with np.load(paths.frozen_eval) as archive:
        assert archive["case_0000__edge_source"].tolist() == [7]
        assert archive["case_0000__edge_target"].tolist() == [11]
        assert archive["case_0000__edge_vote_count"].tolist() == [10]
        assert np.array_equal(
            archive["case_0000__taska_layout"],
            np.arange(runner.COUNT),
        )
    metadata = json.loads(paths.frozen_eval_metadata.read_text(encoding="utf-8"))
    assert metadata["contains_exact_references_or_labels"] is False
    assert metadata["contains_target_ids_or_source_grid_coordinates"] is False
    assert metadata["contains_strict_original_tile_layouts"] is True


def test_scoring_validates_freeze_before_constructing_target_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_constructed = False

    def construct_cache(*_args: object, **_kwargs: object) -> None:
        nonlocal cache_constructed
        cache_constructed = True
        raise AssertionError("target cache must not be constructed")

    def reject_freeze(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("frozen roster changed")

    monkeypatch.setattr(runner, "CleanTileCache", construct_cache)
    monkeypatch.setattr(runner, "_validate_pre_score_freeze", reject_freeze)
    paths = SimpleNamespace(
        pre_score_freeze=tmp_path / "absent",
        frozen_eval_metadata=tmp_path / "absent-metadata",
    )
    with pytest.raises(RuntimeError, match="frozen roster"):
        runner._score_frozen_eval(
            paths,
            [],
            {},
            targets=tmp_path / "targets",
        )
    assert cache_constructed is False


def test_output_directory_is_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "run"
    paths = runner._prepare_run_paths(output)
    assert paths.report.parent == output
    with pytest.raises(FileExistsError, match="refusing"):
        runner._prepare_run_paths(output)


def test_strict_layout_rejects_duplicate_or_wrong_shape() -> None:
    duplicate = np.arange(runner.COUNT, dtype=np.int32)
    duplicate[-1] = duplicate[-2]
    with pytest.raises(ValueError, match="strict"):
        runner._strict_layout(duplicate)
    with pytest.raises(ValueError, match="strict"):
        runner._strict_layout(np.arange(4))
