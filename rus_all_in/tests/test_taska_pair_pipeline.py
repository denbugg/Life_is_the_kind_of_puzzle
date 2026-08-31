from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import aiijc_puzzle.taska_pair_pipeline as pipeline
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge, solve_raw_tail_global
from aiijc_puzzle.taska_edge_calibrator import (
    TaskaEdgeCalibrator,
    extract_taska_edge_features,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_focal_verifier import (
    TASKA_FOCAL_VERIFIER_SHA256,
    TaskaFocalScoreBatch,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail
from aiijc_puzzle.taska_seam_matcher import (
    TASKA_CHECKPOINTS,
    MutualVote,
    TaskaSeamMatchResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKA_FREEZE = (
    PROJECT_ROOT
    / "outputs/taska-seam-replay/opened32-mps-v1/frozen-target-free-eval.npz"
)
TASKA_FREEZE_METADATA = TASKA_FREEZE.with_suffix(".json")
FOCAL_FREEZE = (
    PROJECT_ROOT
    / "outputs/taska-focal-verifier/opened32-train-exact-top5-cpu-v2/"
    "frozen-target-free-eval.npz"
)
TASKA_FREEZE_SHA256 = "1880940897caeec6b87631d53e1aede1f809955a7acd3e56da9bcf432939e994"
FOCAL_FREEZE_SHA256 = "60243ab924da96d8bb49b072458c4710c65b8195b8d2c31eff1132b59ee56fd2"
FROZEN_CASE_FINAL_SHA256 = (
    "f515adf37aaa53382444440088b444e5c5ce9c2a287408d4b75e6ae29bab7414"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def resources() -> pipeline.TaskaPairPipelineResources:
    return pipeline.load_taska_pair_pipeline_resources(device="cpu")


def _match(
    right: np.ndarray,
    down: np.ndarray,
    edges: tuple[RawTailEdge, ...],
    *,
    weights: np.ndarray,
    votes: np.ndarray,
) -> TaskaSeamMatchResult:
    records = tuple(
        MutualVote(
            edge=edge,
            vote_count=int(vote),
            minimum_margin=float(weight),
            maximum_margin=float(weight),
        )
        for edge, weight, vote in zip(edges, weights, votes, strict=True)
    )
    return TaskaSeamMatchResult(
        right_log=np.ascontiguousarray(-right),
        down_log=np.ascontiguousarray(-down),
        cost_right=np.ascontiguousarray(right),
        cost_down=np.ascontiguousarray(down),
        candidate_edges=edges,
        vote_records=records,
        chosen_vote_threshold=10,
        scorer_count=12,
        checkpoint_sha256=(
            TASKA_CHECKPOINTS["v3"].sha256,
            TASKA_CHECKPOINTS["local"].sha256,
        ),
        config=pipeline.MATCHER_CONFIG,
    )


def _frozen_case() -> tuple[TaskaSeamMatchResult, TaskaFocalScoreBatch, np.ndarray, np.ndarray]:
    assert _sha256(TASKA_FREEZE) == TASKA_FREEZE_SHA256
    assert _sha256(FOCAL_FREEZE) == FOCAL_FREEZE_SHA256
    metadata = json.loads(TASKA_FREEZE_METADATA.read_text(encoding="utf-8"))["rows"][0]
    prefix = "case_0000"
    with np.load(TASKA_FREEZE, allow_pickle=False) as archive:
        right = archive[f"{prefix}__cost_right"].copy()
        down = archive[f"{prefix}__cost_down"].copy()
        right_log = archive[f"{prefix}__right_log"].copy()
        down_log = archive[f"{prefix}__down_log"].copy()
        source = archive[f"{prefix}__edge_source"].copy()
        target = archive[f"{prefix}__edge_target"].copy()
        axis = archive[f"{prefix}__edge_axis"].copy()
        weights = archive[f"{prefix}__edge_weight"].copy()
        votes = archive[f"{prefix}__edge_vote_count"].copy()
        frozen_raw = archive[f"{prefix}__taska_layout"].copy()
    edges = tuple(
        RawTailEdge(int(first), int(second), "right" if int(direction) == 0 else "down")
        for first, second, direction in zip(source, target, axis, strict=True)
    )
    records = tuple(
        MutualVote(edge, int(vote), float(weight), float(weight))
        for edge, weight, vote in zip(edges, weights, votes, strict=True)
    )
    matched = TaskaSeamMatchResult(
        right_log=right_log,
        down_log=down_log,
        cost_right=right,
        cost_down=down,
        candidate_edges=edges,
        vote_records=records,
        chosen_vote_threshold=int(metadata["chosen_vote_threshold"]),
        scorer_count=int(metadata["scorer_count"]),
        checkpoint_sha256=tuple(metadata["checkpoint_sha256"]),
        config=pipeline.MATCHER_CONFIG,
    )
    with np.load(FOCAL_FREEZE, allow_pickle=False) as archive:
        logits = archive[f"{prefix}__focal_logits"].copy()
        features = archive[f"{prefix}__focal_features"].copy()
        frozen_focal = archive[f"{prefix}__focal_layout"].copy()
    focal = TaskaFocalScoreBatch(
        logits=logits,
        features=features,
        edges=edges,
        mode=pipeline.FOCAL_MODE,
        checkpoint_sha256=TASKA_FOCAL_VERIFIER_SHA256,
    )
    return matched, focal, frozen_raw, frozen_focal


def test_all_artifacts_and_frozen_solver_are_sha_gated() -> None:
    assert pipeline.verify_taska_pair_artifacts() == pipeline.EXPECTED_ARTIFACT_SHA256
    assert dict(pipeline.EXPECTED_ARTIFACT_SHA256) == {
        "matcher_v3": TASKA_CHECKPOINTS["v3"].sha256,
        "matcher_local": TASKA_CHECKPOINTS["local"].sha256,
        "logistic_calibrator": pipeline.LOGISTIC_CALIBRATOR_SHA256,
        "focal_verifier": TASKA_FOCAL_VERIFIER_SHA256,
        "nonlinear_calibrator": pipeline.NONLINEAR_CALIBRATOR_SHA256,
        "raw_tail_global_solver": pipeline.RAW_TAIL_GLOBAL_SOLVER_SHA256,
    }


def test_bad_artifact_fails_before_any_deserialisation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = tmp_path / "calibrator.npz"
    bad.write_bytes(b"not the trained calibrator")
    paths = replace(pipeline.TaskaPairArtifactPaths(), logistic_calibrator=bad)
    deserialised = False

    def forbidden(*_args: object, **_kwargs: object) -> TaskaEdgeCalibrator:
        nonlocal deserialised
        deserialised = True
        raise AssertionError("deserialisation must happen only after the complete hash gate")

    monkeypatch.setattr(TaskaEdgeCalibrator, "load_npz", forbidden)
    with pytest.raises(pipeline.TaskaPairArtifactError, match="logistic_calibrator SHA"):
        pipeline.load_taska_pair_pipeline_resources(paths, device="cpu")
    assert deserialised is False


def test_frozen_one_case_matches_independent_four_arm_composition(
    resources: pipeline.TaskaPairPipelineResources,
) -> None:
    matched, focal_scores, frozen_raw, frozen_focal = _frozen_case()
    actual = pipeline._compose_taska_pair_layouts(matched, focal_scores, resources)

    weights = np.asarray(
        [record.minimum_margin for record in matched.vote_records], dtype=np.float64
    )
    votes = np.asarray([record.vote_count for record in matched.vote_records], dtype=np.float64)
    features = extract_taska_edge_features(
        matched.cost_right,
        matched.cost_down,
        matched.right_log,
        matched.down_log,
        matched.candidate_edges,
        weights,
        votes,
        grid=24,
    )
    raw = solve_raw_tail_global(
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        grid=24,
        config=pipeline.SOLVER_CONFIG,
    )
    logistic = solve_prioritized_raw_tail_global(
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        resources.logistic_calibrator.predict_priorities(features.values),
        grid=24,
        config=pipeline.SOLVER_CONFIG,
    )
    focal = solve_prioritized_raw_tail_global(
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        focal_scores.logits,
        grid=24,
        config=pipeline.SOLVER_CONFIG,
    )
    nonlinear = solve_prioritized_raw_tail_global(
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        resources.nonlinear_calibrator.predict_priorities(features.values),
        grid=24,
        config=pipeline.SOLVER_CONFIG,
    )
    layouts = {
        name: result.layout
        for name, result in zip(
            pipeline.ARM_NAMES,
            (raw, logistic, focal, nonlinear),
            strict=True,
        )
    }
    selection = select_lowest_taska_seam_cost_layout(
        layouts,
        matched.cost_right,
        matched.cost_down,
        grid=24,
    )
    expected = polish_unprotected_taska_tail(
        selection.layout,
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        grid=24,
        max_swaps=96,
        minimum_gain=1e-9,
    )

    np.testing.assert_array_equal(raw.layout, frozen_raw)
    np.testing.assert_array_equal(focal.layout, frozen_focal)
    np.testing.assert_array_equal(actual.layout, expected.layout)
    assert actual.choice == selection.choice == "nonlinear"
    assert actual.costs == selection.total_costs
    assert actual.layout_sha256 == FROZEN_CASE_FINAL_SHA256
    assert actual.diagnostics.tail.accepted_swap_count == 96
    assert not actual.layout.flags.writeable
    with pytest.raises(ValueError):
        actual.layout[0] = actual.layout[1]


def test_composition_is_equivariant_to_bag_relabelling_when_costs_have_no_ties(
    resources: pipeline.TaskaPairPipelineResources,
) -> None:
    grid = 3
    count = grid * grid
    generator = np.random.default_rng(681)
    right = generator.uniform(0.1, 30.0, size=(count, count))
    down = generator.uniform(0.1, 30.0, size=(count, count))
    np.fill_diagonal(right, 0.0)
    np.fill_diagonal(down, 0.0)
    edges = (
        RawTailEdge(0, 1, "right"),
        RawTailEdge(1, 2, "right"),
        RawTailEdge(0, 3, "down"),
        RawTailEdge(3, 4, "right"),
        RawTailEdge(4, 5, "down"),
        RawTailEdge(6, 7, "right"),
        RawTailEdge(8, 2, "down"),
        RawTailEdge(5, 6, "right"),
        RawTailEdge(7, 3, "down"),
    )
    weights = np.linspace(0.2, 1.8, len(edges))
    votes = np.asarray([10, 12, 11, 10, 12, 11, 10, 12, 11])
    logits = generator.normal(size=len(edges)).astype(np.float32)
    matched = _match(right, down, edges, weights=weights, votes=votes)
    focal = TaskaFocalScoreBatch(
        logits=logits,
        features=np.zeros((len(edges), 6), dtype=np.float32),
        edges=edges,
        mode=pipeline.FOCAL_MODE,
        checkpoint_sha256=TASKA_FOCAL_VERIFIER_SHA256,
    )
    original = pipeline._compose_taska_pair_layouts(matched, focal, resources, grid=grid)

    order = generator.permutation(count)
    inverse = np.empty(count, dtype=np.int64)
    inverse[order] = np.arange(count)
    relabelled_edges = tuple(
        RawTailEdge(int(inverse[edge.source]), int(inverse[edge.target]), edge.axis)
        for edge in edges
    )
    relabelled_match = _match(
        right[np.ix_(order, order)],
        down[np.ix_(order, order)],
        relabelled_edges,
        weights=weights,
        votes=votes,
    )
    relabelled_focal = TaskaFocalScoreBatch(
        logits=logits,
        features=np.zeros((len(edges), 6), dtype=np.float32),
        edges=relabelled_edges,
        mode=pipeline.FOCAL_MODE,
        checkpoint_sha256=TASKA_FOCAL_VERIFIER_SHA256,
    )
    relabelled = pipeline._compose_taska_pair_layouts(
        relabelled_match,
        relabelled_focal,
        resources,
        grid=grid,
    )

    np.testing.assert_array_equal(order[relabelled.layout], original.layout)
    assert relabelled.choice == original.choice
    assert np.allclose(
        [value for _, value in relabelled.costs],
        [value for _, value in original.costs],
        atol=1e-10,
        rtol=1e-10,
    )


def test_cli_contract_is_layout_only() -> None:
    arguments = pipeline.parse_args(
        [
            "tiles.npy",
            "--output-layout",
            "layout.npy",
            "--diagnostics-json",
            "diagnostics.json",
            "--device",
            "cpu",
        ]
    )
    assert arguments.tiles == Path("tiles.npy")
    assert arguments.output_layout == Path("layout.npy")
    assert arguments.diagnostics_json == Path("diagnostics.json")
    assert not hasattr(arguments, "output_image")

