"""Production legal TASKA pair-oriented layout pipeline.

The pipeline consumes one bag of 576 original upright 20x20 RGB fragments and
returns only a strict ``tile_at_position`` permutation.  It never renders,
replaces, rotates, warps, or otherwise changes pixels.

The fixed production composition is:

1. audited TASKA v3 + local matching on raw/median/bilateral views;
2. four layouts using raw, train256 logistic, focal-top5, and portable
   nonlinear component priorities;
3. target-free selection by the original TASKA cost on all 1,104 board bonds;
4. protected-tail seam polish with ``max_swaps=96``.

Every persisted model is SHA-256 gated before deserialisation.  The imported
frozen raw solver is also byte-gated, but is not modified or copied here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

import aiijc_puzzle.raw_tail_global_solver as raw_tail_global_solver
from aiijc_puzzle.raw_tail_global_solver import (
    RawTailGlobalConfig,
    RawTailGlobalDiagnostics,
    solve_raw_tail_global,
)
from aiijc_puzzle.taska_edge_calibrator import (
    PrioritizedRawTailResult,
    TaskaEdgeCalibrator,
    extract_taska_edge_features,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_focal_verifier import (
    TASKA_FOCAL_VERIFIER_SHA256,
    SeamVerifier,
    TaskaFocalScoreBatch,
    load_taska_focal_verifier,
    score_focal_edges,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_nonlinear_calibrator import TaskaNonlinearCalibrator
from aiijc_puzzle.taska_protected_tail_polish import (
    TaskaProtectedTailDiagnostics,
    polish_unprotected_taska_tail,
)
from aiijc_puzzle.taska_seam_matcher import (
    TASKA_CHECKPOINTS,
    MutualVote,
    SeamEmbed,
    TaskaSeamConfig,
    TaskaSeamMatchResult,
    load_taska_checkpoint,
    match_taska_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRID_SIZE = 24
TILE_COUNT = GRID_SIZE * GRID_SIZE
PAIR_DENOMINATOR = 2 * GRID_SIZE * (GRID_SIZE - 1)
ARM_NAMES = ("raw", "logistic", "focal_top5", "nonlinear")
FOCAL_MODE = "train_exact_top5"
TAIL_MAX_SWAPS = 96
TAIL_MINIMUM_GAIN = 1e-9

LOGISTIC_CALIBRATOR_SHA256 = (
    "adc76ee87fc112d4ca3eeb676cdec6b7d103c596d62a9848ba65ee5ef384b1ac"
)
NONLINEAR_CALIBRATOR_SHA256 = (
    "2a5f95bd9d8e08e57b8bd02e242e25ef4661036ed3b1985fda1d70ee1bf9d2a6"
)
RAW_TAIL_GLOBAL_SOLVER_SHA256 = (
    "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
)

MATCHER_CONFIG = TaskaSeamConfig(
    views=("raw", "median", "bilateral"),
    orientations=2,
    votes=10,
    vote_target=350,
    margin=0.0,
    depth=1,
    quad_weight=0.0,
    rounds=3,
    cycle_weight=0.35,
    sinkhorn_iterations=20,
    acyclic_weight=3.0,
)
SOLVER_CONFIG = RawTailGlobalConfig(
    baseline_quantile=0.15,
    search_rounds=6,
    border_weight=0.0,
    random_seed=0,
    component_cap=0,
    fill_rounds=1,
)


class TaskaPairArtifactError(RuntimeError):
    """A production artifact or the frozen raw solver differs from its manifest."""


@dataclass(frozen=True)
class TaskaPairArtifactPaths:
    """Paths for the five model artifacts used by the fixed pipeline."""

    matcher_v3: Path = PROJECT_ROOT / "artifacts/prior-taska/ckpt/seam_embed_v3.pt"
    matcher_local: Path = PROJECT_ROOT / "artifacts/prior-taska/ckpt/seam_embed_local.pt"
    logistic_calibrator: Path = (
        PROJECT_ROOT / "outputs/taska-edge-calibrator/train256-v1/calibrator.npz"
    )
    focal_verifier: Path = (
        PROJECT_ROOT / "artifacts/prior-taska/ckpt/verify_pair_best.pt"
    )
    nonlinear_calibrator: Path = (
        PROJECT_ROOT / "outputs/taska-nonlinear-calibrator/train256-v1/calibrator.npz"
    )


ArtifactName = Literal[
    "matcher_v3",
    "matcher_local",
    "logistic_calibrator",
    "focal_verifier",
    "nonlinear_calibrator",
    "raw_tail_global_solver",
]


EXPECTED_ARTIFACT_SHA256: tuple[tuple[ArtifactName, str], ...] = (
    ("matcher_v3", TASKA_CHECKPOINTS["v3"].sha256),
    ("matcher_local", TASKA_CHECKPOINTS["local"].sha256),
    ("logistic_calibrator", LOGISTIC_CALIBRATOR_SHA256),
    ("focal_verifier", TASKA_FOCAL_VERIFIER_SHA256),
    ("nonlinear_calibrator", NONLINEAR_CALIBRATOR_SHA256),
    ("raw_tail_global_solver", RAW_TAIL_GLOBAL_SOLVER_SHA256),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_files(paths: TaskaPairArtifactPaths) -> tuple[tuple[ArtifactName, Path], ...]:
    solver_path = Path(raw_tail_global_solver.__file__).resolve()
    return (
        ("matcher_v3", Path(paths.matcher_v3).expanduser().resolve()),
        ("matcher_local", Path(paths.matcher_local).expanduser().resolve()),
        ("logistic_calibrator", Path(paths.logistic_calibrator).expanduser().resolve()),
        ("focal_verifier", Path(paths.focal_verifier).expanduser().resolve()),
        ("nonlinear_calibrator", Path(paths.nonlinear_calibrator).expanduser().resolve()),
        ("raw_tail_global_solver", solver_path),
    )


def verify_taska_pair_artifacts(
    paths: TaskaPairArtifactPaths | None = None,
) -> tuple[tuple[ArtifactName, str], ...]:
    """Verify every artifact before any pipeline deserialisation occurs."""

    if paths is None:
        paths = TaskaPairArtifactPaths()
    expected = dict(EXPECTED_ARTIFACT_SHA256)
    records: list[tuple[ArtifactName, str]] = []
    for name, path in _artifact_files(paths):
        if not path.is_file():
            raise TaskaPairArtifactError(f"{name} is absent: {path}")
        digest = _sha256_file(path)
        if digest != expected[name]:
            raise TaskaPairArtifactError(
                f"{name} SHA-256 mismatch: expected {expected[name]}, got {digest}"
            )
        records.append((name, digest))
    return tuple(records)


@dataclass(frozen=True)
class TaskaPairPipelineResources:
    """Loaded, frozen inference objects and their verified provenance."""

    matchers: tuple[SeamEmbed, SeamEmbed]
    logistic_calibrator: TaskaEdgeCalibrator
    focal_verifier: SeamVerifier
    nonlinear_calibrator: TaskaNonlinearCalibrator
    device: torch.device
    artifact_sha256: tuple[tuple[ArtifactName, str], ...]

    def __post_init__(self) -> None:
        if self.artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
            raise ValueError("resource artifact provenance differs from the production manifest")
        if len(self.matchers) != 2:
            raise ValueError("resources require exactly the v3 and local matchers")
        kinds = tuple(
            getattr(getattr(model, "checkpoint_spec", None), "kind", None)
            for model in self.matchers
        )
        if kinds != ("v3", "local"):
            raise ValueError("resource matcher order must be v3 then local")
        if not isinstance(self.logistic_calibrator, TaskaEdgeCalibrator):
            raise TypeError("logistic_calibrator has the wrong type")
        if not isinstance(self.nonlinear_calibrator, TaskaNonlinearCalibrator):
            raise TypeError("nonlinear_calibrator has the wrong type")
        if not isinstance(self.focal_verifier, SeamVerifier):
            raise TypeError("focal_verifier has the wrong type")
        if getattr(self.focal_verifier, "checkpoint_sha256", None) != (
            TASKA_FOCAL_VERIFIER_SHA256
        ):
            raise ValueError("focal verifier provenance differs")
        devices = tuple(next(model.parameters()).device for model in self.matchers)
        focal_device = next(self.focal_verifier.parameters()).device
        if any(device != self.device for device in (*devices, focal_device)):
            raise ValueError("all loaded networks must share the declared device")


def load_taska_pair_pipeline_resources(
    paths: TaskaPairArtifactPaths | None = None,
    *,
    device: str | torch.device | None = "auto",
) -> TaskaPairPipelineResources:
    """SHA-gate and load the complete fixed production model set once."""

    if paths is None:
        paths = TaskaPairArtifactPaths()
    records = verify_taska_pair_artifacts(paths)
    v3 = load_taska_checkpoint(paths.matcher_v3, "v3", device=device)
    local = load_taska_checkpoint(paths.matcher_local, "local", device=device)
    actual_device = next(v3.parameters()).device
    if next(local.parameters()).device != actual_device:
        raise RuntimeError("matcher devices differ after loading")
    logistic = TaskaEdgeCalibrator.load_npz(paths.logistic_calibrator)
    focal = load_taska_focal_verifier(paths.focal_verifier, device=actual_device)
    nonlinear = TaskaNonlinearCalibrator.load_npz(paths.nonlinear_calibrator)
    # Detect a local write racing the parsers above.
    if verify_taska_pair_artifacts(paths) != records:
        raise TaskaPairArtifactError("an artifact changed while resources were loading")
    return TaskaPairPipelineResources(
        matchers=(v3, local),
        logistic_calibrator=logistic,
        focal_verifier=focal,
        nonlinear_calibrator=nonlinear,
        device=actual_device,
        artifact_sha256=records,
    )


@dataclass(frozen=True)
class TaskaPairArmDiagnostics:
    """Compact target-free evidence for one pre-selection layout arm."""

    name: str
    layout_sha256: str
    total_cost: float
    solver: RawTailGlobalDiagnostics


@dataclass(frozen=True)
class TaskaPairPipelineDiagnostics:
    """All target-free matcher, solver, selector, and tail diagnostics."""

    grid_size: int
    pair_denominator: int
    candidate_edge_count: int
    chosen_vote_threshold: int
    scorer_count: int
    matcher_checkpoint_sha256: tuple[str, ...]
    artifact_sha256: tuple[tuple[ArtifactName, str], ...]
    focal_mode: str
    tail_max_swaps: int
    arms: tuple[TaskaPairArmDiagnostics, ...]
    tail: TaskaProtectedTailDiagnostics


@dataclass(frozen=True)
class TaskaPairPipelineResult:
    """Strict read-only layout plus the complete target-free choice record."""

    layout: np.ndarray
    choice: str
    costs: tuple[tuple[str, float], ...]
    diagnostics: TaskaPairPipelineDiagnostics

    def __post_init__(self) -> None:
        count = self.diagnostics.grid_size**2
        raw = np.asarray(self.layout)
        if raw.shape != (count,) or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
            raise ValueError(f"layout must be one integer vector of length {count}")
        layout = np.ascontiguousarray(raw, dtype=np.int32)
        if not np.array_equal(np.sort(layout), np.arange(count)):
            raise ValueError("layout must contain every original tile exactly once")
        if self.choice not in ARM_NAMES:
            raise ValueError("choice is outside the fixed four-arm roster")
        if tuple(name for name, _ in self.costs) != ARM_NAMES:
            raise ValueError("costs must follow the fixed four-arm roster")
        if not all(np.isfinite(value) for _, value in self.costs):
            raise ValueError("all portfolio costs must be finite")
        layout = layout.copy()
        layout.setflags(write=False)
        object.__setattr__(self, "layout", layout)

    @property
    def layout_sha256(self) -> str:
        canonical = self.layout.astype("<i4", copy=False)
        return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self.diagnostics)
        payload.update(
            {
                "choice": self.choice,
                "costs": dict(self.costs),
                "final_total_cost": self.diagnostics.tail.final_total_cost,
                "layout_sha256": self.layout_sha256,
                "strict_original_tile_permutation": True,
                "pixels_emitted": False,
            }
        )
        return payload


def _layout_sha256(layout: Any) -> str:
    canonical = np.asarray(layout, dtype="<i4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _edge_evidence(matched: TaskaSeamMatchResult) -> tuple[np.ndarray, np.ndarray]:
    records: dict[Any, MutualVote] = {}
    for record in matched.vote_records:
        if not isinstance(record, MutualVote):
            raise TypeError("vote_records must contain MutualVote values")
        if record.edge in records:
            raise ValueError("vote_records contain a duplicate edge")
        records[record.edge] = record
    if set(records) != set(matched.candidate_edges):
        raise ValueError("vote_records and candidate_edges differ")
    weights = np.asarray(
        [records[edge].minimum_margin for edge in matched.candidate_edges],
        dtype=np.float64,
    )
    votes = np.asarray(
        [records[edge].vote_count for edge in matched.candidate_edges],
        dtype=np.float64,
    )
    return weights, votes


def _validate_fixed_match(matched: TaskaSeamMatchResult) -> None:
    if not isinstance(matched, TaskaSeamMatchResult):
        raise TypeError("matched must be a TaskaSeamMatchResult")
    if matched.config != MATCHER_CONFIG:
        raise ValueError("matcher configuration differs from the fixed production recipe")
    expected = (TASKA_CHECKPOINTS["v3"].sha256, TASKA_CHECKPOINTS["local"].sha256)
    if matched.checkpoint_sha256 != expected:
        raise ValueError("matcher checkpoint provenance differs from the production recipe")
    if matched.scorer_count != 12:
        raise ValueError("production matcher must expose exactly 12 scorers")
    if not 1 <= matched.chosen_vote_threshold <= matched.scorer_count:
        raise ValueError("chosen vote threshold is malformed")


def _compose_taska_pair_layouts(
    matched: TaskaSeamMatchResult,
    focal_scores: TaskaFocalScoreBatch,
    resources: TaskaPairPipelineResources,
    *,
    grid: int = GRID_SIZE,
) -> TaskaPairPipelineResult:
    """Compose precomputed target-free matcher/focal evidence reproducibly."""

    _validate_fixed_match(matched)
    if resources.artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("resource provenance differs from the production manifest")
    if focal_scores.mode != FOCAL_MODE:
        raise ValueError("focal features must use the checkpoint-exact top-5 contract")
    if focal_scores.edges != matched.candidate_edges:
        raise ValueError("focal scores and matcher harvest are not edge-aligned")
    weights, votes = _edge_evidence(matched)
    features = extract_taska_edge_features(
        matched.cost_right,
        matched.cost_down,
        matched.right_log,
        matched.down_log,
        matched.candidate_edges,
        weights,
        votes,
        grid=grid,
    )
    logistic_priorities = resources.logistic_calibrator.predict_priorities(features.values)
    nonlinear_priorities = resources.nonlinear_calibrator.predict_priorities(features.values)

    raw = solve_raw_tail_global(
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        border_unary=None,
        grid=grid,
        config=SOLVER_CONFIG,
    )
    logistic = solve_prioritized_raw_tail_global(
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        logistic_priorities,
        border_unary=None,
        grid=grid,
        config=SOLVER_CONFIG,
    )
    focal = solve_prioritized_raw_tail_global(
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        focal_scores.logits,
        border_unary=None,
        grid=grid,
        config=SOLVER_CONFIG,
    )
    nonlinear = solve_prioritized_raw_tail_global(
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        nonlinear_priorities,
        border_unary=None,
        grid=grid,
        config=SOLVER_CONFIG,
    )
    solvers: tuple[
        Any,
        PrioritizedRawTailResult,
        PrioritizedRawTailResult,
        PrioritizedRawTailResult,
    ]
    solvers = (raw, logistic, focal, nonlinear)
    layouts = {name: solver.layout for name, solver in zip(ARM_NAMES, solvers, strict=True)}
    selection = select_lowest_taska_seam_cost_layout(
        layouts,
        matched.cost_right,
        matched.cost_down,
        grid=grid,
    )
    tail = polish_unprotected_taska_tail(
        selection.layout,
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        grid=grid,
        max_swaps=TAIL_MAX_SWAPS,
        minimum_gain=TAIL_MINIMUM_GAIN,
    )
    cost_by_name = dict(selection.total_costs)
    arms = tuple(
        TaskaPairArmDiagnostics(
            name=name,
            layout_sha256=_layout_sha256(solver.layout),
            total_cost=cost_by_name[name],
            solver=solver.diagnostics,
        )
        for name, solver in zip(ARM_NAMES, solvers, strict=True)
    )
    return TaskaPairPipelineResult(
        layout=tail.layout,
        choice=selection.choice,
        costs=selection.total_costs,
        diagnostics=TaskaPairPipelineDiagnostics(
            grid_size=grid,
            pair_denominator=2 * grid * (grid - 1),
            candidate_edge_count=len(matched.candidate_edges),
            chosen_vote_threshold=matched.chosen_vote_threshold,
            scorer_count=matched.scorer_count,
            matcher_checkpoint_sha256=matched.checkpoint_sha256,
            artifact_sha256=resources.artifact_sha256,
            focal_mode=FOCAL_MODE,
            tail_max_swaps=TAIL_MAX_SWAPS,
            arms=arms,
            tail=tail.diagnostics,
        ),
    )


def solve_taska_pair_pipeline(
    dirty_tiles: Any,
    resources: TaskaPairPipelineResources | None = None,
    *,
    device: str | torch.device | None = "auto",
    focal_chunk_size: int = 8192,
) -> TaskaPairPipelineResult:
    """Return the fixed legal pair-oriented 576-tile layout for one dirty bag.

    Pass preloaded ``resources`` when solving more than one board.  ``device``
    is used only when resources have not already been loaded.
    """

    if resources is None:
        resources = load_taska_pair_pipeline_resources(device=device)
    matched = match_taska_tiles(
        dirty_tiles,
        resources.matchers,
        config=MATCHER_CONFIG,
        device=resources.device,
        require_verified=True,
    )
    focal_scores = score_focal_edges(
        resources.focal_verifier,
        dirty_tiles,
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        mode=FOCAL_MODE,
        grid=GRID_SIZE,
        device=resources.device,
        chunk_size=focal_chunk_size,
    )
    result = _compose_taska_pair_layouts(matched, focal_scores, resources, grid=GRID_SIZE)
    if result.layout.shape != (TILE_COUNT,):
        raise RuntimeError("production pipeline emitted a non-576 layout")
    return result


def _write_npy_exclusive(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve one 576x20x20x3 uint8 tile bag; emit layout only."
    )
    parser.add_argument("tiles", type=Path, help="input .npy tile bag")
    parser.add_argument("--output-layout", type=Path, required=True)
    parser.add_argument("--diagnostics-json", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--focal-chunk-size", type=int, default=8192)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Layout-only one-board CLI; output files are created exclusively."""

    arguments = parse_args(argv)
    loaded = np.load(arguments.tiles, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        loaded.close()
        raise ValueError("tiles must be one .npy array, not an .npz archive")
    tiles = np.asarray(loaded)
    resources = load_taska_pair_pipeline_resources(device=arguments.device)
    result = solve_taska_pair_pipeline(
        tiles,
        resources,
        focal_chunk_size=arguments.focal_chunk_size,
    )
    _write_npy_exclusive(arguments.output_layout, result.layout)
    if arguments.diagnostics_json is not None:
        _write_json_exclusive(arguments.diagnostics_json, result.as_dict())
    print(json.dumps(result.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the installed CLI.
    raise SystemExit(main())


__all__ = [
    "ARM_NAMES",
    "EXPECTED_ARTIFACT_SHA256",
    "FOCAL_MODE",
    "GRID_SIZE",
    "MATCHER_CONFIG",
    "PAIR_DENOMINATOR",
    "SOLVER_CONFIG",
    "TAIL_MAX_SWAPS",
    "TILE_COUNT",
    "TaskaPairArtifactError",
    "TaskaPairArtifactPaths",
    "TaskaPairArmDiagnostics",
    "TaskaPairPipelineDiagnostics",
    "TaskaPairPipelineResources",
    "TaskaPairPipelineResult",
    "load_taska_pair_pipeline_resources",
    "solve_taska_pair_pipeline",
    "verify_taska_pair_artifacts",
]
