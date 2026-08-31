"""SHA-gated layout-only adapter for the confirmed TASKA pair fusion.

The adapter keeps the existing selective-target500 best-pair pipeline as a
separate fallback.  It runs one target500 matcher pass, reconstructs the same
current350 four layouts, and adds exactly one combined arm containing the
selective accepted edges plus only the unique accepted full-resolution edges.
The denoised 20x20 view is matcher-only.  The returned value is always a strict
permutation of the 576 original upright tile ids; no pixels are rendered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import aiijc_puzzle.raw_tail_global_solver as raw_tail_global_solver
import aiijc_puzzle.taska_fullres_union_voter as fullres_union_voter
import aiijc_puzzle.taska_selective_fullres_fusion as selective_fullres_fusion
import aiijc_puzzle.taska_selective_vote500 as selective_vote500
from aiijc_puzzle.raw_tail_global_solver import solve_raw_tail_global
from aiijc_puzzle.taska_edge_calibrator import (
    extract_taska_edge_features,
    solve_prioritized_raw_tail_global,
)
from aiijc_puzzle.taska_focal_verifier import score_focal_edges
from aiijc_puzzle.taska_pair_pipeline import (
    ARM_NAMES,
    EXPECTED_ARTIFACT_SHA256,
    FOCAL_MODE,
    GRID_SIZE,
    SOLVER_CONFIG,
    TaskaPairPipelineResources,
    load_taska_pair_pipeline_resources,
)
from aiijc_puzzle.taska_seam_matcher import match_taska_tiles
from aiijc_puzzle.taska_vote500 import VOTE500_MATCHER_CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULLRES_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/"
    "fullres_boundary_denoiser.pt"
)

SELECTIVE_SOLVER_SHA256 = "8bb23f6ff6402bfde3a2ec8701ea8ddffff86711fbd71e48993eb6d29a8e1fbc"
FUSION_SOLVER_SHA256 = "13ba0e8f5c09c84dfef8c25711805e334a7afd5f0e9e80db749415f566ed6348"
FULLRES_SUPPLY_SOLVER_SHA256 = (
    "9bf412349380d96ec6f5529a7775870843a2cc99a3251bc0e0cf86b2bb3fbd26"
)
RAW_SOLVER_SHA256 = "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
PARENT_REPORT_SHA256 = "1f9d84c99eae6ba1f03a668163f6e19321e20292e31dd5e51ec00282587517af"
CONFIRMATION_CONFIG_SHA256 = (
    "11b713d0475306d8e1e1397f8563132d74ef5b8957e85e1e58a5e4f57f018190"
)
CONFIRMATION_REPORT_SHA256 = (
    "4d0ea850e101cb56a4f70dc6ff164201c09af047dcb669e3c81e19488661e555"
)

CONFIRMED_FILE_SHA256: tuple[tuple[str, str, str], ...] = (
    (
        "parent_report",
        "outputs/taska-selective-fullres-union-fusion/fixed-v1/report.json",
        PARENT_REPORT_SHA256,
    ),
    (
        "confirmation_config",
        "configs/taska_selective_fullres_union_fusion_fresh32_confirmation_v1.json",
        CONFIRMATION_CONFIG_SHA256,
    ),
    (
        "confirmation_report",
        "outputs/taska-selective-fullres-union-fusion/"
        "fresh32-formal-confirmation-v1/report.json",
        CONFIRMATION_REPORT_SHA256,
    ),
    (
        "fullres_denoiser",
        "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/"
        "fullres_boundary_denoiser.pt",
        fullres_union_voter.FULLRES_DENOISER_SHA256,
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_taska_best_pair_fusion_solver() -> tuple[tuple[str, str], ...]:
    """Fail closed unless every confirmed source and evidence artifact matches."""

    sources = (
        ("selective_solver", Path(selective_vote500.__file__), SELECTIVE_SOLVER_SHA256),
        (
            "fusion_solver",
            Path(selective_fullres_fusion.__file__),
            FUSION_SOLVER_SHA256,
        ),
        (
            "fullres_supply_solver",
            Path(fullres_union_voter.__file__),
            FULLRES_SUPPLY_SOLVER_SHA256,
        ),
        ("raw_solver", Path(raw_tail_global_solver.__file__), RAW_SOLVER_SHA256),
    )
    records: list[tuple[str, str]] = []
    for name, path, expected in sources:
        resolved = path.resolve()
        if not resolved.is_file():
            raise RuntimeError(f"confirmed fusion source is absent: {name}")
        digest = _sha256_file(resolved)
        if digest != expected:
            raise RuntimeError(
                f"confirmed fusion source SHA-256 mismatch for {name}: "
                f"expected {expected}, got {digest}"
            )
        records.append((name, digest))
    for name, relative, expected in CONFIRMED_FILE_SHA256:
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_file():
            raise RuntimeError(f"confirmed fusion artifact is absent: {name}")
        digest = _sha256_file(path)
        if digest != expected:
            raise RuntimeError(
                f"confirmed fusion artifact SHA-256 mismatch for {name}: "
                f"expected {expected}, got {digest}"
            )
        records.append((name, digest))
    return tuple(records)


@dataclass(frozen=True)
class TaskaBestPairFusionResources:
    """SHA-gated pair resources and the fixed matcher-only denoiser."""

    pair: TaskaPairPipelineResources
    denoiser: torch.nn.Module
    confirmed_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.pair.artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
            raise ValueError("pair resources differ from the fixed production manifest")
        expected = verify_taska_best_pair_fusion_solver()
        if self.confirmed_sha256 != expected:
            raise ValueError("confirmed fusion provenance differs from verified files")


def load_taska_best_pair_fusion_resources(
    *, device: str | torch.device = "auto"
) -> TaskaBestPairFusionResources:
    """Verify evidence first, then deserialize only SHA-gated model weights."""

    confirmed = verify_taska_best_pair_fusion_solver()
    pair = load_taska_pair_pipeline_resources(device=device)
    denoiser = fullres_union_voter.load_fullres_denoiser(
        FULLRES_CHECKPOINT,
        device=pair.device,
    )
    return TaskaBestPairFusionResources(
        pair=pair,
        denoiser=denoiser,
        confirmed_sha256=confirmed,
    )


def _edge_evidence(matched: Any) -> tuple[np.ndarray, np.ndarray]:
    records = {record.edge: record for record in matched.vote_records}
    if len(records) != len(matched.vote_records) or set(records) != set(
        matched.candidate_edges
    ):
        raise ValueError("matcher vote records are not uniquely edge-aligned")
    margins = np.asarray(
        [records[edge].minimum_margin for edge in matched.candidate_edges],
        dtype=np.float64,
    )
    votes = np.asarray(
        [records[edge].vote_count for edge in matched.candidate_edges],
        dtype=np.float64,
    )
    return margins, votes


def _same_pass_four_layouts(
    matched: Any,
    focal: Any,
    resources: TaskaPairPipelineResources,
) -> dict[str, np.ndarray]:
    margins, votes = _edge_evidence(matched)
    features = extract_taska_edge_features(
        matched.cost_right,
        matched.cost_down,
        matched.right_log,
        matched.down_log,
        matched.candidate_edges,
        margins,
        votes,
        grid=GRID_SIZE,
    ).values
    priorities = (
        resources.logistic_calibrator.predict_priorities(features),
        focal.logits,
        resources.nonlinear_calibrator.predict_priorities(features),
    )
    raw = solve_raw_tail_global(
        matched.cost_right,
        matched.cost_down,
        matched.candidate_edges,
        grid=GRID_SIZE,
        config=SOLVER_CONFIG,
    )
    prioritized = tuple(
        solve_prioritized_raw_tail_global(
            matched.cost_right,
            matched.cost_down,
            matched.candidate_edges,
            priority,
            grid=GRID_SIZE,
            config=SOLVER_CONFIG,
        )
        for priority in priorities
    )
    return {
        name: selective_fullres_fusion.strict_layout(result.layout, grid=GRID_SIZE)
        for name, result in zip(ARM_NAMES, (raw, *prioritized), strict=True)
    }


@dataclass(frozen=True)
class TaskaBestPairFusionPipelineResult:
    """One strict layout plus target-free selection and SHA provenance."""

    layout: np.ndarray
    selected_arm: str
    costs: tuple[tuple[str, float], ...]
    diagnostics: Mapping[str, Any]
    pair_artifact_sha256: tuple[tuple[str, str], ...]
    confirmed_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        layout = selective_fullres_fusion.strict_layout(self.layout, grid=GRID_SIZE)
        if tuple(name for name, _ in self.costs) != (
            selective_fullres_fusion.FUSION_ARM_NAMES
        ):
            raise ValueError("confirmed fusion six-arm cost roster changed")
        if self.selected_arm not in {name for name, _ in self.costs}:
            raise ValueError("selected arm is outside the fixed six-arm roster")
        if self.pair_artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
            raise ValueError("pair resource provenance differs from the fixed manifest")
        if self.confirmed_sha256 != verify_taska_best_pair_fusion_solver():
            raise ValueError("confirmed fusion provenance differs from verified files")
        layout = layout.copy()
        layout.setflags(write=False)
        object.__setattr__(self, "layout", layout)

    @property
    def layout_sha256(self) -> str:
        values = self.layout.astype("<i4", copy=False).tobytes(order="C")
        return hashlib.sha256(values).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "aiijc-taska-best-pair-fusion-pipeline-result-v1",
            "layout_sha256": self.layout_sha256,
            "selected_arm": self.selected_arm,
            "costs": dict(self.costs),
            "diagnostics": dict(self.diagnostics),
            "pair_artifact_sha256": dict(self.pair_artifact_sha256),
            "confirmed_sha256": dict(self.confirmed_sha256),
            "confirmation_config_sha256": CONFIRMATION_CONFIG_SHA256,
            "confirmation_report_sha256": CONFIRMATION_REPORT_SHA256,
            "layout_only": True,
            "original_upright_tile_permutation": True,
            "restored_pixels_matcher_only": True,
        }


def solve_taska_best_pair_fusion_pipeline(
    dirty_tiles: Any,
    resources: TaskaBestPairFusionResources,
    *,
    focal_chunk_size: int = 8192,
    denoiser_batch_size: int = 576,
) -> TaskaBestPairFusionPipelineResult:
    """Run the fixed confirmed fusion and return only original tile ids."""

    verified = verify_taska_best_pair_fusion_solver()
    if resources.confirmed_sha256 != verified:
        raise ValueError("confirmed fusion resources no longer match disk")
    if resources.pair.artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("pair resources differ from the fixed production manifest")
    if focal_chunk_size <= 0 or denoiser_batch_size <= 0:
        raise ValueError("inference chunk and batch sizes must be positive")

    matched500 = match_taska_tiles(
        dirty_tiles,
        resources.pair.matchers,
        config=VOTE500_MATCHER_CONFIG,
        device=resources.pair.device,
        require_verified=True,
    )
    focal500 = score_focal_edges(
        resources.pair.focal_verifier,
        dirty_tiles,
        matched500.cost_right,
        matched500.cost_down,
        matched500.candidate_edges,
        mode=FOCAL_MODE,
        grid=GRID_SIZE,
        device=resources.pair.device,
        chunk_size=focal_chunk_size,
    )
    selective = selective_vote500.compose_selective_vote500(
        matched500,
        focal500,
        resources.pair,
    )
    matched350, focal350 = selective_vote500.same_pass_target350(
        matched500, focal500
    )
    four = _same_pass_four_layouts(matched350, focal350, resources.pair)
    restored = fullres_union_voter.restore_fixed_matcher_view(
        resources.denoiser,
        dirty_tiles,
        device=resources.pair.device,
        batch_size=denoiser_batch_size,
    )
    scorer_sets = fullres_union_voter.restored_mutual_scorer_sets(
        restored,
        resources.pair.matchers,
        device=resources.pair.device,
    )
    proposed, support = fullres_union_voter.supported_absent_edges(
        selective.supply.current_edges,
        scorer_sets,
    )
    proposed_scores = score_focal_edges(
        resources.pair.focal_verifier,
        dirty_tiles,
        matched350.cost_right,
        matched350.cost_down,
        proposed,
        mode=FOCAL_MODE,
        grid=GRID_SIZE,
        device=resources.pair.device,
        chunk_size=focal_chunk_size,
    )
    accepted, accepted_logits = fullres_union_voter.accept_focal_proposals(
        proposed,
        proposed_scores.logits,
    )
    fusion = selective_fullres_fusion.compose_selective_fullres_fusion(
        cost_right=matched350.cost_right,
        cost_down=matched350.cost_down,
        four_layouts=four,
        frozen_selective_control=selective.candidate_layout,
        current_edges=selective.supply.current_edges,
        current_logits=selective.supply.current_logits,
        selective_new_edges=selective.supply.accepted_new_edges,
        selective_new_logits=selective.supply.accepted_new_logits,
        fullres_accepted_edges=accepted,
        fullres_accepted_logits=accepted_logits,
        grid=GRID_SIZE,
    )
    if not np.array_equal(fusion.control_layout, fusion.mechanical_control_layout):
        raise RuntimeError("mechanical selective control replay mismatch")
    diagnostics = {
        **fusion.diagnostics(),
        "target350_vote_threshold": matched350.chosen_vote_threshold,
        "target500_vote_threshold": matched500.chosen_vote_threshold,
        "target500_candidate_count": len(matched500.candidate_edges),
        "selective_proposed_count": len(selective.supply.proposed_new_edges),
        "fullres_proposed_count": len(proposed),
        "fullres_support_histogram": dict(Counter(int(value) for value in support)),
        "restored_scorer_edge_counts": [len(edges) for edges in scorer_sets],
        "one_target500_matcher_pass": True,
        "standalone_fullres_arm_used": False,
        "mechanical_selective_control_replay_matches": True,
    }
    return TaskaBestPairFusionPipelineResult(
        layout=fusion.candidate_layout,
        selected_arm=fusion.choice,
        costs=fusion.costs,
        diagnostics=diagnostics,
        pair_artifact_sha256=resources.pair.artifact_sha256,
        confirmed_sha256=verified,
    )


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
        description=(
            "Solve one 576x20x20x3 original upright tile bag with the confirmed "
            "selective+unique-fullres pair fusion; emit a layout only."
        )
    )
    parser.add_argument("tiles", type=Path, help="input .npy tile bag")
    parser.add_argument("--output-layout", type=Path, required=True)
    parser.add_argument("--diagnostics-json", type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--focal-chunk-size", type=int, default=8192)
    parser.add_argument("--denoiser-batch-size", type=int, default=576)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Layout-only one-board CLI; output files are created exclusively."""

    arguments = parse_args(argv)
    loaded = np.load(arguments.tiles, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        loaded.close()
        raise ValueError("tiles must be one .npy array, not an .npz archive")
    tiles = np.asarray(loaded)
    resources = load_taska_best_pair_fusion_resources(device=arguments.device)
    result = solve_taska_best_pair_fusion_pipeline(
        tiles,
        resources,
        focal_chunk_size=arguments.focal_chunk_size,
        denoiser_batch_size=arguments.denoiser_batch_size,
    )
    _write_npy_exclusive(arguments.output_layout, result.layout)
    if arguments.diagnostics_json is not None:
        _write_json_exclusive(arguments.diagnostics_json, result.as_dict())
    print(json.dumps(result.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI shim.
    raise SystemExit(main())


__all__ = [
    "CONFIRMATION_CONFIG_SHA256",
    "CONFIRMATION_REPORT_SHA256",
    "FULLRES_SUPPLY_SOLVER_SHA256",
    "FUSION_SOLVER_SHA256",
    "RAW_SOLVER_SHA256",
    "SELECTIVE_SOLVER_SHA256",
    "TaskaBestPairFusionPipelineResult",
    "TaskaBestPairFusionResources",
    "load_taska_best_pair_fusion_resources",
    "main",
    "solve_taska_best_pair_fusion_pipeline",
    "verify_taska_best_pair_fusion_solver",
]
