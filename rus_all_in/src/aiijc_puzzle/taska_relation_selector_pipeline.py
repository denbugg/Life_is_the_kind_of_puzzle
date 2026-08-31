"""SHA-gated layout-only production adapter for the confirmed TASKA selector.

The pipeline runs the already confirmed selective-target500 plus unique-fullres
six-arm parent, independently applies its fixed protected tail to every arm,
and selects one whole layout with the frozen relation-truth classifier.  The
full-resolution restored view is matcher-only.  The only output is a strict
permutation of the 576 original upright tile ids; no image pixels are rendered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import aiijc_puzzle.taska_best_pair_fusion_pipeline as parent_pipeline
import aiijc_puzzle.taska_focal_gated_protected_tail as focal_tail
import aiijc_puzzle.taska_layout_portfolio as layout_portfolio
import aiijc_puzzle.taska_relation_truth_selector as relation_selector
import aiijc_puzzle.taska_six_arm_learned_selector as six_arm_preparer
from aiijc_puzzle.taska_pair_pipeline import (
    EXPECTED_ARTIFACT_SHA256,
    FOCAL_MODE,
    GRID_SIZE,
    TILE_COUNT,
)
from aiijc_puzzle.taska_selective_fullres_fusion import (
    COMBINED_ARM,
    FUSION_ARM_NAMES,
    SELECTIVE_ARM,
)
from aiijc_puzzle.taska_vote500 import VOTE500_MATCHER_CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    PROJECT_ROOT
    / "outputs/taska-relation-truth-selector/fixed-v1/model-local32-held32/"
    "frozen-relation-classifier.pkl"
)
MODEL_SHA256 = "ec4eca99243cdc6be20104d789b9e5d5598b79fa0d1b7e69bc37314375ad8c6b"
RELATION_SELECTOR_SHA256 = (
    "1c91b3f18d2fe08dce59217bbdf446a4638fabf4eec19b91cacd988de8cd48e2"
)
SIX_ARM_PREPARER_SHA256 = (
    "cb37af0e278c7b8143bac65a538b349ade85ad1dd07079d4c13837da469bd1a6"
)
FOCAL_TAIL_SHA256 = "33d64d7202a3b65b925d12c77d10e00429968ab70cfdd4b47a52d738dc1224c1"
LAYOUT_PORTFOLIO_SHA256 = (
    "e8e1064969d1c70483cb62c47af8b3555bc82d77103056317cb39b2d0ee42c0f"
)
PARENT_PIPELINE_SHA256 = (
    "2760708dcca9d9df723886ffb17e202f6f9688a1afa19ba3a5e6f6352556fe0c"
)
DEVELOPMENT_CONFIG_SHA256 = (
    "5fac92c3a2c6c562f18a6e38065d1d2dcc13131a74e29ab5cf079213d1b6bacd"
)
DEVELOPMENT_REPORT_SHA256 = (
    "022739dec8a47465f588a3ad9e45660ffbfa327a6f1647bd1517134c01420c39"
)
CONFIRMATION_CONFIG_SHA256 = (
    "3d903eb595d1c0d152a8b53c7c9fa578b5b012227eeb03ab629a7dd24d5ce4e9"
)
CONFIRMATION_REPORT_SHA256 = (
    "d260872251077e1515251b6c7afc316af25df75045c8119112dff4f36c68ea23"
)

CONFIRMED_SOURCE_SHA256: tuple[tuple[str, Any, str], ...] = (
    ("parent_pipeline", parent_pipeline, PARENT_PIPELINE_SHA256),
    ("relation_selector", relation_selector, RELATION_SELECTOR_SHA256),
    ("six_arm_preparer", six_arm_preparer, SIX_ARM_PREPARER_SHA256),
    ("focal_tail", focal_tail, FOCAL_TAIL_SHA256),
    ("layout_portfolio", layout_portfolio, LAYOUT_PORTFOLIO_SHA256),
)
CONFIRMED_FILE_SHA256: tuple[tuple[str, str, str], ...] = (
    (
        "development_config",
        "configs/taska_relation_truth_selector_v1.json",
        DEVELOPMENT_CONFIG_SHA256,
    ),
    (
        "development_report",
        "outputs/taska-relation-truth-selector/fixed-v1/report.json",
        DEVELOPMENT_REPORT_SHA256,
    ),
    (
        "relation_model",
        "outputs/taska-relation-truth-selector/fixed-v1/model-local32-held32/"
        "frozen-relation-classifier.pkl",
        MODEL_SHA256,
    ),
    (
        "relation_model_freeze",
        "outputs/taska-relation-truth-selector/fixed-v1/model-local32-held32/"
        "pre-evaluation-freeze.json",
        "b8f6a360cdeffb38f80a978aac9305a9e34cb7284af12345c98ba517f586908d",
    ),
    (
        "confirmation_config",
        "configs/taska_relation_truth_selector_confirmation_v1.json",
        CONFIRMATION_CONFIG_SHA256,
    ),
    (
        "confirmation_report",
        "outputs/taska-relation-truth-selector/formal-confirmation-v1/report.json",
        CONFIRMATION_REPORT_SHA256,
    ),
    (
        "confirmation_archive",
        "outputs/taska-relation-truth-selector/formal-confirmation-v1/"
        "frozen-target-free-eval.npz",
        "4cd0346333813cea3576f6db40ea517dcc45fdd5aa81a432a351cf4afdd73131",
    ),
    (
        "confirmation_metadata",
        "outputs/taska-relation-truth-selector/formal-confirmation-v1/"
        "frozen-target-free-eval.json",
        "4ae4b8f27d3d6abef21581b189e84c456768c3935d975cec389952b40fdba64c",
    ),
    (
        "confirmation_pre_score_freeze",
        "outputs/taska-relation-truth-selector/formal-confirmation-v1/"
        "pre-score-freeze.json",
        "97a6d2344669ff6f18ece5085e001de3b6b1f04db89e3154510df42b501757b7",
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_taska_relation_selector_solver() -> tuple[tuple[str, str], ...]:
    """Fail closed unless parent, model, configs, reports and sources match."""

    records = [
        (f"parent_{name}", digest)
        for name, digest in parent_pipeline.verify_taska_best_pair_fusion_solver()
    ]
    for name, module, expected in CONFIRMED_SOURCE_SHA256:
        path = Path(module.__file__).resolve()
        if not path.is_file():
            raise RuntimeError(f"confirmed selector source is absent: {name}")
        digest = _sha256_file(path)
        if digest != expected:
            raise RuntimeError(
                f"confirmed selector source SHA-256 mismatch for {name}: "
                f"expected {expected}, got {digest}"
            )
        records.append((name, digest))
    for name, relative, expected in CONFIRMED_FILE_SHA256:
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_file():
            raise RuntimeError(f"confirmed selector artifact is absent: {name}")
        digest = _sha256_file(path)
        if digest != expected:
            raise RuntimeError(
                f"confirmed selector artifact SHA-256 mismatch for {name}: "
                f"expected {expected}, got {digest}"
            )
        records.append((name, digest))
    return tuple(records)


@dataclass(frozen=True)
class TaskaRelationSelectorResources:
    """SHA-gated six-arm parent resources and frozen relation classifier."""

    parent: parent_pipeline.TaskaBestPairFusionResources
    relation_model: Any
    confirmed_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.parent.pair.artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
            raise ValueError("pair resources differ from the fixed production manifest")
        if self.confirmed_sha256 != verify_taska_relation_selector_solver():
            raise ValueError("relation-selector provenance differs from verified files")
        parameters = self.relation_model.get_params()
        for name, expected in relation_selector.MODEL_PARAMETERS.items():
            if parameters.get(name) != expected:
                raise ValueError(f"frozen relation model parameter changed: {name}")


def load_taska_relation_selector_resources(
    *, device: str | torch.device = "auto"
) -> TaskaRelationSelectorResources:
    """Verify every byte before deserializing models and return fixed resources."""

    confirmed = verify_taska_relation_selector_solver()
    parent_resources = parent_pipeline.load_taska_best_pair_fusion_resources(
        device=device
    )
    with MODEL_PATH.open("rb") as stream:
        relation_model = pickle.load(stream)
    return TaskaRelationSelectorResources(
        parent=parent_resources,
        relation_model=relation_model,
        confirmed_sha256=confirmed,
    )


@dataclass(frozen=True)
class TaskaRelationSelectorPipelineResult:
    """One strict selected layout and complete target-free provenance."""

    layout: np.ndarray
    selected_arm: str
    control_arm: str
    expected_correct_scores: tuple[tuple[str, float], ...]
    parent_costs: tuple[tuple[str, float], ...]
    diagnostics: Mapping[str, Any]
    pair_artifact_sha256: tuple[tuple[str, str], ...]
    confirmed_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        layout = relation_selector.strict_layout(self.layout, grid=GRID_SIZE)
        expected_roster = tuple(FUSION_ARM_NAMES)
        if tuple(name for name, _ in self.expected_correct_scores) != expected_roster:
            raise ValueError("relation-score six-arm roster changed")
        if tuple(name for name, _ in self.parent_costs) != expected_roster:
            raise ValueError("parent-cost six-arm roster changed")
        if self.selected_arm not in expected_roster or self.control_arm not in expected_roster:
            raise ValueError("selected/control arm is outside the fixed six-arm roster")
        if self.pair_artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
            raise ValueError("pair resource provenance differs from the fixed manifest")
        if self.confirmed_sha256 != verify_taska_relation_selector_solver():
            raise ValueError("relation-selector provenance differs from verified files")
        layout = layout.copy()
        layout.setflags(write=False)
        object.__setattr__(self, "layout", layout)

    @property
    def layout_sha256(self) -> str:
        values = self.layout.astype("<i4", copy=False).tobytes(order="C")
        return hashlib.sha256(values).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "aiijc-taska-relation-selector-pipeline-result-v1",
            "layout_sha256": self.layout_sha256,
            "selected_arm": self.selected_arm,
            "control_arm": self.control_arm,
            "changed_from_control": self.selected_arm != self.control_arm,
            "expected_correct_scores": dict(self.expected_correct_scores),
            "parent_costs": dict(self.parent_costs),
            "diagnostics": dict(self.diagnostics),
            "pair_artifact_sha256": dict(self.pair_artifact_sha256),
            "confirmed_sha256": dict(self.confirmed_sha256),
            "development_config_sha256": DEVELOPMENT_CONFIG_SHA256,
            "development_report_sha256": DEVELOPMENT_REPORT_SHA256,
            "confirmation_config_sha256": CONFIRMATION_CONFIG_SHA256,
            "confirmation_report_sha256": CONFIRMATION_REPORT_SHA256,
            "relation_model_sha256": MODEL_SHA256,
            "layout_only": True,
            "original_upright_tile_permutation": True,
            "restored_pixels_matcher_only": True,
            "denoised_output_pixels": False,
        }


def _strict_dirty_tiles(value: Any) -> np.ndarray:
    tiles = np.asarray(value)
    if tiles.shape != (TILE_COUNT, 20, 20, 3) or tiles.dtype != np.uint8:
        raise ValueError("tiles must be the original uint8 array with shape 576x20x20x3")
    return np.ascontiguousarray(tiles)


def solve_taska_relation_selector_pipeline(
    dirty_tiles: Any,
    resources: TaskaRelationSelectorResources,
    *,
    focal_chunk_size: int = 8192,
    denoiser_batch_size: int = 576,
) -> TaskaRelationSelectorPipelineResult:
    """Run the frozen parent and select one whole post-tail layout."""

    verified = verify_taska_relation_selector_solver()
    if resources.confirmed_sha256 != verified:
        raise ValueError("relation-selector resources no longer match disk")
    if resources.parent.pair.artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("pair resources differ from the fixed production manifest")
    if focal_chunk_size <= 0 or denoiser_batch_size <= 0:
        raise ValueError("inference chunk and batch sizes must be positive")
    tiles = _strict_dirty_tiles(dirty_tiles)
    pair = resources.parent.pair
    matched500 = parent_pipeline.match_taska_tiles(
        tiles,
        pair.matchers,
        config=VOTE500_MATCHER_CONFIG,
        device=pair.device,
        require_verified=True,
    )
    focal500 = parent_pipeline.score_focal_edges(
        pair.focal_verifier,
        tiles,
        matched500.cost_right,
        matched500.cost_down,
        matched500.candidate_edges,
        mode=FOCAL_MODE,
        grid=GRID_SIZE,
        device=pair.device,
        chunk_size=focal_chunk_size,
    )
    selective = parent_pipeline.selective_vote500.compose_selective_vote500(
        matched500, focal500, pair
    )
    matched350, focal350 = parent_pipeline.selective_vote500.same_pass_target350(
        matched500, focal500
    )
    four = parent_pipeline._same_pass_four_layouts(matched350, focal350, pair)
    restored = parent_pipeline.fullres_union_voter.restore_fixed_matcher_view(
        resources.parent.denoiser,
        tiles,
        device=pair.device,
        batch_size=denoiser_batch_size,
    )
    scorer_sets = parent_pipeline.fullres_union_voter.restored_mutual_scorer_sets(
        restored, pair.matchers, device=pair.device
    )
    proposed, support = parent_pipeline.fullres_union_voter.supported_absent_edges(
        selective.supply.current_edges, scorer_sets
    )
    proposed_scores = parent_pipeline.score_focal_edges(
        pair.focal_verifier,
        tiles,
        matched350.cost_right,
        matched350.cost_down,
        proposed,
        mode=FOCAL_MODE,
        grid=GRID_SIZE,
        device=pair.device,
        chunk_size=focal_chunk_size,
    )
    accepted, accepted_logits = (
        parent_pipeline.fullres_union_voter.accept_focal_proposals(
            proposed, proposed_scores.logits
        )
    )
    fusion = parent_pipeline.selective_fullres_fusion.compose_selective_fullres_fusion(
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
    pre_tail = {
        **four,
        SELECTIVE_ARM: fusion.selective_union_layout,
        COMBINED_ARM: fusion.combined_union_layout,
    }
    arm_edges = {
        arm: (
            fusion.supply.selective_union_edges
            if arm == SELECTIVE_ARM
            else fusion.supply.combined_union_edges
            if arm == COMBINED_ARM
            else fusion.supply.current_edges
        )
        for arm in FUSION_ARM_NAMES
    }
    arm_logits = {
        arm: (
            fusion.supply.selective_union_logits
            if arm == SELECTIVE_ARM
            else fusion.supply.combined_union_logits
            if arm == COMBINED_ARM
            else fusion.supply.current_logits
        )
        for arm in FUSION_ARM_NAMES
    }
    six_arm = six_arm_preparer.prepare_six_arm_target_free_board(
        pre_tail_layouts=pre_tail,
        cost_right=matched350.cost_right,
        cost_down=matched350.cost_down,
        arm_edges=arm_edges,
        arm_logits=arm_logits,
        control_choice=fusion.choice,
        frozen_control_layout=fusion.candidate_layout,
    )
    post_tail = dict(zip(FUSION_ARM_NAMES, six_arm.layouts, strict=True))
    relations = relation_selector.relation_feature_board(
        post_tail_layouts=post_tail,
        pre_tail_layouts=pre_tail,
        cost_right=matched350.cost_right,
        cost_down=matched350.cost_down,
        arm_edges=arm_edges,
        arm_logits=arm_logits,
        provenance={
            relation_selector.PROVENANCE_NAMES[0]: fusion.supply.current_edges,
            relation_selector.PROVENANCE_NAMES[1]: fusion.supply.selective_new_edges,
            relation_selector.PROVENANCE_NAMES[2]: fusion.supply.unique_fullres_edges,
        },
        control_choice=fusion.choice,
    )
    selected_arm, layout, expected_scores = (
        relation_selector.select_relation_truth_layout(
            relations, resources.relation_model
        )
    )
    diagnostics = {
        "changed_from_control": selected_arm != fusion.choice,
        "parent": {
            **fusion.diagnostics(),
            "target350_vote_threshold": matched350.chosen_vote_threshold,
            "target500_vote_threshold": matched500.chosen_vote_threshold,
            "target500_candidate_count": len(matched500.candidate_edges),
            "selective_proposed_count": len(selective.supply.proposed_new_edges),
            "fullres_proposed_count": len(proposed),
            "fullres_support_histogram": dict(
                Counter(int(value) for value in support)
            ),
            "restored_scorer_edge_counts": [len(edges) for edges in scorer_sets],
            "one_target500_matcher_pass": True,
            "standalone_fullres_arm_used": False,
            "mechanical_selective_control_replay_matches": True,
        },
        "independent_post_tail_arms": list(six_arm.diagnostics),
        "relation_rows_per_arm": 2 * GRID_SIZE * (GRID_SIZE - 1),
        "relation_feature_count": len(relation_selector.FEATURE_NAMES),
    }
    return TaskaRelationSelectorPipelineResult(
        layout=layout,
        selected_arm=selected_arm,
        control_arm=fusion.choice,
        expected_correct_scores=tuple(
            zip(FUSION_ARM_NAMES, expected_scores.tolist(), strict=True)
        ),
        parent_costs=fusion.costs,
        diagnostics=diagnostics,
        pair_artifact_sha256=pair.artifact_sha256,
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
            "Solve one original 576x20x20x3 upright tile bag with the formally "
            "confirmed relation selector; emit a layout only."
        )
    )
    parser.add_argument("tiles", type=Path, help="input uint8 .npy tile bag")
    parser.add_argument("--output-layout", type=Path, required=True)
    parser.add_argument("--diagnostics-json", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
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
    tiles = _strict_dirty_tiles(loaded)
    resources = load_taska_relation_selector_resources(device=arguments.device)
    result = solve_taska_relation_selector_pipeline(
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
    "DEVELOPMENT_CONFIG_SHA256",
    "DEVELOPMENT_REPORT_SHA256",
    "MODEL_SHA256",
    "TaskaRelationSelectorPipelineResult",
    "TaskaRelationSelectorResources",
    "load_taska_relation_selector_resources",
    "main",
    "solve_taska_relation_selector_pipeline",
    "verify_taska_relation_selector_solver",
]
