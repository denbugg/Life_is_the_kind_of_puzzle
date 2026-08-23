"""E24 contextual relation selector over the frozen E23 candidate inventory.

CRS-v1 is deliberately split into a label-free production core and an
explicitly supervised LambdaRank fit helper.  The feature extractor accepts
only the exact :class:`e23_i21_residual_candidate_oracle.CandidatePoolResult`
and label-free arrays that are available at inference time:

* the frozen Rank96 candidate ids and raw directional logits;
* the frozen I21 all-pairs directional logits;
* the corrupted, upright 20x20 RGB tiles.

It cannot accept a permutation, clean target, recovered board, image name, or
source id.  Tile/component/relation ids are retained only as graph/index
metadata and are never columns in ``features``.

Every geometry-valid E23 offset is scored; a synthetic ``NONE`` row is added
to each canonical component-pair query.  At inference, the best offset for a
pair must beat ``NONE`` by a *strictly positive* margin.  Pair winners are
then processed in deterministic order by a rollback-safe signed-potential
DSU.  The number of attempted winners is capped at ``2 * (C - 1)`` for ``C``
E23 components.  Tiles are never rotated or reflected.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

# Normalize transitive-import bytecode before importing NumPy/project modules.
# The process launcher must set the same location before importing this module
# if it also wants this module's own bytecode redirected.
_E24_ARTIFACT_ROOT = Path("E:/pazzle_work/posegraph_e24_selector")
_E24_PYCACHE_ROOT = _E24_ARTIFACT_ROOT / "pycache"
if os.name == "nt" and (
    sys.pycache_prefix is None
    or Path(sys.pycache_prefix).drive.upper() != "E:"
):
    sys.pycache_prefix = str(_E24_PYCACHE_ROOT)

import numpy as np

import e22_rcce4_candidate_oracle as e22
import e23_i21_residual_candidate_oracle as e23
from rank96_lab_selector import scaled_lab_tiles


SCHEMA_VERSION = 1
FEATURE_SCHEMA = "pazzle-e24-context-relation-features-v1"
PROTOCOL_NAME = "e24_crs_v1"
GRID = e23.GRID
NUM_TILES = e23.NUM_TILES
NUM_DIRECTIONS = e23.NUM_DIRECTIONS
RAW_WIDTH = e23.CANDIDATE_WIDTH
SPATIAL_K = e23.SPATIAL_K
SEAM_WIDTHS = (1, 2, 4)
PAIR_CONTEXT_SUMMARY_K = 4
INCIDENT_CONTEXT_PAIR_K = 32
ATTEMPT_MULTIPLIER = 2
NONE_RELATION_ID = -1
NONE_HYPOTHESIS_ID = -1
ROW_OFFSET = np.uint8(0)
ROW_NONE = np.uint8(1)
EPSILON = 1.0e-6


LIGHTGBM_CONFIG: Mapping[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [1],
    "label_gain": [0, 1],
    "n_estimators": 256,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 200,
    "max_bin": 255,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "lambda_l2": 1.0,
    "lambda_l1": 0.0,
    "lambdarank_truncation_level": 30,
    "lambdarank_norm": True,
    "verbosity": -1,
    "n_jobs": 8,
    "deterministic": True,
    "force_col_wise": True,
}


PROTOCOL: Mapping[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "name": PROTOCOL_NAME,
    "orientation": "upright_fixed_no_rotation_or_reflection",
    "candidate_boundary": "all_geometry_valid_e23_hypotheses_no_truncation",
    "query": "canonical_component_pair_u_v",
    "query_rows": "all_exact_offsets_then_one_synthetic_none",
    "context_prior": (
        "e0=max_over_claims(mean(i21_forward_percentile,"
        "i21_reverse_percentile))"
    ),
    "two_hop_context": {
        "offset_summary_k": PAIR_CONTEXT_SUMMARY_K,
        "incident_pair_winners_per_endpoint": INCIDENT_CONTEXT_PAIR_K,
        "context_only_truncation": (
            "top4_offsets_per_pair_then_top32_pair_winners_per_endpoint_"
            "e0_desc_canonical_ties"
        ),
        "scored_rows_truncated": False,
    },
    "model": "fixed_lightgbm_lambdarank",
    "model_config": dict(LIGHTGBM_CONFIG),
    "selection": {
        "pair_winner": "score_desc_then_dr_dc_asc",
        "margin": "best_offset_score_minus_none_score_strictly_gt_zero",
        "global_order": "margin_desc_then_u_v_dr_dc_asc",
        "attempt_cap": "min(selected_pair_count,2*(component_count-1))",
    },
    "decoder": "rollback_safe_signed_component_potential_dsu",
    "forbidden_features": (
        "tile_id",
        "component_id",
        "relation_id",
        "hypothesis_id",
        "image_id",
        "source_group",
        "permutation",
        "purity",
        "clean_target",
        "board",
        "metric",
    ),
}


_BASE_FEATURE_NAMES = (
    "is_none",
    "has_offset",
    "claim_missing",
    "raw_missing_all",
    "size_min_log1p",
    "size_max_log1p",
    "size_absdiff_log1p",
    "size_ratio",
    "height_min_scaled",
    "height_max_scaled",
    "height_absdiff_scaled",
    "width_min_scaled",
    "width_max_scaled",
    "width_absdiff_scaled",
    "area_min_scaled",
    "area_max_scaled",
    "area_absdiff_scaled",
    "density_min",
    "density_max",
    "density_absdiff",
    "both_singleton",
    "one_singleton",
    "dr_scaled",
    "dc_scaled",
    "abs_dr_scaled",
    "abs_dc_scaled",
    "offset_l1_scaled",
    "offset_linf_scaled",
    "merged_height_scaled",
    "merged_width_scaled",
    "merged_area_scaled",
    "merged_density",
    "span_row_slack_scaled",
    "span_col_slack_scaled",
    "claim_count_log1p",
    "unique_pair_count_log1p",
    "unique_endpoint_tile_count_log1p",
    "base_claim_count_log1p",
    "residual_claim_count_log1p",
    "base_claim_fraction",
    "spatial_claim_fraction",
    "horizontal_claim_fraction",
    "vertical_claim_fraction",
    "reciprocal_base_fraction",
    "projected_contact_count_log1p",
    "projected_contact_length_scaled",
    "projected_supporting_contact_count_log1p",
    "projected_support_fraction",
    "incidental_contact_count_log1p",
    "incidental_contact_fraction",
    "raw_observation_count_log1p",
    "raw_forward_present_fraction",
    "raw_reverse_present_fraction",
    "raw_reciprocal_claim_fraction",
    "raw_forward_percentile_mean",
    "raw_reverse_percentile_mean",
    "raw_forward_reverse_abs_z_mean",
    "residual_nomination_missing_fraction",
    "residual_correct_forward_nomination_fraction",
    "residual_correct_reverse_nomination_fraction",
    "residual_wrong_direction_nomination_fraction",
    "residual_reciprocal_nomination_fraction",
    "residual_nomination_up_fraction",
    "residual_nomination_down_fraction",
    "residual_nomination_left_fraction",
    "residual_nomination_right_fraction",
    "spatial_forward_percentile_mean",
    "spatial_reverse_percentile_mean",
    "spatial_forward_reverse_abs_z_mean",
    "spatial_literal_selected_fraction",
    "spatial_nomination_mean",
    "spatial_nomination_max",
    "spatial_nomination_logsumexp",
)


def _aggregate_feature_names(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}_{suffix}" for suffix in ("min", "mean", "max", "logmeanexp"))


_SCORE_FEATURE_NAMES = tuple(
    name
    for prefix in (
        "raw_robust_z",
        "raw_percentile",
        "raw_top1_gap",
        "raw_valid_row_size",
        "spatial_robust_z",
        "spatial_percentile",
        "spatial_top1_gap",
        "spatial_wrong_robust_z",
        "spatial_wrong_percentile",
        "spatial_correct_minus_wrong_robust_z",
        "spatial_correct_minus_wrong_percentile",
        "residual_best_rank_percentile",
    )
    for name in _aggregate_feature_names(prefix)
)


_SEAM_FEATURE_NAMES = tuple(
    f"seam_w{width}_{metric}_{summary}"
    for width in SEAM_WIDTHS
    for metric in (
        "rgb_mse",
        "gradient_mse",
        "tangential_gradient_mse",
        "ncc",
        "lab_mse",
    )
    for summary in ("min", "mean", "max")
)


_INCIDENTAL_FEATURE_NAMES = (
    "incidental_evidence_missing",
    *tuple(
        f"incidental_{metric}_{summary}"
        for metric in ("spatial_e0", "seam_w1_rgb_mse", "seam_w1_lab_mse", "seam_w1_ncc")
        for summary in ("min", "mean", "max")
    ),
)


_CONTEXT_FEATURE_NAMES = (
    "query_offset_count_log1p",
    "query_claim_count_log1p",
    "query_support_mass_log1p",
    "query_best_e0",
    "query_second_e0",
    "query_e0_margin",
    "query_e0_entropy",
    "query_no_alternative",
    "query_top4_e0_min",
    "query_top4_e0_mean",
    "query_top4_e0_max",
    "offset_e0",
    "offset_e0_rank_percentile",
    "offset_e0_gap_best",
    "offset_e0_margin_best_other",
    "offset_e0_robust_z",
    "incident_degree_min_scaled",
    "incident_degree_max_scaled",
    "incident_degree_absdiff_scaled",
    "incident_e0_min",
    "incident_e0_mean_min",
    "incident_e0_mean_max",
    "incident_e0_mean_absdiff",
    "incident_e0_max",
    "incident_e0_absdiff",
    "incident_pair_count_min_log1p",
    "incident_pair_count_max_log1p",
    "incident_pair_count_absdiff_log1p",
    "incident_hypothesis_count_min_log1p",
    "incident_hypothesis_count_max_log1p",
    "incident_hypothesis_count_absdiff_log1p",
    "shared_intermediates_log1p",
    "twohop_path_count_log1p",
    "twohop_exact_count_log1p",
    "twohop_exact_intermediates_log1p",
    "twohop_conflict_count_log1p",
    "twohop_conflict_intermediates_log1p",
    "twohop_exact_support_sum",
    "twohop_exact_support_mean",
    "twohop_exact_support_max",
    "twohop_conflict_support_sum",
    "twohop_conflict_support_mean",
    "twohop_conflict_support_max",
    "twohop_best_match_minus_conflict",
    "twohop_zero_sum_witness_count_log1p",
    "twohop_zero_sum_witness_support_sum",
    "twohop_zero_sum_witness_support_max",
    "twohop_min_l1_scaled",
)


FEATURE_NAMES: tuple[str, ...] = (
    *_BASE_FEATURE_NAMES,
    *_SCORE_FEATURE_NAMES,
    *_SEAM_FEATURE_NAMES,
    *_INCIDENTAL_FEATURE_NAMES,
    *_CONTEXT_FEATURE_NAMES,
)
FEATURE_INDEX: Mapping[str, int] = {name: index for index, name in enumerate(FEATURE_NAMES)}

if len(FEATURE_NAMES) != len(set(FEATURE_NAMES)):
    raise RuntimeError("E24 feature names are not unique")
if any(
    forbidden in name
    for name in FEATURE_NAMES
    for forbidden in PROTOCOL["forbidden_features"]
):
    raise RuntimeError("E24 feature schema contains a forbidden identifier/label feature")


class ContextRelationSelectorError(ValueError):
    """An E24 label-free input, feature, or decoder invariant failed closed."""


class Predictor(Protocol):
    def predict(self, values: np.ndarray) -> Any: ...


@dataclass(frozen=True, slots=True)
class RelationFeatureTable:
    """Canonical all-offset-plus-NONE LambdaRank rows for one or more scenes."""

    features: np.ndarray
    hypothesis_ids: np.ndarray
    relation_ids: np.ndarray
    relations: np.ndarray
    row_kind: np.ndarray
    support: np.ndarray
    query_offsets: np.ndarray
    scene_offsets: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])

    @property
    def queries(self) -> int:
        return int(self.query_offsets.size - 1)

    @property
    def query_sizes(self) -> np.ndarray:
        return np.diff(self.query_offsets)

    @property
    def none_rows(self) -> np.ndarray:
        return self.query_offsets[1:] - 1


@dataclass(frozen=True, slots=True)
class SelectedRelation:
    hypothesis_id: int
    relation_id: int
    u: int
    v: int
    dr: int
    dc: int
    score: float
    none_score: float
    margin: float
    support: int

    @property
    def relation(self) -> tuple[int, int, int, int]:
        return self.u, self.v, self.dr, self.dc


@dataclass(frozen=True, slots=True)
class DSUOutcome:
    selection: SelectedRelation
    accepted: bool
    reason: str
    tree_merge: bool
    cycle: bool


@dataclass(frozen=True, slots=True)
class RelationDecodeResult:
    selected: tuple[SelectedRelation, ...]
    attempted: tuple[SelectedRelation, ...]
    outcomes: tuple[DSUOutcome, ...]
    components: tuple[dict[int, tuple[int, int]], ...]
    attempt_cap: int
    tree_merges: int
    cycle_acceptances: int
    rejection_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _ComponentGeometry:
    size: int
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    height: int
    width: int
    area: int
    density: float


@dataclass(frozen=True, slots=True)
class _DirectionalStatistics:
    robust_z: np.ndarray
    percentile: np.ndarray
    top1_gap: np.ndarray


@dataclass(frozen=True, slots=True)
class _SpatialNomination:
    direction: int
    source: int
    target: int
    rank: int


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value)
    array.setflags(write=False)
    return array


def _finite_scalar(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContextRelationSelectorError(f"{label} is not numeric") from exc
    if not np.isfinite(result):
        raise ContextRelationSelectorError(f"{label} must be finite")
    return result


def _validate_table(table: RelationFeatureTable) -> RelationFeatureTable:
    if type(table) is not RelationFeatureTable:
        raise ContextRelationSelectorError("table must be an exact RelationFeatureTable")
    features = table.features
    rows = int(features.shape[0]) if isinstance(features, np.ndarray) and features.ndim == 2 else -1
    if (
        not isinstance(features, np.ndarray)
        or features.dtype != np.float32
        or features.ndim != 2
        or features.shape[1] != len(FEATURE_NAMES)
        or not features.flags.c_contiguous
        or not np.isfinite(features).all()
    ):
        raise ContextRelationSelectorError("feature matrix contract failed")
    specs = (
        (table.hypothesis_ids, np.int64, (rows,), "hypothesis_ids"),
        (table.relation_ids, np.int64, (rows,), "relation_ids"),
        (table.relations, np.int64, (rows, 4), "relations"),
        (table.row_kind, np.uint8, (rows,), "row_kind"),
        (table.support, np.int64, (rows,), "support"),
    )
    for value, dtype, shape, label in specs:
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != dtype
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ContextRelationSelectorError(f"{label} contract failed")
    offsets = table.query_offsets
    if (
        not isinstance(offsets, np.ndarray)
        or offsets.dtype != np.int64
        or offsets.ndim != 1
        or offsets.size < 2
        or not offsets.flags.c_contiguous
        or int(offsets[0]) != 0
        or int(offsets[-1]) != rows
        or bool((np.diff(offsets) < 2).any())
    ):
        raise ContextRelationSelectorError("query_offsets contract failed")
    scene_offsets = table.scene_offsets
    if (
        not isinstance(scene_offsets, np.ndarray)
        or scene_offsets.dtype != np.int64
        or scene_offsets.ndim != 1
        or scene_offsets.size < 2
        or not scene_offsets.flags.c_contiguous
        or int(scene_offsets[0]) != 0
        or int(scene_offsets[-1]) != rows
        or bool((np.diff(scene_offsets) <= 0).any())
        or not set(scene_offsets.tolist()).issubset(set(offsets.tolist()))
    ):
        raise ContextRelationSelectorError("scene_offsets contract failed")
    if bool(((table.row_kind != ROW_OFFSET) & (table.row_kind != ROW_NONE)).any()):
        raise ContextRelationSelectorError("unknown row kind")
    for start, stop in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        start_i, stop_i = int(start), int(stop)
        if table.row_kind[stop_i - 1] != ROW_NONE or bool(
            (table.row_kind[start_i : stop_i - 1] != ROW_OFFSET).any()
        ):
            raise ContextRelationSelectorError("each query must end in exactly one NONE row")
        pair = table.relations[start_i, :2]
        if not bool(np.all(table.relations[start_i:stop_i, :2] == pair)):
            raise ContextRelationSelectorError("query rows do not share a component pair")
        offsets_relation = [
            tuple(map(int, row))
            for row in table.relations[start_i : stop_i - 1]
        ]
        if offsets_relation != sorted(offsets_relation) or len(offsets_relation) != len(
            set(offsets_relation)
        ):
            raise ContextRelationSelectorError("query offset rows are not canonical")
        if (
            int(table.hypothesis_ids[stop_i - 1]) != NONE_HYPOTHESIS_ID
            or int(table.relation_ids[stop_i - 1]) != NONE_RELATION_ID
            or int(table.support[stop_i - 1]) != 0
        ):
            raise ContextRelationSelectorError("NONE metadata drifted")
    return table


def _component_geometry(component: e23.RigidComponent) -> _ComponentGeometry:
    if not component.entries:
        raise ContextRelationSelectorError("E23 component is empty")
    rows = [int(row) for _tile, row, _col in component.entries]
    cols = [int(col) for _tile, _row, col in component.entries]
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    area = height * width
    return _ComponentGeometry(
        size=len(component.entries),
        min_row=min(rows),
        max_row=max(rows),
        min_col=min(cols),
        max_col=max(cols),
        height=height,
        width=width,
        area=area,
        density=float(len(component.entries) / area),
    )


def _validate_label_free_inputs(
    result: e23.CandidatePoolResult,
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
    spatial_logits: np.ndarray,
    tiles_uint8: np.ndarray,
) -> tuple[np.ndarray, tuple[_ComponentGeometry, ...]]:
    if type(result) is not e23.CandidatePoolResult:
        raise ContextRelationSelectorError(
            "result must be the exact frozen E23 CandidatePoolResult; generic/RawScene objects are forbidden"
        )
    if (
        not isinstance(candidate_ids, np.ndarray)
        or candidate_ids.shape != (NUM_TILES, RAW_WIDTH)
        or candidate_ids.dtype != np.int64
        or not candidate_ids.flags.c_contiguous
    ):
        raise ContextRelationSelectorError("candidate_ids must be contiguous int64[576,128]")
    if (
        not isinstance(raw_logits, np.ndarray)
        or raw_logits.shape != (NUM_DIRECTIONS, NUM_TILES, RAW_WIDTH)
        or raw_logits.dtype != np.float32
        or not raw_logits.flags.c_contiguous
        or bool(np.isnan(raw_logits).any())
        or bool(np.isposinf(raw_logits).any())
    ):
        raise ContextRelationSelectorError("raw_logits must be contiguous float32[4,576,128]")
    finite = np.isfinite(raw_logits)
    if not all(np.array_equal(finite[0], finite[index]) for index in range(1, NUM_DIRECTIONS)):
        raise ContextRelationSelectorError("raw finite mask differs across directions")
    valid = np.ascontiguousarray(finite[0], dtype=np.bool_)
    if not bool(valid.any(axis=1).all()) or not bool(np.isneginf(raw_logits[~finite]).all()):
        raise ContextRelationSelectorError("raw logits contain invalid padding/empty rows")
    valid_ids = candidate_ids[valid]
    if bool(((valid_ids < 0) | (valid_ids >= NUM_TILES)).any()):
        raise ContextRelationSelectorError("valid candidate id lies outside the tile bag")
    if (
        not isinstance(spatial_logits, np.ndarray)
        or spatial_logits.shape != (NUM_DIRECTIONS, NUM_TILES, NUM_TILES)
        or spatial_logits.dtype != np.float32
        or not spatial_logits.flags.c_contiguous
        or not np.isfinite(spatial_logits).all()
    ):
        raise ContextRelationSelectorError("spatial_logits must be contiguous finite float32[4,576,576]")
    if (
        not isinstance(tiles_uint8, np.ndarray)
        or tiles_uint8.shape != (NUM_TILES, 20, 20, 3)
        or tiles_uint8.dtype != np.uint8
        or not tiles_uint8.flags.c_contiguous
    ):
        raise ContextRelationSelectorError("tiles must be contiguous upright uint8[576,20,20,3]")

    components = result.components
    if (
        not isinstance(components, tuple)
        or not components
        or tuple(component.component_id for component in components) != tuple(range(len(components)))
    ):
        raise ContextRelationSelectorError("E23 components are not canonical")
    owner = np.asarray(result.owner)
    local_rows = np.asarray(result.local_rows)
    local_cols = np.asarray(result.local_cols)
    if any(value.shape != (NUM_TILES,) for value in (owner, local_rows, local_cols)):
        raise ContextRelationSelectorError("E23 owner/local-coordinate arrays are malformed")
    seen_tiles: set[int] = set()
    geometries: list[_ComponentGeometry] = []
    for component in components:
        if type(component) is not e23.RigidComponent:
            raise ContextRelationSelectorError("E23 component has the wrong exact type")
        positions: set[tuple[int, int]] = set()
        for tile, row, col in component.entries:
            tile_i, row_i, col_i = int(tile), int(row), int(col)
            if not 0 <= tile_i < NUM_TILES or tile_i in seen_tiles:
                raise ContextRelationSelectorError("E23 component tile partition drifted")
            if (row_i, col_i) in positions:
                raise ContextRelationSelectorError("E23 component has a coordinate collision")
            if (
                int(owner[tile_i]) != component.component_id
                or int(local_rows[tile_i]) != row_i
                or int(local_cols[tile_i]) != col_i
            ):
                raise ContextRelationSelectorError("E23 owner/local-coordinate binding drifted")
            seen_tiles.add(tile_i)
            positions.add((row_i, col_i))
        geometry = _component_geometry(component)
        if geometry.height > GRID or geometry.width > GRID:
            raise ContextRelationSelectorError("E23 component exceeds the 24x24 span")
        geometries.append(geometry)
    if seen_tiles != set(range(NUM_TILES)):
        raise ContextRelationSelectorError("E23 components do not partition all 576 tiles")

    # Bind the supplied raw arrays to every canonical base-pair membership.
    expected_memberships: dict[tuple[int, int], list[int | None]] = {}
    for source in range(NUM_TILES):
        row_ids = candidate_ids[source, valid[source]]
        if np.unique(row_ids).size != row_ids.size or source in row_ids:
            raise ContextRelationSelectorError("raw candidate row repeats/self-selects a tile")
        for slot_value in np.flatnonzero(valid[source]).tolist():
            slot = int(slot_value)
            target = int(candidate_ids[source, slot])
            a, b = (source, target) if source < target else (target, source)
            membership = expected_memberships.setdefault((a, b), [None, None])
            membership[0 if source == a else 1] = slot
    expected_base = tuple(
        (a, b, slots[0], slots[1])
        for (a, b), slots in sorted(expected_memberships.items())
    )
    observed_base = tuple(
        (pair.a, pair.b, pair.a_to_b_slot, pair.b_to_a_slot)
        for pair in result.base_affinity_pairs
    )
    if observed_base != expected_base:
        raise ContextRelationSelectorError("E23 base-pair inventory does not match raw arrays")

    selected = np.asarray(result.spatial_selected_ids)
    if (
        selected.shape != (NUM_DIRECTIONS, NUM_TILES, SPATIAL_K)
        or selected.dtype != np.int64
        or not selected.flags.c_contiguous
    ):
        raise ContextRelationSelectorError("E23 residual selection array is malformed")
    expected_selected, _ = e23._select_spatial_residuals(
        spatial_logits, result.base_affinity_pairs
    )
    if not np.array_equal(selected, expected_selected):
        raise ContextRelationSelectorError("E23 residual selections do not match spatial logits")

    claims = result.claims
    if not isinstance(claims, tuple) or tuple(claim.claim_id for claim in claims) != tuple(
        range(len(claims))
    ):
        raise ContextRelationSelectorError("E23 claim inventory is not canonical")
    for claim in claims:
        if type(claim) is not e23.RCCE4Claim:
            raise ContextRelationSelectorError("E23 claim has the wrong exact type")
        if (
            not 0 <= claim.pair_id < len(result.affinity_pairs)
            or (claim.dy, claim.dx) not in ((0, 1), (1, 0))
            or int(owner[claim.first]) != claim.first_component
            or int(owner[claim.second]) != claim.second_component
            or claim.first_component == claim.second_component
        ):
            raise ContextRelationSelectorError("E23 claim geometry/ownership drifted")
        for observation in claim.observations:
            if type(observation) is not e22.LogitObservation:
                raise ContextRelationSelectorError("claim observation has the wrong exact type")
            if (
                not 0 <= observation.slot < RAW_WIDTH
                or not bool(valid[observation.source, observation.slot])
                or int(candidate_ids[observation.source, observation.slot]) != observation.target
                or float(raw_logits[observation.direction, observation.source, observation.slot])
                != float(observation.logit)
            ):
                raise ContextRelationSelectorError("claim observation is not bound to raw logits")

    hypotheses = result.hypotheses
    if not isinstance(hypotheses, tuple) or tuple(
        hypothesis.hypothesis_id for hypothesis in hypotheses
    ) != tuple(range(len(hypotheses))):
        raise ContextRelationSelectorError("E23 hypothesis IDs are not contiguous")
    previous: tuple[int, int, int, int] | None = None
    for hypothesis in hypotheses:
        if type(hypothesis) is not e23.PoseHypothesis:
            raise ContextRelationSelectorError("E23 hypothesis has the wrong exact type")
        relation = (hypothesis.u, hypothesis.v, hypothesis.dr, hypothesis.dc)
        if (
            not 0 <= hypothesis.u < hypothesis.v < len(components)
            or previous is not None
            and relation <= previous
            or not hypothesis.claim_ids
            or tuple(sorted(hypothesis.claim_ids)) != hypothesis.claim_ids
            or hypothesis.claim_ids[0] < 0
            or hypothesis.claim_ids[-1] >= len(claims)
        ):
            raise ContextRelationSelectorError("E23 hypotheses are not canonical")
        relation_source = result.relation_candidates[hypothesis.relation_id]
        if relation_source.relation != relation or relation_source.claim_ids != hypothesis.claim_ids:
            raise ContextRelationSelectorError("E23 hypothesis/relation binding drifted")
        previous = relation
    if not hypotheses:
        raise ContextRelationSelectorError("E23 returned no geometry-valid hypotheses")
    return valid, tuple(geometries)


def _robust_statistics(
    values: np.ndarray,
    ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 1 or ids.shape != values.shape or not values.size:
        raise ContextRelationSelectorError("directional score row is malformed")
    finite_values = values.astype(np.float64, copy=False)
    median = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - median)))
    scale = 1.4826 * mad + EPSILON
    robust = np.clip((finite_values - median) / scale, -8.0, 8.0)
    order = np.lexsort((ids.astype(np.int64, copy=False), -finite_values))
    rank = np.empty(values.size, dtype=np.int64)
    rank[order] = np.arange(values.size, dtype=np.int64)
    percentile = (
        np.ones(values.size, dtype=np.float64)
        if values.size == 1
        else (values.size - 1 - rank).astype(np.float64) / (values.size - 1)
    )
    maximum = float(finite_values.max())
    gap = finite_values - maximum
    if values.size > 1:
        winner_indices = np.flatnonzero(finite_values == maximum)
        if winner_indices.size == 1:
            winner = int(winner_indices[0])
            best_other = float(
                max(
                    finite_values[:winner].max(initial=-np.inf),
                    finite_values[winner + 1 :].max(initial=-np.inf),
                )
            )
            gap[winner] = maximum - best_other
    return robust.astype(np.float32), percentile.astype(np.float32), gap.astype(np.float32)


def _raw_statistics(
    candidate_ids: np.ndarray, raw_logits: np.ndarray, valid: np.ndarray
) -> _DirectionalStatistics:
    robust = np.zeros_like(raw_logits, dtype=np.float32)
    percentile = np.zeros_like(raw_logits, dtype=np.float32)
    gap = np.zeros_like(raw_logits, dtype=np.float32)
    for direction in range(NUM_DIRECTIONS):
        for source in range(NUM_TILES):
            mask = valid[source]
            one_robust, one_percentile, one_gap = _robust_statistics(
                raw_logits[direction, source, mask], candidate_ids[source, mask]
            )
            robust[direction, source, mask] = one_robust
            percentile[direction, source, mask] = one_percentile
            gap[direction, source, mask] = one_gap
    return _DirectionalStatistics(robust, percentile, gap)


def _spatial_statistics(spatial_logits: np.ndarray) -> _DirectionalStatistics:
    robust = np.zeros_like(spatial_logits, dtype=np.float32)
    percentile = np.zeros_like(spatial_logits, dtype=np.float32)
    gap = np.zeros_like(spatial_logits, dtype=np.float32)
    ids = np.arange(NUM_TILES, dtype=np.int64)
    for direction in range(NUM_DIRECTIONS):
        for source in range(NUM_TILES):
            mask = ids != source
            one_robust, one_percentile, one_gap = _robust_statistics(
                spatial_logits[direction, source, mask], ids[mask]
            )
            robust[direction, source, mask] = one_robust
            percentile[direction, source, mask] = one_percentile
            gap[direction, source, mask] = one_gap
    return _DirectionalStatistics(robust, percentile, gap)


def _logmeanexp(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    maximum = float(array.max())
    return float(maximum + np.log(np.exp(array - maximum).mean()))


def _put_aggregate(row: np.ndarray, prefix: str, values: Sequence[float]) -> None:
    if values:
        array = np.asarray(values, dtype=np.float64)
        result = (float(array.min()), float(array.mean()), float(array.max()), _logmeanexp(values))
    else:
        result = (0.0, 0.0, 0.0, 0.0)
    for suffix, value in zip(("min", "mean", "max", "logmeanexp"), result):
        row[FEATURE_INDEX[f"{prefix}_{suffix}"]] = np.float32(value)


def _put_min_mean_max(row: np.ndarray, prefix: str, values: Sequence[float]) -> None:
    if values:
        array = np.asarray(values, dtype=np.float64)
        result = (float(array.min()), float(array.mean()), float(array.max()))
    else:
        result = (0.0, 0.0, 0.0)
    for suffix, value in zip(("min", "mean", "max"), result):
        row[FEATURE_INDEX[f"{prefix}_{suffix}"]] = np.float32(value)


def _profile_ncc(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt(np.square(left).sum() * np.square(right).sum()))
    return float(np.dot(left, right) / denominator) if denominator > 1.0e-12 else 0.0


def _seam_values_full(
    normalized_rgb: np.ndarray,
    lab: np.ndarray,
    claim: e23.RCCE4Claim,
    width: int,
) -> tuple[float, float, float, float, float]:
    first = int(claim.first)
    second = int(claim.second)
    # Physical upright boundary pixels are index 19 on the first tile and
    # index 0 on the second.  Wider bands walk inward from that exact seam.
    outer = 20 - 1
    first_depth = np.arange(outer, outer - width, -1, dtype=np.int64)
    second_depth = np.arange(0, width, dtype=np.int64)
    if claim.dx == 1:
        a = normalized_rgb[first][:, first_depth, :].transpose(1, 0, 2)
        b = normalized_rgb[second][:, second_depth, :].transpose(1, 0, 2)
        a_lab = lab[first][:, first_depth, :].transpose(1, 0, 2)
        b_lab = lab[second][:, second_depth, :].transpose(1, 0, 2)
        a_inner = normalized_rgb[first][:, first_depth - 1, :].transpose(1, 0, 2)
        b_inner = normalized_rgb[second][:, second_depth + 1, :].transpose(1, 0, 2)
    else:
        a = normalized_rgb[first][first_depth, :, :]
        b = normalized_rgb[second][second_depth, :, :]
        a_lab = lab[first][first_depth, :, :]
        b_lab = lab[second][second_depth, :, :]
        a_inner = normalized_rgb[first][first_depth - 1, :, :]
        b_inner = normalized_rgb[second][second_depth + 1, :, :]
    rgb_mse = float(np.square(a - b).mean(dtype=np.float64))
    # Both derivatives point in the physical first->second direction.
    first_gradient = a - a_inner
    second_gradient = b_inner - b
    gradient_mse = float(np.square(first_gradient - second_gradient).mean(dtype=np.float64))
    first_tangential = np.diff(a, axis=1)
    second_tangential = np.diff(b, axis=1)
    tangential_gradient_mse = float(
        np.square(first_tangential - second_tangential).mean(dtype=np.float64)
    )
    ncc = _profile_ncc(a, b)
    lab_mse = float(np.square(a_lab - b_lab).mean(dtype=np.float64))
    return rgb_mse, gradient_mse, tangential_gradient_mse, ncc, lab_mse


def _seam_values(
    normalized_rgb: np.ndarray,
    lab: np.ndarray,
    claim: e23.RCCE4Claim,
    width: int,
) -> tuple[float, float, float, float]:
    """Compatibility view of the four original seam summaries."""

    rgb_mse, gradient_mse, _tangential, ncc, lab_mse = _seam_values_full(
        normalized_rgb, lab, claim, width
    )
    return rgb_mse, gradient_mse, ncc, lab_mse


def _base_component_features(
    row: np.ndarray,
    u_geometry: _ComponentGeometry,
    v_geometry: _ComponentGeometry,
) -> None:
    size_u = float(u_geometry.size)
    size_v = float(v_geometry.size)
    row[FEATURE_INDEX["size_min_log1p"]] = np.log1p(min(size_u, size_v))
    row[FEATURE_INDEX["size_max_log1p"]] = np.log1p(max(size_u, size_v))
    row[FEATURE_INDEX["size_absdiff_log1p"]] = np.log1p(abs(size_u - size_v))
    row[FEATURE_INDEX["size_ratio"]] = min(size_u, size_v) / max(size_u, size_v)
    for prefix, first, second, scale in (
        ("height", u_geometry.height, v_geometry.height, GRID),
        ("width", u_geometry.width, v_geometry.width, GRID),
        ("area", u_geometry.area, v_geometry.area, GRID * GRID),
    ):
        row[FEATURE_INDEX[f"{prefix}_min_scaled"]] = min(first, second) / scale
        row[FEATURE_INDEX[f"{prefix}_max_scaled"]] = max(first, second) / scale
        row[FEATURE_INDEX[f"{prefix}_absdiff_scaled"]] = abs(first - second) / scale
    row[FEATURE_INDEX["density_min"]] = min(u_geometry.density, v_geometry.density)
    row[FEATURE_INDEX["density_max"]] = max(u_geometry.density, v_geometry.density)
    row[FEATURE_INDEX["density_absdiff"]] = abs(u_geometry.density - v_geometry.density)
    row[FEATURE_INDEX["both_singleton"]] = float(size_u == 1.0 and size_v == 1.0)
    row[FEATURE_INDEX["one_singleton"]] = float((size_u == 1.0) != (size_v == 1.0))


def _merged_geometry_features(
    row: np.ndarray,
    u_geometry: _ComponentGeometry,
    v_geometry: _ComponentGeometry,
    dr: int,
    dc: int,
) -> None:
    minimum_row = min(u_geometry.min_row, v_geometry.min_row + dr)
    maximum_row = max(u_geometry.max_row, v_geometry.max_row + dr)
    minimum_col = min(u_geometry.min_col, v_geometry.min_col + dc)
    maximum_col = max(u_geometry.max_col, v_geometry.max_col + dc)
    height = maximum_row - minimum_row + 1
    width = maximum_col - minimum_col + 1
    area = height * width
    row[FEATURE_INDEX["dr_scaled"]] = dr / GRID
    row[FEATURE_INDEX["dc_scaled"]] = dc / GRID
    row[FEATURE_INDEX["abs_dr_scaled"]] = abs(dr) / GRID
    row[FEATURE_INDEX["abs_dc_scaled"]] = abs(dc) / GRID
    row[FEATURE_INDEX["offset_l1_scaled"]] = (abs(dr) + abs(dc)) / (2 * GRID)
    row[FEATURE_INDEX["offset_linf_scaled"]] = max(abs(dr), abs(dc)) / GRID
    row[FEATURE_INDEX["merged_height_scaled"]] = height / GRID
    row[FEATURE_INDEX["merged_width_scaled"]] = width / GRID
    row[FEATURE_INDEX["merged_area_scaled"]] = area / (GRID * GRID)
    row[FEATURE_INDEX["merged_density"]] = (u_geometry.size + v_geometry.size) / max(1, area)
    row[FEATURE_INDEX["span_row_slack_scaled"]] = (GRID - height) / GRID
    row[FEATURE_INDEX["span_col_slack_scaled"]] = (GRID - width) / GRID


def _oriented_offset(
    relation: tuple[int, int, int, int], source: int, target: int
) -> tuple[int, int]:
    u, v, dr, dc = relation
    if source == u and target == v:
        return dr, dc
    if source == v and target == u:
        return -dr, -dc
    raise ContextRelationSelectorError("requested oriented offset leaves its relation")


def _bounded_context_shortlists(
    relations: np.ndarray,
    e0: np.ndarray,
    group_ranges: Sequence[tuple[int, int]],
    component_count: int,
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], tuple[int, ...]],
    dict[int, dict[int, int]],
    dict[int, int],
]:
    """Return deterministic top-4 offsets and top-32 incident pair winners."""

    best_by_pair: dict[tuple[int, int], int] = {}
    top_by_pair: dict[tuple[int, int], tuple[int, ...]] = {}
    for start, stop in group_ranges:
        ranked = sorted(
            range(int(start), int(stop)),
            key=lambda idx: (
                -float(e0[idx]),
                int(relations[idx, 2]),
                int(relations[idx, 3]),
            ),
        )
        pair = tuple(map(int, relations[int(start), :2]))
        best_by_pair[pair] = ranked[0]
        top_by_pair[pair] = tuple(ranked[:PAIR_CONTEXT_SUMMARY_K])

    incident_candidates: dict[int, list[int]] = {
        component_id: [] for component_id in range(int(component_count))
    }
    for best_index in best_by_pair.values():
        u, v = map(int, relations[best_index, :2])
        incident_candidates[u].append(best_index)
        incident_candidates[v].append(best_index)
    incident: dict[int, dict[int, int]] = {}
    incident_pair_counts: dict[int, int] = {}
    for component_id, indices in incident_candidates.items():
        incident_pair_counts[component_id] = len(indices)
        ranked = sorted(
            indices,
            key=lambda idx: (
                -float(e0[idx]),
                tuple(map(int, relations[idx])),
            ),
        )
        neighbours: dict[int, int] = {}
        for idx in ranked[:INCIDENT_CONTEXT_PAIR_K]:
            u, v = map(int, relations[idx, :2])
            neighbours[v if component_id == u else u] = idx
        incident[component_id] = neighbours
    return best_by_pair, top_by_pair, incident, incident_pair_counts


def _cached_incidental_evidence(
    seam: tuple[int, int, int, int],
    normalized_rgb: np.ndarray,
    lab: np.ndarray,
    spatial_stats: _DirectionalStatistics,
    cache: dict[tuple[int, int, int, int], tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    cached = cache.get(seam)
    if cached is not None:
        return cached
    first, second, dy, dx = seam
    direction = e23.RIGHT if dx == 1 else e23.DOWN
    inverse = e23.LEFT if dx == 1 else e23.UP
    spatial_e0 = 0.5 * (
        float(spatial_stats.percentile[direction, first, second])
        + float(spatial_stats.percentile[inverse, second, first])
    )
    claim = e23.RCCE4Claim(
        -1, -1, first, second, dy, dx, -1, -1, None, None
    )
    rgb_mse, _normal, _tangential, ncc, lab_mse = _seam_values_full(
        normalized_rgb, lab, claim, 1
    )
    evidence = (spatial_e0, rgb_mse, lab_mse, ncc)
    cache[seam] = evidence
    return evidence


def _projected_contact_features(
    row: np.ndarray,
    result: e23.CandidatePoolResult,
    hypothesis: e23.PoseHypothesis,
    normalized_rgb: np.ndarray,
    lab: np.ndarray,
    spatial_stats: _DirectionalStatistics,
    evidence_cache: dict[
        tuple[int, int, int, int], tuple[float, float, float, float]
    ],
) -> None:
    """Summarize all cross-component tile edges after applying the offset."""

    component_u = result.components[int(hypothesis.u)]
    component_v = result.components[int(hypothesis.v)]
    shifted_v = {
        (int(local_row) + int(hypothesis.dr), int(local_col) + int(hypothesis.dc)): int(tile)
        for tile, local_row, local_col in component_v.entries
    }
    contacts: set[tuple[int, int, int, int]] = set()
    for tile_u, local_row, local_col in component_u.entries:
        tile_u_i = int(tile_u)
        row_u, col_u = int(local_row), int(local_col)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            tile_v = shifted_v.get((row_u + dy, col_u + dx))
            if tile_v is None:
                continue
            if (dy, dx) == (0, 1):
                seam = (tile_u_i, tile_v, 0, 1)
            elif (dy, dx) == (1, 0):
                seam = (tile_u_i, tile_v, 1, 0)
            elif (dy, dx) == (0, -1):
                seam = (tile_v, tile_u_i, 0, 1)
            else:
                seam = (tile_v, tile_u_i, 1, 0)
            contacts.add(seam)
    supporting = {
        result.claims[int(claim_id)].physical_seam for claim_id in hypothesis.claim_ids
    }
    supporting_projected = len(contacts.intersection(supporting))
    incidental_seams = sorted(contacts.difference(supporting))
    incidental = len(incidental_seams)
    row[FEATURE_INDEX["projected_contact_count_log1p"]] = np.log1p(len(contacts))
    row[FEATURE_INDEX["projected_contact_length_scaled"]] = len(contacts) / GRID
    row[FEATURE_INDEX["projected_supporting_contact_count_log1p"]] = np.log1p(
        supporting_projected
    )
    row[FEATURE_INDEX["projected_support_fraction"]] = supporting_projected / max(
        1, len(contacts)
    )
    row[FEATURE_INDEX["incidental_contact_count_log1p"]] = np.log1p(incidental)
    row[FEATURE_INDEX["incidental_contact_fraction"]] = incidental / max(1, len(contacts))
    row[FEATURE_INDEX["incidental_evidence_missing"]] = float(not incidental_seams)
    evidence = [
        _cached_incidental_evidence(
            seam, normalized_rgb, lab, spatial_stats, evidence_cache
        )
        for seam in incidental_seams
    ]
    for metric_index, metric in enumerate(
        ("spatial_e0", "seam_w1_rgb_mse", "seam_w1_lab_mse", "seam_w1_ncc")
    ):
        _put_min_mean_max(
            row,
            f"incidental_{metric}",
            [values[metric_index] for values in evidence],
        )


def extract_relation_features(
    result: e23.CandidatePoolResult,
    candidate_ids: np.ndarray,
    raw_logits: np.ndarray,
    spatial_logits: np.ndarray,
    tiles_uint8: np.ndarray,
) -> RelationFeatureTable:
    """Create deterministic, label-free CRS-v1 rows for every E23 hypothesis.

    The signature intentionally has no extensible ``**kwargs`` and no label or
    sample object argument.  Training code must construct relevance labels in a
    separate module after this function returns.
    """

    valid, geometries = _validate_label_free_inputs(
        result, candidate_ids, raw_logits, spatial_logits, tiles_uint8
    )
    raw_stats = _raw_statistics(candidate_ids, raw_logits, valid)
    spatial_stats = _spatial_statistics(spatial_logits)
    selected_literal = np.zeros(
        (NUM_DIRECTIONS, NUM_TILES, NUM_TILES), dtype=np.bool_
    )
    nominations_by_pair: dict[tuple[int, int], list[_SpatialNomination]] = {}
    for direction in range(NUM_DIRECTIONS):
        for source in range(NUM_TILES):
            selected_targets = result.spatial_selected_ids[direction, source]
            selected_literal[direction, source, selected_targets] = True
            for rank, target_value in enumerate(selected_targets.tolist()):
                target = int(target_value)
                pair = (source, target) if source < target else (target, source)
                nominations_by_pair.setdefault(pair, []).append(
                    _SpatialNomination(direction, source, target, int(rank))
                )

    rgb = tiles_uint8.astype(np.float32) / np.float32(255.0)
    mean = rgb.mean(axis=(1, 2, 3), keepdims=True)
    rms = np.sqrt(np.square(rgb - mean).mean(axis=(1, 2, 3), keepdims=True) + EPSILON)
    normalized_rgb = np.clip((rgb - mean) / rms, -5.0, 5.0).astype(np.float32)
    lab = scaled_lab_tiles(tiles_uint8)
    incidental_evidence_cache: dict[
        tuple[int, int, int, int], tuple[float, float, float, float]
    ] = {}

    hypotheses = result.hypotheses
    count = len(hypotheses)
    width = len(FEATURE_NAMES)
    offset_features = np.zeros((count, width), dtype=np.float32)
    relations = np.empty((count, 4), dtype=np.int64)
    hypothesis_ids = np.empty(count, dtype=np.int64)
    relation_ids = np.empty(count, dtype=np.int64)
    support = np.empty(count, dtype=np.int64)
    e0 = np.empty(count, dtype=np.float32)
    pair_count = len(result.base_affinity_pairs)
    affinity_pairs = result.affinity_pairs

    for index, hypothesis in enumerate(hypotheses):
        row = offset_features[index]
        u, v, dr, dc = map(
            int, (hypothesis.u, hypothesis.v, hypothesis.dr, hypothesis.dc)
        )
        relations[index] = (u, v, dr, dc)
        hypothesis_ids[index] = int(hypothesis.hypothesis_id)
        relation_ids[index] = int(hypothesis.relation_id)
        support[index] = len(hypothesis.claim_ids)
        row[FEATURE_INDEX["is_none"]] = 0.0
        row[FEATURE_INDEX["has_offset"]] = 1.0
        row[FEATURE_INDEX["claim_missing"]] = 0.0
        _base_component_features(row, geometries[u], geometries[v])
        _merged_geometry_features(row, geometries[u], geometries[v], dr, dc)
        _projected_contact_features(
            row,
            result,
            hypothesis,
            normalized_rgb,
            lab,
            spatial_stats,
            incidental_evidence_cache,
        )

        claim_values = [result.claims[int(claim_id)] for claim_id in hypothesis.claim_ids]
        claim_count = len(claim_values)
        row[FEATURE_INDEX["claim_count_log1p"]] = np.log1p(claim_count)
        row[FEATURE_INDEX["unique_pair_count_log1p"]] = np.log1p(
            len({int(claim.pair_id) for claim in claim_values})
        )
        base_claims = [claim for claim in claim_values if int(claim.pair_id) < pair_count]
        spatial_claims = [claim for claim in claim_values if int(claim.pair_id) >= pair_count]
        row[FEATURE_INDEX["unique_endpoint_tile_count_log1p"]] = np.log1p(
            len({int(claim.first) for claim in claim_values}.union(
                int(claim.second) for claim in claim_values
            ))
        )
        row[FEATURE_INDEX["base_claim_count_log1p"]] = np.log1p(len(base_claims))
        row[FEATURE_INDEX["residual_claim_count_log1p"]] = np.log1p(len(spatial_claims))
        row[FEATURE_INDEX["base_claim_fraction"]] = len(base_claims) / claim_count
        row[FEATURE_INDEX["spatial_claim_fraction"]] = len(spatial_claims) / claim_count
        row[FEATURE_INDEX["horizontal_claim_fraction"]] = (
            sum(int(claim.dx) == 1 for claim in claim_values) / claim_count
        )
        row[FEATURE_INDEX["vertical_claim_fraction"]] = (
            sum(int(claim.dy) == 1 for claim in claim_values) / claim_count
        )
        reciprocal_base = sum(
            bool(getattr(affinity_pairs[int(claim.pair_id)], "reciprocal", False))
            for claim in base_claims
        )
        row[FEATURE_INDEX["reciprocal_base_fraction"]] = reciprocal_base / max(1, len(base_claims))

        raw_z: list[float] = []
        raw_percentile: list[float] = []
        raw_gap: list[float] = []
        raw_valid_row_size: list[float] = []
        raw_forward_percentile: list[float] = []
        raw_reverse_percentile: list[float] = []
        raw_abs_z: list[float] = []
        spatial_z: list[float] = []
        spatial_percentile: list[float] = []
        spatial_gap: list[float] = []
        spatial_forward_percentile: list[float] = []
        spatial_reverse_percentile: list[float] = []
        spatial_abs_z: list[float] = []
        spatial_wrong_z: list[float] = []
        spatial_wrong_percentile: list[float] = []
        spatial_correct_minus_wrong_z: list[float] = []
        spatial_correct_minus_wrong_percentile: list[float] = []
        literal_selected: list[float] = []
        nominations: list[float] = []
        nomination_best_rank: list[float] = []
        nomination_missing: list[float] = []
        nomination_correct_forward: list[float] = []
        nomination_correct_reverse: list[float] = []
        nomination_wrong_direction: list[float] = []
        nomination_reciprocal: list[float] = []
        nomination_direction_counts = np.zeros(NUM_DIRECTIONS, dtype=np.float64)
        nomination_total = 0
        claim_e0: list[float] = []
        seam: dict[tuple[int, str], list[float]] = {
            (seam_width, metric): []
            for seam_width in SEAM_WIDTHS
            for metric in (
                "rgb_mse",
                "gradient_mse",
                "tangential_gradient_mse",
                "ncc",
                "lab_mse",
            )
        }
        forward_present = 0
        reverse_present = 0
        reciprocal_claims = 0
        for claim in claim_values:
            forward = claim.forward_observation
            reverse = claim.reverse_observation
            for observation, output_percentiles, is_forward in (
                (forward, raw_forward_percentile, True),
                (reverse, raw_reverse_percentile, False),
            ):
                if observation is None:
                    continue
                z_value = float(
                    raw_stats.robust_z[
                        observation.direction, observation.source, observation.slot
                    ]
                )
                percentile_value = float(
                    raw_stats.percentile[
                        observation.direction, observation.source, observation.slot
                    ]
                )
                raw_z.append(z_value)
                raw_percentile.append(percentile_value)
                raw_gap.append(
                    float(
                        raw_stats.top1_gap[
                            observation.direction, observation.source, observation.slot
                        ]
                    )
                )
                raw_valid_row_size.append(
                    float(np.count_nonzero(valid[int(observation.source)])) / RAW_WIDTH
                )
                output_percentiles.append(percentile_value)
                if is_forward:
                    forward_present += 1
                else:
                    reverse_present += 1
            if forward is not None and reverse is not None:
                reciprocal_claims += 1
                raw_abs_z.append(
                    abs(
                        float(
                            raw_stats.robust_z[
                                forward.direction, forward.source, forward.slot
                            ]
                        )
                        - float(
                            raw_stats.robust_z[
                                reverse.direction, reverse.source, reverse.slot
                            ]
                        )
                    )
                )

            direction = e23.RIGHT if claim.dx == 1 else e23.DOWN
            inverse = e23.LEFT if claim.dx == 1 else e23.UP
            forward_key = (direction, int(claim.first), int(claim.second))
            reverse_key = (inverse, int(claim.second), int(claim.first))
            f_z = float(spatial_stats.robust_z[forward_key])
            r_z = float(spatial_stats.robust_z[reverse_key])
            f_pct = float(spatial_stats.percentile[forward_key])
            r_pct = float(spatial_stats.percentile[reverse_key])
            forward_wrong_directions = tuple(
                literal for literal in range(NUM_DIRECTIONS) if literal != direction
            )
            reverse_wrong_directions = tuple(
                literal for literal in range(NUM_DIRECTIONS) if literal != inverse
            )
            f_wrong_z = max(
                float(spatial_stats.robust_z[literal, claim.first, claim.second])
                for literal in forward_wrong_directions
            )
            r_wrong_z = max(
                float(spatial_stats.robust_z[literal, claim.second, claim.first])
                for literal in reverse_wrong_directions
            )
            f_wrong_pct = max(
                float(spatial_stats.percentile[literal, claim.first, claim.second])
                for literal in forward_wrong_directions
            )
            r_wrong_pct = max(
                float(spatial_stats.percentile[literal, claim.second, claim.first])
                for literal in reverse_wrong_directions
            )
            spatial_z.extend((f_z, r_z))
            spatial_percentile.extend((f_pct, r_pct))
            spatial_gap.extend(
                (
                    float(spatial_stats.top1_gap[forward_key]),
                    float(spatial_stats.top1_gap[reverse_key]),
                )
            )
            spatial_forward_percentile.append(f_pct)
            spatial_reverse_percentile.append(r_pct)
            spatial_abs_z.append(abs(f_z - r_z))
            spatial_wrong_z.extend((f_wrong_z, r_wrong_z))
            spatial_wrong_percentile.extend((f_wrong_pct, r_wrong_pct))
            spatial_correct_minus_wrong_z.extend((f_z - f_wrong_z, r_z - r_wrong_z))
            spatial_correct_minus_wrong_percentile.extend(
                (f_pct - f_wrong_pct, r_pct - r_wrong_pct)
            )
            literal_selected.extend(
                (
                    float(selected_literal[forward_key]),
                    float(selected_literal[reverse_key]),
                )
            )
            pair = affinity_pairs[int(claim.pair_id)]
            nominations.append(float(getattr(pair, "nomination_count", 0)))
            identity = (
                (int(claim.first), int(claim.second))
                if int(claim.first) < int(claim.second)
                else (int(claim.second), int(claim.first))
            )
            one_nominations = nominations_by_pair.get(identity, ())
            nomination_missing.append(float(not one_nominations))
            nomination_best_rank.append(
                max(
                    (
                        (SPATIAL_K - 1 - item.rank) / max(1, SPATIAL_K - 1)
                        for item in one_nominations
                    ),
                    default=0.0,
                )
            )
            forward_exact = any(
                item.direction == direction
                and item.source == int(claim.first)
                and item.target == int(claim.second)
                for item in one_nominations
            )
            reverse_exact = any(
                item.direction == inverse
                and item.source == int(claim.second)
                and item.target == int(claim.first)
                for item in one_nominations
            )
            nomination_correct_forward.append(float(forward_exact))
            nomination_correct_reverse.append(float(reverse_exact))
            nomination_wrong_direction.append(
                float(
                    any(
                        not (
                            item.direction == direction
                            and item.source == int(claim.first)
                            and item.target == int(claim.second)
                        )
                        and not (
                            item.direction == inverse
                            and item.source == int(claim.second)
                            and item.target == int(claim.first)
                        )
                        for item in one_nominations
                    )
                )
            )
            nomination_reciprocal.append(
                float(
                    bool(one_nominations)
                    and {item.source for item in one_nominations}
                    == {int(claim.first), int(claim.second)}
                )
            )
            for item in one_nominations:
                nomination_direction_counts[item.direction] += 1.0
                nomination_total += 1
            claim_e0.append(0.5 * (f_pct + r_pct))
            for seam_width in SEAM_WIDTHS:
                values = _seam_values_full(normalized_rgb, lab, claim, seam_width)
                for metric, value in zip(
                    (
                        "rgb_mse",
                        "gradient_mse",
                        "tangential_gradient_mse",
                        "ncc",
                        "lab_mse",
                    ),
                    values,
                ):
                    seam[(seam_width, metric)].append(value)

        observation_count = len(raw_z)
        row[FEATURE_INDEX["raw_missing_all"]] = float(observation_count == 0)
        row[FEATURE_INDEX["raw_observation_count_log1p"]] = np.log1p(observation_count)
        row[FEATURE_INDEX["raw_forward_present_fraction"]] = forward_present / claim_count
        row[FEATURE_INDEX["raw_reverse_present_fraction"]] = reverse_present / claim_count
        row[FEATURE_INDEX["raw_reciprocal_claim_fraction"]] = reciprocal_claims / claim_count
        row[FEATURE_INDEX["raw_forward_percentile_mean"]] = (
            float(np.mean(raw_forward_percentile)) if raw_forward_percentile else 0.0
        )
        row[FEATURE_INDEX["raw_reverse_percentile_mean"]] = (
            float(np.mean(raw_reverse_percentile)) if raw_reverse_percentile else 0.0
        )
        row[FEATURE_INDEX["raw_forward_reverse_abs_z_mean"]] = (
            float(np.mean(raw_abs_z)) if raw_abs_z else 0.0
        )
        row[FEATURE_INDEX["residual_nomination_missing_fraction"]] = float(
            np.mean(nomination_missing)
        )
        row[FEATURE_INDEX["residual_best_rank_percentile_mean"]] = float(
            np.mean(nomination_best_rank)
        )
        row[FEATURE_INDEX["residual_correct_forward_nomination_fraction"]] = float(
            np.mean(nomination_correct_forward)
        )
        row[FEATURE_INDEX["residual_correct_reverse_nomination_fraction"]] = float(
            np.mean(nomination_correct_reverse)
        )
        row[FEATURE_INDEX["residual_wrong_direction_nomination_fraction"]] = float(
            np.mean(nomination_wrong_direction)
        )
        row[FEATURE_INDEX["residual_reciprocal_nomination_fraction"]] = float(
            np.mean(nomination_reciprocal)
        )
        for literal, direction_name in enumerate(("up", "down", "left", "right")):
            row[FEATURE_INDEX[f"residual_nomination_{direction_name}_fraction"]] = (
                nomination_direction_counts[literal] / max(1, nomination_total)
            )
        row[FEATURE_INDEX["spatial_forward_percentile_mean"]] = float(
            np.mean(spatial_forward_percentile)
        )
        row[FEATURE_INDEX["spatial_reverse_percentile_mean"]] = float(
            np.mean(spatial_reverse_percentile)
        )
        row[FEATURE_INDEX["spatial_forward_reverse_abs_z_mean"]] = float(
            np.mean(spatial_abs_z)
        )
        row[FEATURE_INDEX["spatial_literal_selected_fraction"]] = float(
            np.mean(literal_selected)
        )
        row[FEATURE_INDEX["spatial_nomination_mean"]] = float(np.mean(nominations))
        row[FEATURE_INDEX["spatial_nomination_max"]] = float(np.max(nominations))
        row[FEATURE_INDEX["spatial_nomination_logsumexp"]] = _logmeanexp(nominations)
        _put_aggregate(row, "raw_robust_z", raw_z)
        _put_aggregate(row, "raw_percentile", raw_percentile)
        _put_aggregate(row, "raw_top1_gap", raw_gap)
        _put_aggregate(row, "raw_valid_row_size", raw_valid_row_size)
        _put_aggregate(row, "spatial_robust_z", spatial_z)
        _put_aggregate(row, "spatial_percentile", spatial_percentile)
        _put_aggregate(row, "spatial_top1_gap", spatial_gap)
        _put_aggregate(row, "spatial_wrong_robust_z", spatial_wrong_z)
        _put_aggregate(row, "spatial_wrong_percentile", spatial_wrong_percentile)
        _put_aggregate(
            row,
            "spatial_correct_minus_wrong_robust_z",
            spatial_correct_minus_wrong_z,
        )
        _put_aggregate(
            row,
            "spatial_correct_minus_wrong_percentile",
            spatial_correct_minus_wrong_percentile,
        )
        _put_aggregate(row, "residual_best_rank_percentile", nomination_best_rank)
        for seam_width in SEAM_WIDTHS:
            for metric in (
                "rgb_mse",
                "gradient_mse",
                "tangential_gradient_mse",
                "ncc",
                "lab_mse",
            ):
                _put_min_mean_max(
                    row, f"seam_w{seam_width}_{metric}", seam[(seam_width, metric)]
                )
        e0[index] = np.float32(max(claim_e0))

    # Hypotheses are already relation-sorted, so component-pair runs are the
    # canonical LambdaRank queries.  Compute their summaries without dropping
    # any scored offset.
    group_starts = [0]
    for index in range(1, count):
        if not np.array_equal(relations[index, :2], relations[index - 1, :2]):
            group_starts.append(index)
    group_starts.append(count)
    group_ranges = tuple(
        (int(start), int(stop))
        for start, stop in zip(group_starts[:-1], group_starts[1:])
    )
    _best_by_pair, top_by_pair, incident, incident_pair_counts = (
        _bounded_context_shortlists(
            relations, e0, group_ranges, len(result.components)
        )
    )
    for start, stop in group_ranges:
        indices = list(range(start, stop))
        ranked = sorted(
            indices,
            key=lambda idx: (
                -float(e0[idx]),
                int(relations[idx, 2]),
                int(relations[idx, 3]),
            ),
        )
        values = np.asarray([float(e0[idx]) for idx in indices], dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad + EPSILON
        top = [float(e0[idx]) for idx in ranked[:PAIR_CONTEXT_SUMMARY_K]]
        best = top[0]
        second = top[1] if len(top) > 1 else 0.0
        shifted = values - float(values.max())
        probabilities = np.exp(shifted)
        probabilities /= float(probabilities.sum())
        entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, EPSILON))))
        query_claims = int(support[start:stop].sum())
        for rank, idx in enumerate(ranked):
            row = offset_features[idx]
            row[FEATURE_INDEX["query_offset_count_log1p"]] = np.log1p(stop - start)
            row[FEATURE_INDEX["query_claim_count_log1p"]] = np.log1p(query_claims)
            row[FEATURE_INDEX["query_support_mass_log1p"]] = np.log1p(query_claims)
            row[FEATURE_INDEX["query_best_e0"]] = best
            row[FEATURE_INDEX["query_second_e0"]] = second
            row[FEATURE_INDEX["query_e0_margin"]] = best - second if len(ranked) > 1 else 0.0
            row[FEATURE_INDEX["query_e0_entropy"]] = entropy
            row[FEATURE_INDEX["query_no_alternative"]] = float(len(ranked) == 1)
            row[FEATURE_INDEX["query_top4_e0_min"]] = min(top)
            row[FEATURE_INDEX["query_top4_e0_mean"]] = float(np.mean(top))
            row[FEATURE_INDEX["query_top4_e0_max"]] = max(top)
            row[FEATURE_INDEX["offset_e0"]] = float(e0[idx])
            row[FEATURE_INDEX["offset_e0_rank_percentile"]] = (
                (len(ranked) - 1 - rank) / max(1, len(ranked) - 1)
            )
            row[FEATURE_INDEX["offset_e0_gap_best"]] = float(e0[idx]) - best
            if len(ranked) > 1:
                best_other = max(float(e0[other]) for other in ranked if other != idx)
                row[FEATURE_INDEX["offset_e0_margin_best_other"]] = (
                    float(e0[idx]) - best_other
                )
            else:
                row[FEATURE_INDEX["offset_e0_margin_best_other"]] = 0.0
            row[FEATURE_INDEX["offset_e0_robust_z"]] = np.clip(
                (float(e0[idx]) - median) / scale, -8.0, 8.0
            )

    # Endpoint hypothesis counts remain complete; only the two-hop leg graph
    # uses the bounded shortlists above.  All hypotheses remain scored rows.
    incident_hypothesis_counts: dict[int, int] = {
        component.component_id: 0 for component in result.components
    }
    for relation in relations:
        incident_hypothesis_counts[int(relation[0])] += 1
        incident_hypothesis_counts[int(relation[1])] += 1
    for index in range(count):
        row = offset_features[index]
        relation = tuple(map(int, relations[index]))
        u, v, dr, dc = relation
        neighbours_u = incident[u]
        neighbours_v = incident[v]
        degree_u = len(neighbours_u) / max(1, len(result.components) - 1)
        degree_v = len(neighbours_v) / max(1, len(result.components) - 1)
        row[FEATURE_INDEX["incident_degree_min_scaled"]] = min(degree_u, degree_v)
        row[FEATURE_INDEX["incident_degree_max_scaled"]] = max(degree_u, degree_v)
        row[FEATURE_INDEX["incident_degree_absdiff_scaled"]] = abs(degree_u - degree_v)
        endpoint_e0_u = [float(e0[idx]) for idx in neighbours_u.values()]
        endpoint_e0_v = [float(e0[idx]) for idx in neighbours_v.values()]
        e0_u = max(endpoint_e0_u, default=0.0)
        e0_v = max(endpoint_e0_v, default=0.0)
        mean_u = float(np.mean(endpoint_e0_u)) if endpoint_e0_u else 0.0
        mean_v = float(np.mean(endpoint_e0_v)) if endpoint_e0_v else 0.0
        row[FEATURE_INDEX["incident_e0_min"]] = min(e0_u, e0_v)
        row[FEATURE_INDEX["incident_e0_max"]] = max(e0_u, e0_v)
        row[FEATURE_INDEX["incident_e0_absdiff"]] = abs(e0_u - e0_v)
        row[FEATURE_INDEX["incident_e0_mean_min"]] = min(mean_u, mean_v)
        row[FEATURE_INDEX["incident_e0_mean_max"]] = max(mean_u, mean_v)
        row[FEATURE_INDEX["incident_e0_mean_absdiff"]] = abs(mean_u - mean_v)
        pair_u, pair_v = incident_pair_counts[u], incident_pair_counts[v]
        hypothesis_u = incident_hypothesis_counts[u]
        hypothesis_v = incident_hypothesis_counts[v]
        for prefix, first, second in (
            ("incident_pair_count", pair_u, pair_v),
            ("incident_hypothesis_count", hypothesis_u, hypothesis_v),
        ):
            row[FEATURE_INDEX[f"{prefix}_min_log1p"]] = np.log1p(min(first, second))
            row[FEATURE_INDEX[f"{prefix}_max_log1p"]] = np.log1p(max(first, second))
            row[FEATURE_INDEX[f"{prefix}_absdiff_log1p"]] = np.log1p(abs(first - second))
        shared = sorted(set(neighbours_u).intersection(neighbours_v))
        exact_support: list[float] = []
        conflict_support: list[float] = []
        exact_intermediates: set[int] = set()
        conflict_intermediates: set[int] = set()
        residuals: list[int] = []
        for middle in shared:
            pair_left = (u, middle) if u < middle else (middle, u)
            pair_right = (middle, v) if middle < v else (v, middle)
            for left_index in top_by_pair[pair_left]:
                for right_index in top_by_pair[pair_right]:
                    left_relation = tuple(map(int, relations[left_index]))
                    right_relation = tuple(map(int, relations[right_index]))
                    u_to_middle = _oriented_offset(left_relation, u, middle)
                    middle_to_v = _oriented_offset(right_relation, middle, v)
                    path = (
                        u_to_middle[0] + middle_to_v[0],
                        u_to_middle[1] + middle_to_v[1],
                    )
                    residual = abs(path[0] - dr) + abs(path[1] - dc)
                    residuals.append(residual)
                    path_support = min(float(e0[left_index]), float(e0[right_index]))
                    if residual == 0:
                        exact_support.append(path_support)
                        exact_intermediates.add(middle)
                    else:
                        conflict_support.append(path_support)
                        conflict_intermediates.add(middle)
        row[FEATURE_INDEX["shared_intermediates_log1p"]] = np.log1p(len(shared))
        row[FEATURE_INDEX["twohop_path_count_log1p"]] = np.log1p(
            len(exact_support) + len(conflict_support)
        )
        row[FEATURE_INDEX["twohop_exact_count_log1p"]] = np.log1p(len(exact_support))
        row[FEATURE_INDEX["twohop_exact_intermediates_log1p"]] = np.log1p(
            len(exact_intermediates)
        )
        row[FEATURE_INDEX["twohop_conflict_count_log1p"]] = np.log1p(len(conflict_support))
        row[FEATURE_INDEX["twohop_conflict_intermediates_log1p"]] = np.log1p(
            len(conflict_intermediates)
        )
        row[FEATURE_INDEX["twohop_exact_support_sum"]] = sum(exact_support)
        row[FEATURE_INDEX["twohop_exact_support_mean"]] = (
            float(np.mean(exact_support)) if exact_support else 0.0
        )
        row[FEATURE_INDEX["twohop_exact_support_max"]] = max(exact_support, default=0.0)
        row[FEATURE_INDEX["twohop_conflict_support_sum"]] = sum(conflict_support)
        row[FEATURE_INDEX["twohop_conflict_support_mean"]] = (
            float(np.mean(conflict_support)) if conflict_support else 0.0
        )
        row[FEATURE_INDEX["twohop_conflict_support_max"]] = max(conflict_support, default=0.0)
        row[FEATURE_INDEX["twohop_best_match_minus_conflict"]] = (
            max(exact_support, default=0.0) - max(conflict_support, default=0.0)
        )
        row[FEATURE_INDEX["twohop_zero_sum_witness_count_log1p"]] = np.log1p(
            len(exact_support)
        )
        row[FEATURE_INDEX["twohop_zero_sum_witness_support_sum"]] = sum(exact_support)
        row[FEATURE_INDEX["twohop_zero_sum_witness_support_max"]] = max(
            exact_support, default=0.0
        )
        row[FEATURE_INDEX["twohop_min_l1_scaled"]] = (
            min(residuals) / (2 * GRID) if residuals else 0.0
        )

    # Materialize each query contiguously as exact offsets followed by NONE.
    rows = count + len(group_ranges)
    features = np.zeros((rows, width), dtype=np.float32)
    out_hypothesis = np.full(rows, NONE_HYPOTHESIS_ID, dtype=np.int64)
    out_relation_id = np.full(rows, NONE_RELATION_ID, dtype=np.int64)
    out_relations = np.zeros((rows, 4), dtype=np.int64)
    out_kind = np.full(rows, ROW_OFFSET, dtype=np.uint8)
    out_support = np.zeros(rows, dtype=np.int64)
    query_offsets = np.empty(len(group_ranges) + 1, dtype=np.int64)
    cursor = 0
    query_offsets[0] = 0
    none_copy_names = tuple(
        name
        for name in _CONTEXT_FEATURE_NAMES
        if name.startswith("query_")
        or name.startswith("incident_")
        or name == "shared_intermediates_log1p"
    )
    for query_index, (start, stop) in enumerate(group_ranges):
        length = stop - start
        features[cursor : cursor + length] = offset_features[start:stop]
        out_hypothesis[cursor : cursor + length] = hypothesis_ids[start:stop]
        out_relation_id[cursor : cursor + length] = relation_ids[start:stop]
        out_relations[cursor : cursor + length] = relations[start:stop]
        out_support[cursor : cursor + length] = support[start:stop]
        cursor += length
        none = features[cursor]
        u, v = map(int, relations[start, :2])
        none[FEATURE_INDEX["is_none"]] = 1.0
        none[FEATURE_INDEX["has_offset"]] = 0.0
        none[FEATURE_INDEX["claim_missing"]] = 1.0
        none[FEATURE_INDEX["raw_missing_all"]] = 1.0
        _base_component_features(none, geometries[u], geometries[v])
        # NONE receives only candidate-independent query/endpoint context.
        for name in none_copy_names:
            none[FEATURE_INDEX[name]] = offset_features[start, FEATURE_INDEX[name]]
        out_relations[cursor] = (u, v, 0, 0)
        out_kind[cursor] = ROW_NONE
        cursor += 1
        query_offsets[query_index + 1] = cursor
    if cursor != rows or not np.isfinite(features).all():
        raise ContextRelationSelectorError("feature materialization/accounting drifted")

    table = RelationFeatureTable(
        features=_readonly(features.astype(np.float32, copy=False)),
        hypothesis_ids=_readonly(out_hypothesis),
        relation_ids=_readonly(out_relation_id),
        relations=_readonly(out_relations),
        row_kind=_readonly(out_kind),
        support=_readonly(out_support),
        query_offsets=_readonly(query_offsets),
        scene_offsets=_readonly(np.asarray((0, rows), dtype=np.int64)),
    )
    return _validate_table(table)


def concatenate_feature_tables(
    tables: Sequence[RelationFeatureTable],
) -> RelationFeatureTable:
    """Concatenate complete scene tables without exposing a scene-id feature."""

    values = tuple(_validate_table(table) for table in tables)
    if not values:
        raise ContextRelationSelectorError("at least one feature table is required")
    row_cursor = 0
    offsets = [0]
    scene_offsets = [0]
    for table in values:
        offsets.extend((table.query_offsets[1:] + row_cursor).tolist())
        scene_offsets.extend((table.scene_offsets[1:] + row_cursor).tolist())
        row_cursor += table.rows
    combined = RelationFeatureTable(
        features=_readonly(np.concatenate([table.features for table in values], axis=0)),
        hypothesis_ids=_readonly(
            np.concatenate([table.hypothesis_ids for table in values], axis=0)
        ),
        relation_ids=_readonly(
            np.concatenate([table.relation_ids for table in values], axis=0)
        ),
        relations=_readonly(np.concatenate([table.relations for table in values], axis=0)),
        row_kind=_readonly(np.concatenate([table.row_kind for table in values], axis=0)),
        support=_readonly(np.concatenate([table.support for table in values], axis=0)),
        query_offsets=_readonly(np.asarray(offsets, dtype=np.int64)),
        scene_offsets=_readonly(np.asarray(scene_offsets, dtype=np.int64)),
    )
    return _validate_table(combined)


def balanced_query_row_weights(
    table: RelationFeatureTable, relevance: np.ndarray
) -> np.ndarray:
    """Balance positive-offset and NONE-positive query categories, then rows.

    Every query has exactly one relevance-one row.  The two query categories
    receive equal total weight when both are present; within a category every
    query receives equal mass and every row in a query shares that query mass.
    """

    value = _validate_table(table)
    labels = np.asarray(relevance)
    if labels.shape != (value.rows,) or not np.issubdtype(labels.dtype, np.integer):
        raise ContextRelationSelectorError("relevance must be an integer vector aligned to rows")
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ContextRelationSelectorError("LambdaRank relevance must be binary")
    weights = np.empty(value.rows, dtype=np.float32)
    for scene_start, scene_stop in zip(
        value.scene_offsets[:-1], value.scene_offsets[1:]
    ):
        scene_start_i, scene_stop_i = int(scene_start), int(scene_stop)
        first_query = int(np.searchsorted(value.query_offsets, scene_start_i, side="left"))
        last_query = int(np.searchsorted(value.query_offsets, scene_stop_i, side="left"))
        if (
            first_query >= value.query_offsets.size
            or last_query >= value.query_offsets.size
            or int(value.query_offsets[first_query]) != scene_start_i
            or int(value.query_offsets[last_query]) != scene_stop_i
        ):
            raise ContextRelationSelectorError("scene boundary is not a query boundary")
        query_rows = [
            (int(value.query_offsets[index]), int(value.query_offsets[index + 1]))
            for index in range(first_query, last_query)
        ]
        categories: list[bool] = []
        for start_i, stop_i in query_rows:
            one = labels[start_i:stop_i]
            if int(one.sum()) != 1:
                raise ContextRelationSelectorError("every query needs exactly one relevant row")
            categories.append(bool(one[-1] == 1))
        category_counts = {
            category: categories.count(category) for category in set(categories)
        }
        if set(category_counts) != {False, True}:
            raise ContextRelationSelectorError(
                "every training scene needs nonempty positive-offset and NONE-positive categories"
            )
        for category, (start_i, stop_i) in zip(categories, query_rows):
            query_mass = 0.5 / category_counts[category]
            weights[start_i:stop_i] = query_mass / (stop_i - start_i)
    weights /= float(weights.mean())
    return _readonly(weights)


def fit_lambdarank(
    table: RelationFeatureTable,
    relevance: np.ndarray,
    *,
    fold: int,
    row_weights: np.ndarray | None = None,
) -> Any:
    """Fit the one fixed CRS-v1 LambdaRank model; no early stopping or sweep."""

    value = _validate_table(table)
    labels = np.asarray(relevance)
    if labels.shape != (value.rows,) or not np.issubdtype(labels.dtype, np.integer):
        raise ContextRelationSelectorError("relevance is not aligned integer data")
    if int(fold) != fold or fold < 0:
        raise ContextRelationSelectorError("fold must be a non-negative integer")
    frozen_weights = balanced_query_row_weights(value, labels)
    weights = (
        frozen_weights
        if row_weights is None
        else np.asarray(row_weights, dtype=np.float32)
    )
    if weights.shape != (value.rows,) or not np.isfinite(weights).all() or bool((weights <= 0).any()):
        raise ContextRelationSelectorError("row_weights must be finite positive aligned values")
    if not np.array_equal(weights, frozen_weights):
        raise ContextRelationSelectorError(
            "CRS-v1 forbids row-weight overrides that differ from frozen per-scene balancing"
        )
    try:
        from lightgbm import LGBMRanker
    except Exception as exc:  # pragma: no cover - environment failure path
        raise ContextRelationSelectorError("LightGBM is required for CRS-v1 training") from exc
    config = dict(LIGHTGBM_CONFIG)
    seed = 1234 + int(fold)
    config.update(
        random_state=seed,
        data_random_seed=seed,
        feature_fraction_seed=seed,
    )
    model = LGBMRanker(**config)
    model.fit(
        value.features,
        labels.astype(np.int32, copy=False),
        group=value.query_sizes.astype(np.int32, copy=False),
        sample_weight=weights,
    )
    return model


def predict_scores(model: Predictor, table: RelationFeatureTable) -> np.ndarray:
    """Predict one finite float64 score per canonical offset/NONE row."""

    value = _validate_table(table)
    if not hasattr(model, "predict"):
        raise ContextRelationSelectorError("model has no predict method")
    scores = np.asarray(model.predict(value.features), dtype=np.float64)
    if scores.shape != (value.rows,) or not np.isfinite(scores).all():
        raise ContextRelationSelectorError("model predictions are not finite/aligned")
    return _readonly(scores)


def select_pair_winners(
    table: RelationFeatureTable,
    scores: np.ndarray,
    *,
    component_count: int,
) -> tuple[tuple[SelectedRelation, ...], tuple[SelectedRelation, ...], int]:
    """Select strict-positive pair margins and the frozen global attempt prefix."""

    value = _validate_table(table)
    predictions = np.asarray(scores, dtype=np.float64)
    if predictions.shape != (value.rows,) or not np.isfinite(predictions).all():
        raise ContextRelationSelectorError("scores must be finite and aligned")
    if int(component_count) != component_count or component_count < 1:
        raise ContextRelationSelectorError("component_count must be positive")
    selected: list[SelectedRelation] = []
    for start, stop in zip(value.query_offsets[:-1], value.query_offsets[1:]):
        start_i, stop_i = int(start), int(stop)
        none_index = stop_i - 1
        offset_indices = range(start_i, none_index)
        best = min(
            offset_indices,
            key=lambda idx: (
                -float(predictions[idx]),
                int(value.relations[idx, 2]),
                int(value.relations[idx, 3]),
            ),
        )
        none_score = float(predictions[none_index])
        score = float(predictions[best])
        margin = score - none_score
        if not margin > 0.0:
            continue
        u, v, dr, dc = map(int, value.relations[best])
        selected.append(
            SelectedRelation(
                hypothesis_id=int(value.hypothesis_ids[best]),
                relation_id=int(value.relation_ids[best]),
                u=u,
                v=v,
                dr=dr,
                dc=dc,
                score=score,
                none_score=none_score,
                margin=margin,
                support=int(value.support[best]),
            )
        )
    selected.sort(
        key=lambda item: (
            -item.margin,
            item.u,
            item.v,
            item.dr,
            item.dc,
        )
    )
    cap = min(
        len(selected),
        max(0, ATTEMPT_MULTIPLIER * (int(component_count) - 1)),
    )
    attempted = tuple(selected[:cap])
    return tuple(selected), attempted, cap


class _PotentialDSU:
    """Signed component DSU whose rejected operations never mutate state."""

    def __init__(self, result: e23.CandidatePoolResult) -> None:
        if type(result) is not e23.CandidatePoolResult:
            raise ContextRelationSelectorError("DSU requires the exact E23 result")
        self.result = result
        self.components = result.components
        self.parent = {component.component_id: component.component_id for component in self.components}
        self.size = {component.component_id: 1 for component in self.components}
        self.delta = {component.component_id: (0, 0) for component in self.components}
        self.members = {
            component.component_id: {component.component_id} for component in self.components
        }
        self.entries: dict[int, dict[tuple[int, int], int]] = {}
        self.translations: dict[int, dict[int, tuple[int, int]]] = {}
        for component in self.components:
            occupied: dict[tuple[int, int], int] = {}
            for tile, row, col in component.entries:
                coordinate = (int(row), int(col))
                if coordinate in occupied:
                    raise ContextRelationSelectorError("component collision reached DSU")
                occupied[coordinate] = int(tile)
            self.entries[component.component_id] = occupied
            self.translations[component.component_id] = {component.component_id: (0, 0)}

    def find(self, component_id: int) -> tuple[int, tuple[int, int]]:
        if component_id not in self.parent:
            raise ContextRelationSelectorError("relation references an unknown component")
        parent = self.parent[component_id]
        if parent == component_id:
            return component_id, (0, 0)
        root, parent_delta = self.find(parent)
        own = self.delta[component_id]
        total = (own[0] + parent_delta[0], own[1] + parent_delta[1])
        # Deliberately do not path-compress: a rejected proposal must not
        # mutate even parent/potential representation as a read side effect.
        return root, total

    def _contact_valid(
        self,
        selection: SelectedRelation,
        translations: Mapping[int, tuple[int, int]],
    ) -> bool:
        hypothesis = self.result.hypotheses[selection.hypothesis_id]
        if (
            hypothesis.relation_id != selection.relation_id
            or hypothesis.relation != selection.relation
        ):
            raise ContextRelationSelectorError("selected relation metadata does not match E23")
        for claim_id in hypothesis.claim_ids:
            claim = self.result.claims[int(claim_id)]
            first_component = int(self.result.owner[claim.first])
            second_component = int(self.result.owner[claim.second])
            if first_component not in translations or second_component not in translations:
                return False
            first = (
                translations[first_component][0] + int(self.result.local_rows[claim.first]),
                translations[first_component][1] + int(self.result.local_cols[claim.first]),
            )
            second = (
                translations[second_component][0] + int(self.result.local_rows[claim.second]),
                translations[second_component][1] + int(self.result.local_cols[claim.second]),
            )
            if (second[0] - first[0], second[1] - first[1]) != (claim.dy, claim.dx):
                return False
        return True

    def try_accept(self, selection: SelectedRelation) -> tuple[bool, str, bool, bool]:
        root_u, shift_u = self.find(selection.u)
        root_v, shift_v = self.find(selection.v)
        if root_u == root_v:
            observed = (shift_v[0] - shift_u[0], shift_v[1] - shift_u[1])
            if observed != (selection.dr, selection.dc):
                return False, "conflict", False, False
            if not self._contact_valid(selection, self.translations[root_u]):
                return False, "contact", False, False
            return True, "cycle", False, True

        root_v_from_u = (
            selection.dr + shift_u[0] - shift_v[0],
            selection.dc + shift_u[1] - shift_v[1],
        )
        proposed_translations = dict(self.translations[root_u])
        proposed_translations.update(
            {
                component_id: (translation[0] + root_v_from_u[0], translation[1] + root_v_from_u[1])
                for component_id, translation in self.translations[root_v].items()
            }
        )
        if not self._contact_valid(selection, proposed_translations):
            return False, "contact", False, False
        shifted_v = {
            (coordinate[0] + root_v_from_u[0], coordinate[1] + root_v_from_u[1]): tile
            for coordinate, tile in self.entries[root_v].items()
        }
        if set(self.entries[root_u]).intersection(shifted_v):
            return False, "collision", False, False
        proposed_entries = dict(self.entries[root_u])
        proposed_entries.update(shifted_v)
        rows = [row for row, _col in proposed_entries]
        cols = [col for _row, col in proposed_entries]
        if max(rows) - min(rows) + 1 > GRID or max(cols) - min(cols) + 1 > GRID:
            return False, "span", False, False

        keep_u = (self.size[root_u], -root_u) >= (self.size[root_v], -root_v)
        if keep_u:
            self.parent[root_v] = root_u
            self.delta[root_v] = root_v_from_u
            self.size[root_u] += self.size[root_v]
            self.members[root_u].update(self.members.pop(root_v))
            self.entries[root_u] = proposed_entries
            self.translations[root_u] = proposed_translations
            del self.entries[root_v]
            del self.translations[root_v]
        else:
            root_u_from_v = (-root_v_from_u[0], -root_v_from_u[1])
            shifted_u = {
                (coordinate[0] + root_u_from_v[0], coordinate[1] + root_u_from_v[1]): tile
                for coordinate, tile in self.entries[root_u].items()
            }
            retained_entries = dict(self.entries[root_v])
            retained_entries.update(shifted_u)
            retained_translations = dict(self.translations[root_v])
            retained_translations.update(
                {
                    component_id: (
                        translation[0] + root_u_from_v[0],
                        translation[1] + root_u_from_v[1],
                    )
                    for component_id, translation in self.translations[root_u].items()
                }
            )
            self.parent[root_u] = root_v
            self.delta[root_u] = root_u_from_v
            self.size[root_v] += self.size[root_u]
            self.members[root_v].update(self.members.pop(root_u))
            self.entries[root_v] = retained_entries
            self.translations[root_v] = retained_translations
            del self.entries[root_u]
            del self.translations[root_u]
        return True, "tree", True, False

    def normalized_components(self) -> tuple[dict[int, tuple[int, int]], ...]:
        output: list[dict[int, tuple[int, int]]] = []
        roots = sorted(self.entries, key=lambda root: (-len(self.entries[root]), root))
        for root in roots:
            entries = self.entries[root]
            minimum_row = min(row for row, _col in entries)
            minimum_col = min(col for _row, col in entries)
            component = {
                tile: (row - minimum_row, col - minimum_col)
                for (row, col), tile in entries.items()
            }
            output.append(dict(sorted(component.items())))
        if sum(len(component) for component in output) != NUM_TILES:
            raise ContextRelationSelectorError("decoded components lost or duplicated tiles")
        return tuple(output)


def decode_relation_scores(
    result: e23.CandidatePoolResult,
    table: RelationFeatureTable,
    scores: np.ndarray,
) -> RelationDecodeResult:
    """Apply strict pair selection and the frozen rollback-safe DSU decoder."""

    if type(result) is not e23.CandidatePoolResult:
        raise ContextRelationSelectorError("decoder requires the exact E23 result")
    selected, attempted, cap = select_pair_winners(
        table, scores, component_count=len(result.components)
    )
    dsu = _PotentialDSU(result)
    outcomes: list[DSUOutcome] = []
    rejections: dict[str, int] = {}
    tree_merges = 0
    cycles = 0
    for selection in attempted:
        accepted, reason, tree, cycle = dsu.try_accept(selection)
        outcomes.append(DSUOutcome(selection, accepted, reason, tree, cycle))
        tree_merges += int(tree)
        cycles += int(cycle)
        if not accepted:
            rejections[reason] = rejections.get(reason, 0) + 1
    return RelationDecodeResult(
        selected=selected,
        attempted=attempted,
        outcomes=tuple(outcomes),
        components=dsu.normalized_components(),
        attempt_cap=cap,
        tree_merges=tree_merges,
        cycle_acceptances=cycles,
        rejection_counts=dict(sorted(rejections.items())),
    )


_FEATURE_CACHE_KEYS = frozenset(
    {
        "schema_version",
        "feature_names",
        "features",
        "hypothesis_ids",
        "relation_ids",
        "relations",
        "row_kind",
        "support",
        "query_offsets",
        "scene_offsets",
    }
)


def _require_e_drive(path: Path) -> Path:
    resolved = path.resolve()
    root = _E24_ARTIFACT_ROOT.resolve()
    try:
        contained = os.path.commonpath((str(root), str(resolved))) == str(root)
    except ValueError:
        contained = False
    if resolved.drive.upper() != "E:" or not contained or resolved == root:
        raise ContextRelationSelectorError(
            "E24 feature caches must live below E:/pazzle_work/posegraph_e24_selector"
        )
    return resolved


def _npy_member_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, version=(1, 0), allow_pickle=False)
    return stream.getvalue()


def feature_table_npz_bytes(table: RelationFeatureTable) -> bytes:
    """Return the canonical fixed-timestamp feature-cache byte stream."""

    value = _validate_table(table)
    members = (
        ("schema_version", np.asarray([SCHEMA_VERSION], dtype=np.int64)),
        (
            "feature_names",
            np.asarray(FEATURE_NAMES, dtype=f"<U{max(map(len, FEATURE_NAMES))}"),
        ),
        ("features", value.features),
        ("hypothesis_ids", value.hypothesis_ids),
        ("relation_ids", value.relation_ids),
        ("relations", value.relations),
        ("row_kind", value.row_kind),
        ("support", value.support),
        ("query_offsets", value.query_offsets),
        ("scene_offsets", value.scene_offsets),
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, member in members:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_member_bytes(member))
    return stream.getvalue()


def save_feature_table_npz(path: Path, table: RelationFeatureTable) -> None:
    """Publish a strict, create-once, uncompressed feature cache on E:."""

    destination = _require_e_drive(Path(path))
    payload = feature_table_npz_bytes(table)
    if destination.suffix.lower() != ".npz":
        raise ContextRelationSelectorError("feature cache must use .npz")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_feature_table_npz(path: Path) -> RelationFeatureTable:
    """Load only the exact CRS-v1 feature-cache schema from E:."""

    source = _require_e_drive(Path(path))
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as stored:
        if frozenset(stored.files) != _FEATURE_CACHE_KEYS:
            raise ContextRelationSelectorError("feature cache key whitelist failed")
        if not np.array_equal(
            stored["schema_version"], np.asarray([SCHEMA_VERSION], dtype=np.int64)
        ):
            raise ContextRelationSelectorError("feature cache schema version drifted")
        if tuple(stored["feature_names"].tolist()) != FEATURE_NAMES:
            raise ContextRelationSelectorError("feature cache feature names drifted")
        exact_dtypes = {
            "features": np.dtype(np.float32),
            "hypothesis_ids": np.dtype(np.int64),
            "relation_ids": np.dtype(np.int64),
            "relations": np.dtype(np.int64),
            "row_kind": np.dtype(np.uint8),
            "support": np.dtype(np.int64),
            "query_offsets": np.dtype(np.int64),
            "scene_offsets": np.dtype(np.int64),
        }
        for key, expected_dtype in exact_dtypes.items():
            if stored[key].dtype != expected_dtype:
                raise ContextRelationSelectorError(
                    f"feature cache {key} dtype drifted"
                )
        table = RelationFeatureTable(
            features=_readonly(stored["features"].astype(np.float32, copy=False)),
            hypothesis_ids=_readonly(stored["hypothesis_ids"].astype(np.int64, copy=False)),
            relation_ids=_readonly(stored["relation_ids"].astype(np.int64, copy=False)),
            relations=_readonly(stored["relations"].astype(np.int64, copy=False)),
            row_kind=_readonly(stored["row_kind"].astype(np.uint8, copy=False)),
            support=_readonly(stored["support"].astype(np.int64, copy=False)),
            query_offsets=_readonly(stored["query_offsets"].astype(np.int64, copy=False)),
            scene_offsets=_readonly(stored["scene_offsets"].astype(np.int64, copy=False)),
        )
    value = _validate_table(table)
    if source.read_bytes() != feature_table_npz_bytes(value):
        raise ContextRelationSelectorError("feature cache bytes are not canonical")
    return value


__all__ = (
    "ATTEMPT_MULTIPLIER",
    "ContextRelationSelectorError",
    "DSUOutcome",
    "FEATURE_INDEX",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA",
    "LIGHTGBM_CONFIG",
    "NONE_HYPOTHESIS_ID",
    "NONE_RELATION_ID",
    "PAIR_CONTEXT_SUMMARY_K",
    "PROTOCOL",
    "PROTOCOL_NAME",
    "RelationDecodeResult",
    "RelationFeatureTable",
    "SCHEMA_VERSION",
    "SelectedRelation",
    "balanced_query_row_weights",
    "concatenate_feature_tables",
    "decode_relation_scores",
    "extract_relation_features",
    "feature_table_npz_bytes",
    "fit_lambdarank",
    "load_feature_table_npz",
    "predict_scores",
    "save_feature_table_npz",
    "select_pair_winners",
)
