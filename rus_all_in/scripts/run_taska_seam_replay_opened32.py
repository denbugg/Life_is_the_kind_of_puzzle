#!/usr/bin/env python3
"""Replay the legal target-free TASKA seam stack on the opened eval32 panel.

The candidate arm is fixed before execution: the historical v3 and local seam
checkpoints, raw/median/bilateral dirty-tile views, two orientations, a dynamic
350-edge vote target, depth-one mutual edges, raw fused-score ordering, and the
portable raw-tail global solver.  Chooser, verifier, border prior, restoration,
and every target-id-derived feature are excluded.

The historical ``quad=0.4`` default is deliberately *not* reproduced.  Its
boundary mask used ``tile_id % 24`` / ``tile_id // 24`` after validation tiles
had been reordered with target-derived ids.  Applying that mask to a raw bag is
not permutation-equivariant.  This legal replay fixes ``quad_weight=0``.

Candidate scores, harvest membership, and strict layouts are written and
hash-rostered before exact references are recreated.  The panel has already
been opened and is development evidence only; promotion requires a separate
source-disjoint confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.layout_evaluation import LayoutEvaluation, evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import (
    RawTailEdge,
    RawTailGlobalConfig,
    solve_raw_tail_global,
)

try:
    from scripts.run_union_hard_edge_priority_pilot import (
        BOOTSTRAP_SEED,
        DEFAULT_TARGETS,
        PROJECT_ROOT,
        CleanTileCache,
        _dirty_sha256,
        prepare_case,
        source_clustered_delta_ci,
    )
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    from run_union_hard_edge_priority_pilot import (
        BOOTSTRAP_SEED,
        DEFAULT_TARGETS,
        PROJECT_ROOT,
        CleanTileCache,
        _dirty_sha256,
        prepare_case,
        source_clustered_delta_ci,
    )


CONFIG_SCHEMA = "aiijc-taska-seam-replay-opened32-v1"
REPORT_SCHEMA = "aiijc-taska-seam-replay-opened32-report-v1"
FROZEN_SCHEMA = "aiijc-taska-seam-frozen-target-free-eval-v1"
EXPERIMENT = "taska-seam-replay-opened32-v1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_seam_replay_opened32_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-seam-replay/opened32-v1"

GRID = 24
COUNT = GRID * GRID
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
EVAL_SOURCE_COUNT = 16
DRAWS = (0, 1)
EVAL_CASE_COUNT = EVAL_SOURCE_COUNT * len(DRAWS)
EVAL_SOURCE_ORDER_DIGEST = "13f7fe84262f9c4d0aee7ce80dfdc1edeec3ce7f1b5082f06ae5c6aceda6fa5f"
EVAL_CASES_DIGEST = "6b872422b32562b845deb514fd478cd21379384cccdcf07a9e4db257e71b6e04"
SYNTHETIC_SEED = 1_267_233_517

PARENT_COMMITMENT_SHA256 = "575bb43d850ec3276b61aef616cfa9f2f5fa6f31db35f417d6852b9a38dac540"
PARENT_FROZEN_EVAL_SHA256 = "86bf9dfa5f0117e3ea35e3c0806f5909a271c176b90cea24c0f1dc7802e11fcc"
PARENT_FROZEN_METADATA_SHA256 = (
    "b0e26d3fdf2a05169d6ba18c3ec62561470f205f64c4bf454e8142f8ae39edac"
)
V3_CHECKPOINT_SHA256 = "6f0917d66d908f6cc0f4c1fcb949d3bcbadcba2490a6f7b5a12596e61de9730e"
LOCAL_CHECKPOINT_SHA256 = "5932853a73961d261b494368a4db04633fecc5996771c14d64f49ef00c7cfe73"

MATCHER_KINDS = ("v3", "local")
VIEWS = ("raw", "median", "bilateral")
ORIENTATIONS = 2
VOTES_FALLBACK = 10
VOTE_TARGET = 350
HARVEST_DEPTH = 1
QUAD_WEIGHT = 0.0
SINKHORN_ITERATIONS = 20
CYCLE_ROUNDS = 3
CYCLE_WEIGHT = 0.35
ACYCLIC_WEIGHT = 3.0
SINKHORN_SLACK = 0

SOLVER_CONFIG = RawTailGlobalConfig(
    baseline_quantile=0.15,
    search_rounds=6,
    border_weight=0.0,
    random_seed=0,
    component_cap=0,
    fill_rounds=1,
)

ARTIFACT_KEYS = {
    "selection_commitment": PARENT_COMMITMENT_SHA256,
    "parent_frozen_eval": PARENT_FROZEN_EVAL_SHA256,
    "parent_frozen_eval_metadata": PARENT_FROZEN_METADATA_SHA256,
    "matcher_v3": V3_CHECKPOINT_SHA256,
    "matcher_local": LOCAL_CHECKPOINT_SHA256,
}

RUNTIME_SOURCE_PATHS = {
    "replay_runner": Path(__file__).resolve(),
    "taska_seam_matcher": PROJECT_ROOT / "src/aiijc_puzzle/taska_seam_matcher.py",
    "raw_tail_global_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
    "parent_pilot_runner": PROJECT_ROOT / "scripts/run_union_hard_edge_priority_pilot.py",
}


@dataclass(frozen=True)
class Artifacts:
    selection_commitment: Path
    parent_frozen_eval: Path
    parent_frozen_eval_metadata: Path
    matcher_v3: Path
    matcher_local: Path


@dataclass(frozen=True)
class RunPaths:
    frozen_eval: Path
    frozen_eval_metadata: Path
    pre_score_freeze: Path
    report: Path


@dataclass(frozen=True)
class DirtyCase:
    case_id: str
    source_filename: str
    dirty_tiles: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument(
        "--smoke-one",
        action="store_true",
        help="run the first registered case only; never emits a promotable full-panel status",
    )
    return parser.parse_args(argv)


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_path(value: Any, *, name: str) -> Path:
    raw = value.get("path") if isinstance(value, Mapping) else value
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{name} path is malformed")
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{name} does not exist: {path}")
    return path


def _record(path: Path) -> dict[str, str]:
    return {"path": _project_path(path), "sha256": sha256_file(path)}


def _names_digest(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _cases_digest(names: Sequence[str], draws: Sequence[int]) -> str:
    values = [f"{name}\0{int(draw)}" for name in names for draw in draws]
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _require_equal(config: Mapping[str, Any], dotted: str, expected: Any) -> None:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"preregistration field is absent: {dotted}")
        value = value[part]
    if value != expected:
        raise ValueError(f"preregistration field changed: {dotted}")


def _validate_recipe(config: Mapping[str, Any]) -> tuple[str, ...]:
    fixed: dict[str, Any] = {
        "schema": CONFIG_SCHEMA,
        "experiment": EXPERIMENT,
        "protocol.development_eval_panel_previously_opened": True,
        "protocol.single_candidate_arm": True,
        "protocol.hyperparameter_or_arm_sweep": False,
        "protocol.predictions_harvest_and_layouts_frozen_before_reference_recreation": True,
        "protocol.fresh_source_disjoint_confirmation_required_before_promotion": True,
        "matcher.kinds": list(MATCHER_KINDS),
        "matcher.views": list(VIEWS),
        "matcher.orientations": ORIENTATIONS,
        "matcher.fusion": "per-model-sinkhorn-then-elementwise-min",
        "matcher.sinkhorn_iterations": SINKHORN_ITERATIONS,
        "matcher.cycle_rounds": CYCLE_ROUNDS,
        "matcher.cycle_weight": CYCLE_WEIGHT,
        "matcher.acyclic_weight": ACYCLIC_WEIGHT,
        "matcher.sinkhorn_slack": SINKHORN_SLACK,
        "harvest.depth": HARVEST_DEPTH,
        "harvest.votes_fallback": VOTES_FALLBACK,
        "harvest.vote_target": VOTE_TARGET,
        "harvest.weighted": False,
        "harvest.margin": 0.0,
        "harvest.order": "raw_fused_score",
        "harvest.quad_weight": QUAD_WEIGHT,
        "harvest.historical_quad_0_4_excluded_as_target_id_dependent": True,
        "solver.name": "raw-tail-global",
        "solver.baseline_quantile": SOLVER_CONFIG.baseline_quantile,
        "solver.search_rounds": SOLVER_CONFIG.search_rounds,
        "solver.random_seed": SOLVER_CONFIG.random_seed,
        "solver.component_cap": SOLVER_CONFIG.component_cap,
        "solver.fill_rounds": SOLVER_CONFIG.fill_rounds,
        "solver.border_unary": False,
        "solver.border_weight": SOLVER_CONFIG.border_weight,
        "evaluation.arms": ["union_v2", "learned_priority", "taska_legal_raw_tail"],
        "evaluation.primary_metric": "satisfied_adjacent_pairs_per_board",
        "evaluation.pair_denominator": PAIR_DENOMINATOR,
        "evaluation.secondary_metrics": ["adjacency_recall", "exact_tiles_per_board"],
        "evaluation.full_panel_requires_exactly_32_cases": True,
        "evaluation.bootstrap_intervals_are_reported_but_not_used_for_tuning": True,
        "legality.organizer_train_only": True,
        "legality.dirty_tiles_only_for_candidate_inference": True,
        "legality.restored_or_denoised_pixels_emitted": False,
        "legality.target_ids_or_exact_references_in_candidate_inference": False,
        "legality.chooser_used": False,
        "legality.verifier_used": False,
        "legality.border_prior_used": False,
        "legality.index_derived_boundary_mask_used": False,
        "legality.output_uses_each_original_upright_20x20_tile_exactly_once": True,
        "legality.competition_test_forbidden": True,
    }
    for name, expected in fixed.items():
        _require_equal(config, name, expected)

    panel = config.get("panel")
    if not isinstance(panel, Mapping):
        raise ValueError("preregistration needs an explicit panel mapping")
    names = panel.get("source_filenames")
    if not isinstance(names, list) or not all(
        isinstance(name, str) and Path(name).name == name and name.endswith(".png")
        for name in names
    ):
        raise ValueError("panel source_filenames are malformed")
    source_names = tuple(names)
    if len(source_names) != EVAL_SOURCE_COUNT or len(set(source_names)) != len(source_names):
        raise ValueError("panel must contain exactly 16 unique sources")
    if panel.get("draws") != list(DRAWS) or panel.get("case_count") != EVAL_CASE_COUNT:
        raise ValueError("panel draw expansion changed")
    if panel.get("source_order_digest") != EVAL_SOURCE_ORDER_DIGEST:
        raise ValueError("panel source order digest changed")
    if panel.get("cases_digest") != EVAL_CASES_DIGEST:
        raise ValueError("panel case digest changed")
    if _names_digest(source_names) != EVAL_SOURCE_ORDER_DIGEST:
        raise ValueError("panel source names do not match their digest")
    if _cases_digest(source_names, DRAWS) != EVAL_CASES_DIGEST:
        raise ValueError("panel cases do not match their digest")
    return source_names


def _load_preregistration(path: Path) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    path = path.resolve()
    sidecar = Path(f"{path}.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("preregistration and its .sha256 sidecar must both exist")
    config_sha = sha256_file(path)
    tokens = sidecar.read_text(encoding="utf-8").split()
    if not tokens or tokens[0] != config_sha:
        raise ValueError("preregistration sidecar does not match config bytes")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("preregistration root must be a mapping")
    return payload, config_sha, _validate_recipe(payload)


def _validate_record(
    records: Mapping[str, Any],
    name: str,
    *,
    expected_sha256: str | None = None,
    expected_path: Path | None = None,
) -> Path:
    value = records.get(name)
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} needs explicit path and sha256")
    path = _resolve_path(value, name=name)
    observed = sha256_file(path)
    if value.get("sha256") != observed:
        raise ValueError(f"{name} SHA-256 mismatch")
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"{name} differs from its frozen artifact")
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"{name} path differs from its frozen runtime source")
    return path


def _validate_artifacts(config: Mapping[str, Any]) -> Artifacts:
    records = config.get("artifacts")
    if not isinstance(records, Mapping) or set(records) != set(ARTIFACT_KEYS):
        raise ValueError("artifact roster changed")
    paths = {
        name: _validate_record(records, name, expected_sha256=expected)
        for name, expected in ARTIFACT_KEYS.items()
    }
    runtime = config.get("runtime_sources")
    if not isinstance(runtime, Mapping) or set(runtime) != set(RUNTIME_SOURCE_PATHS):
        raise ValueError("runtime source roster changed")
    for name, expected_path in RUNTIME_SOURCE_PATHS.items():
        _validate_record(runtime, name, expected_path=expected_path)
    return Artifacts(**paths)


def _validate_parent(
    artifacts: Artifacts,
    source_names: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    commitment = json.loads(artifacts.selection_commitment.read_text(encoding="utf-8"))
    if commitment.get("schema") != "aiijc-union-hard-priority-pilot-v1":
        raise ValueError("parent commitment schema changed")
    if commitment.get("synthetic_seed") != SYNTHETIC_SEED:
        raise ValueError("parent synthetic seed changed")
    evaluation = commitment.get("eval")
    if not isinstance(evaluation, Mapping):
        raise ValueError("parent eval roster is absent")
    if tuple(evaluation.get("source_filenames", ())) != tuple(source_names):
        raise ValueError("candidate panel differs from parent eval roster")
    if evaluation.get("draws") != list(DRAWS):
        raise ValueError("parent eval draws changed")
    if evaluation.get("source_order_digest") != EVAL_SOURCE_ORDER_DIGEST:
        raise ValueError("parent source order digest changed")
    if evaluation.get("cases_digest") != EVAL_CASES_DIGEST:
        raise ValueError("parent case digest changed")

    metadata = json.loads(artifacts.parent_frozen_eval_metadata.read_text(encoding="utf-8"))
    fixed = {
        "schema": "aiijc-union-hard-edge-frozen-target-free-eval-v1",
        "contains_exact_references_or_labels": False,
        "contains_strict_original_tile_layouts": True,
        "strict_layout_count_per_arm": EVAL_CASE_COUNT,
    }
    if any(metadata.get(name) != value for name, value in fixed.items()):
        raise ValueError("parent frozen eval metadata contract changed")
    rows = metadata.get("rows")
    if not isinstance(rows, list) or len(rows) != EVAL_CASE_COUNT:
        raise ValueError("parent frozen eval row count changed")
    expected = [
        (source_name, draw)
        for source_name in source_names
        for draw in DRAWS
    ]
    for index, (row, (source_name, draw)) in enumerate(zip(rows, expected, strict=True)):
        if row.get("prefix") != f"case_{index:04d}":
            raise ValueError("parent frozen eval prefix order changed")
        if row.get("source_filename") != source_name or row.get("draw_index") != draw:
            raise ValueError("parent frozen eval case roster changed")
        if not isinstance(row.get("case_id"), str) or not isinstance(row.get("dirty_sha256"), str):
            raise ValueError("parent frozen eval row identity is malformed")
    return commitment, [dict(row) for row in rows]


def _manifest_lookup(commitment: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    record = commitment.get("manifest")
    if not isinstance(record, Mapping) or set(record) < {"path", "sha256", "split"}:
        raise ValueError("parent commitment manifest record is malformed")
    path = _resolve_path(record, name="parent_manifest")
    if sha256_file(path) != record["sha256"] or record["split"] != "train":
        raise ValueError("parent organizer-train manifest changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("splits", {}).get("train")
    if not isinstance(rows, list) or len(rows) != 5_600:
        raise ValueError("parent organizer-train manifest rows changed")
    lookup = {str(row["filename"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("parent organizer-train manifest has duplicate filenames")
    return lookup


def _case_specs(
    source_names: Sequence[str],
    lookup: Mapping[str, Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], str, int]]:
    specs = [(lookup[name], name, draw) for name in source_names for draw in DRAWS]
    if len(specs) != EVAL_CASE_COUNT:
        raise RuntimeError("eval panel expansion changed")
    return specs


def _select_device(name: str, *, allow_nondeterministic_mps: bool) -> torch.device:
    if name == "mps":
        if not allow_nondeterministic_mps:
            raise ValueError("MPS requires --allow-nondeterministic-mps")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        torch.use_deterministic_algorithms(False)
        return torch.device("mps")
    if name != "cpu":
        raise ValueError("device must be cpu or mps")
    if allow_nondeterministic_mps:
        raise ValueError("--allow-nondeterministic-mps requires MPS")
    torch.use_deterministic_algorithms(True)
    return torch.device("cpu")


def _prepare_run_paths(output_dir: Path) -> RunPaths:
    root = output_dir.resolve()
    paths = RunPaths(
        frozen_eval=root / "frozen-target-free-eval.npz",
        frozen_eval_metadata=root / "frozen-target-free-eval.json",
        pre_score_freeze=root / "pre-score-freeze.json",
        report=root / "report.json",
    )
    if any(path.exists() for path in asdict(paths).values()):
        raise FileExistsError("refusing to overwrite a TASKA replay run")
    root.mkdir(parents=True, exist_ok=True)
    return paths


def _strict_layout(value: Any) -> np.ndarray:
    layout = np.ascontiguousarray(value, dtype=np.int32)
    if layout.shape != (COUNT,) or not np.array_equal(np.sort(layout), np.arange(COUNT)):
        raise ValueError("layout is not a strict original-tile permutation")
    return layout


def _dirty_case(
    cache: Any,
    record: Mapping[str, Any],
    source_name: str,
    draw: int,
) -> DirtyCase:
    # ``prepare_case`` necessarily creates the synthetic corruption from the
    # organizer clean train image.  The exact mapping is discarded inside this
    # narrow helper and is neither returned nor persisted before the freeze.
    case = prepare_case(cache, record, draw_index=draw, seed=SYNTHETIC_SEED)
    tiles = np.ascontiguousarray(case.dirty_tiles)
    if tiles.shape != (COUNT, 20, 20, 3) or tiles.dtype != np.uint8:
        raise ValueError("dirty tiles violate the 576x20x20 RGB contract")
    if str(case.source_filename) != source_name:
        raise RuntimeError("synthetic case source filename changed")
    return DirtyCase(str(case.case_id), source_name, tiles)


def _load_matchers(artifacts: Artifacts, *, device: torch.device) -> tuple[Any, ...]:
    from aiijc_puzzle.taska_seam_matcher import load_default_taska_ensemble

    if artifacts.matcher_v3.parent != artifacts.matcher_local.parent:
        raise ValueError("TASKA v3/local checkpoints must share one audited directory")
    return load_default_taska_ensemble(artifacts.matcher_v3.parent, device=device)


def _match_dirty_tiles(
    tiles: np.ndarray,
    matchers: Sequence[Any],
    *,
    device: torch.device,
) -> Any:
    from aiijc_puzzle.taska_seam_matcher import TaskaSeamConfig, match_taska_tiles

    return match_taska_tiles(
        tiles,
        matchers,
        config=TaskaSeamConfig(
            views=VIEWS,
            orientations=ORIENTATIONS,
            votes=VOTES_FALLBACK,
            vote_target=VOTE_TARGET,
            margin=0.0,
            depth=HARVEST_DEPTH,
            quad_weight=QUAD_WEIGHT,
            rounds=CYCLE_ROUNDS,
            cycle_weight=CYCLE_WEIGHT,
            sinkhorn_iterations=SINKHORN_ITERATIONS,
            acyclic_weight=ACYCLIC_WEIGHT,
        ),
        device=device,
    )


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    matrix = np.asarray(current, dtype=np.float64)
    if matrix.shape != (COUNT, COUNT):
        raise ValueError(f"{name} must have shape {(COUNT, COUNT)}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(matrix)


def _edge_value(values: Any, edge: RawTailEdge, index: int, *, default: float) -> float:
    if values is None:
        return default
    if isinstance(values, Mapping):
        for key in (edge, (edge.source, edge.target, edge.axis)):
            if key in values:
                return float(values[key])
        return default
    array = np.asarray(values)
    if array.ndim != 1 or index >= len(array):
        raise ValueError("edge diagnostics are not aligned to candidate_edges")
    return float(array[index])


def _diagnostics_dict(value: Any) -> dict[str, Any]:
    diagnostics = getattr(value, "diagnostics", None)
    if diagnostics is None:
        return {}
    if hasattr(diagnostics, "as_dict"):
        diagnostics = diagnostics.as_dict()
    elif is_dataclass(diagnostics):
        diagnostics = asdict(diagnostics)
    if not isinstance(diagnostics, Mapping):
        return {"repr": repr(diagnostics)}
    return json.loads(json.dumps(dict(diagnostics), default=str))


def _vote_record_maps(value: Any) -> tuple[dict[RawTailEdge, float], dict[RawTailEdge, int]]:
    records = getattr(value, "vote_records", ())
    weights: dict[RawTailEdge, float] = {}
    counts: dict[RawTailEdge, int] = {}
    for record in records:
        edge = getattr(record, "edge", None)
        if not isinstance(edge, RawTailEdge):
            raise TypeError("vote_records must contain RawTailEdge identities")
        weights[edge] = float(record.minimum_margin)
        counts[edge] = int(record.vote_count)
    return weights, counts


def _freeze_target_free_eval(
    paths: RunPaths,
    specs: Sequence[tuple[Mapping[str, Any], str, int]],
    parent_rows: Sequence[Mapping[str, Any]],
    artifacts: Artifacts,
    *,
    targets: Path,
    device: torch.device,
) -> tuple[float, int]:
    if len(specs) != len(parent_rows):
        raise ValueError("candidate and parent row rosters differ")
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    matchers = _load_matchers(artifacts, device=device)
    target_cache = CleanTileCache(targets)
    started = perf_counter()
    for index, ((record, source_name, draw), parent_row) in enumerate(
        zip(specs, parent_rows, strict=True)
    ):
        dirty = _dirty_case(target_cache, record, source_name, draw)
        dirty_sha = _dirty_sha256(dirty.dirty_tiles)
        if (
            dirty.case_id != parent_row["case_id"]
            or source_name != parent_row["source_filename"]
            or draw != int(parent_row["draw_index"])
            or dirty_sha != parent_row["dirty_sha256"]
        ):
            raise RuntimeError("candidate replay recreated a different synthetic case")

        matched = _match_dirty_tiles(dirty.dirty_tiles, matchers, device=device)
        cost_right = _finite_matrix(matched.cost_right, name="cost_right")
        cost_down = _finite_matrix(matched.cost_down, name="cost_down")
        right_log = _finite_matrix(matched.right_log, name="right_log")
        down_log = _finite_matrix(matched.down_log, name="down_log")
        edges = tuple(matched.candidate_edges)
        if not all(isinstance(edge, RawTailEdge) for edge in edges):
            raise TypeError("candidate_edges must contain RawTailEdge values")
        edge_weights, vote_counts = _vote_record_maps(matched)
        if set(edge_weights) != set(edges) or set(vote_counts) != set(edges):
            raise ValueError("vote_records and candidate_edges differ")
        if int(matched.scorer_count) != len(MATCHER_KINDS) * len(VIEWS) * ORIENTATIONS:
            raise ValueError("TASKA scorer count differs from the fixed recipe")
        if not 1 <= int(matched.chosen_vote_threshold) <= int(matched.scorer_count):
            raise ValueError("TASKA dynamic vote threshold is malformed")
        if set(matched.checkpoint_sha256) != {
            V3_CHECKPOINT_SHA256,
            LOCAL_CHECKPOINT_SHA256,
        }:
            raise ValueError("TASKA result checkpoint provenance changed")
        solver = solve_raw_tail_global(
            cost_right,
            cost_down,
            edges,
            border_unary=None,
            grid=GRID,
            config=SOLVER_CONFIG,
        )
        layout = _strict_layout(solver.layout)
        prefix = f"case_{index:04d}"
        arrays[f"{prefix}__cost_right"] = cost_right.astype(np.float32)
        arrays[f"{prefix}__cost_down"] = cost_down.astype(np.float32)
        arrays[f"{prefix}__right_log"] = right_log.astype(np.float32)
        arrays[f"{prefix}__down_log"] = down_log.astype(np.float32)
        arrays[f"{prefix}__edge_source"] = np.asarray(
            [edge.source for edge in edges], dtype=np.int32
        )
        arrays[f"{prefix}__edge_target"] = np.asarray(
            [edge.target for edge in edges], dtype=np.int32
        )
        arrays[f"{prefix}__edge_axis"] = np.asarray(
            [0 if edge.axis == "right" else 1 for edge in edges], dtype=np.uint8
        )
        arrays[f"{prefix}__edge_weight"] = np.asarray(
            [
                _edge_value(edge_weights, edge, rank, default=0.0)
                for rank, edge in enumerate(edges)
            ],
            dtype=np.float32,
        )
        arrays[f"{prefix}__edge_vote_count"] = np.asarray(
            [
                round(
                    _edge_value(
                        vote_counts,
                        # The audited frontend keeps one aligned MutualVote
                        # record per candidate; no score is inferred here.
                        edge,
                        rank,
                        default=0.0,
                    )
                )
                for rank, edge in enumerate(edges)
            ],
            dtype=np.int16,
        )
        arrays[f"{prefix}__taska_layout"] = layout
        rows.append(
            {
                "prefix": prefix,
                "case_id": dirty.case_id,
                "source_filename": source_name,
                "draw_index": draw,
                "dirty_sha256": dirty_sha,
                "candidate_edge_count": len(edges),
                "chosen_vote_threshold": int(matched.chosen_vote_threshold),
                "scorer_count": int(matched.scorer_count),
                "checkpoint_sha256": list(matched.checkpoint_sha256),
                "matcher_diagnostics": _diagnostics_dict(matched),
                "solver_diagnostics": solver.diagnostics.as_dict(),
            }
        )
        print(
            json.dumps(
                {
                    "event": "taska_target_free_case_frozen_in_memory",
                    "case": index + 1,
                    "case_count": len(specs),
                    "source_filename": source_name,
                    "draw_index": draw,
                    "harvest_edges": len(edges),
                    "strict": True,
                }
            ),
            flush=True,
        )

    _write_npz_exclusive(paths.frozen_eval, arrays)
    _write_json_exclusive(
        paths.frozen_eval_metadata,
        {
            "schema": FROZEN_SCHEMA,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "contains_dirty_derived_scores": True,
            "contains_frozen_harvest_membership": True,
            "contains_strict_original_tile_layouts": True,
            "quad_weight": QUAD_WEIGHT,
            "border_prior_used": False,
            "rows": rows,
        },
    )
    return perf_counter() - started, len(rows)


def _freeze_pre_score_roster(
    paths: RunPaths,
    artifacts: Artifacts,
    *,
    config_path: Path,
    config_sha256: str,
    device: torch.device,
    allow_nondeterministic_mps: bool,
) -> dict[str, dict[str, str]]:
    sidecar = Path(f"{config_path.resolve()}.sha256")
    frozen = {
        "config": {"path": _project_path(config_path), "sha256": config_sha256},
        "config_sidecar": _record(sidecar),
        **{name: _record(getattr(artifacts, name)) for name in ARTIFACT_KEYS},
        **{name: _record(path) for name, path in RUNTIME_SOURCE_PATHS.items()},
        "frozen_target_free_eval": _record(paths.frozen_eval),
        "frozen_target_free_eval_metadata": _record(paths.frozen_eval_metadata),
    }
    _write_json_exclusive(
        paths.pre_score_freeze,
        {
            "schema": "aiijc-taska-seam-pre-score-freeze-v1",
            "created_before_eval_reference_recreation": True,
            "contains_evaluation_references_or_labels": False,
            "already_opened_development_panel": True,
            "freshness_claimed": False,
            "device": str(device),
            "nondeterministic_mps_explicitly_allowed": allow_nondeterministic_mps,
            "artifacts": frozen,
        },
    )
    return frozen


def _validate_pre_score_freeze(
    paths: RunPaths,
    expected: Mapping[str, Mapping[str, str]],
) -> None:
    payload = json.loads(paths.pre_score_freeze.read_text(encoding="utf-8"))
    if payload.get("created_before_eval_reference_recreation") is not True:
        raise RuntimeError("pre-score freeze timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze unexpectedly contains evaluation labels")
    if payload.get("artifacts") != expected:
        raise RuntimeError("pre-score frozen artifact roster changed")
    for name, record in expected.items():
        path = _resolve_path(record, name=f"pre_score_{name}")
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"pre-score artifact changed: {name}")


def _layout_metrics(evaluation: LayoutEvaluation) -> dict[str, Any]:
    if evaluation.adjacency_total != PAIR_DENOMINATOR:
        raise RuntimeError("adjacency denominator changed")
    pair_count = int(evaluation.adjacency_correct)
    recall = float(evaluation.adjacency)
    if pair_count != round(recall * evaluation.adjacency_total):
        raise RuntimeError("integer adjacency count and recall disagree")
    return {
        "satisfied_adjacent_pairs": pair_count,
        "adjacency_recall": recall,
        "exact_tiles": int(evaluation.correct_tile_count),
        "strict_permutation": True,
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]], *, full_panel: bool) -> dict[str, Any]:
    arms = ("union_v2", "learned_priority", "taska_legal_raw_tail")
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    summary: dict[str, Any] = {
        "arms": {
            arm: {
                metric: float(np.mean([float(row[arm][metric]) for row in rows]))
                for metric in metrics
            }
            for arm in arms
        }
    }
    sources = [str(row["source_filename"]) for row in rows]
    deltas: dict[str, Any] = {}
    for baseline_index, baseline in enumerate(("learned_priority", "union_v2")):
        deltas[baseline] = {}
        for metric_index, metric in enumerate(metrics):
            values = [
                float(row["taska_legal_raw_tail"][metric]) - float(row[baseline][metric])
                for row in rows
            ]
            if full_panel:
                deltas[baseline][metric] = source_clustered_delta_ci(
                    values,
                    sources,
                    seed=BOOTSTRAP_SEED + baseline_index * 10 + metric_index,
                )
            else:
                deltas[baseline][metric] = {
                    "mean": float(np.mean(values)),
                    "ci95_lower": None,
                    "ci95_upper": None,
                    "smoke_only": True,
                }
    summary["candidate_deltas"] = deltas
    return summary


def _score_frozen_eval(
    artifacts: Artifacts,
    paths: RunPaths,
    commitment: Mapping[str, Any],
    expected_hashes: Mapping[str, Mapping[str, str]],
    *,
    targets: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, bool]]:
    # This must remain the first operation: no target cache or target path is
    # touched until every dirty-only prediction artifact has been hash-frozen.
    _validate_pre_score_freeze(paths, expected_hashes)
    candidate_metadata = json.loads(paths.frozen_eval_metadata.read_text(encoding="utf-8"))
    parent_metadata = json.loads(artifacts.parent_frozen_eval_metadata.read_text(encoding="utf-8"))
    candidate_rows = candidate_metadata["rows"]
    parent_rows = parent_metadata["rows"][: len(candidate_rows)]
    lookup = _manifest_lookup(commitment)
    target_cache = CleanTileCache(targets)
    rows: list[dict[str, Any]] = []
    with (
        np.load(artifacts.parent_frozen_eval) as parent_archive,
        np.load(paths.frozen_eval) as candidate_archive,
    ):
        for candidate_row, parent_row in zip(candidate_rows, parent_rows, strict=True):
            identity_fields = ("prefix", "case_id", "source_filename", "draw_index", "dirty_sha256")
            if any(candidate_row[name] != parent_row[name] for name in identity_fields):
                raise RuntimeError("candidate and parent frozen row identities differ")
            source_name = str(candidate_row["source_filename"])
            draw = int(candidate_row["draw_index"])
            case = prepare_case(
                target_cache,
                lookup[source_name],
                draw_index=draw,
                seed=SYNTHETIC_SEED,
            )
            if (
                str(case.case_id) != candidate_row["case_id"]
                or _dirty_sha256(case.dirty_tiles) != candidate_row["dirty_sha256"]
            ):
                raise RuntimeError("eval scoring recreated a different synthetic case")
            reference = _strict_layout(np.argsort(case.input_tile_to_position))
            prefix = str(candidate_row["prefix"])
            layouts = {
                "union_v2": _strict_layout(parent_archive[f"{prefix}__union_v2_layout"]),
                "learned_priority": _strict_layout(
                    parent_archive[f"{prefix}__learned_priority_layout"]
                ),
                "taska_legal_raw_tail": _strict_layout(
                    candidate_archive[f"{prefix}__taska_layout"]
                ),
            }
            row: dict[str, Any] = {
                "source_filename": source_name,
                "draw_index": draw,
                "case_id": str(case.case_id),
            }
            for arm, layout in layouts.items():
                row[arm] = _layout_metrics(
                    evaluate_layout(layout, reference, reference_is_exact=True)
                )
            rows.append(row)

    full_panel = len(rows) == EVAL_CASE_COUNT
    metrics = _summarize_rows(rows, full_panel=full_panel)
    candidate = metrics["arms"]["taska_legal_raw_tail"]
    gate = {
        "full_registered_32_case_panel": full_panel,
        "all_layouts_strict": all(
            bool(row[arm]["strict_permutation"])
            for row in rows
            for arm in ("union_v2", "learned_priority", "taska_legal_raw_tail")
        ),
        "candidate_pairs_exceed_union_v2": (
            candidate["satisfied_adjacent_pairs"]
            > metrics["arms"]["union_v2"]["satisfied_adjacent_pairs"]
        ),
        "candidate_pairs_exceed_learned_priority": (
            candidate["satisfied_adjacent_pairs"]
            > metrics["arms"]["learned_priority"]["satisfied_adjacent_pairs"]
        ),
    }
    gate["passed"] = all(gate.values())
    return rows, metrics, gate


def run(args: argparse.Namespace) -> None:
    config, config_sha, source_names = _load_preregistration(args.config)
    artifacts = _validate_artifacts(config)
    commitment, parent_rows = _validate_parent(artifacts, source_names)
    lookup = _manifest_lookup(commitment)
    specs = _case_specs(source_names, lookup)
    if args.smoke_one:
        specs = specs[:1]
        parent_rows = parent_rows[:1]
    paths = _prepare_run_paths(args.output_dir)
    device = _select_device(
        args.device,
        allow_nondeterministic_mps=args.allow_nondeterministic_mps,
    )
    started = perf_counter()
    inference_seconds, frozen_case_count = _freeze_target_free_eval(
        paths,
        specs,
        parent_rows,
        artifacts,
        targets=args.targets,
        device=device,
    )
    frozen = _freeze_pre_score_roster(
        paths,
        artifacts,
        config_path=args.config,
        config_sha256=config_sha,
        device=device,
        allow_nondeterministic_mps=bool(args.allow_nondeterministic_mps),
    )
    print(
        json.dumps(
            {
                "event": "taska_scores_harvest_and_layouts_frozen_before_scoring",
                "frozen_eval_sha256": sha256_file(paths.frozen_eval),
                "frozen_eval_metadata_sha256": sha256_file(paths.frozen_eval_metadata),
                "pre_score_freeze_sha256": sha256_file(paths.pre_score_freeze),
                "case_count": frozen_case_count,
                "target_reference_persisted": False,
            }
        ),
        flush=True,
    )
    rows, metrics, gate = _score_frozen_eval(
        artifacts,
        paths,
        commitment,
        frozen,
        targets=args.targets,
    )
    _write_json_exclusive(
        paths.report,
        {
            "schema": REPORT_SCHEMA,
            "status": (
                "smoke-only"
                if args.smoke_one
                else "development-replay-pass"
                if gate["passed"]
                else "development-replay-fail"
            ),
            "experiment": EXPERIMENT,
            "panel": {
                "previously_opened": True,
                "freshness_claimed": False,
                "source_disjoint_confirmation_required_before_promotion": True,
                "registered_source_count": EVAL_SOURCE_COUNT,
                "registered_case_count": EVAL_CASE_COUNT,
                "evaluated_case_count": len(rows),
                "smoke_only": bool(args.smoke_one),
                "source_order_digest": EVAL_SOURCE_ORDER_DIGEST,
                "cases_digest": EVAL_CASES_DIGEST,
            },
            "preregistration": {"path": _project_path(args.config), "sha256": config_sha},
            "device": {
                "value": str(device),
                "nondeterministic_mps_explicitly_allowed": bool(
                    args.allow_nondeterministic_mps
                ),
                "determinism_claimed": device.type != "mps",
            },
            "candidate": {
                "single_fixed_arm": True,
                "hyperparameter_or_arm_sweep": False,
                "matcher_kinds": list(MATCHER_KINDS),
                "views": list(VIEWS),
                "orientations": ORIENTATIONS,
                "vote_target": VOTE_TARGET,
                "votes_fallback": VOTES_FALLBACK,
                "depth": HARVEST_DEPTH,
                "quad_weight": QUAD_WEIGHT,
                "historical_quad_0_4_excluded_as_target_id_dependent": True,
                "chooser_used": False,
                "verifier_used": False,
                "border_prior_used": False,
                "solver": asdict(SOLVER_CONFIG),
            },
            "artifacts": {name: _record(getattr(artifacts, name)) for name in ARTIFACT_KEYS},
            "runtime_sources": {
                name: _record(path) for name, path in RUNTIME_SOURCE_PATHS.items()
            },
            "frozen_eval": {
                "archive": _record(paths.frozen_eval),
                "metadata": _record(paths.frozen_eval_metadata),
                "pre_score_freeze": _record(paths.pre_score_freeze),
                "scores_harvest_and_layouts_frozen_before_references": True,
                "contains_exact_references_or_labels": False,
            },
            "metrics": metrics,
            "gate": gate,
            "rows": rows,
            "runtime_seconds": {
                "target_free_inference": inference_seconds,
                "total": perf_counter() - started,
            },
            "legality": {
                "organizer_train_only": True,
                "dirty_tiles_only_for_candidate_inference": True,
                "restored_pixels_emitted": False,
                "target_ids_or_references_used_by_candidate": False,
                "index_derived_boundary_mask_used": False,
                "chooser_verifier_or_border_prior_used": False,
                "original_upright_tile_permutations_only": True,
                "competition_test_accessed": False,
            },
        },
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
