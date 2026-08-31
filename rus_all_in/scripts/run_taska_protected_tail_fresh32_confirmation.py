#!/usr/bin/env python3
"""Confirm two precommitted protected-tail budgets on a current-disjoint panel.

The 16-source, two-draw roster is selected deterministically from
``img_006700..img_006999`` after excluding both the earlier held16 roster and
the opened32 development roster.  The experiment is deliberately limited to
three arms fixed before any selected target is read:

* the frozen legal raw TASKA layout;
* the unchanged protected-tail polish with ``max_swaps=24``;
* one evidence-driven extension with ``max_swaps=96`` because every prior
  held case saturated the 24-swap cap.

This is not a budget sweep.  Dirty-derived matrices, harvested edges, all
three strict layouts, and polish diagnostics are written and hash-frozen
before exact synthetic references are recreated.  The source range remains
historically model-selection-exposed, so the report is a current-disjoint
confirmation and never a formal fresh promotion claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import PIL
import torch
from PIL import Image

from aiijc_puzzle.layout_evaluation import LayoutEvaluation, evaluate_layout
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file, split_tiles
from aiijc_puzzle.raw_tail_global_solver import (
    RawTailEdge,
    RawTailGlobalConfig,
    solve_raw_tail_global,
)
from aiijc_puzzle.restoration_r6 import distort_tiles
from aiijc_puzzle.synthetic_socket_evaluation import (
    DEFAULT_SYNTHETIC_NAMESPACE,
    make_exact_synthetic_case,
)
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail
from aiijc_puzzle.taska_seam_matcher import (
    TaskaSeamConfig,
    load_default_taska_ensemble,
    match_taska_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "aiijc-taska-protected-tail-fresh32-confirmation-v1"
REPORT_SCHEMA = "aiijc-taska-protected-tail-fresh32-confirmation-report-v1"
FROZEN_SCHEMA = "aiijc-taska-protected-tail-fresh32-target-free-v1"
FREEZE_SCHEMA = "aiijc-taska-protected-tail-fresh32-pre-score-freeze-v1"
EXPERIMENT = "taska-protected-tail-fresh32-confirmation-v1"

DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_protected_tail_fresh32_confirmation_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-protected-tail/fresh-held32-mps-v1"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"

GRID = 24
COUNT = GRID * GRID
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
SOURCE_MINIMUM = 6_700
SOURCE_MAXIMUM = 6_999
SOURCE_COUNT = 16
DRAWS = (0, 1)
CASE_COUNT = SOURCE_COUNT * len(DRAWS)
SELECTION_NAMESPACE = "aiijc-taska-protected-tail-fresh32-confirmation-v1-source16xdraw2"
SELECTION_SEED = 2_026_083_101
SELECTION_ALGORITHM = (
    "exclude committed held16 and opened32 names, then "
    "sha256(namespace\\0seed\\0filename) ascending, filename tie-break"
)
SYNTHETIC_SEED = 1_267_233_517
BOOTSTRAP_SEED = 1_331_947_281
BOOTSTRAP_RESAMPLES = 20_000
ARM_BUDGETS = {"taska_legal_raw_tail": 0, "protected_tail_24": 24, "protected_tail_96": 96}

FULL_RANGE_DIGEST = "8f736ea53d6bb377c82b0286d51381d2305365266dd02286fb417a9d028729d5"
EXCLUSION_DIGEST = "5ce2e6b973f2d512dab7f045bebb673498cc8070e989a9acb25dc46ba9977fdb"
ELIGIBLE_DIGEST = "f678696bc1c18171b420de53c00066f16e60618bf0daac25c39ed40a3675d7ee"
SOURCE_ORDER_DIGEST = "b9b19c09ecb05587ae7b9ca8c1c93ad00d8fbbee77cf378e647f0f9db31c99b2"
CASES_DIGEST = "93cbfc8acc8c18b009d7b116c2c4d874f980ddf7d218dc7ace6978ac8bab0936"

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

ARTIFACT_NAMES = (
    "manifest",
    "opened32_recipe",
    "held300_recipe",
    "matcher_v3",
    "matcher_local",
)
RUNTIME_SOURCE_PATHS = {
    "confirmation_runner": Path(__file__).resolve(),
    "layout_evaluation": PROJECT_ROOT / "src/aiijc_puzzle/layout_evaluation.py",
    "protocol": PROJECT_ROOT / "src/aiijc_puzzle/protocol.py",
    "raw_tail_global_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
    "restoration_corruption": PROJECT_ROOT / "src/aiijc_puzzle/restoration_r6.py",
    "synthetic_case_generator": PROJECT_ROOT / "src/aiijc_puzzle/synthetic_socket_evaluation.py",
    "taska_layout_portfolio": PROJECT_ROOT / "src/aiijc_puzzle/taska_layout_portfolio.py",
    "taska_protected_tail_polish": PROJECT_ROOT / "src/aiijc_puzzle/taska_protected_tail_polish.py",
    "taska_seam_matcher": PROJECT_ROOT / "src/aiijc_puzzle/taska_seam_matcher.py",
}


@dataclass(frozen=True)
class Artifacts:
    manifest: Path
    opened32_recipe: Path
    held300_recipe: Path
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
    draw_index: int
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
        help="evaluate only the first committed case; no panel inference is claimed",
    )
    return parser.parse_args(argv)


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _record(path: Path) -> dict[str, str]:
    return {"path": _project_path(path), "sha256": sha256_file(path)}


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


def _names_digest(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _cases_digest(names: Sequence[str]) -> str:
    values = [f"{name}\0{draw}" for name in names for draw in DRAWS]
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _source_index(name: str) -> int:
    if len(name) != len("img_000000.png") or not name.startswith("img_"):
        raise ValueError(f"malformed source filename: {name}")
    return int(name[4:10])


def _full_range_names() -> tuple[str, ...]:
    return tuple(f"img_{index:06d}.png" for index in range(SOURCE_MINIMUM, SOURCE_MAXIMUM + 1))


def _reference_rosters(artifacts: Artifacts) -> tuple[tuple[str, ...], tuple[str, ...]]:
    opened = json.loads(artifacts.opened32_recipe.read_text(encoding="utf-8"))
    held = json.loads(artifacts.held300_recipe.read_text(encoding="utf-8"))
    opened_names = opened.get("panel", {}).get("source_filenames")
    held_names = held.get("panel", {}).get("source_filenames")
    if not isinstance(opened_names, list) or not all(
        isinstance(name, str) for name in opened_names
    ):
        raise ValueError("opened32 source roster is malformed")
    if not isinstance(held_names, list) or not all(isinstance(name, str) for name in held_names):
        raise ValueError("current held16 source roster is malformed")
    return tuple(opened_names), tuple(held_names)


def _eligible_names(opened_names: Sequence[str], held_names: Sequence[str]) -> tuple[str, ...]:
    excluded = set(opened_names) | set(held_names)
    return tuple(name for name in _full_range_names() if name not in excluded)


def _deterministic_source_roster(
    opened_names: Sequence[str],
    held_names: Sequence[str],
) -> tuple[str, ...]:
    prefix = f"{SELECTION_NAMESPACE}\0{SELECTION_SEED}\0".encode()
    eligible = _eligible_names(opened_names, held_names)
    return tuple(
        sorted(
            eligible,
            key=lambda name: (hashlib.sha256(prefix + name.encode()).digest(), name),
        )[:SOURCE_COUNT]
    )


def _write_json_exclusive(path: Path, payload: Any) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
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


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    sidecar = Path(f"{path}.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("preregistration and its .sha256 sidecar must both exist")
    digest = sha256_file(path)
    tokens = sidecar.read_text(encoding="utf-8").split()
    if not tokens or tokens[0] != digest:
        raise ValueError("preregistration sidecar does not match config bytes")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("preregistration root must be a mapping")
    return payload, digest


def _validate_record(
    records: Mapping[str, Any],
    name: str,
    *,
    expected_path: Path | None = None,
) -> Path:
    value = records.get(name)
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} needs explicit path and sha256")
    path = _resolve_path(value, name=name)
    if value.get("sha256") != sha256_file(path):
        raise ValueError(f"{name} SHA-256 mismatch")
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"{name} path differs from its frozen runtime source")
    return path


def _validate_artifacts(config: Mapping[str, Any]) -> Artifacts:
    records = config.get("artifacts")
    if not isinstance(records, Mapping) or set(records) != set(ARTIFACT_NAMES):
        raise ValueError("artifact roster changed")
    paths = {name: _validate_record(records, name) for name in ARTIFACT_NAMES}
    runtime = config.get("runtime_sources")
    if not isinstance(runtime, Mapping) or set(runtime) != set(RUNTIME_SOURCE_PATHS):
        raise ValueError("runtime source roster changed")
    for name, expected_path in RUNTIME_SOURCE_PATHS.items():
        _validate_record(runtime, name, expected_path=expected_path)
    return Artifacts(**paths)


def _fixed_recipe_dicts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    matcher = {
        "kinds": ["v3", "local"],
        "views": list(MATCHER_CONFIG.views),
        "orientations": MATCHER_CONFIG.orientations,
        "fusion": "per-model-sinkhorn-then-elementwise-min",
        "sinkhorn_iterations": MATCHER_CONFIG.sinkhorn_iterations,
        "cycle_rounds": MATCHER_CONFIG.rounds,
        "cycle_weight": MATCHER_CONFIG.cycle_weight,
        "acyclic_weight": MATCHER_CONFIG.acyclic_weight,
        "sinkhorn_slack": 0,
    }
    harvest = {
        "depth": MATCHER_CONFIG.depth,
        "votes_fallback": MATCHER_CONFIG.votes,
        "vote_target": MATCHER_CONFIG.vote_target,
        "weighted": False,
        "margin": MATCHER_CONFIG.margin,
        "order": "raw_fused_score",
        "quad_weight": MATCHER_CONFIG.quad_weight,
        "historical_quad_0_4_excluded_as_target_id_dependent": True,
    }
    solver = {
        "name": "raw-tail-global",
        "baseline_quantile": SOLVER_CONFIG.baseline_quantile,
        "search_rounds": SOLVER_CONFIG.search_rounds,
        "random_seed": SOLVER_CONFIG.random_seed,
        "component_cap": SOLVER_CONFIG.component_cap,
        "fill_rounds": SOLVER_CONFIG.fill_rounds,
        "border_unary": False,
        "border_weight": SOLVER_CONFIG.border_weight,
    }
    return matcher, harvest, solver


def _validate_preregistration(
    config: Mapping[str, Any],
    artifacts: Artifacts,
) -> tuple[str, ...]:
    matcher, harvest, solver = _fixed_recipe_dicts()
    fixed = {
        "schema": CONFIG_SCHEMA,
        "experiment": EXPERIMENT,
        "protocol.current_iteration_source_disjoint": True,
        "protocol.roster_and_arms_committed_before_selected_targets_opened": True,
        "protocol.target_free_outputs_frozen_before_exact_references": True,
        "protocol.historical_model_selection_exposed": True,
        "protocol.formal_fresh_promotion_claimed": False,
        "protocol.budget_sweep": False,
        "panel.selection_namespace": SELECTION_NAMESPACE,
        "panel.selection_seed": SELECTION_SEED,
        "panel.selection_algorithm": SELECTION_ALGORITHM,
        "panel.eligible_minimum": f"img_{SOURCE_MINIMUM:06d}.png",
        "panel.eligible_maximum": f"img_{SOURCE_MAXIMUM:06d}.png",
        "panel.full_range_digest": FULL_RANGE_DIGEST,
        "panel.exclusion_digest": EXCLUSION_DIGEST,
        "panel.eligible_digest": ELIGIBLE_DIGEST,
        "panel.eligible_count": 284,
        "panel.source_count": SOURCE_COUNT,
        "panel.draws": list(DRAWS),
        "panel.case_count": CASE_COUNT,
        "panel.source_order_digest": SOURCE_ORDER_DIGEST,
        "panel.cases_digest": CASES_DIGEST,
        "matcher": matcher,
        "harvest": harvest,
        "solver": solver,
        "polish.arms": [
            {"name": "taska_legal_raw_tail", "max_swaps": 0},
            {"name": "protected_tail_24", "max_swaps": 24},
            {"name": "protected_tail_96", "max_swaps": 96},
        ],
        "polish.minimum_gain": 1e-9,
        "polish.extension_rationale": (
            "all 32 prior held300 cases saturated the pre-existing 24-swap cap; "
            "96 is one precommitted evidence-driven extension, not a sweep"
        ),
        "evaluation.primary_metric": "satisfied_adjacent_pairs_per_board",
        "evaluation.pair_denominator": PAIR_DENOMINATOR,
        "evaluation.secondary_metrics": ["adjacency_recall", "exact_tiles_per_board"],
        "evaluation.bootstrap_unit": "source_with_two_draws",
        "evaluation.bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "evaluation.bootstrap_seed": BOOTSTRAP_SEED,
        "evaluation.full_panel_requires_exactly_32_cases": True,
        "legality.organizer_train_sources_only": True,
        "legality.dirty_tiles_only_for_candidate_inference": True,
        "legality.target_ids_or_exact_references_in_candidate_inference": False,
        "legality.output_uses_each_original_upright_20x20_tile_exactly_once": True,
        "legality.competition_test_forbidden": True,
    }
    for name, expected in fixed.items():
        _require_equal(config, name, expected)

    opened_names, held_names = _reference_rosters(artifacts)
    if len(opened_names) != 16 or len(held_names) != 16:
        raise ValueError("reference panels must each contain 16 sources")
    excluded = tuple(sorted(set(opened_names) | set(held_names)))
    eligible = _eligible_names(opened_names, held_names)
    roster = _deterministic_source_roster(opened_names, held_names)
    if _names_digest(_full_range_names()) != FULL_RANGE_DIGEST:
        raise RuntimeError("full last-300 source universe changed")
    if _names_digest(excluded) != EXCLUSION_DIGEST:
        raise RuntimeError("reference panel exclusion roster changed")
    if len(eligible) != 284 or _names_digest(eligible) != ELIGIBLE_DIGEST:
        raise RuntimeError("current-disjoint eligible source universe changed")
    if set(roster) & (set(opened_names) | set(held_names)):
        raise RuntimeError("deterministic roster overlaps a reference panel")
    if _names_digest(roster) != SOURCE_ORDER_DIGEST or _cases_digest(roster) != CASES_DIGEST:
        raise RuntimeError("deterministic roster digest changed")
    configured_names = config.get("panel", {}).get("source_filenames")
    if configured_names != list(roster):
        raise ValueError("configured panel differs from the deterministic roster")

    held_recipe = json.loads(artifacts.held300_recipe.read_text(encoding="utf-8"))
    for section in ("matcher", "harvest", "solver"):
        if config.get(section) != held_recipe.get(section):
            raise ValueError(f"{section} differs from the frozen held300 recipe")
    return roster


def _load_manifest(artifacts: Artifacts) -> dict[str, Mapping[str, Any]]:
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or compute_protocol_digest(manifest) != manifest.get(
        "protocol_digest"
    ):
        raise ValueError("organizer-train manifest protocol digest is invalid")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("organizer-train manifest has no split mapping")
    rows = [row for split in splits.values() if isinstance(split, list) for row in split]
    if len(rows) != 7_000 or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("organizer-train manifest must contain exactly 7000 records")
    lookup = {str(row["filename"]): row for row in rows}
    if len(lookup) != 7_000:
        raise ValueError("organizer-train manifest has duplicate filenames")
    return lookup


def _case_specs(
    source_names: Sequence[str],
    lookup: Mapping[str, Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], str, int]]:
    if any(name not in lookup for name in source_names):
        raise ValueError("registered source is absent from organizer-train manifest")
    if any(not SOURCE_MINIMUM <= _source_index(name) <= SOURCE_MAXIMUM for name in source_names):
        raise ValueError("registered source escaped the last-300 range")
    specs = [(lookup[name], name, draw) for name in source_names for draw in DRAWS]
    if len(specs) != CASE_COUNT:
        raise RuntimeError("panel expansion changed")
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
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to reuse output directory {root}") from error
    return RunPaths(
        frozen_eval=root / "frozen-target-free-eval.npz",
        frozen_eval_metadata=root / "frozen-target-free-eval.json",
        pre_score_freeze=root / "pre-score-freeze.json",
        report=root / "report.json",
    )


def _strict_layout(value: Any) -> np.ndarray:
    layout = np.ascontiguousarray(value, dtype=np.int32)
    if layout.shape != (COUNT,) or not np.array_equal(np.sort(layout), np.arange(COUNT)):
        raise ValueError("layout is not a strict original-tile permutation")
    return layout


class CleanTileCache:
    """Small verified cache over explicit organizer-train target records."""

    def __init__(self, targets: Path, *, maximum_boards: int = 2) -> None:
        if maximum_boards <= 0:
            raise ValueError("maximum_boards must be positive")
        if not targets.is_dir():
            raise ValueError(f"organizer-train targets directory is absent: {targets}")
        self.targets = targets
        self.maximum_boards = maximum_boards
        self.values: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, record: Mapping[str, Any]) -> np.ndarray:
        filename = str(record["filename"])
        if filename in self.values:
            value = self.values.pop(filename)
            self.values[filename] = value
            return value
        path = self.targets / filename
        expected = record.get("target_sha256")
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ValueError(f"organizer-train target hash mismatch: {filename}")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (480, 480):
                raise ValueError(f"expected RGB 480x480 target: {path}")
            value = np.ascontiguousarray(split_tiles(np.asarray(image, dtype=np.uint8)))
        self.values[filename] = value
        while len(self.values) > self.maximum_boards:
            self.values.popitem(last=False)
        return value


def _dirty_case(
    cache: CleanTileCache,
    record: Mapping[str, Any],
    source_name: str,
    draw: int,
) -> DirtyCase:
    digest = hashlib.sha256(
        f"{DEFAULT_SYNTHETIC_NAMESPACE}\0{SYNTHETIC_SEED}\0{source_name}\0{draw}".encode()
    ).digest()
    corruption_seed = int.from_bytes(digest[:8], "little")
    permutation_seed = int.from_bytes(digest[8:16], "little")
    corrupted = distort_tiles(cache.load(record), np.random.default_rng(corruption_seed))
    shuffle = np.random.default_rng(permutation_seed).permutation(COUNT)
    tiles = np.ascontiguousarray(corrupted[shuffle])
    if tiles.shape != (COUNT, 20, 20, 3) or tiles.dtype != np.uint8:
        raise ValueError("dirty tiles violate the 576x20x20 RGB contract")
    case_digest = hashlib.sha256(
        f"{source_name}\0{draw}\0{SYNTHETIC_SEED}".encode()
    ).hexdigest()[:16]
    return DirtyCase(f"synthetic-{case_digest}", source_name, draw, tiles)


def _dirty_sha256(tiles: np.ndarray) -> str:
    value = np.ascontiguousarray(tiles)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (COUNT, COUNT) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite {(COUNT, COUNT)} matrix")
    return np.ascontiguousarray(matrix)


def _freeze_target_free_eval(
    paths: RunPaths,
    specs: Sequence[tuple[Mapping[str, Any], str, int]],
    artifacts: Artifacts,
    *,
    targets: Path,
    device: torch.device,
) -> tuple[float, int]:
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    if artifacts.matcher_v3.parent != artifacts.matcher_local.parent:
        raise ValueError("TASKA checkpoints must share one audited directory")
    matchers = load_default_taska_ensemble(artifacts.matcher_v3.parent, device=device)
    cache = CleanTileCache(targets)
    started = perf_counter()
    for index, (record, source_name, draw) in enumerate(specs):
        dirty = _dirty_case(cache, record, source_name, draw)
        matched = match_taska_tiles(
            dirty.dirty_tiles,
            matchers,
            config=MATCHER_CONFIG,
            device=device,
        )
        cost_right = _finite_matrix(matched.cost_right, name="cost_right")
        cost_down = _finite_matrix(matched.cost_down, name="cost_down")
        right_log = _finite_matrix(matched.right_log, name="right_log")
        down_log = _finite_matrix(matched.down_log, name="down_log")
        edges = tuple(matched.candidate_edges)
        records = tuple(matched.vote_records)
        if not all(isinstance(edge, RawTailEdge) for edge in edges):
            raise TypeError("candidate_edges must contain RawTailEdge values")
        if tuple(record.edge for record in records) != edges:
            raise ValueError("vote_records and candidate_edges differ")
        if int(matched.scorer_count) != 12:
            raise ValueError("TASKA scorer count differs from the fixed recipe")
        solved = solve_raw_tail_global(
            cost_right,
            cost_down,
            edges,
            border_unary=None,
            grid=GRID,
            config=SOLVER_CONFIG,
        )
        raw_layout = _strict_layout(solved.layout)
        polish24 = polish_unprotected_taska_tail(
            raw_layout,
            cost_right,
            cost_down,
            edges,
            grid=GRID,
            max_swaps=24,
            minimum_gain=1e-9,
        )
        polish96 = polish_unprotected_taska_tail(
            raw_layout,
            cost_right,
            cost_down,
            edges,
            grid=GRID,
            max_swaps=96,
            minimum_gain=1e-9,
        )
        layout24 = _strict_layout(polish24.layout)
        layout96 = _strict_layout(polish96.layout)
        if polish96.diagnostics.accepted_swap_count < polish24.diagnostics.accepted_swap_count:
            raise RuntimeError("96-swap extension accepted fewer swaps than its 24-swap prefix")
        if polish96.diagnostics.final_total_cost > polish24.diagnostics.final_total_cost + 1e-7:
            raise RuntimeError("96-swap extension increased seam cost relative to the fixed 24 arm")

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
        arrays[f"{prefix}__taska_legal_raw_tail"] = raw_layout
        arrays[f"{prefix}__protected_tail_24"] = layout24
        arrays[f"{prefix}__protected_tail_96"] = layout96
        rows.append(
            {
                "prefix": prefix,
                "case_id": dirty.case_id,
                "source_filename": source_name,
                "draw_index": draw,
                "dirty_sha256": _dirty_sha256(dirty.dirty_tiles),
                "candidate_edge_count": len(edges),
                "chosen_vote_threshold": int(matched.chosen_vote_threshold),
                "scorer_count": int(matched.scorer_count),
                "checkpoint_sha256": list(matched.checkpoint_sha256),
                "raw_solver_diagnostics": solved.diagnostics.as_dict(),
                "protected_tail_24_diagnostics": asdict(polish24.diagnostics),
                "protected_tail_96_diagnostics": asdict(polish96.diagnostics),
            }
        )
        print(
            json.dumps(
                {
                    "event": "taska_protected_tail_target_free_case_ready",
                    "case": index + 1,
                    "case_count": len(specs),
                    "source_filename": source_name,
                    "draw_index": draw,
                    "harvest_edges": len(edges),
                    "accepted_swaps_24": polish24.diagnostics.accepted_swap_count,
                    "accepted_swaps_96": polish96.diagnostics.accepted_swap_count,
                    "strict_arms": 3,
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
            "contains_dirty_derived_matcher_matrices": True,
            "contains_frozen_harvest_membership": True,
            "contains_all_precommitted_strict_layouts": True,
            "arm_budgets": ARM_BUDGETS,
            "strict_layout_count_per_arm": len(rows),
            "current_iteration_source_disjoint": True,
            "historical_model_selection_exposed": True,
            "rows": rows,
        },
    )
    return perf_counter() - started, len(rows)


def _dependency_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
        "torch": torch.__version__,
    }


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
        **{name: _record(getattr(artifacts, name)) for name in ARTIFACT_NAMES},
        **{name: _record(path) for name, path in RUNTIME_SOURCE_PATHS.items()},
        "frozen_target_free_eval": _record(paths.frozen_eval),
        "frozen_target_free_eval_metadata": _record(paths.frozen_eval_metadata),
    }
    _write_json_exclusive(
        paths.pre_score_freeze,
        {
            "schema": FREEZE_SCHEMA,
            "created_before_exact_reference_recreation": True,
            "contains_evaluation_references_or_labels": False,
            "roster_and_arms_committed_before_selected_targets_opened": True,
            "current_iteration_source_disjoint": True,
            "historical_model_selection_exposed": True,
            "formal_fresh_promotion_claimed": False,
            "device": str(device),
            "nondeterministic_mps_explicitly_allowed": allow_nondeterministic_mps,
            "dependency_versions": _dependency_versions(),
            "artifacts": frozen,
        },
    )
    return frozen


def _validate_pre_score_freeze(
    paths: RunPaths,
    expected: Mapping[str, Mapping[str, str]],
) -> None:
    payload = json.loads(paths.pre_score_freeze.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_recreation") is not True:
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
    if pair_count != round(recall * PAIR_DENOMINATOR):
        raise RuntimeError("integer adjacency count and recall disagree")
    return {
        "satisfied_adjacent_pairs": pair_count,
        "adjacency_recall": recall,
        "exact_tiles": int(evaluation.correct_tile_count),
        "strict_permutation": True,
    }


def source_clustered_mean_ci(
    values: Sequence[float],
    sources: Sequence[str],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if len(values) != len(sources) or not values:
        raise ValueError("values and sources must be aligned and non-empty")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("clustered values must be finite")
        grouped[str(source)].append(float(value))
    if any(len(group) != len(DRAWS) for group in grouped.values()):
        raise ValueError("every source cluster must contain both registered draws")
    source_means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    generator = np.random.default_rng(seed)
    distribution = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 2048):
        stop = min(start + 2048, resamples)
        indices = generator.integers(
            0,
            len(source_means),
            size=(stop - start, len(source_means)),
        )
        distribution[start:stop] = source_means[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "source_cluster_mean": float(source_means.mean()),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": len(source_means),
        "case_count": len(values),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


def _source_wins_ties_losses(
    deltas: Sequence[float],
    sources: Sequence[str],
) -> dict[str, int]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for delta, source in zip(deltas, sources, strict=True):
        grouped[source].append(float(delta))
    means = [float(np.mean(grouped[name])) for name in sorted(grouped)]
    return {
        "wins": sum(value > 0 for value in means),
        "ties": sum(value == 0 for value in means),
        "losses": sum(value < 0 for value in means),
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]], *, full_panel: bool) -> dict[str, Any]:
    metric_names = (
        ("satisfied_adjacent_pairs", "satisfied_adjacent_pairs_per_board"),
        ("adjacency_recall", "adjacency_recall"),
        ("exact_tiles", "exact_tiles_per_board"),
    )
    sources = [str(row["source_filename"]) for row in rows]
    summary: dict[str, Any] = {"pair_denominator": PAIR_DENOMINATOR, "arms": {}}
    for arm_index, arm in enumerate(ARM_BUDGETS):
        arm_summary: dict[str, Any] = {}
        for metric_index, (row_metric, report_metric) in enumerate(metric_names):
            values = [float(row[arm][row_metric]) for row in rows]
            if full_panel:
                arm_summary[report_metric] = source_clustered_mean_ci(
                    values,
                    sources,
                    seed=BOOTSTRAP_SEED + 100 * arm_index + metric_index,
                )
            else:
                arm_summary[report_metric] = {
                    "mean": float(np.mean(values)),
                    "ci95_lower": None,
                    "ci95_upper": None,
                    "smoke_only": True,
                }
        summary["arms"][arm] = arm_summary

    comparisons = (
        ("protected_tail_24", "taska_legal_raw_tail"),
        ("protected_tail_96", "taska_legal_raw_tail"),
        ("protected_tail_96", "protected_tail_24"),
    )
    summary["comparisons"] = {}
    for comparison_index, (candidate, baseline) in enumerate(comparisons):
        name = f"{candidate}_minus_{baseline}"
        comparison: dict[str, Any] = {}
        for metric_index, (row_metric, report_metric) in enumerate(metric_names):
            deltas = [
                float(row[candidate][row_metric]) - float(row[baseline][row_metric])
                for row in rows
            ]
            if full_panel:
                comparison[report_metric] = source_clustered_mean_ci(
                    deltas,
                    sources,
                    seed=BOOTSTRAP_SEED + 1_000 + 100 * comparison_index + metric_index,
                )
                comparison[report_metric]["source_wins_ties_losses"] = (
                    _source_wins_ties_losses(deltas, sources)
                )
            else:
                comparison[report_metric] = {
                    "mean": float(np.mean(deltas)),
                    "ci95_lower": None,
                    "ci95_upper": None,
                    "smoke_only": True,
                }
        summary["comparisons"][name] = comparison
    return summary


def _score_frozen_eval(
    paths: RunPaths,
    specs: Sequence[tuple[Mapping[str, Any], str, int]],
    expected_hashes: Mapping[str, Mapping[str, str]],
    *,
    targets: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_pre_score_freeze(paths, expected_hashes)
    metadata = json.loads(paths.frozen_eval_metadata.read_text(encoding="utf-8"))
    frozen_rows = metadata.get("rows")
    if not isinstance(frozen_rows, list) or len(frozen_rows) != len(specs):
        raise RuntimeError("frozen candidate row roster changed")
    cache = CleanTileCache(targets)
    rows: list[dict[str, Any]] = []
    with np.load(paths.frozen_eval) as archive:
        for frozen_row, (record, source_name, draw) in zip(frozen_rows, specs, strict=True):
            dirty, reference = make_exact_synthetic_case(
                cache.load(record),
                source_filename=source_name,
                draw_index=draw,
                seed=SYNTHETIC_SEED,
            )
            if (
                frozen_row.get("source_filename") != source_name
                or int(frozen_row.get("draw_index", -1)) != draw
                or frozen_row.get("case_id") != dirty.case_id
                or frozen_row.get("dirty_sha256") != _dirty_sha256(dirty.tiles)
                or reference.case_id != dirty.case_id
            ):
                raise RuntimeError("scoring recreated a different synthetic case")
            prefix = str(frozen_row["prefix"])
            exact = _strict_layout(reference.tile_at_position)
            row: dict[str, Any] = {
                "source_filename": source_name,
                "draw_index": draw,
                "case_id": str(dirty.case_id),
                "accepted_swaps_24": int(
                    frozen_row["protected_tail_24_diagnostics"]["accepted_swap_count"]
                ),
                "accepted_swaps_96": int(
                    frozen_row["protected_tail_96_diagnostics"]["accepted_swap_count"]
                ),
            }
            for arm in ARM_BUDGETS:
                layout = _strict_layout(archive[f"{prefix}__{arm}"])
                row[arm] = _layout_metrics(
                    evaluate_layout(layout, exact, reference_is_exact=True)
                )
            rows.append(row)
    return rows, _summarize_rows(rows, full_panel=len(rows) == CASE_COUNT)


def run(args: argparse.Namespace) -> None:
    config, config_sha = _load_config(args.config)
    artifacts = _validate_artifacts(config)
    source_names = _validate_preregistration(config, artifacts)
    lookup = _load_manifest(artifacts)
    specs = _case_specs(source_names, lookup)
    if args.smoke_one:
        specs = specs[:1]
    paths = _prepare_run_paths(args.output_dir)
    device = _select_device(
        args.device,
        allow_nondeterministic_mps=args.allow_nondeterministic_mps,
    )
    started = perf_counter()
    inference_seconds, frozen_case_count = _freeze_target_free_eval(
        paths,
        specs,
        artifacts,
        targets=args.targets.resolve(),
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
                "event": "taska_protected_tail_target_free_artifacts_hash_frozen",
                "frozen_eval_sha256": sha256_file(paths.frozen_eval),
                "frozen_eval_metadata_sha256": sha256_file(paths.frozen_eval_metadata),
                "pre_score_freeze_sha256": sha256_file(paths.pre_score_freeze),
                "case_count": frozen_case_count,
                "arm_budgets": ARM_BUDGETS,
                "exact_reference_persisted": False,
            }
        ),
        flush=True,
    )
    rows, metrics = _score_frozen_eval(
        paths,
        specs,
        frozen,
        targets=args.targets.resolve(),
    )
    full_panel = len(rows) == CASE_COUNT
    strict = all(
        bool(row[arm]["strict_permutation"])
        for row in rows
        for arm in ARM_BUDGETS
    )
    saturation24 = sum(int(row["accepted_swaps_24"]) == 24 for row in rows)
    saturation96 = sum(int(row["accepted_swaps_96"]) == 96 for row in rows)
    _write_json_exclusive(
        paths.report,
        {
            "schema": REPORT_SCHEMA,
            "status": "smoke-only" if args.smoke_one else "confirmation-complete",
            "experiment": EXPERIMENT,
            "panel": {
                "current_iteration_source_disjoint": True,
                "excluded_opened32_and_current_held16": True,
                "historical_model_selection_exposed": True,
                "formal_fresh_promotion_claimed": False,
                "registered_source_count": SOURCE_COUNT,
                "registered_case_count": CASE_COUNT,
                "evaluated_case_count": len(rows),
                "smoke_only": bool(args.smoke_one),
                "source_order_digest": SOURCE_ORDER_DIGEST,
                "cases_digest": CASES_DIGEST,
            },
            "preregistration": {"path": _project_path(args.config), "sha256": config_sha},
            "device": {
                "value": str(device),
                "nondeterministic_mps_explicitly_allowed": bool(args.allow_nondeterministic_mps),
                "determinism_claimed": device.type != "mps",
            },
            "candidate": {
                "budget_sweep": False,
                "arms_precommitted": True,
                "arm_budgets": ARM_BUDGETS,
                "extension_is_evidence_driven": True,
                "matcher": asdict(MATCHER_CONFIG),
                "solver": asdict(SOLVER_CONFIG),
                "minimum_gain": 1e-9,
                "accepted_swap_cap_saturation": {
                    "protected_tail_24": saturation24,
                    "protected_tail_96": saturation96,
                    "case_count": len(rows),
                },
            },
            "artifacts": {name: _record(getattr(artifacts, name)) for name in ARTIFACT_NAMES},
            "runtime_sources": {name: _record(path) for name, path in RUNTIME_SOURCE_PATHS.items()},
            "dependency_versions": _dependency_versions(),
            "frozen_eval": {
                "archive": _record(paths.frozen_eval),
                "metadata": _record(paths.frozen_eval_metadata),
                "pre_score_freeze": _record(paths.pre_score_freeze),
                "target_free_outputs_frozen_before_exact_references": True,
                "contains_exact_references_or_labels": False,
            },
            "metrics": metrics,
            "measurement": {
                "full_registered_32_case_panel": full_panel,
                "all_three_arm_layouts_strict": strict,
                "valid": full_panel and strict,
                "promotable_as_formally_fresh": False,
            },
            "rows": rows,
            "runtime_seconds": {
                "target_free_inference_and_polish": inference_seconds,
                "total": perf_counter() - started,
            },
            "legality": {
                "organizer_train_sources_only": True,
                "dirty_tiles_only_for_candidate_inference": True,
                "restored_pixels_emitted": False,
                "target_ids_or_references_used_by_candidate": False,
                "original_upright_tile_permutations_only": True,
                "competition_test_accessed": False,
            },
        },
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
