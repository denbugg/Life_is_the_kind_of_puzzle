#!/usr/bin/env python3
"""Measure the fixed legal TASKA solver on the historical last-300 sources.

The 16-source, two-draw roster is committed in a signed JSON file before any
selected target is opened.  Every source is in ``img_006700..img_006999``, the
range excluded by the historical matcher training expression ``names[:-300]``.
That range was nevertheless used for historical model selection, so this run
is explicitly a source-held diagnostic and never a fresh promotion claim.

Candidate inference sees only one shuffled bag of 576 dirty upright 20x20 RGB
tiles.  Dirty-derived matrices, vote harvest, and strict layouts are written
and hash-frozen before exact synthetic references are recreated for scoring.
The fixed recipe is byte-pinned to the opened32 TASKA preregistration.
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
from aiijc_puzzle.protocol import (
    compute_protocol_digest,
    sha256_file,
    split_tiles,
)
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
from aiijc_puzzle.taska_seam_matcher import (
    TaskaSeamConfig,
    load_default_taska_ensemble,
    match_taska_tiles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "aiijc-taska-seam-held300-diagnostic-v1"
REPORT_SCHEMA = "aiijc-taska-seam-held300-diagnostic-report-v1"
FROZEN_SCHEMA = "aiijc-taska-seam-held300-frozen-target-free-v1"
FREEZE_SCHEMA = "aiijc-taska-seam-held300-pre-score-freeze-v1"
EXPERIMENT = "taska-seam-held300-diagnostic-v1"

DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_seam_held300_diagnostic_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-seam-replay/held300-diagnostic-v1"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"

GRID = 24
COUNT = GRID * GRID
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
SOURCE_MINIMUM = 6_700
SOURCE_MAXIMUM = 6_999
UNIVERSE_COUNT = SOURCE_MAXIMUM - SOURCE_MINIMUM + 1
SOURCE_COUNT = 16
DRAWS = (0, 1)
CASE_COUNT = SOURCE_COUNT * len(DRAWS)
SELECTION_NAMESPACE = "aiijc-taska-seam-held300-diagnostic-v1-source16xdraw2"
SELECTION_SEED = 2_087_291_821
SELECTION_ALGORITHM = "sha256(namespace\\0seed\\0filename) ascending, filename tie-break"
SYNTHETIC_SEED = 1_267_233_517
BOOTSTRAP_SEED = 934_711_727
BOOTSTRAP_RESAMPLES = 20_000

UNIVERSE_DIGEST = "8f736ea53d6bb377c82b0286d51381d2305365266dd02286fb417a9d028729d5"
SOURCE_ORDER_DIGEST = "9876aebe37330d923b803d8d01a33e9bf17699a8a9b2cb2363fd22909a92d234"
CASES_DIGEST = "f9d70aa1858564abcf8296b244338c8d5a06817ac29e5e2e48eaf656c6404e1d"

OPENED32_CONFIG_SHA256 = "06396905bdf7b552e165e207262d5a0654d19af3d35b2d557501551ca46da359"
MANIFEST_SHA256 = "4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da"
V3_CHECKPOINT_SHA256 = "6f0917d66d908f6cc0f4c1fcb949d3bcbadcba2490a6f7b5a12596e61de9730e"
LOCAL_CHECKPOINT_SHA256 = "5932853a73961d261b494368a4db04633fecc5996771c14d64f49ef00c7cfe73"

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

ARTIFACT_HASHES = {
    "manifest": MANIFEST_SHA256,
    "opened32_recipe": OPENED32_CONFIG_SHA256,
    "matcher_v3": V3_CHECKPOINT_SHA256,
    "matcher_local": LOCAL_CHECKPOINT_SHA256,
}

RUNTIME_SOURCE_PATHS = {
    "diagnostic_runner": Path(__file__).resolve(),
    "layout_evaluation": PROJECT_ROOT / "src/aiijc_puzzle/layout_evaluation.py",
    "protocol": PROJECT_ROOT / "src/aiijc_puzzle/protocol.py",
    "raw_tail_global_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
    "restoration_corruption": PROJECT_ROOT / "src/aiijc_puzzle/restoration_r6.py",
    "synthetic_case_generator": PROJECT_ROOT / "src/aiijc_puzzle/synthetic_socket_evaluation.py",
    "taska_seam_matcher": PROJECT_ROOT / "src/aiijc_puzzle/taska_seam_matcher.py",
}


@dataclass(frozen=True)
class Artifacts:
    manifest: Path
    opened32_recipe: Path
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
        help="run only the first committed case; no panel-level inference is claimed",
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


def _eligible_names() -> tuple[str, ...]:
    return tuple(f"img_{index:06d}.png" for index in range(SOURCE_MINIMUM, SOURCE_MAXIMUM + 1))


def _deterministic_source_roster() -> tuple[str, ...]:
    prefix = f"{SELECTION_NAMESPACE}\0{SELECTION_SEED}\0".encode()
    return tuple(
        sorted(
            _eligible_names(),
            key=lambda name: (hashlib.sha256(prefix + name.encode("utf-8")).digest(), name),
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


def _validate_recipe(config: Mapping[str, Any]) -> tuple[str, ...]:
    matcher, harvest, solver = _fixed_recipe_dicts()
    fixed: dict[str, Any] = {
        "schema": CONFIG_SCHEMA,
        "experiment": EXPERIMENT,
        "protocol.single_candidate_arm": True,
        "protocol.hyperparameter_or_arm_sweep": False,
        "protocol.roster_committed_before_selected_targets_opened": True,
        "protocol.predictions_matrices_harvest_and_layouts_frozen_before_references": True,
        "protocol.historical_matcher_training_disjoint": True,
        "protocol.historical_model_selection_exposed": True,
        "protocol.fresh_promotion_claimed": False,
        "panel.selection_namespace": SELECTION_NAMESPACE,
        "panel.selection_seed": SELECTION_SEED,
        "panel.selection_algorithm": SELECTION_ALGORITHM,
        "panel.eligible_minimum": f"img_{SOURCE_MINIMUM:06d}.png",
        "panel.eligible_maximum": f"img_{SOURCE_MAXIMUM:06d}.png",
        "panel.eligible_count": UNIVERSE_COUNT,
        "panel.eligible_digest": UNIVERSE_DIGEST,
        "panel.draws": list(DRAWS),
        "panel.source_count": SOURCE_COUNT,
        "panel.case_count": CASE_COUNT,
        "panel.source_order_digest": SOURCE_ORDER_DIGEST,
        "panel.cases_digest": CASES_DIGEST,
        "panel.opened32_overlap_count": 0,
        "matcher": matcher,
        "harvest": harvest,
        "solver": solver,
        "evaluation.primary_metric": "satisfied_adjacent_pairs_per_board",
        "evaluation.pair_denominator": PAIR_DENOMINATOR,
        "evaluation.secondary_metrics": ["adjacency_recall", "exact_tiles_per_board"],
        "evaluation.bootstrap_unit": "source_with_two_draws",
        "evaluation.bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "evaluation.bootstrap_seed": BOOTSTRAP_SEED,
        "evaluation.full_panel_requires_exactly_32_cases": True,
        "legality.organizer_train_sources_only": True,
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
        isinstance(name, str) and Path(name).name == name for name in names
    ):
        raise ValueError("panel source_filenames are malformed")
    source_names = tuple(names)
    expected_names = _deterministic_source_roster()
    if source_names != expected_names:
        raise ValueError("panel differs from the deterministic source roster")
    if len(source_names) != SOURCE_COUNT or len(set(source_names)) != SOURCE_COUNT:
        raise ValueError("panel must contain exactly 16 unique sources")
    if _names_digest(_eligible_names()) != UNIVERSE_DIGEST:
        raise RuntimeError("held300 eligible universe changed")
    if _names_digest(source_names) != SOURCE_ORDER_DIGEST:
        raise ValueError("panel source order digest changed")
    if _cases_digest(source_names, DRAWS) != CASES_DIGEST:
        raise ValueError("panel cases digest changed")
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
    if not isinstance(records, Mapping) or set(records) != set(ARTIFACT_HASHES):
        raise ValueError("artifact roster changed")
    paths = {
        name: _validate_record(records, name, expected_sha256=expected)
        for name, expected in ARTIFACT_HASHES.items()
    }
    runtime = config.get("runtime_sources")
    if not isinstance(runtime, Mapping) or set(runtime) != set(RUNTIME_SOURCE_PATHS):
        raise ValueError("runtime source roster changed")
    for name, expected_path in RUNTIME_SOURCE_PATHS.items():
        _validate_record(runtime, name, expected_path=expected_path)
    return Artifacts(**paths)


def _validate_reference_recipe(artifacts: Artifacts, config: Mapping[str, Any]) -> set[str]:
    reference = json.loads(artifacts.opened32_recipe.read_text(encoding="utf-8"))
    for section in ("matcher", "harvest", "solver"):
        if config.get(section) != reference.get(section):
            raise ValueError(f"held300 {section} differs from the frozen opened32 recipe")
    opened_panel = reference.get("panel", {}).get("source_filenames")
    if not isinstance(opened_panel, list) or not all(
        isinstance(name, str) for name in opened_panel
    ):
        raise ValueError("opened32 source roster is malformed")
    return set(opened_panel)


def _load_manifest(artifacts: Artifacts) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
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
    return manifest, lookup


def _case_specs(
    source_names: Sequence[str],
    lookup: Mapping[str, Mapping[str, Any]],
    opened32_sources: set[str],
) -> list[tuple[Mapping[str, Any], str, int]]:
    if set(source_names) & opened32_sources:
        raise ValueError("held300 panel overlaps the opened32 source panel")
    if any(name not in lookup for name in source_names):
        raise ValueError("held300 panel source is absent from organizer-train manifest")
    if any(not SOURCE_MINIMUM <= int(name[4:10]) <= SOURCE_MAXIMUM for name in source_names):
        raise ValueError("held300 panel escaped the historical last-300 range")
    specs = [(lookup[name], name, draw) for name in source_names for draw in DRAWS]
    if len(specs) != CASE_COUNT:
        raise RuntimeError("held300 panel expansion changed")
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
            value = np.ascontiguousarray(
                split_tiles(np.asarray(image, dtype=np.uint8)),
                dtype=np.uint8,
            )
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
    # Reproduce only the dirty half of make_exact_synthetic_case.  In
    # particular, no inverse shuffle / exact reference is constructed before
    # the dirty-only prediction archive has been written and hash-frozen.
    digest = hashlib.sha256(
        f"{DEFAULT_SYNTHETIC_NAMESPACE}\0{SYNTHETIC_SEED}\0{source_name}\0{draw}".encode()
    ).digest()
    corruption_seed = int.from_bytes(digest[:8], "little")
    permutation_seed = int.from_bytes(digest[8:16], "little")
    corrupted = distort_tiles(
        cache.load(record),
        np.random.default_rng(corruption_seed),
    )
    shuffle = np.random.default_rng(permutation_seed).permutation(COUNT)
    tiles = np.ascontiguousarray(corrupted[shuffle])
    if tiles.shape != (COUNT, 20, 20, 3) or tiles.dtype != np.uint8:
        raise ValueError("dirty tiles violate the 576x20x20 RGB contract")
    case_digest = hashlib.sha256(f"{source_name}\0{draw}\0{SYNTHETIC_SEED}".encode()).hexdigest()[
        :16
    ]
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
        raise ValueError("TASKA v3/local checkpoints must share one audited directory")
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
        if set(matched.checkpoint_sha256) != {
            V3_CHECKPOINT_SHA256,
            LOCAL_CHECKPOINT_SHA256,
        }:
            raise ValueError("TASKA result checkpoint provenance changed")
        solved = solve_raw_tail_global(
            cost_right,
            cost_down,
            edges,
            border_unary=None,
            grid=GRID,
            config=SOLVER_CONFIG,
        )
        layout = _strict_layout(solved.layout)
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
            [record.minimum_margin for record in records], dtype=np.float32
        )
        arrays[f"{prefix}__edge_vote_count"] = np.asarray(
            [record.vote_count for record in records], dtype=np.int16
        )
        arrays[f"{prefix}__taska_layout"] = layout
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
                "solver_diagnostics": solved.diagnostics.as_dict(),
            }
        )
        print(
            json.dumps(
                {
                    "event": "taska_held300_target_free_case_frozen_in_memory",
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
            "strict_layout_count": len(rows),
            "source_range_was_outside_historical_matcher_training": True,
            "source_range_was_historically_model_selection_exposed": True,
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
        **{name: _record(getattr(artifacts, name)) for name in ARTIFACT_HASHES},
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
            "source_roster_was_committed_before_selected_targets_opened": True,
            "fresh_promotion_claimed": False,
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
        raise ValueError("every source cluster must contain exactly both registered draws")
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


def _summarize_rows(rows: Sequence[Mapping[str, Any]], *, full_panel: bool) -> dict[str, Any]:
    metrics = (
        ("satisfied_adjacent_pairs", "satisfied_adjacent_pairs_per_board"),
        ("adjacency_recall", "adjacency_recall"),
        ("exact_tiles", "exact_tiles_per_board"),
    )
    summary: dict[str, Any] = {"pair_denominator": PAIR_DENOMINATOR}
    sources = [str(row["source_filename"]) for row in rows]
    for metric_index, (row_metric, report_metric) in enumerate(metrics):
        values = [float(row["taska_legal_raw_tail"][row_metric]) for row in rows]
        if full_panel:
            summary[report_metric] = source_clustered_mean_ci(
                values,
                sources,
                seed=BOOTSTRAP_SEED + metric_index,
            )
        else:
            summary[report_metric] = {
                "mean": float(np.mean(values)),
                "ci95_lower": None,
                "ci95_upper": None,
                "smoke_only": True,
            }
    return summary


def _score_frozen_eval(
    paths: RunPaths,
    specs: Sequence[tuple[Mapping[str, Any], str, int]],
    expected_hashes: Mapping[str, Mapping[str, str]],
    *,
    targets: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # No target cache is constructed until every dirty-only artifact hash has
    # been checked against the pre-score freeze.
    _validate_pre_score_freeze(paths, expected_hashes)
    metadata = json.loads(paths.frozen_eval_metadata.read_text(encoding="utf-8"))
    frozen_rows = metadata.get("rows")
    if not isinstance(frozen_rows, list) or len(frozen_rows) != len(specs):
        raise RuntimeError("frozen candidate row roster changed")
    cache = CleanTileCache(targets)
    rows: list[dict[str, Any]] = []
    with np.load(paths.frozen_eval) as archive:
        for frozen_row, (record, source_name, draw) in zip(
            frozen_rows,
            specs,
            strict=True,
        ):
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
            layout = _strict_layout(archive[f"{prefix}__taska_layout"])
            exact = _strict_layout(reference.tile_at_position)
            rows.append(
                {
                    "source_filename": source_name,
                    "draw_index": draw,
                    "case_id": str(dirty.case_id),
                    "taska_legal_raw_tail": _layout_metrics(
                        evaluate_layout(layout, exact, reference_is_exact=True)
                    ),
                }
            )
    return rows, _summarize_rows(rows, full_panel=len(rows) == CASE_COUNT)


def run(args: argparse.Namespace) -> None:
    config, config_sha, source_names = _load_preregistration(args.config)
    artifacts = _validate_artifacts(config)
    opened32_sources = _validate_reference_recipe(artifacts, config)
    _manifest, lookup = _load_manifest(artifacts)
    specs = _case_specs(source_names, lookup, opened32_sources)
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
                "event": "taska_held300_predictions_matrices_harvest_and_layouts_frozen",
                "frozen_eval_sha256": sha256_file(paths.frozen_eval),
                "frozen_eval_metadata_sha256": sha256_file(paths.frozen_eval_metadata),
                "pre_score_freeze_sha256": sha256_file(paths.pre_score_freeze),
                "case_count": frozen_case_count,
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
    strict = all(bool(row["taska_legal_raw_tail"]["strict_permutation"]) for row in rows)
    _write_json_exclusive(
        paths.report,
        {
            "schema": REPORT_SCHEMA,
            "status": "smoke-only" if args.smoke_one else "diagnostic-complete",
            "experiment": EXPERIMENT,
            "panel": {
                "historical_matcher_training_disjoint": True,
                "historical_model_selection_exposed": True,
                "fresh_promotion_claimed": False,
                "opened32_overlap_count": 0,
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
                "single_fixed_arm": True,
                "hyperparameter_or_arm_sweep": False,
                "opened32_matcher_harvest_solver_recipe_exactly_reused": True,
                "matcher": asdict(MATCHER_CONFIG),
                "solver": asdict(SOLVER_CONFIG),
                "chooser_used": False,
                "verifier_used": False,
                "border_prior_used": False,
            },
            "artifacts": {name: _record(getattr(artifacts, name)) for name in ARTIFACT_HASHES},
            "runtime_sources": {name: _record(path) for name, path in RUNTIME_SOURCE_PATHS.items()},
            "dependency_versions": _dependency_versions(),
            "frozen_eval": {
                "archive": _record(paths.frozen_eval),
                "metadata": _record(paths.frozen_eval_metadata),
                "pre_score_freeze": _record(paths.pre_score_freeze),
                "predictions_matrices_harvest_and_layouts_frozen_before_references": True,
                "contains_exact_references_or_labels": False,
            },
            "metrics": metrics,
            "measurement": {
                "full_registered_32_case_panel": full_panel,
                "all_layouts_strict": strict,
                "valid": full_panel and strict,
                "promotable": False,
            },
            "rows": rows,
            "runtime_seconds": {
                "target_free_inference": inference_seconds,
                "total": perf_counter() - started,
            },
            "legality": {
                "organizer_train_sources_only": True,
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
