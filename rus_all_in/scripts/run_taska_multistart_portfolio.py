#!/usr/bin/env python3
"""Evaluate the fixed 4-arm x 4-seed TASKA multistart portfolio.

The experiment reuses already frozen target-free matcher matrices, candidate
membership, calibrator features, and focal logits.  It changes only
``RawTailGlobalConfig.random_seed`` to the preregistered set ``(0, 1, 2, 3)``
for raw/logistic/focal/nonlinear edge ordering.  The minimum original all-bond
TASKA seam-cost layout is selected from the 16 strict layouts and receives the
unchanged protected 96-swap tail.

All layouts and selection diagnostics are hash-frozen before an exact
synthetic reference is recreated.  Targets are used only in the final offline
scoring pass.  The script never modifies the frozen raw solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge, RawTailGlobalConfig
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case
from aiijc_puzzle.taska_edge_calibrator import (
    TaskaEdgeCalibrator,
    extract_taska_edge_features,
)
from aiijc_puzzle.taska_layout_portfolio import select_lowest_taska_seam_cost_layout
from aiijc_puzzle.taska_multistart_portfolio import (
    TASKA_MULTISTART_ARMS,
    TASKA_MULTISTART_SEEDS,
    TASKA_MULTISTART_TAIL_SWAPS,
    solve_taska_multistart_portfolio,
)
from aiijc_puzzle.taska_nonlinear_calibrator import TaskaNonlinearCalibrator
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail

try:
    from scripts import run_taska_focal_verifier_replay as focal_parent
    from scripts import run_taska_protected_tail_fresh32_confirmation as fresh_parent
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_verifier_replay as focal_parent
    import run_taska_protected_tail_fresh32_confirmation as fresh_parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID = 24
COUNT = GRID * GRID
PAIR_DENOMINATOR = 2 * GRID * (GRID - 1)
CASE_COUNT = 32
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 1_351_116_901
RAW_SOLVER_SHA256 = "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
LOGISTIC_SHA256 = "adc76ee87fc112d4ca3eeb676cdec6b7d103c596d62a9848ba65ee5ef384b1ac"
NONLINEAR_SHA256 = "2a5f95bd9d8e08e57b8bd02e242e25ef4661036ed3b1985fda1d70ee1bf9d2a6"

FROZEN_SCHEMA = "aiijc-taska-multistart-portfolio-target-free-v1"
FREEZE_SCHEMA = "aiijc-taska-multistart-portfolio-pre-score-freeze-v1"
REPORT_SCHEMA = "aiijc-taska-multistart-portfolio-report-v1"

PanelName = Literal["opened32", "held300", "fresh32"]


@dataclass(frozen=True)
class PanelSpec:
    parent_archive: str
    parent_archive_sha256: str
    parent_metadata: str
    parent_metadata_sha256: str
    priority_archive: str
    priority_archive_sha256: str
    priority_metadata: str
    priority_metadata_sha256: str
    raw_layout_key: str
    current_layout_prefix: str


PANEL_SPECS: dict[PanelName, PanelSpec] = {
    "opened32": PanelSpec(
        parent_archive="outputs/taska-seam-replay/opened32-mps-v1/frozen-target-free-eval.npz",
        parent_archive_sha256="1880940897caeec6b87631d53e1aede1f809955a7acd3e56da9bcf432939e994",
        parent_metadata="outputs/taska-seam-replay/opened32-mps-v1/frozen-target-free-eval.json",
        parent_metadata_sha256="f327664bc9db353b53b8f05738f94a5baaf8eefec1c708ae92f5032c37ce6eaf",
        priority_archive=(
            "outputs/taska-focal-verifier/opened32-train-exact-top5-cpu-v2/"
            "frozen-target-free-eval.npz"
        ),
        priority_archive_sha256="60243ab924da96d8bb49b072458c4710c65b8195b8d2c31eff1132b59ee56fd2",
        priority_metadata=(
            "outputs/taska-focal-verifier/opened32-train-exact-top5-cpu-v2/"
            "frozen-target-free-eval.json"
        ),
        priority_metadata_sha256="8e6be1d0f4b2652b784141d7c53d7fb63394e8bda6af3b076a9fd5721f07c9d5",
        raw_layout_key="taska_layout",
        current_layout_prefix="",
    ),
    "held300": PanelSpec(
        parent_archive=(
            "outputs/taska-seam-replay/held300-diagnostic-mps-v1/"
            "frozen-target-free-eval.npz"
        ),
        parent_archive_sha256="0876d7acd6be7ebe863f831de568586c837da52cc603df4ff8d7c5a6b0d441df",
        parent_metadata=(
            "outputs/taska-seam-replay/held300-diagnostic-mps-v1/"
            "frozen-target-free-eval.json"
        ),
        parent_metadata_sha256="91710486233f45bda2b8aab019c508d1e6a0f75e282a6ce9a47aee93d9bf0a8d",
        priority_archive=(
            "outputs/taska-focal-verifier/held300-train-exact-top5-cpu-v2/"
            "frozen-target-free-eval.npz"
        ),
        priority_archive_sha256="7d4ad494ab572d1ac3c94ab73a49b54e80b26baba489dfbd56f732a5c43394c5",
        priority_metadata=(
            "outputs/taska-focal-verifier/held300-train-exact-top5-cpu-v2/"
            "frozen-target-free-eval.json"
        ),
        priority_metadata_sha256="301ba535f04b63ff8da48a0a83b5f207521d4b57f1bdbb61ceb58dbee57daff2",
        raw_layout_key="taska_layout",
        current_layout_prefix="",
    ),
    "fresh32": PanelSpec(
        parent_archive=(
            "outputs/taska-protected-tail/fresh-held32-mps-v1/"
            "frozen-target-free-eval.npz"
        ),
        parent_archive_sha256="d7b156ff1a8cdab702881242e48797b1a18f750a2d6a60f2a7d769dbfa1bffc1",
        parent_metadata=(
            "outputs/taska-protected-tail/fresh-held32-mps-v1/"
            "frozen-target-free-eval.json"
        ),
        parent_metadata_sha256="1acb5d0000dd76e48fb6c079827fa2113bb56f541905fc97ced9656b8d7fe53f",
        priority_archive=(
            "outputs/taska-fresh32-leader-confirmation/fresh-held32-mps-v1/"
            "frozen-target-free-eval.npz"
        ),
        priority_archive_sha256="f3710cc3b00aaf2e75cb4127c280bc95eeeedf237f51a76ca234bac079c6f75f",
        priority_metadata=(
            "outputs/taska-fresh32-leader-confirmation/fresh-held32-mps-v1/"
            "frozen-target-free-eval.json"
        ),
        priority_metadata_sha256="311a1b3dc42bfb317a2c5cde1cee319de86ceba85622cb376fe4bfb83e2b53b1",
        raw_layout_key="taska_legal_raw_tail",
        current_layout_prefix="",
    ),
}

DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_LOGISTIC = PROJECT_ROOT / "outputs/taska-edge-calibrator/train256-v1/calibrator.npz"
DEFAULT_NONLINEAR = PROJECT_ROOT / "outputs/taska-nonlinear-calibrator/train256-v1/calibrator.npz"
SOLVER_CONFIG = RawTailGlobalConfig(
    baseline_quantile=0.15,
    search_rounds=6,
    border_weight=0.0,
    random_seed=0,
    component_cap=0,
    fill_rounds=1,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=tuple(PANEL_SPECS), required=True)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--logistic", type=Path, default=DEFAULT_LOGISTIC)
    parser.add_argument("--nonlinear", type=Path, default=DEFAULT_NONLINEAR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke-one", action="store_true")
    return parser.parse_args(argv)


def _require_hash(path: Path, expected: str, *, name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{name} does not exist: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    return resolved


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rendered = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


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


def _strict_layout(value: Any) -> np.ndarray:
    layout = np.ascontiguousarray(value, dtype=np.int32)
    if layout.shape != (COUNT,) or not np.array_equal(np.sort(layout), np.arange(COUNT)):
        raise ValueError("layout is not a strict original-tile permutation")
    return layout


def _finite_matrix(archive: Any, key: str) -> np.ndarray:
    matrix = np.asarray(archive[key], dtype=np.float64)
    if matrix.shape != (COUNT, COUNT) or not np.isfinite(matrix).all():
        raise ValueError(f"{key} is not one finite 576x576 matrix")
    return np.ascontiguousarray(matrix)


def _edges_from_archive(archive: Any, prefix: str) -> tuple[RawTailEdge, ...]:
    sources = np.asarray(archive[f"{prefix}__edge_source"], dtype=np.int64)
    targets = np.asarray(archive[f"{prefix}__edge_target"], dtype=np.int64)
    axes = np.asarray(archive[f"{prefix}__edge_axis"], dtype=np.uint8)
    if not (sources.ndim == targets.ndim == axes.ndim == 1):
        raise ValueError("frozen edge arrays must be one-dimensional")
    if not (len(sources) == len(targets) == len(axes)) or not np.isin(axes, (0, 1)).all():
        raise ValueError("frozen edge arrays are misaligned or malformed")
    return tuple(
        RawTailEdge(int(source), int(target), "right" if int(axis) == 0 else "down")
        for source, target, axis in zip(sources, targets, axes, strict=True)
    )


def _load_rows(panel: PanelName, *, smoke_one: bool) -> list[Mapping[str, Any]]:
    spec = PANEL_SPECS[panel]
    metadata_path = PROJECT_ROOT / spec.parent_metadata
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("contains_exact_references_or_labels") is not False:
        raise ValueError("parent metadata is not target-free")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != CASE_COUNT:
        raise ValueError("parent panel must contain exactly 32 cases")
    required = {
        "prefix",
        "case_id",
        "source_filename",
        "draw_index",
        "dirty_sha256",
        "candidate_edge_count",
    }
    if any(not isinstance(row, Mapping) or not required <= set(row) for row in rows):
        raise ValueError("parent metadata row is malformed")
    return rows[:1] if smoke_one else rows


def _case_priorities(
    panel: PanelName,
    prefix: str,
    parent: Any,
    priority_archive: Any,
    cost_right: np.ndarray,
    cost_down: np.ndarray,
    edges: Sequence[RawTailEdge],
    logistic: TaskaEdgeCalibrator,
    nonlinear: TaskaNonlinearCalibrator,
) -> dict[str, np.ndarray]:
    if panel == "fresh32":
        features = np.asarray(priority_archive[f"{prefix}__edge_features"], dtype=np.float64)
    else:
        features = extract_taska_edge_features(
            cost_right,
            cost_down,
            _finite_matrix(parent, f"{prefix}__right_log"),
            _finite_matrix(parent, f"{prefix}__down_log"),
            edges,
            np.asarray(parent[f"{prefix}__edge_weight"], dtype=np.float64),
            np.asarray(parent[f"{prefix}__edge_vote_count"], dtype=np.float64),
            grid=GRID,
        ).values
    focal = np.asarray(priority_archive[f"{prefix}__focal_logits"], dtype=np.float64)
    if focal.shape != (len(edges),) or not np.isfinite(focal).all():
        raise ValueError("frozen focal logits are malformed")
    return {
        "logistic": logistic.predict_priorities(features),
        "focal": focal,
        "nonlinear": nonlinear.predict_priorities(features),
    }


def _solve_target_free_case(
    task: tuple[PanelName, str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    panel, prefix = task
    spec = PANEL_SPECS[panel]
    logistic = TaskaEdgeCalibrator.load_npz(DEFAULT_LOGISTIC)
    nonlinear = TaskaNonlinearCalibrator.load_npz(DEFAULT_NONLINEAR)
    with (
        np.load(PROJECT_ROOT / spec.parent_archive, allow_pickle=False) as parent,
        np.load(PROJECT_ROOT / spec.priority_archive, allow_pickle=False) as priority_archive,
    ):
        cost_right = _finite_matrix(parent, f"{prefix}__cost_right")
        cost_down = _finite_matrix(parent, f"{prefix}__cost_down")
        edges = _edges_from_archive(parent, prefix)
        priorities = _case_priorities(
            panel,
            prefix,
            parent,
            priority_archive,
            cost_right,
            cost_down,
            edges,
            logistic,
            nonlinear,
        )
        result = solve_taska_multistart_portfolio(
            cost_right,
            cost_down,
            edges,
            priorities,
            grid=GRID,
            solver_config=SOLVER_CONFIG,
        )
        layouts = dict(result.layouts)

        parent_raw = _strict_layout(parent[f"{prefix}__{spec.raw_layout_key}"])
        if not np.array_equal(layouts["raw_seed0"], parent_raw):
            raise RuntimeError("seed-0 raw replay differs from the frozen parent")
        parent_focal = _strict_layout(priority_archive[f"{prefix}__focal_layout"])
        if not np.array_equal(layouts["focal_seed0"], parent_focal):
            raise RuntimeError("seed-0 focal replay differs from the frozen focal parent")
        if panel == "fresh32":
            for arm in ("logistic", "nonlinear"):
                frozen = _strict_layout(priority_archive[f"{prefix}__{arm}_layout"])
                if not np.array_equal(layouts[f"{arm}_seed0"], frozen):
                    raise RuntimeError(f"seed-0 {arm} replay differs from frozen fresh32")

        seed0 = {arm: layouts[f"{arm}_seed0"] for arm in TASKA_MULTISTART_ARMS}
        current_selection = select_lowest_taska_seam_cost_layout(
            seed0,
            cost_right,
            cost_down,
            grid=GRID,
        )
        current_polish = polish_unprotected_taska_tail(
            current_selection.layout,
            cost_right,
            cost_down,
            edges,
            grid=GRID,
            max_swaps=TASKA_MULTISTART_TAIL_SWAPS,
            minimum_gain=1e-9,
        )
        if panel == "fresh32":
            frozen_current = _strict_layout(priority_archive[f"{prefix}__portfolio_tail96_layout"])
            if not np.array_equal(current_polish.layout, frozen_current):
                raise RuntimeError("current seed-0 fresh32 leader did not replay exactly")

        arrays = {f"layout__{name}": _strict_layout(layout) for name, layout in result.layouts}
        arrays["current_seed0_portfolio"] = _strict_layout(current_selection.layout)
        arrays["current_seed0_tail96"] = _strict_layout(current_polish.layout)
        arrays["multistart_portfolio"] = _strict_layout(result.selection.layout)
        arrays["multistart_tail96"] = _strict_layout(result.polish.layout)
        row = {
            "prefix": prefix,
            "candidate_edge_count": len(edges),
            "current_seed0_choice": current_selection.choice,
            "current_seed0_total_costs": dict(current_selection.total_costs),
            "current_seed0_tail96_diagnostics": asdict(current_polish.diagnostics),
            "multistart_choice": result.selection.choice,
            "multistart_total_costs": dict(result.selection.total_costs),
            "multistart_tail96_diagnostics": asdict(result.polish.diagnostics),
            "all_20_layouts_strict": all(
                np.array_equal(np.sort(layout), np.arange(COUNT)) for layout in arrays.values()
            ),
        }
    return arrays, row


def _runtime_sources() -> dict[str, Path]:
    return {
        "multistart_runner": Path(__file__).resolve(),
        "multistart_module": PROJECT_ROOT / "src/aiijc_puzzle/taska_multistart_portfolio.py",
        "layout_portfolio": PROJECT_ROOT / "src/aiijc_puzzle/taska_layout_portfolio.py",
        "protected_tail": PROJECT_ROOT / "src/aiijc_puzzle/taska_protected_tail_polish.py",
        "edge_calibrator": PROJECT_ROOT / "src/aiijc_puzzle/taska_edge_calibrator.py",
        "nonlinear_calibrator": PROJECT_ROOT / "src/aiijc_puzzle/taska_nonlinear_calibrator.py",
        "frozen_raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
    }


def _freeze_target_free(
    *,
    panel: PanelName,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    workers: int,
) -> tuple[Path, Path, Path, float]:
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen_path = output_dir / "frozen-target-free-eval.npz"
    metadata_path = output_dir / "frozen-target-free-eval.json"
    freeze_path = output_dir / "pre-score-freeze.json"
    tasks = [(panel, str(row["prefix"])) for row in rows]
    started = perf_counter()
    if workers == 1:
        results = map(_solve_target_free_case, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_solve_target_free_case, tasks)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    try:
        for index, (parent_row, (case_arrays, case_row)) in enumerate(
            zip(rows, results, strict=True)
        ):
            prefix = str(parent_row["prefix"])
            for name, value in case_arrays.items():
                arrays[f"{prefix}__{name}"] = value
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "case_id": parent_row["case_id"],
                    "source_filename": parent_row["source_filename"],
                    "draw_index": int(parent_row["draw_index"]),
                    "dirty_sha256": parent_row["dirty_sha256"],
                    **case_row,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "taska_multistart_target_free_case_ready",
                        "panel": panel,
                        "case": index + 1,
                        "case_count": len(rows),
                        "current_choice": case_row["current_seed0_choice"],
                        "multistart_choice": case_row["multistart_choice"],
                        "strict": case_row["all_20_layouts_strict"],
                    }
                ),
                flush=True,
            )
    finally:
        if workers != 1:
            executor.shutdown(wait=True)

    _write_npz_exclusive(frozen_path, arrays)
    _write_json_exclusive(
        metadata_path,
        {
            "schema": FROZEN_SCHEMA,
            "panel": panel,
            "contains_exact_references_or_labels": False,
            "contains_target_ids_or_source_grid_coordinates": False,
            "candidate_membership_and_costs_unchanged": True,
            "only_solver_random_seed_changes": True,
            "seeds": list(TASKA_MULTISTART_SEEDS),
            "arms": list(TASKA_MULTISTART_ARMS),
            "portfolio_layout_count": len(TASKA_MULTISTART_SEEDS)
            * len(TASKA_MULTISTART_ARMS),
            "selector": "minimum original TASKA cost over all 1104 board bonds",
            "protected_tail_max_swaps": TASKA_MULTISTART_TAIL_SWAPS,
            "rows": frozen_rows,
        },
    )
    spec = PANEL_SPECS[panel]
    artifact_paths = {
        "parent_archive": PROJECT_ROOT / spec.parent_archive,
        "parent_metadata": PROJECT_ROOT / spec.parent_metadata,
        "priority_archive": PROJECT_ROOT / spec.priority_archive,
        "priority_metadata": PROJECT_ROOT / spec.priority_metadata,
        "logistic_calibrator": DEFAULT_LOGISTIC,
        "nonlinear_calibrator": DEFAULT_NONLINEAR,
        "frozen_candidate_archive": frozen_path,
        "frozen_candidate_metadata": metadata_path,
        **_runtime_sources(),
    }
    _write_json_exclusive(
        freeze_path,
        {
            "schema": FREEZE_SCHEMA,
            "created_before_exact_reference_recreation": True,
            "contains_evaluation_references_or_labels": False,
            "panel": panel,
            "artifacts": {name: _record(path) for name, path in artifact_paths.items()},
        },
    )
    return frozen_path, metadata_path, freeze_path, perf_counter() - started


def _validate_freeze(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_recreation") is not True:
        raise RuntimeError("pre-score freeze timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze unexpectedly contains labels")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("pre-score artifact roster is missing")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise RuntimeError(f"malformed frozen artifact record: {name}")
        raw_path, expected = record.get("path"), record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise RuntimeError(f"malformed frozen artifact record: {name}")
        artifact = Path(raw_path)
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if sha256_file(artifact.resolve()) != expected:
            raise RuntimeError(f"pre-score artifact changed: {name}")
    return payload


def _dirty_sha256(panel: PanelName, tiles: np.ndarray) -> str:
    value = np.ascontiguousarray(tiles)
    if panel == "opened32":
        return hashlib.sha256(value.tobytes()).hexdigest()
    if panel == "held300":
        return focal_parent._dirty_sha256(value)
    if panel == "fresh32":
        return fresh_parent._dirty_sha256(value)
    raise ValueError(f"unsupported panel: {panel}")


def _layout_metrics(layout: np.ndarray, exact: np.ndarray) -> dict[str, Any]:
    result = evaluate_layout(layout, exact, reference_is_exact=True)
    if result.adjacency_total != PAIR_DENOMINATOR:
        raise RuntimeError("adjacency denominator changed")
    return {
        "satisfied_adjacent_pairs": int(result.adjacency_correct),
        "adjacency_recall": float(result.adjacency),
        "exact_tiles": int(result.correct_tile_count),
        "strict_permutation": True,
    }


def _clustered_ci(
    values: Sequence[float],
    sources: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("comparison contains a non-finite value")
        grouped[str(source)].append(float(value))
    if any(len(group) != 2 for group in grouped.values()):
        raise ValueError("every source cluster must contain two draws")
    source_means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    generator = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(start + 2048, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(
            0,
            len(source_means),
            size=(stop - start, len(source_means)),
        )
        distribution[start:stop] = source_means[indices].mean(axis=1)
    means = source_means.tolist()
    return {
        "mean": float(np.mean(values)),
        "source_cluster_mean": float(source_means.mean()),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": len(source_means),
        "case_count": len(values),
        "source_wins_ties_losses": {
            "wins": sum(value > 0 for value in means),
            "ties": sum(value == 0 for value in means),
            "losses": sum(value < 0 for value in means),
        },
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
    }


def _score_after_freeze(
    *,
    panel: PanelName,
    rows: Sequence[Mapping[str, Any]],
    targets: Path,
    frozen_path: Path,
    metadata_path: Path,
    freeze_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_freeze(freeze_path)
    candidate_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidate_rows = candidate_metadata.get("rows")
    if not isinstance(candidate_rows, list) or len(candidate_rows) != len(rows):
        raise RuntimeError("frozen candidate row roster changed")
    lookup = focal_parent._load_manifest_lookup()
    cache = focal_parent.CleanTileCache(targets.resolve())
    scored: list[dict[str, Any]] = []
    with np.load(frozen_path, allow_pickle=False) as archive:
        for parent_row, candidate_row in zip(rows, candidate_rows, strict=True):
            identity = ("prefix", "case_id", "source_filename", "draw_index", "dirty_sha256")
            if any(parent_row[field] != candidate_row[field] for field in identity):
                raise RuntimeError("parent and candidate frozen row identities differ")
            prefix = str(parent_row["prefix"])
            source = str(parent_row["source_filename"])
            draw = int(parent_row["draw_index"])
            dirty, reference = make_exact_synthetic_case(
                cache.load(lookup[source]),
                source_filename=source,
                draw_index=draw,
                seed=focal_parent.SYNTHETIC_SEED,
            )
            if (
                dirty.case_id != parent_row["case_id"]
                or reference.case_id != dirty.case_id
                or _dirty_sha256(panel, dirty.tiles) != parent_row["dirty_sha256"]
            ):
                raise RuntimeError("scoring recreated a different synthetic case")
            exact = _strict_layout(reference.tile_at_position)
            values: dict[str, Any] = {}
            for name in (
                "current_seed0_tail96",
                "multistart_portfolio",
                "multistart_tail96",
            ):
                values[name] = _layout_metrics(
                    _strict_layout(archive[f"{prefix}__{name}"]),
                    exact,
                )
            for seed in TASKA_MULTISTART_SEEDS:
                for arm in TASKA_MULTISTART_ARMS:
                    name = f"{arm}_seed{seed}"
                    values[name] = _layout_metrics(
                        _strict_layout(archive[f"{prefix}__layout__{name}"]),
                        exact,
                    )
            scored.append(
                {
                    "prefix": prefix,
                    "case_id": dirty.case_id,
                    "source_filename": source,
                    "draw_index": draw,
                    "current_seed0_choice": candidate_row["current_seed0_choice"],
                    "multistart_choice": candidate_row["multistart_choice"],
                    **values,
                }
            )

    metric_names = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    scored_names = (
        "current_seed0_tail96",
        "multistart_portfolio",
        "multistart_tail96",
        *(
            f"{arm}_seed{seed}"
            for seed in TASKA_MULTISTART_SEEDS
            for arm in TASKA_MULTISTART_ARMS
        ),
    )
    summary: dict[str, Any] = {
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": {
            name: {
                metric: float(np.mean([row[name][metric] for row in scored]))
                for metric in metric_names
            }
            for name in scored_names
        },
        "current_seed0_choice_counts": dict(
            Counter(str(row["current_seed0_choice"]) for row in scored)
        ),
        "multistart_choice_counts": dict(
            Counter(str(row["multistart_choice"]) for row in scored)
        ),
    }
    comparison: dict[str, Any] = {}
    sources = [str(row["source_filename"]) for row in scored]
    for index, metric in enumerate(metric_names):
        deltas = [
            float(row["multistart_tail96"][metric])
            - float(row["current_seed0_tail96"][metric])
            for row in scored
        ]
        comparison[metric] = (
            _clustered_ci(deltas, sources, seed=BOOTSTRAP_SEED + index)
            if len(scored) == CASE_COUNT
            else {
                "mean": float(np.mean(deltas)),
                "ci95_lower": None,
                "ci95_upper": None,
                "smoke_only": True,
            }
        )
    summary["multistart_tail96_minus_current_seed0_tail96"] = comparison
    return scored, summary


def run(args: argparse.Namespace) -> None:
    panel: PanelName = args.panel
    if isinstance(args.workers, bool) or not 1 <= args.workers <= 8:
        raise ValueError("workers must be an integer in [1, 8]")
    spec = PANEL_SPECS[panel]
    _require_hash(
        PROJECT_ROOT / spec.parent_archive,
        spec.parent_archive_sha256,
        name="parent archive",
    )
    _require_hash(
        PROJECT_ROOT / spec.parent_metadata,
        spec.parent_metadata_sha256,
        name="parent metadata",
    )
    _require_hash(
        PROJECT_ROOT / spec.priority_archive,
        spec.priority_archive_sha256,
        name="priority archive",
    )
    _require_hash(
        PROJECT_ROOT / spec.priority_metadata,
        spec.priority_metadata_sha256,
        name="priority metadata",
    )
    _require_hash(args.logistic, LOGISTIC_SHA256, name="logistic calibrator")
    _require_hash(args.nonlinear, NONLINEAR_SHA256, name="nonlinear calibrator")
    _require_hash(
        PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        RAW_SOLVER_SHA256,
        name="frozen raw solver",
    )
    if (
        args.logistic.resolve() != DEFAULT_LOGISTIC.resolve()
        or args.nonlinear.resolve() != DEFAULT_NONLINEAR.resolve()
    ):
        raise ValueError("this fixed experiment accepts only the preregistered calibrators")
    if not args.targets.resolve().is_dir():
        raise ValueError(f"organizer-train target directory is absent: {args.targets}")
    rows = _load_rows(panel, smoke_one=bool(args.smoke_one))
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (PROJECT_ROOT / f"outputs/taska-multistart-portfolio/{panel}-v1").resolve()
    )
    started = perf_counter()
    frozen, metadata, freeze, inference_seconds = _freeze_target_free(
        panel=panel,
        rows=rows,
        output_dir=output_dir,
        workers=int(args.workers),
    )
    print(
        json.dumps(
            {
                "event": "taska_multistart_all_layouts_frozen",
                "panel": panel,
                "case_count": len(rows),
                "frozen_archive_sha256": sha256_file(frozen),
                "reference_reconstructed_yet": False,
            }
        ),
        flush=True,
    )
    scored, metrics = _score_after_freeze(
        panel=panel,
        rows=rows,
        targets=args.targets,
        frozen_path=frozen,
        metadata_path=metadata,
        freeze_path=freeze,
    )
    strict = all(
        row[name]["strict_permutation"]
        for row in scored
        for name in row
        if isinstance(row[name], Mapping) and "strict_permutation" in row[name]
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "smoke-only" if args.smoke_one else "evaluation-complete",
        "panel": panel,
        "case_count": len(scored),
        "candidate": {
            "fixed_seeds_no_sweep": list(TASKA_MULTISTART_SEEDS),
            "fixed_arms": list(TASKA_MULTISTART_ARMS),
            "only_solver_random_seed_changes": True,
            "candidate_membership_and_costs_unchanged": True,
            "selector": "minimum original TASKA all-bond seam cost",
            "protected_tail_max_swaps": TASKA_MULTISTART_TAIL_SWAPS,
            "targets_used_only_after_all_layouts_were_hash_frozen": True,
            "solver_config_except_multistart_seed": asdict(SOLVER_CONFIG),
        },
        "frozen_eval": {
            "archive": _record(frozen),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
            "contains_exact_references_or_labels": False,
        },
        "metrics": metrics,
        "measurement": {"all_layouts_strict": strict, "valid": strict},
        "rows": scored,
        "timing": {
            "target_free_inference_seconds": inference_seconds,
            "total_seconds": perf_counter() - started,
            "worker_count": int(args.workers),
        },
    }
    report_path = output_dir / "report.json"
    _write_json_exclusive(report_path, report)
    print(
        json.dumps(
            {
                "event": "taska_multistart_evaluation_complete",
                "panel": panel,
                "report": str(report_path),
                "metrics": metrics,
                "valid": strict,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run(parse_args())
