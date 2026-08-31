#!/usr/bin/env python3
"""Evaluate one preregistered seven-arm confirmed TASKA portfolio.

The frozen six-arm selective/fullres-fusion candidate is the exact control.
Exactly one independently confirmed standalone fullres-union pre-tail arm is
added.  No matcher, denoiser, threshold, budget, seed, weight, or learned
selector is rerun or swept.  Candidate layouts are frozen before synthetic
organizer-train references are reconstructed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_confirmed_arm_portfolio import (
    CONFIRMED_ARM_NAMES,
    FULLRES_ARM,
    compose_confirmed_arm_portfolio,
)
from aiijc_puzzle.taska_pair_pipeline import PAIR_DENOMINATOR
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES

try:
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_selective_fullres_fusion as fusion
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_current_finetune as finetune
    import run_taska_selective_fullres_fusion as fusion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-confirmed-arm-portfolio/fixed-v1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_confirmed_arm_portfolio_v1.json"
GRID = 24
COUNT = GRID * GRID
LOCAL_GATE = 0.0
HELD_GATE = 0.5
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_210
REPORT_SCHEMA = "aiijc-taska-confirmed-arm-portfolio-report-v1"
SCORED_ARMS = ("confirmed_six_arm_control", "seven_arm_candidate")


@dataclass(frozen=True)
class PanelSpec:
    name: str
    case_count: int
    parent: fusion.PanelSpec
    fusion_archive: Path
    fusion_metadata: Path
    fusion_freeze: Path


FUSION_ROOT = PROJECT_ROOT / "outputs/taska-selective-fullres-union-fusion/fixed-v1"


def _panel(name: str) -> PanelSpec:
    root = FUSION_ROOT / name
    return PanelSpec(
        name=name,
        case_count=32,
        parent=fusion.PANELS[name],
        fusion_archive=root / "frozen-target-free-eval.npz",
        fusion_metadata=root / "frozen-target-free-eval.json",
        fusion_freeze=root / "pre-score-freeze.json",
    )


PANELS = {name: _panel(name) for name in ("local32", "held32", "fresh32")}
FUSION_SHA256 = {
    "local32": (
        "1b17c4a52ae80b58f973ee8aaffd20d0e1d9a125c1ac5e3acdc66f31abddf7df",
        "106ac31d166c1b244a498c3cc76f59d4730601e6fba3a35fa6721eb7f18befa1",
        "3b35db324f46a0368cad5c3f6570c08f9631560fe1f5f47f14defc77b8689720",
    ),
    "held32": (
        "6cfb766c1e693a2fec535d683f187a89f2d63632a282ff199e6aa708caafe469",
        "f37d23bd44c1565ae560c46ed6b6f33b4500b52168147ba114ff8debc59f0bf4",
        "aa5b53abbb3fe5b20900a2102e144f024bf563e42dbe37edd1086811515178bc",
    ),
    "fresh32": (
        "75a9359eb3ac798096437c22e269c8374a0a38bb01f8e7f9fa9745bd054180cb",
        "c65d7e332460001d67b2dc2052a2dd3a2e6c62d08f3a936c7723ecec6dac6794",
        "7fca88f9ea4489bf64d73a060127af73e1adaa598089eadfb79c1597627d5e93",
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--smoke-one",
        action="store_true",
        help="freeze one local case without reconstructing its reference",
    )
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rendered = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path}") from error


def _load_preregistered_config(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed preregistration config is missing")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if expected != actual:
        raise ValueError("preregistration SHA-256 sidecar mismatch")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "selector_roster": list(CONFIRMED_ARM_NAMES),
        "control_roster": list(FUSION_ARM_NAMES),
        "selector": "minimum original TASKA all-1104-bond raw seam cost",
        "tail": "winner-aligned focal-logit-zero non-adjacent tail96",
        "local_pair_gate": LOCAL_GATE,
        "held_pair_gate": HELD_GATE,
        "no_sweep": True,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"preregistration contract mismatch: {key}")
    return config, actual


def _require_inputs() -> None:
    fusion._require_inputs()
    for name, spec in PANELS.items():
        paths = (spec.fusion_archive, spec.fusion_metadata, spec.fusion_freeze)
        for path, expected in zip(paths, FUSION_SHA256[name], strict=True):
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"frozen fusion input SHA-256 mismatch: {path}")
        fusion._validate_freeze(spec.fusion_freeze)


def _rows(path: Path, case_count: int) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) < case_count:
        raise ValueError(f"{path} has fewer than {case_count} rows")
    return rows[:case_count]


def _aligned_rows(spec: PanelSpec) -> list[tuple[Mapping[str, Any], ...]]:
    parent = fusion._aligned_rows(replace(spec.parent, case_count=spec.case_count))
    fusion_rows = _rows(spec.fusion_metadata, spec.case_count)
    identity = ("prefix", "source_filename", "draw_index", "dirty_sha256")
    aligned: list[tuple[Mapping[str, Any], ...]] = []
    for base_records, fused in zip(parent, fusion_rows, strict=True):
        if any(base_records[0].get(field) != fused.get(field) for field in identity):
            raise RuntimeError(f"{spec.name} frozen fusion row identity mismatch")
        aligned.append((*base_records, fused))
    return aligned


def _layout_metrics(layout: Any, reference: Any) -> dict[str, Any]:
    result = evaluate_layout(layout, reference, reference_is_exact=True)
    if result.adjacency_total != PAIR_DENOMINATOR:
        raise RuntimeError("pair denominator changed")
    return {
        "satisfied_adjacent_pairs": int(result.adjacency_correct),
        "adjacency_recall": float(result.adjacency),
        "exact_tiles": int(result.correct_tile_count),
        "strict_original_upright_permutation": True,
    }


def _cluster_ci(values: Sequence[float], sources: Sequence[str], *, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, source in zip(values, sources, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("bootstrap values must be finite")
        grouped[source].append(float(value))
    means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    generator = np.random.default_rng(seed)
    distribution = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 2048):
        stop = min(start + 2048, BOOTSTRAP_RESAMPLES)
        indices = generator.integers(0, len(means), size=(stop - start, len(means)))
        distribution[start:stop] = means[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "source_count": len(means),
        "case_count": len(values),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    result: dict[str, Any] = {
        "case_count": len(rows),
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([row["metrics"][arm][metric] for row in rows]))
                for metric in metrics
            }
            for arm in SCORED_ARMS
        },
        "choice_counts": dict(Counter(row["choice"] for row in rows)),
        "control_replay_match_count": sum(
            bool(row["mechanical_control_matches_frozen"]) for row in rows
        ),
    }
    sources = [str(row["source_filename"]) for row in rows]
    deltas: dict[str, Any] = {}
    for index, metric in enumerate(metrics):
        values = [
            float(row["metrics"]["seven_arm_candidate"][metric])
            - float(row["metrics"]["confirmed_six_arm_control"][metric])
            for row in rows
        ]
        summary = _cluster_ci(values, sources, seed=BOOTSTRAP_SEED + index)
        summary["case_wins_ties_losses"] = {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }
        deltas[metric] = summary
    result["candidate_minus_control"] = deltas
    return result


def _freeze(
    spec: PanelSpec,
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    config_path: Path,
) -> tuple[Path, Path, Path]:
    stage = output_dir / spec.name
    stage.mkdir(parents=True, exist_ok=False)
    archive = stage / "frozen-target-free-eval.npz"
    metadata = stage / "frozen-target-free-eval.json"
    freeze = stage / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-confirmed-arm-portfolio-target-free-v1",
            "stage": spec.name,
            "contains_exact_references_or_candidate_labels": False,
            "matcher_rerun": False,
            "selector_roster": list(CONFIRMED_ARM_NAMES),
            "control_roster": list(FUSION_ARM_NAMES),
            "tail": "winner-aligned focal-logit-zero non-adjacent tail96",
            "all_layouts_strict_original_upright_permutations": True,
            "rows": list(rows),
        },
    )
    artifacts = {
        "frozen_archive": archive,
        "frozen_metadata": metadata,
        "preregistration": config_path,
        "fusion_archive": spec.fusion_archive,
        "fusion_metadata": spec.fusion_metadata,
        "fusion_parent_freeze": spec.fusion_freeze,
        "fullres_archive": spec.parent.fullres_archive,
        "fullres_metadata": spec.parent.fullres_metadata,
        "fullres_parent_freeze": spec.parent.fullres_freeze,
        "runner": Path(__file__).resolve(),
        "portfolio_module": (
            PROJECT_ROOT / "src/aiijc_puzzle/taska_confirmed_arm_portfolio.py"
        ),
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
    }
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-confirmed-arm-portfolio-pre-score-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {name: _record(path) for name, path in artifacts.items()},
        },
    )
    return archive, metadata, freeze


def _score_panel(
    *,
    archive: Path,
    metadata: Path,
    freeze: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fusion._validate_freeze(freeze)
    frozen_rows = json.loads(metadata.read_text(encoding="utf-8"))["rows"]
    scored: list[dict[str, Any]] = []
    with np.load(archive, allow_pickle=False) as candidate:
        for frozen in frozen_rows:
            prefix = str(frozen["prefix"])
            source = str(frozen["source_filename"])
            draw = int(frozen["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != frozen["dirty_sha256"]:
                raise RuntimeError("scoring recreated different dirty bytes")
            reference = finetune._reference(cache, lookup[source], source, draw, dirty.dirty_tiles)
            metrics = {
                arm: _layout_metrics(candidate[f"{prefix}__{arm}_layout"], reference)
                for arm in SCORED_ARMS
            }
            scored.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "choice": frozen["choice"],
                    "control_choice": frozen["control_choice"],
                    "mechanical_control_matches_frozen": frozen[
                        "mechanical_control_matches_frozen"
                    ],
                    "metrics": metrics,
                }
            )
    return scored, _summarize(scored)


def _run_panel(
    spec: PanelSpec,
    *,
    output_dir: Path,
    config_path: Path,
    lookup: Mapping[str, Mapping[str, Any]] | None,
    cache: Any | None,
    target_free_only: bool,
) -> dict[str, Any]:
    aligned = _aligned_rows(spec)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    parent = spec.parent
    with (
        np.load(parent.layout_archive, allow_pickle=False) as layouts,
        np.load(parent.base_archive, allow_pickle=False) as base,
        np.load(parent.fullres_archive, allow_pickle=False) as fullres,
        np.load(spec.fusion_archive, allow_pickle=False) as fused,
    ):
        for index, records in enumerate(aligned):
            row = records[-1]
            prefix = str(row["prefix"])
            current = fusion._edges(fused, prefix, "current")
            selective_new = fusion._edges(fused, prefix, "selective_new")
            fullres_accepted = fusion._edges(fused, prefix, "fullres_accepted")
            combined = fusion._edges(fused, prefix, "combined_union")
            selective_union = current + selective_new
            fullres_union = current + fullres_accepted
            if len(set(fullres_union)) != len(fullres_union):
                raise RuntimeError("standalone fullres union contains an overlap")
            current_logits = np.asarray(fused[f"{prefix}__current_focal_logits"])
            selective_logits = np.concatenate(
                (
                    current_logits,
                    np.asarray(fused[f"{prefix}__selective_new_focal_logits"]),
                )
            )
            combined_logits = np.asarray(fused[f"{prefix}__combined_union_focal_logits"])
            fullres_logits = np.concatenate(
                (
                    current_logits,
                    np.asarray(fused[f"{prefix}__fullres_accepted_focal_logits"]),
                )
            )
            result = compose_confirmed_arm_portfolio(
                cost_right=fusion._matrix(base, f"{prefix}__cost_right"),
                cost_down=fusion._matrix(base, f"{prefix}__cost_down"),
                four_layouts=fusion._four_layouts(layouts, prefix),
                selective_union_layout=fused[f"{prefix}__selective_union_layout"],
                combined_union_layout=fused[f"{prefix}__combined_union_layout"],
                fullres_union_layout=fullres[f"{prefix}__fullres_union_focal_layout"],
                frozen_fusion_control=fused[f"{prefix}__combined_union_candidate_layout"],
                current_edges=current,
                current_logits=current_logits,
                selective_union_edges=selective_union,
                selective_union_logits=selective_logits,
                combined_union_edges=combined,
                combined_union_logits=combined_logits,
                fullres_union_edges=fullres_union,
                fullres_union_logits=fullres_logits,
            )
            replay = bool(np.array_equal(result.mechanical_control_layout, result.control_layout))
            if not replay:
                raise RuntimeError("mechanical six-arm control mismatch")
            arrays[f"{prefix}__confirmed_six_arm_control_layout"] = result.control_layout
            arrays[f"{prefix}__seven_arm_candidate_layout"] = result.candidate_layout
            arrays[f"{prefix}__mechanical_control_layout"] = result.mechanical_control_layout
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": str(row["source_filename"]),
                    "draw_index": int(row["draw_index"]),
                    "dirty_sha256": str(row["dirty_sha256"]),
                    "mechanical_control_matches_frozen": replay,
                    **result.diagnostics(),
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{spec.name}_confirmed_arm_portfolio_target_free",
                        "case": index + 1,
                        "case_count": len(aligned),
                        "choice": result.choice,
                        "control_choice": result.control_choice,
                        "control_replay": replay,
                    }
                ),
                flush=True,
            )
    archive, metadata, freeze = _freeze(
        spec,
        output_dir,
        arrays,
        frozen_rows,
        config_path,
    )
    payload: dict[str, Any] = {
        "status": "target-free-smoke" if target_free_only else "complete",
        "target_free_summary": {
            "case_count": len(frozen_rows),
            "control_replay_match_count": sum(
                row["mechanical_control_matches_frozen"] for row in frozen_rows
            ),
            "choice_counts": dict(Counter(row["choice"] for row in frozen_rows)),
            "new_fullres_winner_count": sum(
                row["choice"] == FULLRES_ARM for row in frozen_rows
            ),
        },
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
        },
    }
    if not target_free_only:
        if lookup is None or cache is None:
            raise RuntimeError("scoring resources are absent")
        scored, summary = _score_panel(
            archive=archive,
            metadata=metadata,
            freeze=freeze,
            lookup=lookup,
            cache=cache,
        )
        payload.update({"rows": scored, "summary": summary})
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_inputs()
    config, config_sha256 = _load_preregistered_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    if args.smoke_one:
        smoke = replace(PANELS["local32"], name="smoke1", case_count=1)
        local = _run_panel(
            smoke,
            output_dir=output_dir,
            config_path=args.config.resolve(),
            lookup=None,
            cache=None,
            target_free_only=True,
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": "target-free-smoke",
            "local32": local,
            "preregistration_sha256": config_sha256,
            "reference_reconstructed": False,
            "competition_test_accessed": False,
        }
        _write_json(output_dir / "report.json", report)
        print(json.dumps(report, indent=2))
        return report

    source_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(source_config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    local = _run_panel(
        PANELS["local32"],
        output_dir=output_dir,
        config_path=args.config.resolve(),
        lookup=lookup,
        cache=cache,
        target_free_only=False,
    )
    local_delta = local["summary"]["candidate_minus_control"]["satisfied_adjacent_pairs"]["mean"]
    held: dict[str, Any] = {"status": "skipped_by_negative_local_pair_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_pair_gate"}
    if local_delta >= LOCAL_GATE:
        held = _run_panel(
            PANELS["held32"],
            output_dir=output_dir,
            config_path=args.config.resolve(),
            lookup=lookup,
            cache=cache,
            target_free_only=False,
        )
        held_delta = held["summary"]["candidate_minus_control"][
            "satisfied_adjacent_pairs"
        ]["mean"]
        if held_delta >= HELD_GATE:
            fresh = _run_panel(
                PANELS["fresh32"],
                output_dir=output_dir,
                config_path=args.config.resolve(),
                lookup=lookup,
                cache=cache,
                target_free_only=False,
            )
        else:
            fresh = {"status": "skipped_by_held_pair_delta_below_0.5"}
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "protocol": config,
        "preregistration_sha256": config_sha256,
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "strict_original_upright_tile_permutations": True,
            "pixels_restored_replaced_rotated_or_warped": False,
            "targets_used_only_after_candidate_freeze": True,
            "competition_test_accessed": False,
            "postprocessing_used": False,
            "production_modified": False,
        },
        "artifacts": {
            "runner": _record(Path(__file__).resolve()),
            "portfolio_module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/taska_confirmed_arm_portfolio.py"
            ),
            "preregistration": _record(args.config.resolve()),
        },
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps({name: report[name] for name in PANELS}, indent=2))
    return report


if __name__ == "__main__":
    run(parse_args())
