#!/usr/bin/env python3
"""Combine the fixed fullres union voter with the fixed focal-gated tail96.

No matcher, selector, threshold or budget is rerun or tuned.  The five-arm
pre-tail winner and its candidate supply are reconstructed from the SHA-frozen
fullres fixed-v1 artifacts.  Candidate layouts are frozen before exact
references are reconstructed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_focal_gated_protected_tail import (
    FOCAL_PROTECTION_LOGIT_THRESHOLD,
)
from aiijc_puzzle.taska_fullres_focal_gated_tail import (
    polish_fullres_winner_with_focal_gate,
)
from aiijc_puzzle.taska_protected_tail_polish import polish_unprotected_taska_tail

try:
    from scripts import run_taska_fullres_union_voter as fullres_parent
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_fullres_union_voter as fullres_parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-fullres-focal-gated-tail/fixed-v1"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
FULLRES_OUTPUT = PROJECT_ROOT / "outputs/taska-fullres-union-voter/fixed-v1"
FULLRES_REPORT = FULLRES_OUTPUT / "report.json"
PAIR_DENOMINATOR = 1104
TAIL_SWAPS = 96
MINIMUM_GAIN = 1e-9
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_187
ARMS = (
    "four_arm_control_tail96",
    "fullres_five_arm_tail96",
    "combo_focal_gated_tail96",
)
REPORT_SCHEMA = "aiijc-taska-fullres-focal-gated-tail-report-v1"

FIXED_INPUTS = {
    FULLRES_REPORT: "d67a7ed7e2cd9e7c333052ab4db9d0b32e444980da83939f0e54e7f88c7195b8",
    FULLRES_OUTPUT / "local32/frozen-target-free-eval.npz": (
        "17dd26e11fbbaf8d79d66c122a1ca7abfbe6e34d6c9a149ed38420381845946d"
    ),
    FULLRES_OUTPUT / "local32/frozen-target-free-eval.json": (
        "e5574bbea6ff83bb5caded5152d14ed22d56e9043229ee577cff7bda65f9ea20"
    ),
    FULLRES_OUTPUT / "local32/pre-score-freeze.json": (
        "dfce253658ff99322cc0c3a08629602f805873def8be23b390e68eea4d7add65"
    ),
    FULLRES_OUTPUT / "held32/frozen-target-free-eval.npz": (
        "a6eef323aecacdf6285b5095d004b0c3f141e1a5ab1958cc8639e7c964f4c48e"
    ),
    FULLRES_OUTPUT / "held32/frozen-target-free-eval.json": (
        "116b5e77bc01329ffab0f850b4aaa79194d79a0ac6239f0243142c9752724b13"
    ),
    FULLRES_OUTPUT / "held32/pre-score-freeze.json": (
        "0f3f44e0bd7d36ba49ed19a469d10b7b5ecda9ed4871dd60cec326e048c2deff"
    ),
    FULLRES_OUTPUT / "fresh32/frozen-target-free-eval.npz": (
        "06805de4cd0d76b3007f0af620e862d1b062413c9629861d0be1825e1e538af0"
    ),
    FULLRES_OUTPUT / "fresh32/frozen-target-free-eval.json": (
        "af8e1a398bbdbb069870e5224857e758720eb252f2314bd9cf0c527d351c5f3f"
    ),
    FULLRES_OUTPUT / "fresh32/pre-score-freeze.json": (
        "69fdd637e7684398a7938442bfb8d808f225c56b3dbe95a69250a3125c7d6547"
    ),
    PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py": (
        "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
    ),
    PROJECT_ROOT / "src/aiijc_puzzle/taska_fullres_union_voter.py": (
        "9bf412349380d96ec6f5529a7775870843a2cc99a3251bc0e0cf86b2bb3fbd26"
    ),
    PROJECT_ROOT / "scripts/run_taska_fullres_union_voter.py": (
        "e5e7d79e944693f81b8d391161f75ecb2e0dc4f43a0f413074b8a8db6a7c5e34"
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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


def _require_inputs() -> None:
    for path, expected in {**fullres_parent.EXPECTED_SHA256, **FIXED_INPUTS}.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA-256 mismatch: {path}")


def _fixed_paths(stage: str) -> tuple[Path, Path, Path]:
    root = FULLRES_OUTPUT / stage
    return (
        root / "frozen-target-free-eval.npz",
        root / "frozen-target-free-eval.json",
        root / "pre-score-freeze.json",
    )


def _identity(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["prefix"]),
        str(row["source_filename"]),
        int(row["draw_index"]),
        str(row["dirty_sha256"]),
    )


def _accepted_logits(
    proposed: Sequence[RawTailEdge],
    proposed_logits: Any,
    accepted: Sequence[RawTailEdge],
) -> np.ndarray:
    logits = np.asarray(proposed_logits, dtype=np.float64)
    if logits.shape != (len(proposed),) or not np.isfinite(logits).all():
        raise ValueError("proposed focal logits are malformed")
    lookup = dict(zip(proposed, logits, strict=True))
    if len(lookup) != len(proposed) or not set(accepted) <= set(proposed):
        raise ValueError("accepted edges are not a subset of proposed edges")
    result = np.asarray([lookup[edge] for edge in accepted], dtype=np.float64)
    if len(result) and float(result.min()) < FOCAL_PROTECTION_LOGIT_THRESHOLD:
        raise RuntimeError("fixed accepted new edge violates focal logit-zero gate")
    return result


def _pre_tail_layout(
    choice: str,
    *,
    prefix: str,
    fixed_archive: Any,
    layout_archive: Any,
) -> np.ndarray:
    if choice == "fullres_union_focal":
        return fullres_parent.strict_layout(
            fixed_archive[f"{prefix}__fullres_union_focal_layout"]
        )
    layouts = fullres_parent._four_layouts(layout_archive, prefix)
    if choice not in layouts:
        raise ValueError(f"unexpected five-arm choice: {choice}")
    return layouts[choice]


def _edge_arrays(prefix: str, name: str, edges: Sequence[RawTailEdge]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__{name}__edge_source": np.asarray(
            [edge.source for edge in edges], dtype=np.int32
        ),
        f"{prefix}__{name}__edge_target": np.asarray(
            [edge.target for edge in edges], dtype=np.int32
        ),
        f"{prefix}__{name}__edge_axis": np.asarray(
            [edge.axis == "down" for edge in edges], dtype=np.uint8
        ),
    }


def _freeze(
    *,
    stage: str,
    spec: Any,
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path, Path]:
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    archive = stage_dir / "frozen-target-free-eval.npz"
    metadata = stage_dir / "frozen-target-free-eval.json"
    freeze = stage_dir / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-fullres-focal-gated-tail-target-free-v1",
            "stage": stage,
            "contains_exact_references_or_labels": False,
            "fullres_pre_tail_winner_reused": True,
            "focal_logit_threshold": FOCAL_PROTECTION_LOGIT_THRESHOLD,
            "tail_max_swaps": TAIL_SWAPS,
            "threshold_or_budget_sweep": False,
            "all_layouts_strict_original_upright_tile_permutations": True,
            "rows": list(rows),
        },
    )
    fixed_archive, fixed_metadata, fixed_freeze = _fixed_paths(stage)
    sources = {
        "candidate_archive": archive,
        "candidate_metadata": metadata,
        "runner": Path(__file__).resolve(),
        "composition_module": PROJECT_ROOT
        / "src/aiijc_puzzle/taska_fullres_focal_gated_tail.py",
        "focal_tail_module": PROJECT_ROOT
        / "src/aiijc_puzzle/taska_focal_gated_protected_tail.py",
        "protected_tail_module": PROJECT_ROOT
        / "src/aiijc_puzzle/taska_protected_tail_polish.py",
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
        "fullres_fixed_archive": fixed_archive,
        "fullres_fixed_metadata": fixed_metadata,
        "fullres_fixed_pre_score_freeze": fixed_freeze,
        "base_archive": spec.base_archive,
        "base_metadata": spec.base_metadata,
        "layout_archive": spec.layout_archive,
        "layout_metadata": spec.layout_metadata,
        "focal_archive": spec.focal_archive,
        "focal_metadata": spec.focal_metadata,
    }
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-fullres-focal-gated-tail-pre-score-freeze-v1",
            "stage": stage,
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {name: _record(path) for name, path in sources.items()},
        },
    )
    return archive, metadata, freeze


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("pre-score timing contract differs")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze contains labels")
    for name, record in payload["artifacts"].items():
        artifact = Path(record["path"])
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"frozen artifact changed before scoring: {name}")


def _generate_panel(
    *,
    stage: str,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    spec = fullres_parent.PANELS[stage]
    fixed_archive_path, fixed_metadata_path, fixed_freeze_path = _fixed_paths(stage)
    fullres_parent._validate_freeze(fixed_freeze_path)
    aligned = fullres_parent._aligned_rows(spec)
    fixed_rows = json.loads(fixed_metadata_path.read_text(encoding="utf-8"))["rows"]
    if len(fixed_rows) != 32 or len(aligned) != 32:
        raise ValueError(f"{stage} must contain exactly 32 cases")
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    with (
        np.load(fixed_archive_path, allow_pickle=False) as fixed_archive,
        np.load(spec.base_archive, allow_pickle=False) as base_archive,
        np.load(spec.layout_archive, allow_pickle=False) as layout_archive,
        np.load(spec.focal_archive, allow_pickle=False) as focal_archive,
    ):
        for index, (records, fixed_row) in enumerate(
            zip(aligned, fixed_rows, strict=True)
        ):
            if _identity(records[0]) != _identity(fixed_row):
                raise RuntimeError(f"{stage} fullres and base row identities differ")
            prefix, source, draw, dirty_sha = _identity(fixed_row)
            right = fullres_parent._matrix(base_archive, f"{prefix}__cost_right")
            down = fullres_parent._matrix(base_archive, f"{prefix}__cost_down")
            current = fullres_parent._edges(base_archive, prefix)
            current_logits = np.asarray(
                focal_archive[f"{prefix}__focal_logits"], dtype=np.float64
            )
            if current_logits.shape != (len(current),) or not np.isfinite(
                current_logits
            ).all():
                raise ValueError("current focal logits are not edge-aligned")
            proposed = fullres_parent._edges(fixed_archive, f"{prefix}__proposed")
            accepted = fullres_parent._edges(fixed_archive, f"{prefix}__accepted")
            new_logits = _accepted_logits(
                proposed,
                fixed_archive[f"{prefix}__proposed_focal_logits"],
                accepted,
            )
            choice = str(fixed_row["five_arm_choice"])
            winner_is_fullres = choice == "fullres_union_focal"
            pre_tail = _pre_tail_layout(
                choice,
                prefix=prefix,
                fixed_archive=fixed_archive,
                layout_archive=layout_archive,
            )
            winner_edges = current + accepted if winner_is_fullres else current
            replay = polish_unprotected_taska_tail(
                pre_tail,
                right,
                down,
                winner_edges,
                grid=fullres_parent.GRID,
                max_swaps=TAIL_SWAPS,
                minimum_gain=MINIMUM_GAIN,
            )
            known_fullres = fullres_parent.strict_layout(
                fixed_archive[f"{prefix}__five_arm_tail96_layout"]
            )
            if not np.array_equal(replay.layout, known_fullres):
                raise RuntimeError("all-edge tail did not replay frozen fullres control")
            combo = polish_fullres_winner_with_focal_gate(
                pre_tail,
                right,
                down,
                current,
                current_logits,
                accepted,
                new_logits,
                winner_is_fullres=winner_is_fullres,
                grid=fullres_parent.GRID,
            )
            four_control = fullres_parent.strict_layout(
                fixed_archive[f"{prefix}__control_tail96_layout"]
            )
            arrays[f"{prefix}__pre_tail_winner_layout"] = pre_tail
            arrays[f"{prefix}__four_arm_control_tail96_layout"] = four_control
            arrays[f"{prefix}__fullres_five_arm_tail96_layout"] = known_fullres
            arrays[f"{prefix}__combo_focal_gated_tail96_layout"] = combo.layout
            arrays.update(_edge_arrays(prefix, "current", current))
            arrays.update(_edge_arrays(prefix, "accepted", accepted))
            arrays[f"{prefix}__current_focal_logits"] = current_logits.astype(
                np.float32
            )
            arrays[f"{prefix}__accepted_focal_logits"] = new_logits.astype(
                np.float32
            )
            rows.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "dirty_sha256": dirty_sha,
                    "five_arm_choice": choice,
                    "winner_is_fullres": winner_is_fullres,
                    "all_edge_control_replayed": True,
                    "combo": asdict(combo.diagnostics),
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{stage}_combo_target_free",
                        "case": index + 1,
                        "case_count": 32,
                        "winner": choice,
                        "kept": combo.diagnostics.focal_gate.focal_kept_edge_count,
                    }
                ),
                flush=True,
            )
    return _freeze(
        stage=stage,
        spec=spec,
        output_dir=output_dir,
        arrays=arrays,
        rows=rows,
    )


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
        "case_wins_ties_losses": {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        },
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    summary: dict[str, Any] = {
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([row["metrics"][arm][metric] for row in rows]))
                for metric in metrics
            }
            for arm in ARMS
        },
        "five_arm_choice_counts": dict(
            Counter(row["five_arm_choice"] for row in frozen_rows)
        ),
        "target_free_diagnostics": {
            "mean_current_edges": float(
                np.mean([row["combo"]["current_edge_count"] for row in frozen_rows])
            ),
            "mean_accepted_new_edges": float(
                np.mean(
                    [row["combo"]["accepted_new_edge_count"] for row in frozen_rows]
                )
            ),
            "mean_focal_kept_edges": float(
                np.mean(
                    [
                        row["combo"]["focal_gate"]["focal_kept_edge_count"]
                        for row in frozen_rows
                    ]
                )
            ),
            "mean_combo_protected_tiles": float(
                np.mean(
                    [
                        row["combo"]["focal_gate"]["tail"]["protected_tile_count"]
                        for row in frozen_rows
                    ]
                )
            ),
            "mean_combo_accepted_swaps": float(
                np.mean(
                    [
                        row["combo"]["focal_gate"]["tail"]["accepted_swap_count"]
                        for row in frozen_rows
                    ]
                )
            ),
        },
    }
    sources = [str(row["source_filename"]) for row in rows]
    comparisons = {
        "combo_minus_fullres": (ARMS[2], ARMS[1]),
        "combo_minus_four_arm": (ARMS[2], ARMS[0]),
    }
    for comparison_index, (name, (candidate, control)) in enumerate(
        comparisons.items()
    ):
        summary[name] = {
            metric: _cluster_ci(
                [
                    float(row["metrics"][candidate][metric])
                    - float(row["metrics"][control][metric])
                    for row in rows
                ],
                sources,
                seed=BOOTSTRAP_SEED + 10 * comparison_index + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }
    return summary


def _score_panel(
    *,
    archive: Path,
    metadata: Path,
    freeze: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_freeze(freeze)
    frozen_rows = json.loads(metadata.read_text(encoding="utf-8"))["rows"]
    scored: list[dict[str, Any]] = []
    with np.load(archive, allow_pickle=False) as candidate:
        for row in frozen_rows:
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty = fullres_parent.finetune._dirty_case(
                cache, lookup[source], source, draw
            )
            if fullres_parent.finetune._dirty_sha256(dirty.dirty_tiles) != row[
                "dirty_sha256"
            ]:
                raise RuntimeError("scoring recreated different dirty bytes")
            reference = fullres_parent.finetune._reference(
                cache, lookup[source], source, draw, dirty.dirty_tiles
            )
            scored.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "metrics": {
                        arm: fullres_parent._layout_metrics(
                            candidate[f"{prefix}__{arm}_layout"], reference
                        )
                        for arm in ARMS
                    },
                }
            )
    return scored, _summarize(scored, frozen_rows)


def _run_panel(
    *,
    stage: str,
    output_dir: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> dict[str, Any]:
    started = perf_counter()
    archive, metadata, freeze = _generate_panel(stage=stage, output_dir=output_dir)
    rows, summary = _score_panel(
        archive=archive,
        metadata=metadata,
        freeze=freeze,
        lookup=lookup,
        cache=cache,
    )
    return {
        "status": "complete",
        "summary": summary,
        "rows": rows,
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_inputs()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    config, _, _ = fullres_parent.finetune._load_config(
        fullres_parent.finetune.DEFAULT_CONFIG
    )
    lookup = fullres_parent.finetune._manifest_lookup(config)
    cache = fullres_parent.finetune.CleanTileCache(
        args.targets.resolve(), maximum_boards=2
    )
    local = _run_panel(
        stage="local32", output_dir=output_dir, lookup=lookup, cache=cache
    )
    local_delta = local["summary"]["combo_minus_fullres"][
        "satisfied_adjacent_pairs"
    ]["mean"]
    held: dict[str, Any] = {"status": "skipped_by_negative_local_combo_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_combo_gate"}
    if local_delta >= 0:
        held = _run_panel(
            stage="held32", output_dir=output_dir, lookup=lookup, cache=cache
        )
        held_delta = held["summary"]["combo_minus_fullres"][
            "satisfied_adjacent_pairs"
        ]["mean"]
        if held_delta >= 0.5:
            fresh = _run_panel(
                stage="fresh32", output_dir=output_dir, lookup=lookup, cache=cache
            )
        else:
            fresh = {"status": "skipped_by_held_combo_delta_below_0.5"}
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "protocol": {
            "fixed_fullres_pre_tail_winner_reused": True,
            "winner_fullres_uses_current_plus_accepted_new": True,
            "old_winner_uses_current_only": True,
            "focal_logit_threshold": FOCAL_PROTECTION_LOGIT_THRESHOLD,
            "tail_max_swaps": TAIL_SWAPS,
            "tail_minimum_gain": MINIMUM_GAIN,
            "local_gate": "combo minus fullres pair delta >= 0",
            "held_gate": "combo minus fullres pair delta >= +0.5",
            "threshold_or_budget_sweep": False,
        },
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "target_free_candidates_frozen_before_reference": True,
            "all_outputs_strict_original_upright_tile_permutations": True,
            "raw_dense_costs_unchanged": True,
            "restored_pixels_matcher_only": True,
            "competition_test_accessed": False,
            "postprocessing_used": False,
        },
        "artifacts": {
            "fullres_parent_report": _record(FULLRES_REPORT),
            "runner": _record(Path(__file__).resolve()),
            "composition_module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/taska_fullres_focal_gated_tail.py"
            ),
            "raw_solver": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
            ),
        },
    }
    _write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {key: report[key] for key in ("local32", "held32", "fresh32")},
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    run(parse_args())
