#!/usr/bin/env python3
"""Evaluate one preregistered component-relation anchor over TASKA six-arm.

The candidate is deliberately narrow: after the frozen confirmed six-arm
layout has been built, selected-supply edges may propose rigid translations
of one already-realised component.  Exactly one proposal can be accepted and
only when the original all-bond seam objective strictly improves.  Candidate
layouts are written and hashed before organizer-train references are rebuilt.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_component_relation_anchor import (
    anchor_one_component_from_relation_votes,
)
from aiijc_puzzle.taska_pair_pipeline import PAIR_DENOMINATOR

try:
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_selective_fullres_fusion as fusion
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_current_finetune as finetune
    import run_taska_selective_fullres_fusion as fusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-component-relation-anchor/fixed-v1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_component_relation_anchor_v1.json"
FUSION_ROOT = PROJECT_ROOT / "outputs/taska-selective-fullres-union-fusion/fixed-v1"
GRID = 24
CONTROL_ARM = "confirmed_six_arm_fusion"
CANDIDATE_ARM = "one_component_relation_anchor"
ARMS = (CONTROL_ARM, CANDIDATE_ARM)
FOCAL_THRESHOLD = 0.0
MINIMUM_COST_GAIN = 0.0
LOCAL_PAIR_GATE = 0.0
HELD_EXACT_GATE = 0.0
HELD_PAIR_GATE = 0.0


@dataclass(frozen=True)
class PanelSpec:
    name: str
    case_count: int
    parent: fusion.PanelSpec
    fusion_archive: Path
    fusion_metadata: Path
    fusion_freeze: Path


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke-one", action="store_true")
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


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed component-anchor preregistration is missing")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise ValueError("component-anchor preregistration SHA-256 mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    required = {
        "candidate": "one post-tail rigid component translation from relation votes",
        "control": "exact frozen confirmed six-arm fusion final layout",
        "component_focal_threshold": FOCAL_THRESHOLD,
        "minimum_all_bond_cost_gain": MINIMUM_COST_GAIN,
        "local_exact_gate": "strictly_positive",
        "local_pair_gate": LOCAL_PAIR_GATE,
        "held_exact_gate": HELD_EXACT_GATE,
        "held_pair_gate": HELD_PAIR_GATE,
        "no_sweep": True,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"component-anchor preregistration mismatch: {key}")
    for relative, expected in config["fixed_source_sha256"].items():
        target = PROJECT_ROOT / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"signed candidate source changed: {relative}")
    return config, digest


def _require_inputs() -> None:
    fusion._require_inputs()
    for name, spec in PANELS.items():
        for path, expected in zip(
            (spec.fusion_archive, spec.fusion_metadata, spec.fusion_freeze),
            FUSION_SHA256[name],
            strict=True,
        ):
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"frozen fusion SHA-256 mismatch: {path}")
        fusion._validate_freeze(spec.fusion_freeze)


def _rows(path: Path, count: int) -> list[Mapping[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows")
    if not isinstance(rows, list) or len(rows) < count:
        raise ValueError(f"{path} contains fewer than {count} rows")
    return rows[:count]


def _aligned_rows(spec: PanelSpec) -> list[tuple[Mapping[str, Any], ...]]:
    parent_rows = fusion._aligned_rows(replace(spec.parent, case_count=spec.case_count))
    frozen_rows = _rows(spec.fusion_metadata, spec.case_count)
    identity = ("prefix", "source_filename", "draw_index", "dirty_sha256")
    result: list[tuple[Mapping[str, Any], ...]] = []
    for records, frozen in zip(parent_rows, frozen_rows, strict=True):
        if any(records[0].get(field) != frozen.get(field) for field in identity):
            raise RuntimeError(f"{spec.name} frozen row identity mismatch")
        result.append((*records, frozen))
    return result


def _edge_family(
    archive: Any,
    prefix: str,
    name: str,
) -> tuple[tuple[RawTailEdge, ...], np.ndarray]:
    source = np.asarray(archive[f"{prefix}__{name}__edge_source"], dtype=np.int32)
    target = np.asarray(archive[f"{prefix}__{name}__edge_target"], dtype=np.int32)
    axis = np.asarray(archive[f"{prefix}__{name}__edge_axis"], dtype=np.uint8)
    logits = np.asarray(archive[f"{prefix}__{name}_focal_logits"], dtype=np.float64)
    if not (source.shape == target.shape == axis.shape == logits.shape):
        raise ValueError("selected-supply edge arrays are not aligned")
    edges = tuple(
        RawTailEdge(int(first), int(second), "right" if int(direction) == 0 else "down")
        for first, second, direction in zip(source, target, axis, strict=True)
    )
    return edges, logits


def _selected_supply(
    archive: Any,
    prefix: str,
    choice: str,
) -> tuple[tuple[RawTailEdge, ...], np.ndarray, str]:
    if choice == "combined_union_focal":
        names = ("combined_union",)
    elif choice == "selective_vote500_focal":
        names = ("current", "selective_new")
    else:
        names = ("current",)
    families = [_edge_family(archive, prefix, name) for name in names]
    edges = tuple(edge for family, _ in families for edge in family)
    logits = np.concatenate([values for _, values in families])
    return edges, logits, "+".join(names)


def _layout_metrics(layout: Any, reference: Any) -> dict[str, Any]:
    result = evaluate_layout(layout, reference, reference_is_exact=True)
    if result.adjacency_total != PAIR_DENOMINATOR:
        raise RuntimeError("pair denominator changed")
    return {
        "exact_tiles": int(result.correct_tile_count),
        "satisfied_adjacent_pairs": int(result.adjacency_correct),
        "adjacency_recall": float(result.adjacency),
        "strict_original_upright_permutation": True,
    }


def _delta_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "wins": int(np.count_nonzero(array > 0)),
        "ties": int(np.count_nonzero(array == 0)),
        "losses": int(np.count_nonzero(array < 0)),
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("exact_tiles", "satisfied_adjacent_pairs", "adjacency_recall")
    arms = {
        arm: {
            metric: float(np.mean([row["metrics"][arm][metric] for row in rows]))
            for metric in metrics
        }
        for arm in ARMS
    }
    deltas = {
        metric: _delta_summary(
            [
                row["metrics"][CANDIDATE_ARM][metric]
                - row["metrics"][CONTROL_ARM][metric]
                for row in rows
            ]
        )
        for metric in metrics
    }
    return {
        "case_count": len(rows),
        "pair_denominator": PAIR_DENOMINATOR,
        "arms": arms,
        "candidate_minus_control": deltas,
        "target_free_diagnostics": {
            "changed_layout_count": sum(row["changed"] for row in rows),
            "mean_relation_hypothesis_count": float(
                np.mean([row["relation_hypothesis_count"] for row in rows])
            ),
            "mean_cost_improving_hypothesis_count": float(
                np.mean([row["cost_improving_hypothesis_count"] for row in rows])
            ),
            "mean_selected_component_size": float(
                np.mean([row["selected_component_size"] for row in rows])
            ),
            "choice_counts": dict(Counter(str(row["fusion_choice"]) for row in rows)),
        },
    }


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("pre-score timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze contains labels")
    for record in payload["artifacts"].values():
        artifact = Path(record["path"])
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"pre-score artifact changed: {artifact}")


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
    stage = output_dir / spec.name
    stage.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    with (
        np.load(spec.parent.base_archive, allow_pickle=False) as base,
        np.load(spec.fusion_archive, allow_pickle=False) as fused,
    ):
        for index, records in enumerate(aligned):
            row = records[-1]
            prefix = str(row["prefix"])
            control = np.ascontiguousarray(
                fused[f"{prefix}__combined_union_candidate_layout"], dtype=np.int32
            )
            edges, logits, selected_family = _selected_supply(
                fused, prefix, str(row["choice"])
            )
            anchored = anchor_one_component_from_relation_votes(
                control,
                fusion._matrix(base, f"{prefix}__cost_right"),
                fusion._matrix(base, f"{prefix}__cost_down"),
                edges,
                logits,
                grid=GRID,
                focal_threshold=FOCAL_THRESHOLD,
                minimum_cost_gain=MINIMUM_COST_GAIN,
            )
            arrays[f"{prefix}__{CONTROL_ARM}_layout"] = control
            arrays[f"{prefix}__{CANDIDATE_ARM}_layout"] = anchored.layout
            diagnostics = asdict(anchored.diagnostics)
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": str(row["source_filename"]),
                    "draw_index": int(row["draw_index"]),
                    "dirty_sha256": str(row["dirty_sha256"]),
                    "fusion_choice": str(row["choice"]),
                    "selected_supply_family": selected_family,
                    **diagnostics,
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{spec.name}_component_relation_anchor_target_free",
                        "case": index + 1,
                        "case_count": len(aligned),
                        "changed": diagnostics["changed"],
                        "component_size": diagnostics["selected_component_size"],
                        "cost_gain": diagnostics["baseline_total_cost"]
                        - diagnostics["selected_total_cost"],
                    }
                ),
                flush=True,
            )
    archive = stage / "frozen-target-free-eval.npz"
    metadata = stage / "frozen-target-free-eval.json"
    freeze = stage / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-component-relation-anchor-target-free-v1",
            "contains_exact_references_or_candidate_labels": False,
            "candidate_family": (
                "one post-tail relation-implied rigid component shift with local fill"
            ),
            "component_definition": (
                "connected components of focal-logit>=0 selected-supply edges "
                "realised by the frozen six-arm layout"
            ),
            "selection": "maximum softplus relation vote among all-bond-improving moves",
            "strict_original_upright_permutations": True,
            "rows": frozen_rows,
        },
    )
    artifacts = {
        "archive": _record(archive),
        "metadata": _record(metadata),
        "preregistration": _record(config_path),
        "runner": _record(Path(__file__).resolve()),
        "solver": _record(
            PROJECT_ROOT / "src/aiijc_puzzle/taska_component_relation_anchor.py"
        ),
        "fusion_archive": _record(spec.fusion_archive),
        "fusion_metadata": _record(spec.fusion_metadata),
        "fusion_parent_freeze": _record(spec.fusion_freeze),
        "base_archive": _record(spec.parent.base_archive),
        "base_metadata": _record(spec.parent.base_metadata),
    }
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-component-relation-anchor-pre-score-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": artifacts,
        },
    )
    payload: dict[str, Any] = {
        "status": "target-free-smoke" if target_free_only else "complete",
        "target_free_summary": {
            "case_count": len(frozen_rows),
            "changed_layout_count": sum(row["changed"] for row in frozen_rows),
            "mean_relation_hypothesis_count": float(
                np.mean([row["relation_hypothesis_count"] for row in frozen_rows])
            ),
            "mean_cost_improving_hypothesis_count": float(
                np.mean([row["cost_improving_hypothesis_count"] for row in frozen_rows])
            ),
        },
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
        },
    }
    if target_free_only:
        return payload
    if lookup is None or cache is None:
        raise RuntimeError("scoring resources are absent")
    _validate_freeze(freeze)
    scored: list[dict[str, Any]] = []
    with np.load(archive, allow_pickle=False) as frozen:
        for row in frozen_rows:
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
                raise RuntimeError("scoring recreated different dirty bytes")
            reference = finetune._reference(cache, lookup[source], source, draw, dirty.dirty_tiles)
            scored.append(
                {
                    **row,
                    "metrics": {
                        arm: _layout_metrics(frozen[f"{prefix}__{arm}_layout"], reference)
                        for arm in ARMS
                    },
                }
            )
    payload.update({"rows": scored, "summary": _summarize(scored)})
    return payload


def _gate(panel: Mapping[str, Any], *, local: bool) -> bool:
    delta = panel["summary"]["candidate_minus_control"]
    exact = float(delta["exact_tiles"]["mean"])
    pairs = float(delta["satisfied_adjacent_pairs"]["mean"])
    if local:
        return exact > 0.0 and pairs >= LOCAL_PAIR_GATE
    return exact >= HELD_EXACT_GATE and pairs >= HELD_PAIR_GATE


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_inputs()
    protocol, protocol_sha256 = _load_config(args.config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.smoke_one:
        smoke = replace(PANELS["local32"], name="smoke1", case_count=1)
        result = _run_panel(
            smoke,
            output_dir=output_dir,
            config_path=args.config.resolve(),
            lookup=None,
            cache=None,
            target_free_only=True,
        )
        report = {
            "schema": "aiijc-taska-component-relation-anchor-report-v1",
            "status": "target-free-smoke",
            "preregistration_sha256": protocol_sha256,
            "smoke1": result,
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
    held: dict[str, Any] = {"status": "skipped_by_local_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_gate"}
    if _gate(local, local=True):
        held = _run_panel(
            PANELS["held32"],
            output_dir=output_dir,
            config_path=args.config.resolve(),
            lookup=lookup,
            cache=cache,
            target_free_only=False,
        )
        if _gate(held, local=False):
            fresh = _run_panel(
                PANELS["fresh32"],
                output_dir=output_dir,
                config_path=args.config.resolve(),
                lookup=lookup,
                cache=cache,
                target_free_only=False,
            )
        else:
            fresh = {"status": "skipped_by_held_gate"}
    report = {
        "schema": "aiijc-taska-component-relation-anchor-report-v1",
        "status": "complete",
        "protocol": protocol,
        "preregistration_sha256": protocol_sha256,
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "legality": {
            "strict_original_upright_tile_permutations": True,
            "pixels_changed_rotated_warped_replaced_or_postprocessed": False,
            "targets_used_only_after_each_panel_candidate_freeze": True,
            "competition_test_accessed": False,
            "production_modified": False,
        },
        "artifacts": {
            "runner": _record(Path(__file__).resolve()),
            "solver": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/taska_component_relation_anchor.py"
            ),
            "preregistration": _record(args.config.resolve()),
        },
    }
    _write_json(output_dir / "report.json", report)
    concise = {
        name: value.get("summary", value) for name, value in (
            ("local32", local),
            ("held32", held),
            ("fresh32", fresh),
        )
    }
    print(json.dumps(concise, indent=2))
    return report


if __name__ == "__main__":
    run(parse_args())
