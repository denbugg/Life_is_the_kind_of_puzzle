#!/usr/bin/env python3
"""Evaluate one preregistered Twin-only edge-supply arm on frozen TASKA fusion.

The frozen fusion layout is replayed as control.  FullResolutionTwin can add
only provably Twin-only Union-v2 top144 hard edges that are absent from the
parent combined union and pass the recovered focal logit>=0 rule.  Predictions
are content-addressed before organizer-train references are reconstructed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.fullres_twin_side_matcher import twin_right_down_scores
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.raw_twin_union_production import (
    infer_raw_twin_union_assignments,
    load_fullres_twin_checkpoint,
    load_raw_twin_union_checkpoint,
)
from aiijc_puzzle.socket_sorter_production import load_socket_checkpoint
from aiijc_puzzle.taska_focal_verifier import (
    TASKA_FOCAL_VERIFIER_SHA256,
    load_taska_focal_verifier,
    score_focal_edges,
)
from aiijc_puzzle.taska_pair_pipeline import FOCAL_MODE, PAIR_DENOMINATOR
from aiijc_puzzle.taska_twin_unique_supply import (
    TWIN_ACCEPT_LOGIT_MINIMUM,
    TWIN_FUSION_ARM_NAMES,
    TWIN_PARENT_TOPK,
    UNION_HARD_BUDGET_PER_AXIS,
    compose_twin_unique_fusion,
    nominate_twin_unique_edges,
)

try:
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_selective_fullres_fusion as parent
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_current_finetune as finetune
    import run_taska_selective_fullres_fusion as parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-twin-unique-supply/fixed-v1"
PREREGISTRATION = PROJECT_ROOT / "configs/taska_twin_unique_supply_preregistered_v1.json"
PREREGISTRATION_SHA256 = "05ac95769646a569573dafdccb4082e5ba33da063f7742597f5cee8bfbb0df53"
PARENT_ROOT = PROJECT_ROOT / "outputs/taska-selective-fullres-union-fusion/fixed-v1"
PARENT_REPORT = PARENT_ROOT / "report.json"
PARENT_REPORT_SHA256 = "1f9d84c99eae6ba1f03a668163f6e19321e20292e31dd5e51ec00282587517af"
SOCKET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
TWIN_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fullres-twin-side-matcher/v1-fit256-s400-eval24/fullres-twin-side-matcher.pt"
)
UNION_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/raw-twin-union-reranker/v2-fit256-s400-eval24/raw-twin-union-reranker-v2.pt"
)
UNION_CONFIG = PROJECT_ROOT / "configs/raw_twin_union_reranker_v2_preregistered.json"
UNION_SELECTION = (
    PROJECT_ROOT
    / "outputs/raw-twin-union-reranker/v2-fit256-s400-eval24/selection-commitment.json"
)
FOCAL_CHECKPOINT = PROJECT_ROOT / "artifacts/prior-taska/ckpt/verify_pair_best.pt"
GRID = 24
COUNT = GRID * GRID
LOCAL_GATE = 0.0
HELD_GATE = 0.5
REPORT_SCHEMA = "aiijc-taska-twin-unique-supply-report-v1"
SCORED_ARMS = ("fusion_parent_control", "twin_unique_candidate")

_PARENT_PANEL_SHA256 = {
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
_MODEL_SHA256 = {
    SOCKET_CHECKPOINT: "0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670",
    TWIN_CHECKPOINT: "c5b44901e8da459e3c48b6e7af7153c5d7eed26f1c1b52c8712c4fa0dc4ea8ae",
    UNION_CHECKPOINT: "a5f882ab3c827e4e3779be3372c62d2a8fb9cd95d3558fd30cc566a9c3137f79",
    UNION_CONFIG: "6741e92e832a630f1b83bde6edc8a341a348f52daa82313c40a8f32c7c1173d4",
    UNION_SELECTION: "71ae4f5095489613857fcd25c541fe496da0d6861f6ff604850147dd04b91cd2",
    FOCAL_CHECKPOINT: TASKA_FOCAL_VERIFIER_SHA256,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--smoke-one",
        action="store_true",
        help="freeze one local target-free prediction without scoring",
    )
    return parser.parse_args(argv)


def _parent_triplet(panel_name: str) -> tuple[Path, Path, Path]:
    archive = PARENT_ROOT / panel_name / "frozen-target-free-eval.npz"
    return archive, archive.with_suffix(".json"), archive.parent / "pre-score-freeze.json"


def _record(path: Path) -> dict[str, str]:
    return parent._record(path)


def _require_inputs() -> None:
    parent._require_inputs()
    fixed = {
        PREREGISTRATION: PREREGISTRATION_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        **_MODEL_SHA256,
    }
    for panel_name, hashes in _PARENT_PANEL_SHA256.items():
        fixed.update(dict(zip(_parent_triplet(panel_name), hashes, strict=True)))
    for path, expected in fixed.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen input SHA-256 mismatch: {path}")
    preregistered = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if preregistered.get("created_before_candidate_target_scoring") is not True:
        raise ValueError("preregistration timing contract differs")
    for panel_name in parent.PANELS:
        parent._validate_parent_freeze(_parent_triplet(panel_name)[2])


def _aligned_rows(
    spec: parent.PanelSpec,
    *,
    case_count: int,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    base_rows = parent._aligned_rows(
        parent.PanelSpec(**{**spec.__dict__, "case_count": case_count})
    )
    fusion_rows = parent._rows(_parent_triplet(spec.name)[1], case_count)
    identity = ("prefix", "source_filename", "draw_index", "dirty_sha256")
    aligned: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for records, fusion in zip(base_rows, fusion_rows, strict=True):
        source = records[0]
        if any(source.get(field) != fusion.get(field) for field in identity):
            raise RuntimeError(f"{spec.name} parent fusion identity mismatch")
        aligned.append((source, fusion))
    return aligned


def _edge_arrays(
    prefix: str,
    name: str,
    edges: Sequence[RawTailEdge],
) -> dict[str, np.ndarray]:
    return parent._edge_arrays(prefix, name, edges)


def _load_models(device: torch.device) -> tuple[Any, Any, Any, Any]:
    socket = load_socket_checkpoint(SOCKET_CHECKPOINT, device=device)
    twin = load_fullres_twin_checkpoint(TWIN_CHECKPOINT, device=device)
    union = load_raw_twin_union_checkpoint(
        UNION_CHECKPOINT,
        config_path=UNION_CONFIG,
        selection_path=UNION_SELECTION,
        device=device,
    )
    focal = load_taska_focal_verifier(FOCAL_CHECKPOINT, device=device)
    return socket, twin, union, focal


def _freeze(
    *,
    output_panel_name: str,
    parent_panel_name: str,
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path, Path]:
    stage = output_dir / output_panel_name
    stage.mkdir(parents=True, exist_ok=False)
    archive = stage / "frozen-target-free-eval.npz"
    metadata = archive.with_suffix(".json")
    freeze = stage / "pre-score-freeze.json"
    parent._write_npz(archive, arrays)
    parent._write_json(
        metadata,
        {
            "schema": "aiijc-taska-twin-unique-supply-target-free-v1",
            "stage": output_panel_name,
            "contains_exact_references_or_candidate_labels": False,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "twin_nomination": "Union-v2 hard top144/axis intersect direct Twin top32",
            "focal_logit_minimum": TWIN_ACCEPT_LOGIT_MINIMUM,
            "selector_roster": list(TWIN_FUSION_ARM_NAMES),
            "tail": "focal-gated non-adjacent tail96",
            "all_layouts_strict_original_upright_permutations": True,
            "rows": list(rows),
        },
    )
    artifacts = {
        "frozen_archive": archive,
        "frozen_metadata": metadata,
        "preregistration": PREREGISTRATION,
        "parent_report": PARENT_REPORT,
        "parent_archive": _parent_triplet(parent_panel_name)[0],
        "parent_metadata": _parent_triplet(parent_panel_name)[1],
        "parent_freeze": _parent_triplet(parent_panel_name)[2],
        "socket_checkpoint": SOCKET_CHECKPOINT,
        "twin_checkpoint": TWIN_CHECKPOINT,
        "union_checkpoint": UNION_CHECKPOINT,
        "union_config": UNION_CONFIG,
        "union_selection": UNION_SELECTION,
        "focal_checkpoint": FOCAL_CHECKPOINT,
        "runner": Path(__file__).resolve(),
        "module": PROJECT_ROOT / "src/aiijc_puzzle/taska_twin_unique_supply.py",
        "parent_runner": PROJECT_ROOT / "scripts/run_taska_selective_fullres_fusion.py",
    }
    parent._write_json(
        freeze,
        {
            "schema": "aiijc-taska-twin-unique-supply-pre-score-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {name: _record(path) for name, path in artifacts.items()},
        },
    )
    return archive, metadata, freeze


def _validate_freeze(path: Path) -> None:
    parent._validate_freeze(path)


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
        "choice_counts": dict(Counter(str(row["choice"]) for row in rows)),
        "parent_replay_match_count": sum(
            bool(row["mechanical_parent_matches_frozen"]) for row in rows
        ),
    }
    sources = [str(row["source_filename"]) for row in rows]
    deltas: dict[str, Any] = {}
    for index, metric in enumerate(metrics):
        values = [
            float(row["metrics"]["twin_unique_candidate"][metric])
            - float(row["metrics"]["fusion_parent_control"][metric])
            for row in rows
        ]
        summary = parent._cluster_ci(
            values,
            sources,
            seed=parent.BOOTSTRAP_SEED + 20 + index,
        )
        summary["case_wins_ties_losses"] = {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }
        deltas[metric] = summary
    result["candidate_minus_control"] = deltas
    fields = (
        "parent_combined_edges",
        "parent_combined_true_edges",
        "proposed_twin_only_edges",
        "proposed_twin_only_true_edges",
        "accepted_twin_only_edges",
        "accepted_twin_only_true_edges",
        "augmented_edges",
        "augmented_true_edges",
    )
    totals = {field: int(sum(row["supply"][field] for row in rows)) for field in fields}
    result["supply_mean_per_board"] = {
        field: float(np.mean([row["supply"][field] for row in rows])) for field in fields
    }
    result["supply_totals"] = totals
    result["supply_quality"] = {
        "proposed_twin_only_precision": float(
            totals["proposed_twin_only_true_edges"]
            / max(1, totals["proposed_twin_only_edges"])
        ),
        "accepted_twin_only_precision": float(
            totals["accepted_twin_only_true_edges"]
            / max(1, totals["accepted_twin_only_edges"])
        ),
        "augmented_union_recall": float(
            result["supply_mean_per_board"]["augmented_true_edges"] / PAIR_DENOMINATOR
        ),
    }
    return result


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
        for frozen in frozen_rows:
            prefix = str(frozen["prefix"])
            source = str(frozen["source_filename"])
            draw = int(frozen["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            if finetune._dirty_sha256(dirty.dirty_tiles) != frozen["dirty_sha256"]:
                raise RuntimeError("scoring recreated different dirty bytes")
            reference = finetune._reference(
                cache,
                lookup[source],
                source,
                draw,
                dirty.dirty_tiles,
            )
            truth = parent._truth_edges(reference)
            parent_edges = set(parent._edges(candidate, prefix, "parent_combined"))
            proposed = set(parent._edges(candidate, prefix, "proposed_twin"))
            accepted = set(parent._edges(candidate, prefix, "accepted_twin"))
            augmented = set(parent._edges(candidate, prefix, "augmented"))
            metrics = {
                arm: parent._layout_metrics(candidate[f"{prefix}__{arm}_layout"], reference)
                for arm in SCORED_ARMS
            }
            scored.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "choice": frozen["choice"],
                    "mechanical_parent_matches_frozen": frozen[
                        "mechanical_parent_matches_frozen"
                    ],
                    "metrics": metrics,
                    "supply": {
                        "parent_combined_edges": len(parent_edges),
                        "parent_combined_true_edges": len(parent_edges & truth),
                        "proposed_twin_only_edges": len(proposed),
                        "proposed_twin_only_true_edges": len(proposed & truth),
                        "accepted_twin_only_edges": len(accepted),
                        "accepted_twin_only_true_edges": len(accepted & truth),
                        "augmented_edges": len(augmented),
                        "augmented_true_edges": len(augmented & truth),
                    },
                }
            )
    return scored, _summarize(scored)


def _run_panel(
    spec: parent.PanelSpec,
    *,
    case_count: int,
    output_dir: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
    models: tuple[Any, Any, Any, Any],
    device: torch.device,
    target_free_only: bool,
) -> dict[str, Any]:
    socket, twin, union, focal = models
    aligned = _aligned_rows(spec, case_count=case_count)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    parent_archive_path = _parent_triplet(spec.name)[0]
    with (
        np.load(spec.layout_archive, allow_pickle=False) as layouts,
        np.load(spec.base_archive, allow_pickle=False) as base,
        np.load(parent_archive_path, allow_pickle=False) as fusion,
    ):
        for index, (_identity, frozen) in enumerate(aligned):
            prefix = str(frozen["prefix"])
            source = str(frozen["source_filename"])
            draw = int(frozen["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            dirty_hash = finetune._dirty_sha256(dirty.dirty_tiles)
            if dirty_hash != frozen["dirty_sha256"]:
                raise RuntimeError("Twin inference recreated different dirty bytes")
            cost_right = parent._matrix(base, f"{prefix}__cost_right")
            cost_down = parent._matrix(base, f"{prefix}__cost_down")
            current = parent._edges(fusion, prefix, "current")
            selective = parent._edges(fusion, prefix, "selective_new")
            fullres = parent._edges(fusion, prefix, "unique_fullres")
            combined = parent._edges(fusion, prefix, "combined_union")
            if combined != current + selective + fullres:
                raise RuntimeError("frozen fusion combined-union order changed")
            inference = infer_raw_twin_union_assignments(
                dirty.dirty_tiles,
                socket,
                twin,
                union,
                device=device,
            )
            twin_right, twin_down = twin_right_down_scores(
                twin.model,
                dirty.dirty_tiles,
                device=device,
            )
            proposed = nominate_twin_unique_edges(
                twin_right_scores=twin_right,
                twin_down_scores=twin_down,
                learned_right_assignment=inference.learned_right_log_assignment,
                learned_down_assignment=inference.learned_down_log_assignment,
                candidate_snapshot=inference.candidate_snapshot,
                excluded_edges=combined,
            )
            focal_scores = score_focal_edges(
                focal,
                dirty.dirty_tiles,
                cost_right,
                cost_down,
                proposed,
                mode=FOCAL_MODE,
                grid=GRID,
                device=device,
            )
            if focal_scores.edges != proposed:
                raise RuntimeError("focal scores changed Twin nomination order")
            result = compose_twin_unique_fusion(
                cost_right=cost_right,
                cost_down=cost_down,
                four_layouts=parent._four_layouts(layouts, prefix),
                selective_union_layout=fusion[f"{prefix}__selective_union_layout"],
                combined_union_layout=fusion[f"{prefix}__combined_union_layout"],
                frozen_parent_layout=fusion[
                    f"{prefix}__combined_union_candidate_layout"
                ],
                frozen_parent_choice=str(frozen["choice"]),
                current_edges=current,
                current_logits=fusion[f"{prefix}__current_focal_logits"],
                selective_new_edges=selective,
                selective_new_logits=fusion[f"{prefix}__selective_new_focal_logits"],
                unique_fullres_edges=fullres,
                unique_fullres_logits=fusion[f"{prefix}__unique_fullres_focal_logits"],
                proposed_twin_edges=proposed,
                proposed_twin_logits=focal_scores.logits,
            )
            replay = bool(
                np.array_equal(
                    result.control_layout,
                    fusion[f"{prefix}__combined_union_candidate_layout"],
                )
            )
            if not replay:
                raise RuntimeError("frozen fusion parent replay mismatch")
            arrays[f"{prefix}__fusion_parent_control_layout"] = result.control_layout
            arrays[f"{prefix}__twin_unique_candidate_layout"] = result.candidate_layout
            arrays[f"{prefix}__twin_unique_union_layout"] = result.twin_union_layout
            for name, edges in (
                ("parent_combined", combined),
                ("proposed_twin", result.proposed_twin_edges),
                ("accepted_twin", result.accepted_twin_edges),
                ("augmented", result.augmented_edges),
            ):
                arrays.update(_edge_arrays(prefix, name, edges))
            arrays[f"{prefix}__proposed_twin_focal_logits"] = result.proposed_twin_logits
            arrays[f"{prefix}__accepted_twin_focal_logits"] = result.accepted_twin_logits
            arrays[f"{prefix}__augmented_focal_logits"] = result.augmented_logits
            diagnostics = result.diagnostics()
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "dirty_sha256": dirty_hash,
                    "mechanical_parent_matches_frozen": replay,
                    "union_candidate_snapshot_sha256": inference.candidate_snapshot.sha256,
                    "union_candidate_count": inference.candidate_count,
                    **diagnostics,
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{spec.name}_twin_unique_target_free",
                        "case": index + 1,
                        "case_count": case_count,
                        "proposed": len(result.proposed_twin_edges),
                        "accepted": len(result.accepted_twin_edges),
                        "choice": result.choice,
                        "parent_replay": replay,
                    }
                ),
                flush=True,
            )
    archive, metadata, freeze = _freeze(
        output_panel_name="smoke1" if target_free_only else spec.name,
        parent_panel_name=spec.name,
        output_dir=output_dir,
        arrays=arrays,
        rows=frozen_rows,
    )
    payload: dict[str, Any] = {
        "status": "target-free-smoke" if target_free_only else "complete",
        "target_free_summary": {
            "case_count": len(frozen_rows),
            "parent_replay_match_count": sum(
                bool(row["mechanical_parent_matches_frozen"]) for row in frozen_rows
            ),
            "choice_counts": dict(Counter(str(row["choice"]) for row in frozen_rows)),
            "mean_proposed_twin_only": float(
                np.mean([row["proposed_twin_only_count"] for row in frozen_rows])
            ),
            "mean_accepted_twin_only": float(
                np.mean([row["accepted_twin_only_count"] for row in frozen_rows])
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
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    models = _load_models(device)
    started = perf_counter()
    if args.smoke_one:
        smoke = _run_panel(
            parent.PANELS["local32"],
            case_count=1,
            output_dir=output_dir,
            lookup=lookup,
            cache=cache,
            models=models,
            device=device,
            target_free_only=True,
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": "target-free-smoke",
            "local32": smoke,
            "reference_reconstructed": False,
            "competition_test_accessed": False,
        }
        parent._write_json(output_dir / "report.json", report)
        print(json.dumps(report, indent=2))
        return report

    local = _run_panel(
        parent.PANELS["local32"],
        case_count=32,
        output_dir=output_dir,
        lookup=lookup,
        cache=cache,
        models=models,
        device=device,
        target_free_only=False,
    )
    local_delta = local["summary"]["candidate_minus_control"][
        "satisfied_adjacent_pairs"
    ]["mean"]
    held: dict[str, Any] = {"status": "skipped_by_negative_local_pair_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_pair_gate"}
    if local_delta >= LOCAL_GATE:
        held = _run_panel(
            parent.PANELS["held32"],
            case_count=32,
            output_dir=output_dir,
            lookup=lookup,
            cache=cache,
            models=models,
            device=device,
            target_free_only=False,
        )
        held_delta = held["summary"]["candidate_minus_control"][
            "satisfied_adjacent_pairs"
        ]["mean"]
        if held_delta >= HELD_GATE:
            fresh = _run_panel(
                parent.PANELS["fresh32"],
                case_count=32,
                output_dir=output_dir,
                lookup=lookup,
                cache=cache,
                models=models,
                device=device,
                target_free_only=False,
            )
        else:
            fresh = {"status": "skipped_by_held_pair_delta_below_0.5"}
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "protocol": {
            "parent": "frozen selective-target500 plus unique-fullres fusion",
            "control": "exact frozen parent final layout",
            "twin_nomination": (
                "Union-v2 hard top144/axis intersect directly evaluated Twin top32"
            ),
            "twin_topk": TWIN_PARENT_TOPK,
            "union_hard_budget_per_axis": UNION_HARD_BUDGET_PER_AXIS,
            "uniqueness": "absent from parent combined union",
            "acceptance": f"recovered focal logit >= {TWIN_ACCEPT_LOGIT_MINIMUM}",
            "selector_roster": list(TWIN_FUSION_ARM_NAMES),
            "raw_dense_costs_unchanged": True,
            "tail": "focal-gated non-adjacent tail96",
            "local_pair_gate": LOCAL_GATE,
            "held_pair_gate": HELD_GATE,
            "no_threshold_budget_or_roster_sweep": True,
        },
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "strict_original_upright_tile_permutations": True,
            "twin_and_socket_pixels_matcher_only": True,
            "pixels_rendered_replaced_rotated_or_warped": False,
            "targets_used_only_after_candidate_freeze": True,
            "competition_test_accessed": False,
            "postprocessing_used": False,
            "production_modified": False,
        },
        "artifacts": {
            "preregistration": _record(PREREGISTRATION),
            "runner": _record(Path(__file__).resolve()),
            "module": _record(PROJECT_ROOT / "src/aiijc_puzzle/taska_twin_unique_supply.py"),
            "parent_report": _record(PARENT_REPORT),
        },
    }
    parent._write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {name: report[name] for name in ("local32", "held32", "fresh32")},
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    run(parse_args())
