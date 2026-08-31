#!/usr/bin/env python3
"""Evaluate one preregistered graph-consensus priority on frozen fusion evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_fullres_translation_consensus import (
    CONSENSUS_ARM_NAMES,
    CONSENSUS_MINIMUM,
    compose_translation_consensus_fusion,
)

try:
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_selective_fullres_fusion as parent
except ModuleNotFoundError:
    import run_taska_focal_current_finetune as finetune
    import run_taska_selective_fullres_fusion as parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_fullres_translation_consensus_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-fullres-translation-consensus/fixed-v1"
CONFIG_SHA256 = "d67c31d2f19c4d44acbf0a3fa039f05304ce5021bc1ed3c952732204ee99ace8"
FUSION_ROOT = PROJECT_ROOT / "outputs/taska-selective-fullres-union-fusion/fixed-v1"
REPORT_SCHEMA = "aiijc-taska-fullres-translation-consensus-report-v1"
LOCAL_PAIR_GATE = 0.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=parent.DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(resolved),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _fusion_archive(panel_name: str) -> Path:
    return FUSION_ROOT / panel_name / "frozen-target-free-eval.npz"


def _fusion_metadata(panel_name: str) -> Path:
    return FUSION_ROOT / panel_name / "frozen-target-free-eval.json"


def _truth_for_row(
    *,
    row: Mapping[str, Any],
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> tuple[np.ndarray, frozenset[RawTailEdge]]:
    source = str(row["source_filename"])
    draw = int(row["draw_index"])
    dirty = finetune._dirty_case(cache, lookup[source], source, draw)
    if finetune._dirty_sha256(dirty.dirty_tiles) != row["dirty_sha256"]:
        raise RuntimeError("recreated dirty bytes differ from frozen evidence")
    reference = finetune._reference(cache, lookup[source], source, draw, dirty.dirty_tiles)
    return parent.strict_layout(reference), parent._truth_edges(reference)


def _freeze_panel(
    *,
    panel_name: str,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    spec = parent.PANELS[panel_name]
    rows = json.loads(_fusion_metadata(panel_name).read_text(encoding="utf-8"))["rows"]
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    with (
        np.load(_fusion_archive(panel_name), allow_pickle=False) as fusion,
        np.load(spec.base_archive, allow_pickle=False) as base,
        np.load(spec.layout_archive, allow_pickle=False) as layouts,
    ):
        for index, row in enumerate(rows):
            prefix = str(row["prefix"])
            cost_right = parent._matrix(base, f"{prefix}__cost_right")
            cost_down = parent._matrix(base, f"{prefix}__cost_down")
            current = parent._edges(fusion, prefix, "current")
            selective = parent._edges(fusion, prefix, "selective_new")
            unique = parent._edges(fusion, prefix, "unique_fullres")
            current_logits = np.asarray(fusion[f"{prefix}__current_focal_logits"])
            selective_logits = np.asarray(fusion[f"{prefix}__selective_new_focal_logits"])
            unique_logits = np.asarray(fusion[f"{prefix}__unique_fullres_focal_logits"])
            result = compose_translation_consensus_fusion(
                cost_right=cost_right,
                cost_down=cost_down,
                four_layouts=parent._four_layouts(layouts, prefix),
                frozen_selective_control=fusion[
                    f"{prefix}__selective_target500_control_layout"
                ],
                frozen_confirmed_fusion_control=fusion[
                    f"{prefix}__combined_union_candidate_layout"
                ],
                current_edges=current,
                current_logits=current_logits,
                selective_new_edges=selective,
                selective_new_logits=selective_logits,
                unique_fullres_edges=unique,
                unique_fullres_logits=unique_logits,
            )
            arrays[f"{prefix}__confirmed_fusion_control_layout"] = result.control_layout
            arrays[f"{prefix}__translation_consensus_candidate_layout"] = (
                result.candidate_layout
            )
            arrays[f"{prefix}__translation_consensus_union_layout"] = (
                result.consensus_union_layout
            )
            arrays[f"{prefix}__consensus_mask"] = result.evidence.mask.astype(np.uint8)
            arrays[f"{prefix}__consensus_support"] = result.evidence.support
            arrays[f"{prefix}__adjusted_unique_priorities"] = (
                result.evidence.adjusted_unique_priorities
            )
            arrays.update(parent._edge_arrays(prefix, "unique_fullres", unique))
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": str(row["source_filename"]),
                    "draw_index": int(row["draw_index"]),
                    "dirty_sha256": str(row["dirty_sha256"]),
                    "choice": result.choice,
                    "strict_control": True,
                    "strict_candidate": True,
                    **result.diagnostics(),
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{panel_name}_translation_consensus_target_free",
                        "case": index + 1,
                        "consensus_edges": int(np.count_nonzero(result.evidence.mask)),
                        "choice": result.choice,
                    }
                ),
                flush=True,
            )
    stage = output_dir / panel_name
    stage.mkdir(parents=True, exist_ok=False)
    archive = stage / "frozen-target-free-eval.npz"
    metadata = stage / "frozen-target-free-eval.json"
    freeze = stage / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-fullres-translation-consensus-target-free-v1",
            "panel": panel_name,
            "contains_exact_references_or_candidate_labels": False,
            "matcher_or_denoiser_rerun": False,
            "consensus_minimum": CONSENSUS_MINIMUM,
            "selector_roster": list(CONSENSUS_ARM_NAMES),
            "confirmed_combined_arm_also_in_candidate_roster": False,
            "all_layouts_strict_original_upright_permutations": True,
            "rows": frozen_rows,
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-fullres-translation-consensus-pre-score-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "config": _record(DEFAULT_CONFIG),
                "parent_fusion_archive": _record(_fusion_archive(panel_name)),
                "runner": _record(Path(__file__).resolve()),
                "module": _record(
                    PROJECT_ROOT / "src/aiijc_puzzle/taska_fullres_translation_consensus.py"
                ),
                "raw_solver": _record(
                    PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
                ),
            },
        },
    )
    return archive, metadata, freeze


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("candidate was not frozen before scoring")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("target-free freeze contains labels")
    for name, record in payload["artifacts"].items():
        artifact = PROJECT_ROOT / record["path"]
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"frozen artifact changed before scoring: {name}")


def _score_panel(
    *,
    archive: Path,
    metadata: Path,
    freeze: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> dict[str, Any]:
    _validate_freeze(freeze)
    rows = json.loads(metadata.read_text(encoding="utf-8"))["rows"]
    scored: list[dict[str, Any]] = []
    with np.load(archive, allow_pickle=False) as frozen:
        for row in rows:
            prefix = str(row["prefix"])
            reference, truth = _truth_for_row(row=row, lookup=lookup, cache=cache)
            control = parent._layout_metrics(
                frozen[f"{prefix}__confirmed_fusion_control_layout"], reference
            )
            candidate = parent._layout_metrics(
                frozen[f"{prefix}__translation_consensus_candidate_layout"], reference
            )
            unique = parent._edges(frozen, prefix, "unique_fullres")
            mask = np.asarray(frozen[f"{prefix}__consensus_mask"], dtype=bool)
            consensus = tuple(edge for edge, keep in zip(unique, mask, strict=True) if keep)
            scored.append(
                {
                    **row,
                    "metrics": {
                        "confirmed_fusion_control": control,
                        "translation_consensus_candidate": candidate,
                    },
                    "target_assisted_diagnostic": {
                        "unique_fullres_edges": len(unique),
                        "unique_fullres_true_edges": len(set(unique) & truth),
                        "consensus_edges": len(consensus),
                        "consensus_true_edges": len(set(consensus) & truth),
                    },
                }
            )
    metric_names = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    sources = [str(row["source_filename"]) for row in scored]
    summary: dict[str, Any] = {
        "case_count": len(scored),
        "pair_denominator": parent.PAIR_DENOMINATOR,
        "arms": {
            arm: {
                metric: float(np.mean([row["metrics"][arm][metric] for row in scored]))
                for metric in metric_names
            }
            for arm in ("confirmed_fusion_control", "translation_consensus_candidate")
        },
        "choice_counts": dict(Counter(row["choice"] for row in scored)),
    }
    deltas: dict[str, Any] = {}
    for index, metric in enumerate(metric_names):
        values = [
            float(row["metrics"]["translation_consensus_candidate"][metric])
            - float(row["metrics"]["confirmed_fusion_control"][metric])
            for row in scored
        ]
        result = parent._cluster_ci(values, sources, seed=202_608_311_106 + index)
        result["case_wins_ties_losses"] = {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }
        deltas[metric] = result
    summary["candidate_minus_control"] = deltas
    consensus_count = sum(
        row["target_assisted_diagnostic"]["consensus_edges"] for row in scored
    )
    consensus_true = sum(
        row["target_assisted_diagnostic"]["consensus_true_edges"] for row in scored
    )
    summary["translation_consensus"] = {
        "edge_count": consensus_count,
        "true_edge_count": consensus_true,
        "precision": consensus_true / max(1, consensus_count),
        "boards_with_consensus": sum(
            row["target_assisted_diagnostic"]["consensus_edges"] > 0 for row in scored
        ),
    }
    return {"status": "complete", "rows": scored, "summary": summary}


def _run_panel(
    *,
    panel_name: str,
    output_dir: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> dict[str, Any]:
    archive, metadata, freeze = _freeze_panel(
        panel_name=panel_name,
        output_dir=output_dir,
    )
    return _score_panel(
        archive=archive,
        metadata=metadata,
        freeze=freeze,
        lookup=lookup,
        cache=cache,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = perf_counter()
    config_path = args.config.resolve()
    if config_path != DEFAULT_CONFIG.resolve() or sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("preregistered config path or SHA-256 changed")
    parent._require_inputs()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    local = _run_panel(
        panel_name="local32",
        output_dir=output_dir,
        lookup=lookup,
        cache=cache,
    )
    local_delta = local["summary"]["candidate_minus_control"][
        "satisfied_adjacent_pairs"
    ]["mean"]
    held: dict[str, Any] = {"status": "skipped_by_nonpositive_local_pair_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_nonpositive_local_pair_gate"}
    if local_delta > LOCAL_PAIR_GATE:
        held = _run_panel(
            panel_name="held32",
            output_dir=output_dir,
            lookup=lookup,
            cache=cache,
        )
        fresh = _run_panel(
            panel_name="fresh32",
            output_dir=output_dir,
            lookup=lookup,
            cache=cache,
        )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "protocol": json.loads(config_path.read_text(encoding="utf-8")),
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "strict_original_upright_tile_permutations": True,
            "targets_absent_from_candidate_inference": True,
            "restored_pixels_matcher_only": True,
            "competition_test_accessed": False,
            "postprocessing_used": False,
            "production_or_submission_modified": False,
        },
        "artifacts": {
            "config": _record(config_path),
            "runner": _record(Path(__file__).resolve()),
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/taska_fullres_translation_consensus.py"
            ),
            "raw_solver": _record(PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"),
        },
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps({"local32": local, "held32": held, "fresh32": fresh}, indent=2))
    return report


if __name__ == "__main__":
    run(parse_args())
