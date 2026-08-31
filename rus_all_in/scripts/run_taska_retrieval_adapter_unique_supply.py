#!/usr/bin/env python3
"""Freeze then gate one adapter-step400 supply above confirmed TASKA fusion."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.fullres_boundary_denoiser import restore_matcher_view
from aiijc_puzzle.fullres_retrieval_adapter import FullResolutionRetrievalAdapter
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.socket_sorter_production import load_socket_checkpoint
from aiijc_puzzle.taska_focal_verifier import (
    TASKA_FOCAL_VERIFIER_SHA256,
    load_taska_focal_verifier,
    score_focal_edges,
)
from aiijc_puzzle.taska_pair_pipeline import FOCAL_MODE, PAIR_DENOMINATOR
from aiijc_puzzle.taska_retrieval_adapter_unique_supply import (
    ADAPTER_FOCAL_LOGIT_MINIMUM,
    ADAPTER_NOMINATOR,
    ADAPTER_TOPK,
    accept_unique_adapter_proposals,
    compose_adapter_unique_fusion,
    reciprocal_rank_topk_edges,
    topk_indices,
    unique_adapter_proposals,
)
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES

try:
    from scripts import run_fullres_boundary_denoiser as socket_eval
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_selective_fullres_fusion as parent
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_fullres_boundary_denoiser as socket_eval
    import run_taska_focal_current_finetune as finetune
    import run_taska_selective_fullres_fusion as parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-retrieval-adapter-unique-supply/fixed-v1"
CONFIG = PROJECT_ROOT / "configs/taska_retrieval_adapter_unique_supply_preregistered_v1.json"
CONFIG_SHA256 = "eb4b2426d4fa6f7f6d3364d1886e5bed9026ddd08ef88001249b125955ebbcfe"
ADAPTER_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fullres-retrieval-adapter/fixed-s100-s400-local16-v1/adapter_step400.pt"
)
ADAPTER_REPORT = (
    PROJECT_ROOT
    / "outputs/fullres-retrieval-adapter/fixed-s100-s400-local16-v1/report.json"
)
SOCKET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
FOCAL_CHECKPOINT = PROJECT_ROOT / "artifacts/prior-taska/ckpt/verify_pair_best.pt"
FUSION_ROOT = PROJECT_ROOT / "outputs/taska-selective-fullres-union-fusion/fixed-v1"
FUSION_REPORT = FUSION_ROOT / "report.json"
GRID = 24
COUNT = GRID * GRID
LOCAL_PAIR_GATE = 0.0
LOCAL_EXACT_GATE = -1.0
HELD_PAIR_GATE = 0.25
FOCAL_PRECISION_MINIMUM = 0.25
FOCAL_TRUE_PER_BOARD_MINIMUM = 0.5
MEAN_UNIQUE_MINIMUM = 2.0
MEAN_ACCEPTED_MINIMUM = 1.0
MEAN_ACCEPTED_MAXIMUM = 64.0
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_213
REPORT_SCHEMA = "aiijc-taska-retrieval-adapter-unique-supply-report-v1"
SCORED_ARMS = ("confirmed_fusion_control", "adapter_unique_candidate")

_FIXED_SHA256 = {
    CONFIG: CONFIG_SHA256,
    ADAPTER_CHECKPOINT: "00ca56f1be2c8e99bc8ef19b0d9190862d6bd5e4fb8b36fbe926087cd3945cb0",
    ADAPTER_REPORT: "5fafb0307586669c7b7c9eaa4699fda1a3bd1250ca921fc48dd7e86af0bdefbb",
    SOCKET_CHECKPOINT: "0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670",
    FOCAL_CHECKPOINT: TASKA_FOCAL_VERIFIER_SHA256,
    FUSION_REPORT: "1f9d84c99eae6ba1f03a668163f6e19321e20292e31dd5e51ec00282587517af",
    PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py": (
        "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
    ),
}

_FUSION_PANEL_SHA256 = {
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


@dataclass(frozen=True)
class PanelSpec:
    name: str
    case_count: int
    parent: parent.PanelSpec
    fusion_archive: Path
    fusion_metadata: Path
    fusion_freeze: Path


def _panel(name: str) -> PanelSpec:
    archive = FUSION_ROOT / name / "frozen-target-free-eval.npz"
    return PanelSpec(
        name=name,
        case_count=32,
        parent=parent.PANELS[name],
        fusion_archive=archive,
        fusion_metadata=archive.with_suffix(".json"),
        fusion_freeze=archive.parent / "pre-score-freeze.json",
    )


PANELS = {name: _panel(name) for name in ("local32", "held32", "fresh32")}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze-local", "continue"), required=True)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--inference-batch", type=int, default=576)
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        shown = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        shown = str(resolved)
    return {"path": shown, "sha256": sha256_file(resolved)}


def _write_json(path: Path, payload: Any) -> None:
    parent._write_json(path, payload)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    parent._write_npz(path, arrays)


def _triplet(spec: PanelSpec, output_dir: Path) -> tuple[Path, Path, Path]:
    archive = output_dir / spec.name / "frozen-target-free-eval.npz"
    return archive, archive.with_suffix(".json"), archive.parent / "pre-score-freeze.json"


def _require_inputs() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen-before-target-assisted-scoring":
        raise RuntimeError("adapter unique-supply preregistration status changed")
    parent._require_inputs()
    for path, expected in _FIXED_SHA256.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen input SHA-256 mismatch: {path}")
    adapter_report = json.loads(ADAPTER_REPORT.read_text(encoding="utf-8"))
    adapter_fit = set(adapter_report["protocol"]["fit_filenames"])
    for name, spec in PANELS.items():
        expected = _FUSION_PANEL_SHA256[name]
        for path, digest in zip(
            (spec.fusion_archive, spec.fusion_metadata, spec.fusion_freeze),
            expected,
            strict=True,
        ):
            if not path.is_file() or sha256_file(path) != digest:
                raise RuntimeError(f"{name} confirmed-fusion parent changed: {path}")
        parent._validate_parent_freeze(spec.fusion_freeze)
        rows = parent._rows(spec.fusion_metadata, spec.case_count)
        if adapter_fit & {str(row["source_filename"]) for row in rows}:
            raise RuntimeError(f"adapter fit overlaps TASKA {name}")


def _aligned_rows(spec: PanelSpec) -> list[tuple[Mapping[str, Any], ...]]:
    parents = parent._aligned_rows(spec.parent)
    fusion = parent._rows(spec.fusion_metadata, spec.case_count)
    identity = ("prefix", "source_filename", "draw_index", "dirty_sha256")
    result: list[tuple[Mapping[str, Any], ...]] = []
    for records, fusion_row in zip(parents, fusion, strict=True):
        if any(records[0].get(field) != fusion_row.get(field) for field in identity):
            raise RuntimeError(f"{spec.name} confirmed-fusion row identity mismatch")
        result.append((*records, fusion_row))
    return result


def _edge_arrays(
    prefix: str, name: str, edges: Sequence[RawTailEdge]
) -> dict[str, np.ndarray]:
    return parent._edge_arrays(prefix, name, edges)


def _load_models(device: torch.device) -> tuple[Any, Any, Any]:
    socket = load_socket_checkpoint(SOCKET_CHECKPOINT, device=device)
    payload = torch.load(ADAPTER_CHECKPOINT, map_location="cpu", weights_only=True)
    if payload.get("step") != 400 or payload.get("config_sha256") != (
        "74bc2f356a5750bd13f19a0911b639831f771522e258313f765027b5a6d0fc95"
    ):
        raise RuntimeError("adapter step400 checkpoint contract changed")
    adapter = FullResolutionRetrievalAdapter().to(device)
    adapter.load_state_dict(payload["state_dict"], strict=True)
    adapter.eval()
    focal = load_taska_focal_verifier(FOCAL_CHECKPOINT, device=device)
    if getattr(focal, "checkpoint_sha256", None) != TASKA_FOCAL_VERIFIER_SHA256:
        raise RuntimeError("loaded focal verifier lineage changed")
    return socket, adapter, focal


def _freeze_artifacts(
    spec: PanelSpec,
    *,
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path, Path]:
    archive, metadata, freeze = _triplet(spec, output_dir)
    archive.parent.mkdir(parents=True, exist_ok=False)
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-retrieval-adapter-unique-supply-target-free-v1",
            "stage": spec.name,
            "contains_exact_references_or_candidate_labels": False,
            "adapter_pixels_matcher_only": True,
            "raw_d64_top32_saved_parallel": True,
            "adapter_nominator": ADAPTER_NOMINATOR,
            "adapter_topk": ADAPTER_TOPK,
            "deduplicate_before_focal": "current + selective + confirmed fullres",
            "dirty_visible_focal_logit_minimum": ADAPTER_FOCAL_LOGIT_MINIMUM,
            "selector_roster": list(FUSION_ARM_NAMES),
            "new_standalone_arm": False,
            "raw_dense_costs_unchanged": True,
            "strict_original_upright_permutations": True,
            "rows": list(rows),
        },
    )
    artifacts = {
        "archive": archive,
        "metadata": metadata,
        "config": CONFIG,
        "adapter_checkpoint": ADAPTER_CHECKPOINT,
        "adapter_report": ADAPTER_REPORT,
        "socket_checkpoint": SOCKET_CHECKPOINT,
        "focal_checkpoint": FOCAL_CHECKPOINT,
        "fusion_parent_archive": spec.fusion_archive,
        "fusion_parent_metadata": spec.fusion_metadata,
        "fusion_parent_freeze": spec.fusion_freeze,
        "runner": Path(__file__).resolve(),
        "module": PROJECT_ROOT / "src/aiijc_puzzle/taska_retrieval_adapter_unique_supply.py",
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
    }
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-retrieval-adapter-unique-supply-pre-score-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": {name: _record(path) for name, path in artifacts.items()},
        },
    )
    return archive, metadata, freeze


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("pre-score freeze timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze contains labels")
    for name, record in payload.get("artifacts", {}).items():
        artifact = Path(record["path"])
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"frozen artifact changed before scoring: {name}")


@torch.inference_mode()
def _freeze_panel(
    spec: PanelSpec,
    *,
    output_dir: Path,
    socket: Any,
    adapter: FullResolutionRetrievalAdapter,
    focal: Any,
    device: torch.device,
    inference_batch: int,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    aligned = _aligned_rows(spec)
    base_spec = spec.parent
    with (
        np.load(base_spec.layout_archive, allow_pickle=False) as layouts,
        np.load(base_spec.base_archive, allow_pickle=False) as base,
        np.load(base_spec.selective_archive, allow_pickle=False) as selective,
        np.load(base_spec.fullres_archive, allow_pickle=False) as fullres,
        np.load(spec.fusion_archive, allow_pickle=False) as fusion,
    ):
        for index, records in enumerate(aligned):
            row = records[2]
            prefix = str(row["prefix"])
            source = str(row["source_filename"])
            draw = int(row["draw_index"])
            dirty = finetune._dirty_case(cache, lookup[source], source, draw)
            dirty_sha = finetune._dirty_sha256(dirty.dirty_tiles)
            if dirty_sha != row["dirty_sha256"]:
                raise RuntimeError(f"{spec.name} recreated different dirty bytes")

            right = parent._matrix(base, f"{prefix}__cost_right")
            down = parent._matrix(base, f"{prefix}__cost_down")
            current = parent._edges(selective, prefix, "current")
            if current != parent._edges(base, prefix) or current != parent._edges(
                fullres, prefix, "current"
            ):
                raise RuntimeError("frozen current-edge identity mismatch")
            current_logits = np.asarray(
                selective[f"{prefix}__current_focal_logits"], dtype=np.float32
            )
            selective_new = parent._edges(selective, prefix, "accepted_new")
            selective_logits = np.asarray(
                selective[f"{prefix}__accepted_new_focal_logits"], dtype=np.float32
            )
            fullres_new, fullres_logits = parent._fullres_accepted_with_logits(
                fullres, prefix
            )

            raw_socket_right, raw_socket_down = socket_eval._socket_scores(
                socket, dirty.dirty_tiles, device=device
            )
            adapted = restore_matcher_view(
                adapter,
                dirty.dirty_tiles,
                device=device,
                batch_size=inference_batch,
            )
            adapter_right, adapter_down = socket_eval._socket_scores(
                socket, adapted, device=device
            )
            nominated = reciprocal_rank_topk_edges(
                adapter_right, axis="right"
            ) + reciprocal_rank_topk_edges(adapter_down, axis="down")
            unique = unique_adapter_proposals(
                nominated_edges=nominated,
                current_edges=current,
                selective_edges=selective_new,
                fullres_edges=fullres_new,
            )
            if unique.unique_edges:
                focal_result = score_focal_edges(
                    focal,
                    dirty.dirty_tiles,
                    right,
                    down,
                    unique.unique_edges,
                    mode=FOCAL_MODE,
                    grid=GRID,
                    device=device,
                )
                proposed_logits = np.asarray(focal_result.logits, dtype=np.float32)
                accepted, accepted_logits = accept_unique_adapter_proposals(
                    unique.unique_edges, proposed_logits
                )
            else:
                proposed_logits = np.empty(0, dtype=np.float32)
                accepted = ()
                accepted_logits = np.empty(0, dtype=np.float32)

            result = compose_adapter_unique_fusion(
                cost_right=right,
                cost_down=down,
                four_layouts=parent._four_layouts(layouts, prefix),
                frozen_selective_control=selective[
                    f"{prefix}__selective_vote500_focal_gated_layout"
                ],
                frozen_fullres_fusion_control=fusion[
                    f"{prefix}__combined_union_candidate_layout"
                ],
                current_edges=current,
                current_logits=current_logits,
                selective_new_edges=selective_new,
                selective_new_logits=selective_logits,
                fullres_accepted_edges=fullres_new,
                fullres_accepted_logits=fullres_logits,
                adapter_accepted_edges=accepted,
                adapter_accepted_logits=accepted_logits,
            )
            replay = bool(
                np.array_equal(
                    result.control_layout,
                    fusion[f"{prefix}__combined_union_candidate_layout"],
                )
            )
            if not replay:
                raise RuntimeError("confirmed-fusion control replay mismatch")

            arrays[f"{prefix}__confirmed_fusion_control_layout"] = result.control_layout
            arrays[f"{prefix}__adapter_unique_candidate_layout"] = result.candidate_layout
            arrays[f"{prefix}__extended_combined_layout"] = result.extended_combined_layout
            arrays[f"{prefix}__raw_d64_top32_right"] = topk_indices(raw_socket_right)
            arrays[f"{prefix}__raw_d64_top32_down"] = topk_indices(raw_socket_down)
            arrays[f"{prefix}__adapter_d64_top32_right"] = topk_indices(adapter_right)
            arrays[f"{prefix}__adapter_d64_top32_down"] = topk_indices(adapter_down)
            for name, edges in (
                ("adapter_nominated", nominated),
                ("adapter_unique_proposed", unique.unique_edges),
                ("adapter_accepted_unique", accepted),
                ("base_combined_union", result.base.supply.combined_union_edges),
                ("extended_combined_union", result.extended_union_edges),
            ):
                arrays.update(_edge_arrays(prefix, name, edges))
            arrays[f"{prefix}__adapter_unique_proposed_focal_logits"] = proposed_logits
            arrays[f"{prefix}__adapter_accepted_unique_focal_logits"] = accepted_logits
            arrays[f"{prefix}__extended_combined_union_focal_logits"] = (
                result.extended_union_logits
            )
            frozen_rows.append(
                {
                    "prefix": prefix,
                    "source_filename": source,
                    "draw_index": draw,
                    "dirty_sha256": dirty_sha,
                    "base_control_replayed": replay,
                    **unique.diagnostics(),
                    **result.diagnostics(),
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{spec.name}_adapter_unique_target_free",
                        "case": index + 1,
                        "case_count": len(aligned),
                        "nominated": len(nominated),
                        "unique": len(unique.unique_edges),
                        "accepted": len(accepted),
                        "choice": result.choice,
                    }
                ),
                flush=True,
            )
    archive, metadata, freeze = _freeze_artifacts(
        spec,
        output_dir=output_dir,
        arrays=arrays,
        rows=frozen_rows,
    )
    summary = {
        "case_count": len(frozen_rows),
        "control_replay_match_count": sum(
            bool(row["base_control_replayed"]) for row in frozen_rows
        ),
        "choice_counts": dict(Counter(str(row["choice"]) for row in frozen_rows)),
        "mean_nominated": float(
            np.mean([row["adapter_nominated_edge_count"] for row in frozen_rows])
        ),
        "mean_unique_proposed": float(
            np.mean([row["adapter_unique_proposed_count"] for row in frozen_rows])
        ),
        "mean_accepted_unique": float(
            np.mean([row["adapter_accepted_unique_count"] for row in frozen_rows])
        ),
    }
    summary["feasibility_passed"] = bool(
        summary["control_replay_match_count"] == spec.case_count
        and summary["mean_unique_proposed"] >= MEAN_UNIQUE_MINIMUM
        and MEAN_ACCEPTED_MINIMUM
        <= summary["mean_accepted_unique"]
        <= MEAN_ACCEPTED_MAXIMUM
    )
    return {
        "status": "target-free-frozen",
        "target_free_summary": summary,
        "runtime_seconds": perf_counter() - started,
        "artifacts": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
        },
    }


def _cluster_ci(
    values: Sequence[float], sources: Sequence[str], *, seed: int
) -> dict[str, Any]:
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
        "choice_counts": dict(Counter(str(row["choice"]) for row in rows)),
        "control_replay_match_count": sum(
            bool(row["base_control_replayed"]) for row in rows
        ),
    }
    sources = [str(row["source_filename"]) for row in rows]
    result["candidate_minus_control"] = {}
    for index, metric in enumerate(metrics):
        values = [
            float(row["metrics"][SCORED_ARMS[1]][metric])
            - float(row["metrics"][SCORED_ARMS[0]][metric])
            for row in rows
        ]
        summary = _cluster_ci(values, sources, seed=BOOTSTRAP_SEED + index)
        summary["case_wins_ties_losses"] = {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }
        result["candidate_minus_control"][metric] = summary
    fields = (
        "adapter_nominated_edges",
        "adapter_nominated_true_edges",
        "adapter_unique_proposed_edges",
        "adapter_unique_proposed_true_edges",
        "adapter_accepted_unique_edges",
        "adapter_accepted_unique_true_edges",
        "base_combined_union_edges",
        "base_combined_union_true_edges",
        "extended_combined_union_edges",
        "extended_combined_union_true_edges",
    )
    totals = {field: int(sum(row["supply"][field] for row in rows)) for field in fields}
    result["supply_totals"] = totals
    result["supply_mean_per_board"] = {
        field: float(np.mean([row["supply"][field] for row in rows]))
        for field in fields
    }
    result["supply_quality"] = {
        "nominated_precision": totals["adapter_nominated_true_edges"]
        / max(1, totals["adapter_nominated_edges"]),
        "unique_proposed_precision": totals["adapter_unique_proposed_true_edges"]
        / max(1, totals["adapter_unique_proposed_edges"]),
        "accepted_unique_precision": totals["adapter_accepted_unique_true_edges"]
        / max(1, totals["adapter_accepted_unique_edges"]),
        "accepted_true_edges_per_board": result["supply_mean_per_board"][
            "adapter_accepted_unique_true_edges"
        ],
        "extended_union_recall": result["supply_mean_per_board"][
            "extended_combined_union_true_edges"
        ]
        / PAIR_DENOMINATOR,
    }
    return result


def _score_panel(
    spec: PanelSpec,
    *,
    output_dir: Path,
    lookup: Mapping[str, Mapping[str, Any]],
    cache: Any,
) -> dict[str, Any]:
    archive, metadata, freeze = _triplet(spec, output_dir)
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
                cache, lookup[source], source, draw, dirty.dirty_tiles
            )
            truth = parent._truth_edges(reference)
            nominated = set(parent._edges(candidate, prefix, "adapter_nominated"))
            unique = set(parent._edges(candidate, prefix, "adapter_unique_proposed"))
            accepted = set(parent._edges(candidate, prefix, "adapter_accepted_unique"))
            base = set(parent._edges(candidate, prefix, "base_combined_union"))
            extended = set(parent._edges(candidate, prefix, "extended_combined_union"))
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
                    "base_control_replayed": frozen["base_control_replayed"],
                    "metrics": metrics,
                    "supply": {
                        "adapter_nominated_edges": len(nominated),
                        "adapter_nominated_true_edges": len(nominated & truth),
                        "adapter_unique_proposed_edges": len(unique),
                        "adapter_unique_proposed_true_edges": len(unique & truth),
                        "adapter_accepted_unique_edges": len(accepted),
                        "adapter_accepted_unique_true_edges": len(accepted & truth),
                        "base_combined_union_edges": len(base),
                        "base_combined_union_true_edges": len(base & truth),
                        "extended_combined_union_edges": len(extended),
                        "extended_combined_union_true_edges": len(extended & truth),
                    },
                }
            )
    return {
        "status": "scored",
        "rows": scored,
        "summary": _summarize(scored),
        "artifacts": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
        },
    }


def _target_free_gate(payload: Mapping[str, Any]) -> bool:
    return bool(payload["target_free_summary"]["feasibility_passed"])


def _focal_calibration(summary: Mapping[str, Any]) -> dict[str, Any]:
    quality = summary["supply_quality"]
    passed = bool(
        quality["accepted_unique_precision"] >= FOCAL_PRECISION_MINIMUM
        and quality["accepted_unique_precision"]
        >= quality["unique_proposed_precision"]
        and quality["accepted_true_edges_per_board"]
        >= FOCAL_TRUE_PER_BOARD_MINIMUM
    )
    return {
        "accepted_precision": quality["accepted_unique_precision"],
        "ungated_unique_precision": quality["unique_proposed_precision"],
        "accepted_true_edges_per_board": quality["accepted_true_edges_per_board"],
        "minimum_precision": FOCAL_PRECISION_MINIMUM,
        "minimum_true_edges_per_board": FOCAL_TRUE_PER_BOARD_MINIMUM,
        "passed": passed,
    }


def _setup_device(args: argparse.Namespace) -> torch.device:
    if args.inference_batch <= 0:
        raise ValueError("inference_batch must be positive")
    if args.allow_nondeterministic_mps != (args.device == "mps"):
        raise ValueError("MPS requires explicit --allow-nondeterministic-mps")
    device = torch.device(args.device)
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)
    return device


def _resources(targets: Path) -> tuple[Any, Any]:
    config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    return (
        finetune._manifest_lookup(config),
        finetune.CleanTileCache(targets.resolve(), maximum_boards=2),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_inputs()
    device = _setup_device(args)
    output_dir = args.output_dir.resolve()
    lookup, cache = _resources(args.targets)
    if args.mode == "freeze-local":
        output_dir.mkdir(parents=True, exist_ok=False)
        socket, adapter, focal = _load_models(device)
        device = next(focal.parameters()).device
        local = _freeze_panel(
            PANELS["local32"],
            output_dir=output_dir,
            socket=socket,
            adapter=adapter,
            focal=focal,
            device=device,
            inference_batch=args.inference_batch,
            lookup=lookup,
            cache=cache,
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": (
                "local-target-free-feasible-awaiting-decode"
                if _target_free_gate(local)
                else "target-free-feasibility-fail-stop-no-decode"
            ),
            "local32": local,
            "reference_reconstructed": False,
            "competition_test_accessed": False,
        }
        _write_json(output_dir / "freeze-local-report.json", report)
        print(json.dumps(report, indent=2))
        return report

    freeze_report_path = output_dir / "freeze-local-report.json"
    freeze_report = json.loads(freeze_report_path.read_text(encoding="utf-8"))
    if not _target_free_gate(freeze_report["local32"]):
        raise RuntimeError("local target-free feasibility failed; decode forbidden")
    started = perf_counter()
    local = _score_panel(
        PANELS["local32"], output_dir=output_dir, lookup=lookup, cache=cache
    )
    focal_calibration = _focal_calibration(local["summary"])
    pair_delta = local["summary"]["candidate_minus_control"][
        "satisfied_adjacent_pairs"
    ]["mean"]
    exact_delta = local["summary"]["candidate_minus_control"]["exact_tiles"][
        "mean"
    ]
    local_gate = bool(
        focal_calibration["passed"]
        and pair_delta >= LOCAL_PAIR_GATE
        and exact_delta >= LOCAL_EXACT_GATE
    )
    local["focal_calibration_gate"] = focal_calibration
    local["continuation_gate_passed"] = local_gate
    held: dict[str, Any] = {"status": "skipped_by_local_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_gate"}
    if local_gate:
        socket, adapter, focal = _load_models(device)
        device = next(focal.parameters()).device
        held_freeze = _freeze_panel(
            PANELS["held32"],
            output_dir=output_dir,
            socket=socket,
            adapter=adapter,
            focal=focal,
            device=device,
            inference_batch=args.inference_batch,
            lookup=lookup,
            cache=cache,
        )
        if _target_free_gate(held_freeze):
            held = _score_panel(
                PANELS["held32"], output_dir=output_dir, lookup=lookup, cache=cache
            )
            held["target_free"] = held_freeze
            held_delta = held["summary"]["candidate_minus_control"][
                "satisfied_adjacent_pairs"
            ]["mean"]
            held["continuation_gate_passed"] = bool(held_delta >= HELD_PAIR_GATE)
            if held["continuation_gate_passed"]:
                fresh_freeze = _freeze_panel(
                    PANELS["fresh32"],
                    output_dir=output_dir,
                    socket=socket,
                    adapter=adapter,
                    focal=focal,
                    device=device,
                    inference_batch=args.inference_batch,
                    lookup=lookup,
                    cache=cache,
                )
                if _target_free_gate(fresh_freeze):
                    fresh = _score_panel(
                        PANELS["fresh32"],
                        output_dir=output_dir,
                        lookup=lookup,
                        cache=cache,
                    )
                    fresh["target_free"] = fresh_freeze
                else:
                    fresh = {
                        "status": "target-free-feasibility-fail-stop-no-decode",
                        "target_free": fresh_freeze,
                    }
        else:
            held = {
                "status": "target-free-feasibility-fail-stop-no-decode",
                "target_free": held_freeze,
            }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "protocol": {
            "adapter_checkpoint_step": 400,
            "adapter_nominator": ADAPTER_NOMINATOR,
            "topk": ADAPTER_TOPK,
            "deduplicate_before_focal": "current + selective + confirmed fullres",
            "dirty_visible_focal_logit_minimum": ADAPTER_FOCAL_LOGIT_MINIMUM,
            "combined_order": "current + selective + unique fullres + unique adapter",
            "raw_dense_matrices_unchanged": True,
            "selector_roster": list(FUSION_ARM_NAMES),
            "new_standalone_arm": False,
            "tail": "unchanged focal-gated non-adjacent tail96",
            "local_pair_gate": LOCAL_PAIR_GATE,
            "local_exact_gate": LOCAL_EXACT_GATE,
            "held_pair_gate": HELD_PAIR_GATE,
            "no_threshold_budget_or_roster_sweep": True,
        },
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "adapter_pixels_matcher_only": True,
            "raw_d64_evidence_saved_parallel": True,
            "strict_original_upright_tile_permutations": True,
            "targets_used_only_after_candidate_freeze": True,
            "adapter_fit_disjoint_from_all_scored_panels": True,
            "competition_test_accessed": False,
            "submission_created": False,
            "postprocessing_used": False,
            "production_or_official_best_modified": False,
        },
        "artifacts": {
            "config": _record(CONFIG),
            "adapter_checkpoint": _record(ADAPTER_CHECKPOINT),
            "adapter_report": _record(ADAPTER_REPORT),
            "socket_checkpoint": _record(SOCKET_CHECKPOINT),
            "focal_checkpoint": _record(FOCAL_CHECKPOINT),
            "runner": _record(Path(__file__).resolve()),
            "module": _record(
                PROJECT_ROOT
                / "src/aiijc_puzzle/taska_retrieval_adapter_unique_supply.py"
            ),
            "raw_solver": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
            ),
        },
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps({name: report[name] for name in PANELS}, indent=2))
    return report


if __name__ == "__main__":
    run(parse_args())
