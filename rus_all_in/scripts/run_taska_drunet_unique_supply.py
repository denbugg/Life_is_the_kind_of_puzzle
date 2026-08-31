#!/usr/bin/env python3
"""Gate one official-DRUNet descriptor nominator above frozen TASKA fusion.

DRUNet pixels are used only to nominate reciprocal border-descriptor edges.
All parent edges are removed, the remainder is accepted only by the frozen
dirty-pixel focal verifier, and only the existing combined arm is extended.
The original dense costs, six-arm roster and focal-gated tail96 are unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.pretrained_tile_denoiser import (
    load_drunet_color,
    render_drunet_tiles,
)
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.raw_tail_global_solver import RawTailEdge
from aiijc_puzzle.taska_drunet_unique_supply import (
    DRUNET_NOMINATOR,
    DRUNET_SIGMA_255,
    accept_unique_drunet_proposals,
    compose_drunet_unique_fusion,
    restored_descriptor_mutual_edges,
    unique_drunet_proposals,
)
from aiijc_puzzle.taska_focal_verifier import (
    TASKA_FOCAL_VERIFIER_SHA256,
    load_taska_focal_verifier,
    score_focal_edges,
)
from aiijc_puzzle.taska_pair_pipeline import (
    FOCAL_MODE,
    PAIR_DENOMINATOR,
    TaskaPairArtifactPaths,
)
from aiijc_puzzle.taska_selective_fullres_fusion import FUSION_ARM_NAMES

try:
    from scripts import run_taska_focal_current_finetune as finetune
    from scripts import run_taska_selective_fullres_fusion as parent
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_taska_focal_current_finetune as finetune
    import run_taska_selective_fullres_fusion as parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/taska-drunet-unique-supply/fixed-v1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_drunet_unique_supply_fixed_v1.json"
DRUNET_CHECKPOINT = (
    PROJECT_ROOT / "artifacts/pretrained-denoisers/kair-fc1732f/drunet_color.pth"
)
DRUNET_CHECKPOINT_SHA256 = (
    "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4"
)
CONFIG_SHA256 = "6090dd715604b333f0e0df37d4673dd30099ef49946fb93dc871b67c24292aa2"
FUSION_ROOT = PROJECT_ROOT / "outputs/taska-selective-fullres-union-fusion/fixed-v1"
GRID = 24
COUNT = GRID * GRID
LOCAL_GATE = 0.0
HELD_GATE = 0.5
FOCAL_LOGIT_MINIMUM = 0.0
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_203
REPORT_SCHEMA = "aiijc-taska-drunet-unique-supply-report-v1"
SCORED_ARMS = ("selective_fullres_control", "drunet_unique_candidate")


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
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--inference-batch", type=int, default=144)
    parser.add_argument("--smoke-one", action="store_true")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rendered = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


def _config(path: Path) -> Mapping[str, Any]:
    resolved = path.resolve()
    if sha256_file(resolved) != CONFIG_SHA256:
        raise RuntimeError("DRUNet unique-supply preregistration hash mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen-before-target-assisted-scoring":
        raise RuntimeError("DRUNet unique-supply preregistration status changed")
    return payload


def _require_inputs(config: Mapping[str, Any]) -> None:
    parent._require_inputs()
    if sha256_file(DRUNET_CHECKPOINT) != DRUNET_CHECKPOINT_SHA256:
        raise RuntimeError("official DRUNet checkpoint hash mismatch")
    if TaskaPairArtifactPaths().focal_verifier.resolve().stat().st_size <= 0:
        raise RuntimeError("focal verifier checkpoint is absent")
    declared = config["frozen_parent_artifacts"]
    for name, spec in PANELS.items():
        expected = declared[name]
        for path, key in (
            (spec.fusion_archive, "archive_sha256"),
            (spec.fusion_metadata, "metadata_sha256"),
            (spec.fusion_freeze, "pre_score_freeze_sha256"),
        ):
            if not path.is_file() or sha256_file(path) != expected[key]:
                raise RuntimeError(f"{name} frozen fusion parent changed: {path.name}")
        parent._validate_freeze(spec.fusion_freeze)


def _rows(path: Path, count: int) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) < count:
        raise ValueError(f"{path} has fewer than {count} rows")
    return rows[:count]


def _aligned_rows(spec: PanelSpec) -> list[tuple[Mapping[str, Any], ...]]:
    parents = parent._aligned_rows(spec.parent)
    fusion = _rows(spec.fusion_metadata, spec.case_count)
    identity = ("prefix", "source_filename", "draw_index", "dirty_sha256")
    result: list[tuple[Mapping[str, Any], ...]] = []
    for records, fusion_row in zip(parents, fusion, strict=True):
        if any(records[0].get(field) != fusion_row.get(field) for field in identity):
            raise RuntimeError(f"{spec.name} fusion-parent row identity mismatch")
        result.append((*records, fusion_row))
    return result


def _write_json(path: Path, payload: Any) -> None:
    parent._write_json(path, payload)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    parent._write_npz(path, arrays)


def _edge_arrays(
    prefix: str, name: str, edges: Sequence[RawTailEdge]
) -> dict[str, np.ndarray]:
    return parent._edge_arrays(prefix, name, edges)


def _freeze(
    spec: PanelSpec,
    *,
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    config_path: Path,
) -> tuple[Path, Path, Path]:
    stage = output_dir / spec.name
    stage.mkdir(parents=True, exist_ok=False)
    archive = stage / "frozen-target-free-eval.npz"
    metadata = archive.with_suffix(".json")
    freeze = stage / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-drunet-unique-supply-target-free-v1",
            "stage": spec.name,
            "contains_exact_references_or_candidate_labels": False,
            "restored_pixels_matcher_only": True,
            "drunet_nominator": DRUNET_NOMINATOR,
            "deduplicate_before_focal": True,
            "dirty_visible_focal_logit_minimum": FOCAL_LOGIT_MINIMUM,
            "selector_roster": list(FUSION_ARM_NAMES),
            "new_standalone_arm": False,
            "raw_dense_costs_unchanged": True,
            "tail": "unchanged focal-gated non-adjacent tail96",
            "all_layouts_strict_original_upright_permutations": True,
            "rows": list(rows),
        },
    )
    artifacts = {
        "frozen_archive": archive,
        "frozen_metadata": metadata,
        "config": config_path,
        "drunet_checkpoint": DRUNET_CHECKPOINT,
        "focal_verifier": TaskaPairArtifactPaths().focal_verifier,
        "fusion_parent_archive": spec.fusion_archive,
        "fusion_parent_metadata": spec.fusion_metadata,
        "fusion_parent_freeze": spec.fusion_freeze,
        "runner": Path(__file__).resolve(),
        "module": PROJECT_ROOT / "src/aiijc_puzzle/taska_drunet_unique_supply.py",
        "raw_solver": PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py",
    }
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-drunet-unique-supply-pre-score-freeze-v1",
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
        "choice_counts": dict(Counter(row["choice"] for row in rows)),
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
        "drunet_nominated_edges",
        "drunet_nominated_true_edges",
        "drunet_unique_proposed_edges",
        "drunet_unique_proposed_true_edges",
        "drunet_accepted_unique_edges",
        "drunet_accepted_unique_true_edges",
        "base_combined_union_edges",
        "base_combined_union_true_edges",
        "extended_combined_union_edges",
        "extended_combined_union_true_edges",
    )
    totals = {field: int(sum(row["supply"][field] for row in rows)) for field in fields}
    result["supply_totals"] = totals
    result["supply_mean_per_board"] = {
        field: float(np.mean([row["supply"][field] for row in rows])) for field in fields
    }
    result["supply_quality"] = {
        "nominated_precision": totals["drunet_nominated_true_edges"]
        / max(1, totals["drunet_nominated_edges"]),
        "unique_proposed_precision": totals["drunet_unique_proposed_true_edges"]
        / max(1, totals["drunet_unique_proposed_edges"]),
        "accepted_unique_precision": totals["drunet_accepted_unique_true_edges"]
        / max(1, totals["drunet_accepted_unique_edges"]),
        "extended_union_recall": result["supply_mean_per_board"][
            "extended_combined_union_true_edges"
        ]
        / PAIR_DENOMINATOR,
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
                cache, lookup[source], source, draw, dirty.dirty_tiles
            )
            truth = parent._truth_edges(reference)
            nominated = set(parent._edges(candidate, prefix, "drunet_nominated"))
            unique = set(parent._edges(candidate, prefix, "drunet_unique_proposed"))
            accepted = set(parent._edges(candidate, prefix, "drunet_accepted_unique"))
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
                        "drunet_nominated_edges": len(nominated),
                        "drunet_nominated_true_edges": len(nominated & truth),
                        "drunet_unique_proposed_edges": len(unique),
                        "drunet_unique_proposed_true_edges": len(unique & truth),
                        "drunet_accepted_unique_edges": len(accepted),
                        "drunet_accepted_unique_true_edges": len(accepted & truth),
                        "base_combined_union_edges": len(base),
                        "base_combined_union_true_edges": len(base & truth),
                        "extended_combined_union_edges": len(extended),
                        "extended_combined_union_true_edges": len(extended & truth),
                    },
                }
            )
    return scored, _summarize(scored)


def _run_panel(
    spec: PanelSpec,
    *,
    output_dir: Path,
    config_path: Path,
    drunet: torch.nn.Module,
    focal: torch.nn.Module,
    device: torch.device,
    inference_batch: int,
    lookup: Mapping[str, Mapping[str, Any]] | None,
    cache: Any | None,
    target_free_only: bool,
) -> dict[str, Any]:
    aligned = _aligned_rows(spec)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
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
            if lookup is None or cache is None:
                raise RuntimeError("dirty-case reconstruction resources are absent")
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

            restored, restoration = render_drunet_tiles(
                drunet,
                dirty.dirty_tiles,
                sigma_255=DRUNET_SIGMA_255,
                device=device,
                batch_size=inference_batch,
            )
            nominated = restored_descriptor_mutual_edges(restored)
            unique = unique_drunet_proposals(
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
                accepted, accepted_logits = accept_unique_drunet_proposals(
                    unique.unique_edges, focal_result.logits
                )
                proposed_logits = np.asarray(focal_result.logits, dtype=np.float32)
            else:
                accepted = ()
                accepted_logits = np.empty(0, dtype=np.float32)
                proposed_logits = np.empty(0, dtype=np.float32)

            result = compose_drunet_unique_fusion(
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
                drunet_accepted_edges=accepted,
                drunet_accepted_logits=accepted_logits,
            )
            replay = bool(
                np.array_equal(
                    result.control_layout,
                    fusion[f"{prefix}__combined_union_candidate_layout"],
                )
            )
            if not replay:
                raise RuntimeError("frozen selective+fullres control mismatch")

            arrays[f"{prefix}__selective_fullres_control_layout"] = result.control_layout
            arrays[f"{prefix}__drunet_unique_candidate_layout"] = result.candidate_layout
            arrays[f"{prefix}__extended_combined_layout"] = result.extended_combined_layout
            for name, edges in (
                ("drunet_nominated", nominated),
                ("drunet_unique_proposed", unique.unique_edges),
                ("drunet_accepted_unique", accepted),
                ("base_combined_union", result.base.supply.combined_union_edges),
                ("extended_combined_union", result.extended_union_edges),
            ):
                arrays.update(_edge_arrays(prefix, name, edges))
            arrays[f"{prefix}__drunet_unique_proposed_focal_logits"] = proposed_logits
            arrays[f"{prefix}__drunet_accepted_unique_focal_logits"] = accepted_logits
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
                    "restoration": asdict(restoration),
                    **unique.diagnostics(),
                    **result.diagnostics(),
                }
            )
            print(
                json.dumps(
                    {
                        "event": f"{spec.name}_drunet_unique_target_free",
                        "case": index + 1,
                        "case_count": len(aligned),
                        "nominated": len(nominated),
                        "unique": len(unique.unique_edges),
                        "accepted": len(accepted),
                        "choice": result.choice,
                        "control_replay": replay,
                    }
                ),
                flush=True,
            )
    archive, metadata, freeze = _freeze(
        spec,
        output_dir=output_dir,
        arrays=arrays,
        rows=frozen_rows,
        config_path=config_path,
    )
    payload: dict[str, Any] = {
        "status": "target-free-smoke" if target_free_only else "complete",
        "target_free_summary": {
            "case_count": len(frozen_rows),
            "control_replay_match_count": sum(
                bool(row["base_control_replayed"]) for row in frozen_rows
            ),
            "choice_counts": dict(Counter(row["choice"] for row in frozen_rows)),
            "mean_nominated": float(
                np.mean([row["drunet_nominated_edge_count"] for row in frozen_rows])
            ),
            "mean_unique_proposed": float(
                np.mean([row["drunet_unique_proposed_count"] for row in frozen_rows])
            ),
            "mean_accepted_unique": float(
                np.mean([row["drunet_accepted_unique_count"] for row in frozen_rows])
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
    config_path = args.config.resolve()
    config = _config(config_path)
    _require_inputs(config)
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
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    drunet = load_drunet_color(DRUNET_CHECKPOINT, device)
    focal = load_taska_focal_verifier(
        TaskaPairArtifactPaths().focal_verifier, device=device
    )
    if getattr(focal, "checkpoint_sha256", None) != TASKA_FOCAL_VERIFIER_SHA256:
        raise RuntimeError("loaded focal verifier lineage changed")
    # ``torch.device('mps')`` and the concrete parameter device ``mps:0`` are
    # not equality-comparable on current PyTorch.  Use the model's resolved
    # device for both DRUNet and focal inference calls.
    device = next(focal.parameters()).device
    finetune_config, _, _ = finetune._load_config(finetune.DEFAULT_CONFIG)
    lookup = finetune._manifest_lookup(finetune_config)
    cache = finetune.CleanTileCache(args.targets.resolve(), maximum_boards=2)

    if args.smoke_one:
        base = PANELS["local32"]
        smoke = PanelSpec(
            name="smoke1",
            case_count=1,
            parent=parent.PanelSpec(**{**base.parent.__dict__, "name": "smoke1", "case_count": 1}),
            fusion_archive=base.fusion_archive,
            fusion_metadata=base.fusion_metadata,
            fusion_freeze=base.fusion_freeze,
        )
        local = _run_panel(
            smoke,
            output_dir=output_dir,
            config_path=config_path,
            drunet=drunet,
            focal=focal,
            device=device,
            inference_batch=args.inference_batch,
            lookup=lookup,
            cache=cache,
            target_free_only=True,
        )
        report = {
            "schema": REPORT_SCHEMA,
            "status": "target-free-smoke",
            "local32": local,
            "reference_reconstructed": False,
            "competition_test_accessed": False,
        }
        _write_json(output_dir / "report.json", report)
        print(json.dumps(report, indent=2))
        return report

    local = _run_panel(
        PANELS["local32"],
        output_dir=output_dir,
        config_path=config_path,
        drunet=drunet,
        focal=focal,
        device=device,
        inference_batch=args.inference_batch,
        lookup=lookup,
        cache=cache,
        target_free_only=False,
    )
    local_delta = local["summary"]["candidate_minus_control"][
        "satisfied_adjacent_pairs"
    ]["mean"]
    held: dict[str, Any] = {"status": "skipped_by_negative_local_pair_gate"}
    fresh: dict[str, Any] = {"status": "skipped_by_local_or_held_pair_gate"}
    if local_delta >= LOCAL_GATE:
        held = _run_panel(
            PANELS["held32"],
            output_dir=output_dir,
            config_path=config_path,
            drunet=drunet,
            focal=focal,
            device=device,
            inference_batch=args.inference_batch,
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
                config_path=config_path,
                drunet=drunet,
                focal=focal,
                device=device,
                inference_batch=args.inference_batch,
                lookup=lookup,
                cache=cache,
                target_free_only=False,
            )
        else:
            fresh = {"status": "skipped_by_held_pair_delta_below_0.5"}
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "protocol": {
            "official_drunet_sigma_255": DRUNET_SIGMA_255,
            "nominator": DRUNET_NOMINATOR,
            "restored_descriptor": "existing normalized grayscale width6",
            "deduplicate_before_focal": "current + selective + confirmed fullres",
            "dirty_visible_focal_logit_minimum": FOCAL_LOGIT_MINIMUM,
            "combined_order": "current + selective + unique fullres + unique DRUNet",
            "raw_dense_matrices_unchanged": True,
            "selector_roster": list(FUSION_ARM_NAMES),
            "new_standalone_arm": False,
            "tail": "unchanged focal-gated non-adjacent tail96",
            "control": "exact frozen selective+fullres fusion final layout",
            "local_pair_gate": LOCAL_GATE,
            "held_pair_gate": HELD_GATE,
            "no_threshold_budget_or_roster_sweep": True,
        },
        "local32": local,
        "held32": held,
        "fresh32": fresh,
        "runtime_seconds": perf_counter() - started,
        "legality": {
            "restored_pixels_matcher_only": True,
            "strict_original_upright_tile_permutations": True,
            "targets_used_only_after_candidate_freeze": True,
            "competition_test_accessed": False,
            "submission_created": False,
            "postprocessing_used": False,
            "production_or_official_best_modified": False,
        },
        "artifacts": {
            "config": _record(config_path),
            "drunet_checkpoint": _record(DRUNET_CHECKPOINT),
            "focal_verifier": _record(TaskaPairArtifactPaths().focal_verifier),
            "runner": _record(Path(__file__).resolve()),
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/taska_drunet_unique_supply.py"
            ),
            "raw_solver": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
            ),
        },
    }
    _write_json(output_dir / "report.json", report)
    print(
        json.dumps(
            {name: report[name] for name in ("local32", "held32", "fresh32")},
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    run(parse_args())
