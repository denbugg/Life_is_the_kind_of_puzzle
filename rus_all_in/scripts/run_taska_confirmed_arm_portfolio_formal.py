#!/usr/bin/env python3
"""Run the preregistered source16xdraw2 confirmed-arm portfolio confirmation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aiijc_puzzle.layout_evaluation import evaluate_layout
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.taska_confirmed_arm_portfolio import (
    CONFIRMED_ARM_NAMES,
    compose_confirmed_arm_portfolio,
)
from aiijc_puzzle.taska_focal_verifier import score_focal_edges
from aiijc_puzzle.taska_fullres_union_voter import (
    accept_focal_proposals,
    compose_fullres_union_focal_arm,
    load_fullres_denoiser,
    restore_fixed_matcher_view,
    restored_mutual_scorer_sets,
    supported_absent_edges,
)
from aiijc_puzzle.taska_pair_pipeline import FOCAL_MODE, GRID_SIZE, PAIR_DENOMINATOR
from aiijc_puzzle.taska_seam_matcher import match_taska_tiles
from aiijc_puzzle.taska_selective_fullres_fusion import compose_selective_fullres_fusion
from aiijc_puzzle.taska_selective_vote500 import (
    compose_selective_vote500,
    same_pass_target350,
)
from aiijc_puzzle.taska_vote500 import VOTE500_MATCHER_CONFIG

try:
    from scripts import run_taska_selective_fullres_union_fusion_fresh32_confirmation as parent
except ModuleNotFoundError:
    import run_taska_selective_fullres_union_fusion_fresh32_confirmation as parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/taska_confirmed_arm_portfolio_v1.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/taska-confirmed-arm-portfolio/formal-source16-draw2-v1"
)
FULLRES_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/"
    "fullres_boundary_denoiser.pt"
)
SOURCE_COUNT = 16
DRAWS = (0, 1)
CASE_COUNT = 32
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 2_026_083_212
CONTROL_ARM = "confirmed_six_arm_control"
CANDIDATE_ARM = "seven_arm_candidate"
ARMS = (CONTROL_ARM, CANDIDATE_ARM)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--inference-batch", type=int, default=576)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        rendered = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        rendered = str(resolved)
    return {"path": rendered, "sha256": sha256_file(resolved)}


def _load_config(path: Path) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed parent preregistration is missing")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256_file(resolved):
        raise ValueError("preregistration SHA-256 sidecar mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    roster = tuple(config["formal_roster_selection"]["source_filenames"])
    if len(roster) != SOURCE_COUNT or len(set(roster)) != SOURCE_COUNT:
        raise ValueError("formal roster must contain 16 distinct sources")
    if config.get("selector_roster") != list(CONFIRMED_ARM_NAMES):
        raise ValueError("fixed seven-arm roster changed")
    if config.get("formal_open_rule") != (
        "open only after local, held, and fresh gates complete successfully"
    ):
        raise ValueError("formal opening rule changed")
    if config.get("no_sweep") is not True:
        raise ValueError("no-sweep contract changed")
    return config, roster


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _target_free_case(
    dirty_tiles: np.ndarray,
    *,
    resources: Any,
    denoiser: Any,
    inference_batch: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    matched500 = match_taska_tiles(
        dirty_tiles,
        resources.matchers,
        config=VOTE500_MATCHER_CONFIG,
        device=resources.device,
        require_verified=True,
    )
    focal500 = score_focal_edges(
        resources.focal_verifier,
        dirty_tiles,
        matched500.cost_right,
        matched500.cost_down,
        matched500.candidate_edges,
        mode=FOCAL_MODE,
        grid=GRID_SIZE,
        device=resources.device,
    )
    selective = compose_selective_vote500(matched500, focal500, resources)
    matched350, focal350 = same_pass_target350(matched500, focal500)
    four = parent._four_layouts(matched350, focal350, resources)
    supply = selective.supply
    restored = restore_fixed_matcher_view(
        denoiser,
        dirty_tiles,
        device=resources.device,
        batch_size=inference_batch,
    )
    scorer_sets = restored_mutual_scorer_sets(
        restored,
        resources.matchers,
        device=resources.device,
    )
    proposed, support = supported_absent_edges(supply.current_edges, scorer_sets)
    proposal_scores = score_focal_edges(
        resources.focal_verifier,
        dirty_tiles,
        matched350.cost_right,
        matched350.cost_down,
        proposed,
        mode=FOCAL_MODE,
        grid=GRID_SIZE,
        device=resources.device,
    )
    accepted, accepted_logits = accept_focal_proposals(proposed, proposal_scores.logits)
    fusion = compose_selective_fullres_fusion(
        cost_right=matched350.cost_right,
        cost_down=matched350.cost_down,
        four_layouts=four,
        frozen_selective_control=selective.candidate_layout,
        current_edges=supply.current_edges,
        current_logits=supply.current_logits,
        selective_new_edges=supply.accepted_new_edges,
        selective_new_logits=supply.accepted_new_logits,
        fullres_accepted_edges=accepted,
        fullres_accepted_logits=accepted_logits,
        grid=GRID_SIZE,
    )
    fullres = compose_fullres_union_focal_arm(
        cost_right=matched350.cost_right,
        cost_down=matched350.cost_down,
        four_layouts=four,
        current_edges=supply.current_edges,
        current_focal_logits=supply.current_logits,
        accepted_new_edges=accepted,
        accepted_new_logits=accepted_logits,
        grid=GRID_SIZE,
    )
    fullres_union_edges = supply.current_edges + accepted
    fullres_union_logits = np.concatenate((supply.current_logits, accepted_logits))
    result = compose_confirmed_arm_portfolio(
        cost_right=matched350.cost_right,
        cost_down=matched350.cost_down,
        four_layouts=four,
        selective_union_layout=fusion.selective_union_layout,
        combined_union_layout=fusion.combined_union_layout,
        fullres_union_layout=fullres.fullres_layout,
        frozen_fusion_control=fusion.candidate_layout,
        current_edges=supply.current_edges,
        current_logits=supply.current_logits,
        selective_union_edges=fusion.supply.selective_union_edges,
        selective_union_logits=fusion.supply.selective_union_logits,
        combined_union_edges=fusion.supply.combined_union_edges,
        combined_union_logits=fusion.supply.combined_union_logits,
        fullres_union_edges=fullres_union_edges,
        fullres_union_logits=fullres_union_logits,
        grid=GRID_SIZE,
    )
    if not np.array_equal(result.mechanical_control_layout, fusion.candidate_layout):
        raise RuntimeError("confirmed six-arm control did not replay")
    return (
        result.control_layout,
        result.candidate_layout,
        {
            **result.diagnostics(),
            "target500_candidate_count": len(matched500.candidate_edges),
            "current_edge_count": len(supply.current_edges),
            "selective_accepted_count": len(supply.accepted_new_edges),
            "fullres_proposed_count": len(proposed),
            "fullres_accepted_count": len(accepted),
            "fullres_support_histogram": dict(Counter(int(value) for value in support)),
            "one_target500_matcher_pass": True,
            "restored_pixels_matcher_only": True,
        },
    )


def _validate_freeze(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("created_before_exact_reference_reconstruction") is not True:
        raise RuntimeError("pre-score freeze timing contract changed")
    if payload.get("contains_evaluation_references_or_labels") is not False:
        raise RuntimeError("pre-score freeze contains labels")
    for name, record in payload["artifacts"].items():
        artifact = Path(record["path"])
        if not artifact.is_absolute():
            artifact = PROJECT_ROOT / artifact
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"pre-score artifact changed: {name}")


def _layout_metrics(layout: Any, exact: Any) -> dict[str, Any]:
    result = evaluate_layout(layout, exact, reference_is_exact=True)
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
        grouped[source].append(float(value))
    if len(grouped) != SOURCE_COUNT or any(len(values) != 2 for values in grouped.values()):
        raise ValueError("formal bootstrap requires 16 sources x two draws")
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
        "case_wins_ties_losses": {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        },
        "source_wins_ties_losses": {
            "wins": int(np.sum(means > 0)),
            "ties": int(np.sum(means == 0)),
            "losses": int(np.sum(means < 0)),
        },
        "source_count": SOURCE_COUNT,
        "case_count": CASE_COUNT,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config, roster = _load_config(args.config)
    manifest_path = PROJECT_ROOT / "data/interim/validation_manifest.json"
    manifest, _ = parent._load_manifest(manifest_path)
    specs = [(manifest[name], name, draw) for name in roster for draw in DRAWS]
    if args.validate_only:
        result = {"status": "validated", "roster": list(roster), "case_count": len(specs)}
        print(json.dumps(result, indent=2))
        return result
    device = parent.synthetic._select_device(
        args.device,
        allow_nondeterministic_mps=bool(args.allow_nondeterministic_mps),
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    resources = parent.load_taska_pair_pipeline_resources(device=device)
    denoiser = load_fullres_denoiser(FULLRES_CHECKPOINT, device=resources.device)
    cache = parent.synthetic.CleanTileCache(args.targets.resolve(), maximum_boards=2)
    arrays: dict[str, np.ndarray] = {}
    frozen_rows: list[dict[str, Any]] = []
    started = perf_counter()
    for index, (record, source, draw) in enumerate(specs):
        prefix = f"case_{index:03d}"
        dirty = parent.synthetic._dirty_case(cache, record, source, draw)
        control, candidate, diagnostics = _target_free_case(
            dirty.dirty_tiles,
            resources=resources,
            denoiser=denoiser,
            inference_batch=args.inference_batch,
        )
        arrays[f"{prefix}__{CONTROL_ARM}_layout"] = control
        arrays[f"{prefix}__{CANDIDATE_ARM}_layout"] = candidate
        frozen_rows.append(
            {
                "prefix": prefix,
                "case_id": dirty.case_id,
                "source_filename": source,
                "draw_index": draw,
                "dirty_sha256": parent.synthetic._dirty_sha256(dirty.dirty_tiles),
                **diagnostics,
            }
        )
        print(
            json.dumps(
                {
                    "event": "confirmed_arm_portfolio_formal_target_free",
                    "case": index + 1,
                    "case_count": CASE_COUNT,
                    "choice": diagnostics["choice"],
                }
            ),
            flush=True,
        )
    archive = output / "frozen-target-free-eval.npz"
    metadata = output / "frozen-target-free-eval.json"
    freeze = output / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-taska-confirmed-arm-portfolio-formal-target-free-v1",
            "contains_exact_references_or_candidate_labels": False,
            "selector_roster": list(CONFIRMED_ARM_NAMES),
            "restored_pixels_matcher_only": True,
            "strict_original_upright_permutations": True,
            "rows": frozen_rows,
        },
    )
    artifacts = {
        "archive": _record(archive),
        "metadata": _record(metadata),
        "preregistration": _record(args.config.resolve()),
        "preregistration_sidecar": _record(
            args.config.resolve().with_suffix(args.config.suffix + ".sha256")
        ),
        "formal_runner": _record(Path(__file__).resolve()),
        "portfolio_module": _record(
            PROJECT_ROOT / "src/aiijc_puzzle/taska_confirmed_arm_portfolio.py"
        ),
        "raw_solver": _record(PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"),
        "fullres_checkpoint": _record(FULLRES_CHECKPOINT),
    }
    _write_json(
        freeze,
        {
            "schema": "aiijc-taska-confirmed-arm-portfolio-formal-freeze-v1",
            "created_before_exact_reference_reconstruction": True,
            "contains_evaluation_references_or_labels": False,
            "artifacts": artifacts,
        },
    )
    inference_seconds = perf_counter() - started
    print(
        json.dumps(
            {
                "event": "formal_all_target_free_evidence_frozen",
                "archive_sha256": sha256_file(archive),
                "reference_reconstructed_yet": False,
            }
        ),
        flush=True,
    )
    _validate_freeze(freeze)
    scored: list[dict[str, Any]] = []
    with np.load(archive, allow_pickle=False) as frozen:
        for (record, source, draw), row in zip(specs, frozen_rows, strict=True):
            dirty, reference = parent.make_exact_synthetic_case(
                cache.load(record),
                source_filename=source,
                draw_index=draw,
                seed=parent.synthetic.SYNTHETIC_SEED,
            )
            if (
                dirty.case_id != row["case_id"]
                or parent.synthetic._dirty_sha256(dirty.tiles) != row["dirty_sha256"]
            ):
                raise RuntimeError("formal scoring recreated a different dirty case")
            exact = parent.strict_layout(reference.tile_at_position, grid=GRID_SIZE)
            prefix = row["prefix"]
            scored.append(
                {
                    "source_filename": source,
                    "draw_index": draw,
                    CONTROL_ARM: _layout_metrics(
                        frozen[f"{prefix}__{CONTROL_ARM}_layout"], exact
                    ),
                    CANDIDATE_ARM: _layout_metrics(
                        frozen[f"{prefix}__{CANDIDATE_ARM}_layout"], exact
                    ),
                }
            )
    metric_names = ("satisfied_adjacent_pairs", "adjacency_recall", "exact_tiles")
    arm_means = {
        arm: {
            metric: float(np.mean([row[arm][metric] for row in scored]))
            for metric in metric_names
        }
        for arm in ARMS
    }
    sources = [row["source_filename"] for row in scored]
    deltas = {
        metric: _cluster_ci(
            [row[CANDIDATE_ARM][metric] - row[CONTROL_ARM][metric] for row in scored],
            sources,
            seed=BOOTSTRAP_SEED + index,
        )
        for index, metric in enumerate(metric_names)
    }
    gate_config = config["formal_gate"]
    primary = deltas["satisfied_adjacent_pairs"]
    gate_passed = (
        primary["mean"] >= gate_config["pair_mean_minimum"]
        and primary["ci95_lower"] >= gate_config["source_cluster_ci95_lower_minimum"]
    )
    report = {
        "schema": "aiijc-taska-confirmed-arm-portfolio-formal-report-v1",
        "status": "confirmed" if gate_passed else "not-confirmed",
        "panel": {"source_filenames": list(roster), "draws": list(DRAWS)},
        "metrics": {
            "pair_denominator": PAIR_DENOMINATOR,
            "arms": arm_means,
            "candidate_minus_control": deltas,
            "confirmation_gate": {
                **gate_config,
                "observed_pair_mean": primary["mean"],
                "observed_pair_ci95_lower": primary["ci95_lower"],
                "passed": gate_passed,
            },
        },
        "rows": scored,
        "frozen_eval": {
            "archive": _record(archive),
            "metadata": _record(metadata),
            "pre_score_freeze": _record(freeze),
        },
        "runtime_seconds": {
            "target_free": inference_seconds,
            "total": perf_counter() - started,
        },
        "legality": {
            "targets_used_only_after_candidate_freeze": True,
            "restored_pixels_matcher_only": True,
            "strict_original_upright_permutations": True,
            "competition_test_accessed": False,
            "postprocessing_used": False,
            "production_modified": False,
        },
    }
    _write_json(output / "report.json", report)
    print(json.dumps(report["metrics"], indent=2), flush=True)
    return report


if __name__ == "__main__":
    run(parse_args())
