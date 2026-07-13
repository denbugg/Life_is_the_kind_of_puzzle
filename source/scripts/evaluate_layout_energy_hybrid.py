#!/usr/bin/env python3
"""Test a failed frozen critic heatmap on strong frozen real16 HBT/QAP layouts.

Prediction is raw-only and target-free.  Every proposed move is an exact tile
swap ranked by raw RGB border-L1 delta.  The critic may (a) choose the suspect
positions and (b) optionally rerank a tiny seam shortlist.  An independent
classical weak-seam-position control gets exactly the same proposal budget,
and a budget-matched no-op records/discards the hybrid proposal pool.

All layouts are atomically serialized before any target path is opened.  The
post-hoc report can diagnose signal but is always ``safe_for_submission=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
from scipy.stats import spearmanr
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puzzle_assembly.layout_energy_hybrid import (  # noqa: E402
    FrozenCriticScorer,
    SwapProposal,
    critic_rerank_or_noop,
    load_failed_frozen_critic,
    local_seam_costs,
    raw_border_l1_seam,
    seam_objective,
    seam_select_or_noop,
    sha256_array,
    sha256_file,
    top_delta_swaps,
    validate_layout,
)
from puzzle_assembly.metrics import predicted_image_metrics  # noqa: E402
from puzzle_denoise_v2.tiles import split_tiles_numpy  # noqa: E402


EXPECTED_CHECKPOINT_SHA256 = (
    "039cd7638731006665a62064f658211fd288d8cdcae6df79347a2f038f5cb717"
)
# Updated only when the deterministic layout-only manifest is intentionally
# regenerated from the pinned cc1b... source report.
EXPECTED_LAYOUT_MANIFEST_SHA256 = (
    "d5eb0f71668be726cea84f6b8a2c9e6ea42c551fe72dbaff60d90aa13d6f4b00"
)
PREDICTIONS_NAME = "layout_energy_hybrid_predictions_frozen.json"
REPORT_NAME = "layout_energy_hybrid_report.json"


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_ks(value: str) -> tuple[int, ...]:
    try:
        ks = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise ValueError("--suspect-k must contain integers") from exc
    if not ks or min(ks) < 2 or max(ks) >= 576:
        raise ValueError("--suspect-k values must be in [2,575]")
    return ks


def _load_layout_manifest(
    path: Path,
    *,
    expected_sha256: str,
    data_root: Path,
) -> dict[str, Any]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"layout manifest sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported layout manifest schema")
    if payload.get("kind") != "frozen_real16_hbt_qap_layout_only_manifest":
        raise ValueError("wrong layout manifest kind")
    if payload.get("safe_for_submission") is not False:
        raise ValueError("layout manifest must remain explicitly unsafe")
    export = payload.get("export_contract")
    if not isinstance(export, dict):
        raise ValueError("layout manifest export contract missing")
    if export.get("target_paths_accessed") is not False:
        raise ValueError("layout export unexpectedly accessed targets")
    if export.get("target_metrics_exported") is not False:
        raise ValueError("layout export contains target metrics")
    names = payload.get("source_names")
    sources = payload.get("sources")
    if (
        not isinstance(names, list)
        or not isinstance(sources, list)
        or len(names) != 16
        or len(set(names)) != 16
    ):
        raise ValueError("layout manifest must contain the fixed real16 panel")
    if [record.get("source") for record in sources] != names:
        raise ValueError("layout manifest source order mismatch")
    for record in sources:
        name = record["source"]
        relative = record.get("raw_input")
        if relative != f"train/inputs/{name}":
            raise ValueError(f"unexpected raw input path for {name}")
        input_path = data_root / relative
        if sha256_file(input_path) != record.get("raw_input_sha256"):
            raise ValueError(f"raw input hash mismatch for {name}")
        layouts = record.get("layouts")
        hashes = record.get("layout_sha256")
        if set(layouts or {}) != {"hbt", "qap"} or set(hashes or {}) != {"hbt", "qap"}:
            raise ValueError(f"missing HBT/QAP layout for {name}")
        for label in ("hbt", "qap"):
            layout = validate_layout(layouts[label], count=576)
            if sha256_array(layout) != hashes[label]:
                raise ValueError(f"layout hash mismatch for {name}/{label}")
    payload["manifest_sha256"] = actual_sha256
    return payload


def _proposal_hash(proposals: tuple[SwapProposal, ...]) -> str:
    payload = json.dumps(
        [proposal.to_dict() for proposal in proposals],
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _variant_record(
    layout: np.ndarray,
    *,
    method: str,
    proposal_budget: int,
    proposal_pool_sha256: str,
    selected: SwapProposal | None,
    base_seam: float,
    seam_after: float,
    critic_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "method": method,
        "position_to_slot": validate_layout(layout, count=576).tolist(),
        "position_to_slot_sha256": sha256_array(np.asarray(layout, dtype=np.int32)),
        "proposal_budget": int(proposal_budget),
        "proposal_pool_sha256": proposal_pool_sha256,
        "selected_swap": selected.to_dict() if selected is not None else None,
        "accepted_move": selected is not None,
        "raw_seam_before": float(base_seam),
        "raw_seam_after": float(seam_after),
        "raw_seam_delta": float(seam_after - base_seam),
        "critic_scores": (
            [float(value) for value in critic_scores] if critic_scores is not None else None
        ),
        "target_accessed": False,
    }


def _predict_source(
    record: dict[str, Any],
    input_path: Path,
    *,
    model: torch.nn.Module,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    suspect_ks: tuple[int, ...],
    proposal_budget: int,
    rerank_budget: int,
    score_batch_size: int,
    seam_chunk_size: int,
) -> dict[str, Any]:
    """Target-free predictor.  This signature deliberately has no target argument."""

    started = time.perf_counter()
    raw_image = _read_rgb(input_path)
    raw_tiles = split_tiles_numpy(raw_image)
    seam = raw_border_l1_seam(raw_tiles, strip=2, chunk_size=seam_chunk_size)
    scorer = FrozenCriticScorer(
        model,
        raw_tiles,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    bases: list[dict[str, Any]] = []
    for base_label in ("hbt", "qap"):
        base_layout = validate_layout(record["layouts"][base_label], count=576)
        base_scores = scorer.score(base_layout, batch_size=1)
        error_probabilities = base_scores.error_probabilities[0]
        local_costs = local_seam_costs(base_layout, seam)
        base_seam = seam_objective(base_layout, seam)
        correlation = spearmanr(error_probabilities, local_costs).statistic
        if not np.isfinite(correlation):
            correlation = 0.0
        configurations: list[dict[str, Any]] = []
        for suspect_k in suspect_ks:
            critic_hot = np.argsort(
                -error_probabilities, kind="stable"
            )[:suspect_k].astype(np.int32)
            seam_hot = np.argsort(-local_costs, kind="stable")[:suspect_k].astype(
                np.int32
            )
            critic_pool = top_delta_swaps(
                base_layout,
                seam,
                critic_hot,
                budget=proposal_budget,
            )
            control_pool = top_delta_swaps(
                base_layout,
                seam,
                seam_hot,
                budget=proposal_budget,
            )
            if len(critic_pool) != proposal_budget or len(control_pool) != proposal_budget:
                raise AssertionError("proposal budgets were not exact")
            critic_pool_hash = _proposal_hash(critic_pool)
            control_pool_hash = _proposal_hash(control_pool)

            hybrid_layout, hybrid_swap = seam_select_or_noop(
                base_layout, critic_pool
            )
            control_layout, control_swap = seam_select_or_noop(
                base_layout, control_pool
            )
            reranked_layout, reranked_swap, rerank_scores = critic_rerank_or_noop(
                base_layout,
                critic_pool,
                scorer,
                rerank_budget=rerank_budget,
                batch_size=score_batch_size,
            )
            variants = {
                "critic_heatmap_seam": _variant_record(
                    hybrid_layout,
                    method="critic_heatmap_top_k_then_best_raw_seam_delta",
                    proposal_budget=proposal_budget,
                    proposal_pool_sha256=critic_pool_hash,
                    selected=hybrid_swap,
                    base_seam=base_seam,
                    seam_after=seam_objective(hybrid_layout, seam),
                ),
                "critic_heatmap_energy_rerank": _variant_record(
                    reranked_layout,
                    method="critic_heatmap_top_k_then_raw_seam_shortlist_then_frozen_energy",
                    proposal_budget=proposal_budget,
                    proposal_pool_sha256=critic_pool_hash,
                    selected=reranked_swap,
                    base_seam=base_seam,
                    seam_after=seam_objective(reranked_layout, seam),
                    critic_scores=rerank_scores,
                ),
                "seam_only_control": _variant_record(
                    control_layout,
                    method="classical_local_seam_top_k_then_best_raw_seam_delta",
                    proposal_budget=proposal_budget,
                    proposal_pool_sha256=control_pool_hash,
                    selected=control_swap,
                    base_seam=base_seam,
                    seam_after=seam_objective(control_layout, seam),
                ),
                "no_op_budget_matched": _variant_record(
                    base_layout,
                    method="reuse_and_discard_all_critic_heatmap_proposals",
                    proposal_budget=proposal_budget,
                    proposal_pool_sha256=critic_pool_hash,
                    selected=None,
                    base_seam=base_seam,
                    seam_after=base_seam,
                ),
            }
            budgets = {item["proposal_budget"] for item in variants.values()}
            if budgets != {proposal_budget}:
                raise AssertionError("variant proposal budgets differ")
            configurations.append(
                {
                    "suspect_k": int(suspect_k),
                    "critic_hot_positions": critic_hot.tolist(),
                    "seam_hot_positions": seam_hot.tolist(),
                    "hot_position_overlap": int(
                        len(set(critic_hot.tolist()) & set(seam_hot.tolist()))
                    ),
                    "rerank_budget_plus_noop": int(rerank_budget + 1),
                    "variants": variants,
                }
            )
        bases.append(
            {
                "base": base_label,
                "base_position_to_slot": base_layout.tolist(),
                "base_position_to_slot_sha256": sha256_array(base_layout),
                "base_energy": float(base_scores.energies[0]),
                "base_error_probabilities": [
                    float(value) for value in error_probabilities
                ],
                "target_free_heatmap_vs_raw_seam_spearman": float(correlation),
                "configurations": configurations,
            }
        )
    return {
        "source": record["source"],
        "raw_input": record["raw_input"],
        "raw_input_sha256": record["raw_input_sha256"],
        "bases": bases,
        "prediction_seconds": time.perf_counter() - started,
        "target_accessed": False,
    }


def paired_bootstrap(
    deltas: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, float]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not bool(np.isfinite(values).all()):
        raise ValueError("bootstrap deltas must be a non-empty finite vector")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def _posthoc_local_damage(
    probabilities: np.ndarray,
    base_layout: np.ndarray,
    raw_tiles: np.ndarray,
    target_tiles: np.ndarray,
    suspect_ks: tuple[int, ...],
) -> dict[str, Any]:
    ordered = raw_tiles[validate_layout(base_layout, count=576)]
    damage = np.mean(
        np.abs(ordered.astype(np.float32) - target_tiles.astype(np.float32)),
        axis=(1, 2, 3),
        dtype=np.float32,
    )
    correlation = spearmanr(probabilities, damage).statistic
    if not np.isfinite(correlation):
        correlation = 0.0
    overall = float(damage.mean())
    lifts = {}
    for suspect_k in suspect_ks:
        hot = np.argsort(-probabilities, kind="stable")[:suspect_k]
        lifts[f"k{suspect_k}"] = float(damage[hot].mean() / max(overall, 1e-12))
    return {
        "definition": "per-position raw-tile MAE to clean target; diagnostic only, not tile identity ground truth",
        "critic_probability_vs_pixel_damage_spearman": float(correlation),
        "critic_top_k_pixel_damage_lift": lifts,
    }


def _score_frozen_predictions(
    predictions_path: Path,
    *,
    data_root: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Open targets only after the immutable prediction artifact exists."""

    prediction_sha256_before = sha256_file(predictions_path)
    frozen = json.loads(predictions_path.read_text(encoding="utf-8"))
    if frozen.get("prediction_stage", {}).get("targets_accessed") is not False:
        raise ValueError("prediction artifact does not certify target-free generation")
    suspect_ks = tuple(int(value) for value in frozen["configuration"]["suspect_k"])
    per_source: list[dict[str, Any]] = []
    aggregate_rows: dict[tuple[str, int, str], list[dict[str, float]]] = {}
    for source_record in frozen["sources"]:
        name = source_record["source"]
        input_path = data_root / source_record["raw_input"]
        if sha256_file(input_path) != source_record["raw_input_sha256"]:
            raise ValueError(f"raw input changed before target scoring: {name}")
        raw_tiles = split_tiles_numpy(_read_rgb(input_path))
        # This is the first target-path access in the entire workflow.
        target_path = data_root / "train" / "targets" / name
        target = _read_rgb(target_path)
        target_tiles = split_tiles_numpy(target)
        source_result: dict[str, Any] = {"source": name, "bases": []}
        for base_record in source_record["bases"]:
            base_label = base_record["base"]
            base_layout = validate_layout(
                base_record["base_position_to_slot"], count=576
            )
            if sha256_array(base_layout) != base_record["base_position_to_slot_sha256"]:
                raise ValueError(f"frozen base layout hash mismatch: {name}/{base_label}")
            baseline_metrics = predicted_image_metrics(base_layout, raw_tiles, target)
            probabilities = np.asarray(
                base_record["base_error_probabilities"], dtype=np.float64
            )
            base_result: dict[str, Any] = {
                "base": base_label,
                "baseline_raw_render": baseline_metrics,
                "posthoc_local_damage": _posthoc_local_damage(
                    probabilities,
                    base_layout,
                    raw_tiles,
                    target_tiles,
                    suspect_ks,
                ),
                "configurations": [],
            }
            for configuration in base_record["configurations"]:
                suspect_k = int(configuration["suspect_k"])
                variant_results: dict[str, Any] = {}
                for method, variant in configuration["variants"].items():
                    layout = validate_layout(variant["position_to_slot"], count=576)
                    if sha256_array(layout) != variant["position_to_slot_sha256"]:
                        raise ValueError(
                            f"frozen variant hash mismatch: {name}/{base_label}/k{suspect_k}/{method}"
                        )
                    metrics = predicted_image_metrics(layout, raw_tiles, target)
                    delta = float(
                        metrics["predicted_layout_ssim"]
                        - baseline_metrics["predicted_layout_ssim"]
                    )
                    variant_results[method] = {
                        **metrics,
                        "ssim_delta_vs_base": delta,
                        "accepted_move": bool(variant["accepted_move"]),
                        "selected_swap": variant["selected_swap"],
                        "proposal_budget": int(variant["proposal_budget"]),
                    }
                    aggregate_rows.setdefault((base_label, suspect_k, method), []).append(
                        {
                            "baseline": float(
                                baseline_metrics["predicted_layout_ssim"]
                            ),
                            "candidate": float(metrics["predicted_layout_ssim"]),
                            "delta": delta,
                        }
                    )
                if abs(variant_results["no_op_budget_matched"]["ssim_delta_vs_base"]) > 1e-12:
                    raise AssertionError("no-op changed post-hoc SSIM")
                base_result["configurations"].append(
                    {"suspect_k": suspect_k, "variants": variant_results}
                )
            source_result["bases"].append(base_result)
        per_source.append(source_result)
    prediction_sha256_after = sha256_file(predictions_path)
    if prediction_sha256_after != prediction_sha256_before:
        raise RuntimeError("frozen prediction artifact mutated during target scoring")

    aggregates: list[dict[str, Any]] = []
    lookup: dict[tuple[str, int, str], np.ndarray] = {}
    for index, ((base, suspect_k, method), rows) in enumerate(sorted(aggregate_rows.items())):
        deltas = np.asarray([row["delta"] for row in rows], dtype=np.float64)
        candidates = np.asarray([row["candidate"] for row in rows], dtype=np.float64)
        baselines = np.asarray([row["baseline"] for row in rows], dtype=np.float64)
        lookup[(base, suspect_k, method)] = candidates
        bootstrap = paired_bootstrap(
            deltas,
            seed=seed + 1009 * (index + 1),
            samples=bootstrap_samples,
        )
        aggregates.append(
            {
                "base": base,
                "suspect_k": suspect_k,
                "method": method,
                "source_count": len(rows),
                "mean_baseline_raw_ssim": float(baselines.mean()),
                "mean_candidate_raw_ssim": float(candidates.mean()),
                "ssim_delta_vs_base": bootstrap,
                "source_win_fraction_vs_base": float(np.mean(deltas > 1e-12)),
                "source_tie_fraction_vs_base": float(np.mean(np.abs(deltas) <= 1e-12)),
            }
        )
    actionable: list[dict[str, Any]] = []
    for row in aggregates:
        if row["method"] not in {
            "critic_heatmap_seam",
            "critic_heatmap_energy_rerank",
        }:
            continue
        key = (row["base"], row["suspect_k"], row["method"])
        control_key = (row["base"], row["suspect_k"], "seam_only_control")
        learned_minus_control = lookup[key] - lookup[control_key]
        comparison = paired_bootstrap(
            learned_minus_control,
            seed=seed + 65537 + 313 * len(actionable),
            samples=bootstrap_samples,
        )
        gates = {
            "positive_paired_ci_vs_base": row["ssim_delta_vs_base"]["lower_95"] > 0.0,
            "source_win_fraction_at_least_0.60": row[
                "source_win_fraction_vs_base"
            ]
            >= 0.60,
            "mean_delta_at_least_0.001": row["ssim_delta_vs_base"]["mean"] >= 0.001,
            "positive_paired_ci_vs_equal_budget_seam_control": comparison[
                "lower_95"
            ]
            > 0.0,
        }
        actionable.append(
            {
                "base": row["base"],
                "suspect_k": row["suspect_k"],
                "method": row["method"],
                "learned_minus_equal_budget_seam_control": comparison,
                "gates": gates,
                "gate_passed": bool(all(gates.values())),
            }
        )
    signal = any(record["gate_passed"] for record in actionable)
    return {
        "schema_version": 1,
        "kind": "frozen_failed_layout_energy_hybrid_real16_diagnostic",
        "status": "actionable_signal" if signal else "no_actionable_signal",
        "safe_for_submission": False,
        "actionable_signal": signal,
        "anti_leakage": {
            "predictor_accepts_target": False,
            "predictions_atomically_frozen_before_target_access": True,
            "prediction_sha256_before_target_access": prediction_sha256_before,
            "prediction_sha256_after_target_scoring": prediction_sha256_after,
            "prediction_artifact_unchanged": prediction_sha256_after
            == prediction_sha256_before,
            "target_used_only_for_posthoc_metrics": True,
        },
        "metric_scope": {
            "primary": "paired raw-render SSIM delta on fixed real16",
            "raw_render_reason": "diagnostic inference uses no denoiser; paired layout delta only",
            "local_damage": "diagnostic correlation only; not exact tile-identity ground truth",
        },
        "aggregates": aggregates,
        "actionability_gates": actionable,
        "per_source": per_source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frozen-layouts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256)
    parser.add_argument(
        "--expected-layout-manifest-sha256",
        default=EXPECTED_LAYOUT_MANIFEST_SHA256,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-t4", action="store_true")
    parser.add_argument("--suspect-k", default="8,16,32")
    parser.add_argument("--proposal-budget", type=int, default=96)
    parser.add_argument("--rerank-budget", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=5)
    parser.add_argument("--seam-chunk-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.proposal_budget,
        args.rerank_budget,
        args.score_batch_size,
        args.seam_chunk_size,
        args.bootstrap_samples,
        args.limit,
    ) <= 0:
        raise SystemExit("budgets, sizes, samples, and limit must be positive")
    if args.rerank_budget > args.proposal_budget:
        raise SystemExit("rerank budget cannot exceed proposal budget")
    suspect_ks = _parse_ks(args.suspect_k)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"output directory exists; pass --overwrite: {output_dir}")
        expected_children = {PREDICTIONS_NAME, REPORT_NAME}
        unknown = sorted(
            path.name for path in output_dir.iterdir() if path.name not in expected_children
        )
        if unknown:
            raise SystemExit(
                f"refusing to overwrite output directory with unknown files: {unknown}"
            )
        for name in expected_children:
            (output_dir / name).unlink(missing_ok=True)
    else:
        output_dir.mkdir(parents=True)
    manifest = _load_layout_manifest(
        Path(args.frozen_layouts),
        expected_sha256=args.expected_layout_manifest_sha256,
        data_root=data_root,
    )
    if args.limit > len(manifest["sources"]):
        raise SystemExit("limit exceeds frozen panel")
    device = _resolve_device(args.device)
    if args.require_t4:
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (7, 5):
            raise SystemExit("--require-t4 requires CUDA capability sm_75")
    autocast_dtype = torch.float16 if device.type == "cuda" else None
    model, checkpoint_payload = load_failed_frozen_critic(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        device=device,
    )
    split_audit = checkpoint_payload.get("split_audit")
    if not isinstance(split_audit, dict):
        raise SystemExit("critic checkpoint split audit is missing")
    seen_names: set[str] = set()
    for key in ("train_names", "selection_names", "holdout_names"):
        values = split_audit.get(key)
        if not isinstance(values, list) or len(set(values)) != len(values):
            raise SystemExit(f"critic checkpoint has invalid {key}")
        seen_names.update(str(value) for value in values)
    evaluation_names = {
        str(record["source"]) for record in manifest["sources"][: args.limit]
    }
    evaluation_overlap = sorted(seen_names & evaluation_names)
    if evaluation_overlap:
        raise SystemExit(
            f"real-layout diagnostic overlaps critic train/selection/holdout: {evaluation_overlap}"
        )
    prediction_started = time.perf_counter()
    predicted_sources = []
    for index, source in enumerate(manifest["sources"][: args.limit]):
        predicted = _predict_source(
            source,
            data_root / source["raw_input"],
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            suspect_ks=suspect_ks,
            proposal_budget=args.proposal_budget,
            rerank_budget=args.rerank_budget,
            score_batch_size=args.score_batch_size,
            seam_chunk_size=args.seam_chunk_size,
        )
        predicted_sources.append(predicted)
        print(
            json.dumps(
                {
                    "event": "hybrid_source_predicted_target_free",
                    "index": index + 1,
                    "count": args.limit,
                    "source": source["source"],
                    "seconds": predicted["prediction_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    predictions = {
        "schema_version": 1,
        "kind": "frozen_failed_layout_energy_hybrid_target_free_predictions",
        "safe_for_submission": False,
        "prediction_stage": {
            "raw_only": True,
            "denoiser_used": False,
            "targets_accessed": False,
            "predictor_accepts_target": False,
            "elapsed_seconds": time.perf_counter() - prediction_started,
        },
        "checkpoint": {
            "path": str(Path(args.checkpoint)),
            "sha256": args.expected_checkpoint_sha256,
            "status": checkpoint_payload.get("status"),
            "development_gate_passed": checkpoint_payload.get(
                "development_gate_passed"
            ),
            "safe_for_submission": checkpoint_payload.get("safe_for_submission"),
            "selected_epoch": checkpoint_payload.get("selected_epoch"),
            "model_config": checkpoint_payload.get("model_config"),
            "training_selection_holdout_source_count": len(seen_names),
            "real16_source_overlap_count": len(evaluation_overlap),
            "real16_source_overlap": evaluation_overlap,
        },
        "frozen_layout_manifest": {
            "path": str(Path(args.frozen_layouts)),
            "sha256": manifest["manifest_sha256"],
            "source_report_sha256": manifest["source_report_sha256"],
            "source_report_contract": manifest["source_report_contract"],
        },
        "configuration": {
            "suspect_k": list(suspect_ks),
            "proposal_budget_each_method": args.proposal_budget,
            "rerank_budget": args.rerank_budget,
            "rerank_budget_plus_noop": args.rerank_budget + 1,
            "seam": "raw RGB border L1 strip=2",
            "move_family": "one exact tile swap",
            "bases": ["hbt", "qap"],
            "seed": args.seed,
        },
        "runtime": {
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "cuda_capability": list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None,
            "autocast": "fp16" if autocast_dtype is torch.float16 else "off",
        },
        "sources": predicted_sources,
    }
    predictions_path = output_dir / PREDICTIONS_NAME
    _atomic_json(predictions_path, predictions)
    prediction_sha256 = sha256_file(predictions_path)
    print(
        json.dumps(
            {
                "event": "hybrid_predictions_atomically_frozen",
                "path": str(predictions_path),
                "sha256": prediction_sha256,
                "targets_accessed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    report = _score_frozen_predictions(
        predictions_path,
        data_root=data_root,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report.update(
        {
            "configuration": predictions["configuration"],
            "runtime": predictions["runtime"],
            "checkpoint": predictions["checkpoint"],
            "frozen_layout_manifest": predictions["frozen_layout_manifest"],
            "prediction_artifact": {
                "path": predictions_path.name,
                "sha256": prediction_sha256,
                "bytes": predictions_path.stat().st_size,
            },
            "limitations": [
                "failed v1 critic is frozen and receives no retraining",
                "real16 targets are used only after predictions are frozen",
                "raw-render paired SSIM tests layout actionability without a denoiser",
                "one-swap search cannot repair component-scale displacement",
                "global energy rerank is an ablation because v1 global ranking failed",
                "any positive diagnostic still requires a separately frozen denoised-render gate",
                "K/method selection on assembly_cal requires a new untouched holdout before promotion",
                "no submission artifact is produced",
            ],
        }
    )
    report_path = output_dir / REPORT_NAME
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "event": "layout_energy_hybrid_diagnostic_complete",
                "status": report["status"],
                "safe_for_submission": False,
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
