#!/usr/bin/env python3
"""One-shot frozen-gate evaluation for a selected 5x5 TileNAF candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity

from puzzle_assembly.compatibility import prediction_compatibility
from puzzle_assembly.geometry import inverse_permutation
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics, retrieval_metrics
from puzzle_assembly.qap import directional_qap
from puzzle_assembly.solvers import greedy_row_major
from puzzle_denoise_v2.block5x5 import (
    canonical_name_hash,
    load_protocol,
    sha256_file,
)
from puzzle_denoise_v2.degradation import SyntheticTileDegrader
from puzzle_denoise_v2.metrics import tile_metrics
from puzzle_denoise_v2.model import TileNAFNet
from puzzle_denoise_v2.tiles import merge_tiles_numpy
from puzzle_denoise_v2.training import (
    choose_device,
    load_manifest,
    make_fixed_validation_plan,
    render_fixed_validation,
    resolved_device_fingerprint,
    runtime_versions,
)


CURRENT_SHA256 = "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734"
NLM_PARAMETERS = {
    "h": 7.0,
    "h_color": 7.0,
    "template_window_size": 5,
    "search_window_size": 11,
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_model(path: Path, device: torch.device, *, candidate: bool) -> tuple[TileNAFNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if candidate:
        if checkpoint.get("schema_version") != 1 or checkpoint.get("kind") != "tile_naf_block5x5_finetune":
            raise ValueError("candidate checkpoint schema mismatch")
    elif checkpoint.get("model_name") != "tile-naf":
        raise ValueError("current checkpoint is not TileNAF")
    state = checkpoint.get("ema_state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no EMA state")
    model = TileNAFNet()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def predict(model: torch.nn.Module, tiles: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        batch = (
            torch.from_numpy(np.ascontiguousarray(tiles[start : start + batch_size].transpose(0, 3, 1, 2)))
            .float()
            .div_(255.0)
            .to(device)
        )
        restored = model(batch)
        parts.append(
            restored.detach()
            .cpu()
            .mul(255.0)
            .round()
            .clamp(0, 255)
            .byte()
            .permute(0, 2, 3, 1)
            .numpy()
        )
    return np.concatenate(parts)


def classical_nlm(tiles: np.ndarray) -> np.ndarray:
    restored = np.empty_like(tiles)
    for index, tile in enumerate(tiles):
        bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        value = cv2.fastNlMeansDenoisingColored(
            bgr,
            None,
            NLM_PARAMETERS["h"],
            NLM_PARAMETERS["h_color"],
            NLM_PARAMETERS["template_window_size"],
            NLM_PARAMETERS["search_window_size"],
        )
        restored[index] = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    return restored


def source_metrics(prediction: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    result = tile_metrics(prediction, clean)
    result["ordered_image_ssim"] = float(
        structural_similarity(
            merge_tiles_numpy(clean),
            merge_tiles_numpy(prediction),
            channel_axis=2,
            data_range=255,
        )
    )
    return result


def evaluate_methods(
    methods: dict[str, np.ndarray],
    clean: np.ndarray,
    names: list[str],
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    if len(clean) != len(names) * 576:
        raise ValueError("gate arrays must contain 576 tiles per source")
    per_source: list[dict] = []
    for source_index, name in enumerate(names):
        start = source_index * 576
        record: dict[str, object] = {"source": name, "methods": {}}
        for method, prediction in methods.items():
            record["methods"][method] = source_metrics(
                prediction[start : start + 576], clean[start : start + 576]
            )
        per_source.append(record)
    macro: dict[str, dict[str, float]] = {}
    for method in methods:
        keys = per_source[0]["methods"][method]
        macro[method] = {
            key: float(
                np.mean([float(record["methods"][method][key]) for record in per_source])
            )
            for key in keys
        }
    return macro, per_source


def paired_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict[str, float | int]:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if delta.ndim != 1 or len(delta) < 2:
        raise ValueError("bootstrap needs at least two paired sources")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 512):
        stop = min(start + 512, resamples)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[indices].mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "resamples": resamples,
        "sources": len(delta),
        "seed": seed,
    }


def balanced_macro(panel_reports: dict[str, dict]) -> dict[str, dict[str, float]]:
    methods = set.intersection(*(set(report["macro"]) for report in panel_reports.values()))
    result: dict[str, dict[str, float]] = {}
    for method in sorted(methods):
        keys = set.intersection(
            *(set(report["macro"][method]) for report in panel_reports.values())
        )
        result[method] = {
            key: float(
                np.mean([report["macro"][method][key] for report in panel_reports.values()])
            )
            for key in sorted(keys)
        }
    return result


def shuffled_source(
    tiles: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    slot_to_target = np.random.default_rng(seed).permutation(576).astype(np.int32)
    return tiles[slot_to_target], slot_to_target


def retrieval_proxy(
    panel_predictions: dict[str, dict[str, np.ndarray]],
    names: list[str],
    *,
    source_count: int,
    master_seed: int,
) -> dict:
    selected_names = names[:source_count]
    report: dict[str, object] = {"source_names": selected_names, "panels": {}}
    for panel_index, (panel_name, methods) in enumerate(panel_predictions.items()):
        panel_sources: list[dict] = []
        for source_index, name in enumerate(selected_names):
            start = source_index * 576
            source_record: dict[str, object] = {"source": name, "methods": {}}
            seed = master_seed + panel_index * 1000 + source_index
            slot_to_target = np.random.default_rng(seed).permutation(576).astype(np.int32)
            for method in ("raw_copy", "current_tilenaf", "candidate_block5x5"):
                slot_tiles = methods[method][start : start + 576][slot_to_target]
                score = prediction_compatibility(slot_tiles, prefix=method, chunk_size=64)
                source_record["methods"][method] = retrieval_metrics(score, slot_to_target)[
                    "combined"
                ]
            panel_sources.append(source_record)
        macro: dict[str, dict[str, float]] = {}
        for method in ("raw_copy", "current_tilenaf", "candidate_block5x5"):
            keys = panel_sources[0]["methods"][method]
            macro[method] = {
                key: float(
                    np.mean(
                        [float(record["methods"][method][key]) for record in panel_sources]
                    )
                )
                for key in keys
            }
        report["panels"][panel_name] = {"macro": macro, "per_source": panel_sources}
    balanced: dict[str, dict[str, float]] = {}
    for method in ("raw_copy", "current_tilenaf", "candidate_block5x5"):
        keys = next(iter(report["panels"].values()))["macro"][method]
        balanced[method] = {
            key: float(
                np.mean(
                    [panel["macro"][method][key] for panel in report["panels"].values()]
                )
            )
            for key in keys
        }
    report["balanced"] = balanced
    return report


def qap_proxy(
    panel_predictions: dict[str, dict[str, np.ndarray]],
    clean: np.ndarray,
    names: list[str],
    *,
    source_count: int,
    master_seed: int,
    qap_config: dict,
) -> dict:
    selected_names = names[:source_count]
    report: dict[str, object] = {"source_names": selected_names, "panels": {}}
    for panel_index, (panel_name, methods) in enumerate(panel_predictions.items()):
        panel_sources: list[dict] = []
        for source_index, name in enumerate(selected_names):
            start = source_index * 576
            clean_image = merge_tiles_numpy(clean[start : start + 576])
            seed = master_seed + panel_index * 1000 + source_index
            slot_to_target = np.random.default_rng(seed).permutation(576).astype(np.int32)
            source_record: dict[str, object] = {"source": name, "methods": {}}
            for method in ("raw_copy", "current_tilenaf", "candidate_block5x5"):
                slot_tiles = methods[method][start : start + 576][slot_to_target]
                score = prediction_compatibility(slot_tiles, prefix=method, chunk_size=64)
                initial = greedy_row_major(score, boundary_weight=0.2)
                result = directional_qap(
                    score,
                    initial=initial,
                    iterations=int(qap_config["iterations"]),
                    restarts=int(qap_config["restarts"]),
                    seed=seed,
                    boundary_weight=float(qap_config["boundary_weight"]),
                    refine_swaps=int(qap_config["refine_swaps"]),
                )
                source_record["methods"][method] = {
                    **layout_metrics(result.position_to_slot, slot_to_target),
                    **predicted_image_metrics(result.position_to_slot, slot_tiles, clean_image),
                    "objective": result.objective,
                    "iterations": result.iterations,
                    "restart": result.restart,
                    "converged": result.converged,
                }
            panel_sources.append(source_record)
        macro: dict[str, dict[str, float]] = {}
        for method in ("raw_copy", "current_tilenaf", "candidate_block5x5"):
            numeric_keys = [
                key
                for key, value in panel_sources[0]["methods"][method].items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            macro[method] = {
                key: float(
                    np.mean(
                        [float(record["methods"][method][key]) for record in panel_sources]
                    )
                )
                for key in numeric_keys
            }
        report["panels"][panel_name] = {"macro": macro, "per_source": panel_sources}
    balanced: dict[str, dict[str, float]] = {}
    for method in ("raw_copy", "current_tilenaf", "candidate_block5x5"):
        keys = next(iter(report["panels"].values()))["macro"][method]
        balanced[method] = {
            key: float(
                np.mean(
                    [panel["macro"][method][key] for panel in report["panels"].values()]
                )
            )
            for key in keys
        }
    report["balanced"] = balanced
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--current-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()

    started = time.perf_counter()
    torch.set_num_threads(args.torch_threads)
    cv2.setNumThreads(args.torch_threads)
    protocol_path = Path(args.protocol)
    protocol = load_protocol(protocol_path)
    selection_path = Path(args.selection)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("decision") != "open_frozen_gate":
        raise ValueError("selection did not authorize frozen gate access")
    candidate_path = Path(args.candidate_checkpoint)
    candidate_sha256 = sha256_file(candidate_path)
    if candidate_sha256 != selection.get("selected_checkpoint_sha256"):
        raise ValueError("candidate checkpoint does not match development selection")
    if sha256_file(protocol_path) != selection.get("protocol_sha256"):
        raise ValueError("selection protocol hash mismatch")

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    if sha256_file(manifest_path) != protocol["inputs"]["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")
    gate_names = list(protocol["source_partitions"]["frozen_gate"]["names"])
    if canonical_name_hash(gate_names) != protocol["source_partitions"]["frozen_gate"][
        "names_sha256"
    ]:
        raise ValueError("frozen gate name hash mismatch")
    if not set(gate_names) <= set(manifest["splits"]["val"]):
        raise ValueError("frozen gate is outside held-out validation sources")
    if set(gate_names) & set(protocol["source_partitions"]["development"]["names"]):
        raise ValueError("frozen gate overlaps development")

    current_path = Path(args.current_checkpoint)
    if sha256_file(current_path) != CURRENT_SHA256:
        raise ValueError("current checkpoint SHA256 mismatch")
    device = choose_device(args.device)
    current_model, current_metadata = load_model(current_path, device, candidate=False)
    candidate_model, candidate_metadata = load_model(candidate_path, device, candidate=True)
    if candidate_metadata.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("candidate protocol mismatch")
    if candidate_metadata.get("development_source_names_sha256") != protocol[
        "source_partitions"
    ]["development"]["names_sha256"]:
        raise ValueError("candidate development source mismatch")

    target_dir = Path(args.data_root) / "train" / "targets"
    degrader = SyntheticTileDegrader()
    plan = make_fixed_validation_plan(
        target_dir,
        gate_names,
        576,
        int(protocol["panel_seeds"]["frozen_gate"]),
        degrader,
    )
    corruptions = {
        "primary_kornia": render_fixed_validation(
            plan, degrader, args.batch_size, codec="kornia"
        ),
        "independent_libjpeg": render_fixed_validation(
            plan, degrader, min(args.batch_size, 512), codec="pillow"
        ),
    }

    panel_reports: dict[str, dict] = {}
    panel_predictions: dict[str, dict[str, np.ndarray]] = {}
    for panel_name, corrupt in corruptions.items():
        current_prediction = predict(current_model, corrupt, device, args.batch_size)
        candidate_prediction = predict(candidate_model, corrupt, device, args.batch_size)
        nlm_prediction = classical_nlm(corrupt)
        methods = {
            "raw_copy": corrupt,
            "classical_nlm": nlm_prediction,
            "current_tilenaf": current_prediction,
            "candidate_block5x5": candidate_prediction,
        }
        macro, per_source = evaluate_methods(methods, plan.clean, gate_names)
        panel_reports[panel_name] = {"macro": macro, "per_source": per_source}
        panel_predictions[panel_name] = methods
        print(
            json.dumps(
                {"event": "block5x5_gate_panel", "panel": panel_name, "macro": macro},
                sort_keys=True,
            ),
            flush=True,
        )

    balanced = balanced_macro(panel_reports)
    bootstrap: dict[str, dict] = {}
    bootstrap_seed = int(protocol["panel_seeds"]["bootstrap"])
    resamples = int(protocol["evaluation"]["bootstrap_resamples"])
    for panel_index, (panel_name, report) in enumerate(panel_reports.items()):
        for metric in ("tile_ssim", "ordered_image_ssim", "boundary_mae", "gradient_mae"):
            candidate_values = np.asarray(
                [record["methods"]["candidate_block5x5"][metric] for record in report["per_source"]]
            )
            current_values = np.asarray(
                [record["methods"]["current_tilenaf"][metric] for record in report["per_source"]]
            )
            bootstrap[f"{panel_name}.{metric}"] = paired_bootstrap(
                candidate_values,
                current_values,
                seed=bootstrap_seed + panel_index * 100 + len(bootstrap),
                resamples=resamples,
            )
    for metric in ("tile_ssim", "ordered_image_ssim"):
        candidate_values = []
        current_values = []
        for report in panel_reports.values():
            candidate_values.append(
                np.asarray(
                    [record["methods"]["candidate_block5x5"][metric] for record in report["per_source"]]
                )
            )
            current_values.append(
                np.asarray(
                    [record["methods"]["current_tilenaf"][metric] for record in report["per_source"]]
                )
            )
        bootstrap[f"balanced.{metric}"] = paired_bootstrap(
            np.mean(candidate_values, axis=0),
            np.mean(current_values, axis=0),
            seed=bootstrap_seed + (900 if metric == "tile_ssim" else 901),
            resamples=resamples,
        )

    retrieval = retrieval_proxy(
        panel_predictions,
        gate_names,
        source_count=int(protocol["evaluation"]["retrieval_sources"]),
        master_seed=int(protocol["panel_seeds"]["retrieval"]),
    )
    qap = qap_proxy(
        panel_predictions,
        plan.clean,
        gate_names,
        source_count=int(protocol["evaluation"]["qap_sources"]),
        master_seed=int(protocol["panel_seeds"]["qap"]),
        qap_config=protocol["evaluation"]["qap"],
    )

    current = balanced["current_tilenaf"]
    candidate = balanced["candidate_block5x5"]
    deltas = {key: candidate[key] - current[key] for key in candidate if key in current}
    retrieval_delta = (
        retrieval["balanced"]["candidate_block5x5"]["recall_at_1"]
        - retrieval["balanced"]["current_tilenaf"]["recall_at_1"]
    )
    qap_delta = (
        qap["balanced"]["candidate_block5x5"]["predicted_layout_ssim"]
        - qap["balanced"]["current_tilenaf"]["predicted_layout_ssim"]
    )
    promotion = protocol["promotion_gate"]
    checks = {
        "balanced_tile_ssim_delta_at_least_0_003": deltas["tile_ssim"]
        >= float(promotion["balanced_tile_ssim_delta_min"]),
        "balanced_ordered_image_ssim_delta_at_least_0_005": deltas[
            "ordered_image_ssim"
        ]
        >= float(promotion["balanced_ordered_image_ssim_delta_min"]),
        "both_panel_ordered_image_ssim_delta_positive": all(
            report["macro"]["candidate_block5x5"]["ordered_image_ssim"]
            > report["macro"]["current_tilenaf"]["ordered_image_ssim"]
            for report in panel_reports.values()
        ),
        "boundary_mae_nonworse": candidate["boundary_mae"] <= current["boundary_mae"],
        "gradient_mae_nonworse": candidate["gradient_mae"] <= current["gradient_mae"],
        "bootstrap_lower_tile_ssim_positive": bootstrap["balanced.tile_ssim"]["lower"] > 0.0,
        "bootstrap_lower_ordered_image_ssim_positive": bootstrap[
            "balanced.ordered_image_ssim"
        ]["lower"]
        > 0.0,
        "pbc_recall_at_1_delta_at_least_0_01": retrieval_delta
        >= float(promotion["pbc_recall_at_1_delta_min"]),
    }
    qap_diagnostic = {
        "delta": qap_delta,
        "threshold": float(promotion["qap_predicted_layout_ssim_delta_min"]),
        "passes": qap_delta >= float(promotion["qap_predicted_layout_ssim_delta_min"]),
        "source_count": int(protocol["evaluation"]["qap_sources"]),
        "role": "diagnostic_not_boolean_because_source_count_below_4",
    }
    verdict = "promote" if all(checks.values()) else "reject_keep_current"
    payload = {
        "schema_version": 1,
        "kind": "denoise_block5x5_frozen_gate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "safe_for_inference": verdict == "promote",
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "selection": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "current_checkpoint": str(current_path.resolve()),
        "current_checkpoint_sha256": sha256_file(current_path),
        "candidate_checkpoint": str(candidate_path.resolve()),
        "candidate_checkpoint_sha256": candidate_sha256,
        "candidate_variant": candidate_metadata["variant"],
        "candidate_best_step": candidate_metadata["best_step"],
        "frozen_gate_source_names": gate_names,
        "frozen_gate_source_names_sha256": canonical_name_hash(gate_names),
        "frozen_gate_accessed_once_by_this_job": True,
        "panels": panel_reports,
        "balanced_macro": balanced,
        "candidate_minus_current": deltas,
        "bootstrap": bootstrap,
        "retrieval_proxy": retrieval,
        "retrieval_recall_at_1_delta": retrieval_delta,
        "qap_proxy": qap,
        "qap_diagnostic": qap_diagnostic,
        "checks": checks,
        "runtime": {
            "versions": runtime_versions(),
            "device": resolved_device_fingerprint(device),
            "python": platform.python_version(),
            "torch_threads": args.torch_threads,
            "opencv_threads": cv2.getNumThreads(),
            "seconds": time.perf_counter() - started,
        },
        "baseline_contract": {
            "raw_copy": "exact corrupt uint8 tiles",
            "classical_nlm": {"kind": "cv2.fastNlMeansDenoisingColored_per_tile", **NLM_PARAMETERS},
            "current_tilenaf": "selected synthetic-50k EMA",
        },
        "anti_leakage": protocol["anti_leakage"],
    }
    output = Path(args.output)
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "event": "block5x5_frozen_gate_complete",
                "verdict": verdict,
                "candidate_minus_current": deltas,
                "retrieval_recall_at_1_delta": retrieval_delta,
                "qap_diagnostic": qap_diagnostic,
                "checks": checks,
                "output": str(output),
                "output_sha256": sha256_file(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
