#!/usr/bin/env python3
"""Leakage-safe QAP scorer/axis-refinement upgrade matrix.

The exact stage selects one input-only configuration on two known-permutation
corruption engines.  Only then are all real16 input layouts generated, hashed,
and persisted; real targets are opened in a separate scoring phase.  The main
question is whether pure HBT or a lighter C1/HBT fusion is a better QAP energy
than the promoted L1w4 fusion.  Whole-row/column refinement is included as a
bounded secondary gate and always retains the promoted baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image

from puzzle_assembly.axis_refine import alternating_axis_refine
from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.learned import learned_compatibility, load_embedding_checkpoint
from puzzle_assembly.metrics import layout_metrics, predicted_image_metrics
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import split_tiles_numpy


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
DEFAULT_EMBEDDING = (
    "runs/assembly_v1/kaggle/edge2vec_gradient_gpu/"
    "hbt_d320_denoised_rgb_sobel.pt"
)
DEFAULT_MANIFEST = "configs/denoise_splits_seed20260710.json"
DEFAULT_QUARANTINE = "configs/denoise_validation_quarantine_v1.json"
BASELINE_LABEL = "qap_w4_b0.05_i25"
EXPECTED_REAL16_BASELINE = 0.18281991502795386


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument("--embedding-checkpoint", default=DEFAULT_EMBEDDING)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument(
        "--exact-panels", default="primary_kornia,independent_libjpeg"
    )
    parser.add_argument("--exact-sources", type=int, default=8)
    parser.add_argument("--real-sources", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--exact-select-count", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen-layouts", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _layout_sha256(layout: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(layout, dtype=np.int32).tobytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _filename_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def _c1_fusion(
    bank: dict[str, CompatibilityMatrices], prefix: str
) -> CompatibilityMatrices:
    names = [
        name
        for name in sorted(bank)
        if name.startswith(prefix + "_") and not name.endswith("_c2")
    ]
    return fuse_ranked_scores(bank, names=names, name=f"{prefix}_C1_equal_rank_fusion")


def _score_bank(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    *,
    embedding_model: object,
    device: object,
    chunk_size: int,
) -> tuple[dict[str, CompatibilityMatrices], dict[str, str]]:
    bank = build_classical_score_bank(raw_tiles, prefix="raw", chunk_size=chunk_size)
    bank.update(
        build_classical_score_bank(
            denoised_tiles, prefix="denoised", chunk_size=chunk_size
        )
    )
    denoised_c1 = _c1_fusion(bank, "denoised")
    bank[denoised_c1.name] = denoised_c1
    hbt, _ = learned_compatibility(
        embedding_model,
        denoised_tiles,
        device=device,
        name="denoised_hbt_l1",
    )
    bank[hbt.name] = hbt
    aliases = {"l1": hbt.name}
    for weight in (1, 2, 4):
        label = f"w{weight}"
        score = fuse_ranked_scores(
            bank,
            names=[denoised_c1.name, hbt.name],
            weights={hbt.name: float(weight)},
            name=f"denoised_C1_HBTw{weight}_rank_fusion",
        )
        bank[score.name] = score
        aliases[label] = score.name
    return bank, aliases


def _qap_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for score_alias in ("l1", "w1", "w2", "w4"):
        for boundary_weight in (0.0, 0.05):
            configs.append(
                {
                    "label": f"qap_{score_alias}_b{boundary_weight:g}_i25",
                    "kind": "qap",
                    "score_alias": score_alias,
                    "iterations": 25,
                    "restarts": 2,
                    "boundary_weight": boundary_weight,
                }
            )
    for score_alias in ("l1", "w2", "w4"):
        configs.append(
            {
                "label": f"qap_{score_alias}_b0.05_i12",
                "kind": "qap",
                "score_alias": score_alias,
                "iterations": 12,
                "restarts": 2,
                "boundary_weight": 0.05,
            }
        )
    if BASELINE_LABEL not in {config["label"] for config in configs}:
        raise RuntimeError("promoted baseline is missing from QAP matrix")
    return configs


def _axis_configs() -> list[dict[str, Any]]:
    configs = []
    for axis_order in (("row", "column"), ("column", "row")):
        suffix = "rc" if axis_order[0] == "row" else "cr"
        for aggregation, fraction, rank_normalize, guard in (
            ("mean", 1.0, True, 1.0),
            ("mean", 1.0, False, 1.0),
            ("best_mean", 0.5, True, 1.02),
            ("best_mean", 1.0 / 3.0, True, 1.02),
        ):
            mode = (
                "mean_rank"
                if aggregation == "mean" and rank_normalize
                else "mean_raw"
                if aggregation == "mean"
                else f"best{int(round(fraction * 100))}_rank"
            )
            configs.append(
                {
                    "label": f"axis_{mode}_{suffix}",
                    "kind": "axis",
                    "base": BASELINE_LABEL,
                    "cycles": 1,
                    "aggregation": aggregation,
                    "best_fraction": fraction,
                    "rank_normalize": rank_normalize,
                    "reciprocal_weight": 0.35,
                    "boundary_weight": 0.05,
                    "random_restarts": 2,
                    "local_passes": 4,
                    "seam_guard_ratio": guard,
                    "axis_order": axis_order,
                }
            )
    return configs


def experiment_configs() -> list[dict[str, Any]]:
    configs = [*_qap_configs(), *_axis_configs()]
    labels = [config["label"] for config in configs]
    if len(labels) != len(set(labels)):
        raise RuntimeError("duplicate experiment labels")
    return configs


def _predict_layouts(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    *,
    source_name: str,
    embedding_model: object,
    device: object,
    chunk_size: int,
    configs: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], float]:
    started = time.perf_counter()
    bank, aliases = _score_bank(
        raw_tiles,
        denoised_tiles,
        embedding_model=embedding_model,
        device=device,
        chunk_size=chunk_size,
    )
    seed_result = soft_cycle_component_solver(
        bank[aliases["l1"]],
        top_k=8,
        keep_per_tile=1,
        proposal_keep_fraction=0.5,
        loop_weight=1.0,
        reciprocal_weight=0.35,
    )
    layouts: dict[str, np.ndarray] = {
        "softcycle_l1_k8": seed_result.position_to_slot.copy()
    }
    diagnostics: dict[str, dict[str, Any]] = {}
    # Keep the QAP random stream identical to the promoted submission builder.
    # This is required for the baseline reproduction guard below and also makes
    # the scorer/config comparisons paired rather than confounded by restarts.
    qap_seed = _filename_seed(source_name) + 7001
    auxiliary_seed = qap_seed + seed
    config_by_label = {config["label"]: config for config in configs}
    for config in (config for config in configs if config["kind"] == "qap"):
        score = bank[aliases[config["score_alias"]]]
        qap = directional_qap(
            score,
            initial=seed_result.position_to_slot,
            iterations=int(config["iterations"]),
            restarts=int(config["restarts"]),
            seed=qap_seed,
            boundary_weight=float(config["boundary_weight"]),
            initial_weight=0.75,
            noisy_components=3,
            noise_scale=1.0,
            refine_swaps=8,
            refine_weak_cells=32,
        )
        layouts[config["label"]] = qap.position_to_slot.copy()
        diagnostics[config["label"]] = {
            "objective": qap.objective,
            "relaxed_objective": qap.relaxed_objective,
            "restart": qap.restart,
            "iterations": qap.iterations,
            "converged": qap.converged,
            "layout_sha256": _layout_sha256(qap.position_to_slot),
        }
    for index, config in enumerate(config for config in configs if config["kind"] == "axis"):
        base_label = str(config["base"])
        if base_label not in layouts or base_label not in config_by_label:
            raise RuntimeError(f"axis base is unavailable: {base_label}")
        refined = alternating_axis_refine(
            layouts[base_label],
            bank[aliases["w4"]],
            cycles=int(config["cycles"]),
            aggregation=str(config["aggregation"]),
            best_fraction=float(config["best_fraction"]),
            rank_normalize=bool(config["rank_normalize"]),
            reciprocal_weight=float(config["reciprocal_weight"]),
            boundary_weight=float(config["boundary_weight"]),
            random_restarts=int(config["random_restarts"]),
            local_passes=int(config["local_passes"]),
            seam_guard_ratio=float(config["seam_guard_ratio"]),
            axis_order=tuple(config["axis_order"]),
            seed=auxiliary_seed + 104729 * (index + 1),
        )
        layouts[config["label"]] = refined.position_to_slot.copy()
        diagnostics[config["label"]] = {
            "objective_before": refined.objective_before,
            "objective_after": refined.objective_after,
            "accepted_steps": refined.accepted_steps,
            "row_orders": refined.row_orders,
            "column_orders": refined.column_orders,
            "layout_sha256": _layout_sha256(refined.position_to_slot),
        }
    return layouts, diagnostics, time.perf_counter() - started


def _mean_records(records: list[dict[str, Any]]) -> dict[str, float]:
    numeric = sorted(
        key
        for key in records[0]
        if all(isinstance(record.get(key), (int, float)) for record in records)
    )
    return {key: float(np.mean([float(record[key]) for record in records])) for key in numeric}


def _bootstrap_ci(values: np.ndarray, resamples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    samples = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def _raw_complexity(image: np.ndarray) -> float:
    values = image.astype(np.float32) / 255.0
    vertical = np.abs(values[1:] - values[:-1]).mean()
    horizontal = np.abs(values[:, 1:] - values[:, :-1]).mean()
    return float(0.5 * (vertical + horizontal))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    frozen_path = Path(args.frozen_layouts)
    if not args.overwrite and (output.exists() or frozen_path.exists()):
        raise SystemExit("output exists; pass --overwrite")
    if min(args.exact_sources, args.real_sources, args.exact_select_count) <= 0:
        raise SystemExit("source/select counts must be positive")
    panels = [value.strip() for value in args.exact_panels.split(",") if value.strip()]
    if set(panels) != {"primary_kornia", "independent_libjpeg"}:
        raise SystemExit("exact panels must contain primary_kornia and independent_libjpeg")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))

    data_root = Path(args.data_root)
    exact_names = source_names_for_split(
        "edge_development",
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )[: args.exact_sources]
    real_names = source_names_for_split(
        "assembly_cal",
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )[: args.real_sources]
    if len(exact_names) != args.exact_sources or len(real_names) != args.real_sources:
        raise SystemExit("requested source count exceeds split")
    overlap = sorted(set(exact_names) & set(real_names))
    if overlap:
        raise SystemExit(f"exact and real splits overlap: {overlap[:5]}")

    restorer, device, denoiser_metadata = load_restorer(
        args.denoiser, device=args.device, state="ema"
    )
    embedding, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    for model in (restorer, embedding):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    configs = experiment_configs()
    config_sha256 = _canonical_sha256(configs)
    exact_records: list[dict[str, Any]] = []
    exact_started = time.perf_counter()
    for panel_name in panels:
        for source_index, name in enumerate(exact_names):
            target = _read_rgb(data_root / "train" / "targets" / name)
            panel_seed = per_source_seed(
                args.seed, f"upgrade-matrix-{panel_name}", name, 0
            )
            panel = make_exact_panel(target, panel=panel_name, seed=panel_seed)
            denoised = restore_tiles_uint8(
                restorer, panel.slot_tiles, device, batch_size=args.batch_size
            )
            layouts, diagnostics, seconds = _predict_layouts(
                panel.slot_tiles,
                denoised,
                source_name=name,
                embedding_model=embedding,
                device=device,
                chunk_size=args.chunk_size,
                configs=configs,
                seed=args.seed,
            )
            for label, layout in layouts.items():
                exact_records.append(
                    {
                        "panel": panel_name,
                        "source": name,
                        "label": label,
                        **layout_metrics(layout, panel.slot_to_target),
                        **predicted_image_metrics(layout, denoised, target),
                        "layout_sha256": _layout_sha256(layout),
                        "solver": diagnostics.get(label),
                    }
                )
            print(
                json.dumps(
                    {
                        "event": "upgrade_exact_source",
                        "panel": panel_name,
                        "index": source_index + 1,
                        "count": len(exact_names),
                        "source": name,
                        "candidate_count": len(layouts),
                        "seconds": seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    labels = sorted({record["label"] for record in exact_records})
    exact_aggregate = {}
    for label in labels:
        records = [record for record in exact_records if record["label"] == label]
        exact_aggregate[label] = _mean_records(records)
    ranked = sorted(
        labels,
        key=lambda label: (
            -exact_aggregate[label]["predicted_layout_ssim"],
            -exact_aggregate[label]["combined_adjacency"],
            exact_aggregate[label]["mean_manhattan"],
            label,
        ),
    )
    selected = ranked[: args.exact_select_count]
    if BASELINE_LABEL not in selected:
        selected.append(BASELINE_LABEL)
    selected = list(dict.fromkeys(selected))
    promoted_label = selected[0]
    exact_selection = {
        "selection_metric": "mean exact-panel denoised-render SSIM, adjacency/manhattan tie-break",
        "ranked_labels": ranked,
        "selected_for_real": selected,
        "promoted_label_before_real_targets": promoted_label,
        "configs_sha256": config_sha256,
    }

    # Phase A: generate only the exact-selected real layouts without opening
    # any real target.  Running every exact candidate on real16 would both
    # waste QAP compute and turn the calibration split into another tuning
    # matrix.  Axis candidates require their promoted QAP base to be present.
    config_by_label = {config["label"]: config for config in configs}
    real_config_labels = [
        label for label in selected if label != "softcycle_l1_k8"
    ]
    if BASELINE_LABEL not in real_config_labels:
        real_config_labels.append(BASELINE_LABEL)
    real_configs = [config_by_label[label] for label in real_config_labels]
    if any(config["kind"] == "axis" for config in real_configs):
        baseline_config = config_by_label[BASELINE_LABEL]
        if BASELINE_LABEL not in {config["label"] for config in real_configs}:
            real_configs.insert(0, baseline_config)

    frozen_layouts: list[np.ndarray] = []
    frozen_hashes: list[list[str]] = []
    real_input_records: list[dict[str, Any]] = []
    candidate_labels = list(selected)
    if BASELINE_LABEL not in candidate_labels:
        candidate_labels.append(BASELINE_LABEL)
    if len(candidate_labels) != len(set(candidate_labels)):
        raise RuntimeError("real candidate labels are not unique")
    for index, name in enumerate(real_names):
        input_image = _read_rgb(data_root / "train" / "inputs" / name)
        raw_tiles = split_tiles_numpy(input_image)
        denoised = restore_tiles_uint8(
            restorer, raw_tiles, device, batch_size=args.batch_size
        )
        layouts, diagnostics, seconds = _predict_layouts(
            raw_tiles,
            denoised,
            source_name=name,
            embedding_model=embedding,
            device=device,
            chunk_size=args.chunk_size,
            configs=real_configs,
            seed=args.seed,
        )
        ordered = np.stack([layouts[label] for label in candidate_labels])
        hashes = [_layout_sha256(layout) for layout in ordered]
        frozen_layouts.append(ordered)
        frozen_hashes.append(hashes)
        real_input_records.append(
            {
                "source": name,
                "input_sha256": hashlib.sha256(input_image.tobytes()).hexdigest(),
                "raw_complexity": _raw_complexity(input_image),
                "layout_hashes": dict(zip(candidate_labels, hashes, strict=True)),
                "solver_diagnostics": diagnostics,
                "seconds": seconds,
                "denoised_tiles": denoised,
            }
        )
        print(
            json.dumps(
                {
                    "event": "upgrade_real_layout_frozen",
                    "index": index + 1,
                    "count": len(real_names),
                    "source": name,
                    "candidate_count": len(candidate_labels),
                    "seconds": seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    frozen_array = np.stack(frozen_layouts).astype(np.int32)
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        frozen_path,
        source_names=np.asarray(real_names),
        candidate_labels=np.asarray(candidate_labels),
        position_to_slot=frozen_array,
        layout_sha256=np.asarray(frozen_hashes),
        configs_json=np.asarray(json.dumps(configs, sort_keys=True)),
        exact_selection_json=np.asarray(json.dumps(exact_selection, sort_keys=True)),
    )
    frozen_sha256 = _sha256(frozen_path)

    # Phase B: targets are opened only after the complete input-only artifact exists.
    real_records: list[dict[str, Any]] = []
    for source_index, record in enumerate(real_input_records):
        name = str(record["source"])
        target = _read_rgb(data_root / "train" / "targets" / name)
        denoised = record.pop("denoised_tiles")
        for candidate_index, label in enumerate(candidate_labels):
            metrics = predicted_image_metrics(
                frozen_array[source_index, candidate_index], denoised, target
            )
            real_records.append(
                {
                    "source": name,
                    "label": label,
                    "raw_complexity": record["raw_complexity"],
                    "layout_sha256": frozen_hashes[source_index][candidate_index],
                    **metrics,
                }
            )

    real_aggregate = {}
    for label in candidate_labels:
        records = [record for record in real_records if record["label"] == label]
        real_aggregate[label] = _mean_records(records)
    baseline_values = np.asarray(
        [
            record["predicted_layout_ssim"]
            for record in real_records
            if record["label"] == BASELINE_LABEL
        ],
        dtype=np.float64,
    )
    if abs(float(baseline_values.mean()) - EXPECTED_REAL16_BASELINE) > 1e-9:
        raise RuntimeError(
            "promoted baseline reproduction failed: "
            f"expected {EXPECTED_REAL16_BASELINE}, got {baseline_values.mean()}"
        )
    paired = {}
    source_order = list(real_names)
    for label in candidate_labels:
        values_by_source = {
            record["source"]: record["predicted_layout_ssim"]
            for record in real_records
            if record["label"] == label
        }
        values = np.asarray([values_by_source[name] for name in source_order])
        delta = values - baseline_values
        paired[label] = {
            "mean_ssim": float(values.mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "wins": int(np.sum(delta > 0)),
            "ties": int(np.sum(delta == 0)),
            "losses": int(np.sum(delta < 0)),
            "bootstrap_95_ci": _bootstrap_ci(
                delta, args.bootstrap_resamples, args.seed + 31
            ),
        }
    promoted = paired[promoted_label]
    all_values = np.stack(
        [
            [
                next(
                    record["predicted_layout_ssim"]
                    for record in real_records
                    if record["source"] == name and record["label"] == label
                )
                for label in candidate_labels
            ]
            for name in source_order
        ]
    )
    oracle = np.max(all_values, axis=1)
    report = {
        "schema_version": 1,
        "kind": "assembly_upgrade_qap_axis_matrix",
        "status": (
            "promotion_gate_passed"
            if promoted["mean_delta"] >= 0.005
            and promoted["wins"] >= 10
            and promoted["bootstrap_95_ci"][0] > 0
            else "promotion_gate_failed"
        ),
        "args": vars(args),
        "anti_leakage": {
            "exact_real_overlap": overlap,
            "predictor_accepts_target": False,
            "all_real_layouts_frozen_before_targets": True,
            "frozen_layouts": str(frozen_path),
            "frozen_layouts_sha256": frozen_sha256,
            "target_opened_after_layouts_frozen": True,
        },
        "assets": {
            "denoiser": args.denoiser,
            "denoiser_sha256": _sha256(Path(args.denoiser)),
            "denoiser_metadata": denoiser_metadata,
            "embedding": args.embedding_checkpoint,
            "embedding_sha256": _sha256(Path(args.embedding_checkpoint)),
            "embedding_metadata": embedding_metadata,
        },
        "configs": configs,
        "configs_sha256": config_sha256,
        "real_configs": real_configs,
        "exact": {
            "source_names": exact_names,
            "panels": panels,
            "records": exact_records,
            "aggregate": exact_aggregate,
            "selection": exact_selection,
            "seconds": time.perf_counter() - exact_started,
        },
        "real16": {
            "source_names": real_names,
            "input_records": real_input_records,
            "records": real_records,
            "aggregate": real_aggregate,
            "paired_vs_promoted_baseline": paired,
            "promoted_label": promoted_label,
            "promoted_result": promoted,
            "target_only_oracle_all_candidates": {
                "mean_ssim": float(oracle.mean()),
                "gain_over_baseline": float((oracle - baseline_values).mean()),
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "upgrade_matrix_complete",
                "status": report["status"],
                "output": str(output),
                "output_sha256": _sha256(output),
                "frozen_sha256": frozen_sha256,
                "promoted_label": promoted_label,
                "promoted": promoted,
                "oracle_mean": float(oracle.mean()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
