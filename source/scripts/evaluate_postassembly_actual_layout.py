#!/usr/bin/env python3
"""Two-phase actual-QAP-layout gate for the frozen post-assembly harmonizer.

``freeze-layouts`` builds only the fixed production QAP-w4 permutations and
input-derived renderer tiles.  It never computes a target metric.  ``score``
first verifies those artifacts, freezes every render arm without modifying a
layout, and only then computes SSIM and seam metrics in a second pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from puzzle_assembly.compatibility import (  # noqa: E402
    build_classical_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.components import soft_cycle_component_solver  # noqa: E402
from puzzle_assembly.geometry import validate_permutation  # noqa: E402
from puzzle_assembly.learned import (  # noqa: E402
    learned_compatibility,
    load_embedding_checkpoint,
)
from puzzle_assembly.panels import make_exact_panel  # noqa: E402
from puzzle_assembly.postassembly_harmonizer import (  # noqa: E402
    SeamGraphConfig,
    apply_rgb_offsets,
    bilateral_tile_offsets,
    blend_tiles_uint8,
    image_quality_metrics,
    naive_local_mean_offsets,
    paired_bootstrap_ci,
    seam_graph_rgb_offsets,
)
from puzzle_assembly.protocol import per_source_seed  # noqa: E402
from puzzle_assembly.qap import directional_qap  # noqa: E402
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8  # noqa: E402


PANELS = ("primary_kornia", "independent_libjpeg")
RAW_ARM = "raw_on_frozen_qap_w4"
OLD_ARM = "selected_tilenaf_on_frozen_qap_w4"
SEAM_ARM = "production_seam_tilenaf_on_frozen_qap_w4"
BASELINE = "fixed_alpha_0_5_on_frozen_qap_w4"
CANDIDATE = "seam_graph_rgb_on_frozen_qap_w4"
PLACEBO = "shuffled_neighbor_placebo_on_frozen_qap_w4"
NAIVE = "naive_5x5_on_frozen_qap_w4"
BILATERAL = "bilateral_offset_on_frozen_qap_w4"
ARMS = (RAW_ARM, OLD_ARM, SEAM_ARM, BASELINE, CANDIDATE, PLACEBO, NAIVE, BILATERAL)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze-layouts", "score"))
    parser.add_argument(
        "--config", default="configs/postassembly_actual_qap_layout_v1.json"
    )
    parser.add_argument("--phase-a-dir", required=True)
    parser.add_argument("--output-dir", help="required for score")
    parser.add_argument("--limit", type=int, choices=(8, 32), default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--classical-chunk-size", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise RuntimeError(f"invalid target shape {values.shape}: {path}")
    return values


def _require_hash(path: Path, expected: str, role: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{role} hash mismatch: expected {expected}, got {actual}: {path}"
        )


def _protocol(config_path: Path, *, limit: int) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    config = _load_json(config_path)
    if config.get("kind") != "postassembly_actual_qap_layout_development_protocol":
        raise RuntimeError("unexpected actual-layout protocol kind")
    if config.get("status") != "precommitted_before_actual_layout_target_metrics":
        raise RuntimeError("actual-layout protocol is not precommitted")
    scope = config["scope"]
    required_scope = {
        "development_only": True,
        "may_promote_submission": False,
        "layout_refinement_forbidden": True,
        "layout_candidate_routing_forbidden": True,
        "same_frozen_qap_w4_layout_for_every_render_arm": True,
        "target_pixels_available_to_layout_predictor": False,
        "target_pixels_available_to_harmonizer": False,
    }
    for key, expected in required_scope.items():
        if scope.get(key) is not expected:
            raise RuntimeError(f"actual-layout scope drift: {key}")

    base_record = config["base_harmonizer_protocol"]
    base_path = REPO_ROOT / base_record["path"]
    _require_hash(base_path, base_record["sha256"], "base harmonizer protocol")
    base = _load_json(base_path)
    names = [str(value) for value in base["source_selection"]["names"]]
    if len(names) != 32 or _names_sha256(names) != config["source_selection"]["names_sha256"]:
        raise RuntimeError("actual-layout source list/hash drift")
    if _names_sha256(names[:8]) != config["source_selection"]["small_smoke_names_sha256"]:
        raise RuntimeError("actual-layout small8 hash drift")

    for role, record in config["assets"].items():
        _require_hash(REPO_ROOT / record["path"], record["sha256"], role)
    if tuple(config["synthetic_panels"]["names"]) != PANELS:
        raise RuntimeError("actual-layout panel order drift")
    predictor = config["frozen_layout_predictor"]
    expected_predictor = {
        "soft_cycle_top_k": 8,
        "soft_cycle_keep_per_tile": 1,
        "soft_cycle_keep_fraction": 0.5,
        "soft_cycle_loop_weight": 1.0,
        "soft_cycle_reciprocal_weight": 0.35,
        "qap_iterations": 25,
        "qap_restarts": 2,
        "qap_boundary_weight": 0.05,
        "qap_initial_weight": 0.75,
        "qap_noisy_components": 3,
        "qap_noise_scale": 1.0,
        "qap_refine_swaps": 8,
        "qap_refine_weak_cells": 32,
    }
    for key, expected in expected_predictor.items():
        if predictor.get(key) != expected:
            raise RuntimeError(f"production QAP-w4 parameter drift: {key}")
    if config["primary_comparison"] != {
        "candidate": CANDIDATE,
        "baseline": BASELINE,
        "only_pixel_render_changes": True,
        "position_to_slot_must_be_identical": True,
    }:
        raise RuntimeError("primary comparison drift")
    return config, base, names[:limit]


def _filename_qap_seed(source: str) -> int:
    return int.from_bytes(hashlib.sha256(source.encode("utf-8")).digest()[:4], "little") + 7001


def _predict_qap_w4(
    slot_tiles: np.ndarray,
    *,
    source: str,
    selected_model: torch.nn.Module,
    seam_model: torch.nn.Module,
    embedding_model: torch.nn.Module,
    device: torch.device,
    config: dict[str, Any],
    denoise_batch_size: int,
    classical_chunk_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Input-only predictor: deliberately has no target or permutation argument."""

    if slot_tiles.shape != (576, 20, 20, 3) or slot_tiles.dtype != np.uint8:
        raise ValueError("slot_tiles must be uint8 576x20x20x3")
    selected = restore_tiles_uint8(
        selected_model, slot_tiles, device, batch_size=denoise_batch_size
    )
    bank = build_classical_score_bank(
        selected, prefix="denoised", chunk_size=classical_chunk_size
    )
    c1_names = [
        key
        for key in sorted(bank)
        if key.startswith("denoised_") and not key.endswith("_c2")
    ]
    c1 = fuse_ranked_scores(bank, names=c1_names, name="denoised_C1")
    hbt, _ = learned_compatibility(
        embedding_model, selected, device=device, name="denoised_hbt_l1"
    )
    predictor = config["frozen_layout_predictor"]
    seed_result = soft_cycle_component_solver(
        hbt,
        top_k=int(predictor["soft_cycle_top_k"]),
        keep_per_tile=int(predictor["soft_cycle_keep_per_tile"]),
        proposal_keep_fraction=float(predictor["soft_cycle_keep_fraction"]),
        loop_weight=float(predictor["soft_cycle_loop_weight"]),
        reciprocal_weight=float(predictor["soft_cycle_reciprocal_weight"]),
    )
    initial = validate_permutation(
        seed_result.position_to_slot, name="soft_cycle_position_to_slot"
    )
    w4 = fuse_ranked_scores(
        {c1.name: c1, hbt.name: hbt},
        names=[c1.name, hbt.name],
        weights={hbt.name: 4.0},
        name="denoised_C1_HBTw4_rank_fusion",
    )
    qap_seed = _filename_qap_seed(source)
    result = directional_qap(
        w4,
        initial=initial,
        iterations=int(predictor["qap_iterations"]),
        restarts=int(predictor["qap_restarts"]),
        seed=qap_seed,
        boundary_weight=float(predictor["qap_boundary_weight"]),
        initial_weight=float(predictor["qap_initial_weight"]),
        noisy_components=int(predictor["qap_noisy_components"]),
        noise_scale=float(predictor["qap_noise_scale"]),
        refine_swaps=int(predictor["qap_refine_swaps"]),
        refine_weak_cells=int(predictor["qap_refine_weak_cells"]),
    )
    layout = validate_permutation(result.position_to_slot, name="qap_w4_position_to_slot")
    seam = restore_tiles_uint8(
        seam_model, slot_tiles, device, batch_size=denoise_batch_size
    )
    arrays = {
        "raw_slot_tiles": np.ascontiguousarray(slot_tiles),
        "selected_slot_tiles": np.ascontiguousarray(selected),
        "seam_slot_tiles": np.ascontiguousarray(seam),
        "position_to_slot": layout.astype(np.int32, copy=True),
    }
    diagnostics = {
        "qap_seed": qap_seed,
        "soft_cycle_accepted_edges": int(seed_result.accepted_edges),
        "soft_cycle_component_sizes": [int(value) for value in seed_result.component_sizes],
        "qap_objective": float(result.objective),
        "qap_relaxed_objective": float(result.relaxed_objective),
        "qap_restart": int(result.restart),
        "qap_iterations": int(result.iterations),
        "qap_converged": bool(result.converged),
    }
    return arrays, diagnostics


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _contained_artifact(root: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != "artifacts":
        raise RuntimeError(f"non-canonical artifact path: {relative_value}")
    path = (root / relative).resolve()
    artifact_root = (root / "artifacts").resolve()
    if path.parent != artifact_root or path.suffix != ".npz":
        raise RuntimeError(f"artifact path escaped root: {relative_value}")
    return path


def _freeze_layouts(
    args: argparse.Namespace,
    config_path: Path,
    config: dict[str, Any],
    names: list[str],
) -> None:
    phase_root = (REPO_ROOT / args.phase_a_dir).resolve()
    if phase_root.exists():
        raise FileExistsError(f"refusing to overwrite Phase A: {phase_root}")
    (phase_root / "artifacts").mkdir(parents=True)
    _write_json(
        phase_root / "PHASE_A_STARTED.json",
        {
            "kind": "postassembly_actual_layout_phase_a_started",
            "created_utc": _utc_now(),
            "config_sha256": _sha256(config_path),
            "source_count": len(names),
            "target_metrics_computed": False,
        },
    )

    torch.set_num_threads(args.torch_threads)
    assets = config["assets"]
    selected_model, device, selected_metadata = load_restorer(
        REPO_ROOT / assets["selected_tilenaf"]["path"],
        device=args.device,
        state=assets["selected_tilenaf"]["state"],
    )
    seam_model, seam_device, seam_metadata = load_restorer(
        REPO_ROOT / assets["production_seam_tilenaf"]["path"],
        device=str(device),
        state=assets["production_seam_tilenaf"]["state"],
    )
    if seam_device != device:
        raise RuntimeError("restorer devices differ")
    embedding, embedding_metadata = load_embedding_checkpoint(
        REPO_ROOT / assets["hbt"]["path"], device=device
    )
    for model in (selected_model, seam_model, embedding):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    records: list[dict[str, Any]] = []
    master_seed = int(config["synthetic_panels"]["master_seed"])
    started = time.perf_counter()
    for source_index, source in enumerate(names):
        # The clean image is used only to instantiate a synthetic input fixture.
        # Neither clean pixels nor make_exact_panel's permutation enter predictor.
        clean = _read_rgb(REPO_ROOT / "puzzle/train/targets" / source)
        for panel_index, panel in enumerate(PANELS):
            record_started = time.perf_counter()
            seed = per_source_seed(
                master_seed, f"postassembly-rgb-offset-{panel}", source, 0
            )
            exact = make_exact_panel(clean, panel=panel, seed=seed)
            arrays, diagnostics = _predict_qap_w4(
                exact.slot_tiles,
                source=source,
                selected_model=selected_model,
                seam_model=seam_model,
                embedding_model=embedding,
                device=device,
                config=config,
                denoise_batch_size=args.denoise_batch_size,
                classical_chunk_size=args.classical_chunk_size,
            )
            key = f"{source.removesuffix('.png')}__{panel}"
            artifact = phase_root / "artifacts" / f"{key}.npz"
            _atomic_npz(artifact, arrays)
            records.append(
                {
                    "source": source,
                    "source_index": source_index,
                    "panel": panel,
                    "panel_index": panel_index,
                    "panel_seed": seed,
                    "artifact": str(artifact.relative_to(phase_root)),
                    "artifact_sha256": _sha256(artifact),
                    "raw_slot_tiles_sha256": _array_sha256(arrays["raw_slot_tiles"]),
                    "selected_slot_tiles_sha256": _array_sha256(arrays["selected_slot_tiles"]),
                    "seam_slot_tiles_sha256": _array_sha256(arrays["seam_slot_tiles"]),
                    "layout_sha256": _array_sha256(arrays["position_to_slot"]),
                    "valid_permutation": True,
                    "diagnostics": diagnostics,
                    "seconds": float(time.perf_counter() - record_started),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "actual_layout_frozen",
                        "completed": len(records),
                        "total": len(names) * 2,
                        "source": source,
                        "panel": panel,
                        "layout_sha256": records[-1]["layout_sha256"],
                        "seconds": records[-1]["seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    manifest = {
        "schema_version": 1,
        "kind": "postassembly_actual_layout_phase_a_manifest",
        "created_utc": _utc_now(),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": _sha256(config_path),
        "base_harmonizer_config_sha256": config["base_harmonizer_protocol"]["sha256"],
        "source_count": len(names),
        "record_count": len(records),
        "source_names": names,
        "source_names_sha256": _names_sha256(names),
        "panels": list(PANELS),
        "target_metrics_computed": False,
        "layout_refinement_after_freeze_allowed": False,
        "all_valid_permutations": all(record["valid_permutation"] for record in records),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "harmonizer_sha256": _sha256(
            REPO_ROOT / "src/puzzle_assembly/postassembly_harmonizer.py"
        ),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "seconds": float(time.perf_counter() - started),
        },
        "model_metadata": {
            "selected_tilenaf": selected_metadata,
            "production_seam_tilenaf": seam_metadata,
            "hbt": embedding_metadata,
        },
        "records": records,
    }
    manifest_path = phase_root / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        phase_root / "PHASE_A_COMPLETE.json",
        {
            "kind": "postassembly_actual_layout_phase_a_complete",
            "created_utc": _utc_now(),
            "manifest_sha256": _sha256(manifest_path),
            "record_count": len(records),
            "target_metrics_computed": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "phase_a_complete_no_target_metrics",
                "phase_a_dir": str(phase_root.relative_to(REPO_ROOT)),
                "manifest_sha256": _sha256(manifest_path),
                "record_count": len(records),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def _load_phase_a(
    phase_root: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    names: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    complete = _load_json(phase_root / "PHASE_A_COMPLETE.json")
    manifest_path = phase_root / "manifest.json"
    if complete.get("manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("Phase A manifest hash mismatch")
    manifest = _load_json(manifest_path)
    expected = {
        "kind": "postassembly_actual_layout_phase_a_manifest",
        "config_sha256": _sha256(config_path),
        "base_harmonizer_config_sha256": config["base_harmonizer_protocol"]["sha256"],
        "source_count": len(names),
        "record_count": len(names) * 2,
        "source_names": names,
        "source_names_sha256": _names_sha256(names),
        "panels": list(PANELS),
        "target_metrics_computed": False,
        "layout_refinement_after_freeze_allowed": False,
        "all_valid_permutations": True,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "harmonizer_sha256": _sha256(
            REPO_ROOT / "src/puzzle_assembly/postassembly_harmonizer.py"
        ),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Phase A manifest drift: {key}")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(names) * 2:
        raise RuntimeError("invalid Phase A record list")
    expected_pairs = [(source, panel) for source in names for panel in PANELS]
    if [(record.get("source"), record.get("panel")) for record in records] != expected_pairs:
        raise RuntimeError("Phase A source/panel order drift")
    expected_files: set[Path] = set()
    for record in records:
        artifact = _contained_artifact(phase_root, str(record["artifact"]))
        expected_files.add(artifact)
        if _sha256(artifact) != record["artifact_sha256"]:
            raise RuntimeError(f"Phase A artifact hash mismatch: {artifact}")
        with np.load(artifact, allow_pickle=False) as payload:
            if set(payload.files) != {
                "raw_slot_tiles",
                "selected_slot_tiles",
                "seam_slot_tiles",
                "position_to_slot",
            }:
                raise RuntimeError(f"unexpected Phase A arrays: {artifact}")
            layout = validate_permutation(payload["position_to_slot"])
            if _array_sha256(layout) != record["layout_sha256"]:
                raise RuntimeError(f"Phase A layout hash mismatch: {artifact}")
            for key in ("raw_slot_tiles", "selected_slot_tiles", "seam_slot_tiles"):
                values = payload[key]
                if values.shape != (576, 20, 20, 3) or values.dtype != np.uint8:
                    raise RuntimeError(f"invalid {key}: {artifact}")
                if _array_sha256(values) != record[f"{key}_sha256"]:
                    raise RuntimeError(f"Phase A {key} hash mismatch: {artifact}")
    actual_files = set((phase_root / "artifacts").glob("*.npz"))
    if actual_files != expected_files:
        raise RuntimeError("Phase A artifact tree has missing or extra NPZ files")
    return manifest, records


def _rgb_config(base: dict[str, Any]) -> SeamGraphConfig:
    values = base["methods"]["seam_graph_rgb_on_blend"]
    return SeamGraphConfig(
        extrapolation_band=int(values["extrapolation_band"]),
        confidence_scale=float(values["confidence_scale"]),
        confidence_floor=float(values["confidence_floor"]),
        ridge=float(values["ridge"]),
        huber_delta=float(values["huber_delta"]),
        irls_steps=int(values["irls_steps"]),
        max_abs_offset=float(values["max_abs_offset"]),
    )


def _construct_arms(
    payload: Any,
    *,
    layout: np.ndarray,
    source: str,
    panel: str,
    config: dict[str, Any],
    base: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = np.asarray(payload["raw_slot_tiles"])[layout]
    selected = np.asarray(payload["selected_slot_tiles"])[layout]
    seam = np.asarray(payload["seam_slot_tiles"])[layout]
    blend = blend_tiles_uint8(selected, seam, auxiliary_weight=0.5)
    rgb_config = _rgb_config(base)
    offsets, diagnostics = seam_graph_rgb_offsets(blend, rgb_config)
    placebo_seed = per_source_seed(
        int(config["synthetic_panels"]["master_seed"]),
        f"postassembly-placebo-{panel}",
        source,
        0,
    )
    placebo_offsets, placebo_diagnostics = seam_graph_rgb_offsets(
        blend, rgb_config, placebo_seed=placebo_seed
    )
    naive_config = base["methods"]["naive_5x5_on_blend"]
    naive_offsets = naive_local_mean_offsets(
        blend,
        radius=int(naive_config["radius_tiles"]),
        strength=float(naive_config["strength"]),
        max_abs_offset=float(naive_config["max_abs_rgb_offset"]),
    )
    bilateral_config = base["methods"]["bilateral_offset_on_blend"]
    bilateral_offsets = bilateral_tile_offsets(
        blend,
        radius=int(bilateral_config["radius_tiles"]),
        sigma_spatial=float(bilateral_config["sigma_spatial"]),
        sigma_colour=float(bilateral_config["sigma_colour"]),
        strength=float(bilateral_config["strength"]),
        max_abs_offset=float(bilateral_config["max_abs_rgb_offset"]),
    )
    arms = {
        RAW_ARM: raw,
        OLD_ARM: selected,
        SEAM_ARM: seam,
        BASELINE: blend,
        CANDIDATE: apply_rgb_offsets(blend, offsets),
        PLACEBO: apply_rgb_offsets(blend, placebo_offsets),
        NAIVE: apply_rgb_offsets(blend, naive_offsets),
        BILATERAL: apply_rgb_offsets(blend, bilateral_offsets),
    }
    return arms, {CANDIDATE: diagnostics, PLACEBO: placebo_diagnostics}


def _layout_metrics(layout: np.ndarray, slot_to_target: np.ndarray) -> dict[str, Any]:
    layout = validate_permutation(layout)
    slot_to_target = validate_permutation(slot_to_target, name="slot_to_target")
    target_positions = slot_to_target[layout]
    grid = target_positions.reshape(24, 24)
    right = (grid[:, 1:] == grid[:, :-1] + 1) & (grid[:, :-1] % 24 < 23)
    down = grid[1:, :] == grid[:-1, :] + 24
    return {
        "valid_permutation": True,
        "strict_position_accuracy": float(
            np.mean(target_positions == np.arange(576, dtype=np.int32))
        ),
        "right_down_adjacency_recall": float(
            (int(right.sum()) + int(down.sum())) / 1104.0
        ),
    }


def _comparison(
    records: list[dict[str, Any]], *, bootstrap_seed: int, resamples: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for panel_index, panel in enumerate(PANELS):
        selected = [record for record in records if record["panel"] == panel]
        deltas = np.asarray(
            [
                record["metrics"][CANDIDATE]["ssim"]
                - record["metrics"][BASELINE]["ssim"]
                for record in selected
            ],
            dtype=np.float64,
        )
        seam_deltas = np.asarray(
            [
                record["metrics"][CANDIDATE]["target_referenced_seam_error"]
                - record["metrics"][BASELINE]["target_referenced_seam_error"]
                for record in selected
            ],
            dtype=np.float64,
        )
        low, high = paired_bootstrap_ci(
            deltas, seed=bootstrap_seed + panel_index, resamples=resamples
        )
        result[panel] = {
            "source_count": len(selected),
            "mean_ssim_delta": float(deltas.mean()),
            "paired_bootstrap_95_ci": [low, high],
            "wins_ties_losses": [
                int(np.sum(deltas > 1e-12)),
                int(np.sum(np.abs(deltas) <= 1e-12)),
                int(np.sum(deltas < -1e-12)),
            ],
            "large_regressions_below_minus_0_01": int(np.sum(deltas < -0.01)),
            "mean_target_referenced_seam_error_delta": float(seam_deltas.mean()),
        }
    return result


def _panel_macro(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metric_names = (
        "ssim",
        "boundary_band_mae",
        "target_referenced_seam_error",
        "untargeted_seam_discontinuity",
        "mae",
    )
    for panel in PANELS:
        selected = [record for record in records if record["panel"] == panel]
        result[panel] = {
            arm: {
                metric: float(
                    np.mean([record["metrics"][arm][metric] for record in selected])
                )
                for metric in metric_names
            }
            for arm in ARMS
        }
        result[panel]["layout"] = {
            "valid_permutation_rate": float(
                np.mean([record["layout_metrics"]["valid_permutation"] for record in selected])
            ),
            "strict_position_accuracy": float(
                np.mean([record["layout_metrics"]["strict_position_accuracy"] for record in selected])
            ),
            "right_down_adjacency_recall": float(
                np.mean(
                    [record["layout_metrics"]["right_down_adjacency_recall"] for record in selected]
                )
            ),
        }
    return result


def _gate(
    comparison: dict[str, Any],
    config: dict[str, Any],
    *,
    source_count: int,
    valid_count: int,
) -> dict[str, Any]:
    if source_count == 8:
        gate = config["small8_advance_rule"]
        per_panel = {}
        for panel in PANELS:
            values = comparison[panel]
            checks = {
                "mean_ssim_delta_above_zero": values["mean_ssim_delta"]
                > float(gate["both_panels_mean_ssim_delta_must_exceed"]),
                "wins_at_least_5_of_8": values["wins_ties_losses"][0]
                >= int(gate["both_panels_wins_minimum_of_8"]),
                "large_regressions_at_most_1": values[
                    "large_regressions_below_minus_0_01"
                ]
                <= int(gate["both_panels_large_regressions_below_minus_0_01_maximum"]),
                "seam_error_nonregression": values[
                    "mean_target_referenced_seam_error_delta"
                ]
                <= float(gate["both_panels_mean_target_referenced_seam_error_delta_maximum"]),
            }
            per_panel[panel] = {"checks": checks, "passed": bool(all(checks.values()))}
        passed = valid_count == 16 and all(value["passed"] for value in per_panel.values())
        return {
            "kind": "small8_advance_rule",
            "evaluable": True,
            "valid_permutation_count": valid_count,
            "per_panel": per_panel,
            "passed": bool(passed),
            "submission_promotion_allowed": False,
        }
    gate = config["full32_gate"]
    per_panel = {}
    for panel in PANELS:
        values = comparison[panel]
        checks = {
            "mean_ssim_delta_at_least_0_005": values["mean_ssim_delta"]
            >= float(gate["per_panel_mean_ssim_delta_minimum"]),
            "bootstrap_lower_above_zero": values["paired_bootstrap_95_ci"][0]
            > float(gate["per_panel_paired_bootstrap_lower_must_exceed"]),
            "seam_error_nonregression": values[
                "mean_target_referenced_seam_error_delta"
            ]
            <= float(gate["per_panel_mean_target_referenced_seam_error_delta_maximum"]),
        }
        per_panel[panel] = {"checks": checks, "passed": bool(all(checks.values()))}
    passed = valid_count == 64 and all(value["passed"] for value in per_panel.values())
    return {
        "kind": "full32_gate",
        "evaluable": True,
        "valid_permutation_count": valid_count,
        "per_panel": per_panel,
        "passed": bool(passed),
        "submission_promotion_allowed": False,
    }


def _score(
    args: argparse.Namespace,
    config_path: Path,
    config: dict[str, Any],
    base: dict[str, Any],
    names: list[str],
) -> None:
    if not args.output_dir:
        raise SystemExit("score requires --output-dir")
    phase_root = (REPO_ROOT / args.phase_a_dir).resolve()
    manifest, phase_records = _load_phase_a(
        phase_root, config_path=config_path, config=config, names=names
    )
    output_root = (REPO_ROOT / args.output_dir).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite score output: {output_root}")
    (output_root / "frozen_renders").mkdir(parents=True)
    _write_json(
        output_root / "RENDER_FREEZE_STARTED.json",
        {
            "kind": "postassembly_actual_layout_render_freeze_started",
            "created_utc": _utc_now(),
            "phase_a_manifest_sha256": _sha256(phase_root / "manifest.json"),
            "target_metrics_computed": False,
        },
    )

    # Pass 1: verify exact synthetic inputs and freeze every target-blind arm.
    render_records: list[dict[str, Any]] = []
    for phase_record in phase_records:
        source = str(phase_record["source"])
        panel = str(phase_record["panel"])
        clean = _read_rgb(REPO_ROOT / "puzzle/train/targets" / source)
        exact = make_exact_panel(clean, panel=panel, seed=int(phase_record["panel_seed"]))
        artifact = _contained_artifact(phase_root, str(phase_record["artifact"]))
        with np.load(artifact, allow_pickle=False) as payload:
            if not np.array_equal(payload["raw_slot_tiles"], exact.slot_tiles):
                raise RuntimeError(f"Phase A synthetic input recomposition mismatch: {source}/{panel}")
            layout = validate_permutation(payload["position_to_slot"])
            arms, diagnostics = _construct_arms(
                payload,
                layout=layout,
                source=source,
                panel=panel,
                config=config,
                base=base,
            )
        key = f"{source.removesuffix('.png')}__{panel}"
        render_path = output_root / "frozen_renders" / f"{key}.npz"
        _atomic_npz(render_path, arms)
        render_records.append(
            {
                "source": source,
                "panel": panel,
                "panel_seed": int(phase_record["panel_seed"]),
                "phase_a_artifact_sha256": phase_record["artifact_sha256"],
                "layout_sha256": phase_record["layout_sha256"],
                "render_artifact": str(render_path.relative_to(output_root)),
                "render_artifact_sha256": _sha256(render_path),
                "arm_sha256": {name: _array_sha256(values) for name, values in arms.items()},
                "diagnostics": diagnostics,
                "layout_changed": False,
            }
        )
    render_manifest = {
        "schema_version": 1,
        "kind": "postassembly_actual_layout_frozen_render_manifest",
        "created_utc": _utc_now(),
        "config_sha256": _sha256(config_path),
        "phase_a_manifest_sha256": _sha256(phase_root / "manifest.json"),
        "record_count": len(render_records),
        "all_arms": list(ARMS),
        "layout_refinement_performed": False,
        "target_metrics_computed": False,
        "records": render_records,
    }
    render_manifest_path = output_root / "render_manifest.json"
    _write_json(render_manifest_path, render_manifest)
    _write_json(
        output_root / "TARGET_METRICS_STARTED.json",
        {
            "kind": "postassembly_actual_layout_target_metrics_started",
            "created_utc": _utc_now(),
            "render_manifest_sha256": _sha256(render_manifest_path),
            "all_layouts_and_arms_frozen": True,
        },
    )

    # Pass 2: target-aware scoring only. No predictor/harmonizer is called here.
    scored: list[dict[str, Any]] = []
    for render_record in render_records:
        source = render_record["source"]
        panel = render_record["panel"]
        clean = _read_rgb(REPO_ROOT / "puzzle/train/targets" / source)
        exact = make_exact_panel(clean, panel=panel, seed=render_record["panel_seed"])
        render_path = output_root / render_record["render_artifact"]
        if _sha256(render_path) != render_record["render_artifact_sha256"]:
            raise RuntimeError(f"frozen render hash mismatch: {render_path}")
        phase_record = next(
            record
            for record in phase_records
            if record["source"] == source and record["panel"] == panel
        )
        phase_artifact = _contained_artifact(phase_root, phase_record["artifact"])
        with np.load(phase_artifact, allow_pickle=False) as phase_payload:
            layout = validate_permutation(phase_payload["position_to_slot"])
        with np.load(render_path, allow_pickle=False) as renders:
            if tuple(renders.files) != ARMS:
                raise RuntimeError(f"frozen render arm order drift: {render_path}")
            for arm in ARMS:
                if _array_sha256(renders[arm]) != render_record["arm_sha256"][arm]:
                    raise RuntimeError(f"frozen arm hash mismatch: {source}/{panel}/{arm}")
            metrics = {
                arm: image_quality_metrics(renders[arm], exact.clean_target_tiles)
                for arm in ARMS
            }
        scored.append(
            {
                "source": source,
                "panel": panel,
                "layout_sha256": render_record["layout_sha256"],
                "layout_metrics": _layout_metrics(layout, exact.slot_to_target),
                "metrics": metrics,
                "diagnostics": render_record["diagnostics"],
            }
        )
        print(
            json.dumps(
                {
                    "event": "actual_layout_scored",
                    "completed": len(scored),
                    "total": len(render_records),
                    "source": source,
                    "panel": panel,
                    "adjacency": scored[-1]["layout_metrics"]["right_down_adjacency_recall"],
                    "baseline_ssim": metrics[BASELINE]["ssim"],
                    "candidate_ssim": metrics[CANDIDATE]["ssim"],
                    "delta": metrics[CANDIDATE]["ssim"] - metrics[BASELINE]["ssim"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    metric_config = config["metrics"]
    comparison = _comparison(
        scored,
        bootstrap_seed=int(metric_config["paired_bootstrap_seed"]),
        resamples=int(metric_config["paired_bootstrap_resamples"]),
    )
    valid_count = sum(
        int(record["layout_metrics"]["valid_permutation"]) for record in scored
    )
    gate = _gate(
        comparison,
        config,
        source_count=len(names),
        valid_count=valid_count,
    )
    if len(names) == 8:
        status = "small8_pass_run_unchanged_full32" if gate["passed"] else "small8_fail_stop_no_retune"
    else:
        status = "full32_pass_requires_new_confirmation" if gate["passed"] else "full32_fail_stop_or_redesign"
    report = {
        "schema_version": 1,
        "kind": "postassembly_actual_qap_layout_report",
        "created_utc": _utc_now(),
        "status": status,
        "development_only": True,
        "submission_promotion_allowed": False,
        "source_count": len(names),
        "record_count": len(scored),
        "source_names": names,
        "source_names_sha256": _names_sha256(names),
        "config_sha256": _sha256(config_path),
        "phase_a_manifest_sha256": _sha256(phase_root / "manifest.json"),
        "render_manifest_sha256": _sha256(render_manifest_path),
        "all_layouts_frozen_before_render_metrics": True,
        "layout_refinement_performed": False,
        "same_layout_for_all_arms": True,
        "correct_layout_ceiling_is_separate_evidence": True,
        "panel_macro": _panel_macro(scored),
        "primary_comparison": comparison,
        "gate": gate,
        "records": scored,
    }
    report_path = output_root / "report.json"
    _write_json(report_path, report)
    result = {
        "status": status,
        "report": str(report_path.relative_to(REPO_ROOT)),
        "report_sha256": _sha256(report_path),
        "source_count": len(names),
        "primary_comparison": comparison,
        "gate": gate,
    }
    _write_json(output_root / "RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main() -> None:
    args = _parse_args()
    if args.denoise_batch_size <= 0 or args.classical_chunk_size <= 0 or args.torch_threads <= 0:
        raise SystemExit("batch/chunk/thread arguments must be positive")
    config_path = (REPO_ROOT / args.config).resolve()
    config, base, names = _protocol(config_path, limit=args.limit)
    if args.action == "freeze-layouts":
        if args.output_dir:
            raise SystemExit("freeze-layouts does not accept --output-dir")
        _freeze_layouts(args, config_path, config, names)
    else:
        _score(args, config_path, config, base, names)


if __name__ == "__main__":
    main()
