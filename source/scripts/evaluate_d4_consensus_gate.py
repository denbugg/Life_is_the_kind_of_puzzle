#!/usr/bin/env python3
"""Frozen two-stage exact gate for D4 test-time compatibility consensus.

Phase A creates every baseline and candidate layout using input pixels only.
Only after all target-blind prerequisites pass are clean permutations and
pixels attached for adjacency and promoted-renderer SSIM scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for value in (REPO_ROOT, SCRIPT_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.d4_consensus import (
    D4_VIEWS,
    d4_rank_consensus,
    inverse_transform_tiles,
    transform_tiles,
)
from puzzle_assembly.geometry import TILE_COUNT, validate_permutation
from puzzle_assembly.learned import learned_compatibility, load_embedding_checkpoint
from puzzle_assembly.metrics import layout_metrics
from puzzle_assembly.panels import make_exact_panel
from puzzle_assembly.postassembly_harmonizer import (
    LuminanceGainConfig,
    SeamGraphConfig,
    apply_luminance_gains,
    apply_rgb_offsets,
    blend_tiles_uint8,
    image_quality_metrics,
    seam_graph_luminance_gains,
    seam_graph_rgb_offsets,
)
from puzzle_assembly.protocol import per_source_seed, source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from build_assembly_submission import _promoted_score_bank


PANELS = ("primary_kornia", "independent_libjpeg")
FROZEN_SPLIT = "edge_development"
FROZEN_SOURCE_OFFSET = 340
FROZEN_SOURCE_COUNT = 8
FROZEN_SOURCE_NAMES_SHA256 = (
    "ddc0a394fbac1f5674a1f724de1b0617e32e0e26d308c6ff5ee369d6452055be"
)
EXPECTED_ASSET_SHA256 = {
    "selected_denoiser": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "seam_denoiser": "f973c7e606a112020c527bb72277b82586df915edc829a22305e587b35aec1b9",
    "embedding": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
}
AUTHORITATIVE_BUILDER_SHA256 = (
    "8433c0e545edfeb49f2512208a3ea062fb1a248a64bcde3f87037cdf30d6ac97"
)
EXPECTED_SLOW_REFERENCE_SHA256 = {
    "primary_kornia": "66710a9e2e42b98658ca7513d8bb08b5458c84152552114f79640c8bd1afd45b",
    "independent_libjpeg": "f9c132f45b1520cb07d03b405509efdf5802cffd71dcf0546c2033387554e845",
}
EXPECTED_SLOW_ARTIFACT_SHA256 = {
    "primary_kornia": "ce426ee278f156257585d1403ed362618fb3b1b2083e28c921e7be796e1a46a3",
    "independent_libjpeg": "f60964e4bbcfc61a7ef19bc7fa21c8426835f6cca3a38b783c1f234599fbdcbf",
}
RGB_CONFIG = SeamGraphConfig(
    extrapolation_band=3,
    confidence_scale=12.0,
    confidence_floor=0.05,
    ridge=0.20,
    huber_delta=4.0,
    irls_steps=4,
    max_abs_offset=12.0,
)
LUMA_CONFIG = LuminanceGainConfig(
    extrapolation_band=3,
    confidence_scale=0.08,
    confidence_floor=0.05,
    ridge=0.50,
    huber_delta=0.025,
    irls_steps=4,
    max_fractional_gain=0.04,
    luminance_floor=12.0,
    luminance_ceiling=243.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=PANELS, required=True)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--selected-denoiser", required=True)
    parser.add_argument("--seam-denoiser", required=True)
    parser.add_argument("--embedding-checkpoint", required=True)
    parser.add_argument("--slow-phase-a-reference", required=True)
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument("--quarantine", default="configs/denoise_validation_quarantine_v1.json")
    parser.add_argument("--audit-exclusion", default="configs/assembly_audit_exclusion_v1.json")
    parser.add_argument("--split", default=FROZEN_SPLIT)
    parser.add_argument("--source-offset", type=int, default=FROZEN_SOURCE_OFFSET)
    parser.add_argument("--sources", type=int, default=FROZEN_SOURCE_COUNT)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--phase", choices=("phase-a", "phase-b"), required=True)
    parser.add_argument("--phase-a-artifact")
    parser.add_argument("--phase-a-report")
    parser.add_argument("--phase-b-authorization")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected RGB shape: {path}: {values.shape}")
    return values


def _build_production_scores(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    embedding_model: Any,
    *,
    device: Any,
) -> tuple[CompatibilityMatrices, CompatibilityMatrices]:
    """Call the authoritative submission score builder for l1 and l1w4."""

    bank, aliases = _promoted_score_bank(
        raw_tiles,
        denoised_tiles,
        embedding_model=embedding_model,
        device=device,
        chunk_size=64,
        include_line_seam=False,
        line_seam_auxiliary_weight=0.0,
        line_seam_fusion_weight=0.0,
    )
    return bank[aliases["l1"]], bank[aliases["l1w4"]]


def _build_fast_equivalent_scores(
    denoised_tiles: np.ndarray,
    embedding_model: Any,
    *,
    device: Any,
) -> tuple[CompatibilityMatrices, CompatibilityMatrices]:
    """Compute only the denoised scores used by production l1/l1w4.

    The authoritative builder also creates raw/cross arms that neither the
    frozen l1 seed nor l1w4 QAP consumes.  Phase A proves this reduced builder
    bit-exact on the identity view before it is allowed for D4 views.
    """

    bank = build_classical_score_bank(
        denoised_tiles, prefix="denoised", chunk_size=64
    )
    c1_names = [
        name
        for name in sorted(bank)
        if name.startswith("denoised_") and not name.endswith("_c2")
    ]
    c1 = fuse_ranked_scores(
        bank,
        names=c1_names,
        name="denoised_C1_equal_rank_fusion",
    )
    l1, _outside = learned_compatibility(
        embedding_model,
        denoised_tiles,
        device=device,
        name="denoised_l1_embedding",
    )
    l1w4 = fuse_ranked_scores(
        {c1.name: c1, l1.name: l1},
        names=[c1.name, l1.name],
        weights={l1.name: 4.0},
        name="denoised_C1_L1w4_rank_fusion",
    )
    return l1, l1w4


def _scores_bit_exact(
    first: CompatibilityMatrices, second: CompatibilityMatrices
) -> bool:
    if first.name != second.name:
        return False
    diagonal = np.diag_indices(TILE_COUNT)
    off_diagonal = ~np.eye(TILE_COUNT, dtype=bool)
    for side in ("right", "down"):
        left = np.asarray(getattr(first, side))
        right = np.asarray(getattr(second, side))
        if left.shape != (TILE_COUNT, TILE_COUNT) or right.shape != left.shape:
            return False
        if left.dtype != np.float32 or right.dtype != np.float32:
            return False
        if not np.isposinf(left[diagonal]).all() or not np.isposinf(right[diagonal]).all():
            return False
        if not np.isfinite(left[off_diagonal]).all() or not np.isfinite(
            right[off_diagonal]
        ).all():
            return False
        if not np.array_equal(left, right):
            return False
    return _score_hash(first) == _score_hash(second)


def names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _assert_finite_payload(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_payload(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_payload(item, path=f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise RuntimeError(f"non-finite report value at {path}")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_write_phase_a_artifact(
    path: Path, phase_a: list[dict[str, Any]]
) -> str:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    arrays: dict[str, np.ndarray] = {
        "names": np.asarray([item["name"] for item in phase_a]),
        "panel_seeds": np.asarray(
            [item["panel_seed"] for item in phase_a], dtype=np.uint64
        ),
    }
    for key in (
        "raw_tiles",
        "selected_tiles",
        "seam_tiles",
        "baseline_layout",
        "candidate_layout",
        "baseline_render",
        "candidate_render",
    ):
        arrays[key] = np.stack([item[key] for item in phase_a], axis=0)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256(path)


def _load_phase_a_artifact(
    path: Path,
    *,
    expected_sha256: str,
    expected_names: Sequence[str],
) -> dict[str, np.ndarray]:
    if sha256(path) != expected_sha256:
        raise RuntimeError("Phase-A artifact hash mismatch")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "names",
            "panel_seeds",
            "raw_tiles",
            "selected_tiles",
            "seam_tiles",
            "baseline_layout",
            "candidate_layout",
            "baseline_render",
            "candidate_render",
        }
        if set(payload.files) != required:
            raise RuntimeError(f"Phase-A artifact schema drift: {payload.files}")
        arrays = {key: np.ascontiguousarray(payload[key]) for key in payload.files}
    if arrays["names"].tolist() != list(expected_names):
        raise RuntimeError("Phase-A artifact source order drift")
    return arrays


def _filename_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def _score_hash(score: CompatibilityMatrices) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(score.right, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(score.down, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _solve_layout(
    seed_score: CompatibilityMatrices,
    qap_score: CompatibilityMatrices,
    name: str,
) -> np.ndarray:
    """Production l1 soft-cycle + l1w4 QAP, with no target-bearing argument."""

    seed_result = soft_cycle_component_solver(
        seed_score,
        top_k=8,
        keep_per_tile=1,
        proposal_keep_fraction=0.5,
        loop_weight=1.0,
        reciprocal_weight=0.35,
    )
    result = directional_qap(
        qap_score,
        initial=seed_result.position_to_slot,
        iterations=25,
        restarts=2,
        seed=_filename_seed(name) + 7001,
        boundary_weight=0.05,
        initial_weight=0.75,
        noisy_components=3,
        noise_scale=1.0,
        refine_swaps=8,
        refine_weak_cells=32,
    )
    return validate_permutation(result.position_to_slot, name="position_to_slot")


def _restore_nonidentity_views(
    restorer: Any,
    raw_tiles: np.ndarray,
    device: Any,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    transformed = [transform_tiles(raw_tiles, view) for view in D4_VIEWS[1:]]
    packed = np.concatenate(transformed, axis=0)
    restored = restore_tiles_uint8(restorer, packed, device, batch_size=batch_size)
    output: dict[str, np.ndarray] = {}
    for index, view in enumerate(D4_VIEWS[1:]):
        start = index * TILE_COUNT
        current = restored[start : start + TILE_COUNT]
        output[view] = inverse_transform_tiles(current, view)
    return output


def _render_promoted(
    selected_slot_tiles: np.ndarray,
    seam_slot_tiles: np.ndarray,
    layout: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    layout = validate_permutation(layout, name="render_layout")
    selected = np.ascontiguousarray(selected_slot_tiles[layout])
    seam = np.ascontiguousarray(seam_slot_tiles[layout])
    blended = blend_tiles_uint8(selected, seam, auxiliary_weight=0.5)
    offsets, rgb_diagnostics = seam_graph_rgb_offsets(blended, RGB_CONFIG)
    rgb = apply_rgb_offsets(blended, offsets)
    gains, luma_diagnostics = seam_graph_luminance_gains(rgb, LUMA_CONFIG)
    promoted = apply_luminance_gains(rgb, gains)
    return promoted, {
        "rgb": rgb_diagnostics,
        "luma": luma_diagnostics,
        "ordered_tiles_sha256": array_sha256(promoted),
    }


def _panel_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    adjacency = np.asarray([record["delta"]["combined_adjacency"] for record in records])
    ssim = np.asarray([record["delta"]["harmonized_ssim"] for record in records])
    runtime = np.asarray([record["phase_a"]["candidate_seconds"] for record in records])
    return {
        "records": len(records),
        "mean_combined_adjacency_delta": float(adjacency.mean()),
        "mean_harmonized_ssim_delta": float(ssim.mean()),
        "ssim_wins": int(np.count_nonzero(ssim > 0)),
        "ssim_ties": int(np.count_nonzero(ssim == 0)),
        "ssim_losses": int(np.count_nonzero(ssim < 0)),
        "worst_harmonized_ssim_delta": float(ssim.min()),
        "mean_candidate_seconds": float(runtime.mean()),
        "max_candidate_seconds": float(runtime.max()),
        "different_layouts": int(
            sum(record["phase_a"]["layout_changed"] for record in records)
        ),
    }


def _panel_gate(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "mean_combined_adjacency_delta_ge_0.003": summary[
            "mean_combined_adjacency_delta"
        ] >= 0.003,
        "mean_harmonized_ssim_delta_ge_0.002": summary[
            "mean_harmonized_ssim_delta"
        ] >= 0.002,
        "ssim_wins_ge_6_of_8": summary["ssim_wins"] >= 6,
        "worst_harmonized_ssim_delta_ge_-0.010": summary[
            "worst_harmonized_ssim_delta"
        ] >= -0.010,
        "max_candidate_seconds_le_20": summary["max_candidate_seconds"] <= 20.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "decision": "panel_pass" if all(checks.values()) else "stop_d4_no_panel_signal",
    }


def _resolve_protocol(args: argparse.Namespace) -> tuple[dict[str, str], list[str]]:
    if (
        args.split != FROZEN_SPLIT
        or args.source_offset != FROZEN_SOURCE_OFFSET
        or args.sources != FROZEN_SOURCE_COUNT
    ):
        raise SystemExit("frozen D4 diagnostic protocol drift")
    actual_assets = {
        "selected_denoiser": sha256(args.selected_denoiser),
        "seam_denoiser": sha256(args.seam_denoiser),
        "embedding": sha256(args.embedding_checkpoint),
    }
    if actual_assets != EXPECTED_ASSET_SHA256:
        raise RuntimeError(f"frozen asset hash drift: {actual_assets}")
    slow_reference_sha256 = sha256(args.slow_phase_a_reference)
    if slow_reference_sha256 != EXPECTED_SLOW_REFERENCE_SHA256[args.panel]:
        raise RuntimeError("slow Phase-A input-only reference hash drift")
    actual_assets["slow_phase_a_reference"] = slow_reference_sha256
    builder = REPO_ROOT / "scripts/build_assembly_submission.py"
    if sha256(builder) != AUTHORITATIVE_BUILDER_SHA256:
        raise RuntimeError("authoritative production score-builder hash drift")
    names = source_names_for_split(
        args.split,
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
        audit_exclusion_path=args.audit_exclusion,
    )[args.source_offset : args.source_offset + args.sources]
    if len(names) != FROZEN_SOURCE_COUNT or names_sha256(names) != FROZEN_SOURCE_NAMES_SHA256:
        raise RuntimeError("frozen D4 source slice drift")
    for protected_split in (
        "assembly_cal",
        "assembly_incremental_gate",
        "assembly_audit_exposed",
        "assembly_final_audit",
    ):
        protected = set(
            source_names_for_split(
                protected_split,
                manifest_path=args.manifest,
                quarantine_path=args.quarantine,
                audit_exclusion_path=args.audit_exclusion,
            )
        )
        if protected.intersection(names):
            raise RuntimeError(f"whole-source overlap with {protected_split}")
    return actual_assets, names


def _protocol(names: Sequence[str]) -> dict[str, Any]:
    return {
        "split": FROZEN_SPLIT,
        "source_offset": FROZEN_SOURCE_OFFSET,
        "source_count": FROZEN_SOURCE_COUNT,
        "source_names": list(names),
        "source_names_sha256": FROZEN_SOURCE_NAMES_SHA256,
        "views": list(D4_VIEWS),
        "formula": "0.50*identity_row_rank + 0.40*median4_row_rank + 0.10*MAD4",
        "parameter_sweeps": 0,
        "production_solver": "softcycle_l1_top8_keep1_fraction0.5_then_qap_l1w4_25x2_b0.05",
        "production_renderer": "selected_seam_half_blend_rgb_offsets_luma_gain",
        "authoritative_score_builder_sha256": AUTHORITATIVE_BUILDER_SHA256,
        "phase_a_target_metrics_opened": False,
        "solver_accepts_target_or_truth": False,
    }


def _run_phase_a(
    args: argparse.Namespace,
    output: Path,
    artifact: Path,
    actual_assets: dict[str, str],
    names: list[str],
) -> None:
    selected_model, device, selected_metadata = load_restorer(
        args.selected_denoiser, device=args.device
    )
    seam_model, seam_device, seam_metadata = load_restorer(
        args.seam_denoiser, device=str(device)
    )
    if seam_device != device:
        raise RuntimeError("selected and seam denoisers resolved to different devices")
    embedding, embedding_metadata = load_embedding_checkpoint(
        args.embedding_checkpoint, device=device
    )
    for model in (selected_model, seam_model, embedding):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    learned_names = (
        set(embedding_metadata.get("train_names", []))
        | set(embedding_metadata.get("val_names", []))
        | set(embedding_metadata.get("validation_names", []))
    )
    if learned_names.intersection(names):
        raise RuntimeError("whole-source overlap with frozen embedding fit/validation")

    slow_reference = json.loads(Path(args.slow_phase_a_reference).read_text())
    if (
        slow_reference.get("kind") != "d4_compatibility_consensus_phase_a"
        or slow_reference.get("panel") != args.panel
        or slow_reference.get("phase_a", {}).get("target_metrics_opened") is not False
        or slow_reference.get("artifact", {}).get("sha256")
        != EXPECTED_SLOW_ARTIFACT_SHA256[args.panel]
        or [record.get("name") for record in slow_reference.get("records", [])]
        != names
    ):
        raise RuntimeError("invalid slow Phase-A input-only reference")

    phase_a: list[dict[str, Any]] = []
    for source_index, name in enumerate(names):
        clean_for_input_generation = read_rgb(Path(args.data_root) / "train/targets" / name)
        panel_seed = per_source_seed(args.seed, f"d4-consensus-{args.panel}", name, 0)
        exact_input = make_exact_panel(
            clean_for_input_generation, panel=args.panel, seed=panel_seed
        )
        raw_tiles = np.ascontiguousarray(exact_input.slot_tiles)
        del exact_input, clean_for_input_generation
        for view in D4_VIEWS:
            if not np.array_equal(
                inverse_transform_tiles(transform_tiles(raw_tiles, view), view), raw_tiles
            ):
                raise RuntimeError(f"non-exact D4 inverse for {view}")

        audit_started = time.perf_counter()
        audit_selected_tiles = restore_tiles_uint8(
            selected_model, raw_tiles, device, batch_size=args.denoise_batch_size
        )
        # Correctness prerequisite (excluded from deployable candidate runtime):
        # the reduced builder must be bit-exact to the hash-pinned production
        # builder on every identity input.
        reference_l1, reference_w4 = _build_production_scores(
            raw_tiles, audit_selected_tiles, embedding, device=device
        )
        audit_fast_l1, audit_fast_w4 = _build_fast_equivalent_scores(
            audit_selected_tiles, embedding, device=device
        )
        identity_l1_exact = _scores_bit_exact(reference_l1, audit_fast_l1)
        identity_w4_exact = _scores_bit_exact(reference_w4, audit_fast_w4)
        if not identity_l1_exact or not identity_w4_exact:
            raise RuntimeError("fast D4 score builder is not bit-exact to production")
        audit_seconds = time.perf_counter() - audit_started

        # Full deployable candidate is recomputed fresh from raw input.  No
        # tensor or score from the untimed equivalence audit is reused.
        candidate_started = time.perf_counter()
        selected_tiles = restore_tiles_uint8(
            selected_model, raw_tiles, device, batch_size=args.denoise_batch_size
        )
        identity_l1, identity_w4 = _build_fast_equivalent_scores(
            selected_tiles, embedding, device=device
        )
        if not _scores_bit_exact(reference_l1, identity_l1) or not _scores_bit_exact(
            reference_w4, identity_w4
        ):
            raise RuntimeError("fresh timed identity scores drifted from production")
        denoised_views = {"identity": selected_tiles}
        denoised_views.update(
            _restore_nonidentity_views(
                selected_model, raw_tiles, device, batch_size=args.denoise_batch_size
            )
        )
        view_scores = {"identity": identity_w4}
        for view in D4_VIEWS[1:]:
            _view_l1, view_scores[view] = _build_fast_equivalent_scores(
                denoised_views[view], embedding, device=device
            )
        consensus = d4_rank_consensus(view_scores)
        # The production l1 seed is deliberately unchanged; D4 changes only
        # the qap compatibility cost.
        candidate_layout = _solve_layout(identity_l1, consensus, name)
        seam_tiles = restore_tiles_uint8(
            seam_model, raw_tiles, seam_device, batch_size=args.denoise_batch_size
        )
        candidate_render, candidate_render_diag = _render_promoted(
            selected_tiles, seam_tiles, candidate_layout
        )
        candidate_seconds = time.perf_counter() - candidate_started

        baseline_started = time.perf_counter()
        baseline_layout = _solve_layout(reference_l1, reference_w4, name)
        baseline_render, baseline_render_diag = _render_promoted(
            selected_tiles, seam_tiles, baseline_layout
        )
        baseline_seconds = time.perf_counter() - baseline_started
        slow_record = slow_reference["records"][source_index]
        slow_fingerprint = {
            "raw_tiles_sha256": slow_record["raw_tiles_sha256"],
            "identity_l1_score_sha256": slow_record["identity_l1_score_sha256"],
            "identity_l1w4_score_sha256": slow_record[
                "identity_l1w4_score_sha256"
            ],
            "consensus_score_sha256": slow_record["consensus_score_sha256"],
            "baseline_layout_sha256": slow_record["baseline_layout_sha256"],
            "candidate_layout_sha256": slow_record["candidate_layout_sha256"],
            "baseline_render_sha256": slow_record["baseline_render"][
                "ordered_tiles_sha256"
            ],
            "candidate_render_sha256": slow_record["candidate_render"][
                "ordered_tiles_sha256"
            ],
        }
        current_fingerprint = {
            "raw_tiles_sha256": array_sha256(raw_tiles),
            "identity_l1_score_sha256": _score_hash(identity_l1),
            "identity_l1w4_score_sha256": _score_hash(identity_w4),
            "consensus_score_sha256": _score_hash(consensus),
            "baseline_layout_sha256": array_sha256(baseline_layout),
            "candidate_layout_sha256": array_sha256(candidate_layout),
            "baseline_render_sha256": baseline_render_diag["ordered_tiles_sha256"],
            "candidate_render_sha256": candidate_render_diag[
                "ordered_tiles_sha256"
            ],
        }
        if current_fingerprint != slow_fingerprint:
            raise RuntimeError(
                f"fast candidate drifted from sealed slow Phase A: {name}"
            )
        phase_a.append(
            {
                "name": name,
                "panel_seed": int(panel_seed),
                "raw_tiles": raw_tiles,
                "selected_tiles": selected_tiles,
                "seam_tiles": seam_tiles,
                "baseline_layout": baseline_layout,
                "candidate_layout": candidate_layout,
                "baseline_render": baseline_render,
                "candidate_render": candidate_render,
                "report": {
                    "raw_tiles_sha256": array_sha256(raw_tiles),
                    "identity_l1_score_sha256": _score_hash(identity_l1),
                    "identity_l1w4_score_sha256": _score_hash(identity_w4),
                    "identity_l1_matches_authoritative_bit_exact": identity_l1_exact,
                    "identity_l1w4_matches_authoritative_bit_exact": identity_w4_exact,
                    "consensus_score_sha256": _score_hash(consensus),
                    "baseline_layout_sha256": array_sha256(baseline_layout),
                    "candidate_layout_sha256": array_sha256(candidate_layout),
                    "baseline_render": baseline_render_diag,
                    "candidate_render": candidate_render_diag,
                    "identity_equivalence_audit_seconds": float(audit_seconds),
                    "baseline_seconds": float(baseline_seconds),
                    "candidate_seconds": float(candidate_seconds),
                    "layout_changed": bool(not np.array_equal(baseline_layout, candidate_layout)),
                    "valid_baseline_layout": True,
                    "valid_candidate_layout": True,
                    "d4_inverse_byte_exact": True,
                    "identity_fast_scores_bit_exact_to_authoritative": bool(
                        identity_l1_exact and identity_w4_exact
                    ),
                    "matches_sealed_slow_phase_a": True,
                    "candidate_uses_unchanged_identity_l1_seed": True,
                    "target_or_truth_passed_to_solver": False,
                },
            }
        )
        print(
            json.dumps(
                {
                    "stage": "phase_a_input_only",
                    "panel": args.panel,
                    "done": source_index + 1,
                    "total": len(names),
                    "candidate_seconds": candidate_seconds,
                }
            ),
            flush=True,
        )

    changed = sum(item["report"]["layout_changed"] for item in phase_a)
    max_runtime = max(item["report"]["candidate_seconds"] for item in phase_a)
    checks = {
        "all_d4_inverses_byte_exact": all(item["report"]["d4_inverse_byte_exact"] for item in phase_a),
        "all_identity_fast_scores_bit_exact_to_authoritative": all(
            item["report"]["identity_fast_scores_bit_exact_to_authoritative"]
            for item in phase_a
        ),
        "all_candidates_use_unchanged_identity_l1_seed": all(
            item["report"]["candidate_uses_unchanged_identity_l1_seed"] for item in phase_a
        ),
        "all_candidates_match_sealed_slow_phase_a": all(
            item["report"]["matches_sealed_slow_phase_a"] for item in phase_a
        ),
        "all_8_panel_local_layouts_valid": all(
            item["report"]["valid_baseline_layout"] and item["report"]["valid_candidate_layout"]
            for item in phase_a
        ),
        "max_candidate_input_to_render_seconds_le_20": max_runtime <= 20.0,
        "solver_target_blind": all(
            not item["report"]["target_or_truth_passed_to_solver"] for item in phase_a
        ),
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact_sha256 = _atomic_write_phase_a_artifact(artifact, phase_a)
    payload = {
        "schema_version": 1,
        "kind": "d4_compatibility_consensus_phase_a",
        "safe_for_submission": False,
        "panel": args.panel,
        "protocol": _protocol(names),
        "phase_a": {
            "checks": checks,
            "passed": all(checks.values()),
            "different_layouts": int(changed),
            "authoritative_score_checks_passed": int(
                sum(
                    item["report"]["identity_l1_matches_authoritative_bit_exact"]
                    + item["report"][
                        "identity_l1w4_matches_authoritative_bit_exact"
                    ]
                    for item in phase_a
                )
            ),
            "max_candidate_seconds": float(max_runtime),
            "target_metrics_opened": False,
        },
        "records": [item["report"] | {"name": item["name"], "panel_seed": item["panel_seed"]} for item in phase_a],
        "artifact": {"path": str(artifact), "sha256": artifact_sha256},
        "assets": actual_assets,
        "metadata": {
            "selected_denoiser": selected_metadata,
            "seam_denoiser": seam_metadata,
            "embedding": embedding_metadata,
        },
    }
    _assert_finite_payload(payload)
    _atomic_write_json(output, payload)


def _run_phase_b(
    args: argparse.Namespace,
    output: Path,
    artifact: Path,
    phase_a_report_path: Path,
    authorization_path: Path,
    actual_assets: dict[str, str],
    names: list[str],
) -> None:
    phase_a_report = json.loads(phase_a_report_path.read_text())
    if (
        phase_a_report.get("kind") != "d4_compatibility_consensus_phase_a"
        or phase_a_report.get("panel") != args.panel
        or not phase_a_report.get("phase_a", {}).get("passed")
        or phase_a_report.get("phase_a", {}).get("target_metrics_opened") is not False
    ):
        raise RuntimeError("invalid or failed Phase-A report")
    authorization = json.loads(authorization_path.read_text())
    panel_auth = authorization.get("panels", {}).get(args.panel, {})
    authorization_checks = authorization.get("checks", {})
    expected_authorization_checks = {
        "both_phase_a_pass",
        "different_layouts_ge_4_of_16",
        "authoritative_score_checks_eq_32",
        "all_phase_a_artifacts_hash_sealed",
        "no_target_metrics_opened",
    }
    if (
        authorization.get("kind") != "d4_global_phase_b_authorization"
        or authorization.get("authorized") is not True
        or authorization.get("source_names_sha256") != FROZEN_SOURCE_NAMES_SHA256
        or set(authorization_checks) != expected_authorization_checks
        or not all(authorization_checks.values())
        or set(authorization.get("panels", {})) != set(PANELS)
        or panel_auth.get("phase_a_report_sha256") != sha256(phase_a_report_path)
        or panel_auth.get("phase_a_artifact_sha256") != sha256(artifact)
    ):
        raise RuntimeError("Phase B lacks a valid global two-panel authorization")
    arrays = _load_phase_a_artifact(
        artifact,
        expected_sha256=phase_a_report["artifact"]["sha256"],
        expected_names=names,
    )
    records: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        clean = read_rgb(Path(args.data_root) / "train/targets" / name)
        panel_seed = int(arrays["panel_seeds"][index])
        exact = make_exact_panel(clean, panel=args.panel, seed=panel_seed)
        if array_sha256(exact.slot_tiles) != phase_a_report["records"][index]["raw_tiles_sha256"]:
            raise RuntimeError("Phase-A input changed before target attachment")
        baseline_geometry = layout_metrics(arrays["baseline_layout"][index], exact.slot_to_target)
        candidate_geometry = layout_metrics(arrays["candidate_layout"][index], exact.slot_to_target)
        baseline_quality = image_quality_metrics(
            arrays["baseline_render"][index], exact.clean_target_tiles
        )
        candidate_quality = image_quality_metrics(
            arrays["candidate_render"][index], exact.clean_target_tiles
        )
        records.append(
            {
                "name": name,
                "panel": args.panel,
                "panel_seed": panel_seed,
                "phase_a": phase_a_report["records"][index],
                "baseline": {
                    "combined_adjacency": baseline_geometry["combined_adjacency"],
                    "harmonized_ssim": baseline_quality["ssim"],
                },
                "candidate": {
                    "combined_adjacency": candidate_geometry["combined_adjacency"],
                    "harmonized_ssim": candidate_quality["ssim"],
                },
                "delta": {
                    "combined_adjacency": candidate_geometry["combined_adjacency"]
                    - baseline_geometry["combined_adjacency"],
                    "harmonized_ssim": candidate_quality["ssim"] - baseline_quality["ssim"],
                },
            }
        )
    summary = _panel_summary(records)
    gate = _panel_gate(summary)
    payload = {
        "schema_version": 1,
        "kind": "d4_compatibility_consensus_exact_panel_gate",
        "safe_for_submission": False,
        "panel": args.panel,
        "protocol": _protocol(names),
        "phase_a": phase_a_report["phase_a"],
        "phase_b_authorization_sha256": sha256(authorization_path),
        "records": records,
        "summary": summary,
        "gate": gate,
        "assets": actual_assets,
    }
    _assert_finite_payload(payload)
    _atomic_write_json(output, payload)
    print(json.dumps({"panel": args.panel, "gate": gate, "summary": summary}, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    actual_assets, names = _resolve_protocol(args)
    if not args.phase_a_artifact:
        raise SystemExit("--phase-a-artifact is required")
    artifact = Path(args.phase_a_artifact)
    if args.phase == "phase-a":
        if args.phase_a_report or args.phase_b_authorization:
            raise SystemExit("Phase A must not receive Phase-B inputs")
        _run_phase_a(args, output, artifact, actual_assets, names)
        return
    if not args.phase_a_report or not args.phase_b_authorization:
        raise SystemExit("Phase B requires --phase-a-report and --phase-b-authorization")
    _run_phase_b(
        args,
        output,
        artifact,
        Path(args.phase_a_report),
        Path(args.phase_b_authorization),
        actual_assets,
        names,
    )


if __name__ == "__main__":
    main()
