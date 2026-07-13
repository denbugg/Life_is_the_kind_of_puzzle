#!/usr/bin/env python3
"""Build a deterministic 700-image assembly submission with the promoted pipeline."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image

from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.components import (
    ComponentSolveResult,
    reciprocal_component_solver,
    soft_cycle_component_solver,
)
from puzzle_assembly.learned import learned_compatibility, load_embedding_checkpoint
from puzzle_assembly.line_seam import line_seam_compatibility
from puzzle_assembly.qap import DirectionalQAPResult, directional_qap
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import merge_tiles_numpy, split_tiles_numpy


DEFAULT_DENOISER = "runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
ARCHIVE_TIMESTAMP = (2026, 7, 10, 0, 0, 0)
QAP_SCORE_CHOICES = (
    "denoised_c1",
    "cross_c1",
    "l1",
    "l1w4",
    "cross_l1w4",
    "line",
    "l1w4line",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="puzzle/test")
    parser.add_argument("--denoiser", default=DEFAULT_DENOISER)
    parser.add_argument(
        "--embedding-checkpoint",
        help=(
            "optional HBT side-embedding checkpoint; supplying it activates the "
            "promoted soft-cycle + directional-QAP pipeline unless "
            "--qap-iterations=0"
        ),
    )
    parser.add_argument(
        "--renderer-denoiser",
        help="optional separate denoiser used only for final pixels after layout is frozen",
    )
    parser.add_argument(
        "--line-seam",
        action="store_true",
        help="add the optional raw+denoised structural line-seam score",
    )
    parser.add_argument("--line-seam-auxiliary-weight", type=float, default=0.35)
    parser.add_argument("--line-seam-fusion-weight", type=float, default=0.5)
    parser.add_argument("--soft-cycle-score", choices=QAP_SCORE_CHOICES, default="l1")
    parser.add_argument("--soft-cycle-topk", type=int, default=8)
    parser.add_argument("--soft-cycle-keep-per-tile", type=int, default=1)
    parser.add_argument("--soft-cycle-keep-fraction", type=float, default=0.5)
    parser.add_argument("--soft-cycle-loop-weight", type=float, default=1.0)
    parser.add_argument("--soft-cycle-reciprocal-weight", type=float, default=0.35)
    parser.add_argument("--qap-score", choices=QAP_SCORE_CHOICES, default="l1w4")
    parser.add_argument(
        "--qap-iterations",
        type=int,
        default=25,
        help="Frank-Wolfe iterations; zero keeps the legacy solver",
    )
    parser.add_argument("--qap-restarts", type=int, default=2)
    parser.add_argument("--qap-initial-weight", type=float, default=0.75)
    parser.add_argument("--qap-noisy-components", type=int, default=3)
    parser.add_argument("--qap-noise-scale", type=float, default=1.0)
    parser.add_argument("--qap-boundary-weight", type=float, default=0.0)
    parser.add_argument("--qap-refine-swaps", type=int, default=8)
    parser.add_argument("--qap-refine-weak-cells", type=int, default=32)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--reference-zip")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-count", type=int, default=700)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise ValueError(f"unexpected image shape for {path}: {values.shape}")
    return values


def _reference_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    if any(Path(name).name != name for name in names):
        raise ValueError("reference zip contains nested paths")
    return sorted(names)


def _png_bytes(values: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(values, mode="RGB").save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def _filename_seed(name: str) -> int:
    """Return the same stable per-source seed on every Python/platform run."""
    return int.from_bytes(
        hashlib.sha256(name.encode("utf-8")).digest()[:4], "little"
    )


def _layout_sha256(layout: np.ndarray) -> str:
    values = np.asarray(layout, dtype=np.int32)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _c1_fusion(
    bank: dict[str, CompatibilityMatrices], prefix: str
) -> CompatibilityMatrices:
    names = [
        name
        for name in sorted(bank)
        if name.startswith(f"{prefix}_") and not name.endswith("_c2")
    ]
    return fuse_ranked_scores(
        bank,
        names=names,
        name=f"{prefix}_C1_equal_rank_fusion",
    )


def _promoted_score_bank(
    raw_tiles: np.ndarray,
    denoised_tiles: np.ndarray,
    *,
    embedding_model: object,
    device: object,
    chunk_size: int,
    include_line_seam: bool,
    line_seam_auxiliary_weight: float,
    line_seam_fusion_weight: float,
) -> tuple[dict[str, CompatibilityMatrices], dict[str, str]]:
    """Build only input-derived scores used by the promoted submission path."""
    bank = build_classical_score_bank(raw_tiles, prefix="raw", chunk_size=chunk_size)
    bank.update(
        build_classical_score_bank(
            denoised_tiles, prefix="denoised", chunk_size=chunk_size
        )
    )
    raw_c1 = _c1_fusion(bank, "raw")
    denoised_c1 = _c1_fusion(bank, "denoised")
    bank[raw_c1.name] = raw_c1
    bank[denoised_c1.name] = denoised_c1

    cross_c1 = fuse_ranked_scores(
        bank,
        names=[raw_c1.name, denoised_c1.name],
        weights={denoised_c1.name: 2.0},
        name="raw_denoised_C1_dn2_rank_fusion",
    )
    bank[cross_c1.name] = cross_c1

    l1, _outside_logits = learned_compatibility(
        embedding_model,
        denoised_tiles,
        device=device,
        name="denoised_l1_embedding",
    )
    bank[l1.name] = l1
    l1w4 = fuse_ranked_scores(
        bank,
        names=[denoised_c1.name, l1.name],
        weights={l1.name: 4.0},
        name="denoised_C1_L1w4_rank_fusion",
    )
    cross_l1w4 = fuse_ranked_scores(
        bank,
        names=[cross_c1.name, l1.name],
        weights={l1.name: 4.0},
        name="raw_denoised_C1_L1w4_rank_fusion",
    )
    bank[l1w4.name] = l1w4
    bank[cross_l1w4.name] = cross_l1w4
    aliases = {
        "denoised_c1": denoised_c1.name,
        "cross_c1": cross_c1.name,
        "l1": l1.name,
        "l1w4": l1w4.name,
        "cross_l1w4": cross_l1w4.name,
    }

    if include_line_seam:
        line = line_seam_compatibility(
            denoised_tiles,
            prefix="denoised",
            auxiliary_tiles=raw_tiles,
            auxiliary_prefix="raw",
            auxiliary_weight=line_seam_auxiliary_weight,
        )
        bank[line.name] = line
        l1w4line = fuse_ranked_scores(
            bank,
            names=[l1w4.name, line.name],
            weights={line.name: line_seam_fusion_weight},
            name="denoised_C1_L1w4_line_rank_fusion",
        )
        bank[l1w4line.name] = l1w4line
        aliases["line"] = line.name
        aliases["l1w4line"] = l1w4line.name
    return bank, aliases


def _layout_cost_diagnostics(
    layout: np.ndarray, score: CompatibilityMatrices
) -> dict[str, object]:
    grid = np.asarray(layout, dtype=np.int32).reshape(24, 24)
    values = np.concatenate(
        [
            score.right[grid[:, :-1], grid[:, 1:]].ravel(),
            score.down[grid[:-1, :], grid[1:, :]].ravel(),
        ]
    )
    values = values[np.isfinite(values)].astype(np.float64, copy=False)
    if not len(values):
        return {
            "score": score.name,
            "finite_edges": 0,
            "mean": None,
            "sum": None,
            "median": None,
            "p90": None,
        }
    return {
        "score": score.name,
        "finite_edges": int(len(values)),
        "mean": float(np.mean(values)),
        "sum": float(np.sum(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
    }


def _component_diagnostics(result: ComponentSolveResult) -> dict[str, object]:
    return {
        "accepted_edges": int(result.accepted_edges),
        "proposed_edges": int(result.proposed_edges),
        "component_sizes": [int(value) for value in result.component_sizes],
        "placed_component_tiles": int(result.placed_component_tiles),
        "unresolved_tiles_before_assignment": int(
            result.unresolved_tiles_before_assignment
        ),
        "consensus_added_tiles": int(result.consensus_added_tiles),
    }


def _qap_diagnostics(result: DirectionalQAPResult) -> dict[str, object]:
    history = [float(value) for value in result.history]
    return {
        "objective": float(result.objective),
        "relaxed_objective": float(result.relaxed_objective),
        "restart": int(result.restart),
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
        "history": history,
        "history_start": history[0] if history else None,
        "history_end": history[-1] if history else None,
        "history_delta": history[-1] - history[0] if len(history) >= 2 else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.offset < 0 or args.expected_count <= 0:
        raise SystemExit("offset must be non-negative and expected-count positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("limit must be positive")
    if args.batch_size <= 0 or args.chunk_size <= 0:
        raise SystemExit("batch-size and chunk-size must be positive")
    if args.qap_iterations < 0:
        raise SystemExit("qap-iterations must be non-negative")
    if args.soft_cycle_topk <= 0 or args.soft_cycle_keep_per_tile <= 0:
        raise SystemExit("soft-cycle-topk and keep-per-tile must be positive")
    if not 0.0 < args.soft_cycle_keep_fraction <= 1.0:
        raise SystemExit("soft-cycle-keep-fraction must lie in (0, 1]")
    if not all(
        np.isfinite(value) and value >= 0.0
        for value in (
            args.soft_cycle_loop_weight,
            args.soft_cycle_reciprocal_weight,
        )
    ):
        raise SystemExit("soft-cycle weights must be non-negative")
    if not (
        np.isfinite(args.line_seam_auxiliary_weight)
        and 0.0 <= args.line_seam_auxiliary_weight <= 1.0
    ):
        raise SystemExit("line-seam-auxiliary-weight must lie in [0, 1]")
    if not (
        np.isfinite(args.line_seam_fusion_weight)
        and args.line_seam_fusion_weight >= 0.0
    ):
        raise SystemExit("line-seam-fusion-weight must be non-negative")
    embedding_path = (
        Path(args.embedding_checkpoint) if args.embedding_checkpoint else None
    )
    qap_enabled = embedding_path is not None and args.qap_iterations > 0
    if args.line_seam and not qap_enabled:
        raise SystemExit(
            "--line-seam is part of the promoted QAP path; provide "
            "--embedding-checkpoint and use positive --qap-iterations"
        )
    if qap_enabled:
        if args.qap_restarts <= 0 or args.qap_noisy_components <= 0:
            raise SystemExit("qap-restarts and qap-noisy-components must be positive")
        if not (
            np.isfinite(args.qap_initial_weight)
            and 0.0 <= args.qap_initial_weight <= 1.0
        ):
            raise SystemExit("qap-initial-weight must lie in [0, 1]")
        if not all(
            np.isfinite(value) and value >= 0.0
            for value in (args.qap_noise_scale, args.qap_boundary_weight)
        ):
            raise SystemExit("qap noise and boundary weights must be non-negative")
        if args.qap_refine_swaps < 0:
            raise SystemExit("qap-refine-swaps must be non-negative")
        if args.qap_refine_swaps > 0 and args.qap_refine_weak_cells < 2:
            raise SystemExit("qap-refine-weak-cells must be at least 2")
        unavailable = {args.soft_cycle_score, args.qap_score} & {
            "line",
            "l1w4line",
        }
        if unavailable and not args.line_seam:
            raise SystemExit(
                f"scores {sorted(unavailable)} require --line-seam"
            )
    output = Path(args.output)
    report = Path(args.report) if args.report else output.with_suffix(".json")
    if output.resolve() == report.resolve():
        raise SystemExit("output zip and report must differ")
    if not args.overwrite and (output.exists() or report.exists()):
        raise SystemExit(f"output exists; pass --overwrite: {output} or {report}")
    input_dir = Path(args.input_dir)
    all_paths = sorted(input_dir.glob("*.png"))
    paths = all_paths[args.offset :]
    if args.limit is not None:
        paths = paths[: args.limit]
    if len(paths) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} selected PNGs, found {len(paths)}"
        )
    names = [path.name for path in paths]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate input filenames")
    reference_validation = None
    if args.reference_zip:
        reference = _reference_names(Path(args.reference_zip))
        reference_shard = reference[args.offset :]
        if args.limit is not None:
            reference_shard = reference_shard[: args.limit]
        if sorted(names) == reference:
            reference_validation = "exact_member_set"
        elif sorted(names) == reference_shard:
            reference_validation = "offset_limit_slice_of_full_member_set"
        else:
            raise SystemExit("selected filenames differ from the reference submission")

    denoiser_path = Path(args.denoiser)
    restorer, device, denoiser_metadata = load_restorer(
        denoiser_path, device=args.device, state="ema"
    )
    embedding_model = None
    embedding_metadata = None
    embedding_config = None
    if embedding_path is not None:
        embedding_model, embedding_metadata = load_embedding_checkpoint(
            embedding_path, device=device
        )
        model_config = getattr(embedding_model, "config", None)
        if callable(model_config):
            embedding_config = model_config()
    renderer_path = Path(args.renderer_denoiser) if args.renderer_denoiser else denoiser_path
    if renderer_path.resolve() == denoiser_path.resolve():
        renderer = restorer
        renderer_metadata = denoiser_metadata
    else:
        renderer, renderer_device, renderer_metadata = load_restorer(
            renderer_path, device=device, state="ema"
        )
        if renderer_device != device:
            raise RuntimeError("scoring and rendering denoisers resolved to different devices")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()

    source_records: list[dict[str, object]] = []
    promoted_alias_names: dict[str, str] | None = None
    started = time.perf_counter()
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for index, path in enumerate(paths):
                source_started = time.perf_counter()
                input_image = _read_rgb(path)
                input_pixel_sha256 = hashlib.sha256(input_image.tobytes()).hexdigest()
                raw_tiles = split_tiles_numpy(input_image)
                denoise_started = time.perf_counter()
                denoised = restore_tiles_uint8(
                    restorer, raw_tiles, device, batch_size=args.batch_size
                )
                denoise_seconds = time.perf_counter() - denoise_started
                score_started = time.perf_counter()
                if qap_enabled:
                    assert embedding_model is not None
                    bank, aliases = _promoted_score_bank(
                        raw_tiles,
                        denoised,
                        embedding_model=embedding_model,
                        device=device,
                        chunk_size=args.chunk_size,
                        include_line_seam=args.line_seam,
                        line_seam_auxiliary_weight=args.line_seam_auxiliary_weight,
                        line_seam_fusion_weight=args.line_seam_fusion_weight,
                    )
                    promoted_alias_names = aliases
                    seed_score = bank[aliases[args.soft_cycle_score]]
                    qap_score = bank[aliases[args.qap_score]]
                    score_seconds = time.perf_counter() - score_started

                    seed_started = time.perf_counter()
                    seed_result = soft_cycle_component_solver(
                        seed_score,
                        top_k=args.soft_cycle_topk,
                        keep_per_tile=args.soft_cycle_keep_per_tile,
                        proposal_keep_fraction=args.soft_cycle_keep_fraction,
                        loop_weight=args.soft_cycle_loop_weight,
                        reciprocal_weight=args.soft_cycle_reciprocal_weight,
                    )
                    seed_seconds = time.perf_counter() - seed_started
                    filename_seed = _filename_seed(path.name)
                    qap_seed = filename_seed + 7001
                    qap_started = time.perf_counter()
                    qap_result = directional_qap(
                        qap_score,
                        initial=seed_result.position_to_slot,
                        iterations=args.qap_iterations,
                        restarts=args.qap_restarts,
                        seed=qap_seed,
                        boundary_weight=args.qap_boundary_weight,
                        initial_weight=args.qap_initial_weight,
                        noisy_components=args.qap_noisy_components,
                        noise_scale=args.qap_noise_scale,
                        refine_swaps=args.qap_refine_swaps,
                        refine_weak_cells=args.qap_refine_weak_cells,
                    )
                    qap_seconds = time.perf_counter() - qap_started
                    layout = qap_result.position_to_slot
                    solver_seconds = seed_seconds + qap_seconds
                    objective_diagnostics = {
                        "qap_score_seed": _layout_cost_diagnostics(
                            seed_result.position_to_slot, qap_score
                        ),
                        "qap_score_final": _layout_cost_diagnostics(layout, qap_score),
                        "final_layout_by_alias": {
                            alias: _layout_cost_diagnostics(layout, bank[score_name])
                            for alias, score_name in sorted(aliases.items())
                        },
                    }
                    solver_details: dict[str, object] = {
                        "filename_seed": filename_seed,
                        "qap_seed": qap_seed,
                        "qap_seed_formula": "filename_sha256_first4_le + 7001",
                        "soft_cycle_score_alias": args.soft_cycle_score,
                        "soft_cycle_score_name": seed_score.name,
                        "soft_cycle_layout_sha256": _layout_sha256(
                            seed_result.position_to_slot
                        ),
                        "soft_cycle": _component_diagnostics(seed_result),
                        "qap_score_alias": args.qap_score,
                        "qap_score_name": qap_score.name,
                        "qap": _qap_diagnostics(qap_result),
                        "objective_diagnostics": objective_diagnostics,
                        "soft_cycle_seconds": seed_seconds,
                        "qap_seconds": qap_seconds,
                    }
                else:
                    bank = build_classical_score_bank(
                        denoised, prefix="denoised", chunk_size=args.chunk_size
                    )
                    fusion = _c1_fusion(bank, "denoised")
                    score_seconds = time.perf_counter() - score_started
                    solver_started = time.perf_counter()
                    legacy_result = reciprocal_component_solver(
                        fusion, include_verified_loops=True, refine=False
                    )
                    layout = legacy_result.position_to_slot
                    solver_seconds = time.perf_counter() - solver_started
                    solver_details = {
                        "legacy_component": _component_diagnostics(legacy_result),
                        "objective_diagnostics": {
                            "selected_layout": _layout_cost_diagnostics(layout, fusion)
                        },
                    }
                layout = np.asarray(layout, dtype=np.int32)
                render_started = time.perf_counter()
                render_tiles = (
                    denoised
                    if renderer is restorer
                    else restore_tiles_uint8(
                        renderer, raw_tiles, device, batch_size=args.batch_size
                    )
                )
                restored = merge_tiles_numpy(render_tiles[layout])
                if restored.shape != (480, 480, 3) or restored.dtype != np.uint8:
                    raise RuntimeError(f"invalid output image for {path.name}")
                png_payload = _png_bytes(restored)
                render_seconds = time.perf_counter() - render_started
                info = zipfile.ZipInfo(path.name, date_time=ARCHIVE_TIMESTAMP)
                info.create_system = 3
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, png_payload, compresslevel=6)
                record = {
                    "source": path.name,
                    "input_pixel_sha256": input_pixel_sha256,
                    "output_png_sha256": hashlib.sha256(png_payload).hexdigest(),
                    "layout_sha256": _layout_sha256(layout),
                    "position_to_slot": layout.tolist(),
                    "pipeline": "directional_qap" if qap_enabled else "legacy",
                    "denoise_seconds": denoise_seconds,
                    "score_seconds": score_seconds,
                    "solver_seconds": solver_seconds,
                    "render_seconds": render_seconds,
                    **solver_details,
                    "total_seconds": time.perf_counter() - source_started,
                }
                source_records.append(record)
                print(
                    json.dumps(
                        {
                            "event": "submission_source_complete",
                            "index": index + 1,
                            "count": len(paths),
                            "source": path.name,
                            "pipeline": record["pipeline"],
                            "layout_sha256": record["layout_sha256"],
                            "seconds": record["total_seconds"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        with zipfile.ZipFile(temporary) as archive:
            corrupt = archive.testzip()
            archive_names = archive.namelist()
            if corrupt is not None:
                raise RuntimeError(f"corrupt zip member: {corrupt}")
            if archive_names != names:
                raise RuntimeError("final zip member order differs from selected inputs")
            for name in archive_names:
                if Path(name).name != name:
                    raise RuntimeError("final zip contains nested paths")
                if archive.getinfo(name).date_time != ARCHIVE_TIMESTAMP:
                    raise RuntimeError(f"non-reproducible zip timestamp for {name}")
                with archive.open(name) as handle, Image.open(handle) as image:
                    if image.mode != "RGB" or image.size != (480, 480):
                        raise RuntimeError(f"invalid archived PNG: {name}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    denoiser_sha256 = _sha256(denoiser_path)
    renderer_sha256 = (
        denoiser_sha256
        if renderer_path.resolve() == denoiser_path.resolve()
        else _sha256(renderer_path)
    )
    embedding_sha256 = _sha256(embedding_path) if embedding_path is not None else None
    pipeline = {
        "mode": "promoted_directional_qap" if qap_enabled else "legacy",
        "scoring_view": (
            "raw_plus_selected_tilenaf_denoised"
            if qap_enabled
            else "selected_tilenaf_denoised"
        ),
        "score": (
            promoted_alias_names[args.qap_score]
            if qap_enabled and promoted_alias_names is not None
            else "denoised_C1_equal_rank_fusion"
        ),
        "score_alias": args.qap_score if qap_enabled else "denoised_c1",
        "seed_solver": (
            "soft_cycle_component_solver" if qap_enabled else None
        ),
        "solver": (
            "directional_qap"
            if qap_enabled
            else "reciprocal_component_loops_no_refine"
        ),
        "render_view": (
            "selected_tilenaf_denoised"
            if renderer is restorer
            else "separate_renderer_denoised"
        ),
    }
    payload = {
        "schema_version": 2,
        "kind": "assembly_v1_submission_report",
        "pipeline": pipeline,
        "configuration": {
            "batch_size": args.batch_size,
            "chunk_size": args.chunk_size,
            "line_seam": {
                "enabled": args.line_seam,
                "auxiliary_weight": args.line_seam_auxiliary_weight,
                "fusion_weight": args.line_seam_fusion_weight,
            },
            "soft_cycle": {
                "score": args.soft_cycle_score,
                "top_k": args.soft_cycle_topk,
                "keep_per_tile": args.soft_cycle_keep_per_tile,
                "keep_fraction": args.soft_cycle_keep_fraction,
                "loop_weight": args.soft_cycle_loop_weight,
                "reciprocal_weight": args.soft_cycle_reciprocal_weight,
            },
            "qap": {
                "enabled": qap_enabled,
                "score": args.qap_score,
                "iterations": args.qap_iterations,
                "restarts": args.qap_restarts,
                "initial_weight": args.qap_initial_weight,
                "noisy_components": args.qap_noisy_components,
                "noise_scale": args.qap_noise_scale,
                "boundary_weight": args.qap_boundary_weight,
                "refine_swaps": args.qap_refine_swaps,
                "refine_weak_cells": args.qap_refine_weak_cells,
                "seed_formula": "filename_sha256_first4_le + 7001",
            },
        },
        "anti_leakage": {
            "target_paths_or_pixels_read": False,
            "layout_inputs": [
                "selected test input PNG pixels",
                "denoiser checkpoint",
                *(["embedding checkpoint"] if embedding_path is not None else []),
            ],
            "reference_zip_usage": "member_names_only" if args.reference_zip else None,
        },
        "input_dir": str(input_dir),
        "offset": args.offset,
        "limit": args.limit,
        "expected_count": args.expected_count,
        "available_input_count": len(all_paths),
        "count": len(paths),
        "source_names": names,
        "reference_zip": str(args.reference_zip) if args.reference_zip else None,
        "reference_validation": reference_validation,
        "denoiser": denoiser_metadata,
        "denoiser_checkpoint": str(denoiser_path),
        "denoiser_checkpoint_sha256": denoiser_sha256,
        "embedding_metadata": embedding_metadata,
        "embedding_model_config": embedding_config,
        "embedding_model_type": (
            type(embedding_model).__name__ if embedding_model is not None else None
        ),
        "embedding_checkpoint": str(embedding_path) if embedding_path is not None else None,
        "embedding_checkpoint_sha256": embedding_sha256,
        "renderer_denoiser": renderer_metadata,
        "renderer_denoiser_checkpoint": str(renderer_path),
        "renderer_denoiser_checkpoint_sha256": renderer_sha256,
        "device": str(device),
        "archive": {
            "member_order": names,
            "member_timestamp": list(ARCHIVE_TIMESTAMP),
            "compression": "ZIP_DEFLATED",
            "compresslevel": 6,
            "flat_member_names": True,
            "unix_mode": "100644",
        },
        "output": str(output),
        "output_sha256": _sha256(output),
        "output_bytes": output.stat().st_size,
        "sources": source_records,
        "seconds": time.perf_counter() - started,
    }
    report.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "submission_complete",
                "output": str(output),
                "sha256": payload["output_sha256"],
                "seconds": payload["seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
