#!/usr/bin/env python3
"""Leakage-safe fixed confirmation of QAP HBT weight 1 versus weight 4.

The evaluator is deliberately split into three separate actions.  ``phase-a``
can run in isolated input-only GPU containers, ``finalize-phase-a`` merges and
independently validates the shards, and ``phase-b`` is the only action that may
construct a target path.  Phase B requires an out-of-band SHA256 anchor for the
canonical Phase-A manifest and fails closed on any integrity drift.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from puzzle_assembly.compatibility import (
    CompatibilityMatrices,
    build_classical_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.components import soft_cycle_component_solver
from puzzle_assembly.geometry import TILE_COUNT, validate_permutation
from puzzle_assembly.learned import learned_compatibility, load_embedding_checkpoint
from puzzle_assembly.protocol import source_names_for_split
from puzzle_assembly.qap import directional_qap
from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8
from puzzle_denoise_v2.tiles import merge_tiles_numpy, split_tiles_numpy


EXPECTED_CONFIG_SHA256 = (
    "30732463fb200bdff8f909ef06be6cb6c4e7859692e01c9d33c5d55175ffe262"
)
EXPECTED_NAMES_SHA256 = (
    "e5fb7fc6b3d24e9c080b4f33224b863c181e72452de4e54e602a80a321c13251"
)
SHARD_MANIFEST = "FROZEN_INPUT_ONLY_SHARD_MANIFEST.json"
FINAL_MANIFEST = "FROZEN_INPUT_ONLY_MANIFEST.json"
TARGET_MARKER = "TARGET_ACCESS_STARTED.json"
BASELINE_KEY = "baseline"
CANDIDATE_KEY = "candidate"


@dataclass(frozen=True)
class PhaseAPrediction:
    layouts: dict[str, np.ndarray]
    renders: dict[str, np.ndarray]
    initial_layout: np.ndarray
    qap_seed: int
    denoised_tiles_sha256: str
    diagnostics: dict[str, dict[str, Any]]


Predictor = Callable[[str, np.ndarray], PhaseAPrediction]
TargetReader = Callable[[Path], np.ndarray]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        required=True,
        choices=("phase-a", "finalize-phase-a", "phase-b"),
    )
    parser.add_argument("--config", default="configs/qap_weight_confirmation_v1.json")
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser")
    parser.add_argument("--hbt-checkpoint")
    parser.add_argument("--manifest")
    parser.add_argument("--quarantine")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--classical-chunk-size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--phase-a-dir")
    parser.add_argument("--phase-a-dirs", nargs="+")
    parser.add_argument("--phase-a-envelope-sha256s", nargs="+")
    parser.add_argument("--finalized-phase-a-dir")
    parser.add_argument("--phase-a-envelope-sha256")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _names_sha256(names: Sequence[str]) -> str:
    # This protocol intentionally has no trailing newline.
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return _bytes_sha256(_canonical_bytes(payload))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_envelope(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    envelope = {"payload": payload, "payload_sha256": _canonical_sha256(payload)}
    _atomic_bytes(path, _canonical_bytes(envelope) + b"\n")
    return envelope


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _load_exact_envelope(path: Path, expected_file_sha256: str | None = None) -> dict[str, Any]:
    if expected_file_sha256 is not None and _sha256(path) != expected_file_sha256:
        raise RuntimeError(f"Phase-A manifest SHA256 anchor mismatch: {path}")
    envelope = _load_json(path)
    if set(envelope) != {"payload", "payload_sha256"}:
        raise RuntimeError(f"non-canonical manifest envelope keys: {path}")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"manifest payload is not an object: {path}")
    if envelope.get("payload_sha256") != _canonical_sha256(payload):
        raise RuntimeError(f"manifest payload hash mismatch: {path}")
    if path.read_bytes() != _canonical_bytes(envelope) + b"\n":
        raise RuntimeError(f"manifest file is not canonical JSON: {path}")
    return envelope


def _read_rgb_bytes(path: Path) -> tuple[bytes, np.ndarray]:
    payload = path.read_bytes()
    with Image.open(BytesIO(payload)) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise RuntimeError(f"unexpected RGB image shape for {path}: {values.shape}")
    return payload, values


def _read_rgb(path: Path) -> np.ndarray:
    return _read_rgb_bytes(path)[1]


def _png_bytes(values: np.ndarray) -> bytes:
    values = np.asarray(values)
    if values.dtype != np.uint8 or values.shape != (480, 480, 3):
        raise RuntimeError("frozen render must be uint8 RGB 480x480")
    output = BytesIO()
    Image.fromarray(values, mode="RGB").save(output, format="PNG", compress_level=6)
    return output.getvalue()


def _npy_bytes(values: np.ndarray) -> bytes:
    output = BytesIO()
    np.save(output, values, allow_pickle=False)
    return output.getvalue()


def _decode_frozen_png(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise RuntimeError(f"frozen render has unexpected shape {values.shape}")
    return values


def _decode_layout(payload: bytes, *, name: str) -> np.ndarray:
    layout = np.load(BytesIO(payload), allow_pickle=False)
    return validate_permutation(layout, name=name)


def _resolve_asset_path(config_path: Path, configured: str, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    path = Path(configured).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.resolve().parent.parent / path).resolve()


def _validated_protocol_and_assets(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    config_path = Path(args.config).expanduser().resolve()
    if _sha256(config_path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("fixed confirmation config SHA256 mismatch")
    protocol = _load_json(config_path)
    if protocol.get("kind") != "fixed_qap_weight_confirmation":
        raise RuntimeError("unexpected confirmation config kind")
    if protocol.get("safe_for_submission_before_gate") is not False:
        raise RuntimeError("confirmation config is not fail-closed")
    _validate_fixed_protocol(protocol)
    assets = protocol.get("assets")
    if not isinstance(assets, dict):
        raise RuntimeError("confirmation config has no assets")
    specifications = {
        "denoiser": ("denoiser", "denoiser_sha256", args.denoiser),
        "hbt": ("hbt", "hbt_sha256", args.hbt_checkpoint),
        "manifest": ("manifest", "manifest_sha256", args.manifest),
        "quarantine": ("quarantine", "quarantine_sha256", args.quarantine),
    }
    resolved: dict[str, dict[str, str]] = {}
    for label, (path_key, hash_key, override) in specifications.items():
        expected = str(assets.get(hash_key, ""))
        path = _resolve_asset_path(config_path, str(assets.get(path_key, "")), override)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"pinned {label} SHA256 mismatch")
        resolved[label] = {
            "path": str(path),
            "sha256": actual,
            "configured_path": str(assets[path_key]),
        }
    return protocol, resolved


def _portable_assets(assets: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Checkpoint/config bindings that survive relocation of the frozen tree."""

    return {
        key: {
            "sha256": str(value["sha256"]),
            "configured_path": str(value["configured_path"]),
        }
        for key, value in sorted(assets.items())
    }


def _validate_fixed_protocol(protocol: dict[str, Any]) -> None:
    """Assert every scientific degree of freedom in the pinned A/B protocol."""

    confirmation = protocol.get("original_real_confirmation")
    if not isinstance(confirmation, dict):
        raise RuntimeError("confirmation config has no original-real gate")
    expected_confirmation = {
        "split": "assembly_incremental_gate",
        "offset": 128,
        "count": 64,
        "names_sha256": EXPECTED_NAMES_SHA256,
    }
    for key, expected in expected_confirmation.items():
        if confirmation.get(key) != expected:
            raise RuntimeError(f"fixed confirmation field drift: {key}")
    common = protocol.get("common_solver")
    expected_common = {
        "renderer": "selected_tilenaf_synth_50k_ema_uint8",
        "seed_score": "denoised_hbt_l1",
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
        "qap_seed_formula": "sha256(filename)[:4]_little + 7001",
    }
    if common != expected_common:
        raise RuntimeError("fixed common solver contract drift")
    expected_variants = {
        BASELINE_KEY: {
            "label": "qap_w4_b0.05_i25",
            "score": "denoised_C1_HBTw4_rank_fusion",
            "hbt_weight": 4.0,
        },
        CANDIDATE_KEY: {
            "label": "qap_w1_b0.05_i25",
            "score": "denoised_C1_HBTw1_rank_fusion",
            "hbt_weight": 1.0,
        },
    }
    for key, expected in expected_variants.items():
        if protocol.get(key) != expected:
            raise RuntimeError(f"fixed {key} solver contract drift")
    metric = confirmation.get("metric")
    expected_metric = {
        "name": "RGB_SSIM",
        "call": "skimage.metrics.structural_similarity(target_rgb_uint8, frozen_render_rgb_uint8, channel_axis=2, data_range=255)",
        "expected_shape": [480, 480, 3],
        "bootstrap_unit": "paired_whole_source_delta_candidate_minus_baseline",
        "bootstrap_method": "two_sided_percentile",
        "bootstrap_quantiles": [0.025, 0.975],
        "bootstrap_resamples": 20000,
        "bootstrap_seed": 20260711,
        "tie_tolerance": 1e-12,
        "win_definition": "delta > tie_tolerance",
        "tie_definition": "abs(delta) <= tie_tolerance",
        "loss_definition": "delta < -tie_tolerance",
        "large_regression_definition": "delta < -0.01",
    }
    if metric != expected_metric:
        raise RuntimeError("fixed metric/bootstrap contract drift")
    gate = confirmation.get("gate")
    expected_gate = {
        "logic": "all_of",
        "mean_ssim_delta_min": 0.005,
        "mean_ssim_delta_operator": ">=",
        "ssim_bootstrap_95_lower_gt": 0.0,
        "ssim_bootstrap_95_lower_operator": ">",
        "wins_min": 40,
        "wins_operator": ">=",
        "large_regressions_max": 6,
        "large_regressions_operator": "<=",
        "valid_permutation_count": 64,
        "valid_permutation_operator": "==",
    }
    if gate != expected_gate:
        raise RuntimeError("fixed promotion gate contract drift")


def _expected_names(protocol: dict[str, Any], assets: dict[str, dict[str, str]]) -> list[str]:
    confirmation = protocol["original_real_confirmation"]
    split = str(confirmation["split"])
    offset = int(confirmation["offset"])
    count = int(confirmation["count"])
    all_names = source_names_for_split(
        split,
        manifest_path=assets["manifest"]["path"],
        quarantine_path=assets["quarantine"]["path"],
    )
    names = all_names[offset : offset + count]
    if len(names) != 64 or count != 64 or _names_sha256(names) != EXPECTED_NAMES_SHA256:
        raise RuntimeError("authoritative confirmation source slice/hash mismatch")
    if confirmation.get("names_sha256") != EXPECTED_NAMES_SHA256:
        raise RuntimeError("config confirmation names hash mismatch")
    return names


def _filename_qap_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little") + 7001


def _layout_sha256(layout: np.ndarray) -> str:
    layout = validate_permutation(layout)
    return hashlib.sha256(layout.astype(np.int32, copy=False).tobytes()).hexdigest()


def _build_default_predictor(
    protocol: dict[str, Any], assets: dict[str, dict[str, str]], args: argparse.Namespace
) -> Predictor:
    restorer, device, _ = load_restorer(
        assets["denoiser"]["path"], device=args.device, state="ema"
    )
    embedding, _ = load_embedding_checkpoint(assets["hbt"]["path"], device=device)
    for model in (restorer, embedding):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    common = protocol["common_solver"]
    variant_specs = {
        BASELINE_KEY: protocol["baseline"],
        CANDIDATE_KEY: protocol["candidate"],
    }

    def predict(name: str, input_image: np.ndarray) -> PhaseAPrediction:
        raw = split_tiles_numpy(input_image)
        denoised = restore_tiles_uint8(
            restorer, raw, device, batch_size=args.denoise_batch_size
        )
        bank = build_classical_score_bank(
            denoised, prefix="denoised", chunk_size=args.classical_chunk_size
        )
        c1_names = [
            key
            for key in sorted(bank)
            if key.startswith("denoised_") and not key.endswith("_c2")
        ]
        c1 = fuse_ranked_scores(bank, names=c1_names, name="denoised_C1")
        hbt, _ = learned_compatibility(
            embedding, denoised, device=device, name="denoised_hbt_l1"
        )
        seed_result = soft_cycle_component_solver(
            hbt,
            top_k=int(common["soft_cycle_top_k"]),
            keep_per_tile=int(common["soft_cycle_keep_per_tile"]),
            proposal_keep_fraction=float(common["soft_cycle_keep_fraction"]),
            loop_weight=float(common["soft_cycle_loop_weight"]),
            reciprocal_weight=float(common["soft_cycle_reciprocal_weight"]),
        )
        initial = validate_permutation(seed_result.position_to_slot, name="initial_layout")
        qap_seed = _filename_qap_seed(name)
        layouts: dict[str, np.ndarray] = {}
        renders: dict[str, np.ndarray] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for key in (BASELINE_KEY, CANDIDATE_KEY):
            spec = variant_specs[key]
            score = fuse_ranked_scores(
                {c1.name: c1, hbt.name: hbt},
                names=[c1.name, hbt.name],
                weights={hbt.name: float(spec["hbt_weight"])},
                name=str(spec["score"]),
            )
            result = directional_qap(
                score,
                initial=initial.copy(),
                iterations=int(common["qap_iterations"]),
                restarts=int(common["qap_restarts"]),
                seed=qap_seed,
                boundary_weight=float(common["qap_boundary_weight"]),
                initial_weight=float(common["qap_initial_weight"]),
                noisy_components=int(common["qap_noisy_components"]),
                noise_scale=float(common["qap_noise_scale"]),
                refine_swaps=int(common["qap_refine_swaps"]),
                refine_weak_cells=int(common["qap_refine_weak_cells"]),
            )
            layout = validate_permutation(result.position_to_slot, name=f"{key}_layout")
            layouts[key] = layout.copy()
            renders[key] = merge_tiles_numpy(denoised[layout])
            diagnostics[key] = {
                "objective": float(result.objective),
                "relaxed_objective": float(result.relaxed_objective),
                "restart": int(result.restart),
                "iterations": int(result.iterations),
                "converged": bool(result.converged),
            }
        return PhaseAPrediction(
            layouts=layouts,
            renders=renders,
            initial_layout=initial.copy(),
            qap_seed=qap_seed,
            denoised_tiles_sha256=hashlib.sha256(denoised.tobytes()).hexdigest(),
            diagnostics=diagnostics,
        )

    return predict


def _assert_empty_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise RuntimeError(f"output directory must be empty: {path}")
    return path


def _contained(path_value: str, root: Path, expected_name: str) -> Path:
    relative = Path(path_value)
    if relative.is_absolute() or relative.parts != ("artifacts", expected_name):
        raise RuntimeError(f"frozen artifact path is not canonical-relative: {path_value}")
    artifact_root = (root.resolve() / "artifacts").resolve()
    path = (root.resolve() / relative).resolve()
    if path.parent != artifact_root or path.name != expected_name:
        raise RuntimeError(f"frozen artifact escaped its canonical directory: {path}")
    return path


def _artifact_record(
    *,
    root: Path,
    name: str,
    source_index: int,
    input_path: Path,
    input_sha256: str,
    prediction: PhaseAPrediction,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    stem = Path(name).stem
    initial = validate_permutation(prediction.initial_layout, name="initial_layout")
    expected_qap_seed = _filename_qap_seed(name)
    if prediction.qap_seed != expected_qap_seed:
        raise RuntimeError(f"QAP seed drift for {name}")
    variants: dict[str, Any] = {}
    for key in (BASELINE_KEY, CANDIDATE_KEY):
        layout = validate_permutation(prediction.layouts[key], name=f"{key}_layout")
        render = np.asarray(prediction.renders[key])
        if render.dtype != np.uint8 or render.shape != (480, 480, 3):
            raise RuntimeError(f"invalid frozen {key} render for {name}")
        layout_name = f"{stem}.{key}.layout.npy"
        render_name = f"{stem}.{key}.png"
        layout_path = root / layout_name
        render_path = root / render_name
        _atomic_bytes(layout_path, _npy_bytes(layout.astype(np.int32, copy=False)))
        _atomic_bytes(render_path, _png_bytes(render))
        spec = protocol[key]
        variants[key] = {
            "label": str(spec["label"]),
            "score": str(spec["score"]),
            "hbt_weight": float(spec["hbt_weight"]),
            "layout_path": str(Path("artifacts") / layout_name),
            "layout_sha256": _sha256(layout_path),
            "layout_value_sha256": _layout_sha256(layout),
            "render_path": str(Path("artifacts") / render_name),
            "render_sha256": _sha256(render_path),
            "qap_seed": expected_qap_seed,
            "initial_layout_value_sha256": _layout_sha256(initial),
            "valid_permutation": True,
            "solver": prediction.diagnostics.get(key, {}),
        }
    return {
        "source_index": source_index,
        "name": name,
        "input_path": str(Path("train") / "inputs" / name),
        "input_sha256": input_sha256,
        "denoised_tiles_sha256": prediction.denoised_tiles_sha256,
        "qap_seed": expected_qap_seed,
        "initial_layout_value_sha256": _layout_sha256(initial),
        "variants": variants,
    }


def _artifact_snapshot(
    record: dict[str, Any], artifact_root: Path, protocol: dict[str, Any]
) -> dict[str, str]:
    name = str(record["name"])
    stem = Path(name).stem
    snapshot: dict[str, str] = {}
    if int(record["qap_seed"]) != _filename_qap_seed(name):
        raise RuntimeError(f"record QAP seed mismatch: {name}")
    variants = record.get("variants")
    if not isinstance(variants, dict) or set(variants) != {BASELINE_KEY, CANDIDATE_KEY}:
        raise RuntimeError(f"record does not contain exactly w4/w1 variants: {name}")
    initial_hashes: set[str] = set()
    qap_seeds: set[int] = set()
    for key in (BASELINE_KEY, CANDIDATE_KEY):
        variant = variants[key]
        expected_variant = protocol[key]
        for field in ("label", "score", "hbt_weight"):
            if variant.get(field) != expected_variant[field]:
                raise RuntimeError(f"frozen {key} variant binding mismatch: {field}")
        layout_path = _contained(
            str(variant["layout_path"]), artifact_root, f"{stem}.{key}.layout.npy"
        )
        render_path = _contained(
            str(variant["render_path"]), artifact_root, f"{stem}.{key}.png"
        )
        layout_payload = layout_path.read_bytes()
        render_payload = render_path.read_bytes()
        if _bytes_sha256(layout_payload) != variant.get("layout_sha256"):
            raise RuntimeError(f"frozen layout hash mismatch: {layout_path}")
        if _bytes_sha256(render_payload) != variant.get("render_sha256"):
            raise RuntimeError(f"frozen render hash mismatch: {render_path}")
        layout = _decode_layout(layout_payload, name=f"{key}_layout")
        if _layout_sha256(layout) != variant.get("layout_value_sha256"):
            raise RuntimeError(f"frozen layout value hash mismatch: {layout_path}")
        _decode_frozen_png(render_payload)
        if variant.get("valid_permutation") is not True:
            raise RuntimeError(f"manifest does not attest valid permutation: {name}")
        initial_hashes.add(str(variant.get("initial_layout_value_sha256")))
        qap_seeds.add(int(variant.get("qap_seed")))
        for relative, digest in (
            (str(variant["layout_path"]), _bytes_sha256(layout_payload)),
            (str(variant["render_path"]), _bytes_sha256(render_payload)),
        ):
            if relative in snapshot:
                raise RuntimeError(f"duplicate frozen artifact path: {relative}")
            snapshot[relative] = digest
    if initial_hashes != {str(record["initial_layout_value_sha256"])}:
        raise RuntimeError(f"variants did not share one HBT initializer: {name}")
    if qap_seeds != {int(record["qap_seed"])}:
        raise RuntimeError(f"variants did not share one QAP seed: {name}")
    return snapshot


def _merge_snapshot(destination: dict[str, str], incoming: dict[str, str]) -> None:
    overlap = set(destination).intersection(incoming)
    if overlap:
        raise RuntimeError(f"duplicate frozen artifact paths: {sorted(overlap)[:3]}")
    destination.update(incoming)


def run_phase_a(args: argparse.Namespace, *, predictor: Predictor | None = None) -> dict[str, Any]:
    if args.phase_a_dir is None:
        raise RuntimeError("phase-a requires --phase-a-dir")
    if args.world_size != 2 or args.rank not in {0, 1}:
        raise RuntimeError("phase-a requires exactly rank 0/1 and world-size 2")
    protocol, assets = _validated_protocol_and_assets(args)
    names = _expected_names(protocol, assets)
    output_dir = _assert_empty_output_dir(Path(args.phase_a_dir))
    artifacts_dir = (output_dir / "artifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=False)
    if predictor is None:
        predictor = _build_default_predictor(protocol, assets, args)
    records: list[dict[str, Any]] = []
    for source_index in range(args.rank, len(names), args.world_size):
        name = names[source_index]
        input_path = (Path(args.data_root).resolve() / "train" / "inputs" / name).resolve()
        input_root = (Path(args.data_root).resolve() / "train" / "inputs").resolve()
        if input_path.parent != input_root:
            raise RuntimeError(f"input path escaped root: {input_path}")
        input_payload, input_image = _read_rgb_bytes(input_path)
        prediction = predictor(name, input_image)
        records.append(
            _artifact_record(
                root=artifacts_dir,
                name=name,
                source_index=source_index,
                input_path=input_path,
                input_sha256=_bytes_sha256(input_payload),
                prediction=prediction,
                protocol=protocol,
            )
        )
    expected_indices = list(range(args.rank, len(names), args.world_size))
    if [int(record["source_index"]) for record in records] != expected_indices:
        raise RuntimeError("phase-a shard source assignment drift")
    for record in records:
        _artifact_snapshot(record, output_dir, protocol)
    code_path = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "kind": "qap_weight_confirmation_phase_a_shard",
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "code_path": str(code_path),
        "code_sha256": _sha256(code_path),
        "assets": _portable_assets(assets),
        "split": "assembly_incremental_gate[128:192]",
        "source_names_sha256": EXPECTED_NAMES_SHA256,
        "source_count_total": 64,
        "rank": args.rank,
        "world_size": args.world_size,
        "assigned_source_indices": expected_indices,
        "artifact_root": "artifacts",
        "common_solver": protocol["common_solver"],
        "baseline": protocol["baseline"],
        "candidate": protocol["candidate"],
        "target_paths_constructed": False,
        "target_files_opened": False,
        "records": records,
    }
    manifest_path = output_dir / SHARD_MANIFEST
    envelope = _atomic_envelope(manifest_path, payload)
    _load_exact_envelope(manifest_path, _sha256(manifest_path))
    result = {
        "event": "qap_weight_confirmation_phase_a_complete",
        "action": "phase-a",
        "rank": args.rank,
        "world_size": args.world_size,
        "record_count": len(records),
        "sources": [
            {"source_index": record["source_index"], "name": record["name"]}
            for record in records
        ],
        "manifest": str(manifest_path),
        "phase_a_envelope_sha256": _sha256(manifest_path),
        "payload_sha256": envelope["payload_sha256"],
        "target_paths_or_pixels_read": False,
        "target_access_count": 0,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _validate_shard_payload(
    payload: dict[str, Any], *, protocol: dict[str, Any],
    assets: dict[str, dict[str, str]], expected_shard_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    required_equal = {
        "kind": "qap_weight_confirmation_phase_a_shard",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "source_names_sha256": EXPECTED_NAMES_SHA256,
        "source_count_total": 64,
        "target_paths_constructed": False,
        "target_files_opened": False,
        "common_solver": protocol["common_solver"],
        "baseline": protocol["baseline"],
        "candidate": protocol["candidate"],
        "assets": _portable_assets(assets),
    }
    for key, expected in required_equal.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Phase-A shard binding mismatch: {key}")
    rank, world_size = int(payload["rank"]), int(payload["world_size"])
    if world_size != 2 or rank not in {0, 1}:
        raise RuntimeError("Phase-A shard must be rank 0/1 of world size 2")
    current_code = Path(__file__).resolve()
    if payload.get("code_sha256") != _sha256(current_code):
        raise RuntimeError("Phase-A shard evaluator code hash mismatch")
    if Path(str(payload.get("code_path"))).resolve() != current_code:
        raise RuntimeError("Phase-A shard evaluator code path mismatch")
    expected_indices = list(range(rank, 64, world_size))
    if payload.get("assigned_source_indices") != expected_indices:
        raise RuntimeError("Phase-A shard modulo assignment mismatch")
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Phase-A shard records are missing")
    if [int(record["source_index"]) for record in records] != expected_indices:
        raise RuntimeError("Phase-A shard record indices mismatch")
    if payload.get("artifact_root") != "artifacts":
        raise RuntimeError("Phase-A artifact_root must be canonical-relative")
    snapshot: dict[str, str] = {}
    for record in records:
        name = str(record["name"])
        if record.get("input_path") != str(Path("train") / "inputs" / name):
            raise RuntimeError("Phase-A input path is not canonical-relative")
        _merge_snapshot(
            snapshot, _artifact_snapshot(record, expected_shard_dir, protocol)
        )
    return records, snapshot


def run_finalize_phase_a(args: argparse.Namespace) -> dict[str, Any]:
    if not args.phase_a_dirs or not args.phase_a_envelope_sha256s:
        raise RuntimeError("finalize-phase-a requires shard dirs and SHA256 anchors")
    if len(args.phase_a_dirs) != len(args.phase_a_envelope_sha256s):
        raise RuntimeError("phase-a dirs/anchors length mismatch")
    if len(args.phase_a_dirs) != 2:
        raise RuntimeError("finalize-phase-a requires exactly two shard directories")
    if args.finalized_phase_a_dir is None:
        raise RuntimeError("finalize-phase-a requires --finalized-phase-a-dir")
    protocol, assets = _validated_protocol_and_assets(args)
    names = _expected_names(protocol, assets)
    output_dir = _assert_empty_output_dir(Path(args.finalized_phase_a_dir))
    all_records: list[dict[str, Any]] = []
    shard_bindings: list[dict[str, Any]] = []
    shard_dirs_by_rank: dict[int, Path] = {}
    seen_ranks: set[int] = set()
    world_sizes: set[int] = set()
    for directory, anchor in zip(
        args.phase_a_dirs, args.phase_a_envelope_sha256s, strict=True
    ):
        manifest_path = Path(directory).resolve() / SHARD_MANIFEST
        envelope = _load_exact_envelope(manifest_path, anchor)
        payload = envelope["payload"]
        records, snapshot = _validate_shard_payload(
            payload,
            protocol=protocol,
            assets=assets,
            expected_shard_dir=Path(directory).resolve(),
        )
        rank = int(payload["rank"])
        if rank in seen_ranks:
            raise RuntimeError("duplicate Phase-A shard rank")
        seen_ranks.add(rank)
        shard_dirs_by_rank[rank] = Path(directory).resolve()
        world_sizes.add(int(payload["world_size"]))
        all_records.extend(records)
        shard_bindings.append(
            {
                "rank": rank,
                "world_size": int(payload["world_size"]),
                "manifest_path": str(manifest_path),
                "manifest_sha256": anchor,
                "payload_sha256": envelope["payload_sha256"],
                "artifact_snapshot_sha256": _canonical_sha256(snapshot),
            }
        )
    if len(world_sizes) != 1 or world_sizes != {2}:
        raise RuntimeError("Phase-A shards disagree on world size")
    world_size = next(iter(world_sizes))
    if seen_ranks != set(range(world_size)) or len(shard_bindings) != world_size:
        raise RuntimeError("incomplete Phase-A shard set")
    all_records.sort(key=lambda record: int(record["source_index"]))
    if [int(record["source_index"]) for record in all_records] != list(range(64)):
        raise RuntimeError("Phase-A merge has missing or duplicate source indices")
    if [str(record["name"]) for record in all_records] != names:
        raise RuntimeError("Phase-A merge differs from authoritative source order")
    finalized_artifacts = output_dir / "artifacts"
    finalized_artifacts.mkdir(parents=True, exist_ok=False)
    finalized_snapshot: dict[str, str] = {}
    for record in all_records:
        source_root = shard_dirs_by_rank[int(record["source_index"]) % 2]
        name = str(record["name"])
        stem = Path(name).stem
        for key in (BASELINE_KEY, CANDIDATE_KEY):
            variant = record["variants"][key]
            for field, expected_name in (
                ("layout_path", f"{stem}.{key}.layout.npy"),
                ("render_path", f"{stem}.{key}.png"),
            ):
                source_path = _contained(
                    str(variant[field]), source_root, expected_name
                )
                destination = _contained(
                    str(variant[field]), output_dir, expected_name
                )
                _atomic_bytes(destination, source_path.read_bytes())
        _merge_snapshot(
            finalized_snapshot, _artifact_snapshot(record, output_dir, protocol)
        )
    shard_bindings.sort(key=lambda item: int(item["rank"]))
    payload = {
        "schema_version": 1,
        "kind": "qap_weight_confirmation_finalized_phase_a",
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "code_path": str(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "assets": _portable_assets(assets),
        "split": "assembly_incremental_gate[128:192]",
        "source_names": names,
        "source_names_sha256": EXPECTED_NAMES_SHA256,
        "source_count": 64,
        "common_solver": protocol["common_solver"],
        "baseline": protocol["baseline"],
        "candidate": protocol["candidate"],
        "artifact_root": "artifacts",
        "artifact_snapshot_sha256": _canonical_sha256(finalized_snapshot),
        "shards": shard_bindings,
        "target_paths_constructed": False,
        "target_files_opened": False,
        "final_audit_opened": False,
        "confirmation_audit_opened": False,
        "records": all_records,
    }
    manifest_path = output_dir / FINAL_MANIFEST
    envelope = _atomic_envelope(manifest_path, payload)
    _load_exact_envelope(manifest_path, _sha256(manifest_path))
    result = {
        "event": "qap_weight_confirmation_phase_a_finalized",
        "action": "finalize-phase-a",
        "record_count": 64,
        "sources": [
            {"source_index": record["source_index"], "name": record["name"]}
            for record in all_records
        ],
        "shard_envelope_sha256s": [
            binding["manifest_sha256"] for binding in shard_bindings
        ],
        "manifest": str(manifest_path),
        "phase_a_envelope_sha256": _sha256(manifest_path),
        "payload_sha256": envelope["payload_sha256"],
        "target_paths_or_pixels_read": False,
        "target_access_count": 0,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _validate_finalized_phase_a(
    manifest_path: Path,
    anchor: str,
    *,
    protocol: dict[str, Any],
    assets: dict[str, dict[str, str]],
    names: list[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    envelope = _load_exact_envelope(manifest_path, anchor)
    payload = envelope["payload"]
    required_equal = {
        "kind": "qap_weight_confirmation_finalized_phase_a",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "assets": _portable_assets(assets),
        "source_names": names,
        "source_names_sha256": EXPECTED_NAMES_SHA256,
        "source_count": 64,
        "common_solver": protocol["common_solver"],
        "baseline": protocol["baseline"],
        "candidate": protocol["candidate"],
        "artifact_root": "artifacts",
        "target_paths_constructed": False,
        "target_files_opened": False,
        "final_audit_opened": False,
        "confirmation_audit_opened": False,
    }
    for key, expected in required_equal.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"finalized Phase-A binding mismatch: {key}")
    if payload.get("code_sha256") != _sha256(Path(__file__).resolve()):
        raise RuntimeError("evaluator code changed since Phase A")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 64:
        raise RuntimeError("finalized Phase-A record count mismatch")
    if [int(record["source_index"]) for record in records] != list(range(64)):
        raise RuntimeError("finalized Phase-A source indices mismatch")
    if [str(record["name"]) for record in records] != names:
        raise RuntimeError("finalized Phase-A names/order mismatch")
    snapshot: dict[str, str] = {}
    finalized_root = manifest_path.resolve().parent
    for record in records:
        _merge_snapshot(snapshot, _artifact_snapshot(record, finalized_root, protocol))
    if payload.get("artifact_snapshot_sha256") != _canonical_sha256(snapshot):
        raise RuntimeError("finalized Phase-A artifact snapshot mismatch")
    return payload, snapshot


def _bootstrap_ci(deltas: np.ndarray, metric: dict[str, Any]) -> list[float]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.shape != (64,):
        raise RuntimeError("bootstrap requires exactly 64 paired whole-source deltas")
    rng = np.random.default_rng(int(metric["bootstrap_seed"]))
    indices = rng.integers(
        0, len(values), size=(int(metric["bootstrap_resamples"]), len(values))
    )
    means = values[indices].mean(axis=1)
    return [
        float(value)
        for value in np.quantile(means, metric["bootstrap_quantiles"])
    ]


def _official_ssim(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.uint8)
    prediction = np.asarray(prediction, dtype=np.uint8)
    if target.shape != (480, 480, 3) or prediction.shape != target.shape:
        raise RuntimeError("official SSIM inputs must both be uint8 RGB 480x480")
    return float(
        structural_similarity(target, prediction, channel_axis=2, data_range=255)
    )


def _gate(aggregate: dict[str, Any], gate_spec: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "mean_ssim_delta_ge_0.005": (
            aggregate["mean_ssim_delta"] >= gate_spec["mean_ssim_delta_min"]
        ),
        "bootstrap_95_lower_gt_0": (
            aggregate["bootstrap_95_ci"][0]
            > gate_spec["ssim_bootstrap_95_lower_gt"]
        ),
        "wins_ge_40": aggregate["wins"] >= gate_spec["wins_min"],
        "large_regressions_le_6": (
            aggregate["large_regressions"] <= gate_spec["large_regressions_max"]
        ),
        "valid_permutation_count_eq_64": (
            aggregate["valid_permutation_count"]
            == gate_spec["valid_permutation_count"]
        ),
    }
    return {"logic": gate_spec["logic"], "passed": bool(all(checks.values())), "checks": checks}


def run_phase_b(
    args: argparse.Namespace, *, target_reader: TargetReader = _read_rgb
) -> dict[str, Any]:
    if args.finalized_phase_a_dir is None or args.phase_a_envelope_sha256 is None:
        raise RuntimeError("phase-b requires finalized Phase-A dir and SHA256 anchor")
    if args.output is None:
        raise RuntimeError("phase-b requires --output")
    protocol, assets = _validated_protocol_and_assets(args)
    names = _expected_names(protocol, assets)
    finalized_dir = Path(args.finalized_phase_a_dir).resolve()
    manifest_path = finalized_dir / FINAL_MANIFEST
    anchor = str(args.phase_a_envelope_sha256)
    phase_a_payload, before_snapshot = _validate_finalized_phase_a(
        manifest_path, anchor, protocol=protocol, assets=assets, names=names
    )
    before_integrity = {
        "config_sha256": _sha256(Path(args.config).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "assets": {key: _sha256(value["path"]) for key, value in assets.items()},
        "artifacts": before_snapshot,
    }
    if before_integrity["config_sha256"] != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("config changed before target access")
    marker_path = finalized_dir / TARGET_MARKER
    if marker_path.exists():
        raise RuntimeError("target-access marker already exists; Phase B is one-shot")
    marker_payload = {
        "schema_version": 1,
        "kind": "qap_weight_confirmation_target_access_event",
        "phase_a_manifest_sha256": anchor,
        "phase_a_payload_sha256": _canonical_sha256(phase_a_payload),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "code_sha256": before_integrity["code_sha256"],
        "source_names_sha256": EXPECTED_NAMES_SHA256,
        "target_access_started": True,
        "target_files_may_have_been_opened": True,
    }
    marker_envelope = _atomic_envelope(marker_path, marker_payload)
    marker_sha256 = _sha256(marker_path)
    _load_exact_envelope(marker_path, marker_sha256)

    records: list[dict[str, Any]] = []
    target_access_count = 0
    for record in phase_a_payload["records"]:
        # The marker has been durably written and validated before this target
        # root/path is constructed for the first time.
        name = str(record["name"])
        target_path = (
            Path(args.data_root).resolve() / "train" / "targets" / name
        ).resolve()
        target_root = (Path(args.data_root).resolve() / "train" / "targets").resolve()
        if target_path.parent != target_root:
            raise RuntimeError(f"target path escaped root: {target_path}")
        target = target_reader(target_path)
        target_access_count += 1
        scores: dict[str, float] = {}
        for key in (BASELINE_KEY, CANDIDATE_KEY):
            variant = record["variants"][key]
            render_path = _contained(
                str(variant["render_path"]),
                finalized_dir,
                f"{Path(name).stem}.{key}.png",
            )
            render_payload = render_path.read_bytes()
            if _bytes_sha256(render_payload) != variant["render_sha256"]:
                raise RuntimeError("render changed while attaching targets")
            scores[key] = _official_ssim(target, _decode_frozen_png(render_payload))
        delta = scores[CANDIDATE_KEY] - scores[BASELINE_KEY]
        records.append(
            {
                "source_index": int(record["source_index"]),
                "name": name,
                "baseline_ssim": scores[BASELINE_KEY],
                "candidate_ssim": scores[CANDIDATE_KEY],
                "delta_ssim": delta,
                "baseline_layout_sha256": record["variants"][BASELINE_KEY][
                    "layout_value_sha256"
                ],
                "candidate_layout_sha256": record["variants"][CANDIDATE_KEY][
                    "layout_value_sha256"
                ],
            }
        )

    # Reconstruct every binding after scoring.  A TOCTOU failure raises and no
    # report containing an accepted metric is written.
    _, after_snapshot = _validate_finalized_phase_a(
        manifest_path, anchor, protocol=protocol, assets=assets, names=names
    )
    after_integrity = {
        "config_sha256": _sha256(Path(args.config).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "assets": {key: _sha256(value["path"]) for key, value in assets.items()},
        "artifacts": after_snapshot,
    }
    if after_integrity != before_integrity:
        raise RuntimeError("Phase-A/config/checkpoint integrity changed during target scoring")
    if _sha256(marker_path) != marker_sha256:
        raise RuntimeError("target-access marker changed during target scoring")
    _load_exact_envelope(marker_path, marker_sha256)
    if target_access_count != 64:
        raise RuntimeError("Phase B did not score exactly 64 targets")

    deltas = np.asarray([record["delta_ssim"] for record in records], dtype=np.float64)
    metric_spec = protocol["original_real_confirmation"]["metric"]
    gate_spec = protocol["original_real_confirmation"]["gate"]
    tolerance = float(metric_spec["tie_tolerance"])
    valid_permutations = sum(
        bool(record["variants"][BASELINE_KEY]["valid_permutation"])
        and bool(record["variants"][CANDIDATE_KEY]["valid_permutation"])
        for record in phase_a_payload["records"]
    )
    aggregate = {
        "source_count": 64,
        "bootstrap_unit": "paired_whole_source_delta_candidate_minus_baseline",
        "mean_baseline_ssim": float(
            np.mean([record["baseline_ssim"] for record in records])
        ),
        "mean_candidate_ssim": float(
            np.mean([record["candidate_ssim"] for record in records])
        ),
        "mean_ssim_delta": float(deltas.mean()),
        "median_ssim_delta": float(np.median(deltas)),
        "bootstrap_95_ci": _bootstrap_ci(deltas, metric_spec),
        "wins": int(np.sum(deltas > tolerance)),
        "ties": int(np.sum(np.abs(deltas) <= tolerance)),
        "losses": int(np.sum(deltas < -tolerance)),
        "large_regressions": int(np.sum(deltas < -0.01)),
        "valid_permutation_count": int(valid_permutations),
    }
    gate = _gate(aggregate, gate_spec)
    if gate["passed"]:
        status = "promotion_gate_passed"
    elif (
        aggregate["bootstrap_95_ci"][0] > 0.0
        and aggregate["mean_ssim_delta"] > 0.0
        and aggregate["mean_ssim_delta"] < 0.005
    ):
        status = "confirmed_small_gain_no_promotion"
    else:
        status = "promotion_gate_failed"
    protocol_report = {
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "split": "assembly_incremental_gate",
        "offset": 128,
        "count": 64,
        "source_names_sha256": EXPECTED_NAMES_SHA256,
    }
    solver_report = {
        "common": protocol["common_solver"],
        "baseline": protocol["baseline"],
        "candidate": protocol["candidate"],
    }
    target_access_report = {
        "marker": str(marker_path),
        "marker_sha256": marker_sha256,
        "marker_payload_sha256": marker_envelope["payload_sha256"],
        "marker_preceded_first_target_path_construction": True,
        "target_access_count": target_access_count,
    }
    phase_b_report = {
        **target_access_report,
        "integrity_before_sha256": _canonical_sha256(before_integrity),
        "integrity_after_sha256": _canonical_sha256(after_integrity),
        "post_score_rehash_matched": True,
    }
    report = {
        "schema_version": 1,
        "kind": "qap_weight_confirmation_report",
        "status": status,
        "safe_for_submission": False,
        "eligible_for_final_audit": bool(gate["passed"]),
        "protocol": protocol_report,
        "config": {"path": str(Path(args.config).resolve()), "sha256": EXPECTED_CONFIG_SHA256},
        "code": {"path": str(Path(__file__).resolve()), "sha256": before_integrity["code_sha256"]},
        "assets": assets,
        "split": "assembly_incremental_gate[128:192]",
        "source_names": names,
        "source_names_sha256": EXPECTED_NAMES_SHA256,
        "baseline": protocol["baseline"],
        "candidate": protocol["candidate"],
        "common_solver": protocol["common_solver"],
        "solver": solver_report,
        "phase_a": {
            "manifest": str(manifest_path),
            "manifest_sha256": anchor,
            "payload_sha256": _canonical_sha256(phase_a_payload),
            "source_count": 64,
            "source_names_sha256": EXPECTED_NAMES_SHA256,
            "shards": phase_a_payload["shards"],
            "integrity_before_sha256": _canonical_sha256(before_integrity),
            "integrity_after_sha256": _canonical_sha256(after_integrity),
        },
        "phase_b": phase_b_report,
        "target_access": target_access_report,
        "metric": protocol["original_real_confirmation"]["metric"],
        "records": records,
        "aggregate": aggregate,
        "paired_metrics": aggregate,
        "gate": gate,
        "sealed_sets": {
            "final_audit_opened": False,
            "confirmation_audit_opened": False,
            "must_remain_unopened": True,
        },
        "post_phase_b_mutation_policy": protocol["original_real_confirmation"][
            "post_phase_b_mutation_policy"
        ],
    }
    output_path = Path(args.output).expanduser().resolve()
    _atomic_bytes(output_path, _canonical_bytes(report) + b"\n")
    result = {
        "event": "qap_weight_confirmation_complete",
        "status": status,
        "safe_for_submission": False,
        "eligible_for_final_audit": bool(gate["passed"]),
        "target_access_count": target_access_count,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "gate": gate,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.denoise_batch_size <= 0 or args.classical_chunk_size <= 0:
        raise SystemExit("batch/chunk sizes must be positive")
    try:
        if args.action == "phase-a":
            run_phase_a(args)
        elif args.action == "finalize-phase-a":
            run_finalize_phase_a(args)
        else:
            run_phase_b(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "event": "qap_weight_confirmation_failed",
                    "action": args.action,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "safe_for_submission": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
