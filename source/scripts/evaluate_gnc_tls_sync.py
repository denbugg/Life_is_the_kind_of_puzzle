#!/usr/bin/env python3
"""Leakage-safe calibration of a four-side GNC-TLS initializer for production w4 QAP.

The calibration intentionally reuses an already exposed development slice.  It
is not a confirmation and it can never make a submission-ready claim.  Clean
targets are used before prediction only by ``make_exact_panel`` to construct
the two deterministic corrupted inputs; neither the clean pixels nor the known
permutation are passed to the predictor.  All layouts and rendered PNG bytes
are durably frozen before the known permutation is reconstructed for scoring.

The production comparator is copied exactly from
``scripts/evaluate_qap_weight_confirmation.py``:

* EMA TileNAF rendering;
* HBT soft-cycle initialization;
* C1/HBTw4 rank fusion;
* filename-seeded w4 QAP with 25 iterations and two restarts.

The candidate changes only the QAP initializer.  Four fixed configurations use
R/L/U/D top-k candidates from w4, GNC-TLS synchronization, restricted Hungarian
projection, and then the same w4 QAP budget as the comparator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import skimage
from skimage.metrics import structural_similarity
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puzzle_assembly.compatibility import (  # noqa: E402
    CompatibilityMatrices,
    build_classical_score_bank,
    fuse_ranked_scores,
)
from puzzle_assembly.components import soft_cycle_component_solver  # noqa: E402
from puzzle_assembly.geometry import GRID, TILE_COUNT, validate_permutation  # noqa: E402
from puzzle_assembly.gnc_tls_sync import GncTlsConfig, solve_gnc_tls  # noqa: E402
from puzzle_assembly.learned import (  # noqa: E402
    learned_compatibility,
    load_embedding_checkpoint,
)
from puzzle_assembly.metrics import layout_metrics  # noqa: E402
from puzzle_assembly.panels import make_exact_panel  # noqa: E402
from puzzle_assembly.protocol import per_source_seed, source_names_for_split  # noqa: E402
from puzzle_assembly.qap import directional_qap  # noqa: E402
from puzzle_denoise_v2.inference import (  # noqa: E402
    load_restorer,
    restore_tiles_uint8,
)
from puzzle_denoise_v2.tiles import merge_tiles_numpy  # noqa: E402


SCHEMA_VERSION = 1
MASTER_SEED = 20260713
PANELS = ("primary_kornia", "independent_libjpeg")
SOURCE_SPLIT = "assembly_incremental_gate"
SOURCE_OFFSET = 0
SOURCE_NAMES = (
    "img_001485.png",
    "img_005748.png",
    "img_003783.png",
    "img_001693.png",
    "img_006659.png",
    "img_004510.png",
    "img_005403.png",
    "img_005200.png",
)
SOURCE_NAMES_SHA256 = "bc2f89e49371486ffece5d8ca9881f7de15b22948bab2e0e0749dbfdbffc3581"

PRODUCTION_CONFIG_SHA256 = "30732463fb200bdff8f909ef06be6cb6c4e7859692e01c9d33c5d55175ffe262"
PRODUCTION_EVALUATOR_SHA256 = "4083d11146f62a91d007a553cfb1ae0ec943141e7a0b3a4639fac3d2f1d9559a"
ASSET_SHA256 = {
    "denoiser": "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734",
    "hbt": "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787",
    "manifest": "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345",
    "quarantine": "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215",
}
EXPECTED_COMMON_SOLVER = {
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
FROZEN_CONFIGS = (
    ("k2_c0p50", 2, 0.50),
    ("k2_c0p75", 2, 0.75),
    ("k4_c0p50", 4, 0.50),
    ("k4_c0p75", 4, 0.75),
)
FORBIDDEN_PATH_TOKENS = (
    "candidate_graph_oracle",
    "fixture_label",
    "assembly_final_audit",
    "assembly_audit_exposed",
    "candidate_graph_oracle_v4",
)
PANEL_MARKER = "PANEL_INPUT_CONSTRUCTION_STARTED.json"
FROZEN_MANIFEST = "FROZEN_INPUT_ONLY_MANIFEST.json"
SCORING_MARKER = "SCORING_TRUTH_ACCESS_STARTED.json"
REPORT_NAME = "gnc_tls_sync_report.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument(
        "--production-config", default="configs/qap_weight_confirmation_v1.json"
    )
    parser.add_argument(
        "--denoiser", default="runs/denoise_v2/release/selected_tilenaf_synth_50k.pt"
    )
    parser.add_argument(
        "--hbt-checkpoint",
        default="runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt",
    )
    parser.add_argument("--manifest", default="configs/denoise_splits_seed20260710.json")
    parser.add_argument(
        "--quarantine", default="configs/denoise_validation_quarantine_v1.json"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--denoise-batch-size", type=int, default=512)
    parser.add_argument("--classical-chunk-size", type=int, default=64)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return _bytes_sha256(_canonical_bytes(payload))


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    header = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": [int(value) for value in array.shape]}
    )
    return _bytes_sha256(header + b"\0" + array.tobytes(order="C"))


def _names_sha256(names: Sequence[str]) -> str:
    return _bytes_sha256("\n".join(names).encode("utf-8"))


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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _load_exact_envelope(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    envelope = json.loads(raw)
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "payload_sha256"}:
        raise RuntimeError(f"invalid canonical envelope: {path}")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid envelope payload: {path}")
    if envelope["payload_sha256"] != _canonical_sha256(payload):
        raise RuntimeError(f"payload hash mismatch: {path}")
    if raw != _canonical_bytes(envelope) + b"\n":
        raise RuntimeError(f"non-canonical JSON envelope: {path}")
    return envelope


def _npy_bytes(values: np.ndarray) -> bytes:
    output = BytesIO()
    np.save(output, values, allow_pickle=False)
    return output.getvalue()


def _png_bytes(values: np.ndarray) -> bytes:
    array = np.asarray(values)
    if array.dtype != np.uint8 or array.shape != (480, 480, 3):
        raise RuntimeError("render must be uint8 RGB 480x480")
    output = BytesIO()
    Image.fromarray(array, mode="RGB").save(output, format="PNG", compress_level=6)
    return output.getvalue()


def _read_rgb_bytes(path: Path) -> tuple[bytes, np.ndarray]:
    payload = path.read_bytes()
    with Image.open(BytesIO(payload)) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise RuntimeError(f"unexpected RGB image shape for {path}: {values.shape}")
    return payload, values


def _decode_png(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if values.shape != (480, 480, 3):
        raise RuntimeError("frozen PNG is not uint8 RGB 480x480")
    return values


def _decode_layout(payload: bytes, *, name: str) -> np.ndarray:
    return validate_permutation(np.load(BytesIO(payload), allow_pickle=False), name=name)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def _reject_forbidden_path(path: Path, *, label: str) -> None:
    lowered = path.as_posix().lower()
    hits = [token for token in FORBIDDEN_PATH_TOKENS if token in lowered]
    v4_components = [
        part
        for part in (value.lower() for value in path.parts)
        if part == "v4" or part.startswith("v4_") or part.endswith("_v4")
    ]
    if v4_components:
        hits.append("v4_path_component")
    if hits:
        raise RuntimeError(f"forbidden {label} path tokens {hits}: {path}")


def _prepare_output_dir(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output}")
    (output / "artifacts").mkdir()
    _fsync_directory(output)
    return output


def _validate_protocol_and_assets(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = {
        "production_config": _resolve(args.production_config),
        "production_evaluator": REPO_ROOT / "scripts/evaluate_qap_weight_confirmation.py",
        "data_root": _resolve(args.data_root),
        "denoiser": _resolve(args.denoiser),
        "hbt": _resolve(args.hbt_checkpoint),
        "manifest": _resolve(args.manifest),
        "quarantine": _resolve(args.quarantine),
    }
    for label, path in paths.items():
        _reject_forbidden_path(path, label=label)
    if _file_sha256(paths["production_config"]) != PRODUCTION_CONFIG_SHA256:
        raise RuntimeError("production config SHA256 drift")
    if _file_sha256(paths["production_evaluator"]) != PRODUCTION_EVALUATOR_SHA256:
        raise RuntimeError("authoritative production evaluator SHA256 drift")
    protocol = _load_json(paths["production_config"])
    if protocol.get("kind") != "fixed_qap_weight_confirmation":
        raise RuntimeError("unexpected production config kind")
    if protocol.get("common_solver") != EXPECTED_COMMON_SOLVER:
        raise RuntimeError("production common solver contract drift")
    configured_assets = protocol.get("assets")
    if not isinstance(configured_assets, dict):
        raise RuntimeError("production config has no asset bindings")
    configured_hashes = {
        "denoiser": configured_assets.get("denoiser_sha256"),
        "hbt": configured_assets.get("hbt_sha256"),
        "manifest": configured_assets.get("manifest_sha256"),
        "quarantine": configured_assets.get("quarantine_sha256"),
    }
    if configured_hashes != ASSET_SHA256:
        raise RuntimeError("production asset hash contract drift")
    for label, expected in ASSET_SHA256.items():
        if _file_sha256(paths[label]) != expected:
            raise RuntimeError(f"pinned {label} SHA256 mismatch")
    authoritative_names = source_names_for_split(
        SOURCE_SPLIT,
        manifest_path=paths["manifest"],
        quarantine_path=paths["quarantine"],
    )[SOURCE_OFFSET : SOURCE_OFFSET + len(SOURCE_NAMES)]
    if tuple(authoritative_names) != SOURCE_NAMES:
        raise RuntimeError("reusable calibration source slice drift")
    if _names_sha256(authoritative_names) != SOURCE_NAMES_SHA256:
        raise RuntimeError("reusable calibration names SHA256 drift")
    return protocol, paths


def _code_pins(paths: Mapping[str, Path]) -> dict[str, str]:
    code_paths = {
        Path(__file__).resolve(),
        paths["production_evaluator"].resolve(),
    }
    for package in (
        REPO_ROOT / "src/puzzle_assembly",
        REPO_ROOT / "src/puzzle_denoise_v2",
    ):
        code_paths.update(path.resolve() for path in package.rglob("*.py"))
    relative_paths = {
        path.relative_to(REPO_ROOT).as_posix(): path for path in code_paths
    }
    return {
        relative: _file_sha256(relative_paths[relative])
        for relative in sorted(relative_paths)
    }


def _asset_pins(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        label: {"path": str(paths[label]), "sha256": _file_sha256(paths[label])}
        for label in ("production_config", "denoiser", "hbt", "manifest", "quarantine")
    }


def _hardware() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "skimage": skimage.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }


def _filename_qap_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little") + 7001


def _build_scores(
    denoised: np.ndarray,
    embedding: torch.nn.Module,
    *,
    device: torch.device,
    classical_chunk_size: int,
) -> tuple[CompatibilityMatrices, CompatibilityMatrices, CompatibilityMatrices]:
    bank = build_classical_score_bank(
        denoised, prefix="denoised", chunk_size=classical_chunk_size
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
    w4 = fuse_ranked_scores(
        {c1.name: c1, hbt.name: hbt},
        names=[c1.name, hbt.name],
        weights={hbt.name: 4.0},
        name="denoised_C1_HBTw4_rank_fusion",
    )
    return c1, hbt, w4


def _production_initial(hbt: CompatibilityMatrices) -> tuple[np.ndarray, dict[str, Any]]:
    result = soft_cycle_component_solver(
        hbt,
        top_k=8,
        keep_per_tile=1,
        proposal_keep_fraction=0.5,
        loop_weight=1.0,
        reciprocal_weight=0.35,
    )
    initial = validate_permutation(result.position_to_slot, name="production_initial")
    return initial, {
        "accepted_edges": int(len(result.accepted_edges)),
        "proposed_edges": int(result.proposed_edges),
        "component_sizes": [int(value) for value in result.component_sizes],
        "placed_component_tiles": int(result.placed_component_tiles),
        "unresolved_tiles_before_assignment": int(result.unresolved_tiles_before_assignment),
    }


def _run_production_qap(
    w4: CompatibilityMatrices, initial: np.ndarray, *, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    result = directional_qap(
        w4,
        initial=validate_permutation(initial, name="qap_initial").copy(),
        iterations=25,
        restarts=2,
        seed=seed,
        boundary_weight=0.05,
        initial_weight=0.75,
        noisy_components=3,
        noise_scale=1.0,
        refine_swaps=8,
        refine_weak_cells=32,
    )
    layout = validate_permutation(result.position_to_slot, name="qap_layout")
    return layout, {
        "objective": float(result.objective),
        "relaxed_objective": float(result.relaxed_objective),
        "restart": int(result.restart),
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
    }


def _stable_top_k(
    values: np.ndarray,
    order: np.ndarray,
    *,
    query: int,
    axis: int,
    top_k: int,
) -> list[int]:
    candidates = (
        order[query].tolist() if axis == 1 else order[:, query].tolist()
    )
    selected: list[int] = []
    for candidate in candidates:
        candidate = int(candidate)
        value = values[query, candidate] if axis == 1 else values[candidate, query]
        if candidate != query and np.isfinite(value):
            selected.append(candidate)
            if len(selected) == top_k:
                break
    if len(selected) != top_k:
        raise RuntimeError("not enough finite four-side candidates")
    return selected


def _four_side_edges(
    w4: CompatibilityMatrices, *, top_k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    source: list[int] = []
    destination: list[int] = []
    offsets: list[tuple[float, float]] = []
    confidence: list[float] = []
    side_code: list[int] = []
    counts = {"R": 0, "L": 0, "D": 0, "U": 0}
    orders = {
        "right_row": np.argsort(w4.right, axis=1, kind="stable"),
        "right_column": np.argsort(w4.right, axis=0, kind="stable"),
        "down_row": np.argsort(w4.down, axis=1, kind="stable"),
        "down_column": np.argsort(w4.down, axis=0, kind="stable"),
    }

    def add(query: int, candidate: int, offset: tuple[float, float], rank: int, side: str) -> None:
        source.append(query)
        destination.append(candidate)
        offsets.append(offset)
        confidence.append(1.0 / float(rank + 1))
        side_code.append(("R", "L", "D", "U").index(side))
        counts[side] += 1

    for query in range(TILE_COUNT):
        for rank, candidate in enumerate(
            _stable_top_k(
                w4.right, orders["right_row"], query=query, axis=1, top_k=top_k
            )
        ):
            add(query, candidate, (1.0, 0.0), rank, "R")
        for rank, candidate in enumerate(
            _stable_top_k(
                w4.right, orders["right_column"], query=query, axis=0, top_k=top_k
            )
        ):
            # Query ``query`` asks for its left neighbour ``candidate``.
            # The reversed constraint deliberately remains in this query-side
            # group instead of being canonicalized into the matching R edge.
            add(query, candidate, (-1.0, 0.0), rank, "L")
        for rank, candidate in enumerate(
            _stable_top_k(
                w4.down, orders["down_row"], query=query, axis=1, top_k=top_k
            )
        ):
            add(query, candidate, (0.0, 1.0), rank, "D")
        for rank, candidate in enumerate(
            _stable_top_k(
                w4.down, orders["down_column"], query=query, axis=0, top_k=top_k
            )
        ):
            add(query, candidate, (0.0, -1.0), rank, "U")

    expected = TILE_COUNT * top_k
    if counts != {key: expected for key in counts}:
        raise RuntimeError("four-side candidate count drift")
    return (
        np.asarray(source, dtype=np.int32),
        np.asarray(destination, dtype=np.int32),
        np.asarray(offsets, dtype=np.float64),
        np.asarray(confidence, dtype=np.float64),
        np.asarray(side_code, dtype=np.uint8),
        counts,
    )


def _gnc_config(cutoff: float) -> GncTlsConfig:
    return GncTlsConfig(
        grid_size=GRID,
        gnc_stages=8,
        irls_iterations=4,
        gnc_mu_initial=0.05,
        gnc_mu_final=100.0,
        robust_cutoff=cutoff,
        initial_anchor_weight=1e-3,
        current_anchor_weight=0.0,
        regularization=1e-8,
        max_candidates_per_tile=64,
        max_candidate_radius=4.0,
        restarts=2,
        start_perturbation=0.05,
        assignment_jitter=1e-9,
    )


def _qap_score_hashes(
    c1: CompatibilityMatrices,
    hbt: CompatibilityMatrices,
    w4: CompatibilityMatrices,
) -> dict[str, dict[str, str]]:
    return {
        score.name: {
            "right": _array_sha256(score.right),
            "down": _array_sha256(score.down),
        }
        for score in (c1, hbt, w4)
    }


def _write_array_artifact(output: Path, name: str, values: np.ndarray) -> dict[str, Any]:
    relative = Path("artifacts") / name
    path = output / relative
    payload = _npy_bytes(np.asarray(values))
    _atomic_bytes(path, payload)
    return {
        "path": relative.as_posix(),
        "file_sha256": _bytes_sha256(payload),
        "array_sha256": _array_sha256(values),
        "dtype": np.asarray(values).dtype.str,
        "shape": [int(value) for value in np.asarray(values).shape],
    }


def _write_render_artifact(output: Path, name: str, values: np.ndarray) -> dict[str, Any]:
    relative = Path("artifacts") / name
    path = output / relative
    payload = _png_bytes(values)
    _atomic_bytes(path, payload)
    return {
        "path": relative.as_posix(),
        "file_sha256": _bytes_sha256(payload),
        "pixel_sha256": _array_sha256(values),
        "shape": [480, 480, 3],
        "dtype": "uint8",
    }


def _variant_artifacts(
    output: Path,
    *,
    stem: str,
    label: str,
    layout: np.ndarray,
    denoised: np.ndarray,
) -> dict[str, Any]:
    layout = validate_permutation(layout, name=f"{label}_layout")
    render = merge_tiles_numpy(denoised[layout])
    return {
        "layout": _write_array_artifact(
            output, f"{stem}__{label}.layout.npy", layout.astype(np.int32, copy=False)
        ),
        "render": _write_render_artifact(output, f"{stem}__{label}.png", render),
        "valid_permutation": True,
    }


def _predict_record(
    *,
    output: Path,
    name: str,
    panel: str,
    panel_seed: int,
    raw_tiles: np.ndarray,
    clean_target_file_sha256: str,
    clean_target_pixel_sha256: str,
    restorer: torch.nn.Module,
    embedding: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    denoised = restore_tiles_uint8(
        restorer,
        raw_tiles,
        device,
        batch_size=args.denoise_batch_size,
    )
    c1, hbt, w4 = _build_scores(
        denoised,
        embedding,
        device=device,
        classical_chunk_size=args.classical_chunk_size,
    )
    initial, softcycle_diagnostics = _production_initial(hbt)
    qap_seed = _filename_qap_seed(name)
    baseline_layout, baseline_qap = _run_production_qap(w4, initial, seed=qap_seed)
    stem = f"{Path(name).stem}__{panel}"
    baseline = _variant_artifacts(
        output,
        stem=stem,
        label="baseline_w4_qap",
        layout=baseline_layout,
        denoised=denoised,
    )
    baseline["qap"] = baseline_qap

    edges_by_top_k: dict[int, tuple[np.ndarray, ...]] = {}
    edge_metadata: dict[str, Any] = {}
    for top_k in (2, 4):
        edges = _four_side_edges(w4, top_k=top_k)
        edges_by_top_k[top_k] = edges[:-1]
        source, destination, offsets, confidence, side_code = edges[:-1]
        edge_metadata[str(top_k)] = {
            "counts": edges[-1],
            "source_sha256": _array_sha256(source),
            "destination_sha256": _array_sha256(destination),
            "offsets_sha256": _array_sha256(offsets),
            "confidence_sha256": _array_sha256(confidence),
            "query_side_code_sha256": _array_sha256(side_code),
            "raw_edge_count": int(len(source)),
            "query_side_encoding": {"R": 0, "L": 1, "D": 2, "U": 3},
            "equivalent_reciprocal_constraints_retained": True,
            "confidence": "1/(stable side rank + 1); normalized by GNC per (query,side)",
        }

    candidates: dict[str, Any] = {}
    initial_grid = initial.reshape(GRID, GRID)
    for config_id, top_k, cutoff in FROZEN_CONFIGS:
        source, destination, offsets, confidence, _side_code = edges_by_top_k[top_k]
        config = _gnc_config(cutoff)
        gnc = solve_gnc_tls(
            source,
            destination,
            offsets,
            confidence,
            initial_grid,
            config=config,
            seed=qap_seed,
        )
        pre_qap = validate_permutation(
            gnc.grid.ravel(), name=f"{config_id}_gnc_hungarian_layout"
        )
        final_layout, qap_diagnostics = _run_production_qap(
            w4, pre_qap, seed=qap_seed
        )
        candidate = _variant_artifacts(
            output,
            stem=stem,
            label=f"{config_id}__w4_qap",
            layout=final_layout,
            denoised=denoised,
        )
        candidate.update(
            {
                "config": {"config_id": config_id, "top_k": top_k, **asdict(config)},
                "pre_qap_layout": _write_array_artifact(
                    output,
                    f"{stem}__{config_id}__gnc_hungarian.layout.npy",
                    pre_qap.astype(np.int32, copy=False),
                ),
                "continuous_positions_sha256": _array_sha256(
                    gnc.continuous_positions
                ),
                "gnc": gnc.diagnostics,
                "qap": qap_diagnostics,
            }
        )
        candidates[config_id] = candidate

    return {
        "name": name,
        "panel": panel,
        "panel_seed": int(panel_seed),
        "qap_seed": int(qap_seed),
        "clean_target_file_sha256": clean_target_file_sha256,
        "clean_target_pixel_sha256": clean_target_pixel_sha256,
        "raw_tiles_sha256": _array_sha256(raw_tiles),
        "denoised_tiles_sha256": _array_sha256(denoised),
        "score_hashes": _qap_score_hashes(c1, hbt, w4),
        "initial_layout_sha256": _array_sha256(initial),
        "initial_grid_sha256": _array_sha256(initial_grid),
        "softcycle": softcycle_diagnostics,
        "edges": edge_metadata,
        "baseline": baseline,
        "candidates": candidates,
    }


def _artifact_path(output: Path, record: Mapping[str, Any], expected_suffix: str) -> Path:
    relative = Path(str(record["path"]))
    if relative.is_absolute() or relative.parts[0] != "artifacts":
        raise RuntimeError("frozen artifact path is not canonical-relative")
    path = (output / relative).resolve()
    artifact_root = (output / "artifacts").resolve()
    if path.parent != artifact_root or not path.name.endswith(expected_suffix):
        raise RuntimeError("frozen artifact escaped the artifact directory")
    return path


def _verify_layout_artifact(output: Path, record: Mapping[str, Any], *, name: str) -> np.ndarray:
    path = _artifact_path(output, record, ".npy")
    payload = path.read_bytes()
    if _bytes_sha256(payload) != record["file_sha256"]:
        raise RuntimeError(f"frozen layout file hash drift: {path}")
    layout = _decode_layout(payload, name=name)
    if _array_sha256(layout) != record["array_sha256"]:
        raise RuntimeError(f"frozen layout value hash drift: {path}")
    return layout


def _verify_render_artifact(output: Path, record: Mapping[str, Any]) -> np.ndarray:
    path = _artifact_path(output, record, ".png")
    payload = path.read_bytes()
    if _bytes_sha256(payload) != record["file_sha256"]:
        raise RuntimeError(f"frozen render file hash drift: {path}")
    render = _decode_png(payload)
    if _array_sha256(render) != record["pixel_sha256"]:
        raise RuntimeError(f"frozen render pixel hash drift: {path}")
    return render


def _official_ssim(target: np.ndarray, prediction: np.ndarray) -> float:
    if (
        target.dtype != np.uint8
        or prediction.dtype != np.uint8
        or target.shape != (480, 480, 3)
        or prediction.shape != target.shape
    ):
        raise RuntimeError("official SSIM requires matching uint8 RGB 480x480 arrays")
    return float(
        structural_similarity(target, prediction, channel_axis=2, data_range=255)
    )


def _score_variant(
    *,
    output: Path,
    variant: Mapping[str, Any],
    target: np.ndarray,
    slot_to_target: np.ndarray,
    label: str,
) -> dict[str, Any]:
    layout = _verify_layout_artifact(output, variant["layout"], name=f"{label}_layout")
    render = _verify_render_artifact(output, variant["render"])
    geometry = layout_metrics(layout, slot_to_target)
    return {
        "valid_permutation": True,
        "ssim": _official_ssim(target, render),
        "combined_adjacency": float(geometry["combined_adjacency"]),
        "right_adjacency": float(geometry["right_adjacency"]),
        "down_adjacency": float(geometry["down_adjacency"]),
        "largest_correct_component": int(geometry["largest_correct_component"]),
        "position_accuracy": float(geometry["position_accuracy"]),
    }


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _gnc_health(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    stages = diagnostics.get("stages")
    mu_schedule = diagnostics.get("mu_schedule")
    final_stage = len(mu_schedule) - 1 if isinstance(mu_schedule, list) else -1
    final_records = (
        [record for record in stages if int(record.get("stage", -1)) == final_stage]
        if isinstance(stages, list)
        else []
    )
    checks = {
        "finite_diagnostics": _finite_tree(diagnostics),
        "positive_confidence_edges": int(diagnostics.get("positive_confidence_edges", 0)) > 0,
        "nonzero_query_groups": int(diagnostics.get("query_side_group_count", 0)) > 0,
        "zero_confidence_groups_absent": int(
            diagnostics.get("zero_confidence_group_count", -1)
        )
        == 0,
        "final_stage_present_for_every_restart": len(final_records)
        == int(diagnostics.get("restarts", -1)),
        "final_weights_noncollapsed": bool(final_records)
        and all(
            float(record.get("max_robust_weight", 0.0)) > 0.0
            and float(record.get("mean_robust_weight", 0.0)) > 1e-8
            for record in final_records
        ),
        "no_forbidden_assignments": bool(stages)
        and all(int(record.get("outside_candidate_assignments", -1)) == 0 for record in stages),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _score_frozen_records(
    output: Path, frozen_records: list[dict[str, Any]], *, data_root: Path
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for frozen in frozen_records:
        name = str(frozen["name"])
        panel = str(frozen["panel"])
        seed = int(frozen["panel_seed"])
        target_payload, target = _read_rgb_bytes(data_root / "train/targets" / name)
        if _bytes_sha256(target_payload) != frozen["clean_target_file_sha256"]:
            raise RuntimeError(f"clean target file hash drift before scoring: {name}")
        if _array_sha256(target) != frozen["clean_target_pixel_sha256"]:
            raise RuntimeError(f"clean target pixel hash drift before scoring: {name}")
        # ExactPanel defines this permutation solely from its seed.  Access is
        # intentionally delayed until every layout and render has been frozen.
        seed_mapping = np.random.default_rng(seed).permutation(TILE_COUNT).astype(
            np.int32
        )
        seed_mapping = validate_permutation(
            seed_mapping, name=f"{name}_{panel}_seed_slot_to_target"
        )
        regenerated = make_exact_panel(target, panel=panel, seed=seed)
        if _array_sha256(regenerated.slot_tiles) != frozen["raw_tiles_sha256"]:
            raise RuntimeError(f"regenerated exact panel input hash drift: {name} {panel}")
        regenerated_mapping = validate_permutation(
            regenerated.slot_to_target,
            name=f"{name}_{panel}_regenerated_slot_to_target",
        )
        if not np.array_equal(regenerated_mapping, seed_mapping):
            raise RuntimeError(f"regenerated exact panel mapping drift: {name} {panel}")
        slot_to_target = seed_mapping
        del regenerated, regenerated_mapping
        baseline = _score_variant(
            output=output,
            variant=frozen["baseline"],
            target=target,
            slot_to_target=slot_to_target,
            label=f"{name}_{panel}_baseline",
        )
        candidates: dict[str, Any] = {}
        for config_id, _, _ in FROZEN_CONFIGS:
            variant = frozen["candidates"][config_id]
            metrics = _score_variant(
                output=output,
                variant=variant,
                target=target,
                slot_to_target=slot_to_target,
                label=f"{name}_{panel}_{config_id}",
            )
            health = _gnc_health(variant["gnc"])
            candidates[config_id] = {
                **metrics,
                "ssim_delta": float(metrics["ssim"] - baseline["ssim"]),
                "adjacency_delta": float(
                    metrics["combined_adjacency"] - baseline["combined_adjacency"]
                ),
                "gnc_health": health,
            }
        scored.append(
            {
                "name": name,
                "panel": panel,
                "panel_seed": seed,
                "truth_mapping_sha256": _array_sha256(slot_to_target),
                "baseline": baseline,
                "candidates": candidates,
            }
        )
    return scored


def _summarize(scored: list[dict[str, Any]], config_id: str) -> dict[str, Any]:
    panels: dict[str, Any] = {}
    for panel in PANELS:
        records = [record for record in scored if record["panel"] == panel]
        if len(records) != len(SOURCE_NAMES):
            raise RuntimeError(f"incomplete scored panel: {panel}")
        ssim_delta = np.asarray(
            [record["candidates"][config_id]["ssim_delta"] for record in records],
            dtype=np.float64,
        )
        adjacency_delta = np.asarray(
            [record["candidates"][config_id]["adjacency_delta"] for record in records],
            dtype=np.float64,
        )
        panels[panel] = {
            "records": int(len(records)),
            "mean_baseline_ssim": float(
                np.mean([record["baseline"]["ssim"] for record in records])
            ),
            "mean_candidate_ssim": float(
                np.mean([record["candidates"][config_id]["ssim"] for record in records])
            ),
            "mean_ssim_delta": float(ssim_delta.mean()),
            "mean_adjacency_delta": float(adjacency_delta.mean()),
            "ssim_wins": int(np.sum(ssim_delta > 1e-12)),
            "ssim_ties": int(np.sum(np.abs(ssim_delta) <= 1e-12)),
            "ssim_losses": int(np.sum(ssim_delta < -1e-12)),
            "worst_ssim_delta": float(ssim_delta.min()),
        }
    source_deltas: list[dict[str, Any]] = []
    for name in SOURCE_NAMES:
        records = [record for record in scored if record["name"] == name]
        if len(records) != len(PANELS):
            raise RuntimeError(f"incomplete source-macro pair: {name}")
        delta = float(
            np.mean([record["candidates"][config_id]["ssim_delta"] for record in records])
        )
        source_deltas.append({"name": name, "mean_ssim_delta": delta})
    all_valid = all(
        record["baseline"]["valid_permutation"]
        and record["candidates"][config_id]["valid_permutation"]
        for record in scored
    )
    all_healthy = all(
        record["candidates"][config_id]["gnc_health"]["passed"]
        for record in scored
    )
    no_large_regression = all(
        record["candidates"][config_id]["ssim_delta"] >= -0.01
        for record in scored
    )
    source_values = np.asarray(
        [record["mean_ssim_delta"] for record in source_deltas], dtype=np.float64
    )
    checks = {
        "all_16_baseline_and_candidate_permutations_valid": all_valid
        and len(scored) == len(SOURCE_NAMES) * len(PANELS),
        "source_macro_mean_ssim_delta_ge_0p003": float(source_values.mean()) >= 0.003,
        "each_panel_mean_ssim_delta_ge_0p001": all(
            panels[panel]["mean_ssim_delta"] >= 0.001 for panel in PANELS
        ),
        "source_macro_wins_ge_6_of_8": int(np.sum(source_values > 1e-12)) >= 6,
        "each_panel_mean_adjacency_delta_ge_0p005": all(
            panels[panel]["mean_adjacency_delta"] >= 0.005 for panel in PANELS
        ),
        "no_record_ssim_delta_lt_minus_0p01": no_large_regression,
        "all_gnc_coordinates_weights_and_projections_healthy": all_healthy,
    }
    return {
        "config_id": config_id,
        "panels": panels,
        "source_macro": {
            "records": source_deltas,
            "mean_ssim_delta": float(source_values.mean()),
            "wins": int(np.sum(source_values > 1e-12)),
            "ties": int(np.sum(np.abs(source_values) <= 1e-12)),
            "losses": int(np.sum(source_values < -1e-12)),
        },
        "gate": {"passed": all(checks.values()), "logic": "all_of", "checks": checks},
    }


def _selection(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [
        config_id for config_id, summary in summaries.items() if summary["gate"]["passed"]
    ]
    selected = min(
        eligible,
        key=lambda config_id: (
            -min(
                summaries[config_id]["panels"][panel]["mean_ssim_delta"]
                for panel in PANELS
            ),
            -summaries[config_id]["source_macro"]["mean_ssim_delta"],
            -summaries[config_id]["source_macro"]["wins"],
            next(top_k for key, top_k, _ in FROZEN_CONFIGS if key == config_id),
            next(cutoff for key, _, cutoff in FROZEN_CONFIGS if key == config_id),
            config_id,
        ),
        default=None,
    )
    return {
        "eligible": sorted(eligible),
        "selected": selected,
        "status": "go_fresh_confirmation_only" if selected is not None else "stop_no_eligible_config",
        "safe_for_submission": False,
        "go_authorizes": "one new source-disjoint fresh confirmation only",
        "go_does_not_authorize": [
            "V4 access",
            "assembly_final_audit access",
            "submission generation",
            "retuning on this reusable slice",
        ],
    }


def _assert_pins_unchanged(
    initial_code: Mapping[str, str],
    initial_assets: Mapping[str, Mapping[str, str]],
    paths: Mapping[str, Path],
) -> None:
    if _code_pins(paths) != initial_code:
        raise RuntimeError("code pin drift during calibration")
    if _asset_pins(paths) != initial_assets:
        raise RuntimeError("asset pin drift during calibration")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.denoise_batch_size <= 0 or args.classical_chunk_size <= 0:
        raise SystemExit("batch and chunk sizes must be positive")
    output = _prepare_output_dir(args.output_dir)
    _protocol, paths = _validate_protocol_and_assets(args)
    initial_code_pins = _code_pins(paths)
    initial_asset_pins = _asset_pins(paths)
    started = time.time()

    panel_marker_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "gnc_tls_exact_panel_input_construction_started",
        "source_split": SOURCE_SPLIT,
        "source_offset": SOURCE_OFFSET,
        "source_names_sha256": SOURCE_NAMES_SHA256,
        "panels": list(PANELS),
        "clean_target_use": "deterministic exact corrupted-input construction only",
        "evaluator_must_not_read_or_pass_truth_mapping_to_predictor": True,
        "solver_accepts_target": False,
    }
    _atomic_envelope(output / PANEL_MARKER, panel_marker_payload)

    restorer, device, _ = load_restorer(
        paths["denoiser"], device=args.device, state="ema"
    )
    embedding, _ = load_embedding_checkpoint(paths["hbt"], device=device)
    for model in (restorer, embedding):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    frozen_records: list[dict[str, Any]] = []
    for source_index, name in enumerate(SOURCE_NAMES):
        for panel in PANELS:
            panel_seed = per_source_seed(MASTER_SEED, f"gnc-tls-{panel}", name, 0)
            clean_payload, clean = _read_rgb_bytes(
                paths["data_root"] / "train/targets" / name
            )
            clean_target_file_sha256 = _bytes_sha256(clean_payload)
            clean_target_pixel_sha256 = _array_sha256(clean)
            exact_panel = make_exact_panel(clean, panel=panel, seed=panel_seed)
            # Deliberately extract only the solver input.  The ExactPanel truth
            # fields are neither accessed nor retained before the freeze.
            raw_tiles = np.ascontiguousarray(exact_panel.slot_tiles)
            del exact_panel, clean, clean_payload
            with torch.inference_mode():
                frozen_records.append(
                    _predict_record(
                        output=output,
                        name=name,
                        panel=panel,
                        panel_seed=panel_seed,
                        raw_tiles=raw_tiles,
                        clean_target_file_sha256=clean_target_file_sha256,
                        clean_target_pixel_sha256=clean_target_pixel_sha256,
                        restorer=restorer,
                        embedding=embedding,
                        device=device,
                        args=args,
                    )
                )
            print(
                json.dumps(
                    {
                        "stage": "input_only_prediction",
                        "source_index": source_index,
                        "name": name,
                        "panel": panel,
                        "frozen_records": len(frozen_records),
                        "total": len(SOURCE_NAMES) * len(PANELS),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    frozen_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "gnc_tls_sync_frozen_input_only_predictions",
        "safe_for_submission": False,
        "source_provenance": {
            "split": SOURCE_SPLIT,
            "offset": SOURCE_OFFSET,
            "count": len(SOURCE_NAMES),
            "names": list(SOURCE_NAMES),
            "names_sha256": SOURCE_NAMES_SHA256,
            "freshness": "reusable_exposed_calibration_not_fresh",
            "known_prior_use": "positional diffusion development calibration",
            "whole_source_disjoint_from_recent_edge_development_gates": True,
        },
        "panels": list(PANELS),
        "master_seed": MASTER_SEED,
        "production": {
            "authoritative_evaluator": "scripts/evaluate_qap_weight_confirmation.py",
            "common_solver": EXPECTED_COMMON_SOLVER,
            "baseline": "HBT soft-cycle -> w4 QAP 25x2",
            "candidate": "four-side w4 GNC-TLS Hungarian -> same w4 QAP 25x2",
            "renderer": "same EMA TileNAF denoised uint8 tiles",
        },
        "frozen_configs": [
            {"config_id": config_id, "top_k": top_k, "robust_cutoff": cutoff}
            for config_id, top_k, cutoff in FROZEN_CONFIGS
        ],
        "anti_leakage": {
            "solver_accepts_target": False,
            "exact_panel_builder_constructs_truth_mapping": True,
            "truth_mapping_accessed_by_evaluator_scoring_before_frozen_manifest": False,
            "truth_mapping_passed_to_predictor_before_frozen_manifest": False,
            "truth_present_in_solver_input": False,
            "clean_target_pre_freeze_use": "exact corrupted panel construction only",
            "candidate_graph_oracle_loaded": False,
            "fixture_label_loaded": False,
            "v4_loaded": False,
            "assembly_final_audit_loaded": False,
            "forbidden_paths_fail_closed": list(FORBIDDEN_PATH_TOKENS),
        },
        "code_pins": initial_code_pins,
        "asset_pins": initial_asset_pins,
        "records": frozen_records,
    }
    frozen_envelope = _atomic_envelope(output / FROZEN_MANIFEST, frozen_payload)
    frozen_file_sha256 = _file_sha256(output / FROZEN_MANIFEST)
    verified_frozen = _load_exact_envelope(output / FROZEN_MANIFEST)
    if verified_frozen != frozen_envelope:
        raise RuntimeError("frozen manifest readback mismatch")
    reloaded_records = verified_frozen["payload"].get("records")
    if not isinstance(reloaded_records, list) or len(reloaded_records) != len(frozen_records):
        raise RuntimeError("frozen manifest record set is incomplete")
    for frozen in reloaded_records:
        _verify_layout_artifact(
            output, frozen["baseline"]["layout"], name="baseline_pre_score_verify"
        )
        _verify_render_artifact(output, frozen["baseline"]["render"])
        for config_id, _, _ in FROZEN_CONFIGS:
            candidate = frozen["candidates"][config_id]
            _verify_layout_artifact(
                output, candidate["pre_qap_layout"], name="gnc_pre_qap_pre_score_verify"
            )
            _verify_layout_artifact(
                output, candidate["layout"], name="candidate_pre_score_verify"
            )
            _verify_render_artifact(output, candidate["render"])
    _assert_pins_unchanged(initial_code_pins, initial_asset_pins, paths)

    scoring_marker_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "gnc_tls_sync_scoring_truth_access_started",
        "frozen_manifest_file_sha256": frozen_file_sha256,
        "frozen_manifest_payload_sha256": frozen_envelope["payload_sha256"],
        "all_layouts_and_renders_verified_before_marker": True,
        "truth_use": "post-freeze permutation-aware metrics and official RGB SSIM only",
    }
    _atomic_envelope(output / SCORING_MARKER, scoring_marker_payload)
    scored_records = _score_frozen_records(
        output, reloaded_records, data_root=paths["data_root"]
    )
    summaries = {
        config_id: _summarize(scored_records, config_id)
        for config_id, _, _ in FROZEN_CONFIGS
    }
    selection = _selection(summaries)
    _assert_pins_unchanged(initial_code_pins, initial_asset_pins, paths)

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "gnc_tls_sync_calibration_report",
        "status": selection["status"],
        "safe_for_submission": False,
        "submission_ready": False,
        "source_provenance": frozen_payload["source_provenance"],
        "panels": list(PANELS),
        "master_seed": MASTER_SEED,
        "metric": {
            "name": "RGB_SSIM",
            "call": "skimage.metrics.structural_similarity(target_rgb_uint8, frozen_render_rgb_uint8, channel_axis=2, data_range=255)",
            "unit": "paired whole source-panel; source macro averages both panels",
        },
        "production": frozen_payload["production"],
        "frozen_configs": frozen_payload["frozen_configs"],
        "frozen_input_manifest": {
            "path": FROZEN_MANIFEST,
            "file_sha256": frozen_file_sha256,
            "payload_sha256": frozen_envelope["payload_sha256"],
        },
        "gate_contract": {
            "logic": "all_of",
            "valid_permutations": "all 16 baseline and candidate records",
            "source_macro_mean_post_qap_ssim_delta_min": 0.003,
            "each_panel_mean_post_qap_ssim_delta_min": 0.001,
            "source_macro_wins_min": 6,
            "each_panel_mean_adjacency_delta_min": 0.005,
            "individual_ssim_delta_min": -0.01,
            "gnc": "finite, positive query groups, noncollapsed final weights, no forbidden Hungarian assignments",
        },
        "selection": selection,
        "summaries": summaries,
        "records": scored_records,
        "anti_leakage": {
            **frozen_payload["anti_leakage"],
            "scoring_started_after_frozen_manifest": True,
            "frozen_manifest_file_sha256": frozen_file_sha256,
            "frozen_manifest_payload_sha256": frozen_envelope["payload_sha256"],
            "panel_construction_marker": PANEL_MARKER,
            "scoring_truth_access_marker": SCORING_MARKER,
        },
        "code_pins": initial_code_pins,
        "asset_pins": initial_asset_pins,
        "environment": _hardware(),
        "seconds": float(time.time() - started),
    }
    _atomic_envelope(output / REPORT_NAME, report)
    print(
        json.dumps(
            {
                "status": selection["status"],
                "selected": selection["selected"],
                "report": str(output / REPORT_NAME),
                "safe_for_submission": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
