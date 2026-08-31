#!/usr/bin/env python3
"""Train and gate one vectorized raw+adapter1600+DINO neighbour verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from aiijc_puzzle.dinov2_boundary_matcher import (
    extract_patch_tokens,
    load_official_dinov2,
    scores_from_patch_tokens,
)
from aiijc_puzzle.fullres_boundary_denoiser import restore_matcher_view
from aiijc_puzzle.fullres_retrieval_adapter import FullResolutionRetrievalAdapter
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.socket_sorter_production import load_socket_checkpoint
from aiijc_puzzle.synthetic_socket_evaluation import (
    ExactSyntheticReference,
    SyntheticSocketInput,
    exact_local_retrieval_metrics,
    make_exact_synthetic_case,
    names_digest,
)
from aiijc_puzzle.tri_emitter_edge_verifier import (
    TOP_K,
    CandidatePool,
    TriEmitterEdgeVerifier,
    build_candidate_pool,
    candidate_pool_digest,
    compress_dino_boundary_tokens,
    fixed_dino_projection,
    ordered_raw_side_sequences,
    sparse_reciprocal_evidence,
    verifier_contract,
)

try:
    from scripts import run_fullres_boundary_denoiser as boundary
    from scripts import run_fullres_retrieval_adapter as roster
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_fullres_boundary_denoiser as boundary
    import run_fullres_retrieval_adapter as roster

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCH_CONFIG = (
    PROJECT_ROOT / "configs/tri_emitter_edge_verifier_architecture_preregistered_v1.json"
)
DEFAULT_FULL_CONFIG = (
    PROJECT_ROOT / "configs/tri_emitter_edge_verifier_full_preregistered_v1.json"
)
DEFAULT_CAPACITY_OUTPUT = (
    PROJECT_ROOT / "outputs/tri-emitter-edge-verifier/capacity4x4-v1"
)
DEFAULT_BENCHMARK_OUTPUT = (
    PROJECT_ROOT / "outputs/tri-emitter-edge-verifier/fullboard-benchmark-v1"
)
DEFAULT_FULL_OUTPUT = (
    PROJECT_ROOT / "outputs/tri-emitter-edge-verifier/fit32-draw2-s3-local16-v1"
)
SOCKET_CHECKPOINT = roster.DEFAULT_SOCKET
ADAPTER_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/fullres-retrieval-adapter/scale1600-local16-v1/adapter_step1600.pt"
)
DINO_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts/foundation-semantics/dinov2-vits14-official/dinov2_vits14_pretrain.pth"
)
EXISTING_ADAPTER_ARCHIVE = (
    PROJECT_ROOT
    / "outputs/fullres-retrieval-adapter/scale1600-local16-v1/local16/"
    "frozen-target-free-retrieval.npz"
)
EXISTING_ADAPTER_METADATA = EXISTING_ADAPTER_ARCHIVE.with_suffix(".json")
EXISTING_DINO_ARCHIVE = (
    PROJECT_ROOT
    / "outputs/dinov2-boundary-candidate-screen/opened-local16-v1/frozen-target-free-candidates.npz"
)
EXISTING_DINO_METADATA = EXISTING_DINO_ARCHIVE.with_suffix(".json")
TERMINAL_CONFIG = (
    PROJECT_ROOT / "configs/tri_emitter_edge_verifier_terminal_preregistered_v1.json"
)
TRAIN_SEED = 20260913
FIT_CASE_SEED = 20260914
EVAL_CASE_SEED = 20260908
FIT_DRAWS = (0, 1)
TRAIN_EPOCHS = 3
QUERY_BATCH = 96
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
CAPACITY_STEPS = 200
CAPACITY_R1_MINIMUM = 0.90
CAPACITY_LOSS_RATIO_MAXIMUM = 0.25
LOCAL_GATE = {
    "pooled_r1_gain_minimum": 0.01,
    "pooled_r5_gain_minimum": 0.0,
    "matched_reciprocal_precision_gain_minimum": 0.005,
    "matched_reciprocal_coverage_minimum": 0.03,
}
TERMINAL_GATE = {
    "pooled_r1_gain_minimum": 0.005,
    "pooled_r5_gain_minimum": 0.0,
    "matched_reciprocal_precision_gain_minimum": 0.0,
    "matched_reciprocal_coverage_minimum": 0.03,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("capacity", "benchmark", "full"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=roster.DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=roster.DEFAULT_TARGETS)
    parser.add_argument("--socket-checkpoint", type=Path, default=SOCKET_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **arrays)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(resolved),
    }


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError("signed tri-emitter verifier config is missing")
    digest = sha256_file(resolved)
    if sidecar.read_text(encoding="utf-8").split()[0] != digest:
        raise RuntimeError("tri-emitter verifier config sidecar mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("status") != "signed-fixed-protocol":
        raise RuntimeError("tri-emitter verifier protocol is not signed/fixed")
    for artifact in config["frozen_inputs"].values():
        target = PROJECT_ROOT / artifact["path"]
        if not target.is_file() or sha256_file(target) != artifact["sha256"]:
            raise RuntimeError(f"frozen tri-emitter input changed: {target}")
    return config, digest


def _device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def _load_adapter(device: torch.device) -> FullResolutionRetrievalAdapter:
    payload = torch.load(ADAPTER_CHECKPOINT, map_location="cpu", weights_only=True)
    if payload.get("step") != 1600:
        raise RuntimeError("adapter1600 checkpoint step changed")
    model = FullResolutionRetrievalAdapter().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _truth_by_anchor(reference: np.ndarray, *, axis: int) -> np.ndarray:
    layout = np.asarray(reference, dtype=np.int64)
    count = len(layout)
    grid = round(math.sqrt(count))
    result = np.full(count, -1, dtype=np.int32)
    if axis == 0:
        positions = np.arange(count).reshape(grid, grid)[:, :-1].ravel()
        neighbour = positions + 1
    elif axis == 1:
        positions = np.arange(count).reshape(grid, grid)[:-1].ravel()
        neighbour = positions + grid
    else:
        raise ValueError("axis must be 0 or 1")
    result[layout[positions]] = layout[neighbour]
    return result


@torch.inference_mode()
def _socket_scores(
    socket: Any,
    tiles: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(tiles)
    grid = round(math.sqrt(count))
    if grid * grid != count:
        raise ValueError("tile count must be square")
    output = socket.model(boundary._tensor(tiles, device).unsqueeze(0), grid=grid)
    normaliser = math.log(float(count + grid))
    return (
        np.ascontiguousarray(
            output.right_log_assignment[0, :count, :count].float().cpu().numpy()
            + normaliser,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            output.down_log_assignment[0, :count, :count].float().cpu().numpy()
            + normaliser,
            dtype=np.float32,
        ),
    )


def _target_slots(pool: CandidatePool, reference: np.ndarray) -> np.ndarray:
    result = np.full(pool.candidates.shape[:2], -1, dtype=np.int16)
    for axis in range(2):
        truth = _truth_by_anchor(reference, axis=axis)
        for anchor in np.flatnonzero(truth >= 0):
            match = np.flatnonzero(
                pool.valid[axis, anchor]
                & (pool.candidates[axis, anchor] == truth[anchor])
            )
            if len(match):
                result[axis, anchor] = int(match[0])
    return result


def _extract_case(
    item: SyntheticSocketInput,
    *,
    socket: Any,
    adapter: FullResolutionRetrievalAdapter,
    dino: torch.nn.Module,
    projection: np.ndarray,
    device: torch.device,
    executor: ThreadPoolExecutor,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    started = perf_counter()
    dino_started = perf_counter()
    dino_future = executor.submit(
        extract_patch_tokens,
        dino,
        item.tiles,
        device=torch.device("cpu"),
        batch_size=64,
    )
    socket_started = perf_counter()
    raw_right, raw_down = _socket_scores(socket, item.tiles, device=device)
    raw_seconds = perf_counter() - socket_started
    adapter_started = perf_counter()
    adapted = restore_matcher_view(
        adapter,
        item.tiles,
        device=device,
        batch_size=len(item.tiles),
    )
    adapt_seconds = perf_counter() - adapter_started
    socket_started = perf_counter()
    adapt_right, adapt_down = _socket_scores(socket, adapted, device=device)
    adapt_socket_seconds = perf_counter() - socket_started
    tokens = dino_future.result()
    dino_seconds = perf_counter() - dino_started
    dino_scores = scores_from_patch_tokens(tokens)
    pool = build_candidate_pool(
        {
            "raw_d64_ot": (raw_right, raw_down),
            "adapter_step1600": (adapt_right, adapt_down),
            "dinov2_boundary": (dino_scores.right, dino_scores.down),
        },
        top_k=min(TOP_K, len(item.tiles) - 1),
    )
    arrays = {
        "raw_sides": ordered_raw_side_sequences(item.tiles),
        "dino_sides": compress_dino_boundary_tokens(tokens, projection),
        "candidates": pool.candidates,
        "valid": pool.valid,
        "auxiliary": pool.auxiliary,
        "raw_baseline": pool.raw_baseline,
        "emitter_topk": pool.emitter_topk,
        "raw_dense": np.stack((raw_right, raw_down)),
    }
    return arrays, {
        "raw_socket_seconds": raw_seconds,
        "adapter_seconds": adapt_seconds,
        "adapted_socket_seconds": adapt_socket_seconds,
        "dino_cpu_concurrent_seconds": dino_seconds,
        "total_seconds": perf_counter() - started,
        "union_identity_digest": pool.identity_digest,
    }


def _to_device_case(
    values: Mapping[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "raw_sides": torch.from_numpy(values["raw_sides"].astype(np.float32)).to(device),
        "dino_sides": torch.from_numpy(values["dino_sides"].astype(np.float32)).to(device),
        "candidates": torch.from_numpy(values["candidates"].astype(np.int64)).to(device),
        "valid": torch.from_numpy(values["valid"].astype(bool)).to(device),
        "auxiliary": torch.from_numpy(values["auxiliary"].astype(np.float32)).to(device),
        "raw_baseline": torch.from_numpy(values["raw_baseline"].astype(np.float32)).to(device),
    }


def _model_rows(
    model: TriEmitterEdgeVerifier,
    case: Mapping[str, torch.Tensor],
    axes: np.ndarray,
    anchors: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.from_numpy(np.ascontiguousarray(axes, dtype=np.int64)).to(
        case["raw_sides"].device
    )
    query = torch.from_numpy(np.ascontiguousarray(anchors, dtype=np.int64)).to(
        case["raw_sides"].device
    )
    return model(
        case["raw_sides"],
        case["dino_sides"],
        query,
        case["candidates"][axis, query],
        case["valid"][axis, query],
        axis,
        case["auxiliary"][axis, query],
        case["raw_baseline"][axis, query],
    )


@torch.inference_mode()
def _score_case(
    model: TriEmitterEdgeVerifier,
    values: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    query_batch: int = 64,
) -> np.ndarray:
    case = _to_device_case(values, device)
    count = values["candidates"].shape[1]
    logits = np.full(values["candidates"].shape, -1e4, dtype=np.float32)
    model.eval()
    for axis in range(2):
        for start in range(0, count, query_batch):
            anchors = np.arange(start, min(start + query_batch, count), dtype=np.int64)
            axes = np.full(len(anchors), axis, dtype=np.int64)
            current, _ = _model_rows(model, case, axes, anchors)
            logits[axis, anchors] = current.float().cpu().numpy()
    return logits


def _rank_topk(
    candidates: np.ndarray,
    valid: np.ndarray,
    logits: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    count = candidates.shape[0]
    result = np.empty((count, k), dtype=np.int32)
    for anchor in range(count):
        ids = candidates[anchor, valid[anchor]]
        scores = logits[anchor, valid[anchor]]
        order = np.argsort(-scores, kind="stable")[:k]
        if len(order) != k:
            raise RuntimeError("candidate union is smaller than requested learned top-k")
        result[anchor] = ids[order]
    return result


def _capacity_crop(board: Any) -> np.ndarray:
    indices = np.array([row * 24 + column for row in range(4) for column in range(4)])
    return np.ascontiguousarray(board.tiles[indices])


def _make_models(args: argparse.Namespace) -> tuple[Any, Any, Any, np.ndarray, torch.device]:
    device = _device(args.device)
    socket = load_socket_checkpoint(args.socket_checkpoint, device=device)
    adapter = _load_adapter(device)
    dino = load_official_dinov2(DINO_CHECKPOINT, device=torch.device("cpu"))
    projection = fixed_dino_projection(384)
    return socket, adapter, dino, projection, device


def run_capacity(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_sha: str,
    output: Path,
) -> dict[str, Any]:
    protocol, fit_boards, _, _ = roster._load_protocol(args)
    socket, adapter, dino, projection, device = _make_models(args)
    clean = _capacity_crop(fit_boards[0])
    item, reference = make_exact_synthetic_case(
        clean,
        source_filename=f"{fit_boards[0].filename}#top-left-4x4",
        draw_index=0,
        seed=FIT_CASE_SEED + 100,
    )
    output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        values, runtime = _extract_case(
            item,
            socket=socket,
            adapter=adapter,
            dino=dino,
            projection=projection,
            device=device,
            executor=executor,
        )
    targets = _target_slots(
        CandidatePool(
            candidates=values["candidates"],
            valid=values["valid"],
            auxiliary=values["auxiliary"],
            raw_baseline=values["raw_baseline"],
            emitter_topk=values["emitter_topk"],
            identity_digest=str(runtime["union_identity_digest"]),
        ),
        reference.tile_at_position,
    )
    queries = np.argwhere(targets >= 0)
    if len(queries) != 24:
        raise RuntimeError("4x4 union must contain all 24 exact directed neighbours")
    torch.manual_seed(TRAIN_SEED)
    model = TriEmitterEdgeVerifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    case = _to_device_case(values, device)
    target = torch.from_numpy(targets[queries[:, 0], queries[:, 1]].astype(np.int64)).to(device)
    losses: list[float] = []
    model.train()
    for _ in range(CAPACITY_STEPS):
        logits, _ = _model_rows(model, case, queries[:, 0], queries[:, 1])
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    logits = _score_case(model, values, device=device, query_batch=32)
    learned = np.stack(
        [
            _rank_topk(
                values["candidates"][axis],
                values["valid"][axis],
                logits[axis],
                k=15,
            )
            for axis in range(2)
        ]
    )
    metrics = exact_local_retrieval_metrics(
        learned[0], learned[1], reference.tile_at_position, ks=(1, 5, 15)
    )
    ratio = losses[-1] / losses[0]
    passed = bool(
        metrics["pooled_r1"] >= CAPACITY_R1_MINIMUM
        and ratio <= CAPACITY_LOSS_RATIO_MAXIMUM
    )
    checkpoint = output / "capacity_model.pt"
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "contract": verifier_contract(model),
            "capacity_only_not_reusable_for_full": True,
            "config_sha256": config_sha,
        },
        checkpoint,
    )
    report = {
        "schema": "aiijc-tri-emitter-verifier-capacity-v1",
        "status": "pass" if passed else "fail-stop",
        "protocol": protocol,
        "config_sha256": config_sha,
        "case": {
            "source": item.source_filename,
            "draw_index": item.draw_index,
            "query_count": len(queries),
        },
        "runtime": runtime,
        "training": {
            "steps": CAPACITY_STEPS,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "final_to_initial_ratio": ratio,
        },
        "retrieval": metrics,
        "gate": {
            "r1_minimum": CAPACITY_R1_MINIMUM,
            "loss_ratio_maximum": CAPACITY_LOSS_RATIO_MAXIMUM,
            "passed": passed,
        },
        "artifacts": {
            "config": _record(args.config),
            "checkpoint": _record(checkpoint),
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/tri_emitter_edge_verifier.py"
            ),
            "runner": _record(Path(__file__)),
        },
        "competition_test_accessed": False,
        "decoder_run": False,
    }
    _write_json(output / "report.json", report)
    return report


def _fit_cache_case(
    path: Path,
    values: Mapping[str, np.ndarray],
    targets: np.ndarray,
) -> None:
    _write_npz(
        path,
        {
            "raw_sides": values["raw_sides"].astype(np.float16),
            "dino_sides": values["dino_sides"].astype(np.float16),
            "candidates": values["candidates"].astype(np.int32),
            "valid": values["valid"].astype(bool),
            "auxiliary": values["auxiliary"].astype(np.float16),
            "raw_baseline": values["raw_baseline"].astype(np.float16),
            "emitter_topk": values["emitter_topk"].astype(np.int32),
            "target_slots": targets.astype(np.int16),
        },
    )


def _load_fit_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.ascontiguousarray(archive[key]) for key in archive.files}


def run_benchmark(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_sha: str,
    output: Path,
) -> dict[str, Any]:
    _, fit_boards, _, _ = roster._load_protocol(args)
    socket, adapter, dino, projection, device = _make_models(args)
    item, reference = make_exact_synthetic_case(
        fit_boards[0].tiles,
        source_filename=fit_boards[0].filename,
        draw_index=FIT_DRAWS[0],
        seed=FIT_CASE_SEED,
    )
    output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        values, prep = _extract_case(
            item,
            socket=socket,
            adapter=adapter,
            dino=dino,
            projection=projection,
            device=device,
            executor=executor,
        )
    pool = CandidatePool(
        candidates=values["candidates"],
        valid=values["valid"],
        auxiliary=values["auxiliary"],
        raw_baseline=values["raw_baseline"],
        emitter_topk=values["emitter_topk"],
        identity_digest=str(prep["union_identity_digest"]),
    )
    targets = _target_slots(pool, reference.tile_at_position)
    queries = np.argwhere(targets >= 0)
    generator = np.random.default_rng(TRAIN_SEED)
    generator.shuffle(queries)
    queries = queries[:QUERY_BATCH]
    torch.manual_seed(TRAIN_SEED)
    model = TriEmitterEdgeVerifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    case = _to_device_case(values, device)
    target = torch.from_numpy(targets[queries[:, 0], queries[:, 1]].astype(np.int64)).to(device)
    updates: list[float] = []
    for _ in range(5):
        started = perf_counter()
        logits, _ = _model_rows(model, case, queries[:, 0], queries[:, 1])
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        updates.append(perf_counter() - started)
    batches_per_case = math.ceil(len(np.argwhere(targets >= 0)) / QUERY_BATCH)
    projected_steps = batches_per_case * 32 * len(FIT_DRAWS) * TRAIN_EPOCHS
    estimate = {
        "fit_prep_seconds": prep["total_seconds"] * 32 * len(FIT_DRAWS),
        "training_seconds": float(np.mean(updates[1:])) * projected_steps,
        "local_prep_seconds": prep["total_seconds"] * 16,
        "projected_train_steps": projected_steps,
        "total_before_local_score_seconds": (
            prep["total_seconds"] * 32 * len(FIT_DRAWS)
            + float(np.mean(updates[1:])) * projected_steps
        ),
    }
    report = {
        "schema": "aiijc-tri-emitter-verifier-benchmark-v1",
        "status": "complete-no-quality-metric",
        "config_sha256": config_sha,
        "prep": prep,
        "eligible_queries": int(np.count_nonzero(targets >= 0)),
        "mean_warm_update_seconds": float(np.mean(updates[1:])),
        "runtime_estimate": estimate,
        "competition_test_accessed": False,
        "decoder_run": False,
    }
    _write_json(output / "report.json", report)
    return report


def _prepare_fit_cache(
    fit_boards: Sequence[Any],
    *,
    output: Path,
    socket: Any,
    adapter: FullResolutionRetrievalAdapter,
    dino: torch.nn.Module,
    projection: np.ndarray,
    device: torch.device,
) -> tuple[list[Path], list[dict[str, Any]]]:
    cache_dir = output / "fit-cache"
    cache_dir.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for source_index, board in enumerate(fit_boards):
            for draw_index in FIT_DRAWS:
                item, reference = make_exact_synthetic_case(
                    board.tiles,
                    source_filename=board.filename,
                    draw_index=draw_index,
                    seed=FIT_CASE_SEED,
                )
                values, runtime = _extract_case(
                    item,
                    socket=socket,
                    adapter=adapter,
                    dino=dino,
                    projection=projection,
                    device=device,
                    executor=executor,
                )
                pool = CandidatePool(
                    candidates=values["candidates"],
                    valid=values["valid"],
                    auxiliary=values["auxiliary"],
                    raw_baseline=values["raw_baseline"],
                    emitter_topk=values["emitter_topk"],
                    identity_digest=str(runtime["union_identity_digest"]),
                )
                targets = _target_slots(pool, reference.tile_at_position)
                path = cache_dir / f"source_{source_index:02d}_draw_{draw_index}.npz"
                _fit_cache_case(path, values, targets)
                paths.append(path)
                rows.append(
                    {
                        "path": str(path.relative_to(PROJECT_ROOT)),
                        "sha256": sha256_file(path),
                        "source_filename": board.filename,
                        "draw_index": draw_index,
                        "case_id": item.case_id,
                        "dirty_sha256": hashlib.sha256(item.tiles.tobytes()).hexdigest(),
                        "eligible_queries": int(np.count_nonzero(targets >= 0)),
                        "runtime": runtime,
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "fit_cache",
                            "case": len(paths),
                            "count": len(fit_boards) * len(FIT_DRAWS),
                            "source": board.filename,
                            "draw": draw_index,
                            "eligible_queries": rows[-1]["eligible_queries"],
                            "seconds": runtime["total_seconds"],
                        }
                    ),
                    flush=True,
                )
    return paths, rows


def _train_full(
    cache_paths: Sequence[Path],
    *,
    device: torch.device,
    config_sha: str,
    protocol: Mapping[str, Any],
    output: Path,
) -> tuple[TriEmitterEdgeVerifier, dict[str, Any], Path]:
    torch.manual_seed(TRAIN_SEED)
    model = TriEmitterEdgeVerifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    eligible = []
    for path in cache_paths:
        values = _load_fit_cache(path)
        eligible.append(int(np.count_nonzero(values["target_slots"] >= 0)))
    total_steps = TRAIN_EPOCHS * sum(math.ceil(value / QUERY_BATCH) for value in eligible)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_steps, eta_min=LEARNING_RATE * 0.05
    )
    generator = np.random.default_rng(TRAIN_SEED + 1)
    losses: list[float] = []
    history: list[dict[str, Any]] = []
    step = 0
    started = perf_counter()
    model.train()
    for epoch in range(TRAIN_EPOCHS):
        order = generator.permutation(len(cache_paths))
        for case_index in order:
            values = _load_fit_cache(cache_paths[int(case_index)])
            queries = np.argwhere(values["target_slots"] >= 0)
            generator.shuffle(queries)
            case = _to_device_case(values, device)
            for start in range(0, len(queries), QUERY_BATCH):
                batch = queries[start : start + QUERY_BATCH]
                target = torch.from_numpy(
                    values["target_slots"][batch[:, 0], batch[:, 1]].astype(np.int64)
                ).to(device)
                update_started = perf_counter()
                logits, _ = _model_rows(model, case, batch[:, 0], batch[:, 1])
                loss = F.cross_entropy(logits, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                optimizer.step()
                scheduler.step()
                if device.type == "mps":
                    torch.mps.synchronize()
                step += 1
                losses.append(float(loss.detach().cpu()))
                if step == 1 or step % 100 == 0 or step == total_steps:
                    row = {
                        "step": step,
                        "epoch": epoch,
                        "loss": float(np.mean(losses[-min(100, len(losses)) :])),
                        "grad_norm": grad_norm,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_seconds": perf_counter() - started,
                        "last_update_seconds": perf_counter() - update_started,
                    }
                    history.append(row)
                    print(json.dumps({"event": "train", **row}), flush=True)
    checkpoint = output / "tri_emitter_edge_verifier.pt"
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "contract": verifier_contract(model),
            "config_sha256": config_sha,
            "selection": {
                "train_filenames": protocol["fit_filenames"],
                "train_digest": protocol["fit_digest"],
                "lineage_train_filenames": sorted(protocol["fit_filenames"]),
                "lineage_train_digest": names_digest(
                    protocol["fit_filenames"], sort_names=True
                ),
                "lineage_exposed_filenames": sorted(protocol["fit_filenames"]),
                "lineage_exposed_digest": names_digest(
                    protocol["fit_filenames"], sort_names=True
                ),
            },
            "training": {
                "seed": TRAIN_SEED,
                "fit_case_seed": FIT_CASE_SEED,
                "draw_indices": list(FIT_DRAWS),
                "epochs": TRAIN_EPOCHS,
                "steps": total_steps,
                "query_batch": QUERY_BATCH,
                "history": history,
            },
        },
        checkpoint,
    )
    return model, {
        "steps": total_steps,
        "seconds": perf_counter() - started,
        "history": history,
        "initial_loss": losses[0],
        "final_100_mean_loss": float(np.mean(losses[-100:])),
    }, checkpoint


def _expected_local_emitters() -> tuple[list[dict[str, Any]], Any, Any]:
    adapter_rows = json.loads(EXISTING_ADAPTER_METADATA.read_text(encoding="utf-8"))["rows"]
    dino_rows = json.loads(EXISTING_DINO_METADATA.read_text(encoding="utf-8"))["rows"]
    if len(adapter_rows) != len(dino_rows):
        raise RuntimeError("existing opened local emitter panels are misaligned")
    return adapter_rows, np.load(EXISTING_ADAPTER_ARCHIVE), np.load(EXISTING_DINO_ARCHIVE)


def _check_existing_local_identities(
    index: int,
    item: SyntheticSocketInput,
    emitter_topk: np.ndarray,
    expected: tuple[list[dict[str, Any]], Any, Any],
) -> None:
    rows, adapter_archive, dino_archive = expected
    row = rows[index]
    for key, value in (
        ("case_id", item.case_id),
        ("source_filename", item.source_filename),
        ("draw_index", item.draw_index),
        ("dirty_sha256", hashlib.sha256(item.tiles.tobytes()).hexdigest()),
    ):
        if row[key] != value:
            raise RuntimeError(f"existing local identity mismatch: {key}")
    prefix = row["prefix"]
    for axis_index, axis in enumerate(("right", "down")):
        raw = adapter_archive[f"{prefix}__candidate__raw_d64_ot__{axis}"]
        adapted = adapter_archive[f"{prefix}__candidate__adapter_step1600__{axis}"]
        dino = dino_archive[f"{prefix}__candidate__dinov2_boundary__{axis}"]
        for observed, frozen in zip(
            emitter_topk[:, axis_index], (raw, adapted, dino), strict=True
        ):
            if not np.array_equal(observed, frozen):
                raise RuntimeError("recomputed local emitter top32 identities changed")


def _freeze_panel(
    records: Sequence[dict[str, Any]],
    *,
    panel_name: str,
    targets_path: Path,
    model: TriEmitterEdgeVerifier,
    socket: Any,
    adapter: FullResolutionRetrievalAdapter,
    dino: torch.nn.Module,
    projection: np.ndarray,
    device: torch.device,
    output: Path,
    check_opened_local: bool,
) -> tuple[list[dict[str, Any]], dict[str, ExactSyntheticReference], dict[str, Any]]:
    boards = boundary._prepare_boards(tuple(records), targets_path)
    references: dict[str, ExactSyntheticReference] = {}
    cases: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    expected = _expected_local_emitters() if check_opened_local else None
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            for index, board in enumerate(boards):
                item, reference = make_exact_synthetic_case(
                    board.tiles,
                    source_filename=board.filename,
                    draw_index=0,
                    seed=EVAL_CASE_SEED,
                )
                values, runtime = _extract_case(
                    item,
                    socket=socket,
                    adapter=adapter,
                    dino=dino,
                    projection=projection,
                    device=device,
                    executor=executor,
                )
                if expected is not None:
                    _check_existing_local_identities(
                        index, item, values["emitter_topk"], expected
                    )
                before = candidate_pool_digest(
                    values["candidates"], values["valid"], values["emitter_topk"]
                )
                logits = _score_case(model, values, device=device)
                after = candidate_pool_digest(
                    values["candidates"], values["valid"], values["emitter_topk"]
                )
                if before != after or before != runtime["union_identity_digest"]:
                    raise RuntimeError("model scoring mutated candidate union identities")
                raw_candidates = values["emitter_topk"][0]
                learned_candidates = np.stack(
                    [
                        _rank_topk(
                            values["candidates"][axis],
                            values["valid"][axis],
                            logits[axis],
                            k=TOP_K,
                        )
                        for axis in range(2)
                    ]
                )
                raw_reciprocal = [
                    boundary._reciprocal_evidence(values["raw_dense"][axis])
                    for axis in range(2)
                ]
                learned_reciprocal = [
                    sparse_reciprocal_evidence(
                        values["candidates"][axis],
                        values["valid"][axis],
                        logits[axis],
                    )
                    for axis in range(2)
                ]
                prefix = f"case_{index:04d}"
                arrays[f"{prefix}__union_candidates"] = values["candidates"]
                arrays[f"{prefix}__union_valid"] = values["valid"]
                arrays[f"{prefix}__emitter_topk"] = values["emitter_topk"]
                arrays[f"{prefix}__learned_logits"] = logits
                arrays[f"{prefix}__raw_top32"] = raw_candidates
                arrays[f"{prefix}__learned_top32"] = learned_candidates
                for name, evidence in (
                    ("raw", raw_reciprocal),
                    ("learned", learned_reciprocal),
                ):
                    for axis_index, axis in enumerate(("right", "down")):
                        for key, value in evidence[axis_index].items():
                            arrays[f"{prefix}__{name}_reciprocal__{axis}__{key}"] = value
                row = {
                    "prefix": prefix,
                    "case_id": item.case_id,
                    "source_filename": item.source_filename,
                    "draw_index": item.draw_index,
                    "dirty_sha256": hashlib.sha256(item.tiles.tobytes()).hexdigest(),
                    "union_identity_digest_before_model": before,
                    "union_identity_digest_after_model": after,
                    "opened_local_emitter_identity_match": bool(check_opened_local),
                    "runtime": runtime,
                }
                rows.append(row)
                cases.append(
                    {
                        "case_id": item.case_id,
                        "raw_candidates": raw_candidates,
                        "learned_candidates": learned_candidates,
                        "union_candidates": values["candidates"],
                        "union_valid": values["valid"],
                        "raw_reciprocal": raw_reciprocal,
                        "learned_reciprocal": learned_reciprocal,
                    }
                )
                references[item.case_id] = reference
                print(
                    json.dumps(
                        {
                            "event": "freeze_panel",
                            "panel": panel_name,
                            "case": index + 1,
                            "count": len(boards),
                            "seconds": runtime["total_seconds"],
                        }
                    ),
                    flush=True,
                )
    finally:
        if expected is not None:
            expected[1].close()
            expected[2].close()
    panel = output / panel_name
    archive = panel / "frozen-target-free-predictions.npz"
    metadata = panel / "frozen-target-free-predictions.json"
    freeze = panel / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-tri-emitter-verifier-target-free-freeze-v1",
            "panel": panel_name,
            "contains_exact_references_or_labels": False,
            "contains_clean_or_output_pixels": False,
            "union_identities_immutable": True,
            "raw_top32_preserved_in_union": True,
            "rows": rows,
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-tri-emitter-verifier-pre-score-v1",
            "created_before_exact_reference_scoring": True,
            "contains_exact_references_or_labels": False,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "module": _record(
                    PROJECT_ROOT / "src/aiijc_puzzle/tri_emitter_edge_verifier.py"
                ),
                "runner": _record(Path(__file__)),
            },
        },
    )
    return cases, references, {
        "archive": _record(archive),
        "metadata": _record(metadata),
        "pre_score_freeze": _record(freeze),
    }


def _score_panel(
    cases: Sequence[dict[str, Any]],
    references: Mapping[str, ExactSyntheticReference],
) -> dict[str, Any]:
    totals = {
        variant: {
            f"{scope}_{field}": 0
            for scope in ("right", "down", "pooled")
            for field in ("total", "hits_at_1", "hits_at_5", "hits_at_32")
        }
        for variant in ("raw_d64_ot", "tri_emitter_verifier")
    }
    reciprocal_rows: dict[str, list[tuple[float, bool]]] = {
        "raw_d64_ot": [],
        "tri_emitter_verifier": [],
    }
    union_total = 0
    union_hits = 0
    raw_hits = 0
    identity_ok = True
    for case in cases:
        reference = references[case["case_id"]].tile_at_position
        for variant, candidates in (
            ("raw_d64_ot", case["raw_candidates"]),
            ("tri_emitter_verifier", case["learned_candidates"]),
        ):
            metrics = exact_local_retrieval_metrics(
                candidates[0], candidates[1], reference, ks=(1, 5, 32)
            )
            for key in totals[variant]:
                totals[variant][key] += int(metrics[key])
        for axis in range(2):
            truth = _truth_by_anchor(reference, axis=axis)
            valid_truth = truth >= 0
            anchors = np.flatnonzero(valid_truth)
            raw_hit = np.any(
                case["raw_candidates"][axis, anchors] == truth[anchors, None], axis=1
            )
            union_hit = np.array(
                [
                    truth[anchor]
                    in case["union_candidates"][axis, anchor, case["union_valid"][axis, anchor]]
                    for anchor in anchors
                ],
                dtype=bool,
            )
            union_total += len(anchors)
            raw_hits += int(raw_hit.sum())
            union_hits += int(union_hit.sum())
            for name, evidence in (
                ("raw_d64_ot", case["raw_reciprocal"][axis]),
                ("tri_emitter_verifier", case["learned_reciprocal"][axis]),
            ):
                admitted = valid_truth & evidence["reciprocal"]
                correct = evidence["target"] == truth
                reciprocal_rows[name].extend(
                    (float(confidence), bool(ok))
                    for confidence, ok in zip(
                        evidence["confidence"][admitted],
                        correct[admitted],
                        strict=True,
                    )
                )
        identity_ok &= True
    for metrics in totals.values():
        for scope in ("right", "down", "pooled"):
            denominator = metrics[f"{scope}_total"]
            for k in (1, 5, 32):
                metrics[f"{scope}_r{k}"] = metrics[f"{scope}_hits_at_{k}"] / denominator

    def precision_at(rows: list[tuple[float, bool]], count: int) -> float:
        ordered = sorted(rows, key=lambda row: -row[0])[:count]
        return sum(int(correct) for _, correct in ordered) / count if count else 0.0

    count = min(len(reciprocal_rows["raw_d64_ot"]), len(reciprocal_rows["tri_emitter_verifier"]))
    learned_precision = precision_at(reciprocal_rows["tri_emitter_verifier"], count)
    raw_precision = precision_at(reciprocal_rows["raw_d64_ot"], count)
    native = {
        name: {
            "reciprocal_queries": len(rows),
            "coverage": len(rows) / union_total,
            "precision": sum(int(correct) for _, correct in rows) / len(rows) if rows else 0.0,
        }
        for name, rows in reciprocal_rows.items()
    }
    return {
        "case_count": len(cases),
        "retrieval": totals,
        "reciprocal": {
            "native": native,
            "matched_vs_raw": {
                "matched_query_count": count,
                "matched_coverage": count / union_total,
                "candidate_precision": learned_precision,
                "raw_d64_ot_precision": raw_precision,
                "precision_gain": learned_precision - raw_precision,
            },
        },
        "union_top32_each_emitter": {
            "total": union_total,
            "raw_hits": raw_hits,
            "union_hits": union_hits,
            "raw_coverage": raw_hits / union_total,
            "union_coverage": union_hits / union_total,
            "coverage_gain": (union_hits - raw_hits) / union_total,
            "identities_unchanged_by_model": identity_ok,
        },
    }


def _gate(metrics: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    raw = metrics["retrieval"]["raw_d64_ot"]
    learned = metrics["retrieval"]["tri_emitter_verifier"]
    reciprocal = metrics["reciprocal"]["matched_vs_raw"]
    r1 = learned["pooled_r1"] - raw["pooled_r1"]
    r5 = learned["pooled_r5"] - raw["pooled_r5"]
    passed = bool(
        r1 >= thresholds["pooled_r1_gain_minimum"]
        and r5 >= thresholds["pooled_r5_gain_minimum"]
        and reciprocal["precision_gain"]
        >= thresholds["matched_reciprocal_precision_gain_minimum"]
        and reciprocal["matched_coverage"]
        >= thresholds["matched_reciprocal_coverage_minimum"]
        and metrics["union_top32_each_emitter"]["identities_unchanged_by_model"]
    )
    return {
        "r1_gain": r1,
        "r5_gain": r5,
        "matched_reciprocal_precision_gain": reciprocal["precision_gain"],
        "matched_reciprocal_coverage": reciprocal["matched_coverage"],
        "union_identities_unchanged": metrics["union_top32_each_emitter"][
            "identities_unchanged_by_model"
        ],
        "thresholds": dict(thresholds),
        "passed": passed,
    }


def _write_terminal_config(
    *,
    main_config: Path,
    checkpoint: Path,
    protocol: Mapping[str, Any],
    local_gate: Mapping[str, Any],
) -> dict[str, str]:
    if TERMINAL_CONFIG.exists() or TERMINAL_CONFIG.with_suffix(".json.sha256").exists():
        raise FileExistsError("terminal verifier config already exists")
    payload = {
        "schema": "aiijc-tri-emitter-verifier-terminal-preregistration-v1",
        "status": "signed-before-terminal-target-access",
        "created_after_local_gate_passed": True,
        "main_config": _record(main_config),
        "model": _record(checkpoint),
        "source": {
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/tri_emitter_edge_verifier.py"
            ),
            "runner": _record(Path(__file__)),
        },
        "rosters": {
            "fit_filenames": protocol["fit_filenames"],
            "fit_digest": protocol["fit_digest"],
            "local_filenames": protocol["local_filenames"],
            "local_digest": protocol["local_digest"],
            "terminal_filenames": protocol["terminal_filenames"],
            "terminal_digest": protocol["terminal_digest"],
        },
        "local_gate": dict(local_gate),
        "terminal_gate": TERMINAL_GATE,
        "decoder_included": False,
        "competition_test_access": False,
    }
    _write_json(TERMINAL_CONFIG, payload)
    digest = sha256_file(TERMINAL_CONFIG)
    sidecar = TERMINAL_CONFIG.with_suffix(".json.sha256")
    with sidecar.open("x", encoding="utf-8") as stream:
        stream.write(f"{digest}  {TERMINAL_CONFIG.name}\n")
    return _record(TERMINAL_CONFIG)


def run_full(
    args: argparse.Namespace,
    config: dict[str, Any],
    config_sha: str,
    output: Path,
) -> dict[str, Any]:
    protocol, fit_boards, local_records, terminal_records = roster._load_protocol(args)
    if names_digest(protocol["fit_filenames"]) != config["source_protocol"]["fit_digest"]:
        raise RuntimeError("full config fit roster changed")
    socket, adapter, dino, projection, device = _make_models(args)
    output.mkdir(parents=True, exist_ok=False)
    fit_paths, fit_rows = _prepare_fit_cache(
        fit_boards,
        output=output,
        socket=socket,
        adapter=adapter,
        dino=dino,
        projection=projection,
        device=device,
    )
    model, training, checkpoint = _train_full(
        fit_paths,
        device=device,
        config_sha=config_sha,
        protocol=protocol,
        output=output,
    )
    print(
        json.dumps(
            {
                "event": "ready_for_local_scoring",
                "config_sha256": config_sha,
                "checkpoint_sha256": sha256_file(checkpoint),
                "training_seconds": training["seconds"],
            }
        ),
        flush=True,
    )
    local_cases, local_references, local_artifacts = _freeze_panel(
        local_records,
        panel_name="opened-local16",
        targets_path=args.targets,
        model=model,
        socket=socket,
        adapter=adapter,
        dino=dino,
        projection=projection,
        device=device,
        output=output,
        check_opened_local=True,
    )
    local_metrics = _score_panel(local_cases, local_references)
    local_gate = _gate(local_metrics, LOCAL_GATE)
    terminal: dict[str, Any] = {"status": "skipped_by_local_gate"}
    terminal_gate: dict[str, Any] | None = None
    terminal_config: dict[str, str] | None = None
    if local_gate["passed"]:
        terminal_config = _write_terminal_config(
            main_config=args.config,
            checkpoint=checkpoint,
            protocol=protocol,
            local_gate=local_gate,
        )
        terminal_cases, terminal_references, terminal_artifacts = _freeze_panel(
            terminal_records,
            panel_name="reserved-terminal16",
            targets_path=args.targets,
            model=model,
            socket=socket,
            adapter=adapter,
            dino=dino,
            projection=projection,
            device=device,
            output=output,
            check_opened_local=False,
        )
        terminal_metrics = _score_panel(terminal_cases, terminal_references)
        terminal_gate = _gate(terminal_metrics, TERMINAL_GATE)
        terminal = {
            "status": "complete",
            "metrics": terminal_metrics,
            "gate": terminal_gate,
            "artifacts": terminal_artifacts,
        }
    status = (
        "terminal-transfer-passed-decoder-separately-eligible"
        if terminal_gate is not None and terminal_gate["passed"]
        else "terminal-transfer-failed-stop"
        if terminal_gate is not None
        else "local-gate-failed-stop-no-terminal"
    )
    report = {
        "schema": "aiijc-tri-emitter-edge-verifier-report-v1",
        "status": status,
        "config": _record(args.config),
        "protocol": protocol,
        "contract": verifier_contract(model),
        "fit_cache": {
            "case_count": len(fit_rows),
            "draws": list(FIT_DRAWS),
            "rows": fit_rows,
        },
        "training": training,
        "checkpoint": _record(checkpoint),
        "local16": {
            "metrics": local_metrics,
            "gate": local_gate,
            "artifacts": local_artifacts,
        },
        "terminal_preregistration": terminal_config,
        "terminal16": terminal,
        "decoder": {
            "run": False,
            "eligible": bool(terminal_gate is not None and terminal_gate["passed"]),
        },
        "legality": {
            "organizer_train_only": True,
            "candidate_union_target_blind": True,
            "raw_top32_always_preserved": True,
            "pixels_modified": False,
            "competition_test_accessed": False,
            "submission_or_production_modified": False,
        },
        "artifacts": {
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/tri_emitter_edge_verifier.py"
            ),
            "runner": _record(Path(__file__)),
            "socket": _record(args.socket_checkpoint),
            "adapter": _record(ADAPTER_CHECKPOINT),
            "dino": _record(DINO_CHECKPOINT),
        },
    }
    _write_json(output / "report.json", report)
    return report


def main() -> None:
    args = parse_args()
    if args.config is None:
        args.config = DEFAULT_ARCH_CONFIG if args.mode == "capacity" else DEFAULT_FULL_CONFIG
    if args.output_dir is None:
        args.output_dir = {
            "capacity": DEFAULT_CAPACITY_OUTPUT,
            "benchmark": DEFAULT_BENCHMARK_OUTPUT,
            "full": DEFAULT_FULL_OUTPUT,
        }[args.mode]
    config, config_sha = _load_config(args.config)
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    torch.use_deterministic_algorithms(
        True, warn_only=args.allow_nondeterministic_mps
    )
    if args.mode == "capacity":
        report = run_capacity(args, config, config_sha, args.output_dir.resolve())
    elif args.mode == "benchmark":
        report = run_benchmark(args, config, config_sha, args.output_dir.resolve())
    else:
        report = run_full(args, config, config_sha, args.output_dir.resolve())
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
