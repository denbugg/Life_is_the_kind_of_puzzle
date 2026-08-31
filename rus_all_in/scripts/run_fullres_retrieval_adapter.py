#!/usr/bin/env python3
"""Benchmark, train and retrieval-gate one frozen-Socket fullres adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.fullres_boundary_denoiser import restore_matcher_view
from aiijc_puzzle.fullres_retrieval_adapter import (
    FullResolutionRetrievalAdapter,
    retrieval_adapter_contract,
    retrieval_adapter_loss,
)
from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file
from aiijc_puzzle.restoration_r6 import distort_tiles
from aiijc_puzzle.socket_sorter_production import load_socket_checkpoint
from aiijc_puzzle.synthetic_socket_evaluation import (
    ExactSyntheticReference,
    SyntheticSocketInput,
    exact_local_retrieval_metrics,
    freeze_topk_candidates,
    make_exact_synthetic_case,
    names_digest,
)

try:
    from scripts import run_fullres_boundary_denoiser as parent
except ModuleNotFoundError:
    import run_fullres_boundary_denoiser as parent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/fullres_retrieval_adapter_preregistered_v1.json"
CONFIG_SHA256 = "74bc2f356a5750bd13f19a0911b639831f771522e258313f765027b5a6d0fc95"
PARENT_REPORT = (
    PROJECT_ROOT
    / "outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/report.json"
)
PARENT_REPORT_SHA256 = "780f6b065ba769bef8b3ffd30cf0bcb781b2040258835964b122725e732fccc7"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_SOCKET = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
DEFAULT_BENCHMARK_OUTPUT = (
    PROJECT_ROOT / "outputs/fullres-retrieval-adapter/one-step-benchmark-v1"
)
DEFAULT_RUN_OUTPUT = (
    PROJECT_ROOT / "outputs/fullres-retrieval-adapter/fixed-s100-s400-local16-v1"
)
GRID = 24
COUNT = GRID * GRID
CHECKPOINT_STEPS = (100, 400)
TRAIN_SEED = 20260911
EVAL_SEED = 20260908
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 2e-4
LOCAL_KS = (1, 5, 32)
TOP_K = 32


@dataclass(frozen=True)
class TrainSpec:
    source_index: int
    corruption_seed: int
    permutation_seed: int


@dataclass(frozen=True)
class FrozenCase:
    case_id: str
    source_filename: str
    draw_index: int
    dirty_sha256: str
    candidates: dict[str, dict[str, np.ndarray]]
    reciprocal: dict[str, dict[str, dict[str, np.ndarray]]]
    runtime_seconds: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "run"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--socket-checkpoint", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(resolved),
    }


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def _load_protocol(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    list[parent.CleanBoard],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise RuntimeError("preregistered config SHA mismatch")
    if sha256_file(PARENT_REPORT) != PARENT_REPORT_SHA256:
        raise RuntimeError("frozen source-roster parent report SHA mismatch")
    if sha256_file(args.socket_checkpoint) != json.loads(
        CONFIG.read_text(encoding="utf-8")
    )["frozen_inputs"]["socket_checkpoint"]["sha256"]:
        raise RuntimeError("frozen Socket checkpoint SHA mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol_digest") != compute_protocol_digest(manifest):
        raise RuntimeError("manifest protocol digest mismatch")
    parent_report = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    selection = parent_report["selection"]
    names = {
        "fit": tuple(selection["train_filenames"]),
        "local": tuple(selection["eval_filenames"]),
        "terminal": tuple(selection["terminal_filenames"]),
    }
    expected = {
        "fit": "6c0d605b60d9f437a9676dbee653185e62ffb44c42e012e05228b8f3901a0d1c",
        "local": "25ea956a8514d72cb09b8093f12999534995cf75fb18b383834acf38693ca47f",
        "terminal": "2a39d853772aa2c6d23d8b7dbc59f726e2f3a3ecfe098e96ad065c1bbd6d65a6",
    }
    if any(names_digest(names[key]) != expected[key] for key in names):
        raise RuntimeError("source roster digest changed from preregistration")
    if set(names["fit"]) & set(names["local"]):
        raise RuntimeError("fit/local source overlap")
    if set(names["fit"]) & set(names["terminal"]):
        raise RuntimeError("fit/terminal source overlap")
    if set(names["local"]) & set(names["terminal"]):
        raise RuntimeError("local/terminal source overlap")
    record_by_name = {
        str(record["filename"]): record for record in manifest["splits"]["train"]
    }

    def records(kind: str) -> tuple[dict[str, Any], ...]:
        try:
            return tuple(record_by_name[name] for name in names[kind])
        except KeyError as error:
            raise RuntimeError(f"{kind} roster is absent from manifest train") from error

    fit_records = records("fit")
    local_records = records("local")
    terminal_records = records("terminal")
    fit_boards = parent._prepare_boards(fit_records, args.targets)
    protocol = {
        "manifest_digest": compute_protocol_digest(manifest),
        "fit_filenames": list(names["fit"]),
        "fit_digest": expected["fit"],
        "local_filenames": list(names["local"]),
        "local_digest": expected["local"],
        "terminal_filenames": list(names["terminal"]),
        "terminal_digest": expected["terminal"],
    }
    return protocol, fit_boards, local_records, terminal_records


def _training_specs() -> tuple[TrainSpec, ...]:
    generator = np.random.default_rng(TRAIN_SEED + 17)
    return tuple(
        TrainSpec(
            source_index=int(generator.integers(32)),
            corruption_seed=int(generator.integers(0, 2**31 - 1)),
            permutation_seed=int(generator.integers(0, 2**31 - 1)),
        )
        for _ in range(CHECKPOINT_STEPS[-1])
    )


def _materialise_train_board(
    board: parent.CleanBoard,
    spec: TrainSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dirty = distort_tiles(board.tiles, np.random.default_rng(spec.corruption_seed))
    permutation = np.random.default_rng(spec.permutation_seed).permutation(COUNT)
    tile_at_position = np.ascontiguousarray(np.argsort(permutation), dtype=np.int64)
    return (
        np.ascontiguousarray(dirty[permutation]),
        np.ascontiguousarray(board.tiles[permutation]),
        tile_at_position,
    )


def _one_update(
    adapter: FullResolutionRetrievalAdapter,
    socket: Any,
    optimizer: torch.optim.Optimizer,
    dirty_array: np.ndarray,
    clean_array: np.ndarray,
    tile_at_position: np.ndarray,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], float]:
    started = perf_counter()
    dirty = parent._tensor(dirty_array, device)
    clean = parent._tensor(clean_array, device)
    layout = torch.from_numpy(tile_at_position).to(device=device).unsqueeze(0)
    result = retrieval_adapter_loss(
        adapter,
        socket.model,
        dirty,
        clean,
        layout,
        grid=GRID,
    )
    optimizer.zero_grad(set_to_none=True)
    result.total.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0))
    optimizer.step()
    _sync(device)
    elapsed = perf_counter() - started
    diagnostics = {
        "loss": float(result.total.detach().cpu()),
        "socket_loss": float(result.socket.detach().cpu()),
        "boundary_loss": float(result.boundary.detach().cpu()),
        "grad_norm": grad_norm,
        "socket": result.socket_diagnostics,
        "boundary": result.boundary_diagnostics,
    }
    return diagnostics, elapsed


def run_benchmark(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    fit_boards: list[parent.CleanBoard],
    output_dir: Path,
) -> dict[str, Any]:
    spec = _training_specs()[0]
    dirty, clean, layout = _materialise_train_board(
        fit_boards[spec.source_index], spec
    )
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    results: dict[str, Any] = {}
    for name in devices:
        device = torch.device(name)
        socket = load_socket_checkpoint(args.socket_checkpoint, device=device)
        adapter = FullResolutionRetrievalAdapter().to(device).train()
        optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        diagnostics, elapsed = _one_update(
            adapter,
            socket,
            optimizer,
            dirty,
            clean,
            layout,
            device=device,
        )
        results[name] = {
            "cold_complete_update_seconds": elapsed,
            "projected_400_update_minutes": elapsed * 400 / 60.0,
            "diagnostics": diagnostics,
        }
        del socket, adapter, optimizer
        if device.type == "mps":
            torch.mps.empty_cache()
    report = {
        "schema": "aiijc-fullres-retrieval-adapter-one-step-benchmark-v1",
        "status": "complete",
        "target_access": "one organizer-train fit source only",
        "local_or_terminal_targets_opened": False,
        "protocol": protocol,
        "results": results,
        "artifacts": {
            "config": _record(CONFIG),
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/fullres_retrieval_adapter.py"
            ),
            "runner": _record(Path(__file__)),
            "socket": _record(args.socket_checkpoint),
        },
    }
    _write_json(output_dir / "benchmark.json", report)
    return report


def _save_checkpoint(
    adapter: FullResolutionRetrievalAdapter,
    *,
    step: int,
    output_dir: Path,
    protocol: dict[str, Any],
    history: list[dict[str, Any]],
) -> Path:
    path = output_dir / f"adapter_step{step}.pt"
    exposed = tuple(
        sorted(
            set(protocol["fit_filenames"])
            | set(protocol["local_filenames"])
            | set(protocol["terminal_filenames"])
        )
    )
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in adapter.state_dict().items()
            },
            "contract": retrieval_adapter_contract(adapter),
            "step": step,
            "config_sha256": CONFIG_SHA256,
            "selection": {
                **protocol,
                "lineage_train_filenames": protocol["fit_filenames"],
                "lineage_train_digest": names_digest(
                    protocol["fit_filenames"], sort_names=True
                ),
                "lineage_exposed_filenames": list(exposed),
                "lineage_exposed_digest": names_digest(exposed, sort_names=True),
            },
            "training_history": history.copy(),
        },
        path,
    )
    return path


def train(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    fit_boards: list[parent.CleanBoard],
    output_dir: Path,
    *,
    device: torch.device,
) -> tuple[dict[int, Path], list[dict[str, Any]], dict[str, float]]:
    socket = load_socket_checkpoint(args.socket_checkpoint, device=device)
    adapter = FullResolutionRetrievalAdapter().to(device).train()
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        CHECKPOINT_STEPS[-1],
        eta_min=LEARNING_RATE * 0.05,
    )
    specs = _training_specs()
    history: list[dict[str, Any]] = []
    checkpoints: dict[int, Path] = {}
    started = perf_counter()
    wait_seconds = 0.0
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures: dict[int, Future[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        submit = 0
        while submit < 4:
            spec = specs[submit]
            futures[submit] = executor.submit(
                _materialise_train_board,
                fit_boards[spec.source_index],
                spec,
            )
            submit += 1
        for index, spec in enumerate(specs):
            wait_started = perf_counter()
            dirty, clean, layout = futures.pop(index).result()
            wait_seconds += perf_counter() - wait_started
            if submit < len(specs):
                next_spec = specs[submit]
                futures[submit] = executor.submit(
                    _materialise_train_board,
                    fit_boards[next_spec.source_index],
                    next_spec,
                )
                submit += 1
            diagnostics, update_seconds = _one_update(
                adapter,
                socket,
                optimizer,
                dirty,
                clean,
                layout,
                device=device,
            )
            scheduler.step()
            step = index + 1
            row = {
                "step": step,
                "source_filename": fit_boards[spec.source_index].filename,
                "corruption_seed": spec.corruption_seed,
                "permutation_seed": spec.permutation_seed,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "update_seconds": update_seconds,
                **diagnostics,
            }
            history.append(row)
            if step in CHECKPOINT_STEPS:
                checkpoints[step] = _save_checkpoint(
                    adapter,
                    step=step,
                    output_dir=output_dir,
                    protocol=protocol,
                    history=history,
                )
            if step == 1 or step % 10 == 0 or step in CHECKPOINT_STEPS:
                recent = history[-min(10, len(history)) :]
                print(
                    json.dumps(
                        {
                            "event": "train",
                            "step": step,
                            "loss": float(np.mean([item["loss"] for item in recent])),
                            "socket_loss": float(
                                np.mean([item["socket_loss"] for item in recent])
                            ),
                            "boundary_loss": float(
                                np.mean([item["boundary_loss"] for item in recent])
                            ),
                            "mean_update_seconds": float(
                                np.mean([item["update_seconds"] for item in recent])
                            ),
                            "elapsed_seconds": perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    return checkpoints, history, {
        "training_seconds": perf_counter() - started,
        "prefetch_wait_seconds": wait_seconds,
    }


def _load_adapters(
    checkpoint_paths: dict[int, Path],
    *,
    device: torch.device,
) -> dict[str, FullResolutionRetrievalAdapter]:
    result: dict[str, FullResolutionRetrievalAdapter] = {}
    for step, path in checkpoint_paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("config_sha256") != CONFIG_SHA256 or payload.get("step") != step:
            raise RuntimeError("adapter checkpoint contract mismatch")
        model = FullResolutionRetrievalAdapter().to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        result[f"adapter_step{step}"] = model
    return result


@torch.inference_mode()
def _freeze_case(
    item: SyntheticSocketInput,
    *,
    socket: Any,
    adapters: dict[str, FullResolutionRetrievalAdapter],
    device: torch.device,
) -> FrozenCase:
    candidates: dict[str, dict[str, np.ndarray]] = {}
    reciprocal: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    runtimes: dict[str, float] = {}

    started = perf_counter()
    raw_right, raw_down = parent._socket_scores(socket, item.tiles, device=device)
    runtimes["raw_d64"] = perf_counter() - started
    candidates["raw_d64_ot"] = {
        "right": freeze_topk_candidates(raw_right, max_k=TOP_K),
        "down": freeze_topk_candidates(raw_down, max_k=TOP_K),
    }
    reciprocal["raw_d64_ot"] = {
        "right": parent._reciprocal_evidence(raw_right),
        "down": parent._reciprocal_evidence(raw_down),
    }
    for name, adapter in adapters.items():
        started = perf_counter()
        adapted = restore_matcher_view(
            adapter,
            item.tiles,
            device=device,
            batch_size=COUNT,
        )
        runtimes[f"{name}_adapt"] = perf_counter() - started
        started = perf_counter()
        right, down = parent._socket_scores(socket, adapted, device=device)
        runtimes[f"{name}_d64"] = perf_counter() - started
        candidates[name] = {
            "right": freeze_topk_candidates(right, max_k=TOP_K),
            "down": freeze_topk_candidates(down, max_k=TOP_K),
        }
        reciprocal[name] = {
            "right": parent._reciprocal_evidence(right),
            "down": parent._reciprocal_evidence(down),
        }
    return FrozenCase(
        case_id=item.case_id,
        source_filename=item.source_filename,
        draw_index=item.draw_index,
        dirty_sha256=hashlib.sha256(item.tiles.tobytes()).hexdigest(),
        candidates=candidates,
        reciprocal=reciprocal,
        runtime_seconds=runtimes,
    )


def _freeze_panel(
    records: tuple[dict[str, Any], ...],
    *,
    panel_name: str,
    targets: Path,
    socket: Any,
    adapters: dict[str, FullResolutionRetrievalAdapter],
    device: torch.device,
    output_dir: Path,
) -> tuple[list[FrozenCase], dict[str, ExactSyntheticReference]]:
    boards = parent._prepare_boards(records, targets)
    cases: list[FrozenCase] = []
    references: dict[str, ExactSyntheticReference] = {}
    for index, board in enumerate(boards, start=1):
        item, reference = make_exact_synthetic_case(
            board.tiles,
            source_filename=board.filename,
            draw_index=0,
            seed=EVAL_SEED,
        )
        cases.append(
            _freeze_case(item, socket=socket, adapters=adapters, device=device)
        )
        references[item.case_id] = reference
        print(
            json.dumps(
                {
                    "event": "freeze",
                    "panel": panel_name,
                    "case": index,
                    "count": len(boards),
                    "source": board.filename,
                }
            ),
            flush=True,
        )
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        prefix = f"case_{index:04d}"
        for variant, axes in case.candidates.items():
            for axis, value in axes.items():
                arrays[f"{prefix}__candidate__{variant}__{axis}"] = value
        for variant, axes in case.reciprocal.items():
            for axis, evidence in axes.items():
                for key, value in evidence.items():
                    arrays[f"{prefix}__reciprocal__{variant}__{axis}__{key}"] = value
        rows.append(
            {
                "prefix": prefix,
                "case_id": case.case_id,
                "source_filename": case.source_filename,
                "draw_index": case.draw_index,
                "dirty_sha256": case.dirty_sha256,
                "variants": list(case.candidates),
                "runtime_seconds": case.runtime_seconds,
            }
        )
    panel = output_dir / panel_name
    archive = panel / "frozen-target-free-retrieval.npz"
    metadata = panel / "frozen-target-free-retrieval.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-fullres-retrieval-adapter-freeze-v1",
            "panel": panel_name,
            "contains_clean_pixels_or_exact_references": False,
            "contains_restored_pixels_or_layouts": False,
            "raw_evidence_preserved": True,
            "matcher_view_only": True,
            "rows": rows,
        },
    )
    _write_json(
        panel / "pre-score-freeze.json",
        {
            "schema": "aiijc-fullres-retrieval-adapter-pre-score-freeze-v1",
            "created_before_exact_reference_scoring": True,
            "contains_exact_references_or_labels": False,
            "artifacts": {
                "archive": _record(archive),
                "metadata": _record(metadata),
                "config": _record(CONFIG),
                "module": _record(
                    PROJECT_ROOT / "src/aiijc_puzzle/fullres_retrieval_adapter.py"
                ),
                "runner": _record(Path(__file__)),
            },
        },
    )
    return cases, references


def _score_panel(
    cases: list[FrozenCase],
    references: dict[str, ExactSyntheticReference],
) -> dict[str, Any]:
    variants = tuple(cases[0].candidates)
    totals: dict[str, dict[str, float | int]] = {}
    reciprocal_rows: dict[str, list[tuple[float, bool]]] = {
        name: [] for name in variants
    }
    supply: dict[str, dict[str, dict[str, int]]] = {
        name: {
            axis: {"total": 0, "raw_hits": 0, "union_hits": 0}
            for axis in ("right", "down")
        }
        for name in variants
        if name != "raw_d64_ot"
    }
    valid_query_count = 0
    for case in cases:
        reference = references[case.case_id].tile_at_position
        for variant in variants:
            metrics = exact_local_retrieval_metrics(
                case.candidates[variant]["right"],
                case.candidates[variant]["down"],
                reference,
                ks=LOCAL_KS,
            )
            aggregate = totals.setdefault(variant, {})
            for key, value in metrics.items():
                if "_hits_at_" in key or key.endswith("_total"):
                    aggregate[key] = int(aggregate.get(key, 0)) + int(value)
        for axis in ("right", "down"):
            truth = parent._truth_by_anchor(reference, axis=axis)
            valid = truth >= 0
            valid_query_count += int(valid.sum())
            raw_candidates = case.candidates["raw_d64_ot"][axis]
            anchors = np.flatnonzero(valid)
            raw_hit = np.any(
                raw_candidates[anchors] == truth[anchors, None], axis=1
            )
            for variant in variants:
                evidence = case.reciprocal[variant][axis]
                admitted = valid & evidence["reciprocal"]
                correct = evidence["target"] == truth
                reciprocal_rows[variant].extend(
                    (float(confidence), bool(ok))
                    for confidence, ok in zip(
                        evidence["confidence"][admitted],
                        correct[admitted],
                        strict=True,
                    )
                )
                if variant == "raw_d64_ot":
                    continue
                auxiliary = case.candidates[variant][axis]
                auxiliary_hit = np.any(
                    auxiliary[anchors] == truth[anchors, None], axis=1
                )
                values = supply[variant][axis]
                values["total"] += len(anchors)
                values["raw_hits"] += int(raw_hit.sum())
                values["union_hits"] += int(np.count_nonzero(raw_hit | auxiliary_hit))
    for metrics in totals.values():
        for scope in ("right", "down", "pooled"):
            denominator = int(metrics[f"{scope}_total"])
            for k in LOCAL_KS:
                metrics[f"{scope}_r{k}"] = (
                    int(metrics[f"{scope}_hits_at_{k}"]) / denominator
                )

    native = {}
    for variant, rows in reciprocal_rows.items():
        native[variant] = {
            "reciprocal_queries": len(rows),
            "coverage": len(rows) / valid_query_count,
            "precision": sum(int(ok) for _, ok in rows) / len(rows) if rows else 0.0,
        }

    def precision_at(rows: list[tuple[float, bool]], count: int) -> float:
        ordered = sorted(rows, key=lambda value: -value[0])[:count]
        return sum(int(ok) for _, ok in ordered) / count if count else 0.0

    matched = {}
    raw_rows = reciprocal_rows["raw_d64_ot"]
    for variant in variants:
        if variant == "raw_d64_ot":
            continue
        count = min(len(raw_rows), len(reciprocal_rows[variant]))
        candidate_precision = precision_at(reciprocal_rows[variant], count)
        raw_precision = precision_at(raw_rows, count)
        matched[variant] = {
            "matched_query_count": count,
            "matched_coverage": count / valid_query_count,
            "candidate_precision": candidate_precision,
            "raw_d64_ot_precision": raw_precision,
            "precision_gain": candidate_precision - raw_precision,
        }

    supply_metrics: dict[str, Any] = {}
    for variant, axes in supply.items():
        axis_output = {}
        for axis, values in axes.items():
            total = values["total"]
            raw_coverage = values["raw_hits"] / total
            union_coverage = values["union_hits"] / total
            axis_output[axis] = {
                **values,
                "raw_coverage": raw_coverage,
                "union_coverage": union_coverage,
                "coverage_gain": union_coverage - raw_coverage,
            }
        pooled_total = sum(values["total"] for values in axes.values())
        pooled_raw = sum(values["raw_hits"] for values in axes.values())
        pooled_union = sum(values["union_hits"] for values in axes.values())
        supply_metrics[variant] = {
            "axes": axis_output,
            "pooled_total": pooled_total,
            "pooled_raw_hits": pooled_raw,
            "pooled_union_hits": pooled_union,
            "pooled_raw_coverage": pooled_raw / pooled_total,
            "pooled_union_coverage": pooled_union / pooled_total,
            "pooled_coverage_gain": (pooled_union - pooled_raw) / pooled_total,
        }
    return {
        "case_count": len(cases),
        "retrieval": totals,
        "reciprocal": {"native": native, "matched_vs_raw": matched},
        "supply": supply_metrics,
    }


def _local_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    baseline = metrics["retrieval"]["raw_d64_ot"]
    candidate = metrics["retrieval"]["adapter_step400"]
    supply = metrics["supply"]["adapter_step400"]
    precision = metrics["reciprocal"]["matched_vs_raw"]["adapter_step400"]
    r1_gain = candidate["pooled_r1"] - baseline["pooled_r1"]
    r5_gain = candidate["pooled_r5"] - baseline["pooled_r5"]
    supply_passed = all(
        supply["axes"][axis]["coverage_gain"] >= 0.01
        for axis in ("right", "down")
    )
    ranking_passed = r1_gain >= 0.005 and r5_gain >= 0.0
    precision_passed = (
        precision["precision_gain"] >= 0.03
        and precision["matched_coverage"] >= 0.03
    )
    return {
        "r1_gain": r1_gain,
        "r5_gain": r5_gain,
        "directional_supply_gains": {
            axis: supply["axes"][axis]["coverage_gain"]
            for axis in ("right", "down")
        },
        "precision_gain": precision["precision_gain"],
        "matched_coverage": precision["matched_coverage"],
        "supply_passed": supply_passed,
        "ranking_passed": ranking_passed,
        "precision_passed": precision_passed,
        "decoder_gate_passed": supply_passed and (ranking_passed or precision_passed),
    }


def _terminal_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    baseline = metrics["retrieval"]["raw_d64_ot"]
    candidate = metrics["retrieval"]["adapter_step400"]
    supply = metrics["supply"]["adapter_step400"]
    precision = metrics["reciprocal"]["matched_vs_raw"]["adapter_step400"]
    r1_gain = candidate["pooled_r1"] - baseline["pooled_r1"]
    r5_gain = candidate["pooled_r5"] - baseline["pooled_r5"]
    supply_passed = all(
        supply["axes"][axis]["coverage_gain"] >= 0.0
        for axis in ("right", "down")
    )
    ranking_passed = r1_gain >= 0.0 and r5_gain >= 0.0
    precision_passed = (
        precision["precision_gain"] >= 0.0
        and precision["matched_coverage"] >= 0.03
    )
    return {
        "r1_gain": r1_gain,
        "r5_gain": r5_gain,
        "directional_supply_gains": {
            axis: supply["axes"][axis]["coverage_gain"]
            for axis in ("right", "down")
        },
        "precision_gain": precision["precision_gain"],
        "matched_coverage": precision["matched_coverage"],
        "supply_passed": supply_passed,
        "ranking_passed": ranking_passed,
        "precision_passed": precision_passed,
        "transfer_passed": supply_passed and (ranking_passed or precision_passed),
    }


def run_experiment(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    fit_boards: list[parent.CleanBoard],
    local_records: tuple[dict[str, Any], ...],
    terminal_records: tuple[dict[str, Any], ...],
    output_dir: Path,
) -> dict[str, Any]:
    device = _device(args.device)
    checkpoints, history, training_runtime = train(
        args,
        protocol,
        fit_boards,
        output_dir,
        device=device,
    )
    socket = load_socket_checkpoint(args.socket_checkpoint, device=device)
    adapters = _load_adapters(checkpoints, device=device)
    local_cases, local_references = _freeze_panel(
        local_records,
        panel_name="local16",
        targets=args.targets,
        socket=socket,
        adapters=adapters,
        device=device,
        output_dir=output_dir,
    )
    local = _score_panel(local_cases, local_references)
    local_gate = _local_gate(local)
    terminal: dict[str, Any] = {"status": "skipped_by_local_decoder_gate"}
    terminal_gate: dict[str, Any] | None = None
    if local_gate["decoder_gate_passed"]:
        terminal_cases, terminal_references = _freeze_panel(
            terminal_records,
            panel_name="terminal16",
            targets=args.targets,
            socket=socket,
            adapters={"adapter_step400": adapters["adapter_step400"]},
            device=device,
            output_dir=output_dir,
        )
        terminal = _score_panel(terminal_cases, terminal_references)
        terminal["status"] = "complete"
        terminal_gate = _terminal_gate(terminal)
    scaling = {
        key: local["retrieval"]["adapter_step400"][key]
        - local["retrieval"]["adapter_step100"][key]
        for key in (
            "pooled_r1",
            "pooled_r5",
            "right_r1",
            "right_r5",
            "down_r1",
            "down_r5",
        )
    }
    report = {
        "schema": "aiijc-fullres-retrieval-adapter-report-v1",
        "status": (
            "terminal-transfer-passed-decoder-separately-eligible"
            if terminal_gate is not None and terminal_gate["transfer_passed"]
            else "local-gate-passed-terminal-transfer-failed"
            if terminal_gate is not None
            else "local-gate-failed-stop-no-terminal-no-decoder"
        ),
        "protocol": protocol,
        "contract": retrieval_adapter_contract(adapters["adapter_step400"]),
        "configuration": {
            "config_sha256": CONFIG_SHA256,
            "device": str(device),
            "steps": CHECKPOINT_STEPS[-1],
            "fixed_scaling_checkpoints": list(CHECKPOINT_STEPS),
            "training_seed": TRAIN_SEED,
            "evaluation_seed": EVAL_SEED,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "training": {
            "runtime": training_runtime,
            "history": history,
            "checkpoints": {
                str(step): _record(path) for step, path in checkpoints.items()
            },
        },
        "local16": local,
        "scaling_step400_minus_step100": scaling,
        "local_decoder_gate": local_gate,
        "terminal16": terminal,
        "terminal_transfer_gate": terminal_gate,
        "decoder": {
            "run": False,
            "eligible": bool(
                terminal_gate is not None and terminal_gate["transfer_passed"]
            ),
        },
        "legality": {
            "organizer_train_only": True,
            "raw_d64_evidence_preserved": True,
            "adapter_pixels_matcher_only": True,
            "strict_original_upright_tiles_required_for_any_future_output": True,
            "competition_test_accessed": False,
            "submission_or_production_modified": False,
        },
        "artifacts": {
            "config": _record(CONFIG),
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/fullres_retrieval_adapter.py"
            ),
            "runner": _record(Path(__file__)),
            "socket": _record(args.socket_checkpoint),
            "parent_roster_report": _record(PARENT_REPORT),
        },
    }
    _write_json(output_dir / "report.json", report)
    return report


def main() -> None:
    args = parse_args()
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    torch.use_deterministic_algorithms(
        True,
        warn_only=args.allow_nondeterministic_mps,
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_BENCHMARK_OUTPUT
        if args.mode == "benchmark"
        else DEFAULT_RUN_OUTPUT
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol, fit_boards, local_records, terminal_records = _load_protocol(args)
    if args.mode == "benchmark":
        report = run_benchmark(args, protocol, fit_boards, output_dir)
    else:
        report = run_experiment(
            args,
            protocol,
            fit_boards,
            local_records,
            terminal_records,
            output_dir,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
