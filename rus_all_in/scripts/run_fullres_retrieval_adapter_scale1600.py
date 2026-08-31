#!/usr/bin/env python3
"""Run the signed 400-to-1600 full-resolution retrieval-adapter scale test."""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from aiijc_puzzle.fullres_retrieval_adapter import (
    FullResolutionRetrievalAdapter,
    retrieval_adapter_contract,
)
from aiijc_puzzle.protocol import sha256_file
from aiijc_puzzle.socket_sorter_production import load_socket_checkpoint
from aiijc_puzzle.synthetic_socket_evaluation import (
    ExactSyntheticReference,
    make_exact_synthetic_case,
    names_digest,
)

try:
    from scripts import run_fullres_retrieval_adapter as pilot
except ModuleNotFoundError:  # Direct ``python scripts/run_*.py`` execution.
    import run_fullres_retrieval_adapter as pilot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/fullres_retrieval_adapter_scale1600_preregistered_v1.json"
CONFIG_SHA256 = "840a50cba2dea4c7c57300f65ce18613d62ec26f696dd25311a58a25e0605563"
PILOT_REPORT = (
    PROJECT_ROOT
    / "outputs/fullres-retrieval-adapter/fixed-s100-s400-local16-v1/report.json"
)
PILOT_REPORT_SHA256 = "5fafb0307586669c7b7c9eaa4699fda1a3bd1250ca921fc48dd7e86af0bdefbb"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/fullres-retrieval-adapter/scale1600-local16-v1"
)
DEFAULT_SERVER_CONFIG = (
    PROJECT_ROOT / "configs/fullres_retrieval_adapter_server_scale1600_v1.json"
)
CHECKPOINT_STEPS = (400, 1600)
TRAIN_SEED = pilot.TRAIN_SEED
EVAL_SEED = pilot.EVAL_SEED
LEARNING_RATE = pilot.LEARNING_RATE
WEIGHT_DECAY = pilot.WEIGHT_DECAY
TOP_K = pilot.TOP_K
LOCAL_KS = pilot.LOCAL_KS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=pilot.DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=pilot.DEFAULT_TARGETS)
    parser.add_argument(
        "--socket-checkpoint", type=Path, default=pilot.DEFAULT_SOCKET
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        shown = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        shown = str(resolved)
    return {"path": shown, "sha256": sha256_file(resolved)}


def _write_json(path: Path, payload: Any) -> None:
    pilot._write_json(path, payload)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    pilot._write_npz(path, arrays)


def _require_signed_inputs(args: argparse.Namespace) -> None:
    expected = {
        CONFIG: CONFIG_SHA256,
        PILOT_REPORT: PILOT_REPORT_SHA256,
        PROJECT_ROOT / "src/aiijc_puzzle/fullres_retrieval_adapter.py": (
            "fc28b6c361a2e637ae23fcff1d1b0c03fc85aada8076c06035c899a491be35b6"
        ),
        pilot.PARENT_REPORT: pilot.PARENT_REPORT_SHA256,
        args.socket_checkpoint.resolve(): (
            "0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670"
        ),
        args.manifest.resolve(): (
            "4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da"
        ),
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"signed scale1600 input changed: {path}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("status") != "signed-fixed-protocol":
        raise RuntimeError("scale1600 preregistration status changed")


def _training_specs() -> tuple[pilot.TrainSpec, ...]:
    generator = np.random.default_rng(TRAIN_SEED + 17)
    return tuple(
        pilot.TrainSpec(
            source_index=int(generator.integers(32)),
            corruption_seed=int(generator.integers(0, 2**31 - 1)),
            permutation_seed=int(generator.integers(0, 2**31 - 1)),
        )
        for _ in range(CHECKPOINT_STEPS[-1])
    )


def _verify_stream_prefix() -> None:
    old = pilot._training_specs()
    new = _training_specs()
    if tuple(new[: len(old)]) != tuple(old):
        raise RuntimeError("extended training stream does not preserve the first 400 specs")


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
                key: value.detach().cpu()
                for key, value in adapter.state_dict().items()
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


def _train(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    fit_boards: list[Any],
    output_dir: Path,
    *,
    device: torch.device,
    checkpoint_callback: Any | None = None,
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
                pilot._materialise_train_board,
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
                    pilot._materialise_train_board,
                    fit_boards[next_spec.source_index],
                    next_spec,
                )
                submit += 1
            diagnostics, update_seconds = pilot._one_update(
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
                if checkpoint_callback is not None:
                    checkpoint_callback(
                        step,
                        checkpoints[step],
                        adapter,
                        socket,
                    )
            if step == 1 or step % 25 == 0 or step in CHECKPOINT_STEPS:
                recent = history[-min(25, len(history)) :]
                print(
                    json.dumps(
                        {
                            "event": "train_scale1600",
                            "step": step,
                            "loss": float(
                                np.mean([item["loss"] for item in recent])
                            ),
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
    checkpoints: dict[int, Path], *, device: torch.device
) -> dict[str, FullResolutionRetrievalAdapter]:
    result: dict[str, FullResolutionRetrievalAdapter] = {}
    for step, path in checkpoints.items():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("config_sha256") != CONFIG_SHA256 or payload.get("step") != step:
            raise RuntimeError("scale1600 checkpoint contract mismatch")
        model = FullResolutionRetrievalAdapter().to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        result[f"adapter_step{step}"] = model
    return result


def _freeze_panel(
    records: tuple[dict[str, Any], ...],
    *,
    panel_name: str,
    targets: Path,
    socket: Any,
    adapters: dict[str, FullResolutionRetrievalAdapter],
    device: torch.device,
    output_dir: Path,
) -> tuple[list[pilot.FrozenCase], dict[str, ExactSyntheticReference]]:
    boards = pilot.parent._prepare_boards(records, targets)
    cases: list[pilot.FrozenCase] = []
    references: dict[str, ExactSyntheticReference] = {}
    for index, board in enumerate(boards, start=1):
        item, reference = make_exact_synthetic_case(
            board.tiles,
            source_filename=board.filename,
            draw_index=0,
            seed=EVAL_SEED,
        )
        cases.append(
            pilot._freeze_case(
                item,
                socket=socket,
                adapters=adapters,
                device=device,
            )
        )
        references[item.case_id] = reference
        print(
            json.dumps(
                {
                    "event": "freeze_scale1600",
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
    freeze = panel / "pre-score-freeze.json"
    _write_npz(archive, arrays)
    _write_json(
        metadata,
        {
            "schema": "aiijc-fullres-retrieval-adapter-scale1600-freeze-v1",
            "panel": panel_name,
            "contains_clean_pixels_or_exact_references": False,
            "contains_restored_pixels_or_layouts": False,
            "raw_evidence_preserved": True,
            "matcher_view_only": True,
            "rows": rows,
        },
    )
    _write_json(
        freeze,
        {
            "schema": "aiijc-fullres-retrieval-adapter-scale1600-pre-score-v1",
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


def _reciprocal_rows(
    cases: list[pilot.FrozenCase],
    references: dict[str, ExactSyntheticReference],
    variant: str,
) -> list[tuple[float, bool]]:
    rows: list[tuple[float, bool]] = []
    for case in cases:
        reference = references[case.case_id].tile_at_position
        for axis in ("right", "down"):
            truth = pilot.parent._truth_by_anchor(reference, axis=axis)
            valid = truth >= 0
            evidence = case.reciprocal[variant][axis]
            admitted = valid & evidence["reciprocal"]
            correct = evidence["target"] == truth
            rows.extend(
                (float(confidence), bool(ok))
                for confidence, ok in zip(
                    evidence["confidence"][admitted],
                    correct[admitted],
                    strict=True,
                )
            )
    return rows


def _precision_at(rows: list[tuple[float, bool]], count: int) -> float:
    if count <= 0:
        return 0.0
    ordered = sorted(rows, key=lambda value: -value[0])[:count]
    return sum(int(correct) for _, correct in ordered) / count


def _score_panel(
    cases: list[pilot.FrozenCase],
    references: dict[str, ExactSyntheticReference],
) -> dict[str, Any]:
    result = pilot._score_panel(cases, references)
    rows400 = _reciprocal_rows(cases, references, "adapter_step400")
    rows1600 = _reciprocal_rows(cases, references, "adapter_step1600")
    count = min(len(rows400), len(rows1600))
    valid_queries = int(result["retrieval"]["raw_d64_ot"]["pooled_total"])
    precision400 = _precision_at(rows400, count)
    precision1600 = _precision_at(rows1600, count)
    result["checkpoint_matched_reciprocal"] = {
        "matched_query_count": count,
        "matched_coverage": count / valid_queries,
        "adapter_step400_precision": precision400,
        "adapter_step1600_precision": precision1600,
        "step1600_minus_step400_precision": precision1600 - precision400,
    }
    return result


def _scaling(local: dict[str, Any]) -> dict[str, Any]:
    retrieval400 = local["retrieval"]["adapter_step400"]
    retrieval1600 = local["retrieval"]["adapter_step1600"]
    supply400 = local["supply"]["adapter_step400"]
    supply1600 = local["supply"]["adapter_step1600"]
    return {
        "retrieval": {
            key: retrieval1600[key] - retrieval400[key]
            for key in (
                "pooled_r1",
                "pooled_r5",
                "right_r1",
                "right_r5",
                "down_r1",
                "down_r5",
            )
        },
        "raw_union_top32": {
            "pooled_coverage": (
                supply1600["pooled_union_coverage"]
                - supply400["pooled_union_coverage"]
            ),
            "right_coverage": (
                supply1600["axes"]["right"]["union_coverage"]
                - supply400["axes"]["right"]["union_coverage"]
            ),
            "down_coverage": (
                supply1600["axes"]["down"]["union_coverage"]
                - supply400["axes"]["down"]["union_coverage"]
            ),
        },
        "matched_reciprocal": local["checkpoint_matched_reciprocal"],
    }


def _local_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    raw = metrics["retrieval"]["raw_d64_ot"]
    candidate = metrics["retrieval"]["adapter_step1600"]
    precision = metrics["reciprocal"]["matched_vs_raw"]["adapter_step1600"]
    r1_gain = candidate["pooled_r1"] - raw["pooled_r1"]
    r5_gain = candidate["pooled_r5"] - raw["pooled_r5"]
    passed = bool(
        r1_gain >= 0.005
        and r5_gain >= 0.0
        and precision["precision_gain"] >= 0.002
        and precision["matched_coverage"] >= 0.03
    )
    return {
        "r1_gain": r1_gain,
        "r5_gain": r5_gain,
        "matched_reciprocal_precision_gain": precision["precision_gain"],
        "matched_reciprocal_coverage": precision["matched_coverage"],
        "terminal_open_gate_passed": passed,
    }


def _terminal_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    raw = metrics["retrieval"]["raw_d64_ot"]
    candidate = metrics["retrieval"]["adapter_step1600"]
    precision = metrics["reciprocal"]["matched_vs_raw"]["adapter_step1600"]
    r1_gain = candidate["pooled_r1"] - raw["pooled_r1"]
    r5_gain = candidate["pooled_r5"] - raw["pooled_r5"]
    passed = bool(
        r1_gain >= 0.0
        and r5_gain >= 0.0
        and precision["precision_gain"] >= 0.0
        and precision["matched_coverage"] >= 0.03
    )
    return {
        "r1_gain": r1_gain,
        "r5_gain": r5_gain,
        "matched_reciprocal_precision_gain": precision["precision_gain"],
        "matched_reciprocal_coverage": precision["matched_coverage"],
        "transfer_passed": passed,
    }


def _write_server_config(
    *,
    checkpoint: Path,
    local: dict[str, Any],
    scaling: dict[str, Any],
) -> dict[str, Any] | None:
    positive = bool(
        scaling["retrieval"]["pooled_r5"] > 0.0
        and scaling["raw_union_top32"]["pooled_coverage"] > 0.0
    )
    if not positive:
        return None
    payload = {
        "schema": "aiijc-fullres-retrieval-adapter-server-scale1600-v1",
        "status": "server-ready-training-artifact-only-no-solver-promotion",
        "created_only_after_positive_fixed_scaling_slope": True,
        "preregistration": _record(CONFIG),
        "checkpoint": _record(checkpoint),
        "architecture_and_loss_unchanged": True,
        "training": {
            "steps": 1600,
            "fit_sources": 32,
            "seed": TRAIN_SEED,
            "device_recommendation": "CUDA or MPS; checkpoint is portable",
        },
        "local16_scaling_step1600_minus_step400": scaling,
        "local16_step1600_retrieval": local["retrieval"]["adapter_step1600"],
        "raw_control_required": True,
        "matcher_view_only": True,
        "decoder_or_production_authorized": False,
    }
    _write_json(DEFAULT_SERVER_CONFIG, payload)
    return _record(DEFAULT_SERVER_CONFIG)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_signed_inputs(args)
    _verify_stream_prefix()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol, fit_boards, local_records, terminal_records = pilot._load_protocol(args)
    device = pilot._device(args.device)
    checkpoint400_interim: dict[str, Any] = {}

    def checkpoint_callback(
        step: int,
        checkpoint: Path,
        adapter: FullResolutionRetrievalAdapter,
        socket: Any,
    ) -> None:
        if step != 400:
            return
        adapter.eval()
        cases, references = _freeze_panel(
            local_records,
            panel_name="checkpoint400-local16",
            targets=args.targets,
            socket=socket,
            adapters={"adapter_step400": adapter},
            device=device,
            output_dir=output_dir,
        )
        metrics = pilot._score_panel(cases, references)
        checkpoint400_interim.update(
            {
                "checkpoint": _record(checkpoint),
                "metrics": metrics,
            }
        )
        _write_json(
            output_dir / "checkpoint400-local16-metrics.json",
            checkpoint400_interim,
        )
        raw = metrics["retrieval"]["raw_d64_ot"]
        candidate = metrics["retrieval"]["adapter_step400"]
        precision = metrics["reciprocal"]["matched_vs_raw"]["adapter_step400"]
        print(
            json.dumps(
                {
                    "event": "checkpoint400_local16_metrics_ready",
                    "raw_pooled_r1": raw["pooled_r1"],
                    "adapter_pooled_r1": candidate["pooled_r1"],
                    "r1_gain": candidate["pooled_r1"] - raw["pooled_r1"],
                    "raw_pooled_r5": raw["pooled_r5"],
                    "adapter_pooled_r5": candidate["pooled_r5"],
                    "r5_gain": candidate["pooled_r5"] - raw["pooled_r5"],
                    "matched_reciprocal_precision_gain": precision[
                        "precision_gain"
                    ],
                    "matched_reciprocal_coverage": precision[
                        "matched_coverage"
                    ],
                }
            ),
            flush=True,
        )
        adapter.train()

    checkpoints, history, runtime = _train(
        args,
        protocol,
        fit_boards,
        output_dir,
        device=device,
        checkpoint_callback=checkpoint_callback,
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
    scaling = _scaling(local)
    local_gate = _local_gate(local)
    server_config = _write_server_config(
        checkpoint=checkpoints[1600], local=local, scaling=scaling
    )
    terminal: dict[str, Any] = {"status": "skipped_by_local_gate"}
    terminal_gate: dict[str, Any] | None = None
    if local_gate["terminal_open_gate_passed"]:
        terminal_cases, terminal_references = _freeze_panel(
            terminal_records,
            panel_name="terminal16",
            targets=args.targets,
            socket=socket,
            adapters={"adapter_step1600": adapters["adapter_step1600"]},
            device=device,
            output_dir=output_dir,
        )
        terminal = pilot._score_panel(terminal_cases, terminal_references)
        terminal["status"] = "complete"
        terminal_gate = _terminal_gate(terminal)
    report = {
        "schema": "aiijc-fullres-retrieval-adapter-scale1600-report-v1",
        "status": (
            "terminal-transfer-passed-decoder-separately-eligible"
            if terminal_gate is not None and terminal_gate["transfer_passed"]
            else "local-gate-passed-terminal-transfer-failed"
            if terminal_gate is not None
            else "local-gate-failed-stop-no-terminal-no-decoder"
        ),
        "protocol": protocol,
        "contract": retrieval_adapter_contract(adapters["adapter_step1600"]),
        "configuration": {
            "config_sha256": CONFIG_SHA256,
            "device": str(device),
            "steps": 1600,
            "fixed_checkpoints": list(CHECKPOINT_STEPS),
            "training_seed": TRAIN_SEED,
            "evaluation_seed": EVAL_SEED,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler_t_max": 1600,
        },
        "training": {
            "runtime": runtime,
            "history": history,
            "checkpoints": {
                str(step): _record(path) for step, path in checkpoints.items()
            },
        },
        "checkpoint400_interim_local16": checkpoint400_interim,
        "local16": local,
        "scaling_step1600_minus_step400": scaling,
        "local_terminal_open_gate": local_gate,
        "terminal16": terminal,
        "terminal_transfer_gate": terminal_gate,
        "decoder": {
            "run": False,
            "eligible": bool(
                terminal_gate is not None and terminal_gate["transfer_passed"]
            ),
        },
        "server_ready_config": server_config,
        "legality": {
            "organizer_train_only": True,
            "raw_d64_evidence_immutable": True,
            "adapter_pixels_matcher_only": True,
            "competition_test_accessed": False,
            "submission_output_or_production_modified": False,
        },
        "artifacts": {
            "config": _record(CONFIG),
            "module": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/fullres_retrieval_adapter.py"
            ),
            "runner": _record(Path(__file__)),
            "socket": _record(args.socket_checkpoint),
            "parent_roster_report": _record(pilot.PARENT_REPORT),
            "pilot_report": _record(PILOT_REPORT),
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
    report = run(args)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
