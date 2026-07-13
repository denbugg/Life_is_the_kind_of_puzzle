#!/usr/bin/env python3
"""Confirm a frozen latent-edge alpha against the production W4 anchor.

This is a sequential, source-disjoint follow-up to Stage-1.  It never retunes
alpha and never runs QAP.  Its only question is whether the learned residual
reliably improves the actual W4 QAP cost on a previously unopened retrieval
holdout, even though it did not beat standalone HBT retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

import train_evaluate_latent_edge_embedding as stage1
from puzzle_assembly.latent_edge_embedding import load_latent_edge_checkpoint
from puzzle_assembly.learned import load_embedding_checkpoint
from puzzle_assembly.protocol import source_names_for_split
from puzzle_denoise_v2.inference import load_restorer


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="puzzle")
    parser.add_argument("--denoiser", default=stage1.DEFAULT_DENOISER)
    parser.add_argument("--hbt-checkpoint", default=stage1.DEFAULT_HBT)
    parser.add_argument("--latent-checkpoint", required=True)
    parser.add_argument("--manifest", default=stage1.DEFAULT_MANIFEST)
    parser.add_argument("--quarantine", default=stage1.DEFAULT_QUARANTINE)
    parser.add_argument("--split", default="assembly_incremental_gate")
    parser.add_argument("--offset", type=int, default=208)
    parser.add_argument("--sources", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--denoise-batch-size", type=int, default=192)
    parser.add_argument("--classical-chunk-size", type=int, default=64)
    parser.add_argument("--candidate-top-k", type=int, default=32)
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.split != "assembly_incremental_gate" or args.offset < 208:
        raise ValueError("production-anchor holdout must start at gate offset >=208")
    if args.sources <= 0 or args.alpha <= 0.0:
        raise ValueError("sources and alpha must be positive")
    return args


def _paired(values: list[float], label: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("paired deltas must be finite and non-empty")
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(array), size=(20000, len(array)))
    bootstrap = array[index].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "bootstrap_95_lower": float(np.quantile(bootstrap, 0.025)),
        "wins": int(np.sum(array > 0.0)),
        "win_fraction": float(np.mean(array > 0.0)),
        "worst": float(array.min()),
        "count": int(len(array)),
    }


def assess(records: list[dict[str, Any]], *, alpha: float) -> dict[str, Any]:
    score = f"alpha_{alpha:g}"
    panels = {}
    for panel in ("primary_kornia", "independent_libjpeg"):
        selected = [record for record in records if record["panel"] == panel]
        if not selected:
            raise ValueError(f"no records for panel {panel}")
        paired = {}
        for metric in ("recall_at_1", "mrr", "recall_at_5", "recall_at_32"):
            paired[metric] = _paired(
                [
                    float(record["scores"][score][metric])
                    - float(record["scores"]["w4"][metric])
                    for record in selected
                ],
                f"production-anchor:{panel}:{score}:{metric}",
            )
        coverage = float(np.mean([record["candidate_coverage"] for record in selected]))
        checks = {
            "recall_at_1_mean_ge_0.008": paired["recall_at_1"]["mean"] >= 0.008,
            "mrr_mean_ge_0.008": paired["mrr"]["mean"] >= 0.008,
            "recall_at_5_mean_ge_0": paired["recall_at_5"]["mean"] >= 0.0,
            "recall_at_32_mean_ge_0": paired["recall_at_32"]["mean"] >= 0.0,
            "recall_at_1_bootstrap_lower_gt_0": paired["recall_at_1"][
                "bootstrap_95_lower"
            ]
            > 0.0,
            "mrr_bootstrap_lower_gt_0": paired["mrr"]["bootstrap_95_lower"] > 0.0,
            "recall_at_1_wins_ge_10": paired["recall_at_1"]["wins"] >= 10,
            "recall_at_1_worst_ge_minus_0.02": paired["recall_at_1"]["worst"] >= -0.02,
            "candidate_coverage_ge_0.75": coverage >= 0.75,
        }
        panels[panel] = {
            "source_count": len(selected),
            "candidate_coverage": coverage,
            "paired_candidate_minus_w4": paired,
            "checks": checks,
            "passed": all(checks.values()),
        }
    return {
        "alpha": alpha,
        "comparator": "frozen_C1_HBTw4",
        "panels": panels,
        "passed": all(value["passed"] for value in panels.values()),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output}")
    names = source_names_for_split(
        args.split,
        manifest_path=args.manifest,
        quarantine_path=args.quarantine,
    )[args.offset : args.offset + args.sources]
    if len(names) != args.sources:
        raise RuntimeError("requested holdout extends beyond its split")
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    runtime = stage1.Runtime(rank=0, world_size=1, local_rank=0, device=device)
    restorer, _, denoiser_metadata = load_restorer(
        args.denoiser, device=str(device), state="ema"
    )
    hbt, hbt_metadata = load_embedding_checkpoint(args.hbt_checkpoint, device=device)
    model, latent_metadata = load_latent_edge_checkpoint(
        args.latent_checkpoint, device=device
    )
    frozen_alpha = latent_metadata.get("selected_alpha")
    if frozen_alpha is None or abs(float(frozen_alpha) - args.alpha) > 1e-12:
        raise RuntimeError(
            f"checkpoint selected_alpha={frozen_alpha!r} does not match frozen {args.alpha}"
        )
    eval_args = argparse.Namespace(
        data_root=args.data_root,
        seed=args.seed,
        denoise_batch_size=args.denoise_batch_size,
        classical_chunk_size=args.classical_chunk_size,
        candidate_top_k=args.candidate_top_k,
        candidate_cap=args.candidate_cap,
        alphas=f"0,{args.alpha:g}",
        panels="primary_kornia,independent_libjpeg",
    )
    started = time.perf_counter()
    records = stage1._evaluate_split(
        names,
        stage="production_anchor_holdout",
        args=eval_args,
        runtime=runtime,
        restorer=restorer,
        hbt_model=hbt,
        model=model,
    )
    aggregate = stage1.aggregate_records(records)
    gate = assess(records, alpha=args.alpha)
    report = {
        "schema_version": 1,
        "kind": "latent_edge_production_anchor_holdout",
        "status": "passed_qap_still_not_run" if gate["passed"] else "stop_no_w4_holdout_signal",
        "safe_for_submission": False,
        "qap_run": False,
        "args": vars(args),
        "device": str(device),
        "partition": f"{args.split}[{args.offset}:{args.offset + len(names)}]",
        "source_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "latent_checkpoint_sha256": _sha256(args.latent_checkpoint),
        "latent_metadata": latent_metadata,
        "denoiser_metadata": denoiser_metadata,
        "hbt_metadata": hbt_metadata,
        "aggregate": aggregate,
        "gate": gate,
        "records": records,
        "seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"event": "production_anchor_complete", "status": report["status"]}))


if __name__ == "__main__":
    main()
