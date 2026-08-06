"""Bounded information gate for the relative 2D Frontier Pointer.

This script deliberately stops before full puzzle rollout.  It asks the one
question that decides whether the branch is viable: can a relative 5x5 halo of
correctly placed dirty tiles identify the missing dirty tile among the whole
remaining bag?  No absolute cell coordinate is ever provided to the model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from canvas_data import CanvasDataset
from config import GRID, NFRAG, SEED
from frontier_pointer import FrontierPointer
from imgio import train_val_split


WINDOW = 5
CENTER = WINDOW * WINDOW // 2
OFFSETS = sorted(
    ((dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if (dr, dc) != (0, 0)),
    key=lambda item: (max(abs(item[0]), abs(item[1])), abs(item[0]) + abs(item[1]), item),
)


def parse_contexts(value: str) -> tuple[int, ...]:
    try:
        contexts = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as error:
        raise argparse.ArgumentTypeError("contexts must be comma-separated integers") from error
    if not contexts or min(contexts) < 1 or max(contexts) > len(OFFSETS):
        raise argparse.ArgumentTypeError(f"contexts must lie in [1,{len(OFFSETS)}]")
    return contexts


def make_queries(
    perm: Tensor,
    *,
    queries: int,
    contexts: Sequence[int],
    rng: np.random.Generator,
    corrupt_probability: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create relative halo queries from ``perm[input_tile] -> clean_cell``."""
    values = perm.detach().cpu().numpy().astype(np.int64, copy=False)
    if values.shape != (NFRAG,) or np.unique(values).size != NFRAG:
        raise ValueError("perm must be a complete 576-element permutation")
    inverse = np.empty(NFRAG, dtype=np.int64)
    inverse[values] = np.arange(NFRAG, dtype=np.int64)
    context_indices = np.full((queries, WINDOW * WINDOW), -1, dtype=np.int64)
    targets = np.empty(queries, dtype=np.int64)
    valid = np.ones(queries, dtype=bool)
    context_schedule = np.resize(np.asarray(contexts, dtype=np.int64), queries)
    rng.shuffle(context_schedule)

    for query, amount in enumerate(context_schedule):
        # An interior gate isolates the information question from boundary
        # padding.  Boundary contexts are introduced only after this gate.
        row = int(rng.integers(2, GRID - 2))
        column = int(rng.integers(2, GRID - 2))
        target_cell = row * GRID + column
        targets[query] = inverse[target_cell]
        pool_size = max(8, int(amount))
        picked = rng.choice(pool_size, size=int(amount), replace=False)
        used_slots: list[int] = []
        for offset_index in picked:
            dr, dc = OFFSETS[int(offset_index)]
            slot = (dr + 2) * WINDOW + (dc + 2)
            neighbour_cell = (row + dr) * GRID + column + dc
            context_indices[query, slot] = inverse[neighbour_cell]
            used_slots.append(slot)

        if corrupt_probability > 0.0 and rng.random() < corrupt_probability:
            valid[query] = False
            replace_slot = int(rng.choice(used_slots))
            banned = set(context_indices[query, context_indices[query] >= 0].tolist())
            banned.add(int(targets[query]))
            choices = np.asarray([tile for tile in range(NFRAG) if tile not in banned], dtype=np.int64)
            context_indices[query, replace_slot] = int(rng.choice(choices))

    occupied = context_indices >= 0
    available = np.ones((queries, NFRAG), dtype=bool)
    for query in range(queries):
        available[query, context_indices[query, occupied[query]]] = False
        if not available[query, targets[query]]:
            raise AssertionError("the true tile was accidentally masked")
    return (
        torch.from_numpy(context_indices),
        torch.from_numpy(occupied),
        torch.from_numpy(available),
        torch.from_numpy(targets),
        torch.from_numpy(valid),
    )


def stack_queries(
    perms: Tensor,
    *,
    queries: int,
    contexts: Sequence[int],
    rng: np.random.Generator,
    corrupt_probability: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    rows = [
        make_queries(
            perm,
            queries=queries,
            contexts=contexts,
            rng=rng,
            corrupt_probability=corrupt_probability,
        )
        for perm in perms
    ]
    return tuple(torch.stack([row[index] for row in rows]) for index in range(5))  # type: ignore[return-value]


def _device_batch(batch: tuple[Tensor, ...], device: torch.device) -> tuple[Tensor, ...]:
    return tuple(item.to(device, non_blocking=True) for item in batch)


@torch.no_grad()
def evaluate(
    model: FrontierPointer,
    samples: Sequence[dict[str, Tensor]],
    *,
    contexts: Sequence[int],
    eval_queries: int,
    confidence: float,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model.eval()
    result: dict[str, dict[str, float]] = {}
    per_image = max(1, math.ceil(eval_queries / len(samples)))
    for amount in contexts:
        ranks: list[np.ndarray] = []
        confidences: list[np.ndarray] = []
        correct: list[np.ndarray] = []
        values: list[np.ndarray] = []
        rng = np.random.default_rng(seed + 10_007 * int(amount))
        for sample in samples:
            tiles = sample["tiles"].unsqueeze(0).to(device)
            query_batch = make_queries(
                sample["perm"], queries=per_image, contexts=(amount,), rng=rng
            )
            context_indices, occupied, available, targets, _valid = _device_batch(
                tuple(item.unsqueeze(0) for item in query_batch), device
            )
            output = model(tiles, context_indices, occupied, available)
            logits = output["logits"][0].float()
            target = targets[0]
            true_score = logits.gather(1, target[:, None])
            rank = 1 + (logits > true_score).sum(1)
            probability = torch.softmax(logits, dim=-1)
            conf, prediction = probability.max(1)
            ranks.append(rank.cpu().numpy())
            confidences.append(conf.cpu().numpy())
            correct.append(prediction.eq(target).cpu().numpy())
            values.append(torch.sigmoid(output["value"][0]).cpu().numpy())
        rank = np.concatenate(ranks)[:eval_queries]
        conf = np.concatenate(confidences)[:eval_queries]
        hit = np.concatenate(correct)[:eval_queries]
        value = np.concatenate(values)[:eval_queries]
        selected = conf >= confidence
        result[str(amount)] = {
            "queries": int(len(rank)),
            "r1": float(np.mean(rank <= 1)),
            "r5": float(np.mean(rank <= 5)),
            "mrr": float(np.mean(1.0 / rank)),
            "mean_confidence": float(np.mean(conf)),
            "high_confidence_threshold": float(confidence),
            "high_confidence_precision": float(np.mean(hit[selected])) if selected.any() else 0.0,
            "high_confidence_coverage": float(np.mean(selected)),
            "mean_context_value": float(np.mean(value)),
        }
    model.train()
    return result


def gate(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    four = metrics.get("4", {})
    eight = metrics.get("8", {})
    passed = bool(
        four.get("r1", 0.0) >= 0.30
        and four.get("r5", 0.0) >= 0.60
        and eight.get("r1", 0.0) >= 0.50
        and max(
            four.get("high_confidence_precision", 0.0),
            eight.get("high_confidence_precision", 0.0),
        ) >= 0.75
        and max(
            four.get("high_confidence_coverage", 0.0),
            eight.get("high_confidence_coverage", 0.0),
        ) >= 0.05
    )
    return {
        "pass": passed,
        "rule": (
            "context4 R@1>=0.30 and R@5>=0.60; context8 R@1>=0.50; "
            "high-confidence precision>=0.75 and coverage>=0.05"
        ),
    }


def save_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_smoke() -> None:
    train_names, _ = train_val_split()
    np.random.seed(91)
    sample = CanvasDataset(train_names[:1], real_prob=0.0, seed=91)[0]
    model = FrontierPointer(d=40, layers=1, heads=5)
    query_batch = make_queries(
        sample["perm"], queries=6, contexts=(2, 4, 8), rng=np.random.default_rng(92),
        corrupt_probability=0.25,
    )
    context, occupied, available, targets, valid = tuple(item.unsqueeze(0) for item in query_batch)
    output = model(sample["tiles"].unsqueeze(0), context, occupied, available)
    pointer = F.cross_entropy(output["logits"][valid], targets[valid])
    value = F.binary_cross_entropy_with_logits(output["value"], valid.float())
    loss = pointer + 0.2 * value
    loss.backward()
    if not torch.isfinite(loss) or output["logits"].shape != (1, 6, NFRAG):
        raise AssertionError("frontier trainer smoke failed")
    print(
        f"frontier trainer smoke: loss={float(loss.detach()):.4f} valid={int(valid.sum())}/6 "
        "sampler=ok gradient=ok"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-images", type=int, default=1)
    parser.add_argument("--queries", type=int, default=24)
    parser.add_argument("--contexts", type=parse_contexts, default=parse_contexts("2,4,8"))
    parser.add_argument("--eval-queries", type=int, default=512)
    parser.add_argument("--eval-images", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--corrupt-context", type=float, default=0.15)
    parser.add_argument("--value-weight", type=float, default=0.20)
    parser.add_argument("--d", type=int, default=160)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=SEED + 9187)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("E:/pazzle_work/ckpt/frontier_pointer_best.pt"))
    parser.add_argument("--report", type=Path, default=Path("E:/pazzle_work/gates/frontier_pointer_gate.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_images, args.queries, args.eval_queries, args.eval_images, args.eval_interval) < 1:
        parser.error("all count arguments must be positive")
    if not 0.0 <= args.corrupt_context < 1.0 or not 0.0 <= args.confidence <= 1.0:
        parser.error("corrupt-context and confidence must lie in [0,1]")
    return args


def main() -> None:
    args = parse_args()
    if args.smoke:
        run_smoke()
        return
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    train_names, val_names = train_val_split()
    train_data = CanvasDataset(train_names, real_prob=0.0, seed=args.seed)
    eval_data = CanvasDataset(val_names[: args.eval_images], real_prob=0.0, seed=args.seed + 1_000_000)
    # Materialize once: every evaluation interval sees identical distortion,
    # shuffle, and subsequently identical queries.
    np.random.seed(args.seed + 77)
    eval_samples = [eval_data[index] for index in range(len(eval_data))]
    np.random.seed(args.seed)

    model = FrontierPointer(d=args.d, layers=args.layers, heads=args.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, Any]] = []
    best = -1.0
    started = time.monotonic()
    print(
        f"frontier information gate: device={device} steps={args.steps} queries={args.queries} "
        f"contexts={args.contexts} params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        indices = rng.integers(0, len(train_data), size=args.batch_images)
        samples = [train_data[int(index)] for index in indices]
        tiles = torch.stack([sample["tiles"] for sample in samples]).to(device)
        perms = torch.stack([sample["perm"] for sample in samples])
        query_batch = _device_batch(
            stack_queries(
                perms,
                queries=args.queries,
                contexts=args.contexts,
                rng=rng,
                corrupt_probability=args.corrupt_context,
            ),
            device,
        )
        context_indices, occupied, available, targets, valid = query_batch
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(tiles, context_indices, occupied, available)
            pointer_loss = F.cross_entropy(output["logits"][valid], targets[valid])
            value_loss = F.binary_cross_entropy_with_logits(output["value"], valid.float())
            loss = pointer_loss + args.value_weight * value_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 50 == 0:
            elapsed = time.monotonic() - started
            eta = elapsed / step * (args.steps - step)
            print(
                f"step={step}/{args.steps} loss={float(loss.detach()):.4f} "
                f"ptr={float(pointer_loss.detach()):.4f} "
                f"value={float(value_loss.detach()):.4f} eta={eta / 60:.1f}m",
                flush=True,
            )
        if step % args.eval_interval == 0 or step == args.steps:
            metrics = evaluate(
                model,
                eval_samples,
                contexts=args.contexts,
                eval_queries=args.eval_queries,
                confidence=args.confidence,
                seed=args.seed + 123_000,
                device=device,
            )
            current_gate = gate(metrics)
            row = {"step": step, "metrics": metrics, "gate": current_gate}
            history.append(row)
            score = metrics.get("4", {}).get("r1", 0.0)
            if score > best:
                best = score
                save_checkpoint(
                    args.out,
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "metrics": metrics,
                        "contexts": args.contexts,
                        "model_config": {"d": args.d, "window": WINDOW, "layers": args.layers, "heads": args.heads},
                    },
                )
            report = {
                "experiment": "frontier_inpainting_pointer_information_gate",
                "absolute_position_input": False,
                "test_used": False,
                "data": {
                    "train_images": len(train_names),
                    "held_out_images": val_names[: args.eval_images],
                },
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "best_context4_r1": best,
                "history": history,
                "final_gate": current_gate,
                "checkpoint": str(args.out),
            }
            save_report(args.report, report)
            print(f"eval step={step}: {json.dumps(metrics)} gate={current_gate}", flush=True)
    print(f"saved report {args.report}; best context4 R@1={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
