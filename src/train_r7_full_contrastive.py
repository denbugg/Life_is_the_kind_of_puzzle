"""R7: directional full-board contrastive retriever for fixed-orientation 24x24 puzzles.

The model receives only the unordered, independently corrupted tile bag.  Ground-truth
permutations are consumed after score construction solely to form supervised oriented
neighbour labels.  For every valid source-direction edge, the positive is ranked against
all 575 non-self tile candidates in one InfoNCE denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from canvas_data import CanvasDataset

GRID = 24
NFRAG = GRID * GRID
DIR_NAMES = ("up", "right", "down", "left")
# Grid offsets and which candidate-facing key head must match the source query.
DROW = (-1, 0, 1, 0)
DCOL = (0, 1, 0, -1)
OPPOSITE = (2, 3, 0, 1)
IGNORE = -100

DEFAULT_SPLIT = Path(
    r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json"
)
DEFAULT_WORK = Path(
    r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def path_jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): path_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [path_jsonable(v) for v in value]
    return value


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class DirectionalFullContrastNet(nn.Module):
    """Shared full-tile CNN plus independent oriented query/key projections."""

    def __init__(self, width: int = 64, embedding_dim: int = 128, blocks: int = 4) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        self.width = int(width)
        self.embedding_dim = int(embedding_dim)
        self.blocks = int(blocks)
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
        )
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.tail = nn.Sequential(
            nn.GroupNorm(groups, width),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, width * 2),
            nn.SiLU(),
        )
        # Heads are purposefully direction-specific.  A rightward query is
        # compared with the candidate's left-facing key, etc.
        self.query_heads = nn.ModuleList(
            [nn.Linear(width * 2, embedding_dim, bias=False) for _ in range(4)]
        )
        self.key_heads = nn.ModuleList(
            [nn.Linear(width * 2, embedding_dim, bias=False) for _ in range(4)]
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

    def encode(self, tiles: Tensor) -> Tensor:
        if tiles.ndim != 5 or tiles.shape[1:] != (NFRAG, 3, 20, 20):
            raise ValueError(f"expected (B,{NFRAG},3,20,20), got {tuple(tiles.shape)}")
        b = tiles.shape[0]
        x = tiles.reshape(b * NFRAG, 3, 20, 20)
        x = self.stem(x)
        x = self.body(x)
        x = self.tail(x)
        return x.reshape(b, NFRAG, -1)

    def forward(self, tiles: Tensor) -> Tensor:
        features = self.encode(tiles)
        queries = torch.stack([F.normalize(head(features), dim=-1) for head in self.query_heads], dim=1)
        # keys[d] represents the tile boundary that faces direction d.
        keys = torch.stack([F.normalize(head(features), dim=-1) for head in self.key_heads], dim=1)
        facing_keys = keys[:, list(OPPOSITE)]
        scores = torch.einsum("bdne,bdme->bdnm", queries, facing_keys)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scores * scale


def neighbour_targets(perm: Tensor) -> Tensor:
    """Return input-tile target index for each valid oriented direct neighbour.

    `perm[b, i]` maps observed input tile i to its unshuffled clean-grid cell.
    Invalid external-grid directions use IGNORE.  A valid direct neighbour can
    never be self because the board cells are distinct.
    """
    if perm.ndim != 2 or perm.shape[1] != NFRAG:
        raise ValueError(f"expected perm (B,{NFRAG}), got {tuple(perm.shape)}")
    b = perm.shape[0]
    inv = torch.empty_like(perm)
    source = torch.arange(NFRAG, device=perm.device).expand(b, -1)
    inv.scatter_(1, perm, source)
    cells = perm
    rows = torch.div(cells, GRID, rounding_mode="floor")
    cols = cells.remainder(GRID)
    targets = torch.full((b, 4, NFRAG), IGNORE, dtype=torch.long, device=perm.device)
    for d in range(4):
        nr = rows + DROW[d]
        nc = cols + DCOL[d]
        valid = (nr >= 0) & (nr < GRID) & (nc >= 0) & (nc < GRID)
        target_cells = (nr * GRID + nc).clamp(0, NFRAG - 1)
        targets[:, d] = torch.where(valid, inv.gather(1, target_cells), targets[:, d])
    return targets


def masked_scores(raw_scores: Tensor) -> Tensor:
    """Exclude an anchor itself from every retrieval denominator."""
    if raw_scores.ndim != 4 or raw_scores.shape[1:] != (4, NFRAG, NFRAG):
        raise ValueError(f"expected (B,4,{NFRAG},{NFRAG}), got {tuple(raw_scores.shape)}")
    diagonal = torch.eye(NFRAG, dtype=torch.bool, device=raw_scores.device).view(1, 1, NFRAG, NFRAG)
    return raw_scores.masked_fill(diagonal, -1.0e9)


def contrastive_loss(scores: Tensor, targets: Tensor) -> Tuple[Tensor, int]:
    scores = masked_scores(scores)
    flat_scores = scores.reshape(-1, NFRAG)
    flat_targets = targets.reshape(-1)
    valid = int(flat_targets.ne(IGNORE).sum().item())
    if valid == 0:
        raise RuntimeError("no valid internal-grid neighbour labels")
    return F.cross_entropy(flat_scores, flat_targets, ignore_index=IGNORE), valid


def recall_at_k(scores: Tensor, targets: Tensor, k: int) -> Tuple[int, int]:
    scores = masked_scores(scores)
    guesses = scores.topk(k=min(k, NFRAG - 1), dim=-1).indices
    valid = targets.ne(IGNORE)
    hits = guesses.eq(targets.unsqueeze(-1)).any(dim=-1) & valid
    return int(hits.sum().item()), int(valid.sum().item())


def validate_targets(targets: Tensor) -> Dict[str, int]:
    valid = targets.ne(IGNORE)
    anchors = torch.arange(NFRAG, device=targets.device).view(1, 1, NFRAG)
    self_count = int(((targets == anchors) & valid).sum().item())
    return {"valid_edges": int(valid.sum().item()), "self_targets": self_count}


def stack_samples(dataset: CanvasDataset, batch_size: int, generator: random.Random, device: torch.device) -> Tuple[Tensor, Tensor]:
    # CanvasDataset itself applies the project-standard independent corruption,
    # then arbitrary bag permutation.  It is the only source of observed tiles.
    rows = [dataset[generator.randrange(len(dataset))] for _ in range(batch_size)]
    if any(not bool(row["has_perm"].item()) for row in rows):
        raise RuntimeError("R7 supervised retrieval must use synthetic labelled bags only")
    tiles = torch.stack([row["tiles"] for row in rows]).to(device, non_blocking=True)
    perm = torch.stack([row["perm"] for row in rows]).to(device, non_blocking=True)
    return tiles, perm


def load_split(path: Path) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = payload.get("splits", payload)
    required = ("fit", "cal", "dev", "reserve")
    result = {name: list(splits[name]) for name in required}
    if any(not result[name] for name in required):
        raise ValueError("source-disjoint manifest contains an empty required split")
    all_sets = {name: set(values) for name, values in result.items()}
    for left in required:
        for right in required:
            if left < right and all_sets[left] & all_sets[right]:
                raise RuntimeError(f"source leakage: {left}/{right} overlap")
    return result


def make_dataset(names: Sequence[str], seed: int) -> CanvasDataset:
    return CanvasDataset(names, patch=4, real_prob=0.0, seed=seed)


@torch.inference_mode()
def evaluate(model: nn.Module, dataset: CanvasDataset, *, examples: int, k_values: Sequence[int], seed: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    rng = random.Random(seed)
    totals = {int(k): 0 for k in k_values}
    denominator = 0
    for _ in range(examples):
        tiles, perm = stack_samples(dataset, 1, rng, device)
        targets = neighbour_targets(perm)
        scores = model(tiles)
        for k in k_values:
            hit, count = recall_at_k(scores, targets, int(k))
            totals[int(k)] += hit
            denominator += count if int(k) == int(k_values[0]) else 0
    report: Dict[str, float] = {"examples": float(examples), "valid_directed_edges": float(denominator)}
    for k in k_values:
        report[f"recall_at_{int(k)}"] = float(totals[int(k)] / max(1, denominator))
    return report


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_payload(model: nn.Module, args: argparse.Namespace, step: int, metrics: Dict[str, object], split_hash: str) -> Dict[str, object]:
    return {
        "experiment": "R7_directional_full_board_InfoNCE_retriever",
        "model": model.state_dict(),
        "architecture": {"width": args.width, "embedding_dim": args.embedding_dim, "blocks": args.blocks},
        "args": path_jsonable(vars(args)),
        "step": int(step),
        "metrics": metrics,
        "split_sha256": split_hash,
    }


def cpu_smoke(model: nn.Module, fit_dataset: CanvasDataset, device: torch.device, seed: int) -> Dict[str, object]:
    rng = random.Random(seed)
    tiles, perm = stack_samples(fit_dataset, 1, rng, device)
    targets = neighbour_targets(perm)
    sanity = validate_targets(targets)
    scores = model(tiles)
    loss, valid_edges = contrastive_loss(scores, targets)
    if tuple(scores.shape) != (1, 4, NFRAG, NFRAG):
        raise RuntimeError(f"R7 score tensor invariant failed: {tuple(scores.shape)}")
    if sanity["self_targets"] != 0:
        raise RuntimeError("R7 label invariant failed: self target present")
    if not torch.isfinite(loss):
        raise RuntimeError("R7 smoke loss is non-finite")
    return {
        "passed": True,
        "score_shape": list(scores.shape),
        "loss": float(loss.item()),
        "valid_edges": valid_edges,
        "target_validation": sanity,
        "model_inputs": ["tiles"],
        "label_only_tensors": ["perm"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--cal-examples", type=int, default=24)
    parser.add_argument("--fit-n", type=int, default=5360)
    parser.add_argument("--cal-n", type=int, default=670)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 and not args.eval_only:
        raise ValueError("steps must be positive")
    if args.batch_size < 1 or args.fit_n < 1 or args.cal_n < 1:
        raise ValueError("batch-size, fit-n, and cal-n must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu only for G0 smoke")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    splits = load_split(args.split)
    fit_names = splits["fit"][: args.fit_n]
    cal_names = splits["cal"][: args.cal_n]
    if len(fit_names) != args.fit_n or len(cal_names) != args.cal_n:
        raise ValueError("requested source count exceeds pinned split")
    args.work.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.work / "r7_report.json"
    split_hash = sha256(args.split)
    provenance = {
        "split": str(args.split),
        "split_sha256": split_hash,
        "fit_count": len(fit_names),
        "cal_count": len(cal_names),
        "fit_cal_overlap": len(set(fit_names) & set(cal_names)),
        "real_prob": 0.0,
        "orientation": "fixed_no_rotations",
        "model_inputs": ["synthetically_corrupted_permuted_tiles"],
        "labels": ["synthetic_perm_after_score_construction"],
    }
    (args.work / "r7_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    fit_dataset = make_dataset(fit_names, args.seed + 11)
    cal_dataset = make_dataset(cal_names, args.seed + 29)
    model = DirectionalFullContrastNet(args.width, args.embedding_dim, args.blocks).to(device)
    smoke = cpu_smoke(model, fit_dataset, device, args.seed + 101)
    if args.eval_only:
        checkpoint = args.checkpoint or args.work / "r7_best.pt"
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        cal_metrics = evaluate(model, cal_dataset, examples=args.cal_examples, k_values=(1, 5, 20, 96, 128), seed=args.seed + 203, device=device)
        final = {"experiment": "R7_directional_full_board_InfoNCE_retriever", "mode": "eval_only", "smoke": smoke, "cal": cal_metrics, "provenance": provenance}
        report_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        print(json.dumps(final, indent=2), flush=True)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.10)
    train_rng = random.Random(args.seed + 307)
    best_recall = -1.0
    history: List[Dict[str, object]] = []
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        tiles, perm = stack_samples(fit_dataset, args.batch_size, train_rng, device)
        targets = neighbour_targets(perm)
        optimizer.zero_grad(set_to_none=True)
        scores = model(tiles)
        loss, valid_edges = contrastive_loss(scores, targets)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item())
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            cal_metrics = evaluate(model, cal_dataset, examples=args.cal_examples, k_values=(1, 5, 20, 96, 128), seed=args.seed + step, device=device)
            row: Dict[str, object] = {
                "step": step,
                "train_loss": float(loss.item()),
                "valid_train_edges": valid_edges,
                "grad_norm": grad_norm,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_s": round(time.time() - started, 2),
                "cal": cal_metrics,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            selection = float(cal_metrics["recall_at_20"])
            if selection > best_recall:
                best_recall = selection
                torch.save(checkpoint_payload(model, args, step, row, split_hash), args.work / "r7_best.pt")
    last_metrics = history[-1]
    torch.save(checkpoint_payload(model, args, args.steps, last_metrics, split_hash), args.work / "r7_last.pt")
    final: Dict[str, object] = {
        "experiment": "R7_directional_full_board_InfoNCE_retriever",
        "gate": "G0_smoke" if args.device == "cpu" and args.steps == 1 else "G1_capacity",
        "parameters": count_parameters(model),
        "smoke": smoke,
        "best_cal_recall_at_20": best_recall,
        "history": history,
        "provenance": provenance,
        "artifacts": {"best": str(args.work / "r7_best.pt"), "last": str(args.work / "r7_last.pt")},
    }
    report_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
