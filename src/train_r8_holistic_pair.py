"""R8: holistic full-pair compatibility retriever for the fixed-orientation 24x24 puzzle.

R8 scores RGB pairs jointly as canonical 3x20x40 images.  It is deliberately
non-factorized: an anchor/candidate score cannot be written as a dot product of
independent tile embeddings.  The model sees pixels only; synthetic permutations
are used by the trainer to select supervised pair lists and evaluate retrieval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from canvas_data import CanvasDataset

GRID = 24
NFRAG = GRID * GRID
UP, RIGHT, DOWN, LEFT = range(4)
DROW = (-1, 0, 1, 0)
DCOL = (0, 1, 0, -1)
IGNORE = -100
DEFAULT_SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
DEFAULT_WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R8_holistic_full_pair")
R2L_MATCHED_CAL = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R7_full_contrastive_retriever\g1_capacity\r2l_matched_cal_report.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + identity


class HolisticPairNet(nn.Module):
    """Joint pair CNN with direction-specific scalar compatibility heads."""

    def __init__(self, width: int = 96, blocks: int = 5) -> None:
        super().__init__()
        if width % 8:
            raise ValueError("width must divide into GroupNorm groups of 8")
        self.width = int(width)
        self.blocks = int(blocks)
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
        )
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.pool = nn.Sequential(
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, width * 2),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, 1))
            for _ in range(4)
        ])

    def forward(self, pairs: Tensor, directions: Tensor) -> Tensor:
        if pairs.ndim != 4 or pairs.shape[1:] != (3, 20, 40):
            raise ValueError(f"expected (M,3,20,40) pair pixels, got {tuple(pairs.shape)}")
        if directions.ndim != 1 or directions.shape[0] != pairs.shape[0]:
            raise ValueError("directions must be M long")
        if torch.any((directions < 0) | (directions > 3)):
            raise ValueError("invalid direction")
        x = self.pool(self.body(self.stem(pairs)))
        all_scores = torch.cat([head(x) for head in self.heads], dim=1)
        return all_scores.gather(1, directions[:, None]).squeeze(1)


def clean_neighbour_map(perm: Tensor) -> Tensor:
    """Input-tile index of each directional neighbour; outside cells are IGNORE."""
    if perm.ndim != 1 or perm.numel() != NFRAG:
        raise ValueError(f"expected one permutation of {NFRAG} tiles")
    inv = torch.empty_like(perm)
    inv.scatter_(0, perm, torch.arange(NFRAG, device=perm.device))
    cells = perm.long()
    rows = torch.div(cells, GRID, rounding_mode="floor")
    cols = cells.remainder(GRID)
    targets = torch.full((4, NFRAG), IGNORE, dtype=torch.long, device=perm.device)
    for direction in range(4):
        nr = rows + DROW[direction]
        nc = cols + DCOL[direction]
        valid = (nr >= 0) & (nr < GRID) & (nc >= 0) & (nc < GRID)
        target_cells = (nr * GRID + nc).clamp(0, NFRAG - 1)
        targets[direction] = torch.where(valid, inv[target_cells], targets[direction])
    return targets


def make_joint_pairs(tiles: Tensor, anchors: Tensor, candidates: Tensor, directions: Tensor) -> Tensor:
    """Make physical directional pair images without altering any output tile orientation.

    Horizontal pairs are ordered directly.  Vertical physical top/bottom pairs
    are transposed only in the *model representation* so every CNN input has
    20x40 shape; reconstruction tiles themselves are never rotated or transposed.
    """
    if tiles.shape != (NFRAG, 3, 20, 20):
        raise ValueError(f"expected tile bag ({NFRAG},3,20,20), got {tuple(tiles.shape)}")
    if not (anchors.ndim == candidates.ndim == directions.ndim == 1):
        raise ValueError("anchors, candidates, directions must be flat tensors")
    if not (anchors.shape == candidates.shape == directions.shape):
        raise ValueError("pair index vectors must match")
    a = tiles[anchors]
    c = tiles[candidates]
    output = torch.empty((anchors.numel(), 3, 20, 40), dtype=tiles.dtype, device=tiles.device)
    right = directions == RIGHT
    left = directions == LEFT
    down = directions == DOWN
    up = directions == UP
    if right.any():
        output[right] = torch.cat((a[right], c[right]), dim=-1)
    if left.any():
        output[left] = torch.cat((c[left], a[left]), dim=-1)
    if down.any():
        output[down] = torch.cat((a[down], c[down]), dim=-2).transpose(-2, -1)
    if up.any():
        output[up] = torch.cat((c[up], a[up]), dim=-2).transpose(-2, -1)
    return output


def _direct_cells(cell: int) -> set[int]:
    r, c = divmod(cell, GRID)
    result: set[int] = set()
    for dr, dc in zip(DROW, DCOL):
        rr, cc = r + dr, c + dc
        if 0 <= rr < GRID and 0 <= cc < GRID:
            result.add(rr * GRID + cc)
    return result


def _hard_cells(cell: int) -> List[int]:
    r, c = divmod(cell, GRID)
    output: List[int] = []
    for rr in range(max(0, r - 3), min(GRID, r + 4)):
        for cc in range(max(0, c - 3), min(GRID, c + 4)):
            manhattan = abs(rr - r) + abs(cc - c)
            if 2 <= manhattan <= 3:
                output.append(rr * GRID + cc)
    return output


def sampled_pair_lists(perm: Tensor, *, anchors_per_board: int, negatives: int, rng: random.Random) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, int]]:
    """Return (anchor, direction, candidate-list, positive-index) for sampled InfoNCE.

    The first candidate is always the directed positive.  Negatives never include
    self or any true direct neighbour in any cardinal direction, preventing
    contradictory labels across directional rows.
    """
    if negatives < 5:
        raise ValueError("need at least 5 negatives for structural hard-negative quota")
    perm_cpu = [int(x) for x in perm.detach().cpu().tolist()]
    inverse = [0] * NFRAG
    for input_id, clean_cell in enumerate(perm_cpu):
        inverse[clean_cell] = input_id
    selected = rng.sample(range(NFRAG), k=min(anchors_per_board, NFRAG))
    anchors: List[int] = []
    directions: List[int] = []
    lists: List[List[int]] = []
    hard_count = 0
    for anchor in selected:
        cell = perm_cpu[anchor]
        row, col = divmod(cell, GRID)
        forbidden_cells = _direct_cells(cell) | {cell}
        allowed = [inverse[x] for x in range(NFRAG) if x not in forbidden_cells]
        for direction in range(4):
            rr, cc = row + DROW[direction], col + DCOL[direction]
            if not (0 <= rr < GRID and 0 <= cc < GRID):
                continue
            positive = inverse[rr * GRID + cc]
            hard = [inverse[x] for x in _hard_cells(cell) if x not in forbidden_cells]
            rng.shuffle(hard)
            hard = hard[: min(4, negatives)]
            hard_count += len(hard)
            used = set(hard) | {positive}
            pool = [x for x in allowed if x not in used]
            filler = rng.sample(pool, k=negatives - len(hard))
            candidate_list = [positive, *hard, *filler]
            if len(candidate_list) != negatives + 1 or len(set(candidate_list)) != len(candidate_list):
                raise RuntimeError("invalid sampled pair candidate list")
            anchors.append(anchor)
            directions.append(direction)
            lists.append(candidate_list)
    anchor_t = torch.tensor(anchors, dtype=torch.long, device=perm.device)
    direction_t = torch.tensor(directions, dtype=torch.long, device=perm.device)
    candidate_t = torch.tensor(lists, dtype=torch.long, device=perm.device)
    positive_t = torch.zeros(len(lists), dtype=torch.long, device=perm.device)
    stats = {"rows": len(lists), "hard_negatives": hard_count, "negatives_per_row": negatives}
    return anchor_t, direction_t, candidate_t, positive_t, stats


def sampled_loss(model: nn.Module, tiles: Tensor, perm: Tensor, *, anchors_per_board: int, negatives: int, rng: random.Random) -> Tuple[Tensor, Dict[str, int]]:
    anchors, directions, candidates, positive, stats = sampled_pair_lists(
        perm, anchors_per_board=anchors_per_board, negatives=negatives, rng=rng
    )
    rows, choices = candidates.shape
    flat_anchor = anchors[:, None].expand(rows, choices).reshape(-1)
    flat_direction = directions[:, None].expand(rows, choices).reshape(-1)
    pairs = make_joint_pairs(tiles, flat_anchor, candidates.reshape(-1), flat_direction)
    logits = model(pairs, flat_direction).reshape(rows, choices)
    loss = F.cross_entropy(logits, positive)
    if not torch.isfinite(loss):
        raise RuntimeError("R8 sampled loss is non-finite")
    stats["pair_tensors"] = int(pairs.shape[0])
    return loss, stats


def all_direct_targets(perm: Tensor) -> Tensor:
    return clean_neighbour_map(perm)


@torch.inference_mode()
def dense_scores(model: nn.Module, tiles: Tensor, *, pair_chunk: int) -> Tensor:
    """Score every non-self candidate in each direction, chunked for one RTX 2070."""
    if tiles.shape != (NFRAG, 3, 20, 20):
        raise ValueError("dense scoring expects one board tile bag")
    device = tiles.device
    scores = torch.full((4, NFRAG, NFRAG), -1.0e9, dtype=torch.float32, device=device)
    base = torch.arange(NFRAG, device=device)
    for direction in range(4):
        anchors = base[:, None].expand(NFRAG, NFRAG).reshape(-1)
        candidates = base[None, :].expand(NFRAG, NFRAG).reshape(-1)
        valid = anchors.ne(candidates)
        anchors, candidates = anchors[valid], candidates[valid]
        dirs = torch.full_like(anchors, direction)
        out: List[Tensor] = []
        for start in range(0, anchors.numel(), pair_chunk):
            end = min(anchors.numel(), start + pair_chunk)
            pairs = make_joint_pairs(tiles, anchors[start:end], candidates[start:end], dirs[start:end])
            out.append(model(pairs, dirs[start:end]))
        scores[direction, anchors, candidates] = torch.cat(out)
    return scores


@torch.inference_mode()
def evaluate_retrieval(model: nn.Module, dataset: CanvasDataset, *, examples: int, pair_chunk: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    totals = {1: 0, 5: 0, 20: 0, 96: 0, 128: 0}
    denominator = 0
    for index in range(examples):
        row = dataset[index % len(dataset)]
        if not bool(row["has_perm"].item()):
            raise RuntimeError("R8 capacity evaluation requires labelled synthetic tile bags")
        tiles = row["tiles"].to(device)
        targets = all_direct_targets(row["perm"].to(device))
        scores = dense_scores(model, tiles, pair_chunk=pair_chunk)
        for k in totals:
            guess = scores.topk(k=min(k, NFRAG - 1), dim=-1).indices
            valid = targets.ne(IGNORE)
            hit = guess.eq(targets.unsqueeze(-1)).any(dim=-1) & valid
            totals[k] += int(hit.sum().item())
            if k == 1:
                denominator += int(valid.sum().item())
    result: Dict[str, float] = {"examples": float(examples), "valid_directed_edges": float(denominator)}
    for k, value in totals.items():
        result[f"recall_at_{k}"] = value / max(1, denominator)
    return result


def load_split(path: Path) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = payload.get("splits", payload)
    names = {key: list(split[key]) for key in ("fit", "cal", "dev", "reserve")}
    for left in names:
        for right in names:
            if left < right and set(names[left]) & set(names[right]):
                raise RuntimeError(f"source leakage: {left}/{right}")
    return names


def stack_boards(dataset: CanvasDataset, batch_size: int, rng: random.Random, device: torch.device) -> List[Tuple[Tensor, Tensor]]:
    boards: List[Tuple[Tensor, Tensor]] = []
    for _ in range(batch_size):
        row = dataset[rng.randrange(len(dataset))]
        if not bool(row["has_perm"].item()):
            raise RuntimeError("R8 training requires synthetic labelled tile bags")
        boards.append((row["tiles"].to(device, non_blocking=True), row["perm"].to(device, non_blocking=True)))
    return boards


def model_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def smoke(model: nn.Module, dataset: CanvasDataset, device: torch.device, seed: int) -> Dict[str, object]:
    rng = random.Random(seed)
    tiles, perm = stack_boards(dataset, 1, rng, device)[0]
    anchors, directions, candidates, positive, stats = sampled_pair_lists(perm, anchors_per_board=4, negatives=15, rng=rng)
    flat_anchor = anchors[:, None].expand_as(candidates).reshape(-1)
    flat_direction = directions[:, None].expand_as(candidates).reshape(-1)
    pairs = make_joint_pairs(tiles, flat_anchor, candidates.reshape(-1), flat_direction)
    targets = all_direct_targets(perm)
    direct_set = {int(x) for x in targets[targets.ne(IGNORE)].detach().cpu().tolist()}
    negative_values = candidates[:, 1:].reshape(-1)
    self_negatives = int(negative_values.eq(flat_anchor.reshape(-1, candidates.shape[1])[:, 1:].reshape(-1)).sum().item())
    # Per-row direct-neighbour exclusion is checked using original clean cells.
    forbidden_direct_negatives = 0
    for row in range(candidates.shape[0]):
        anchor = int(anchors[row].item())
        direct_values = set(int(x) for x in targets[:, anchor][targets[:, anchor].ne(IGNORE)].tolist())
        forbidden_direct_negatives += sum(int(x) in direct_values for x in candidates[row, 1:].tolist())
    logits = model(pairs, flat_direction).reshape(candidates.shape)
    loss = F.cross_entropy(logits, positive)
    if pairs.shape[1:] != (3, 20, 40) or self_negatives or forbidden_direct_negatives or not torch.isfinite(loss):
        raise RuntimeError("R8 G0 pair/negative/loss invariant failed")
    return {
        "passed": True,
        "pair_shape": list(pairs.shape),
        "sampled_logit_shape": list(logits.shape),
        "loss": float(loss.item()),
        "negative_checks": {"self_negatives": self_negatives, "direct_neighbour_negatives": forbidden_direct_negatives},
        "sampler": stats,
        "model_inputs": ["joint_pair_pixels"],
        "label_only": ["synthetic_perm"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--anchors-per-board", type=int, default=96)
    parser.add_argument("--negatives", type=int, default=15)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--cal-examples", type=int, default=32)
    parser.add_argument("--pair-chunk", type=int, default=4096)
    parser.add_argument("--fit-n", type=int, default=5360)
    parser.add_argument("--cal-n", type=int, default=670)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.anchors_per_board < 1:
        raise ValueError("steps, batch-size and anchors-per-board must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    splits = load_split(args.split)
    fit_names, cal_names = splits["fit"][:args.fit_n], splits["cal"][:args.cal_n]
    if len(fit_names) != args.fit_n or len(cal_names) != args.cal_n:
        raise ValueError("requested source count exceeds pinned split")
    args.work.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.work / "r8_report.json"
    provenance = {
        "split": str(args.split), "split_sha256": sha256(args.split), "fit_count": len(fit_names), "cal_count": len(cal_names),
        "fit_cal_overlap": len(set(fit_names) & set(cal_names)), "real_prob": 0.0, "orientation": "fixed_no_rotations",
        "model_inputs": ["joint_pair_pixels"], "label_only": ["synthetic_perm"],
    }
    (args.work / "r8_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    fit_dataset = CanvasDataset(fit_names, patch=4, real_prob=0.0, seed=args.seed + 11)
    cal_dataset = CanvasDataset(cal_names, patch=4, real_prob=0.0, seed=args.seed + 29)
    model = HolisticPairNet(width=args.width, blocks=args.blocks).to(device)
    smoke_report = smoke(model, fit_dataset, device, args.seed + 101)
    smoke_mode = device.type == "cpu" and args.steps == 1
    effective_batch = 1 if smoke_mode else args.batch_size
    effective_anchors = 4 if smoke_mode else args.anchors_per_board
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)
    rng = random.Random(args.seed + 307)
    history: List[Dict[str, object]] = []
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        boards = stack_boards(fit_dataset, effective_batch, rng, device)
        losses: List[Tensor] = []
        rows, pairs = 0, 0
        optimizer.zero_grad(set_to_none=True)
        for tiles, perm in boards:
            loss, stats = sampled_loss(model, tiles, perm, anchors_per_board=effective_anchors, negatives=args.negatives, rng=rng)
            losses.append(loss)
            rows += stats["rows"]
            pairs += stats["pair_tensors"]
        aggregate = torch.stack(losses).mean()
        aggregate.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item())
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            row = {"step": step, "train_loss": float(aggregate.item()), "grad_norm": grad_norm, "sampled_rows": rows, "pair_tensors": pairs, "lr": float(optimizer.param_groups[0]["lr"]), "elapsed_s": round(time.time() - started, 2)}
            history.append(row)
            print(json.dumps(row), flush=True)
            torch.save({"model": model.state_dict(), "architecture": {"width": args.width, "blocks": args.blocks}, "args": jsonable(vars(args)), "step": step, "row": row, "provenance": provenance}, args.work / "r8_last.pt")
    if smoke_mode:
        final: Dict[str, object] = {"experiment": "R8_holistic_full_pair", "gate": "G0_smoke", "parameters": model_parameters(model), "smoke": smoke_report, "history": history, "provenance": provenance}
    else:
        cal_metrics = evaluate_retrieval(model, cal_dataset, examples=args.cal_examples, pair_chunk=args.pair_chunk, device=device)
        r2l_metric = None
        if R2L_MATCHED_CAL.exists():
            r2l_metric = json.loads(R2L_MATCHED_CAL.read_text(encoding="utf-8"))["metrics"].get("r20")
        final = {"experiment": "R8_holistic_full_pair", "gate": "G1_capacity", "parameters": model_parameters(model), "smoke": smoke_report, "history": history, "cal": cal_metrics, "matched_frozen_r2l_recall_at_20": r2l_metric, "r8_minus_r2l_pp": None if r2l_metric is None else 100.0 * (cal_metrics["recall_at_20"] - r2l_metric), "required_margin_pp": 3.0, "provenance": provenance, "artifacts": {"last": str(args.work / "r8_last.pt")}}
    report_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
