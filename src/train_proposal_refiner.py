"""Train and gate the proposal-conditioned 24x24 permutation refiner.

Synthetic drafts contain translated intact domino or 2x2 components and are
constructed to have roughly the same 0.10--0.25 directed-neighbour accuracy as
the current candidate-ranker/buddies solver.  The paired-alignment encoders are
frozen.  The model must improve both exact placement and neighbour accuracy on
held-out images; otherwise this branch is rejected before real-solver wiring.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from config import CKPT_DIR, GRID, NFRAG, SEED, TRAIN_TGT
from distort import distort_frags
from eval_paired_alignment import PairedAlignment
from imgio import load, to_frags, train_val_split
from placement_metrics import neighbour_accuracy
from proposal_refiner import ProposalRefiner, count_parameters


def _fixed_units(mode: str) -> list[np.ndarray]:
    if mode == "domino_h":
        return [
            np.array([row * GRID + col, row * GRID + col + 1], np.int64)
            for row in range(GRID) for col in range(0, GRID, 2)
        ]
    if mode == "domino_v":
        return [
            np.array([row * GRID + col, (row + 1) * GRID + col], np.int64)
            for row in range(0, GRID, 2) for col in range(GRID)
        ]
    if mode == "block2":
        return [
            np.array(
                [
                    row * GRID + col,
                    row * GRID + col + 1,
                    (row + 1) * GRID + col,
                    (row + 1) * GRID + col + 1,
                ],
                np.int64,
            )
            for row in range(0, GRID, 2) for col in range(0, GRID, 2)
        ]
    raise ValueError(mode)


def component_proposal(
    rng: np.random.Generator,
    target_neighbour: float,
    mode: str | None = None,
) -> np.ndarray:
    """Return ``draft[cell] = true_tile`` with controlled intact components."""
    mode = mode or str(rng.choice(("domino_h", "domino_v", "block2")))
    if mode == "swap":
        # If a fraction f of cells stays fixed, roughly f^2 of physical edges
        # survives.  This supplies an easy-to-hard curriculum while retaining
        # an exact permutation and without expensive rejection sampling.
        fixed_fraction = float(np.sqrt(np.clip(target_neighbour, 0.0, 1.0)))
        moved = int(round((1.0 - fixed_fraction) * NFRAG))
        moved = min(NFRAG, max(2, moved))
        cells = rng.choice(NFRAG, size=moved, replace=False)
        draft = np.arange(NFRAG, dtype=np.int64)
        values = draft[cells].copy()
        for _ in range(8):
            rng.shuffle(values)
            if np.all(values != cells):
                break
        draft[cells] = values
        return draft
    units = _fixed_units(mode)
    edges_per_unit = 4 if mode == "block2" else 1
    count = int(round(float(target_neighbour) * (2 * GRID * (GRID - 1)) / edges_per_unit))
    count = max(0, min(count, len(units)))
    source_ids = rng.choice(len(units), size=count, replace=False)
    target_ids = rng.choice(len(units), size=count, replace=False)
    rng.shuffle(source_ids)

    draft = np.full(NFRAG, -1, np.int64)
    used = np.zeros(NFRAG, dtype=bool)
    for source_id, target_id in zip(source_ids, target_ids):
        source, target = units[int(source_id)], units[int(target_id)]
        # Fixed tilings make selected units disjoint.  Keeping the member order
        # preserves the correct right/down direction after a component move.
        draft[target] = source
        used[source] = True
    remaining_cells = np.flatnonzero(draft < 0)
    remaining_tiles = np.flatnonzero(~used)
    rng.shuffle(remaining_tiles)
    draft[remaining_cells] = remaining_tiles
    if np.unique(draft).size != NFRAG:
        raise AssertionError("proposal is not a permutation")
    return draft


class RefinerDataset(Dataset):
    def __init__(
        self,
        names: list[str],
        *,
        seed: int,
        min_neighbour: float,
        max_neighbour: float,
        modes: tuple[str, ...] = ("domino_h", "domino_v", "block2"),
        deterministic: bool = False,
    ) -> None:
        self.names = list(names)
        self.seed = int(seed)
        self.minimum = float(min_neighbour)
        self.maximum = float(max_neighbour)
        self.modes = tuple(modes)
        self.deterministic = bool(deterministic)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        entropy = 0 if self.deterministic else int(np.random.randint(2**31))
        rng = np.random.default_rng(self.seed + index * 1_000_003 + entropy)
        clean = to_frags(load(os.path.join(TRAIN_TGT, self.names[index])))
        dirty = distort_frags(clean, rng)

        # Random key order matches the arbitrary input tile IDs used at inference.
        key_to_true = rng.permutation(NFRAG).astype(np.int64)
        true_to_key = np.argsort(key_to_true)
        quality = float(rng.uniform(self.minimum, self.maximum))
        draft_true = component_proposal(rng, quality, str(rng.choice(self.modes)))
        draft_key = true_to_key[draft_true]

        def tensor(value: np.ndarray) -> Tensor:
            return torch.from_numpy(np.ascontiguousarray(value)).permute(0, 3, 1, 2).float().div(255.0)

        return {
            "dirty": tensor(dirty[key_to_true]),
            "clean": tensor(clean),
            "draft_key": torch.from_numpy(draft_key),
            "truth_place": torch.from_numpy(true_to_key),
            "draft_true": torch.from_numpy(draft_true),
        }


def _load_alignment(path: str, device: torch.device) -> PairedAlignment:
    payload = torch.load(path, map_location=device, weights_only=False)
    embed_dim = int(payload.get("embed_dim", 128))
    model = PairedAlignment(embed_dim).to(device)
    model.load_state_dict(payload["model"] if "model" in payload else payload, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _amp(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def _encode(
    alignment: PairedAlignment, dirty: Tensor, clean: Tensor, device: torch.device
) -> tuple[Tensor, Tensor]:
    batch = dirty.shape[0]
    with torch.no_grad(), _amp(device):
        dirty_embed = alignment.dirty_encoder(dirty.flatten(0, 1))
        clean_embed = alignment.clean_encoder(clean.flatten(0, 1))
    return (
        dirty_embed.float().reshape(batch, NFRAG, -1),
        clean_embed.float().reshape(batch, NFRAG, -1),
    )


def _hungarian(scores: np.ndarray) -> np.ndarray:
    rows, cols = linear_sum_assignment(-scores)
    place = np.empty(NFRAG, np.int64)
    place[rows] = cols
    return place


@torch.inference_mode()
def evaluate(
    model: ProposalRefiner,
    alignment: PairedAlignment,
    loader: DataLoader,
    device: torch.device,
    maximum_images: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {
        "draft_placement": [], "draft_neighbour": [],
        "refined_placement": [], "refined_neighbour": [],
        "oracle_placement": [], "oracle_neighbour": [],
        "query_r1": [], "query_r5": [],
    }
    seen = 0
    for batch in loader:
        dirty = batch["dirty"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        draft_key = batch["draft_key"].to(device, non_blocking=True).long()
        truth = batch["truth_place"].numpy()
        dirty_embed, clean_embed = _encode(alignment, dirty, clean, device)
        draft = torch.gather(
            dirty_embed, 1, draft_key.unsqueeze(-1).expand(-1, -1, dirty_embed.shape[-1])
        )
        with _amp(device):
            query = model(draft)
        scores = torch.bmm(query.float(), dirty_embed.transpose(1, 2))
        oracle_scores = torch.bmm(clean_embed, dirty_embed.transpose(1, 2))
        order = scores.argsort(dim=-1, descending=True)
        rank = (order == torch.from_numpy(truth).to(device).unsqueeze(-1)).nonzero()[:, -1] + 1
        totals["query_r1"].append(float(rank.le(1).float().mean()))
        totals["query_r5"].append(float(rank.le(5).float().mean()))
        for item in range(dirty.shape[0]):
            truth_place = truth[item].astype(np.int64)
            draft_place = draft_key[item].cpu().numpy().astype(np.int64)
            refined = _hungarian(scores[item].cpu().numpy())
            oracle = _hungarian(oracle_scores[item].cpu().numpy())
            for label, place in (("draft", draft_place), ("refined", refined), ("oracle", oracle)):
                totals[f"{label}_placement"].append(float(np.mean(place == truth_place)))
                totals[f"{label}_neighbour"].append(neighbour_accuracy(place, truth_place)[0])
            seen += 1
            if seen >= maximum_images:
                break
        if seen >= maximum_images:
            break
    model.train()
    return {key: float(np.mean(value)) for key, value in totals.items()}


def _save(path: Path, model: ProposalRefiner, step: int, args: argparse.Namespace, metrics: dict[str, float]) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "model_kwargs": {
                "embed_dim": model.embed_dim, "hidden": model.hidden,
                "layers": model.layers, "heads": model.heads,
            },
            "step": int(step),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def smoke() -> None:
    rng = np.random.default_rng(123)
    truth = np.arange(NFRAG, dtype=np.int64)
    for mode in ("domino_h", "domino_v", "block2"):
        proposal = component_proposal(rng, 0.17, mode)
        score = neighbour_accuracy(proposal, truth)[0]
        if not 0.14 <= score <= 0.20:
            raise AssertionError(f"{mode} generated unexpected neighbour={score:.4f}")
    model = ProposalRefiner(32, 48, 2, 4)
    output = model(torch.randn(1, NFRAG, 32))
    if output.shape != (1, NFRAG, 32) or not torch.isfinite(output).all():
        raise AssertionError("model smoke failed")
    print("smoke ok", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-images", type=int, default=8)
    parser.add_argument("--min-neighbour", type=float, default=0.10)
    parser.add_argument("--max-neighbour", type=float, default=0.25)
    parser.add_argument(
        "--modes", default="domino_h,domino_v,block2",
        help="training proposal modes: swap,domino_h,domino_v,block2",
    )
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--clean-weight", type=float, default=2.0)
    parser.add_argument("--alignment", default=str(Path(CKPT_DIR) / "paired_alignment_best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path(CKPT_DIR) / "proposal_refiner")
    parser.add_argument("--report", type=Path, default=Path("E:/pazzle_work/gates/proposal_refiner_gate.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--init", default=None, help="optional ProposalRefiner checkpoint")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if args.steps < 1 or args.bs < 1 or args.eval_images < 1:
        parser.error("--steps, --bs and --eval-images must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    alignment = _load_alignment(args.alignment, device)
    model = ProposalRefiner(
        alignment.dirty_encoder.embed_dim, args.hidden, args.layers, args.heads
    ).to(device)
    if args.init:
        initial = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(initial["model"] if "model" in initial else initial, strict=True)
        print(f"loaded refiner init {args.init}", flush=True)
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    allowed_modes = {"swap", "domino_h", "domino_v", "block2"}
    if not modes or any(value not in allowed_modes for value in modes):
        parser.error(f"--modes must be a comma-separated subset of {sorted(allowed_modes)}")
    train_names, val_names = train_val_split()
    train_loader = DataLoader(
        RefinerDataset(
            train_names, seed=args.seed, min_neighbour=args.min_neighbour,
            max_neighbour=args.max_neighbour, modes=modes,
        ),
        batch_size=args.bs, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        RefinerDataset(
            val_names[: args.eval_images], seed=args.seed + 90_000,
            min_neighbour=0.16, max_neighbour=0.18, deterministic=True,
        ),
        batch_size=1, shuffle=False, num_workers=min(args.workers, 1),
        pin_memory=device.type == "cuda",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.08
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    iterator = iter(train_loader)
    best = -float("inf")
    best_metrics: dict[str, float] = {}
    started = time.time()
    print(
        f"device={device} params={count_parameters(model):,} "
        f"train={len(train_names)} val={args.eval_images}",
        flush=True,
    )
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        dirty = batch["dirty"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        draft_key = batch["draft_key"].to(device, non_blocking=True).long()
        truth_place = batch["truth_place"].to(device, non_blocking=True).long()
        dirty_embed, clean_embed = _encode(alignment, dirty, clean, device)
        draft = torch.gather(
            dirty_embed, 1, draft_key.unsqueeze(-1).expand(-1, -1, dirty_embed.shape[-1])
        )
        with _amp(device):
            query = model(draft)
            logits = torch.bmm(query, dirty_embed.transpose(1, 2)) * alignment.scale().detach()
            ce = F.cross_entropy(logits.flatten(0, 1), truth_place.flatten())
            clean_loss = (1.0 - (query * clean_embed).sum(dim=-1)).mean()
            loss = args.ce_weight * ce + args.clean_weight * clean_loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step == 1 or step % 25 == 0:
            elapsed = time.time() - started
            print(
                f"step {step}/{args.steps} loss={float(loss):.4f} ce={float(ce):.4f} "
                f"clean={float(clean_loss):.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                f"{elapsed / step:.2f}s/it",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, alignment, val_loader, device, args.eval_images)
            score = metrics["refined_neighbour"] + metrics["refined_placement"]
            print("[VAL] " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items()), flush=True)
            _save(args.out_dir / "last.pt", model, step, args, metrics)
            if score > best:
                best, best_metrics = score, metrics
                _save(args.out_dir / "best.pt", model, step, args, metrics)
                print(f"saved best score={best:.4f}", flush=True)

    gate = {
        "experiment": "proposal_conditioned_permutation_refiner",
        "checkpoint": str(args.out_dir / "best.pt"),
        "metrics": best_metrics,
        "thresholds": {"neighbour_delta": 0.05, "placement_delta": 0.03},
    }
    gate["delta"] = {
        "placement": best_metrics["refined_placement"] - best_metrics["draft_placement"],
        "neighbour": best_metrics["refined_neighbour"] - best_metrics["draft_neighbour"],
    }
    gate["passed"] = (
        gate["delta"]["placement"] >= gate["thresholds"]["placement_delta"]
        and gate["delta"]["neighbour"] >= gate["thresholds"]["neighbour_delta"]
    )
    args.report.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
