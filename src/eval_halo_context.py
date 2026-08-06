"""Stage 2 oracle gate for the factorized 2D-hole scorer (branch E).

Stage 1 (``eval_paired_alignment.py``) proved the *retrieval endpoint* is not
the bottleneck: a dirty tile's content still identifies its own clean
counterpart among ~576 same-photo candidates with R@1 ~76%, even after one
independent application of the challenge degradation. That isolates the
remaining open question cleanly: does a *purely clean* 2D neighbourhood
determine the specific missing centre tile's identity well enough to serve
as a query?

This gate never touches corruption or dirty tiles.  Training data is clean
target tiles at their true, known grid positions (no permutation cache
needed -- ``to_frags`` is already row-major).  A frozen stage-1
``clean_encoder`` supplies tile keys; only the spatial-context transformer is
trained.  This isolates exactly what Frontier Pointer (branch B) could not:
that experiment jointly trained a weak dirty representation *and* 2D
reasoning with one listwise CE and failed at context 4/8 (R@1 2.3%/2.2%).
If this clean-only version also fails at similar context sizes, the
diagnosis is that local photographic content itself does not determine tile
identity (repetitive skies/walls/textures) rather than that the encoder was
weak. If it succeeds, stage 3 combines it with the stage-1 dirty encoder to
rank real dirty candidates from predicted-clean context.

Examples
--------

    python src/eval_halo_context.py --smoke
    python src/eval_halo_context.py --steps 1500 --bs 4 --queries 24 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import CKPT_DIR, FS, GRID, NFRAG, SEED, TRAIN_TGT
from eval_paired_alignment import PairedAlignment, TileEncoder, rank_of_diagonal, retrieval_summary
from imgio import load, to_frags, train_val_split


WINDOW = 5
SLOTS = WINDOW * WINDOW
CENTER = SLOTS // 2
_MARGIN = WINDOW // 2
CONTEXT_SIZES: tuple[int, ...] = (2, 4, 8)


def _offsets() -> Tensor:
    """Relative (row,col) offset for every window slot, row-major, centre excluded."""
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(-_MARGIN, _MARGIN + 1), torch.arange(-_MARGIN, _MARGIN + 1), indexing="ij"
        ),
        dim=-1,
    ).reshape(SLOTS, 2)
    return grid


_OFFSETS = _offsets()


def load_frozen_clean_encoder(path: str, device: torch.device) -> tuple[TileEncoder, int]:
    """Load stage 1's clean_encoder only, frozen, from a paired-alignment checkpoint."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"paired-alignment checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    embed_dim = int(payload["embed_dim"])
    full = PairedAlignment(embed_dim=embed_dim)
    full.load_state_dict(payload["model"], strict=True)
    encoder = full.clean_encoder
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, embed_dim


class HaloContextModel(nn.Module):
    """Predict a masked centre tile's clean-embedding-space query from 2D context."""

    def __init__(self, d: int, *, layers: int = 3, heads: int = 5) -> None:
        super().__init__()
        if d < 20 or d % heads:
            raise ValueError("d must be at least 20 and divisible by heads")
        self.d = int(d)
        self.mask_token = nn.Parameter(torch.empty(self.d))
        self.relative_embedding = nn.Embedding(SLOTS, self.d)
        encoder_layer = nn.TransformerEncoderLayer(
            self.d, heads, dim_feedforward=4 * self.d, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, layers, norm=nn.LayerNorm(self.d))
        self.query_head = nn.Sequential(
            nn.LayerNorm(self.d), nn.Linear(self.d, self.d), nn.GELU(), nn.Linear(self.d, self.d)
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.relative_embedding.weight, std=0.02)

    def scale(self) -> Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def forward(self, keys: Tensor, context_indices: Tensor, occupied: Tensor) -> Tensor:
        """``keys``: (B,576,d) clean tile embeddings. Returns normalized (B,Q,d) queries."""
        if keys.ndim != 3 or keys.shape[-1] != self.d:
            raise ValueError(f"keys must be (B,576,{self.d}), got {tuple(keys.shape)}")
        if context_indices.ndim != 3 or context_indices.shape[-1] != SLOTS:
            raise ValueError(f"context_indices must be (B,Q,{SLOTS})")
        if occupied.shape != context_indices.shape or occupied.dtype != torch.bool:
            raise ValueError("occupied must be bool with the same shape as context_indices")
        if torch.any(context_indices[..., CENTER] != -1):
            raise ValueError(f"context centre slot {CENTER} must always be -1")
        batch, queries = context_indices.shape[:2]
        safe = context_indices.clamp(min=0)
        batch_index = torch.arange(batch, device=keys.device)[:, None, None]
        gathered = keys[batch_index, safe]
        visible = occupied & context_indices.ge(0)
        tokens = torch.where(visible.unsqueeze(-1), gathered, self.mask_token.view(1, 1, 1, self.d))
        tokens = tokens + self.relative_embedding.weight.view(1, 1, SLOTS, self.d)
        encoded = self.context_encoder(tokens.reshape(batch * queries, SLOTS, self.d))
        centre = encoded[:, CENTER].reshape(batch, queries, self.d)
        return F.normalize(self.query_head(centre), dim=-1)


def build_queries(
    *, batch: int, queries_per_image: int, context_sizes: tuple[int, ...], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample (centre_tile_id, context_indices, occupied, context_size) for one batch.

    Centres are restricted to interior cells so the full 5x5 window stays in
    bounds; this keeps the gate focused on the context-sufficiency question
    without adding boundary bookkeeping.
    """
    interior = GRID - 2 * _MARGIN
    if interior < 1:
        raise ValueError("GRID too small for the configured window")
    centre_ids = np.empty((batch, queries_per_image), dtype=np.int64)
    context_indices = np.full((batch, queries_per_image, SLOTS), -1, dtype=np.int64)
    occupied = np.zeros((batch, queries_per_image, SLOTS), dtype=bool)
    context_size_used = np.empty((batch, queries_per_image), dtype=np.int64)
    non_centre_slots = np.array([slot for slot in range(SLOTS) if slot != CENTER])
    for image in range(batch):
        rows = rng.integers(_MARGIN, _MARGIN + interior, size=queries_per_image)
        cols = rng.integers(_MARGIN, _MARGIN + interior, size=queries_per_image)
        centre_ids[image] = rows * GRID + cols
        for query in range(queries_per_image):
            size = int(rng.choice(context_sizes))
            context_size_used[image, query] = size
            chosen_slots = rng.choice(non_centre_slots, size=size, replace=False)
            offsets = _OFFSETS.numpy()[chosen_slots]
            neighbour_rows = rows[query] + offsets[:, 0]
            neighbour_cols = cols[query] + offsets[:, 1]
            tile_ids = neighbour_rows * GRID + neighbour_cols
            context_indices[image, query, chosen_slots] = tile_ids
            occupied[image, query, chosen_slots] = True
    return centre_ids, context_indices, occupied, context_size_used


@torch.inference_mode()
def _encode_image_clean(
    encoder: TileEncoder, name: str, device: torch.device
) -> Tensor:
    clean = to_frags(load(os.path.join(TRAIN_TGT, name)))
    tiles = (
        torch.from_numpy(np.ascontiguousarray(clean)).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    )
    return encoder(tiles)


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def listwise_loss_and_logits(
    query: Tensor, keys: Tensor, context_indices: Tensor, centre_ids: Tensor, scale: Tensor
) -> tuple[Tensor, Tensor]:
    """Rank every one of the 576 same-image tiles; context tiles are excluded.

    Empty slots use ``-1``; clamping them to a valid gather index before a
    scatter would make every empty slot alias tile 0, and ``scatter_``'s
    behaviour with duplicate destination indices is undefined.  A one-hot
    sum sidesteps that: ``any`` over slots is commutative, so duplicate
    (invalid) slots aliasing index 0 never corrupt a genuine exclusion.
    """
    batch, tile_count, _ = keys.shape
    logits = torch.einsum("bqd,bnd->bqn", query, keys) * scale
    valid_context = context_indices.ge(0)
    one_hot = F.one_hot(context_indices.clamp(min=0), num_classes=tile_count).to(torch.bool)
    excluded = (one_hot & valid_context.unsqueeze(-1)).any(dim=2)
    logits = logits.masked_fill(excluded, -torch.inf)
    flat_logits = logits.reshape(-1, tile_count)
    flat_target = centre_ids.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_target)
    return loss, logits


@torch.inference_mode()
def evaluate(
    model: HaloContextModel,
    clean_encoder: TileEncoder,
    names: list[str],
    *,
    queries_per_image: int,
    context_sizes: tuple[int, ...],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    per_size_ranks: dict[int, list[Tensor]] = {size: [] for size in context_sizes}
    for index, name in enumerate(names):
        keys = _encode_image_clean(clean_encoder, name, device).unsqueeze(0)
        centre_ids, context_indices, occupied, sizes_used = build_queries(
            batch=1, queries_per_image=queries_per_image, context_sizes=context_sizes,
            rng=np.random.default_rng(seed + index * 7919),
        )
        centre_t = torch.from_numpy(centre_ids).to(device)
        context_t = torch.from_numpy(context_indices).to(device)
        occupied_t = torch.from_numpy(occupied).to(device)
        with _autocast(device):
            query = model(keys, context_t, occupied_t)
        _, logits = listwise_loss_and_logits(query.float(), keys.float(), context_t, centre_t, model.scale())
        true_score = logits.gather(2, centre_t.unsqueeze(-1)).squeeze(-1)
        rank = logits.gt(true_score.unsqueeze(-1)).sum(dim=-1) + 1
        for size in context_sizes:
            mask = torch.from_numpy(sizes_used[0] == size).to(device)
            if bool(mask.any()):
                per_size_ranks[size].append(rank[0][mask])
    if was_training:
        model.train()
    result: dict[str, Any] = {}
    for size in context_sizes:
        ranks = torch.cat(per_size_ranks[size]) if per_size_ranks[size] else torch.empty(0)
        if ranks.numel():
            result[f"context{size}"] = retrieval_summary(ranks, f"context{size}")
        else:
            result[f"context{size}"] = {"note": "no queries sampled at this context size"}
    return result


def _checkpoint(path: str, model: HaloContextModel, *, step: int, metrics: dict[str, Any]) -> None:
    torch.save({"model": model.state_dict(), "d": model.d, "step": int(step), "metrics": metrics}, path)


def smoke(device: torch.device = torch.device("cpu")) -> dict[str, float]:
    """Data-free contract test: shapes, masking, gradients, and query-slot invariant."""
    torch.manual_seed(99)
    d = 20
    model = HaloContextModel(d=d, layers=1, heads=4).to(device)
    keys = F.normalize(torch.randn(2, NFRAG, d, device=device, requires_grad=True), dim=-1)
    keys.retain_grad()  # keys is a non-leaf (post-normalize) tensor; retain to inspect its grad
    rng = np.random.default_rng(7)
    centre_ids, context_indices, occupied, sizes = build_queries(
        batch=2, queries_per_image=6, context_sizes=CONTEXT_SIZES, rng=rng
    )
    if context_indices.shape != (2, 6, SLOTS):
        raise AssertionError(f"unexpected context_indices shape {context_indices.shape}")
    if np.any(sizes < min(CONTEXT_SIZES)) or np.any(sizes > max(CONTEXT_SIZES)):
        raise AssertionError("sampled context sizes fell outside the configured set")
    if np.any(context_indices[..., CENTER] != -1):
        raise AssertionError("centre slot leaked a context tile id")
    context_t = torch.from_numpy(context_indices).to(device)
    occupied_t = torch.from_numpy(occupied).to(device)
    centre_t = torch.from_numpy(centre_ids).to(device)
    query = model(keys, context_t, occupied_t)
    if query.shape != (2, 6, d):
        raise AssertionError(f"unexpected query shape {query.shape}")
    if not torch.allclose(query.norm(dim=-1), torch.ones(2, 6, device=device), atol=1.0e-5):
        raise AssertionError("query embeddings are not L2-normalized")
    loss, logits = listwise_loss_and_logits(query, keys, context_t, centre_t, model.scale())
    if not torch.isfinite(loss):
        raise AssertionError("listwise loss is not finite")
    for image in range(2):
        for row in range(6):
            occupied_ids = context_indices[image, row][occupied[image, row]]
            for tile_id in occupied_ids:
                if torch.isfinite(logits[image, row, tile_id]):
                    raise AssertionError("an occupied context tile leaked into the candidate logits")
            if not torch.isfinite(logits[image, row, centre_ids[image, row]]):
                raise AssertionError("the true centre tile was masked out of its own candidate row")
    loss.backward()
    if keys.grad is None or not torch.isfinite(keys.grad).all():
        raise AssertionError("listwise loss lost a finite gradient into the frozen keys")
    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()):
        raise AssertionError("HaloContextModel parameters received no gradient")
    return {"loss": float(loss.detach()), "parameters": sum(p.numel() for p in model.parameters())}


def _parse_args() -> argparse.Namespace:
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-encoder-ckpt", "--clean_encoder_ckpt", dest="clean_encoder_ckpt",
        default=os.path.join(CKPT_DIR, "paired_alignment_best.pt"),
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--bs", type=int, default=4, help="images per optimizer step")
    parser.add_argument("--queries", type=int, default=24, help="queries per image per step")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8, help="must divide the stage-1 embed_dim (default 128)")
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int, default=300)
    parser.add_argument("--eval-images", "--eval_images", dest="eval_images", type=int, default=8)
    parser.add_argument("--eval-queries", "--eval_queries", dest="eval_queries", type=int, default=192)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="halo_context")
    parser.add_argument("--ckpt-dir", "--ckpt_dir", dest="ckpt_dir", default=CKPT_DIR)
    parser.add_argument(
        "--report", type=Path, default=Path("E:/pazzle_work/gates/halo_context_gate.json")
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        return args
    if args.steps < 1 or args.bs < 1 or args.queries < 1 or args.eval_images < 1 or args.eval_queries < 1:
        parser.error("--steps, --bs, --queries, --eval-images, and --eval-queries must be positive")
    if args.layers < 1 or args.heads < 1 or args.lr <= 0.0 or args.eval_every < 1:
        parser.error("invalid --layers/--heads/--lr/--eval-every")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.smoke:
        print(f"[halo-context smoke] device={device} {smoke(device)}", flush=True)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    clean_encoder, embed_dim = load_frozen_clean_encoder(args.clean_encoder_ckpt, device)
    train_names, val_names = train_val_split()
    if len(val_names) < args.eval_images:
        raise ValueError(f"--eval-images exceeds the held-out pool ({len(val_names)})")

    model = HaloContextModel(d=embed_dim, layers=args.layers, heads=args.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device} HaloContextModel params={sum(p.numel() for p in model.parameters()):,} "
        f"(frozen clean_encoder from {args.clean_encoder_ckpt}, embed_dim={embed_dim}) "
        f"bs={args.bs} queries/image={args.queries} context_sizes={CONTEXT_SIZES} window={WINDOW}",
        flush=True,
    )
    print(
        "question: does a purely clean 2x24/etc. neighbourhood identify the specific masked "
        "centre tile among all 576 same-photo candidates -- with NO corruption anywhere?",
        flush=True,
    )

    rng = np.random.default_rng(args.seed + 3)
    best_gate_metric = -float("inf")
    started = time.time()
    for step in range(1, args.steps + 1):
        names = [train_names[int(i)] for i in rng.integers(0, len(train_names), size=args.bs)]
        keys_list = [_encode_image_clean(clean_encoder, name, device) for name in names]
        keys = torch.stack(keys_list, dim=0)
        centre_ids, context_indices, occupied, _sizes = build_queries(
            batch=args.bs, queries_per_image=args.queries, context_sizes=CONTEXT_SIZES,
            rng=np.random.default_rng(rng.integers(0, 2**31 - 1)),
        )
        centre_t = torch.from_numpy(centre_ids).to(device)
        context_t = torch.from_numpy(context_indices).to(device)
        occupied_t = torch.from_numpy(occupied).to(device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            query = model(keys, context_t, occupied_t)
            loss, _ = listwise_loss_and_logits(query.float(), keys.float(), context_t, centre_t, model.scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step == 1 or step % 25 == 0:
            elapsed = time.time() - started
            print(
                f"step {step}/{args.steps} loss={float(loss.detach()):.4f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} {elapsed / step:.2f}s/it",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model, clean_encoder, val_names[: args.eval_images],
                queries_per_image=args.eval_queries, context_sizes=CONTEXT_SIZES,
                device=device, seed=args.seed + 9973,
            )
            print(f"[SYN halo-context held-out] step={step} {metrics}", flush=True)
            last_path = os.path.join(args.ckpt_dir, f"{args.tag}_last.pt")
            _checkpoint(last_path, model, step=step, metrics=metrics)
            gate_metric = metrics.get("context4", {}).get(f"context4_r1", -1.0)
            if gate_metric > best_gate_metric:
                best_gate_metric = gate_metric
                best_path = os.path.join(args.ckpt_dir, f"{args.tag}_best.pt")
                _checkpoint(best_path, model, step=step, metrics=metrics)
                print(f"saved best context4_r1={best_gate_metric:.4f}", flush=True)

    final_metrics = evaluate(
        model, clean_encoder, val_names[: args.eval_images],
        queries_per_image=args.eval_queries, context_sizes=CONTEXT_SIZES,
        device=device, seed=args.seed + 9973,
    )
    context4 = final_metrics.get("context4", {})
    context8 = final_metrics.get("context8", {})
    passed = (
        context4.get("context4_r1", 0.0) >= 0.30
        and context4.get("context4_r5", 0.0) >= 0.60
        and context8.get("context8_r1", 0.0) >= 0.50
    )
    report = {
        "experiment": "stage2_clean_halo_context_center_tile_identity",
        "question": (
            "does a purely clean 2D neighbourhood (no corruption anywhere) determine the "
            "specific missing centre tile's identity among ~576 same-photo candidates?"
        ),
        "config": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "device": str(device)},
        "final_metrics": final_metrics,
        "gate": {
            "rule": "context4 R@1>=0.30 AND context4 R@5>=0.60 AND context8 R@1>=0.50",
            "chance_r1": 1.0 / NFRAG,
            "chance_r5": 5.0 / NFRAG,
            "pass": bool(passed),
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    verdict = "PASSED -> proceed to stage 3 (combine with stage-1 dirty encoder)" if passed else "FAILED -> close branch E"
    print(f"\n=== stage 2 gate {verdict} ===", flush=True)
    print(json.dumps(report["gate"], indent=2), flush=True)
    print(f"report saved to {args.report}", flush=True)


if __name__ == "__main__":
    main()
