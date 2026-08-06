"""Train the oracle component-anchored dense all-bag pointer gate.

Every example is freshly distorted from a clean training target and shuffled.
The trainer uses the exact synthetic permutation only to select a valid,
oracle-correct A -> B -> C chain and to obtain the optional clean low-frequency
code for C.  The model receives only the noisy unordered bag, A/B identities,
and a physical rotation operation that canonicalizes the chain direction.  It
never receives a recovered real permutation, an absolute coordinate, an input
tile position, or an affinity/seam candidate list.

The held-out metric is all-bag retrieval: C is ranked among all 574 unused
tiles, not among a prefiltered graph.  This makes the script a strict oracle
information gate before attempting to grow high-confidence components.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from canvas_data import CanvasDataset
from config import FS, GRID, NFRAG, SEED
from dense_pointer import (
    NUM_DIRECTIONS,
    DensePointerNet,
    count_params,
    smoke as model_smoke,
)
from imgio import train_val_split


_DELTA_ROWS = (-1, 1, 0, 0)
_DELTA_COLS = (0, 0, -1, 1)


@dataclass(frozen=True)
class OracleRows:
    """Synthetic supervision for one oracle A -> B -> C chain per bag."""

    anchor_indices: Tensor
    middle_indices: Tensor
    target_indices: Tensor
    directions: Tensor
    target_code: Tensor


def _autocast(device: torch.device, *, enabled: bool):
    """Use optional fp16 autocast; full fp32 is the numerical-safe default."""
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if enabled and device.type == "cuda"
        else nullcontext()
    )


def _make_scaler(*, enabled: bool):
    """Create a GradScaler only for explicit CUDA AMP runs.

    The default all-bag pointer path is fp32.  Keeping ``None`` rather than a
    disabled scaler makes it impossible to accidentally infer that an optimizer
    step happened when GradScaler actually skipped it.
    """
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        # Compatibility with older PyTorch builds used in this workspace.
        return torch.cuda.amp.GradScaler(enabled=True)


def _nonfinite_gradient_names(model: nn.Module, *, limit: int = 6) -> list[str]:
    """Return a short list of non-finite gradient locations, if any."""
    bad: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            bad.append(name)
            if len(bad) >= limit:
                break
    return bad


def _global_grad_norm(model: nn.Module) -> Tensor:
    """Return an fp32 global gradient norm after finite-value validation."""
    squared: Tensor | None = None
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        contribution = parameter.grad.detach().float().square().sum()
        squared = contribution if squared is None else squared + contribution
    if squared is None:
        raise FloatingPointError("backward produced no gradients")
    return squared.sqrt()


def _make_loader(
    dataset: CanvasDataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Build a DataLoader without persistent-worker edge cases at workers=0."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if workers < 0:
        raise ValueError("workers must be non-negative")
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": shuffle and len(dataset) >= batch_size,
    }
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


def _next_batch(
    iterator: Iterable[dict[str, Tensor]], loader: DataLoader
) -> tuple[dict[str, Tensor], Iterable[dict[str, Tensor]]]:
    """Cycle a loader forever, preserving newly randomized synthetic exposure."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def sample_oracle_rows(perm: Tensor, target_patches: Tensor) -> OracleRows:
    """Create oracle A -> B -> C rows from exact synthetic permutation labels.

    B is sampled away from the clean-grid boundary in both axes, then a uniform
    cardinal direction determines A=B-delta and C=B+delta.  This makes all
    four orientations equally valid without exposing any coordinate to the
    model.  ``perm[input_tile] == clean_cell``; its inverse maps the selected
    clean cells back to shuffled bag identities.
    """
    if perm.ndim != 2 or perm.shape[1] != NFRAG:
        raise ValueError(f"perm must have shape (B,{NFRAG}), got {tuple(perm.shape)}")
    if target_patches.ndim != 5 or target_patches.shape[:2] != perm.shape:
        raise ValueError(
            "target_patches must have shape (B,576,patch,patch,3) matching perm, "
            f"got {tuple(target_patches.shape)}"
        )
    if perm.device != target_patches.device:
        raise ValueError("perm and target_patches must be on the same device")
    if torch.any(perm < 0) or torch.any(perm >= NFRAG):
        raise ValueError("perm contains a clean cell outside the puzzle grid")

    batch = perm.shape[0]
    device = perm.device
    # B lies in [1, GRID-2] in both axes, so A and C remain valid for every
    # cardinal direction.  Coordinates live only in this label sampler.
    middle_rows = torch.randint(1, GRID - 1, (batch,), device=device)
    middle_cols = torch.randint(1, GRID - 1, (batch,), device=device)
    directions = torch.randint(NUM_DIRECTIONS, (batch,), device=device)
    delta_rows = torch.tensor(_DELTA_ROWS, device=device, dtype=torch.long)[directions]
    delta_cols = torch.tensor(_DELTA_COLS, device=device, dtype=torch.long)[directions]

    anchor_cells = (middle_rows - delta_rows) * GRID + (middle_cols - delta_cols)
    middle_cells = middle_rows * GRID + middle_cols
    target_cells = (middle_rows + delta_rows) * GRID + (middle_cols + delta_cols)

    inverse = torch.argsort(perm.long(), dim=1)
    anchor_indices = inverse.gather(1, anchor_cells[:, None]).squeeze(1)
    middle_indices = inverse.gather(1, middle_cells[:, None]).squeeze(1)
    target_indices = inverse.gather(1, target_cells[:, None]).squeeze(1)
    if torch.any(anchor_indices.eq(middle_indices)) or torch.any(
        anchor_indices.eq(target_indices)
    ) or torch.any(middle_indices.eq(target_indices)):
        raise AssertionError("a synthetic A/B/C chain must contain three distinct tiles")

    batch_index = torch.arange(batch, device=device)
    # CanvasDataset keeps target patches in clean row-major order.  This target
    # is supervision only; the clean patch never enters DensePointerNet.
    target_code = target_patches[batch_index, target_cells].reshape(batch, -1).float()
    return OracleRows(
        anchor_indices=anchor_indices,
        middle_indices=middle_indices,
        target_indices=target_indices,
        directions=directions,
        target_code=target_code,
    )


def pointer_metrics(
    logits: Tensor,
    target_indices: Tensor,
    *,
    code: Tensor | None = None,
    target_code: Tensor | None = None,
) -> dict[str, float]:
    """Calculate exact all-bag retrieval metrics for a batch of pointer rows."""
    if logits.ndim != 2 or target_indices.ndim != 1 or logits.shape[0] != target_indices.shape[0]:
        raise ValueError("logits must be (B,N) and target_indices must be (B,)")
    if torch.any(target_indices < 0) or torch.any(target_indices >= logits.shape[1]):
        raise ValueError("target_indices contain an out-of-range candidate")
    target_logits = logits.gather(1, target_indices[:, None])
    if not torch.isfinite(target_logits).all():
        raise AssertionError("the true C tile was masked from dense pointer logits")
    ranks = logits.gt(target_logits).sum(dim=1).add(1).float()
    topk = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
    result = {
        "pointer_ce": float(F.cross_entropy(logits, target_indices).detach().cpu()),
        "r1": float(topk[:, :1].eq(target_indices[:, None]).any(dim=1).float().mean().cpu()),
        "r5": float(topk.eq(target_indices[:, None]).any(dim=1).float().mean().cpu()),
        "mrr": float(ranks.reciprocal().mean().cpu()),
        "mean_rank": float(ranks.mean().cpu()),
    }
    if code is not None or target_code is not None:
        if code is None or target_code is None:
            raise ValueError("code and target_code must either both be supplied or both be omitted")
        if code.shape != target_code.shape:
            raise ValueError(f"code shape {tuple(code.shape)} != target code {tuple(target_code.shape)}")
        result["code_l1"] = float(F.l1_loss(code, target_code).detach().cpu())
    return result


@torch.no_grad()
def evaluate(
    model: DensePointerNet,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    max_images: int,
    queries_per_image: int,
) -> dict[str, float]:
    """Evaluate C retrieval among every unused tile of fresh held-out bags."""
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    if queries_per_image <= 0:
        raise ValueError("queries_per_image must be positive")
    model.eval()
    totals: dict[str, float] = {}
    rows_seen = 0
    images_seen = 0
    for batch in loader:
        if images_seen >= max_images:
            break
        take = min(max_images - images_seen, int(batch["tiles"].shape[0]))
        if not bool(batch["has_perm"][:take].all()):
            raise RuntimeError("dense-pointer evaluation requires exact synthetic examples")
        tiles = batch["tiles"][:take].to(device, non_blocking=True)
        perm = batch["perm"][:take].to(device, non_blocking=True).long()
        target_patches = batch["target_patches"][:take].to(device, non_blocking=True)
        for _ in range(queries_per_image):
            rows = sample_oracle_rows(perm, target_patches)
            with _autocast(device, enabled=amp_enabled):
                output = model(
                    tiles,
                    rows.anchor_indices,
                    rows.middle_indices,
                    rows.directions,
                )
            stats = pointer_metrics(
                output["logits"].float(),
                rows.target_indices,
                code=output["code"].float(),
                target_code=rows.target_code.float(),
            )
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value * take
            rows_seen += take
        images_seen += take
    if rows_seen == 0:
        raise RuntimeError("evaluation saw no held-out oracle rows")
    metrics = {key: value / rows_seen for key, value in totals.items()}
    metrics["images"] = float(images_seen)
    metrics["rows"] = float(rows_seen)
    # Baseline is exactly uniform over NFRAG-2 unused tiles for every row.
    metrics["random_r1"] = 1.0 / float(NFRAG - 2)
    metrics["random_r5"] = min(5, NFRAG - 2) / float(NFRAG - 2)
    return metrics


def _checkpoint(
    path: str,
    model: DensePointerNet,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    *,
    step: int,
    args: argparse.Namespace,
    metrics: Mapping[str, float],
) -> None:
    """Write a self-describing checkpoint with the no-leakage data contract."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "experiment": "dense_component_pointer_oracle_gate",
            "model": model.state_dict(),
            "model_kwargs": {
                "tile_size": model.tile_size,
                "width": model.width,
                "embedding_dim": model.embedding_dim,
                "code_patch": model.code_patch,
                "dropout": model.dropout,
                "temperature": model.initial_temperature,
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "args": vars(args),
            "metrics": dict(metrics),
            "data_contract": {
                "source": "fresh CanvasDataset(real_prob=0) synthetic shuffle/distortion",
                "oracle_seed_pair": "exact synthetic A->B only; no recovered real labels",
                "query_target": "next clean-grid cell C in the same direction after B",
                "candidate_set": "all 576 bag tiles with only A and B masked",
                "affinity_candidates": False,
                "seam_cross_encoder": False,
                "absolute_coordinates_input_to_model": False,
                "input_position_features": False,
                "direction_input": "physical bag rotation to canonical A->B left-to-right only",
                "clean_target_code": "training/evaluation supervision only, never model input",
            },
        },
        path,
    )


def _parse_args() -> argparse.Namespace:
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument("--bs", type=int, default=1, help="full puzzle bags per optimizer step")
    parser.add_argument("--eval-bs", type=int, default=1, help="full puzzle bags per evaluation forward")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-n", type=int, default=0, help="0 uses all non-held-out images")
    parser.add_argument("--val-n", type=int, default=48, help="held-out images per evaluation")
    parser.add_argument("--eval-queries", type=int, default=3, help="oracle chains sampled per held-out bag")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--code-patch", type=int, default=4)
    parser.add_argument("--code-weight", type=float, default=0.10)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="dense_pointer_v1")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(workspace, "artifacts", "dense_pointer"),
        help="checkpoint directory inside the workspace by default",
    )
    parser.add_argument("--device", default="", help="defaults to CUDA when available")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="opt in to CUDA fp16 AMP; disabled by default because fp32 is safer for full-bag pointers",
    )
    parser.add_argument("--tiny-smoke", action="store_true", help="CPU-only architecture/label smoke; no data")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_tiny_smoke() -> None:
    """Verify model masking/backprop plus exact synthetic row construction."""
    _set_seed(SEED)
    perm = torch.randperm(NFRAG).reshape(1, NFRAG)
    target_patches = torch.rand(1, NFRAG, 4, 4, 3)
    rows = sample_oracle_rows(perm, target_patches)
    if rows.target_code.shape != (1, 4 * 4 * 3):
        raise AssertionError(f"unexpected target-code shape {tuple(rows.target_code.shape)}")
    identities = torch.stack((rows.anchor_indices, rows.middle_indices, rows.target_indices), dim=1)
    if torch.unique(identities).numel() != 3:
        raise AssertionError("oracle sampler returned duplicate A/B/C identities")
    clean_a = perm.gather(1, rows.anchor_indices[:, None]).squeeze(1)
    clean_b = perm.gather(1, rows.middle_indices[:, None]).squeeze(1)
    clean_c = perm.gather(1, rows.target_indices[:, None]).squeeze(1)
    delta_cells = torch.tensor((-GRID, GRID, -1, 1), dtype=torch.long)[rows.directions]
    if not torch.equal(clean_b - clean_a, delta_cells) or not torch.equal(clean_c - clean_b, delta_cells):
        raise AssertionError("oracle sampler did not preserve the requested A->B->C direction")
    result = model_smoke(device="cpu")
    print(
        "tiny smoke passed "
        f"rows=(A={rows.anchor_indices.item()}, B={rows.middle_indices.item()}, C={rows.target_indices.item()}) "
        f"direction={int(rows.directions.item())} model={result}",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    if args.tiny_smoke:
        _run_tiny_smoke()
        return
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.eval_every <= 0 or args.log_every <= 0:
        raise ValueError("eval-every and log-every must be positive")
    if args.val_n <= 0 or args.eval_queries <= 0:
        raise ValueError("val-n and eval-queries must be positive")
    if args.code_weight < 0.0:
        raise ValueError("code-weight must be non-negative")
    if FS % args.code_patch:
        raise ValueError(f"code-patch ({args.code_patch}) must divide tile size {FS}")

    _set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp_enabled = bool(args.amp and device.type == "cuda")
    if args.amp and device.type != "cuda":
        print("--amp ignored because the selected device is not CUDA", flush=True)

    train_names, validation_names = train_val_split()
    if args.train_n:
        train_names = train_names[: args.train_n]
    validation_names = validation_names[: args.val_n]
    if not train_names or not validation_names:
        raise RuntimeError("empty train or held-out split")
    train_dataset = CanvasDataset(
        train_names,
        patch=args.code_patch,
        real_prob=0.0,
        seed=args.seed,
    )
    validation_dataset = CanvasDataset(
        validation_names,
        patch=args.code_patch,
        real_prob=0.0,
        seed=args.seed + 100_000,
    )
    train_loader = _make_loader(train_dataset, args.bs, args.workers, shuffle=True, device=device)
    validation_loader = _make_loader(
        validation_dataset, args.eval_bs, min(args.workers, 2), shuffle=False, device=device
    )

    model = DensePointerNet(
        tile_size=FS,
        width=args.width,
        embedding_dim=args.embedding_dim,
        code_patch=args.code_patch,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.steps, 1), eta_min=args.lr * 0.10
    )
    scaler = _make_scaler(enabled=amp_enabled)
    print(
        f"device={device} params={count_params(model):,} train_images={len(train_names)} "
        f"heldout_images={len(validation_names)} candidates_per_row={NFRAG - 2} "
        f"precision={'amp-fp16' if amp_enabled else 'fp32'}",
        flush=True,
    )

    iterator: Iterable[dict[str, Tensor]] = iter(train_loader)
    best = float("-inf")
    last_log = time.monotonic()
    running_loss = 0.0
    running_pointer = 0.0
    running_code = 0.0
    for step in range(1, args.steps + 1):
        batch, iterator = _next_batch(iterator, train_loader)
        if not bool(batch["has_perm"].all()):
            raise RuntimeError("dense-pointer training requires exact synthetic CanvasDataset examples")
        tiles = batch["tiles"].to(device, non_blocking=True)
        perm = batch["perm"].to(device, non_blocking=True).long()
        target_patches = batch["target_patches"].to(device, non_blocking=True)
        rows = sample_oracle_rows(perm, target_patches)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, enabled=amp_enabled):
            output = model(tiles, rows.anchor_indices, rows.middle_indices, rows.directions)
        # Keep both supervision terms in fp32 even when --amp is explicitly
        # requested.  The masked all-bag softmax is the numerically sensitive
        # operation, so loss arithmetic must never be autocast to fp16.
        logits = output["logits"].float()
        code = output["code"].float()
        target_code = rows.target_code.float()
        if not torch.isfinite(logits).all() or not torch.isfinite(code).all():
            raise FloatingPointError(f"non-finite model output before backward at step {step}")
        pointer_loss = F.cross_entropy(logits, rows.target_indices)
        code_loss = F.l1_loss(code, target_code)
        loss = pointer_loss + args.code_weight * code_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss before backward at step {step}: "
                f"pointer_ce={float(pointer_loss.detach().cpu())} code_l1={float(code_loss.detach().cpu())}"
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        bad_gradients = _nonfinite_gradient_names(model)
        if bad_gradients:
            optimizer.zero_grad(set_to_none=True)
            scale_text = f" scaler_scale={scaler.get_scale():.0f}" if scaler is not None else ""
            raise FloatingPointError(
                f"non-finite gradients at step {step} ({', '.join(bad_gradients)}){scale_text}; "
                "optimizer and scheduler were not stepped"
            )
        grad_norm = _global_grad_norm(model)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"non-finite global gradient norm at step {step}; optimizer and scheduler were not stepped"
            )
        if args.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Scheduler stepping is deliberately inside the verified-update branch.
        # With fp32 this is a direct optimizer step; with AMP GradScaler has
        # already unscaled and passed the explicit finite-gradient gate above.
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
            scaler_scale = float(scaler.get_scale())
        else:
            optimizer.step()
            scaler_scale = 1.0
        optimizer_updated = True
        if not optimizer_updated:  # Defensive guard for future optimizer modes.
            raise RuntimeError("optimizer update was skipped; scheduler will not advance")
        scheduler.step()

        running_loss += float(loss.detach().cpu())
        running_pointer += float(pointer_loss.detach().cpu())
        running_code += float(code_loss.detach().cpu())
        if step % args.log_every == 0 or step == 1:
            elapsed = time.monotonic() - last_log
            divisor = 1 if step == 1 else args.log_every
            print(
                f"step={step:05d}/{args.steps} loss={running_loss / divisor:.4f} "
                f"pointer_ce={running_pointer / divisor:.4f} code_l1={running_code / divisor:.4f} "
                f"grad_norm={float(grad_norm.detach().cpu()):.3g} updated={int(optimizer_updated)} "
                f"scaler_scale={scaler_scale:.0f} lr={optimizer.param_groups[0]['lr']:.2e} dt={elapsed:.1f}s",
                flush=True,
            )
            last_log = time.monotonic()
            running_loss = running_pointer = running_code = 0.0

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                validation_loader,
                device=device,
                amp_enabled=amp_enabled,
                max_images=args.val_n,
                queries_per_image=args.eval_queries,
            )
            metric_text = " ".join(
                f"{key}={value:.4f}" for key, value in metrics.items() if key not in {"images", "rows"}
            )
            print(
                f"eval step={step:05d} images={int(metrics['images'])} rows={int(metrics['rows'])} {metric_text}",
                flush=True,
            )
            last_path = os.path.join(args.out_dir, f"{args.tag}_last.pt")
            _checkpoint(last_path, model, optimizer, scheduler, step=step, args=args, metrics=metrics)
            if metrics["r1"] > best:
                best = metrics["r1"]
                best_path = os.path.join(args.out_dir, f"{args.tag}_best.pt")
                _checkpoint(best_path, model, optimizer, scheduler, step=step, args=args, metrics=metrics)
                print(f"saved best all_bag_r1={best:.4f}: {best_path}", flush=True)
            model.train()


if __name__ == "__main__":
    main()
