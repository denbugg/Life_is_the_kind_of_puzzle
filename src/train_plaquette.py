"""Train a 2x2 plaquette classifier on synthetically corrupted target tiles.

The positive class is a true 2x2 patch from a clean training target, after each
of its four 20x20 tiles has been independently corrupted.  Negatives are built
from the *same* source image.  They deliberately retain one or two real seams:

* replace one corner, leaving the opposite L-shaped pair of true seams intact;
* swap two adjacent pieces, leaving one true seam intact; or
* replace a row/column with a true adjacent pair from elsewhere in the image.

This makes the classifier learn four-way plaquette consistency rather than just
detecting an obviously bad single seam.

Run from the repository root, for example:

    python src/train_plaquette.py --steps 12000 --bs 64 --workers 6 --tag plaquette

``PlaquetteNet`` is imported lazily so ``--help`` remains available while the
model module is being developed.  The preferred interface receives the four
explicit tiles as ``(B, 4, 3, 20, 20)`` in TL/TR/BL/BR order and returns logits.
For a minimally different model that expects an already stitched 40x40 block,
the call wrapper falls back to ``(B, 3, 40, 40)``.
"""

from __future__ import annotations

import argparse
import inspect
import os
import random
import time
from contextlib import nullcontext
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import FS, GRID, SEED, TRAIN_TGT
from distort import distort_frags
from imgio import load, to_frags, train_val_split


def _idx(row: int, col: int) -> int:
    return row * GRID + col


def _block_indices(row: int, col: int) -> list[int]:
    """Return [top-left, top-right, bottom-left, bottom-right]."""
    tl = _idx(row, col)
    return [tl, tl + 1, tl + GRID, tl + GRID + 1]


def _ring_candidates(row: int, col: int, forbidden: set[int]) -> list[int]:
    """Tiles immediately surrounding a 2x2 block, excluding its own tiles."""
    out: list[int] = []
    for rr in range(max(0, row - 1), min(GRID, row + 3)):
        for cc in range(max(0, col - 1), min(GRID, col + 3)):
            j = _idx(rr, cc)
            if j not in forbidden:
                out.append(j)
    return out


def _sample_external_pair(
    rng: np.random.Generator, forbidden: set[int], horizontal: bool
) -> tuple[int, int] | None:
    """Sample a real adjacent pair not touching the source plaquette."""
    for _ in range(64):
        if horizontal:
            rr = int(rng.integers(GRID))
            cc = int(rng.integers(GRID - 1))
            pair = (_idx(rr, cc), _idx(rr, cc + 1))
        else:
            rr = int(rng.integers(GRID - 1))
            cc = int(rng.integers(GRID))
            pair = (_idx(rr, cc), _idx(rr + 1, cc))
        if pair[0] not in forbidden and pair[1] not in forbidden:
            return pair
    return None


def hard_negative_indices(row: int, col: int, rng: np.random.Generator) -> list[int]:
    """Make a difficult non-plaquette from the same source image.

    Every branch changes one or two pieces but intentionally leaves at least one
    true local seam in the displayed 2x2 block.  This avoids a trivial training
    distribution of four unrelated tiles.
    """
    base = _block_indices(row, col)
    forbidden = set(base)
    mode = int(rng.choice(3, p=(0.45, 0.30, 0.25)))

    if mode == 0:
        # One wrong corner: the two seams opposite that corner remain genuine.
        out = base.copy()
        slot = int(rng.integers(4))
        candidates = _ring_candidates(row, col, forbidden)
        if not candidates:
            candidates = [j for j in range(GRID * GRID) if j not in forbidden]
        out[slot] = int(candidates[int(rng.integers(len(candidates)))])
        return out

    if mode == 1:
        # Adjacent swap: e.g. swap a column, retaining the other column seam.
        out = base.copy()
        pairs = ((0, 1), (0, 2), (1, 3), (2, 3))
        a, b = pairs[int(rng.integers(len(pairs)))]
        out[a], out[b] = out[b], out[a]
        return out

    # Replace an entire row/column by a coherent pair from elsewhere.  The
    # imported pair has a true seam of its own, and the untouched row/column
    # retains another, so this is harder than two independent replacements.
    out = base.copy()
    horizontal = bool(rng.integers(2))
    pair = _sample_external_pair(rng, forbidden, horizontal=horizontal)
    if pair is None:
        # Extremely defensive fallback for malformed geometry.
        out[0], out[1] = out[1], out[0]
        return out
    if horizontal:
        if bool(rng.integers(2)):
            out[0], out[1] = pair  # leave original bottom seam
        else:
            out[2], out[3] = pair  # leave original top seam
    else:
        if bool(rng.integers(2)):
            out[0], out[2] = pair  # leave original right seam
        else:
            out[1], out[3] = pair  # leave original left seam
    return out


class PlaquetteDataset(Dataset):
    """Balanced true / hard-negative 2x2 blocks from a list of clean targets."""

    def __init__(self, names: Iterable[str], training: bool, seed: int = SEED):
        self.names = list(names)
        if not self.names:
            raise ValueError("PlaquetteDataset received no image names")
        self.training = bool(training)
        self.seed = int(seed)

    def __len__(self) -> int:
        # Each image contributes one deterministic positive and one negative per
        # validation pass; shuffled training naturally remains class-balanced.
        return 2 * len(self.names)

    def _rng(self, index: int) -> np.random.Generator:
        if not self.training:
            # Stable validation corruption, patch and negative construction.
            return np.random.default_rng(self.seed + 1_000_003 * int(index))
        # NumPy is seeded per worker in _seed_worker.  A fresh seed makes a
        # repeated image index see a new distortion and different hard negative.
        return np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_index = (int(index) // 2) % len(self.names)
        positive = (int(index) % 2) == 0
        rng = self._rng(index)

        clean = to_frags(load(os.path.join(TRAIN_TGT, self.names[image_index])))
        row = int(rng.integers(GRID - 1))
        col = int(rng.integers(GRID - 1))
        source = _block_indices(row, col)
        order = source if positive else hard_negative_indices(row, col, rng)

        # Only corrupt the four tiles used by this example, rather than all 576.
        # They are still independently corrupted by distort_frags(), exactly as
        # in the challenge, and this keeps full-holdout validation inexpensive.
        unique, inverse = np.unique(np.asarray(order, dtype=np.int64), return_inverse=True)
        corrupted = distort_frags(clean[unique], rng)
        # Explicit TL/TR/BL/BR tiles; PlaquetteNet stitches its 40x40 block
        # internally, so it can retain the shared centre-junction evidence.
        tiles = corrupted[inverse]  # (4,20,20,3)
        x = torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2).float().div_(255.0)
        y = torch.tensor(1.0 if positive else 0.0, dtype=torch.float32)
        return x, y


def _seed_worker(worker_id: int) -> None:
    # torch.initial_seed() is distinct for each DataLoader worker.
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _make_loader(
    dataset: Dataset, batch_size: int, workers: int, shuffle: bool, drop_last: bool
) -> DataLoader:
    kw: dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=_seed_worker,
    )
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kw)


def _build_model() -> torch.nn.Module:
    """Import/instantiate PlaquetteNet while tolerating tiny API variations."""
    try:
        from plaquette import PlaquetteNet  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - gives a useful runtime error
        raise RuntimeError(
            "Could not import PlaquetteNet from src/plaquette.py. "
            "Expected PlaquetteNet(block) -> binary logits."
        ) from exc

    # The intended API is PlaquetteNet().  The two fallbacks make the trainer
    # usable if the model chooses an explicit standard input-channel argument.
    attempts = ({}, {"in_channels": 3}, {"in_ch": 3})
    errors: list[str] = []
    for kwargs in attempts:
        try:
            return PlaquetteNet(**kwargs)
        except TypeError as exc:
            errors.append(f"PlaquetteNet({kwargs}): {exc}")
    try:
        sig = inspect.signature(PlaquetteNet)
    except (TypeError, ValueError):
        sig = None
    hint = f" Constructor signature: {sig}." if sig is not None else ""
    raise TypeError("Could not instantiate PlaquetteNet. " + " | ".join(errors) + hint)


def binary_logits(output: Any, batch_size: int) -> torch.Tensor:
    """Normalize common binary-model outputs to a (B,) positive-class logit."""
    if isinstance(output, dict):
        for key in ("logits", "logit", "score", "scores"):
            if key in output:
                output = output[key]
                break
        else:
            raise TypeError(f"PlaquetteNet returned dict without logits keys: {tuple(output)}")
    elif isinstance(output, (tuple, list)):
        if not output:
            raise TypeError("PlaquetteNet returned an empty tuple/list")
        output = output[0]
    elif hasattr(output, "logits"):
        output = output.logits

    if not torch.is_tensor(output):
        raise TypeError(f"PlaquetteNet output must be a tensor, got {type(output)!r}")
    if output.ndim == 0:
        if batch_size != 1:
            raise ValueError("scalar PlaquetteNet output is only valid for batch_size=1")
        return output.reshape(1)
    if output.ndim == 1 and output.shape[0] == batch_size:
        return output
    if output.ndim == 2 and output.shape == (batch_size, 1):
        return output[:, 0]
    if output.ndim == 2 and output.shape == (batch_size, 2):
        # Convert two-class logits to an equivalent binary log-odds score.
        return output[:, 1] - output[:, 0]
    raise ValueError(
        "Expected PlaquetteNet output shaped (B,), (B,1), or (B,2); "
        f"got {tuple(output.shape)} for B={batch_size}"
    )


def _stitch_tiles(tiles: torch.Tensor) -> torch.Tensor:
    """Convert explicit TL/TR/BL/BR tiles to a conventional 40x40 tensor."""
    if tiles.ndim != 5 or tuple(tiles.shape[1:]) != (4, 3, FS, FS):
        raise ValueError(f"expected (B,4,3,{FS},{FS}) tiles, got {tuple(tiles.shape)}")
    tl, tr, bl, br = tiles.unbind(dim=1)
    top = torch.cat((tl, tr), dim=-1)
    bottom = torch.cat((bl, br), dim=-1)
    return torch.cat((top, bottom), dim=-2)


def forward_logits(model: torch.nn.Module, tiles: torch.Tensor) -> torch.Tensor:
    """Call the current four-tile API, then a stitched-block fallback if needed."""
    try:
        output = model(tiles)
    except (ValueError, AssertionError, RuntimeError) as tile_error:
        # Older/minimally different implementations may accept (B,3,40,40).
        # Keep the original exception available if that call fails too.
        try:
            output = model(_stitch_tiles(tiles))
        except Exception as stitched_error:
            raise RuntimeError(
                "PlaquetteNet rejected both explicit tiles (B,4,3,20,20) and "
                f"a stitched block (B,3,40,40). Explicit-tile error: {tile_error}"
            ) from stitched_error
    return binary_logits(output, tiles.shape[0])


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Dependency-free ROC AUC with average ranks for tied scores."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        # Ranks are one-indexed; all tied values receive their mean rank.
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _autocast(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    for batch_index, (x, y) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with _autocast(amp):
            logits = forward_logits(model, x)
            loss = F.binary_cross_entropy_with_logits(logits, y, reduction="sum")
        loss_sum += float(loss.item())
        count += int(y.numel())
        scores.append(logits.float().cpu().numpy())
        labels.append(y.cpu().numpy())

    if count == 0:
        raise RuntimeError("validation loader yielded no batches")
    s = np.concatenate(scores)
    y = np.concatenate(labels).astype(np.int64)
    return {
        "loss": loss_sum / count,
        "auc": _roc_auc(y, s),
        "acc": float(((s >= 0.0) == (y == 1)).mean()),
        "n": float(count),
    }


def _state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return (model.module if hasattr(model, "module") else model).state_dict()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA but CUDA is unavailable")
    return device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=12_000)
    ap.add_argument("--bs", type=int, default=64, help="number of 40x40 blocks per step")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="plaquette")
    ap.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "plaquette"),
        help="checkpoint directory (workspace-local by default)",
    )
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_n", type=int, default=0, help="0=all held-out images; otherwise first N")
    ap.add_argument("--val_batches", type=int, default=0, help="0=all validation batches")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    if args.steps <= 0 or args.bs <= 0:
        raise ValueError("--steps and --bs must be positive")
    if args.workers < 0 or args.val_every <= 0:
        raise ValueError("--workers must be >= 0 and --val_every must be positive")
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = _resolve_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    train_names, val_names = train_val_split()
    if args.val_n > 0:
        val_names = val_names[: args.val_n]

    train_ds = PlaquetteDataset(train_names, training=True, seed=args.seed)
    val_ds = PlaquetteDataset(val_names, training=False, seed=args.seed + 10_000)
    train_loader = _make_loader(train_ds, args.bs, args.workers, shuffle=True, drop_last=True)
    val_loader = _make_loader(val_ds, args.bs, args.workers, shuffle=False, drop_last=False)

    model = _build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"PlaquetteNet params: {n_params:,}; device={device}; amp={amp}; "
        f"train_images={len(train_names)} val_images={len(val_names)}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):  # older torch fallback
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_auc = -float("inf")
    saved_checkpoint = False
    ema_loss: float | None = None
    ema_acc: float | None = None
    step = 0
    started = time.time()

    while step < args.steps:
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with _autocast(amp):
                logits = forward_logits(model, x)
                loss = F.binary_cross_entropy_with_logits(logits, y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            acc = float(((logits.detach() >= 0.0) == (y >= 0.5)).float().mean().item())
            ema_loss = float(loss.item()) if ema_loss is None else 0.98 * ema_loss + 0.02 * float(loss.item())
            ema_acc = acc if ema_acc is None else 0.98 * ema_acc + 0.02 * acc

            if step % args.log_every == 0:
                sec_per_step = (time.time() - started) / max(1, step)
                print(
                    f"step {step}/{args.steps} loss {ema_loss:.4f} acc {ema_acc:.3f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {sec_per_step:.2f}s/it",
                    flush=True,
                )

            do_val = step > 0 and ((step % args.val_every == 0) or (step + 1 == args.steps))
            if do_val:
                metrics = evaluate(model, val_loader, device, amp, args.val_batches)
                print(
                    f"  [VAL] loss={metrics['loss']:.4f} auc={metrics['auc']:.4f} "
                    f"acc={metrics['acc']:.4f} n={int(metrics['n'])}",
                    flush=True,
                )
                payload = {
                    "model": _state_dict(model),
                    "step": step,
                    "val_auc": metrics["auc"],
                    "val_acc": metrics["acc"],
                    "val_loss": metrics["loss"],
                    "args": vars(args),
                }
                torch.save(payload, os.path.join(args.out_dir, f"{args.tag}_last.pt"))
                saved_checkpoint = True
                if np.isfinite(metrics["auc"]) and metrics["auc"] > best_auc:
                    best_auc = metrics["auc"]
                    torch.save(payload, os.path.join(args.out_dir, f"{args.tag}_best.pt"))
                    print(f"  saved best AUC={best_auc:.4f}", flush=True)
                model.train()

            step += 1
            if step >= args.steps:
                break

    # Ensure a usable last checkpoint even for very short smoke runs where no
    # scheduled validation occurred.
    last_path = os.path.join(args.out_dir, f"{args.tag}_last.pt")
    if not saved_checkpoint:
        metrics = evaluate(model, val_loader, device, amp, args.val_batches)
        payload = {
            "model": _state_dict(model),
            "step": step,
            "val_auc": metrics["auc"],
            "val_acc": metrics["acc"],
            "val_loss": metrics["loss"],
            "args": vars(args),
        }
        torch.save(payload, last_path)
        if np.isfinite(metrics["auc"]):
            best_auc = metrics["auc"]
            torch.save(payload, os.path.join(args.out_dir, f"{args.tag}_best.pt"))
        print(
            f"  [FINAL VAL] loss={metrics['loss']:.4f} auc={metrics['auc']:.4f} "
            f"acc={metrics['acc']:.4f}",
            flush=True,
        )
    print(f"done. best val AUC={best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
