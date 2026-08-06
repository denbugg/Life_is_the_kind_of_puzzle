"""Evaluate PlaquetteNet as a 2x2 candidate re-ranker on synthetic puzzles.

Each held-out clean target is independently corrupted tile-by-tile and shuffled.
For every input fragment ``a`` we construct candidate blocks in this order:

``a --R--> b`` and ``a --D--> c``, followed by the best ``d`` under
``D[b, d] + R[c, d]``.  This deliberately measures two separate things:

* whether the full PairwiseNet candidate pool contains the real 2x2 block; and
* conditional on that event, whether PlaquetteNet ranks it better than the
  sum of the four pairwise seams.

The latter is *not* a full puzzle solve.  It is a compact integration gate for
the proposed local 2x2 scorer.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

import pipeline
from config import GRID, NFRAG, SEED, TRAIN_TGT
from distort import distort_frags
from imgio import load, to_frags, train_val_split
from plaquette import PlaquetteNet
from solve import pairwise_scores_full


def _top_k(scores: np.ndarray, k: int, excluded: tuple[int, ...]) -> np.ndarray:
    """Return indices of the best non-excluded scores, in descending order."""
    allowed = np.ones(scores.shape[0], dtype=bool)
    allowed[list(excluded)] = False
    idx = np.flatnonzero(allowed)
    if not len(idx):
        return np.empty(0, dtype=np.int64)
    order = np.argsort(-scores[idx], kind="stable")
    return idx[order[: min(k, len(idx))]].astype(np.int64, copy=False)


def generate_anchor_candidates(
    R: np.ndarray,
    D: np.ndarray,
    anchor: int,
    k: int,
    diag_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate distinct ``(a,b,c,d)`` candidate blocks for one anchor.

    Tile order is always ``(top-left, top-right, bottom-left, bottom-right)``.
    The returned seam score is the four-edge pairwise baseline used for the
    comparison with PlaquetteNet.
    """
    right = _top_k(R[anchor], k, (anchor,))
    down = _top_k(D[anchor], k, (anchor,))
    blocks: list[tuple[int, int, int, int]] = []
    pair_sums: list[float] = []

    for b in right:
        for c in down:
            # A real 2x2 block cannot reuse its top-right tile as bottom-left.
            if b == c:
                continue
            diagonal = D[b] + R[c]
            for d in _top_k(diagonal, diag_k, (anchor, int(b), int(c))):
                blocks.append((anchor, int(b), int(c), int(d)))
                pair_sums.append(float(R[anchor, b] + D[anchor, c] + D[b, d] + R[c, d]))

    if not blocks:
        return np.empty((0, 4), dtype=np.int64), np.empty(0, dtype=np.float32)
    return np.asarray(blocks, dtype=np.int64), np.asarray(pair_sums, dtype=np.float32)


def _inverse_permutation(perm: np.ndarray) -> np.ndarray:
    """Map clean-grid cell -> shuffled input-fragment index."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm), dtype=perm.dtype)
    return inv


def true_block_for_anchor(perm: np.ndarray, inverse: np.ndarray, anchor: int) -> np.ndarray | None:
    """Return the true block whose top-left input tile is ``anchor``, if any.

    ``perm[input_index]`` is the original row-major target cell.  Anchors from
    the final row or column have no complete 2x2 block below/right of them.
    """
    cell = int(perm[anchor])
    row, col = divmod(cell, GRID)
    if row >= GRID - 1 or col >= GRID - 1:
        return None
    return np.array(
        (anchor, inverse[cell + 1], inverse[cell + GRID], inverse[cell + GRID + 1]),
        dtype=np.int64,
    )


def build_candidate_pool(
    R: np.ndarray,
    D: np.ndarray,
    perm: np.ndarray,
    k: int,
    diag_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], int]:
    """Build all anchor pools and locate generated true blocks.

    Returns ``(blocks, pair_sums, offsets, true_global_indices, valid_count)``.
    ``offsets[a]:offsets[a + 1]`` is the candidate range for anchor ``a``.
    ``true_global_indices`` contains only anchors for which the real block was
    generated; this is intentionally the denominator for re-ranking metrics.
    """
    if R.shape != (NFRAG, NFRAG) or D.shape != (NFRAG, NFRAG):
        raise ValueError(f"expected R/D shape {(NFRAG, NFRAG)}, got {R.shape} and {D.shape}")
    if perm.shape != (NFRAG,):
        raise ValueError(f"expected perm shape {(NFRAG,)}, got {perm.shape}")

    inverse = _inverse_permutation(perm)
    all_blocks: list[np.ndarray] = []
    all_pair_sums: list[np.ndarray] = []
    offsets = np.zeros(NFRAG + 1, dtype=np.int64)
    true_global_indices: list[int] = []
    valid_count = 0

    for anchor in range(NFRAG):
        blocks, pair_sums = generate_anchor_candidates(R, D, anchor, k, diag_k)
        start = int(offsets[anchor])
        offsets[anchor + 1] = start + len(blocks)
        all_blocks.append(blocks)
        all_pair_sums.append(pair_sums)

        truth = true_block_for_anchor(perm, inverse, anchor)
        if truth is None:
            continue
        valid_count += 1
        match = np.flatnonzero(np.all(blocks == truth[None, :], axis=1))
        if len(match):
            # The construction has one unique path for every (a,b,c,d).  Keep
            # the assertion close to the metric in case that ever changes.
            if len(match) != 1:
                raise RuntimeError(f"duplicate true candidate for anchor {anchor}")
            true_global_indices.append(start + int(match[0]))

    nonempty_blocks = [x for x in all_blocks if len(x)]
    nonempty_scores = [x for x in all_pair_sums if len(x)]
    flat_blocks = (
        np.concatenate(nonempty_blocks, axis=0)
        if nonempty_blocks
        else np.empty((0, 4), dtype=np.int64)
    )
    flat_pair_sums = (
        np.concatenate(nonempty_scores, axis=0)
        if nonempty_scores
        else np.empty(0, dtype=np.float32)
    )
    return flat_blocks, flat_pair_sums, offsets, true_global_indices, valid_count


@torch.inference_mode()
def score_plaquette_candidates(
    model: torch.nn.Module,
    frags: np.ndarray,
    blocks: np.ndarray,
    device: torch.device,
    bs: int,
) -> np.ndarray:
    """Run PlaquetteNet over all proposed blocks in bounded GPU batches."""
    if blocks.ndim != 2 or blocks.shape[1:] != (4,):
        raise ValueError(f"expected blocks shaped (M,4), got {blocks.shape}")
    if not len(blocks):
        return np.empty(0, dtype=np.float32)

    x = torch.from_numpy(np.ascontiguousarray(frags)).permute(0, 3, 1, 2).float().div_(255).to(device)
    out = np.empty(len(blocks), dtype=np.float32)
    amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    model.eval()
    for start in range(0, len(blocks), bs):
        stop = min(start + bs, len(blocks))
        idx = torch.as_tensor(blocks[start:stop], device=device)
        with amp_context:
            logits = model(x[idx])
        logits = torch.as_tensor(logits).reshape(-1)
        if logits.numel() != stop - start:
            raise ValueError(
                f"PlaquetteNet returned {logits.numel()} logits for a batch of {stop - start} candidates"
            )
        out[start:stop] = logits.float().cpu().numpy()
    return out


def _r_at_k(scores: np.ndarray, true_index: int, k: int) -> bool:
    """Whether a local candidate index is in a deterministic top-k ranking."""
    k = min(k, len(scores))
    if k <= 0:
        return False
    order = np.argsort(-scores, kind="stable")
    return bool(np.any(order[:k] == true_index))


def _load_plaquette(path: str, device: torch.device) -> tuple[PlaquetteNet, dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Plaquette checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch builds
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported Plaquette checkpoint payload: {type(payload)!r}")
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise TypeError("Plaquette checkpoint has no state dict under 'model'")

    # Width is inferable from the checkpoint, so an experiment with a wider
    # scorer still evaluates without a script edit.  Dropout has no eval effect.
    stem = state.get("stem.0.weight")
    width = int(stem.shape[0]) if torch.is_tensor(stem) else 48
    model = PlaquetteNet(width=width).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, payload


def _validate_args(args: argparse.Namespace) -> None:
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if not (1 <= args.K < NFRAG):
        raise ValueError(f"--K must be in [1, {NFRAG - 1}]")
    if not (1 <= args.diag_k < NFRAG):
        raise ValueError(f"--diag_k must be in [1, {NFRAG - 1}]")
    if args.bs <= 0:
        raise ValueError("--bs must be positive")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10, help="number of held-out targets")
    ap.add_argument("--K", type=int, default=8, help="top right/down candidates per anchor")
    ap.add_argument("--diag_k", type=int, default=4, help="top bottom-right candidates per (b,c)")
    ap.add_argument("--bs", type=int, default=1024, help="PlaquetteNet candidate batch size")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ckpt", default="artifacts/plaquette/plaquette_v1_best.pt")
    args = ap.parse_args()
    _validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pipeline.load_pair historically uses its module-level DEV for checkpoint
    # loading.  Set it once so this evaluation also remains usable on a CPU-only
    # machine; pairwise_scores_full still remains the sole pairwise scorer.
    pipeline.DEV = str(device)
    pair, pair_ckpt = pipeline.load_pair()
    if pair is None:
        raise FileNotFoundError("no PairwiseNet checkpoint found via pipeline.load_pair()")
    plaquette, plaquette_ckpt = _load_plaquette(args.ckpt, device)

    _, val_names = train_val_split()
    names = val_names[: args.n]
    if not names:
        raise RuntimeError("held-out target split is empty")
    print(
        f"device={device} pair_step={pair_ckpt.get('step') if pair_ckpt else None} "
        f"plaquette_step={plaquette_ckpt.get('step')} images={len(names)} "
        f"K={args.K} diag_k={args.diag_k}",
        flush=True,
    )

    totals = {
        "anchors": 0,
        "valid": 0,
        "candidates": 0,
        "generated_true": 0,
        "plaq_r1": 0,
        "plaq_r5": 0,
        "pair_r1": 0,
        "pair_r5": 0,
    }
    started = time.time()
    rng = np.random.default_rng(args.seed)

    for image_index, name in enumerate(names, start=1):
        clean = load(os.path.join(TRAIN_TGT, name))
        # perm maps shuffled input index -> original clean grid cell.  The
        # inverse is retained only inside build_candidate_pool for exact labels.
        perm = rng.permutation(NFRAG).astype(np.int64)
        frags = distort_frags(to_frags(clean), rng)[perm]

        R, D = pairwise_scores_full(pair, frags, device=str(device), bs=args.bs)
        blocks, pair_sums, offsets, true_global, valid_count = build_candidate_pool(
            R, D, perm, args.K, args.diag_k
        )
        plaquette_scores = score_plaquette_candidates(plaquette, frags, blocks, device, args.bs)

        image_hits = {"plaq_r1": 0, "plaq_r5": 0, "pair_r1": 0, "pair_r5": 0}
        for global_true in true_global:
            # Every block begins with its anchor, so use that to recover the
            # candidate interval and make the ranking local to that anchor.
            anchor = int(blocks[global_true, 0])
            start, stop = int(offsets[anchor]), int(offsets[anchor + 1])
            local_true = global_true - start
            plaq_local = plaquette_scores[start:stop]
            pair_local = pair_sums[start:stop]
            image_hits["plaq_r1"] += _r_at_k(plaq_local, local_true, 1)
            image_hits["plaq_r5"] += _r_at_k(plaq_local, local_true, 5)
            image_hits["pair_r1"] += _r_at_k(pair_local, local_true, 1)
            image_hits["pair_r5"] += _r_at_k(pair_local, local_true, 5)

        totals["anchors"] += NFRAG
        totals["valid"] += valid_count
        totals["candidates"] += len(blocks)
        totals["generated_true"] += len(true_global)
        for key, value in image_hits.items():
            totals[key] += int(value)

        recall = len(true_global) / valid_count if valid_count else float("nan")
        denom = max(1, len(true_global))
        print(
            f"[{image_index:>3}/{len(names)}] {name} candidates={len(blocks):,} "
            f"true={len(true_global)}/{valid_count} recall={recall:.3f} "
            f"plaq(R@1/R@5)={image_hits['plaq_r1']/denom:.3f}/{image_hits['plaq_r5']/denom:.3f} "
            f"pair(R@1/R@5)={image_hits['pair_r1']/denom:.3f}/{image_hits['pair_r5']/denom:.3f}",
            flush=True,
        )

    rankable = totals["generated_true"]
    print(f"\n== Plaquette candidate evaluation: held-out synthetic, N={len(names)} ==")
    print(f"anchors_total          {totals['anchors']:,}")
    print(f"valid_true_blocks      {totals['valid']:,}")
    print(f"generated_candidates   {totals['candidates']:,}")
    print(f"candidates/anchor      {totals['candidates'] / max(1, totals['anchors']):.1f}")
    print(f"true_blocks_generated  {rankable:,}")
    print(f"candidate_recall       {rankable / max(1, totals['valid']):.4f}")
    print(f"rankable_anchors       {rankable:,}")
    print(f"plaquette R@1          {totals['plaq_r1'] / max(1, rankable):.4f}")
    print(f"plaquette R@5          {totals['plaq_r5'] / max(1, rankable):.4f}")
    print(f"pair_sum R@1           {totals['pair_r1'] / max(1, rankable):.4f}")
    print(f"pair_sum R@5           {totals['pair_r5'] / max(1, rankable):.4f}")
    print(f"seconds/image          {(time.time() - started) / len(names):.2f}")


if __name__ == "__main__":
    main()
