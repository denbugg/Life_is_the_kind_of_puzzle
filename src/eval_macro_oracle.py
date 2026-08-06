"""Oracle-macro evaluation for the existing PairwiseNet.

This is an intentionally favourable diagnostic for a hierarchical assembler.  A
held-out clean target is corrupted synthetically and shuffled just as in the
normal synthetic validation setup.  The evaluator then reveals *only* the
membership of each non-overlapping source ``side x side`` macro (``4 x 4`` by
default); it does not reveal the order of the fragments inside that macro.

Each macro is solved independently from all directed PairwiseNet scores within
its known member set.  Thus this answers a narrow but useful question: if a
coarse/global stage could supply the correct 4x4 groups, is the current local
edge scorer strong enough to arrange the pieces inside them?

The reported placement metric is strict local-cell accuracy.  The neighbour
metrics are directed (right/down) and are evaluated only on edges internal to a
macro.  Macro locations and orientation are fixed by the oracle, so reflected,
rotated, and translated reconstructions are counted as errors.
"""
from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from numba import njit

import pipeline
from config import GRID, NFRAG, SEED, TRAIN_TGT
from distort import distort_frags
from imgio import load, to_frags, train_val_split


@njit(cache=True, fastmath=True)
def _grid_objective(place: np.ndarray, R: np.ndarray, D: np.ndarray, side: int) -> float:
    """Directed right/down seam objective for an arbitrary square local grid."""
    total = 0.0
    for row in range(side):
        for col in range(side):
            p = row * side + col
            if col + 1 < side:
                total += R[place[p], place[p + 1]]
            if row + 1 < side:
                total += D[place[p], place[p + side]]
    return total


@njit(cache=True, fastmath=True)
def _swap_delta(
    place: np.ndarray,
    p: int,
    q: int,
    R: np.ndarray,
    D: np.ndarray,
    side: int,
) -> float:
    """Objective change caused by swapping two local grid positions.

    There are only ``2 * side * (side - 1)`` directed edges in a macro, so
    scanning these edges is simpler and less error-prone than a special-case
    neighbourhood bookkeeping routine.  Edges unrelated to the swap are
    skipped, making this still cheap for the intended 4x4 case.
    """
    old_total = 0.0
    new_total = 0.0
    fp = place[p]
    fq = place[q]
    for row in range(side):
        for col in range(side):
            u = row * side + col
            if col + 1 < side:
                v = u + 1
                if u == p or u == q or v == p or v == q:
                    a = place[u]
                    b = place[v]
                    an = fq if u == p else (fp if u == q else a)
                    bn = fq if v == p else (fp if v == q else b)
                    old_total += R[a, b]
                    new_total += R[an, bn]
            if row + 1 < side:
                v = u + side
                if u == p or u == q or v == p or v == q:
                    a = place[u]
                    b = place[v]
                    an = fq if u == p else (fp if u == q else a)
                    bn = fq if v == p else (fp if v == q else b)
                    old_total += D[a, b]
                    new_total += D[an, bn]
    return new_total - old_total


@njit(cache=True, fastmath=True)
def _greedy_start(R: np.ndarray, D: np.ndarray, side: int, first_fragment: int) -> np.ndarray:
    """Fill a local grid row-major after choosing its provisional top-left tile."""
    n = side * side
    place = -np.ones(n, dtype=np.int64)
    used = np.zeros(n, dtype=np.uint8)
    place[0] = first_fragment
    used[first_fragment] = 1

    for p in range(1, n):
        row = p // side
        col = p % side
        best_fragment = -1
        best_score = -1.0e30
        for fragment in range(n):
            if used[fragment]:
                continue
            score = 0.0
            if col > 0:
                score += R[place[p - 1], fragment]
            if row > 0:
                score += D[place[p - side], fragment]
            if score > best_score:
                best_score = score
                best_fragment = fragment
        place[p] = best_fragment
        used[best_fragment] = 1
    return place


@njit(cache=True)
def _random_start(n: int, seed: int) -> np.ndarray:
    """A Fisher--Yates start for restarts beyond the one-per-corner-tile sweep."""
    np.random.seed(seed)
    place = np.arange(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(i + 1)
        tmp = place[i]
        place[i] = place[j]
        place[j] = tmp
    return place


@njit(cache=True, fastmath=True)
def _anneal_once(
    initial: np.ndarray,
    R: np.ndarray,
    D: np.ndarray,
    side: int,
    iterations: int,
    temp_start: float,
    temp_end: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """One generic swap-SA run over a square local jigsaw."""
    np.random.seed(seed)
    n = side * side
    place = initial.copy()
    current = _grid_objective(place, R, D, side)
    best = place.copy()
    best_value = current
    ratio = temp_end / temp_start
    denom = max(1, iterations - 1)

    for iteration in range(iterations):
        p = np.random.randint(n)
        q = np.random.randint(n)
        if p == q:
            continue
        delta = _swap_delta(place, p, q, R, D, side)
        temperature = temp_start * ratio ** (iteration / denom)
        if delta > 0.0 or np.random.random() < np.exp(delta / temperature):
            tmp = place[p]
            place[p] = place[q]
            place[q] = tmp
            current += delta
            if current > best_value:
                best_value = current
                best = place.copy()
    return best, best_value


def solve_grid_sa(
    R: np.ndarray,
    D: np.ndarray,
    *,
    side: int = 4,
    iterations: int = 6_000,
    restarts: int = 16,
    temp_scale: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Multi-start SA solver for a square local group.

    ``R[a, b]`` and ``D[a, b]`` are the scores of fragment ``b`` immediately
    right/below fragment ``a``.  The first ``side**2`` restarts cover every
    possible provisional top-left fragment with a directional greedy start;
    additional restarts use random permutations.  It is deliberately local to
    this diagnostic rather than borrowing the 24x24 production solver.
    """
    n = side * side
    if R.shape != (n, n) or D.shape != (n, n):
        raise ValueError(f"expected {(n, n)} local score matrices, got {R.shape} and {D.shape}")
    if iterations <= 0 or restarts <= 0:
        raise ValueError("iterations and restarts must be positive")
    if temp_scale <= 0:
        raise ValueError("temp_scale must be positive")

    R = np.ascontiguousarray(R, dtype=np.float32)
    D = np.ascontiguousarray(D, dtype=np.float32)
    score_std = 0.5 * (float(R.std()) + float(D.std()))
    temp_start = max(1.0e-3, score_std * temp_scale)
    temp_end = temp_start * 0.01
    best_place: np.ndarray | None = None
    best_value = -np.inf

    # numba's RandomState uses a signed 32-bit seed.  Keep all arithmetic in
    # that safe range while retaining reproducible independent macro runs.
    seed_base = int(seed) & 0x7FFFFFFF
    for restart in range(restarts):
        run_seed = (seed_base + 7919 * (restart + 1)) & 0x7FFFFFFF
        if restart < n:
            initial = _greedy_start(R, D, side, (seed_base + restart) % n)
        else:
            initial = _random_start(n, run_seed)
        place, value = _anneal_once(
            initial, R, D, side, iterations, temp_start, temp_end, run_seed
        )
        if value > best_value:
            best_place = place
            best_value = float(value)

    if best_place is None:  # defensive: validation above guarantees a restart
        raise RuntimeError("local simulated annealing produced no candidate")
    return best_place, best_value


def make_oracle_macro_groups(
    perm: np.ndarray,
    *,
    side: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unordered oracle macro memberships and their hidden local labels.

    ``perm[input_fragment]`` is the source target-cell index for that shuffled
    input.  The returned group ordering is randomized independently inside
    every macro: this prevents the construction itself from leaking the target
    cell through a local array index.
    """
    if perm.ndim != 1 or len(perm) != NFRAG:
        raise ValueError(f"expected a permutation of {NFRAG} fragments, got {perm.shape}")
    if GRID % side:
        raise ValueError(f"side={side} must divide global grid side {GRID}")
    n_local = side * side
    n_macros = (GRID // side) ** 2
    source_to_input = np.empty(NFRAG, dtype=np.int64)
    source_to_input[perm] = np.arange(NFRAG, dtype=np.int64)
    groups = np.empty((n_macros, n_local), dtype=np.int64)
    truths = np.empty((n_macros, n_local), dtype=np.int64)

    macro = 0
    for row0 in range(0, GRID, side):
        for col0 in range(0, GRID, side):
            cells = np.array(
                [(row0 + row) * GRID + col0 + col for row in range(side) for col in range(side)],
                dtype=np.int64,
            )
            # Membership is known, but the order passed into PairwiseNet and SA
            # must remain arbitrary.  `cells` is source ordered only long
            # enough to construct the evaluation label below.
            member_inputs = source_to_input[cells]
            member_inputs = member_inputs[rng.permutation(n_local)]
            groups[macro] = member_inputs

            source_to_local = {int(source_cell): local for local, source_cell in enumerate(perm[member_inputs])}
            truths[macro] = np.array([source_to_local[int(cell)] for cell in cells], dtype=np.int64)
            macro += 1

    return groups, truths


@torch.inference_mode()
def score_groups_all_pairs(
    pair_models: torch.nn.Module | list[torch.nn.Module] | tuple[torch.nn.Module, ...],
    frags: np.ndarray,
    group_inputs: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every ordered pair within every known macro in bounded GPU batches.

    Group-to-group comparisons are intentionally absent: oracle membership is
    the premise being tested.  All horizontal and vertical candidates across
    all macros are packed into each batch, avoiding hundreds of tiny model
    calls.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if group_inputs.ndim != 2:
        raise ValueError(f"expected macro members shaped (M,K), got {group_inputs.shape}")
    models = list(pair_models) if isinstance(pair_models, (list, tuple)) else [pair_models]
    if not models:
        raise ValueError("at least one PairwiseNet is required")
    for model in models:
        model.eval()

    groups = np.ascontiguousarray(frags[group_inputs])
    macros, n_local, height, width, channels = groups.shape
    if height != width or channels != 3:
        raise ValueError(f"expected RGB square fragments, got {groups.shape}")
    x = torch.from_numpy(groups).permute(0, 1, 4, 2, 3).float().div_(255).to(device)
    xt = x.transpose(-1, -2)

    # Index order is (macro, left/top local tile, right/bottom local tile),
    # which reshapes directly into (macro, n_local, n_local) below.
    local_a = torch.arange(n_local, device=device).repeat_interleave(n_local).repeat(macros)
    local_b = torch.arange(n_local, device=device).repeat(n_local).repeat(macros)
    macro_id = torch.arange(macros, device=device).repeat_interleave(n_local * n_local)
    total = macros * n_local * n_local
    right = np.empty(total, dtype=np.float32)
    down = np.empty(total, dtype=np.float32)
    amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        m = macro_id[start:stop]
        a = local_a[start:stop]
        b = local_b[start:stop]
        horizontal = torch.cat((x[m, a], x[m, b]), dim=-1)
        vertical = torch.cat((xt[m, a], xt[m, b]), dim=-1)
        with amp:
            right_logits = sum(model(horizontal).float() for model in models) / len(models)
            down_logits = sum(model(vertical).float() for model in models) / len(models)
        right[start:stop] = right_logits.cpu().numpy()
        down[start:stop] = down_logits.cpu().numpy()

    shape = (macros, n_local, n_local)
    return right.reshape(shape), down.reshape(shape)


def local_metrics(place: np.ndarray, truth: np.ndarray, side: int) -> tuple[float, float, float, float, float]:
    """Strict local placement plus directed local right/down edge accuracy."""
    place = np.asarray(place, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    n = side * side
    if place.shape != (n,) or truth.shape != (n,):
        raise ValueError(f"expected two local placements of length {n}")
    true_right = {(int(truth[p]), int(truth[p + 1])) for p in range(n) if p % side < side - 1}
    true_down = {(int(truth[p]), int(truth[p + side])) for p in range(n - side)}
    pred_right = {(int(place[p]), int(place[p + 1])) for p in range(n) if p % side < side - 1}
    pred_down = {(int(place[p]), int(place[p + side])) for p in range(n - side)}
    n_edges = side * (side - 1)
    right = len(pred_right & true_right) / n_edges
    down = len(pred_down & true_down) / n_edges
    return (
        float(np.mean(place == truth)),
        float((len(pred_right & true_right) + len(pred_down & true_down)) / (2 * n_edges)),
        float(right),
        float(down),
        float(np.all(place == truth)),
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.side <= 1 or GRID % args.side:
        raise ValueError(f"--side must be greater than one and divide {GRID}")
    if args.iters <= 0 or args.restarts <= 0 or args.bs_score <= 0:
        raise ValueError("--iters, --restarts, and --bs_score must be positive")
    if args.temp_scale <= 0:
        raise ValueError("--temp_scale must be positive")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10, help="number of held-out targets to evaluate")
    ap.add_argument("--side", type=int, default=4, help="known square macro side (default: 4)")
    ap.add_argument("--iters", type=int, default=6_000, help="swap proposals per SA restart")
    ap.add_argument("--restarts", type=int, default=16, help="greedy/random SA starts per macro")
    ap.add_argument("--temp_scale", type=float, default=1.0, help="multiplier for SA initial temperature")
    ap.add_argument("--bs_score", type=int, default=4096, help="PairwiseNet ordered-pair batch size")
    ap.add_argument("--pair_tag", default="pair", help="PairwiseNet checkpoint tag")
    ap.add_argument("--which", default="best", help="checkpoint preference passed to pipeline.load_pair")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    _validate_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pipeline.load_pair uses pipeline.DEV while loading checkpoint tensors.
    pipeline.DEV = str(device)
    pair, pair_ckpt = pipeline.load_pair(args.pair_tag, args.which)
    if pair is None:
        raise FileNotFoundError("no PairwiseNet checkpoint found via pipeline.load_pair()")

    _, val_names = train_val_split()
    names = val_names[: args.n]
    if not names:
        raise RuntimeError("held-out validation split is empty")
    macros_per_image = (GRID // args.side) ** 2
    ckpt_step = pair_ckpt.get("step") if pair_ckpt else None
    ckpt_val = pair_ckpt.get("val") if pair_ckpt else None
    print(
        f"device={device} pair_step={ckpt_step} pair_val={ckpt_val} "
        f"images={len(names)} macros/image={macros_per_image} "
        f"macro={args.side}x{args.side} iters={args.iters} restarts={args.restarts}",
        flush=True,
    )

    totals = np.zeros(5, dtype=np.float64)  # placement, neigh, right, down, exact macro
    groups_total = 0
    started = time.time()

    for image_index, name in enumerate(names):
        rng = np.random.default_rng(args.seed + image_index)
        clean = load(os.path.join(TRAIN_TGT, name))
        # `perm` maps shuffled input fragment id -> clean source cell.  It is
        # never passed to the scorer/solver; it is used only to create the
        # synthetic input and the deliberately granted macro-membership oracle.
        perm = rng.permutation(NFRAG).astype(np.int64)
        frags = distort_frags(to_frags(clean), rng)[perm]
        group_inputs, truths = make_oracle_macro_groups(perm, side=args.side, rng=rng)
        R_all, D_all = score_groups_all_pairs(
            pair, frags, group_inputs, device=device, batch_size=args.bs_score
        )

        image_metrics = np.zeros((len(group_inputs), 5), dtype=np.float64)
        for macro in range(len(group_inputs)):
            place, _ = solve_grid_sa(
                R_all[macro],
                D_all[macro],
                side=args.side,
                iterations=args.iters,
                restarts=args.restarts,
                temp_scale=args.temp_scale,
                seed=args.seed + image_index * 100_003 + macro,
            )
            image_metrics[macro] = local_metrics(place, truths[macro], args.side)

        mean_metrics = image_metrics.mean(axis=0)
        totals += image_metrics.sum(axis=0)
        groups_total += len(image_metrics)
        print(
            f"[{image_index + 1:>3}/{len(names)}] {name} "
            f"place={mean_metrics[0]:.3f} neigh={mean_metrics[1]:.3f} "
            f"R={mean_metrics[2]:.3f} D={mean_metrics[3]:.3f} "
            f"exact={mean_metrics[4]:.3f}",
            flush=True,
        )

    aggregate = totals / max(1, groups_total)
    print(f"\n== Macro-oracle PairwiseNet evaluation: held-out synthetic, N={len(names)} ==")
    print(f"macro_side             {args.side}")
    print(f"groups_total           {groups_total}")
    print(f"local_place_acc        {aggregate[0]:.4f}")
    print(f"local_neighbour_acc    {aggregate[1]:.4f}")
    print(f"local_right_acc        {aggregate[2]:.4f}")
    print(f"local_down_acc         {aggregate[3]:.4f}")
    print(f"exact_macro_solve_rate {aggregate[4]:.4f}")
    print(f"seconds/image          {(time.time() - started) / len(names):.2f}")


if __name__ == "__main__":
    main()
