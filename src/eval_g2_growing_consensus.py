"""G2: sparse 2x2 growing-consensus diagnostic on U1 candidate graphs.

Candidate construction and closure enumeration are label-blind. Exact synthetic
labels are consulted only after supported edge masks are frozen for held-out
metrics. This is a pre-gate diagnostic, not an assignment solver.
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor

from canvas_data import CanvasDataset
from candidate_rank import DOWN, RIGHT
from config import GRID, NFRAG
from direct_pose import NON_DIRECT_CLASS
from eval_r2l_affinity_union import DEFAULT_AFFINITY_A, DEFAULT_AFFINITY_B, DEFAULT_R2L, _load_r2, _union_candidates
from imgio import train_val_split
from train_direct_pose import candidate_direct_labels
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates

DIRECT_EDGES_PER_BOARD = 4 * GRID * (GRID - 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G2 sparse 2x2 growing-consensus diagnostic")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--prefixes", default="8,16,32")
    p.add_argument("--affinity-k", type=int, default=64)
    p.add_argument("--r2-topk", type=int, default=8)
    p.add_argument("--affinity-ckpt", default=DEFAULT_AFFINITY_A)
    p.add_argument("--affinity-ckpt2", default=DEFAULT_AFFINITY_B)
    p.add_argument("--r2-ckpt", default=DEFAULT_R2L)
    p.add_argument("--seed", type=int, default=240815)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


@torch.inference_mode()
def build_u1(
    tiles: Tensor,
    affinity: torch.nn.Module,
    affinity2: torch.nn.Module,
    r2: torch.nn.Module,
    affinity_k: int,
    r2_topk: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    base, base_valid = mine_affinity_candidates(
        affinity, tiles.unsqueeze(0), candidate_k=affinity_k, device=device, affinity_secondary=affinity2
    )
    with autocast(device):
        r2_scores = r2(tiles.unsqueeze(0))[0].float()
    return _union_candidates(base[0], base_valid[0], r2_scores, r2_topk)


def prefix_mask(valid: Tensor, width: int) -> Tensor:
    """Keep an ordering-preserving candidate prefix for every tile/direction."""
    if width < 1:
        raise ValueError("prefix width must be positive")
    cap = min(width, valid.shape[-1])
    keep = torch.zeros_like(valid)
    keep[..., :cap] = True
    return keep & valid


def support_2x2(candidates: Tensor, valid: Tensor) -> tuple[Tensor, int]:
    """Mark directed edges participating in one or more candidate-supported 2x2 loops.

    Geometry uses canonical directions: a--RIGHT-->b, a--DOWN-->c,
    b--DOWN-->d, c--RIGHT-->d.  Membership is label-blind and candidate sparse.
    """
    support = torch.zeros_like(valid)
    closures = 0
    for a in range(NFRAG):
        right_a = candidates[a, RIGHT, valid[a, RIGHT]].tolist()
        down_a = candidates[a, DOWN, valid[a, DOWN]].tolist()
        if not right_a or not down_a:
            continue
        for b in right_a:
            down_b = candidates[b, DOWN, valid[b, DOWN]].tolist()
            if not down_b:
                continue
            down_b_set = set(down_b)
            for c in down_a:
                right_c = candidates[c, RIGHT, valid[c, RIGHT]].tolist()
                shared = down_b_set.intersection(right_c)
                if not shared:
                    continue
                for d in shared:
                    # Find exact candidate slots. Candidates can contain duplicates;
                    # supporting every repeated slot preserves graph semantics.
                    support[a, RIGHT] |= valid[a, RIGHT] & candidates[a, RIGHT].eq(b)
                    support[a, DOWN] |= valid[a, DOWN] & candidates[a, DOWN].eq(c)
                    support[b, DOWN] |= valid[b, DOWN] & candidates[b, DOWN].eq(d)
                    support[c, RIGHT] |= valid[c, RIGHT] & candidates[c, RIGHT].eq(d)
                    closures += 1
    return support, closures


def metrics(selected: Tensor, labels: Tensor) -> dict[str, float]:
    total = int(selected.sum())
    direct = selected & labels.ne(NON_DIRECT_CLASS)
    direction = torch.arange(4, device=labels.device).view(1, 4, 1)
    exact = direct & labels.eq(direction)
    return {
        "selected_edges": float(total),
        "selected_edges_per_tile": float(total) / NFRAG,
        "direct_precision": float(direct.sum()) / total if total else 0.0,
        "direct_recall_all_true": float(direct.sum()) / DIRECT_EDGES_PER_BOARD,
        "exact_direction_precision": float(exact.sum()) / total if total else 0.0,
        "exact_direction_recall_all_true": float(exact.sum()) / DIRECT_EDGES_PER_BOARD,
    }


def aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    return {key: float(sum(record[key] for record in records) / len(records)) for key in records[0]}


def main() -> None:
    args = parse_args()
    prefixes = sorted({int(x) for x in args.prefixes.split(",") if x.strip()})
    if args.n < 1 or not prefixes or min(prefixes) < 1:
        raise ValueError("n and all prefixes must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    affinity, _, _ = load_frozen_affinity(args.affinity_ckpt, device)
    affinity2, _, _ = load_frozen_affinity(args.affinity_ckpt2, device)
    r2 = _load_r2(args.r2_ckpt, device)
    _, val_names = train_val_split()
    dataset = CanvasDataset(val_names[: args.n], real_prob=0.0, seed=args.seed)
    result: dict[str, object] = {
        "experiment": "G2_sparse_growing_consensus",
        "images": args.n,
        "affinity_k": args.affinity_k,
        "r2_topk": args.r2_topk,
        "prefixes": {},
    }
    all_prefix: dict[int, dict[str, list[dict[str, float]]]] = {
        p: {"raw": [], "supported": [], "closure_count": []} for p in prefixes
    }
    for image_index in range(args.n):
        sample = dataset[image_index]
        tiles = sample["tiles"].to(device)
        perm = sample["perm"].to(device).long()
        candidates, valid = build_u1(tiles, affinity, affinity2, r2, args.affinity_k, args.r2_topk, device)
        flat_candidates = candidates.reshape(1, NFRAG, -1)
        labels = candidate_direct_labels(perm.unsqueeze(0), flat_candidates)[0].reshape_as(candidates)
        for width in prefixes:
            raw = prefix_mask(valid, width)
            supported, closures = support_2x2(candidates, raw)
            raw_m = metrics(raw, labels)
            supported_m = metrics(supported, labels)
            all_prefix[width]["raw"].append(raw_m)
            all_prefix[width]["supported"].append(supported_m)
            all_prefix[width]["closure_count"].append({"closures": float(closures)})
            print(
                f"image={image_index+1}/{args.n} prefix={width} closures={closures} "
                f"raw_p={raw_m['direct_precision']:.4f} support_p={supported_m['direct_precision']:.4f} "
                f"support_r={supported_m['direct_recall_all_true']:.4f}",
                flush=True,
            )
    for width in prefixes:
        raw_m = aggregate(all_prefix[width]["raw"])
        supported_m = aggregate(all_prefix[width]["supported"])
        closures = aggregate(all_prefix[width]["closure_count"])
        lift = supported_m["direct_precision"] / raw_m["direct_precision"] if raw_m["direct_precision"] else 0.0
        result["prefixes"][str(width)] = {
            "raw": raw_m,
            "consensus_supported": supported_m,
            "mean_closures": closures["closures"],
            "precision_lift_vs_raw": lift,
            "pre_gate_pass": bool(lift >= 2.0 and supported_m["direct_recall_all_true"] >= 0.10),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
