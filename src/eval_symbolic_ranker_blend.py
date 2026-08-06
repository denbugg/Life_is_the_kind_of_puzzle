"""Blend PuzLM-style symbolic border evidence with frozen neural ranker rows.

The full-graph caches contain the exact candidate rows and synthetic
permutations used by the existing solver.  This script deterministically
recreates their corrupted tile bags, fits the symbolic tokenizer only on the
training split, and measures whether a row-wise blend improves held-out edge
ranking and ordinary buddies assembly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from candidate_rank import DOWN, LEFT, RIGHT, UP, neighbor_targets
from canvas_data import CanvasDataset
from config import NFRAG, SEED, WORK_ROOT
from eval_seeded_qap import dense_rd
from eval_symbolic_border_tokens import (
    directional_scores,
    fit_tokenizer,
    transition_counts,
)
from imgio import train_val_split
from placement_metrics import neighbour_accuracy, placement_accuracy
from solve_buddies import solve_buddies_from_scores


def _standardize_rows(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full_like(values, -20.0, dtype=np.float32)
    for anchor in range(values.shape[0]):
        for direction in range(values.shape[1]):
            mask = valid[anchor, direction]
            row = values[anchor, direction, mask].astype(np.float32)
            if not len(row):
                continue
            out[anchor, direction, mask] = (row - row.mean()) / max(float(row.std()), 1.0e-4)
    return out


def _recreate_group(
    validation_names: list[str], start: int, count: int, *, seed: int
) -> dict[int, dict[str, torch.Tensor]]:
    # eval_confident_islands.py reset these RNGs before constructing each cache
    # range.  CanvasDataset's extra global draw is therefore reproducible.
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = CanvasDataset(
        validation_names[start : start + count],
        real_prob=0.0,
        seed=seed + 400_000,
    )
    return {start + local: dataset[local] for local in range(count)}


def _parse_groups(text: str) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    for item in text.split(","):
        start, count = item.split(":")
        groups.append((int(start), int(count)))
    return groups


def _token_direction_scores(tokens: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    right = directional_scores(tokens, horizontal, "right")
    down = directional_scores(tokens, vertical, "down")
    return np.stack((down.T, down, right.T, right), axis=0)  # UP, DOWN, LEFT, RIGHT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="0:2,10:12")
    parser.add_argument("--alphas", default="0,0.05,0.1,0.2,0.4,0.7")
    parser.add_argument("--fit-images", type=int, default=32)
    parser.add_argument("--vocab", type=int, default=32)
    parser.add_argument("--components", type=int, default=24)
    parser.add_argument("--augmentations", type=int, default=2)
    parser.add_argument("--max-patches", type=int, default=200_000)
    parser.add_argument("--max-edges", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "symbolic_ranker_blend.json",
    )
    args = parser.parse_args()
    alphas = [float(value) for value in args.alphas.split(",")]
    train_names, validation_names = train_val_split()

    tokenizer = fit_tokenizer(
        train_names[: args.fit_images],
        bins=4,
        components=args.components,
        vocab=args.vocab,
        normalization="raw",
        seed=args.seed,
        max_patches=args.max_patches,
    )
    horizontal, vertical = transition_counts(
        tokenizer,
        train_names[: args.fit_images],
        seed=args.seed + 20_000,
        augmentations=args.augmentations,
    )

    samples: dict[int, dict[str, torch.Tensor]] = {}
    for start, count in _parse_groups(args.groups):
        samples.update(_recreate_group(validation_names, start, count, seed=args.seed))

    rows: dict[str, list[dict[str, float]]] = {str(alpha): [] for alpha in alphas}
    for image_id, sample in samples.items():
        cache_path = args.cache_dir / f"image_{image_id:04d}_k64.npz"
        stored = np.load(cache_path, allow_pickle=False)
        permutation = stored["permutation"].astype(np.int64)
        if not np.array_equal(sample["perm"].numpy(), permutation):
            raise RuntimeError(f"could not reproduce cached synthetic bag {image_id}")
        candidates = stored["candidate_ids"].astype(np.int64)
        raw = stored["candidate_scores"].reshape(NFRAG, 4, -1).astype(np.float32)
        valid = np.isfinite(raw)

        tiles = sample["tiles"].permute(0, 2, 3, 1).numpy()
        tokens = tokenizer.transform(np.rint(tiles * 255.0).clip(0, 255).astype(np.uint8))
        full_symbolic = _token_direction_scores(tokens, horizontal, vertical)
        symbolic = np.empty_like(raw)
        for direction in range(4):
            symbolic[:, direction] = full_symbolic[direction][
                np.arange(NFRAG)[:, None], candidates
            ]
        raw_z = _standardize_rows(raw, valid)
        symbolic_z = _standardize_rows(symbolic, valid)

        truth, exists = neighbor_targets(torch.from_numpy(permutation)[None].long())
        truth = truth[0].numpy()
        exists = exists[0].numpy()
        for alpha in alphas:
            blend = raw_z + alpha * symbolic_z
            blend[~valid] = -np.inf
            top = candidates[np.arange(NFRAG)[:, None], np.argmax(blend, axis=2)]
            exact = top == truth
            edge_r1 = float(exact[exists].mean())

            score_tensor = torch.from_numpy(blend).permute(1, 0, 2).contiguous()
            candidate_tensor = torch.from_numpy(candidates).long()
            right, down = dense_rd(candidate_tensor, score_tensor)
            placement, _ = solve_buddies_from_scores(
                right.numpy(),
                down.numpy(),
                max_edges=args.max_edges,
                min_margin=0.0,
                repair_passes=0,
            )
            target_board = np.argsort(permutation)
            placement_score, _ = placement_accuracy(placement, target_board)
            neighbour, _, _ = neighbour_accuracy(placement, target_board)
            metrics = {
                "image": float(image_id),
                "edge_r1": edge_r1,
                "placement": float(placement_score),
                "neighbour": float(neighbour),
            }
            rows[str(alpha)].append(metrics)
            print(json.dumps({"alpha": alpha, **metrics}), flush=True)

    summary = {
        alpha: {
            key: float(np.mean([row[key] for row in values]))
            for key in ("edge_r1", "placement", "neighbour")
        }
        for alpha, values in rows.items()
    }
    baseline = summary[str(alphas[0])]
    best_alpha = max(summary, key=lambda value: summary[value]["neighbour"])
    report = {
        "experiment": "puzlm_symbolic_plus_neural_ranker_blend",
        "configuration": {
            "groups": args.groups,
            "images": len(samples),
            "fit_images": args.fit_images,
            "vocab": args.vocab,
            "components": args.components,
            "max_edges": args.max_edges,
        },
        "baseline": baseline,
        "best_alpha": best_alpha,
        "best": summary[best_alpha],
        "delta_vs_baseline": {
            key: summary[best_alpha][key] - baseline[key]
            for key in baseline
        },
        "summary": summary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
