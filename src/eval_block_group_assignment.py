"""Branch G stage 2: inference-realistic capacitated grouping of dirty tiles.

Stage G1 (``eval_block_identity.py``) proved a single dirty tile's embedding
carries information about which of 36 same-image 4x4 macro-blocks it belongs
to (R@1=24.6%, R@5=64.1%, median rank 3 of 36) -- but that gate queried
against the TRUE clean block embeddings, which do not exist at test time
(only the shuffled dirty bag exists).

This gate is the realistic version: it uses ONLY the trained ``tile_encoder``
(frozen) on the 576 dirty tiles of one image and asks whether *unsupervised,
capacity-constrained clustering* of their embeddings alone -- no clean
reference of any kind -- recovers the true 16-tile block partition. This
works precisely because training pulled every one of a block's 16 sibling
dirty tiles toward the *same* shared clean-block target, so their embeddings
should cluster together even without querying that target directly.

Algorithm: alternating balanced k-means. Fix 36 centroids -> solve one global
optimal-transport-style assignment (scipy Hungarian on a 576x576 cost matrix,
each of the 36 centroids repeated as 16 identical candidate columns so every
cluster receives exactly 16 tiles) -> recompute centroids as the mean
(renormalized) embedding of newly assigned members -> repeat.

Purity is reported after optimally matching found clusters to true blocks
(Hungarian on the 36x36 overlap matrix) -- this is the metric that decides
whether branch G's grouping can feed the *already validated* macro_oracle
16-tile local solver (placement~=0.68 given a clean group), rather than
building yet another new assembler.

Examples
--------

    python src/eval_block_group_assignment.py --smoke
    python src/eval_block_group_assignment.py --ckpt E:/pazzle_work/ckpt/block_identity_best.pt --images 8 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from config import CKPT_DIR, NFRAG, SEED, TRAIN_TGT
from distort import distort_frags
from eval_block_identity import MACRO, NUM_BLOCKS, TILE_BLOCK_ID, BlockIdentity
from imgio import load, to_frags, train_val_split


def load_frozen_tile_encoder(path: str, device: torch.device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"block-identity checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    embed_dim = int(payload["embed_dim"])
    full = BlockIdentity(embed_dim=embed_dim)
    full.load_state_dict(payload["model"], strict=True)
    encoder = full.tile_encoder
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, embed_dim


def balanced_kmeans(
    embeddings: np.ndarray, *, num_clusters: int, capacity: int, iterations: int, rng: np.random.Generator
) -> np.ndarray:
    """Alternating capacitated-assignment k-means; returns a (N,) cluster id array.

    Each of ``num_clusters`` centroids is repeated ``capacity`` times as
    identical candidate columns so a single Hungarian solve on the expanded
    cost matrix yields a globally optimal *balanced* assignment (every
    cluster receives exactly ``capacity`` items) for the current centroids.
    """
    n, dim = embeddings.shape
    if n != num_clusters * capacity:
        raise ValueError(f"{n} embeddings does not equal {num_clusters}*{capacity}")
    init_idx = rng.choice(n, size=num_clusters, replace=False)
    centroids = embeddings[init_idx].copy()
    assignment = np.zeros(n, dtype=np.int64)
    for _ in range(iterations):
        cost = 1.0 - embeddings @ centroids.T  # cosine distance, embeddings are unit-norm
        expanded_cost = np.repeat(cost, capacity, axis=1)
        rows, columns = linear_sum_assignment(expanded_cost)
        assignment[rows] = columns // capacity
        for cluster in range(num_clusters):
            members = embeddings[assignment == cluster]
            if members.size:
                mean = members.mean(axis=0)
                norm = np.linalg.norm(mean)
                centroids[cluster] = mean / norm if norm > 1.0e-8 else centroids[cluster]
    return assignment


def group_purity(assignment: np.ndarray, true_labels: np.ndarray, num_clusters: int) -> dict[str, float]:
    """Optimally match found clusters to true blocks, then report overlap purity."""
    overlap = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    for cluster in range(num_clusters):
        members = true_labels[assignment == cluster]
        if members.size:
            overlap[cluster] += np.bincount(members, minlength=num_clusters)
    rows, columns = linear_sum_assignment(-overlap)
    matched_correct = overlap[rows, columns]
    capacity = int(np.bincount(assignment, minlength=num_clusters).max())
    perfect_blocks = int(np.sum(matched_correct == capacity))
    near_perfect_blocks = int(np.sum(matched_correct >= capacity - 2))
    return {
        "purity": float(matched_correct.sum() / true_labels.size),
        "perfect_blocks": perfect_blocks,
        "perfect_block_fraction": perfect_blocks / num_clusters,
        "near_perfect_blocks_ge_capacity_minus_2": near_perfect_blocks,
        "near_perfect_block_fraction": near_perfect_blocks / num_clusters,
        "mean_block_purity": float(np.mean(matched_correct) / capacity),
    }


@torch.inference_mode()
def _dirty_embeddings(tile_encoder, name: str, rng: np.random.Generator, device: torch.device) -> np.ndarray:
    clean_tiles = to_frags(load(os.path.join(TRAIN_TGT, name)))
    dirty_tiles = distort_frags(clean_tiles, rng)
    tensor = (
        torch.from_numpy(np.ascontiguousarray(dirty_tiles)).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    )
    return tile_encoder(tensor).float().cpu().numpy()


def evaluate(
    tile_encoder, names: list[str], *, iterations: int, seed: int, device: torch.device
) -> dict[str, Any]:
    capacity = MACRO * MACRO
    per_image: list[dict[str, float]] = []
    for index, name in enumerate(names):
        rng = np.random.default_rng(seed + index * 7919)
        embeddings = _dirty_embeddings(tile_encoder, name, rng, device)
        assignment = balanced_kmeans(
            embeddings, num_clusters=NUM_BLOCKS, capacity=capacity, iterations=iterations,
            rng=np.random.default_rng(seed + index * 131 + 1),
        )
        metrics = group_purity(assignment, TILE_BLOCK_ID, NUM_BLOCKS)
        metrics["image"] = name
        per_image.append(metrics)
        print(
            f"  {name}: purity={metrics['purity']:.4f} perfect_blocks={metrics['perfect_blocks']}/{NUM_BLOCKS} "
            f"near_perfect={metrics['near_perfect_blocks_ge_capacity_minus_2']}/{NUM_BLOCKS}",
            flush=True,
        )
    purity = np.array([item["purity"] for item in per_image])
    perfect_fraction = np.array([item["perfect_block_fraction"] for item in per_image])
    near_perfect_fraction = np.array([item["near_perfect_blocks_ge_capacity_minus_2"] for item in per_image]) / NUM_BLOCKS
    return {
        "per_image": per_image,
        "mean_purity": float(purity.mean()),
        "mean_perfect_block_fraction": float(perfect_fraction.mean()),
        "mean_near_perfect_block_fraction": float(near_perfect_fraction.mean()),
        "images": len(names),
    }


def _null_baseline(seed: int, dim: int = 128, iterations: int = 15) -> dict[str, float]:
    """Chance purity from unit-random embeddings carrying no block signal."""
    capacity = MACRO * MACRO
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(NFRAG, dim)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    assignment = balanced_kmeans(
        embeddings, num_clusters=NUM_BLOCKS, capacity=capacity, iterations=iterations,
        rng=np.random.default_rng(seed + 1),
    )
    return group_purity(assignment, TILE_BLOCK_ID, NUM_BLOCKS)


def smoke() -> dict[str, float]:
    """Data-free contract test: a perfectly-separated toy embedding must recover exact groups."""
    rng = np.random.default_rng(3)
    capacity = MACRO * MACRO
    dim = 40
    centers = rng.normal(size=(NUM_BLOCKS, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    embeddings = np.empty((NFRAG, dim), dtype=np.float32)
    for block in range(NUM_BLOCKS):
        members = np.flatnonzero(TILE_BLOCK_ID == block)
        noise = rng.normal(scale=0.01, size=(len(members), dim))
        vectors = centers[block][None, :] + noise
        embeddings[members] = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    assignment = balanced_kmeans(
        embeddings, num_clusters=NUM_BLOCKS, capacity=capacity, iterations=20, rng=np.random.default_rng(7)
    )
    metrics = group_purity(assignment, TILE_BLOCK_ID, NUM_BLOCKS)
    if metrics["purity"] < 0.999 or metrics["perfect_blocks"] != NUM_BLOCKS:
        raise AssertionError(f"near-separable toy embedding failed to recover exact groups: {metrics}")

    # The empirical null is not a small number: optimally matching 36 found
    # groups to 36 true groups (best-of-36! Hungarian matching) inflates
    # overlap well above the naive 1/36 per-item chance rate. Rather than
    # assert an a-priori guessed bound, check the toy near-separable case
    # clears the null by a wide, unambiguous margin.
    null_metrics = _null_baseline(seed=99, dim=dim, iterations=10)
    if metrics["purity"] < 3.0 * null_metrics["purity"]:
        raise AssertionError(
            f"near-separable toy embedding did not clear the empirical null baseline: "
            f"toy={metrics['purity']} null={null_metrics['purity']}"
        )
    return {"toy_purity": metrics["purity"], "null_purity": null_metrics["purity"]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=os.path.join(CKPT_DIR, "block_identity_best.pt"))
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=SEED + 9973)
    parser.add_argument("--report", type=Path, default=Path("E:/pazzle_work/gates/block_group_assignment_gate.json"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        return args
    if args.images < 1 or args.iterations < 1:
        parser.error("--images and --iterations must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if args.smoke:
        print(f"[block-group-assignment smoke] {smoke()}", flush=True)
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tile_encoder, embed_dim = load_frozen_tile_encoder(args.ckpt, device)
    _, val_names = train_val_split()
    if len(val_names) < args.images:
        raise ValueError(f"--images exceeds the held-out pool ({len(val_names)})")

    print(
        f"device={device} tile_encoder from {args.ckpt} (embed_dim={embed_dim}) "
        f"images={args.images} iterations={args.iterations} "
        f"blocks={NUM_BLOCKS} capacity={MACRO * MACRO} "
        "method=alternating capacitated k-means (Hungarian per round), NO clean reference used",
        flush=True,
    )
    result = evaluate(tile_encoder, val_names[: args.images], iterations=args.iterations, seed=args.seed, device=device)
    null_metrics = _null_baseline(seed=args.seed, dim=embed_dim, iterations=args.iterations)
    print(f"null (random-embedding) baseline: {null_metrics}", flush=True)

    passed = result["mean_purity"] >= 0.30 and result["mean_purity"] >= 3.0 * null_metrics["purity"]
    report = {
        "experiment": "stage_g2_capacitated_dirty_tile_block_grouping",
        "question": (
            "does unsupervised capacity-constrained clustering of ONLY dirty-tile embeddings "
            "(no clean reference, inference-realistic) recover the true 16-tile macro-block partition?"
        ),
        "checkpoint": args.ckpt,
        "null_baseline": null_metrics,
        "result": result,
        "gate": {
            "rule": "mean_purity >= 0.30 AND mean_purity >= 3x the random-embedding null baseline",
            "pass": bool(passed),
            "note": (
                "a pass means real, clean-reference-free macro-block groups of 16 are recoverable "
                "at test time and can feed the already-validated macro_oracle local solver "
                "(placement~=0.68 given a clean 16-tile group) -- the next step would be an "
                "end-to-end SSIM measurement using these recovered groups."
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    verdict = "PASSED -> feed groups into the macro_oracle local solver" if passed else "FAILED -> close branch G"
    print(f"\n=== stage G2 gate {verdict} ===", flush=True)
    print(json.dumps(report["gate"], indent=2), flush=True)
    print(f"report saved to {args.report}", flush=True)


if __name__ == "__main__":
    main()
