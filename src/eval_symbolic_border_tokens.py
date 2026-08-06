"""Gate the PuzLM border-token idea on the PAZZLE corruption process.

The experiment is deliberately smaller than a full sequence model.  It asks
whether PCA + k-means quantisation retains any useful information after every
20x20 tile is degraded independently.  A codebook is fitted on border patches,
then direction-specific token co-occurrence tables are learned from true
synthetic neighbours.  Held-out neighbour retrieval is label-free until the
final rank calculation.

This is based on the tokenizer described in PuzLM (ECCV 2026): divide every
tile into a BxB patch grid, project patches with PCA, quantise them with
k-means, and keep the clockwise border tokens.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from config import GRID, NFRAG, SEED, TRAIN_TGT, WORK_ROOT
from distort import distort_frags
from imgio import load, to_frags, train_val_split


@dataclass
class Tokenizer:
    pca: PCA
    kmeans: MiniBatchKMeans
    bins: int
    normalization: str

    def transform(self, tiles: np.ndarray) -> np.ndarray:
        patches = patch_grid(tiles, self.bins, self.normalization)
        shape = patches.shape[:-1]
        flat = patches.reshape(-1, patches.shape[-1])
        projected = self.pca.transform(flat)
        return self.kmeans.predict(projected).reshape(shape)


def normalize_tiles(tiles: np.ndarray, mode: str) -> np.ndarray:
    x = tiles.astype(np.float32) / 255.0
    if mode == "raw":
        return x
    if mode == "tile":
        mean = x.mean(axis=(1, 2), keepdims=True)
        std = x.std(axis=(1, 2), keepdims=True)
        return (x - mean) / np.maximum(std, 0.04)
    if mode == "gray_tile":
        gray = x @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
        mean = gray.mean(axis=(1, 2), keepdims=True)
        std = gray.std(axis=(1, 2), keepdims=True)
        return ((gray - mean) / np.maximum(std, 0.04))[..., None]
    raise ValueError(f"unknown normalization: {mode}")


def patch_grid(tiles: np.ndarray, bins: int, normalization: str) -> np.ndarray:
    """Return (tiles,bins,bins,flattened-patch) without crossing tile seams."""
    if tiles.ndim != 4 or tiles.shape[1:3] != (20, 20):
        raise ValueError(f"expected (N,20,20,C), got {tiles.shape}")
    if 20 % bins:
        raise ValueError("bins must divide the 20-pixel tile size")
    x = normalize_tiles(tiles, normalization)
    stride = 20 // bins
    x = x.reshape(len(x), bins, stride, bins, stride, x.shape[-1])
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(x).reshape(len(x), bins, bins, -1)


def border_mask(bins: int) -> np.ndarray:
    mask = np.zeros((bins, bins), dtype=bool)
    mask[0] = mask[-1] = True
    mask[:, 0] = mask[:, -1] = True
    return mask


def fit_tokenizer(
    names: list[str],
    *,
    bins: int,
    components: int,
    vocab: int,
    normalization: str,
    seed: int,
    max_patches: int,
) -> Tokenizer:
    rng = np.random.default_rng(seed)
    mask = border_mask(bins)
    chunks: list[np.ndarray] = []
    for index, name in enumerate(names):
        clean = to_frags(load(str(Path(TRAIN_TGT) / name)))
        dirty = distort_frags(clean, np.random.default_rng(seed + 1009 * index))
        for tiles in (clean, dirty):
            patches = patch_grid(tiles, bins, normalization)[:, mask]
            chunks.append(patches.reshape(-1, patches.shape[-1]))
    samples = np.concatenate(chunks, axis=0)
    if len(samples) > max_patches:
        samples = samples[rng.choice(len(samples), max_patches, replace=False)]
    n_components = min(components, samples.shape[1], len(samples) - 1)
    pca = PCA(n_components=n_components, whiten=True, random_state=seed)
    projected = pca.fit_transform(samples)
    kmeans = MiniBatchKMeans(
        n_clusters=vocab,
        batch_size=min(8192, len(projected)),
        n_init=3,
        max_iter=120,
        reassignment_ratio=0.01,
        random_state=seed,
    ).fit(projected)
    return Tokenizer(pca=pca, kmeans=kmeans, bins=bins, normalization=normalization)


def transition_counts(
    tokenizer: Tokenizer,
    names: list[str],
    *,
    seed: int,
    augmentations: int,
) -> tuple[np.ndarray, np.ndarray]:
    vocab = tokenizer.kmeans.n_clusters
    horizontal = np.ones((vocab, vocab), dtype=np.float64)
    vertical = np.ones((vocab, vocab), dtype=np.float64)
    cells = np.arange(NFRAG).reshape(GRID, GRID)
    left, right = cells[:, :-1].reshape(-1), cells[:, 1:].reshape(-1)
    top, bottom = cells[:-1].reshape(-1), cells[1:].reshape(-1)
    for image_index, name in enumerate(names):
        clean = to_frags(load(str(Path(TRAIN_TGT) / name)))
        for aug in range(augmentations):
            tiles = distort_frags(
                clean,
                np.random.default_rng(seed + 100_003 * image_index + 997 * aug),
            )
            tokens = tokenizer.transform(tiles)
            # Matching positions along a side remain aligned because rotations
            # are absent in this competition.
            for offset in range(tokenizer.bins):
                np.add.at(horizontal, (tokens[left, offset, -1], tokens[right, offset, 0]), 1)
                np.add.at(vertical, (tokens[top, -1, offset], tokens[bottom, 0, offset]), 1)
    return _pmi(horizontal), _pmi(vertical)


def _pmi(counts: np.ndarray) -> np.ndarray:
    """Smoothed pointwise mutual information prevents frequent codes winning."""
    total = counts.sum()
    expected = counts.sum(1, keepdims=True) @ counts.sum(0, keepdims=True) / total
    return np.log(counts) - np.log(np.maximum(expected, 1e-12))


def directional_scores(tokens: np.ndarray, table: np.ndarray, direction: str) -> np.ndarray:
    n, bins = len(tokens), tokens.shape[1]
    scores = np.zeros((n, n), dtype=np.float32)
    for offset in range(bins):
        if direction == "right":
            a, b = tokens[:, offset, -1], tokens[:, offset, 0]
        elif direction == "down":
            a, b = tokens[:, -1, offset], tokens[:, 0, offset]
        else:
            raise ValueError(direction)
        scores += table[a[:, None], b[None, :]].astype(np.float32)
    np.fill_diagonal(scores, -np.inf)
    return scores


def ranks_for_truth(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    truth_scores = scores[np.arange(len(scores)), truth]
    # Rank 1 means no candidate has a strictly larger score.
    return 1 + np.sum(scores > truth_scores[:, None], axis=1)


def evaluate(
    tokenizer: Tokenizer,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    names: list[str],
    *,
    seed: int,
) -> dict[str, float]:
    all_ranks: list[np.ndarray] = []
    agreements: list[float] = []
    cells = np.arange(NFRAG).reshape(GRID, GRID)
    true_right = np.full(NFRAG, -1, dtype=np.int64)
    true_down = np.full(NFRAG, -1, dtype=np.int64)
    true_right[cells[:, :-1].reshape(-1)] = cells[:, 1:].reshape(-1)
    true_down[cells[:-1].reshape(-1)] = cells[1:].reshape(-1)
    for image_index, name in enumerate(names):
        clean = to_frags(load(str(Path(TRAIN_TGT) / name)))
        dirty = distort_frags(clean, np.random.default_rng(seed + 1_000_003 * image_index))
        clean_tokens = tokenizer.transform(clean)
        dirty_tokens = tokenizer.transform(dirty)
        agreements.append(float(np.mean(clean_tokens[:, border_mask(tokenizer.bins)] == dirty_tokens[:, border_mask(tokenizer.bins)])))
        for direction, table, truth in (
            ("right", horizontal, true_right),
            ("down", vertical, true_down),
        ):
            valid = truth >= 0
            scores = directional_scores(dirty_tokens, table, direction)
            all_ranks.append(ranks_for_truth(scores[valid], truth[valid]))
    ranks = np.concatenate(all_ranks)
    return {
        "clean_dirty_exact_token_agreement": float(np.mean(agreements)),
        "neighbor_r1": float(np.mean(ranks <= 1)),
        "neighbor_r5": float(np.mean(ranks <= 5)),
        "neighbor_r10": float(np.mean(ranks <= 10)),
        "neighbor_r64": float(np.mean(ranks <= 64)),
        "neighbor_median_rank": float(np.median(ranks)),
        "neighbor_mean_rank": float(np.mean(ranks)),
        "evaluated_edges": float(len(ranks)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-images", type=int, default=32)
    parser.add_argument("--eval-images", type=int, default=6)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--vocab", type=int, default=128)
    parser.add_argument("--augmentations", type=int, default=2)
    parser.add_argument("--max-patches", type=int, default=250_000)
    parser.add_argument("--normalization", choices=("raw", "tile", "gray_tile"), default="tile")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(WORK_ROOT) / "gates" / "symbolic_border_tokens.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_names, val_names = train_val_split()
    if args.fit_images > len(train_names) or args.eval_images > len(val_names):
        raise ValueError("requested more images than the split contains")
    fit_names = train_names[: args.fit_images]
    eval_names = val_names[: args.eval_images]
    tokenizer = fit_tokenizer(
        fit_names,
        bins=args.bins,
        components=args.components,
        vocab=args.vocab,
        normalization=args.normalization,
        seed=args.seed,
        max_patches=args.max_patches,
    )
    horizontal, vertical = transition_counts(
        tokenizer,
        fit_names,
        seed=args.seed + 20_000,
        augmentations=args.augmentations,
    )
    metrics = evaluate(
        tokenizer,
        horizontal,
        vertical,
        eval_names,
        seed=args.seed + 40_000,
    )
    report = {
        "experiment": "puzlm_symbolic_border_token_gate",
        "configuration": {
            "fit_images": args.fit_images,
            "eval_images": args.eval_images,
            "bins": args.bins,
            "components": tokenizer.pca.n_components_,
            "vocab": args.vocab,
            "augmentations": args.augmentations,
            "normalization": args.normalization,
        },
        "metrics": metrics,
        "chance_neighbor_r1": 1.0 / (NFRAG - 1),
        "chance_neighbor_r5": 5.0 / (NFRAG - 1),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
