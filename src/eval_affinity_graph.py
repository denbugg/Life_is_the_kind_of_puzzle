"""Evaluate learned tile-proximity affinities on exact synthetic puzzles.

This is a diagnostic for the *relative local-proximity* branch, not an
assembler.  It asks a deliberately narrow question: after shuffling and
corrupting a complete image, do tiles that came from nearby clean-grid cells
become each other's nearest neighbours in the learned embedding graph?

The evaluator never touches the real-input recovered-permutation cache.
``CanvasDataset(real_prob=0)`` creates a fresh distortion and shuffle from a
clean training target, so ``perm[input_tile] -> clean_cell`` is exact.

Example (held-out synthetic images):

    python src/eval_affinity_graph.py ^
      --ckpt artifacts/macro_affinity/affinity_best.pt --n 12 --device cuda

The ``raw_zpixel`` report is a no-learning baseline: each raw corrupted tile
is zero-centred, RMS-normalised, flattened, and compared by cosine similarity.
That cancels independent brightness/contrast scale while retaining all pixel
structure.  The learned report uses cosine similarity of ``MacroAffinityNet``
embeddings from the supplied checkpoint.
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from canvas_data import CanvasDataset
from config import FS, GRID, NFRAG, SEED
from imgio import train_val_split
from macro_affinity import MacroAffinityNet, count_params


RADII = (1, 3, 4)
TOPK = (5, 15, 32, 64)


def _autocast(device: torch.device):
    """Use fp16 inference on CUDA while preserving an uncomplicated CPU path."""
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )


def _is_tensor_state_dict(value: object) -> bool:
    """Return whether ``value`` looks like a plain PyTorch module state dict."""
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) and isinstance(item, Tensor) for key, item in value.items())
    )


def _torch_load(path: str) -> object:
    """Load a trusted local experiment checkpoint across recent torch versions."""
    try:
        # Explicitly retain metadata such as the trainer's ``args`` and metrics.
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the ``weights_only`` keyword.
        return torch.load(path, map_location="cpu")


def _checkpoint_state(payload: object) -> dict[str, Tensor]:
    """Extract a model state dict from common checkpoint layouts.

    The affinity trainer stores ``{"model": state_dict, "args": ...}``, but
    accepting a raw state dict and common aliases keeps this evaluator useful
    for interrupted/manual experimental saves as well.
    """
    if isinstance(payload, torch.nn.Module):
        return dict(payload.state_dict())
    if _is_tensor_state_dict(payload):
        return dict(payload)
    if isinstance(payload, Mapping):
        for key in ("model", "model_state_dict", "state_dict", "network", "net"):
            candidate = payload.get(key)
            if isinstance(candidate, torch.nn.Module):
                return dict(candidate.state_dict())
            if _is_tensor_state_dict(candidate):
                return dict(candidate)
    raise RuntimeError(
        "checkpoint does not contain a recognizable model state dict; expected a raw "
        "state_dict or one under model/model_state_dict/state_dict"
    )


def _strip_uniform_prefix(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Remove DDP/wrapper prefixes only when every key uses the same prefix."""
    cleaned = dict(state)
    for prefix in ("module.", "model."):
        keys = tuple(cleaned)
        if keys and all(key.startswith(prefix) for key in keys):
            cleaned = {key[len(prefix) :]: value for key, value in cleaned.items()}
    return cleaned


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Read saved argparse metadata without requiring a particular checkpoint type."""
    if isinstance(value, Mapping):
        return value
    namespace = getattr(value, "__dict__", None)
    return namespace if isinstance(namespace, Mapping) else {}


def _first_int(*values: object, fallback: int) -> int:
    for value in values:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return fallback


def _first_float(*values: object, fallback: float) -> float:
    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= parsed < 1.0:
            return parsed
    return fallback


def _state_shape(state: Mapping[str, Tensor], key: str, dimension: int) -> int | None:
    tensor = state.get(key)
    if tensor is None or tensor.ndim <= dimension:
        return None
    return int(tensor.shape[dimension])


def _model_kwargs(payload: object, state: Mapping[str, Tensor]) -> dict[str, Any]:
    """Recover MacroAffinityNet architecture from metadata, then state shapes."""
    metadata = _as_mapping(payload)
    args = _as_mapping(metadata.get("args"))
    saved_kwargs = _as_mapping(metadata.get("model_kwargs"))

    embedding_dim = _first_int(
        saved_kwargs.get("embedding_dim"),
        saved_kwargs.get("d"),
        args.get("embedding_dim"),
        args.get("d"),
        _state_shape(state, "backbone.head.5.weight", 0),
        _state_shape(state, "backbone.head.4.weight", 0),
        fallback=128,
    )
    width = _first_int(
        saved_kwargs.get("width"),
        args.get("width"),
        _state_shape(state, "backbone.stem.0.weight", 0),
        fallback=48,
    )
    has_stats = any(key.startswith("stats.") for key in state)
    stats_hidden = _first_int(
        saved_kwargs.get("stats_hidden"),
        args.get("stats_hidden"),
        _state_shape(state, "stats.net.1.weight", 0),
        fallback=32,
    )
    # Dropout has no learned state, and models are evaluated in eval mode; use
    # saved metadata when present solely to reconstruct the same module shape.
    dropout = _first_float(saved_kwargs.get("dropout"), args.get("dropout"), fallback=0.0)
    return {
        "tiles": NFRAG,
        "tile_size": FS,
        "embedding_dim": embedding_dim,
        "width": width,
        "use_stats": has_stats,
        "stats_hidden": stats_hidden,
        "dropout": dropout,
    }


def load_model(path: str, device: torch.device) -> tuple[MacroAffinityNet, Mapping[str, Any]]:
    """Load and validate a MacroAffinityNet checkpoint with useful failure text."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = _torch_load(path)
    state = _strip_uniform_prefix(_checkpoint_state(payload))
    kwargs = _model_kwargs(payload, state)
    model = MacroAffinityNet(**kwargs)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        missing = ", ".join(incompatible.missing_keys[:8]) or "none"
        unexpected = ", ".join(incompatible.unexpected_keys[:8]) or "none"
        raise RuntimeError(
            "checkpoint architecture does not match MacroAffinityNet "
            f"(missing: {missing}; unexpected: {unexpected}; inferred kwargs: {kwargs})"
        )
    model.to(device).eval()
    return model, _as_mapping(payload)


def clean_coordinates(perm: Tensor) -> tuple[Tensor, Tensor]:
    """Map ``input tile -> clean row-major cell`` labels to row and column."""
    if perm.ndim != 1 or perm.numel() != NFRAG:
        raise ValueError(f"perm must have shape ({NFRAG},), got {tuple(perm.shape)}")
    perm = perm.long()
    if not torch.equal(perm.sort().values.cpu(), torch.arange(NFRAG)):
        raise ValueError("synthetic sample's perm is not a valid clean-cell permutation")
    return torch.div(perm, GRID, rounding_mode="floor"), torch.remainder(perm, GRID)


def chebyshev_distance(perm: Tensor) -> Tensor:
    """Return exact clean-grid Chebyshev distances between every input-tile pair."""
    rows, cols = clean_coordinates(perm)
    return torch.maximum(
        (rows[:, None] - rows[None, :]).abs(),
        (cols[:, None] - cols[None, :]).abs(),
    )


def raw_zpixel_affinity(tiles: Tensor) -> Tensor:
    """Return a raw-pixel correlation affinity with per-tile exposure normalisation."""
    if tiles.ndim != 4 or tuple(tiles.shape[1:]) != (3, FS, FS):
        raise ValueError(f"tiles must have shape ({NFRAG},3,{FS},{FS}), got {tuple(tiles.shape)}")
    pixels = tiles.float().reshape(tiles.shape[0], -1)
    pixels = pixels - pixels.mean(dim=-1, keepdim=True)
    pixels = pixels / pixels.square().mean(dim=-1, keepdim=True).add(1.0e-6).sqrt()
    pixels = F.normalize(pixels, dim=-1, eps=1.0e-6)
    return pixels @ pixels.transpose(0, 1)


def learned_affinity(model: MacroAffinityNet, tiles: Tensor, device: torch.device) -> Tensor:
    """Run the tile encoder and return fp32 cosine affinities for one full puzzle."""
    batch = tiles.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
    with torch.inference_mode(), _autocast(device):
        embeddings = model.embed(batch)
    if not isinstance(embeddings, Tensor) or tuple(embeddings.shape[:2]) != (1, NFRAG):
        raise RuntimeError(
            "MacroAffinityNet.embed must return (1,576,D), got "
            f"{tuple(embeddings.shape) if isinstance(embeddings, Tensor) else type(embeddings)}"
        )
    unit = F.normalize(embeddings.float(), dim=-1, eps=1.0e-6)[0]
    return unit @ unit.transpose(0, 1)


def top_neighbours(affinity: Tensor, maximum_k: int) -> Tensor:
    """Select highest-affinity non-self neighbours for every anchor tile."""
    if tuple(affinity.shape) != (NFRAG, NFRAG):
        raise ValueError(f"affinity must have shape ({NFRAG},{NFRAG}), got {tuple(affinity.shape)}")
    if not 1 <= maximum_k < NFRAG:
        raise ValueError(f"maximum_k must be in [1,{NFRAG - 1}]")
    mask = torch.eye(NFRAG, dtype=torch.bool, device=affinity.device)
    return affinity.float().masked_fill(mask, -torch.inf).topk(maximum_k, dim=-1).indices


def retrieval_metrics(distance: Tensor, neighbours: Tensor, ks: Sequence[int]) -> dict[str, float]:
    """Compute average anchor-level precision and recall at K for exact radii."""
    if tuple(distance.shape) != (NFRAG, NFRAG):
        raise ValueError("distance must be a full puzzle square matrix")
    if neighbours.ndim != 2 or neighbours.shape[0] != NFRAG:
        raise ValueError("neighbours must have one ranked row per input tile")
    result: dict[str, float] = {}
    gathered_distance = distance.gather(1, neighbours)
    off_diagonal = ~torch.eye(NFRAG, dtype=torch.bool, device=distance.device)
    for radius in RADII:
        positive_count = ((distance <= radius) & off_diagonal).sum(dim=-1).float()
        hits = (gathered_distance <= radius).float().cumsum(dim=-1)
        for k in ks:
            found = hits[:, k - 1]
            result[f"r{radius}_precision@{k}"] = float((found / float(k)).mean())
            result[f"r{radius}_recall@{k}"] = float((found / positive_count).mean())
    return result


class _UnionFind:
    """Tiny union-find used only for a 576-node undirected kNN graph."""

    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]

    def component_sizes(self) -> list[int]:
        counts: dict[int, int] = defaultdict(int)
        for node in range(len(self.parent)):
            counts[self.find(node)] += 1
        return list(counts.values())


def graph_metrics(distance: Tensor, neighbours: Tensor, ks: Sequence[int]) -> dict[str, float]:
    """Summarize spatial locality and connectivity of symmetrized kNN graphs."""
    result: dict[str, float] = {}
    selected_distance = distance.gather(1, neighbours)
    neighbours_cpu = neighbours.detach().cpu().numpy()
    for k in ks:
        edge_distance = selected_distance[:, :k].float()
        result[f"graph_mean_cheb@{k}"] = float(edge_distance.mean())
        for radius in RADII:
            result[f"graph_edge_r{radius}@{k}"] = float((edge_distance <= radius).float().mean())

        union_find = _UnionFind(NFRAG)
        for source, targets in enumerate(neighbours_cpu[:, :k]):
            for target in targets:
                union_find.union(source, int(target))
        sizes = union_find.component_sizes()
        result[f"graph_components@{k}"] = float(len(sizes))
        result[f"graph_largest_fraction@{k}"] = float(max(sizes) / NFRAG)
        # Expected component size for a uniformly random node; this exposes a
        # handful of tiny components even when a giant component is present.
        result[f"graph_node_mean_component@{k}"] = float(sum(size * size for size in sizes) / NFRAG)
    return result


def _largest_component(neighbours: Tensor, k: int) -> np.ndarray:
    """Return node indices belonging to the largest undirected kNN component."""
    union_find = _UnionFind(NFRAG)
    for source, targets in enumerate(neighbours[:, :k].detach().cpu().numpy()):
        for target in targets:
            union_find.union(source, int(target))
    groups: dict[int, list[int]] = defaultdict(list)
    for node in range(NFRAG):
        groups[union_find.find(node)].append(node)
    return np.asarray(max(groups.values(), key=len), dtype=np.int64)


def spectral_affine_r2(
    neighbours: Tensor,
    perm: Tensor,
    *,
    k: int,
) -> dict[str, float]:
    """Probe a kNN graph's low-frequency coordinates on its largest component.

    This opt-in diagnostic uses only binary graph edges and a small dense
    eigendecomposition for one 576-node image.  Four non-trivial normalized
    adjacency eigenvectors are linearly fitted to the true clean (row, col)
    coordinates.  It is deliberately an *affine probe*, not a proposed solver.
    """
    indices = _largest_component(neighbours, k)
    count = int(indices.size)
    result = {
        "spectral_component_fraction": count / NFRAG,
        "spectral_affine_r2": float("nan"),
    }
    if count < 8:
        return result

    # Build a binary, symmetrized adjacency on the largest component only.
    local_of_global = np.full(NFRAG, -1, dtype=np.int64)
    local_of_global[indices] = np.arange(count)
    adjacency = np.zeros((count, count), dtype=np.float64)
    full_neighbours = neighbours[:, :k].detach().cpu().numpy()
    for source in indices:
        local_source = local_of_global[source]
        for target in full_neighbours[source]:
            local_target = local_of_global[int(target)]
            if local_target >= 0:
                adjacency[local_source, local_target] = 1.0
                adjacency[local_target, local_source] = 1.0
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    if np.any(degree == 0.0):
        return result
    normalized = adjacency / np.sqrt(degree[:, None] * degree[None, :])
    _, eigenvectors = np.linalg.eigh(normalized)
    feature_count = min(4, count - 1)
    # The leading eigenvector is the trivial degree signal in a connected
    # component.  The immediately preceding vectors are the graph's slow modes.
    features = eigenvectors[:, -(feature_count + 1) : -1]
    rows, cols = clean_coordinates(perm.detach().cpu())
    target = torch.stack((rows, cols), dim=-1).numpy()[indices].astype(np.float64)
    target = (target - target.mean(axis=0, keepdims=True)) / target.std(axis=0, keepdims=True).clip(1e-8)
    design = np.concatenate((np.ones((count, 1), dtype=np.float64), features), axis=1)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ coefficients
    residual = np.square(target - prediction).sum(axis=0)
    total = np.square(target - target.mean(axis=0, keepdims=True)).sum(axis=0)
    result["spectral_affine_r2"] = float(np.mean(1.0 - residual / total.clip(1e-12)))
    return result


def _mean_metrics(totals: Mapping[str, float], count: int) -> dict[str, float]:
    if count < 1:
        raise ValueError("cannot average zero images")
    return {name: value / count for name, value in totals.items()}


def _format_value(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def print_report(
    label: str,
    metrics: Mapping[str, float],
    *,
    images: int,
) -> None:
    """Print a compact, side-by-side-readable evaluator report."""
    print(f"[{label}] exact synthetic images={images}")
    for radius in RADII:
        chunks = []
        for k in TOPK:
            chunks.append(
                f"K={k}: p={metrics[f'r{radius}_precision@{k}']:.4f} "
                f"r={metrics[f'r{radius}_recall@{k}']:.4f}"
            )
        print(f"  retrieval Cheb<={radius}: " + " | ".join(chunks))
    for k in TOPK:
        print(
            f"  graph K={k}: mean_cheb={metrics[f'graph_mean_cheb@{k}']:.3f} "
            f"edges(r1/r3/r4)="
            f"{metrics[f'graph_edge_r1@{k}']:.3f}/"
            f"{metrics[f'graph_edge_r3@{k}']:.3f}/"
            f"{metrics[f'graph_edge_r4@{k}']:.3f} "
            f"components={metrics[f'graph_components@{k}']:.1f} "
            f"largest={metrics[f'graph_largest_fraction@{k}']:.3f} "
            f"node_mean_component={metrics[f'graph_node_mean_component@{k}']:.1f}"
        )
def _chance_precision() -> dict[int, float]:
    """Expected random-neighbour precision for each clean-grid relation."""
    cells = torch.arange(NFRAG)
    rows = torch.div(cells, GRID, rounding_mode="floor")
    cols = torch.remainder(cells, GRID)
    distance = torch.maximum(
        (rows[:, None] - rows[None, :]).abs(),
        (cols[:, None] - cols[None, :]).abs(),
    )
    off_diagonal = ~torch.eye(NFRAG, dtype=torch.bool)
    return {
        radius: float((((distance <= radius) & off_diagonal).sum(dim=-1) / float(NFRAG - 1)).float().mean())
        for radius in RADII
    }


def _parse_device(value: str | None) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="MacroAffinityNet .pt checkpoint")
    parser.add_argument("--n", type=int, default=12, help="held-out synthetic images to evaluate")
    parser.add_argument("--device", default=None, help="cuda when available by default")
    parser.add_argument("--seed", type=int, default=SEED, help="seed for fresh synthetic corruptions")
    parser.add_argument(
        "--spectral",
        action="store_true",
        help="also run a small opt-in spectral affine-coordinate diagnostic",
    )
    parser.add_argument(
        "--spectral_k",
        type=int,
        default=15,
        choices=TOPK,
        help="kNN graph degree used by the optional spectral diagnostic",
    )
    parser.add_argument(
        "--spectral_images",
        type=int,
        default=1,
        help="at most this many images get the optional dense eigendecomposition",
    )
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    if args.spectral_images < 1:
        parser.error("--spectral_images must be positive")

    device = _parse_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model, checkpoint_metadata = load_model(args.ckpt, device)
    step = checkpoint_metadata.get("step") if isinstance(checkpoint_metadata, Mapping) else None
    print(
        f"device={device} checkpoint={os.path.abspath(args.ckpt)} "
        f"params={count_params(model):,}" + (f" step={step}" if step is not None else ""),
        flush=True,
    )
    train_names, val_names = train_val_split()
    del train_names
    if not val_names:
        raise RuntimeError("held-out validation split is empty")
    if args.n > len(val_names):
        parser.error(f"--n={args.n} exceeds held-out split size {len(val_names)}")
    dataset = CanvasDataset(val_names[: args.n], real_prob=0.0, seed=args.seed)

    raw_totals: defaultdict[str, float] = defaultdict(float)
    learned_totals: defaultdict[str, float] = defaultdict(float)
    raw_spectral_totals: defaultdict[str, float] = defaultdict(float)
    learned_spectral_totals: defaultdict[str, float] = defaultdict(float)
    spectral_seen = 0
    maximum_k = max(TOPK)

    for index in range(args.n):
        sample = dataset[index]
        if not bool(sample["has_perm"]):
            raise RuntimeError("evaluator requires exact synthetic CanvasDataset examples")
        tiles = sample["tiles"]
        perm = sample["perm"]
        distance_cpu = chebyshev_distance(perm)

        raw_affinity = raw_zpixel_affinity(tiles)
        raw_neighbours = top_neighbours(raw_affinity, maximum_k)
        raw_metrics = retrieval_metrics(distance_cpu, raw_neighbours, TOPK)
        raw_metrics.update(graph_metrics(distance_cpu, raw_neighbours, TOPK))
        for name, value in raw_metrics.items():
            raw_totals[name] += value

        affinity = learned_affinity(model, tiles, device)
        distance = distance_cpu.to(affinity.device)
        neighbours = top_neighbours(affinity, maximum_k)
        model_metrics = retrieval_metrics(distance, neighbours, TOPK)
        model_metrics.update(graph_metrics(distance, neighbours, TOPK))
        for name, value in model_metrics.items():
            learned_totals[name] += value

        if args.spectral and spectral_seen < args.spectral_images:
            raw_spectral = spectral_affine_r2(
                raw_neighbours, perm, k=args.spectral_k
            )
            learned_spectral = spectral_affine_r2(
                neighbours, perm, k=args.spectral_k
            )
            for name, value in raw_spectral.items():
                raw_spectral_totals[name] += value
            for name, value in learned_spectral.items():
                learned_spectral_totals[name] += value
            spectral_seen += 1

        print(f"processed {index + 1}/{args.n}", flush=True)

    raw_report = _mean_metrics(raw_totals, args.n)
    learned_report = _mean_metrics(learned_totals, args.n)
    if spectral_seen:
        raw_report.update(_mean_metrics(raw_spectral_totals, spectral_seen))
        learned_report.update(_mean_metrics(learned_spectral_totals, spectral_seen))

    chance = _chance_precision()
    print(
        "random-neighbour expected precision: "
        + " ".join(f"Cheb<={radius}: {chance[radius]:.4f}" for radius in RADII)
        + f"; expected recall@K = K/{NFRAG - 1}",
        flush=True,
    )
    print_report("raw_zpixel", raw_report, images=args.n)
    print_report("learned_affinity", learned_report, images=args.n)
    if spectral_seen:
        for label, metrics in (("raw_zpixel", raw_report), ("learned_affinity", learned_report)):
            print(
                f"[{label}] spectral K={args.spectral_k} images={spectral_seen} "
                f"largest_component={_format_value(metrics['spectral_component_fraction'])} "
                f"affine_coordinate_R2={_format_value(metrics['spectral_affine_r2'])}",
                flush=True,
            )


if __name__ == "__main__":
    main()
