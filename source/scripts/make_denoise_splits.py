#!/usr/bin/env python3
"""Create leakage-resistant source-image splits for denoising experiments.

Every test filename is excluded from all training/evaluation source targets. The
remaining target images are grouped by exact hash and conservative perceptual
near-duplicate checks before clusters are assigned to train/val/audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from tqdm import tqdm


@dataclass
class Signature:
    name: str
    sha256: str
    phash: np.uint64
    thumb: np.ndarray


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def signature(path: Path) -> Signature:
    data = path.read_bytes()
    rgb = Image.open(path).convert("RGB")
    gray = np.asarray(rgb.resize((32, 32), Image.Resampling.LANCZOS).convert("L"), dtype=np.float32)
    coeff = dctn(gray, norm="ortho")[:8, :8].reshape(-1)
    bits = coeff[1:] > np.median(coeff[1:])
    value = np.uint64(0)
    for bit in bits:
        value = (value << np.uint64(1)) | np.uint64(bit)
    thumb = np.asarray(rgb.resize((16, 16), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    return Signature(path.name, hashlib.sha256(data).hexdigest(), value, thumb)


def hamming_candidates(hashes: np.ndarray, max_distance: int, chunk: int = 256):
    popcount = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    n = len(hashes)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        xor = np.bitwise_xor(hashes[start:stop, None], hashes[None, :])
        distances = popcount[xor.view(np.uint8).reshape(stop - start, n, 8)].sum(axis=2)
        for local_i, row in enumerate(distances):
            i = start + local_i
            js = np.flatnonzero((row <= max_distance) & (np.arange(n) > i))
            for j in js:
                yield i, int(j), int(row[j])


def assign_clusters(clusters: list[list[str]], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    rng.shuffle(clusters)
    clusters.sort(key=len, reverse=True)
    desired = {"train": 4900, "val": 700, "audit": 700}
    result = {key: [] for key in desired}
    for cluster in clusters:
        deficits = {key: desired[key] - len(result[key]) for key in desired}
        destination = max(deficits, key=lambda key: (deficits[key], rng.random()))
        result[destination].extend(cluster)
    for names in result.values():
        names.sort()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("puzzle"))
    parser.add_argument("--out", type=Path, default=Path("configs/denoise_splits_seed20260710.json"))
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--thumb-mse", type=float, default=2.5e-4)
    args = parser.parse_args()

    target_dir = args.data_root / "train" / "targets"
    test_dir = args.data_root / "test"
    target_names = sorted(path.name for path in target_dir.glob("*.png"))
    test_names = sorted(path.name for path in test_dir.glob("*.png"))
    excluded = sorted(set(target_names) & set(test_names))
    eligible = sorted(set(target_names) - set(excluded))
    if len(target_names) != 7000 or len(test_names) != 700 or len(excluded) != 700:
        raise SystemExit(
            f"unexpected counts targets={len(target_names)} test={len(test_names)} overlap={len(excluded)}"
        )

    signatures = [signature(target_dir / name) for name in tqdm(eligible, desc="signatures")]
    uf = UnionFind(len(signatures))

    exact: dict[str, int] = {}
    for i, sig in enumerate(signatures):
        if sig.sha256 in exact:
            uf.union(i, exact[sig.sha256])
        else:
            exact[sig.sha256] = i

    hashes = np.asarray([sig.phash for sig in signatures], dtype=np.uint64)
    near_pairs = []
    for i, j, distance in tqdm(
        hamming_candidates(hashes, args.phash_distance),
        desc="near-duplicate candidates",
        total=None,
    ):
        mse = float(np.mean((signatures[i].thumb - signatures[j].thumb) ** 2))
        if mse <= args.thumb_mse:
            uf.union(i, j)
            near_pairs.append(
                {
                    "a": signatures[i].name,
                    "b": signatures[j].name,
                    "phash_distance": distance,
                    "thumb_mse": mse,
                }
            )

    grouped: dict[int, list[str]] = {}
    for i, sig in enumerate(signatures):
        grouped.setdefault(uf.find(i), []).append(sig.name)
    clusters = [sorted(names) for names in grouped.values()]
    splits = assign_clusters(clusters, args.seed)

    payload = {
        "schema_version": 1,
        "seed": args.seed,
        "data_root": str(args.data_root),
        "policy": {
            "exclude_all_test_filename_overlaps": True,
            "phash_distance": args.phash_distance,
            "thumb_mse": args.thumb_mse,
            "split_unit": "perceptual_duplicate_cluster",
        },
        "counts": {
            "targets": len(target_names),
            "test": len(test_names),
            "excluded_test_overlap": len(excluded),
            "eligible": len(eligible),
            "clusters": len(clusters),
            "non_singleton_clusters": sum(len(cluster) > 1 for cluster in clusters),
            **{key: len(value) for key, value in splits.items()},
        },
        "excluded_test_overlap": excluded,
        "near_duplicate_pairs": near_pairs,
        "splits": splits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
