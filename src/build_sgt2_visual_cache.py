"""Build a manifest-aligned visual cache for SGT2 from frozen candidate graphs.

The adapter uses only corrupted TRAIN_INP mosaics and pre-existing graph-cache
labels.  It never opens targets or test images.  Every artifact records source
hashes, tile-order contract and source-disjoint split membership.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

GRID = 24
TILE = 20
IMAGE = GRID * TILE
NAME_RE = re.compile(r"^image_(\d+)_k64\.npz$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        if source.mode != "RGB" or source.size != (IMAGE, IMAGE):
            raise ValueError(f"expected strict 480x480 RGB input: {path}, got {source.mode} {source.size}")
        return np.ascontiguousarray(np.asarray(source, dtype=np.uint8))


def split_tiles(image: np.ndarray) -> np.ndarray:
    if image.shape != (IMAGE, IMAGE, 3) or image.dtype != np.uint8:
        raise ValueError(f"invalid RGB image: {image.shape} {image.dtype}")
    # Tile index is the input mosaic's row-major physical tile position, which is
    # the same indexing used by rank96 split_upright_tiles and graph caches.
    return np.ascontiguousarray(image.reshape(GRID, TILE, GRID, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(GRID * GRID, TILE, TILE, 3))


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.stem + ".", suffix=".npz", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        np.savez_compressed(temp, **arrays)
        # numpy appends .npz only when passed a string, not an open handle.
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-cache-dir", type=Path, default=Path(r"E:\pazzle_work\edge_confidence\full_graph_cache"))
    parser.add_argument("--input-dir", type=Path, default=Path(r"E:\pazzle_data\train\inputs"))
    parser.add_argument("--split", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json"))
    parser.add_argument("--out-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\SGT2_visual_graph\visual_cache"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split_payload = json.loads(args.split.read_text(encoding="utf-8"))
    membership: dict[str, str] = {}
    for label, names in split_payload["splits"].items():
        for name in names:
            if name in membership:
                raise RuntimeError(f"duplicate source name in split manifest: {name}")
            membership[name] = label
    caches = sorted(args.graph_cache_dir.glob("*.npz"))
    if not caches:
        raise FileNotFoundError(f"no graph caches in {args.graph_cache_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for cache_path in caches:
        match = NAME_RE.fullmatch(cache_path.name)
        if not match:
            raise ValueError(f"unexpected graph cache filename: {cache_path.name}")
        source_name = f"img_{int(match.group(1)):06d}.png"
        input_path = args.input_dir / source_name
        if not input_path.is_file():
            raise FileNotFoundError(f"mapped input unavailable for {cache_path.name}: {input_path}")
        if source_name not in membership:
            raise RuntimeError(f"mapped input absent from pinned split: {source_name}")
        destination = args.out_dir / cache_path.name
        input_hash = sha256(input_path)
        cache_hash = sha256(cache_path)
        if destination.is_file() and not args.overwrite:
            with np.load(destination, allow_pickle=False) as old:
                old_input = str(old["input_sha256"].item())
                old_graph = str(old["graph_sha256"].item())
                if old_input == input_hash and old_graph == cache_hash:
                    rows.append({"graph_cache": cache_path.name, "source_name": source_name, "split": membership[source_name], "visual_cache": destination.name, "status": "reused", "input_sha256": input_hash, "graph_sha256": cache_hash})
                    continue
            raise RuntimeError(f"existing visual cache has mismatched provenance: {destination}; use --overwrite")
        image = load_rgb(input_path)
        tiles = split_tiles(image)
        with np.load(cache_path, allow_pickle=False) as graph:
            required = ("permutation", "candidate_ids", "candidate_scores", "features", "labels", "anchors", "directions", "predicted")
            missing = [key for key in required if key not in graph.files]
            if missing:
                raise ValueError(f"graph cache missing fields {missing}: {cache_path}")
            if graph["permutation"].shape != (GRID * GRID,) or graph["candidate_ids"].shape[0] != GRID * GRID or graph["candidate_scores"].shape[0] != GRID * GRID * 4:
                raise ValueError(f"unexpected graph tile/query shapes: {cache_path}")
            arrays = {key: np.ascontiguousarray(graph[key]) for key in required}
        atomic_npz(
            destination,
            tiles_rgb=tiles,
            source_name=np.asarray(source_name),
            split_name=np.asarray(membership[source_name]),
            input_sha256=np.asarray(input_hash),
            graph_sha256=np.asarray(cache_hash),
            tile_order=np.asarray("row_major_input_mosaic_24x24_no_rotation"),
            **arrays,
        )
        rows.append({"graph_cache": cache_path.name, "source_name": source_name, "split": membership[source_name], "visual_cache": destination.name, "status": "created", "input_sha256": input_hash, "graph_sha256": cache_hash})
        print(json.dumps(rows[-1]), flush=True)

    summary = {label: sum(row["split"] == label for row in rows) for label in sorted(set(row["split"] for row in rows))}
    manifest = {
        "experiment": "SGT2_visual_cache_G0",
        "scope": "corrupted train-input pixels plus frozen graph caches only; no train targets; no test data",
        "split": str(args.split),
        "split_sha256": sha256(args.split),
        "input_dir": str(args.input_dir),
        "graph_cache_dir": str(args.graph_cache_dir),
        "out_dir": str(args.out_dir),
        "tile_order": "row_major_input_mosaic_24x24_no_rotation",
        "rows": rows,
        "summary_by_split": summary,
    }
    destination = args.manifest or args.out_dir / "visual_cache_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(destination), "summary_by_split": summary, "count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
