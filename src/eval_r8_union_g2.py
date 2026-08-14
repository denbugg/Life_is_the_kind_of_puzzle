"""R8-G2: label-blind fixed-width union coverage of R8 and frozen rank96 candidates.

For each cached DEV input mosaic, R8 scores all directed tile pairs.  The four
scores are maximized only to make an undirected candidate *membership* list;
true directions remain hidden until post-hoc coverage.  A fixed rank-interleaved
fusion alternates frozen rank96 and R8 ranks, deduplicates without labels, and
stops at exactly 128 candidates per anchor.  This avoids score-scale tuning and
keeps active density directly comparable with the cached rank96 graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from train_r8_holistic_pair import GRID, NFRAG, HolisticPairNet, dense_scores

CACHE = Path(r"E:\pazzle_work\edge_confidence\full_graph_cache")
INPUTS = Path(r"E:\pazzle_data\train\inputs")
CKPT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R8_holistic_full_pair\g1_capacity_resume1500_retry1\r8_last.pt")
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R8_holistic_full_pair\g2_union_coverage")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_tiles(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        if im.mode != "RGB" or im.size != (480, 480):
            raise ValueError(f"expected strict RGB 480x480 mosaic, got {path}")
        rgb = np.asarray(im, dtype=np.uint8)
    tiles = rgb.reshape(GRID, 20, GRID, 20, 3).transpose(0, 2, 1, 3, 4).reshape(NFRAG, 20, 20, 3)
    return torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2).float() / 255.0


def coverage(permutation: np.ndarray, candidates: np.ndarray) -> Tuple[float, float]:
    if permutation.shape != (NFRAG,) or candidates.shape[0] != NFRAG:
        raise ValueError("coverage input shape mismatch")
    inverse = np.empty(NFRAG, dtype=np.int64)
    inverse[permutation] = np.arange(NFRAG, dtype=np.int64)
    hit = 0
    total = 0
    for source in range(NFRAG):
        row, col = divmod(int(permutation[source]), GRID)
        values = set(int(x) for x in candidates[source] if int(x) != source)
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            rr, cc = row + dr, col + dc
            if 0 <= rr < GRID and 0 <= cc < GRID:
                total += 1
                hit += int(int(inverse[rr * GRID + cc]) in values)
    density = float(np.mean([len(set(int(x) for x in row if int(x) != i)) for i, row in enumerate(candidates)]))
    return hit / max(1, total), density


def fuse_rank_interleaved(base: np.ndarray, r8: np.ndarray, width: int) -> np.ndarray:
    """Fixed label-blind rank interleaving, base then R8 at each rank."""
    if base.shape != (NFRAG, width) or r8.shape != (NFRAG, width):
        raise ValueError(f"both candidate lists must be ({NFRAG},{width})")
    fused = np.full((NFRAG, width), -1, dtype=np.int64)
    for anchor in range(NFRAG):
        chosen: List[int] = []
        seen = {anchor}
        for rank in range(width):
            for source in (base, r8):
                candidate = int(source[anchor, rank])
                if candidate not in seen:
                    chosen.append(candidate)
                    seen.add(candidate)
                    if len(chosen) == width:
                        break
            if len(chosen) == width:
                break
        if len(chosen) < width:
            # Dense R8 top-K cannot contain duplicates/self, but assert rather
            # than label-fill should a malformed frozen cache violate the contract.
            raise RuntimeError(f"could not fill fixed width for anchor {anchor}: {len(chosen)}")
        fused[anchor] = np.asarray(chosen, dtype=np.int64)
    return fused


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=CACHE)
    p.add_argument("--input-dir", type=Path, default=INPUTS)
    p.add_argument("--checkpoint", type=Path, default=CKPT)
    p.add_argument("--dev", default="image_0014_k64.npz,image_0020_k64.npz")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--pair-chunk", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--report", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.width != 128:
        raise ValueError("R8-G2 is pre-registered at active width 128")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    architecture = state.get("architecture")
    if not architecture:
        raise RuntimeError("R8 checkpoint lacks architecture")
    model = HolisticPairNet(**architecture).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    names = [item.strip() for item in args.dev.split(",") if item.strip()]
    if len(names) != 2:
        raise ValueError("R8-G2 requires the two pinned DEV caches")
    args.work.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for cache_name in names:
        cache_path = args.cache_dir / cache_name
        # These two filenames are the pre-registered pinned DEV cache contract.
        # Legacy graph archives do not embed a split_name scalar; provenance is
        # established by the immutable filename list plus their saved permutation.
        if cache_name not in {"image_0014_k64.npz", "image_0020_k64.npz"}:
            raise RuntimeError(f"R8-G2 rejects non-pinned cache: {cache_name}")
        with np.load(cache_path, allow_pickle=False) as z:
            permutation = np.asarray(z["permutation"], dtype=np.int64)
            base = np.asarray(z["candidate_ids"], dtype=np.int64)
        split = "dev_pinned_filename_contract"
        if base.shape != (NFRAG, args.width):
            raise RuntimeError(f"cached base candidates malformed: {base.shape}")
        # Mirror build_sgt2_visual_cache.py: image_0014_k64.npz maps to
        # the row-major raw input mosaic img_000014.png.
        cache_index = int(cache_name.removeprefix("image_").removesuffix("_k64.npz"))
        source_name = f"img_{cache_index:06d}.png"
        stem = source_name.removesuffix(".png")
        input_path = args.input_dir / source_name
        tiles = load_tiles(input_path).to(device)
        with torch.inference_mode():
            directed = dense_scores(model, tiles, pair_chunk=args.pair_chunk)
            pooled = directed.max(dim=0).values
            r8 = pooled.topk(k=args.width, dim=-1).indices.detach().cpu().numpy().astype(np.int64)
        fused = fuse_rank_interleaved(base, r8, args.width)
        base_cov, base_density = coverage(permutation, base)
        r8_cov, r8_density = coverage(permutation, r8)
        union_cov, union_density = coverage(permutation, fused)
        np.savez_compressed(args.work / f"{stem}_r8g2_candidates.npz", r8_candidates=r8, union_candidates=fused)
        rows.append({
            "name": cache_name, "split": split, "input": str(input_path), "base_coverage": base_cov,
            "r8_coverage": r8_cov, "union_coverage": union_cov, "base_density": base_density,
            "r8_density": r8_density, "union_density": union_density,
        })
        print(json.dumps(rows[-1]), flush=True)
    mean_base = float(np.mean([float(row["base_coverage"]) for row in rows]))
    mean_r8 = float(np.mean([float(row["r8_coverage"]) for row in rows]))
    mean_union = float(np.mean([float(row["union_coverage"]) for row in rows]))
    min_density = float(min(float(row["union_density"]) for row in rows))
    pass_gate = mean_union >= 0.73 and min_density >= 128.0
    report = {
        "experiment": "R8_holistic_full_pair_G2_union_coverage",
        "gate": {"coverage_threshold": 0.73, "active_width": 128, "density_requirement": 128.0},
        "protocol": {
            "fusion": "label_blind_rank_interleaving_base_then_R8_per_rank_deduplicate_stop_at_128",
            "r8_membership": "max_over_4_directional_full_pair_scores_then_top128",
            "tile_order": "row_major_input_mosaic_24x24_no_rotation",
            "cached_sources": names,
        },
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint), "step": int(state.get("step", -1))},
        "rows": rows,
        "summary": {"mean_base_coverage": mean_base, "mean_r8_coverage": mean_r8, "mean_union_coverage": mean_union, "min_union_density": min_density, "passes_G2": pass_gate},
        "decision": "advance_to_R8_G3" if pass_gate else "reject_R8_before_layout",
    }
    destination = args.report or args.work / "r8_g2_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
