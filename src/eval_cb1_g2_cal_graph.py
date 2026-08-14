"""P1/CB1 G2: target-safe CAL candidate-extension coverage.

Uses one pre-existing raw candidate cache and its input mosaic. CB1 scores the
label-blind union of frozen cache membership and directional L1 candidates; only
after every list is frozen is cache permutation metadata used for coverage.
No target image, layout, restorer, solver or submission is accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from infer_rank96 import load_rgb_strict, split_upright_tiles
from train_eval_cb1_g1_capacity import BoundaryBuddyNet, pair_band, sha256_file

GRID = 24
NFRAG = 576
INPUT = Path(r"E:\pazzle_data\train\inputs\img_000051.png")
CACHE = Path(r"E:\pazzle_work\edge_confidence\full_graph_cache\image_0051_k64.npz")
CHECKPOINT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\full_fit\cb1_full_fit.pt")
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\g2_cal_graph")


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def directional_l1_order(tiles: np.ndarray, anchor: int, direction: int) -> np.ndarray:
    if direction == 0:  # anchor right candidate
        diff = np.abs(tiles[anchor, :, -2:, :][None].astype(np.float32) - tiles[:, :, :2, :].astype(np.float32))
    elif direction == 1:  # anchor down candidate
        diff = np.abs(tiles[anchor, -2:, :, :][None].astype(np.float32) - tiles[:, :2, :, :].astype(np.float32))
    elif direction == 2:  # candidate right of source must equal anchor
        diff = np.abs(tiles[:, :, -2:, :].astype(np.float32) - tiles[anchor, :, :2, :][None].astype(np.float32))
    elif direction == 3:  # candidate below source must equal anchor
        diff = np.abs(tiles[:, -2:, :, :].astype(np.float32) - tiles[anchor, :2, :, :][None].astype(np.float32))
    else:
        raise ValueError(direction)
    scores = diff.mean(axis=(1, 2, 3))
    scores[anchor] = np.inf
    return np.argsort(scores, kind="stable")


def score_candidates(model: BoundaryBuddyNet, tiles: np.ndarray, anchor: int, candidates: list[int], direction: int, device: torch.device, chunk: int) -> np.ndarray:
    bands = np.stack([pair_band(tiles, anchor, candidate, direction) for candidate in candidates]).astype(np.float32)
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(candidates), chunk):
            x = torch.from_numpy(bands[start:start + chunk]).to(device)
            values.append(model(x).detach().cpu().numpy())
    return np.concatenate(values)


def membership_coverage(permutation: np.ndarray, candidates: np.ndarray) -> tuple[float, float, int, int]:
    if permutation.shape != (NFRAG,) or candidates.shape[0] != NFRAG:
        raise ValueError("malformed permutation or candidate matrix")
    inverse = np.empty((NFRAG,), dtype=np.int64)
    inverse[permutation] = np.arange(NFRAG, dtype=np.int64)
    hits = total = 0
    for source in range(NFRAG):
        row, col = divmod(int(permutation[source]), GRID)
        actual = set()
        for rr, cc in ((row, col + 1), (row + 1, col), (row, col - 1), (row - 1, col)):
            if 0 <= rr < GRID and 0 <= cc < GRID:
                actual.add(int(inverse[rr * GRID + cc]))
        offered = set(int(x) for x in candidates[source] if int(x) >= 0 and int(x) != source)
        hits += len(actual & offered)
        total += len(actual)
    density = float(np.mean([len(set(int(x) for x in row if int(x) >= 0 and int(x) != i)) for i, row in enumerate(candidates)]))
    return hits / total, density, hits, total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=INPUT)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--l1-width", type=int, default=128)
    p.add_argument("--cb1-width", type=int, default=32)
    p.add_argument("--pair-chunk", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    if (cfg.l1_width, cfg.cb1_width, cfg.pair_chunk) != (128, 32, 4096):
        raise ValueError("CB1 G2 is frozen at L1-width=128, CB1-width=32, pair-chunk=4096")
    for path in (cfg.input, cfg.cache, cfg.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(json.dumps({"stage": "open_frozen_cache"}), flush=True)
    with np.load(cfg.cache, allow_pickle=False) as cache:
        if "candidate_ids" not in cache.files or "permutation" not in cache.files:
            raise ValueError("cache lacks frozen candidate_ids/permutation metadata")
        base = np.asarray(cache["candidate_ids"], dtype=np.int64).copy()
        permutation = np.asarray(cache["permutation"], dtype=np.int64).copy()
    if base.shape != (NFRAG, 128) or permutation.shape != (NFRAG,):
        raise ValueError(f"unexpected cache shapes base={base.shape} permutation={permutation.shape}")
    if np.any(base < 0) or np.any(base >= NFRAG) or np.any(base == np.arange(NFRAG)[:, None]):
        raise ValueError("frozen candidate cache contains invalid identifiers")
    print(json.dumps({"stage": "load_raw_input_and_model"}), flush=True)
    image = load_rgb_strict(cfg.input)
    tiles = split_upright_tiles(image)
    state = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    model = BoundaryBuddyNet().to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    print(json.dumps({"stage": "score_candidate_pools", "anchors": NFRAG}), flush=True)
    cb1 = np.full((NFRAG, 4, cfg.cb1_width), -1, dtype=np.int64)
    for anchor in range(NFRAG):
        if anchor % 48 == 0:
            print(json.dumps({"stage": "score_candidate_pools", "anchor": anchor}), flush=True)
        frozen = [int(x) for x in base[anchor]]
        for direction in range(4):
            l1 = [int(x) for x in directional_l1_order(tiles, anchor, direction)[:cfg.l1_width]]
            candidates = list(dict.fromkeys(frozen + l1))
            if anchor in candidates or len(candidates) < cfg.cb1_width:
                raise RuntimeError(f"malformed label-blind candidate pool for anchor={anchor} direction={direction}")
            scores = score_candidates(model, tiles, anchor, candidates, direction, device, cfg.pair_chunk)
            selected = np.argsort(-scores, kind="stable")[:cfg.cb1_width]
            cb1[anchor, direction] = np.asarray([candidates[int(i)] for i in selected], dtype=np.int64)
    print(json.dumps({"stage": "freeze_union_and_measure"}), flush=True)
    union_rows: list[np.ndarray] = []
    for anchor in range(NFRAG):
        members = list(dict.fromkeys([int(x) for x in base[anchor]] + [int(x) for x in cb1[anchor].reshape(-1)]))
        if anchor in members or len(members) < 128:
            raise RuntimeError(f"malformed union candidate list for anchor={anchor}")
        union_rows.append(np.asarray(members, dtype=np.int64))
    max_width = max(len(row) for row in union_rows)
    union = np.full((NFRAG, max_width), -1, dtype=np.int64)
    for anchor, row in enumerate(union_rows):
        union[anchor, :len(row)] = row
    base_cov, base_density, base_hits, total = membership_coverage(permutation, base)
    cb1_flat = cb1.reshape(NFRAG, -1)
    cb1_cov, cb1_density, cb1_hits, _ = membership_coverage(permutation, cb1_flat)
    union_cov, union_density, union_hits, _ = membership_coverage(permutation, union)
    cfg.work.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.work / "cb1_g2_lists.npz", cb1_candidates=cb1, union_candidates=union)
    report = {
        "experiment": "P1_CB1_boundary_buddies", "gate": "G2_target_safe_CAL_candidate_graph",
        "cache": str(cfg.cache), "input": str(cfg.input), "checkpoint": str(cfg.checkpoint),
        "cache_sha256": sha256_file(cfg.cache), "input_sha256": sha256_file(cfg.input), "checkpoint_sha256": sha256_file(cfg.checkpoint),
        "base_candidate_ids_sha256": sha256_array(base), "cb1_candidate_ids_sha256": sha256_array(cb1), "union_candidate_ids_sha256": sha256_array(union),
        "candidate_construction": {"frozen_width": 128, "directional_l1_width": cfg.l1_width, "cb1_width_per_direction": cfg.cb1_width, "directions": ["right", "down", "left", "up"]},
        "coverage": {
            "total_directed_neighbours": total,
            "base": {"coverage": base_cov, "density": base_density, "hits": base_hits},
            "cb1_only": {"coverage": cb1_cov, "density": cb1_density, "hits": cb1_hits},
            "base_union_cb1": {"coverage": union_cov, "density": union_density, "hits": union_hits, "delta_vs_base": union_cov - base_cov},
        },
        "target_images_opened": False, "cache_labels_opened": False, "layouts_assembled": False, "restorer_used": False, "test_accessed": False,
    }
    report["passes_G2"] = bool((union_cov - base_cov) >= 0.02)
    report["decision"] = "advance_to_CB1_G3_DEV_graph" if report["passes_G2"] else "reject_CB1_before_DEV"
    destination = cfg.report or cfg.work / "cb1_g2_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
