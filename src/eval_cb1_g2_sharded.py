"""P1/CB1 G2 deterministic shard scorer and post-freeze coverage assembler."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval_cb1_g2_cal_graph import (
    CACHE, CHECKPOINT, INPUT, NFRAG, WORK, BoundaryBuddyNet,
    directional_l1_order, membership_coverage, score_candidates, sha256_array,
)
from infer_rank96 import load_rgb_strict, split_upright_tiles
from train_eval_cb1_g1_capacity import sha256_file

SHARDS = ((0, 144), (144, 288), (288, 432), (432, 576))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("score", "assemble"), required=True)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--input", type=Path, default=INPUT)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def shard_path(work: Path, start: int, end: int) -> Path:
    return work / f"cb1_g2_shard_{start:03d}_{end:03d}.npz"


def score(cfg: argparse.Namespace) -> None:
    if (cfg.start, cfg.end) not in SHARDS:
        raise ValueError(f"shard {(cfg.start, cfg.end)} is not pre-registered")
    for path in (cfg.input, cfg.cache, cfg.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    with np.load(cfg.cache, allow_pickle=False) as cache:
        base = np.asarray(cache["candidate_ids"], dtype=np.int64).copy()
    if base.shape != (NFRAG, 128) or np.any(base < 0) or np.any(base >= NFRAG) or np.any(base == np.arange(NFRAG)[:, None]):
        raise ValueError("malformed frozen base candidate cache")
    tiles = split_upright_tiles(load_rgb_strict(cfg.input))
    state = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    model = BoundaryBuddyNet().to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    rows = np.full((cfg.end - cfg.start, 4, 32), -1, dtype=np.int64)
    with torch.no_grad():
        for local, anchor in enumerate(range(cfg.start, cfg.end)):
            print(json.dumps({"mode": "score", "start": cfg.start, "end": cfg.end, "anchor": anchor}), flush=True)
            frozen = [int(x) for x in base[anchor]]
            for direction in range(4):
                l1 = [int(x) for x in directional_l1_order(tiles, anchor, direction)[:128]]
                candidates = list(dict.fromkeys(frozen + l1))
                if anchor in candidates or len(candidates) < 32:
                    raise RuntimeError(f"malformed candidate pool {anchor}/{direction}")
                scores = score_candidates(model, tiles, anchor, candidates, direction, device, 4096)
                pick = np.argsort(-scores, kind="stable")[:32]
                rows[local, direction] = np.asarray([candidates[int(i)] for i in pick], dtype=np.int64)
    cfg.work.mkdir(parents=True, exist_ok=True)
    out = shard_path(cfg.work, cfg.start, cfg.end)
    np.savez_compressed(out, anchors=np.arange(cfg.start, cfg.end, dtype=np.int64), cb1_candidates=rows)
    print(json.dumps({"mode": "score", "shard": str(out), "sha256": sha256_file(out)}), flush=True)


def assemble(cfg: argparse.Namespace) -> None:
    if cfg.start is not None or cfg.end is not None:
        raise ValueError("assemble has no start/end")
    if not cfg.cache.is_file():
        raise FileNotFoundError(cfg.cache)
    with np.load(cfg.cache, allow_pickle=False) as cache:
        base = np.asarray(cache["candidate_ids"], dtype=np.int64).copy()
        permutation = np.asarray(cache["permutation"], dtype=np.int64).copy()
    cb1 = np.full((NFRAG, 4, 32), -1, dtype=np.int64)
    receipts = []
    for start, end in SHARDS:
        path = shard_path(cfg.work, start, end)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as part:
            anchors = np.asarray(part["anchors"], dtype=np.int64)
            rows = np.asarray(part["cb1_candidates"], dtype=np.int64)
        expected = np.arange(start, end, dtype=np.int64)
        if not np.array_equal(anchors, expected) or rows.shape != (end - start, 4, 32):
            raise ValueError(f"invalid shard {path}")
        cb1[start:end] = rows
        receipts.append({"range": [start, end], "path": str(path), "sha256": sha256_file(path)})
    union_rows = []
    for anchor in range(NFRAG):
        members = list(dict.fromkeys([int(x) for x in base[anchor]] + [int(x) for x in cb1[anchor].reshape(-1)]))
        if anchor in members or not members:
            raise RuntimeError(f"invalid union at {anchor}")
        union_rows.append(np.asarray(members, dtype=np.int64))
    width = max(len(row) for row in union_rows)
    union = np.full((NFRAG, width), -1, dtype=np.int64)
    for anchor, row in enumerate(union_rows):
        union[anchor, :len(row)] = row
    base_cov, base_density, base_hits, total = membership_coverage(permutation, base)
    cb1_cov, cb1_density, cb1_hits, _ = membership_coverage(permutation, cb1.reshape(NFRAG, -1))
    union_cov, union_density, union_hits, _ = membership_coverage(permutation, union)
    cfg.work.mkdir(parents=True, exist_ok=True)
    lists = cfg.work / "cb1_g2_lists.npz"
    np.savez_compressed(lists, cb1_candidates=cb1, union_candidates=union)
    report = {
        "experiment": "P1_CB1_boundary_buddies", "gate": "G2_target_safe_CAL_candidate_graph",
        "execution": {"shards": receipts, "candidate_lists_sha256": sha256_file(lists)},
        "cache": str(cfg.cache), "input": str(cfg.input), "checkpoint": str(cfg.checkpoint),
        "cache_sha256": sha256_file(cfg.cache), "input_sha256": sha256_file(cfg.input), "checkpoint_sha256": sha256_file(cfg.checkpoint),
        "base_candidate_ids_sha256": sha256_array(base), "cb1_candidate_ids_sha256": sha256_array(cb1), "union_candidate_ids_sha256": sha256_array(union),
        "coverage": {"total_directed_neighbours": total, "base": {"coverage": base_cov, "density": base_density, "hits": base_hits}, "cb1_only": {"coverage": cb1_cov, "density": cb1_density, "hits": cb1_hits}, "base_union_cb1": {"coverage": union_cov, "density": union_density, "hits": union_hits, "delta_vs_base": union_cov - base_cov}},
        "target_images_opened": False, "cache_labels_opened": False, "layouts_assembled": False, "restorer_used": False, "test_accessed": False,
    }
    report["passes_G2"] = bool((union_cov - base_cov) >= 0.02)
    report["decision"] = "advance_to_CB1_G3_DEV_graph" if report["passes_G2"] else "reject_CB1_before_DEV"
    destination = cfg.report or cfg.work / "cb1_g2_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    cfg = parse_args()
    if cfg.mode == "score":
        score(cfg)
    else:
        assemble(cfg)


if __name__ == "__main__":
    main()
