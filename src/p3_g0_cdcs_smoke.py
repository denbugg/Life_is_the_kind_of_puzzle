"""P3 G0: FIT-only CDCS cache and label-contract smoke test.

Constructs rank96-derived 32-way directional hard lists on four synthetic
per-tile-corrupted FIT bags.  No CAL/DEV/test files, targets, layouts, or
solver calls are reachable from this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

import infer_rank96 as rank96
from eval_candidate_rank import score_full_graph
from train_eval_cb1_g1_capacity import distort_frags, load_rgb, sha256_file, to_frags
from train_offset_pose import mine_affinity_candidates

GRID = 24
N = GRID * GRID
FIT_TARGETS = Path(r"E:\pazzle_data\train\targets")
SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P3_CDCS\g0_smoke")
SOURCE_ROOT = Path(r"C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", type=Path, default=FIT_TARGETS)
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--sources", type=int, default=4)
    return p.parse_args()


def digest_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def config() -> rank96.InferenceConfig:
    return rank96.InferenceConfig(
        input_dir=Path(r"E:\pazzle_data\train\inputs"), output_dir=WORK / "unused",
        output_zip=None,
        ranker_checkpoint=SOURCE_ROOT / "artifacts" / "candidate_rank" / "rank_v2w64_best.pt",
        affinity_primary_checkpoint=SOURCE_ROOT / "artifacts" / "macro_affinity" / "affinity_r1_1200_best.pt",
        affinity_secondary_checkpoint=SOURCE_ROOT / "artifacts" / "macro_affinity" / "affinity_r3_1000_best.pt",
        device="cuda",
    )


def neighbour(slot: int, direction: int) -> int | None:
    row, col = divmod(slot, GRID)
    if direction == 0:
        return None if col == GRID - 1 else slot + 1
    if direction == 1:
        return None if row == GRID - 1 else slot + GRID
    if direction == 2:
        return None if col == 0 else slot - 1
    if direction == 3:
        return None if row == 0 else slot - GRID
    raise ValueError(direction)


def build_lists(candidates: np.ndarray, valid: np.ndarray, scores: np.ndarray, permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (anchors, directions, IDs), positives forcibly at list index zero."""
    if candidates.shape != (N, 128) or valid.shape != (N, 128):
        raise ValueError((candidates.shape, valid.shape))
    if scores.shape != (4, N, 128):
        raise ValueError(scores.shape)
    inverse = np.empty(N, dtype=np.int32)
    inverse[permutation] = np.arange(N, dtype=np.int32)
    anchors: list[int] = []
    directions: list[int] = []
    rows: list[np.ndarray] = []
    for anchor in range(N):
        source_slot = int(permutation[anchor])
        for direction in range(4):
            true_slot = neighbour(source_slot, direction)
            if true_slot is None:
                continue
            positive = int(inverse[true_slot])
            eligible = valid[anchor] & np.isfinite(scores[direction, anchor]) & (candidates[anchor] != anchor) & (candidates[anchor] != positive)
            candidates_d = candidates[anchor, eligible].astype(np.int32, copy=False)
            scores_d = scores[direction, anchor, eligible]
            order = np.argsort(-scores_d, kind="stable")
            hard = candidates_d[order][:31]
            if hard.shape != (31,) or np.unique(hard).size != 31 or positive in hard:
                raise RuntimeError((anchor, direction, positive, hard.shape, np.unique(hard).size))
            members = np.concatenate((np.asarray([positive], dtype=np.int32), hard))
            if np.unique(members).size != 32 or anchor in members:
                raise RuntimeError((anchor, direction, members))
            anchors.append(anchor)
            directions.append(direction)
            rows.append(members)
    a = np.asarray(anchors, dtype=np.int16)
    d = np.asarray(directions, dtype=np.int8)
    members = np.stack(rows).astype(np.int16, copy=False)
    expected = 4 * GRID * (GRID - 1)
    if a.shape != (expected,) or d.shape != (expected,) or members.shape != (expected, 32):
        raise RuntimeError((a.shape, d.shape, members.shape, expected))
    return a, d, members


def main() -> None:
    cfg = args()
    if cfg.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("P3 G0 requires one local CUDA GPU")
    if cfg.sources != 4:
        raise ValueError("P3 G0 is pre-registered at exactly four FIT sources")
    if not cfg.split.is_file():
        raise FileNotFoundError(cfg.split)
    split = json.loads(cfg.split.read_text(encoding="utf-8"))
    fit = list(split["splits"]["fit"])
    cal = set(split["splits"]["cal"])
    dev = set(split["splits"]["dev"])
    if len(fit) != 5360 or set(fit) & (cal | dev):
        raise RuntimeError("invalid pinned source-disjoint split")
    names = fit[:cfg.sources]
    if any(name in cal or name in dev for name in names):
        raise RuntimeError("non-FIT source selected")
    for name in names:
        if not (cfg.targets / name).is_file():
            raise FileNotFoundError(cfg.targets / name)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    rank96._set_deterministic_runtime(cfg.seed, device)
    models = rank96.load_models(config(), device)
    all_source = []
    all_anchor = []
    all_direction = []
    all_members = []
    all_permutation = []
    per_source = []
    for source_index, name in enumerate(names):
        clean = load_rgb(cfg.targets / name)
        fragments = distort_frags(to_frags(clean), np.random.default_rng(cfg.seed * 1009 + source_index))
        permutation = np.random.default_rng(cfg.seed * 2029 + source_index).permutation(N).astype(np.int32)
        bag = fragments[permutation]
        tile_tensor = torch.from_numpy(bag).permute(0, 3, 1, 2).contiguous().float().to(device)
        with torch.no_grad():
            candidates, valid = mine_affinity_candidates(
                models.affinity_primary, tile_tensor.unsqueeze(0), candidate_k=64, device=device,
                affinity_secondary=models.affinity_secondary,
            )
            scores = score_full_graph(models.ranker, tile_tensor, candidates[0], valid[0], pair_batch=4096, device=device)
        anchors, directions, members = build_lists(
            candidates[0].detach().cpu().numpy(), valid[0].detach().cpu().numpy(), scores.detach().cpu().numpy(), permutation,
        )
        all_source.append(np.full(anchors.shape[0], source_index, dtype=np.int8))
        all_anchor.append(anchors); all_direction.append(directions); all_members.append(members); all_permutation.append(permutation)
        per_source.append({
            "source": name, "target_sha256": sha256_file(cfg.targets / name), "queries": int(anchors.size),
            "candidate_shape": list(candidates.shape), "valid_shape": list(valid.shape), "score_shape": list(scores.shape),
            "lists_sha256": digest_array(members), "permutation_sha256": digest_array(permutation),
        })
    source_index = np.concatenate(all_source)
    anchors = np.concatenate(all_anchor)
    directions = np.concatenate(all_direction)
    members = np.concatenate(all_members)
    permutations = np.stack(all_permutation)
    if members.shape != (cfg.sources * 4 * GRID * (GRID - 1), 32):
        raise RuntimeError(members.shape)
    cfg.work.mkdir(parents=True, exist_ok=True)
    artifact = cfg.work / "p3_g0_fit_hardlists.npz"
    np.savez_compressed(
        artifact, source_names=np.asarray(names), source_index=source_index, anchors=anchors, directions=directions,
        members=members, permutations=permutations, positive_index=np.zeros(members.shape[0], dtype=np.int8), seed=np.int64(cfg.seed),
    )
    report = {
        "experiment": "P3_CDCS", "gate": "G0_FIT_only_contract_smoke", "decision": "pass_to_G1_capacity",
        "fit_sources": per_source, "fit_source_count": len(names), "split": str(cfg.split), "split_sha256": sha256_file(cfg.split),
        "teacher": {"ranker": str(config().ranker_checkpoint), "ranker_sha256": sha256_file(config().ranker_checkpoint),
                    "affinity_primary": str(config().affinity_primary_checkpoint), "affinity_primary_sha256": sha256_file(config().affinity_primary_checkpoint),
                    "affinity_secondary": str(config().affinity_secondary_checkpoint), "affinity_secondary_sha256": sha256_file(config().affinity_secondary_checkpoint)},
        "hardlist_artifact": str(artifact), "hardlist_artifact_sha256": sha256_file(artifact),
        "hardlist_shape": list(members.shape), "hardlist_ids_sha256": digest_array(members), "positive_index": 0,
        "uniqueness_verified": True, "queries_per_source": int(4 * GRID * (GRID - 1)), "corruption_seed": cfg.seed,
        "CAL_target_opened": False, "DEV_targets_opened": False, "test_accessed": False, "layouts_assembled": False, "restorer_used": False,
    }
    destination = cfg.work / "p3_g0_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
