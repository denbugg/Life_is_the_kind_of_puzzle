"""P3 G1: FIT-only listwise CDCS capacity test.

Phase `prepare` caches one deterministic challenge-matched corrupted bag and
rank96-derived 32-way hard lists for 96 FIT train and 32 held-out FIT sources.
Phase `train` optimizes a directional band CNN with InfoNCE over the exact
cached lists, then compares held-out CDCS top-1 to a matched pixel-boundary-L1
baseline.  CAL/DEV/test paths are rejected by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import infer_rank96 as rank96
from eval_candidate_rank import score_full_graph
from train_eval_cb1_g1_capacity import BoundaryBuddyNet, distort_frags, load_rgb, pair_band, sha256_file, to_frags
from train_offset_pose import mine_affinity_candidates

GRID = 24
N = GRID * GRID
K = 32
FIT_TARGETS = Path(r"E:\pazzle_data\train\targets")
SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P3_CDCS\g1_capacity")
SOURCE_ROOT = Path(r"C:\Users\pasha\Documents\GitHub\pazzle_will_be_killed")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "train"))
    p.add_argument("--targets", type=Path, default=FIT_TARGETS)
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260816)
    p.add_argument("--train-sources", type=int, default=96)
    p.add_argument("--eval-sources", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--queries-per-step", type=int, default=12)
    p.add_argument("--eval-queries-per-source", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--width", type=int, default=48)
    return p.parse_args()


def digest_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def config() -> rank96.InferenceConfig:
    return rank96.InferenceConfig(
        input_dir=Path(r"E:\pazzle_data\train\inputs"), output_dir=WORK / "unused", output_zip=None,
        ranker_checkpoint=SOURCE_ROOT / "artifacts" / "candidate_rank" / "rank_v2w64_best.pt",
        affinity_primary_checkpoint=SOURCE_ROOT / "artifacts" / "macro_affinity" / "affinity_r1_1200_best.pt",
        affinity_secondary_checkpoint=SOURCE_ROOT / "artifacts" / "macro_affinity" / "affinity_r3_1000_best.pt", device="cuda",
    )


def neighbour(slot: int, direction: int) -> int | None:
    r, c = divmod(slot, GRID)
    if direction == 0: return None if c == GRID - 1 else slot + 1
    if direction == 1: return None if r == GRID - 1 else slot + GRID
    if direction == 2: return None if c == 0 else slot - 1
    if direction == 3: return None if r == 0 else slot - GRID
    raise ValueError(direction)


def hardlists(candidates: np.ndarray, valid: np.ndarray, scores: np.ndarray, permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse = np.empty(N, dtype=np.int32); inverse[permutation] = np.arange(N, dtype=np.int32)
    anchors: list[int] = []; directions: list[int] = []; rows: list[np.ndarray] = []
    for anchor in range(N):
        source_slot = int(permutation[anchor])
        for direction in range(4):
            true_slot = neighbour(source_slot, direction)
            if true_slot is None: continue
            positive = int(inverse[true_slot])
            mask = valid[anchor] & np.isfinite(scores[direction, anchor]) & (candidates[anchor] != anchor) & (candidates[anchor] != positive)
            choices = candidates[anchor, mask].astype(np.int32, copy=False)
            ranking = np.argsort(-scores[direction, anchor, mask], kind="stable")
            hard = choices[ranking][: K - 1]
            if hard.shape != (K - 1,) or np.unique(hard).size != K - 1 or positive in hard: raise RuntimeError((anchor, direction, positive, hard.shape))
            row = np.concatenate((np.asarray([positive], dtype=np.int32), hard))
            if np.unique(row).size != K or anchor in row: raise RuntimeError((anchor, direction, row))
            anchors.append(anchor); directions.append(direction); rows.append(row)
    a = np.asarray(anchors, dtype=np.int16); d = np.asarray(directions, dtype=np.int8); m = np.stack(rows).astype(np.int16, copy=False)
    expected = 4 * GRID * (GRID - 1)
    if a.shape != (expected,) or d.shape != (expected,) or m.shape != (expected, K): raise RuntimeError((a.shape, d.shape, m.shape))
    return a, d, m


def source_names(cfg: argparse.Namespace) -> tuple[list[str], list[str]]:
    split = json.loads(cfg.split.read_text(encoding="utf-8"))
    fit = list(split["splits"]["fit"]); cal = set(split["splits"]["cal"]); dev = set(split["splits"]["dev"])
    total = cfg.train_sources + cfg.eval_sources
    if len(fit) != 5360 or total != 128 or total > len(fit) or set(fit) & (cal | dev): raise RuntimeError("P3 G1 source contract violated")
    train, evaluation = fit[:cfg.train_sources], fit[cfg.train_sources:total]
    if set(train) & set(evaluation) or any(name in cal or name in dev for name in train + evaluation): raise RuntimeError("non-FIT overlap")
    return train, evaluation


def cache_path(work: Path, name: str) -> Path:
    return work / "cache" / name.replace(".png", ".npz")


def build_one(name: str, source_index: int, cfg: argparse.Namespace, models: rank96.LoadedModels, device: torch.device) -> dict[str, object]:
    destination = cache_path(cfg.work, name)
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as data:
            if tuple(data["tiles"].shape) == (N, 20, 20, 3) and tuple(data["members"].shape) == (4 * GRID * (GRID - 1), K):
                return {"source": name, "cached": True, "cache": str(destination), "cache_sha256": sha256_file(destination)}
    clean = load_rgb(cfg.targets / name)
    fragments = distort_frags(to_frags(clean), np.random.default_rng(cfg.seed * 1009 + source_index))
    perm = np.random.default_rng(cfg.seed * 2029 + source_index).permutation(N).astype(np.int32)
    tiles = fragments[perm]
    tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).contiguous().float().to(device)
    with torch.no_grad():
        candidates, valid = mine_affinity_candidates(models.affinity_primary, tensor.unsqueeze(0), candidate_k=64, device=device, affinity_secondary=models.affinity_secondary)
        scores = score_full_graph(models.ranker, tensor, candidates[0], valid[0], pair_batch=4096, device=device)
    anchors, directions, members = hardlists(candidates[0].cpu().numpy(), valid[0].cpu().numpy(), scores.cpu().numpy(), perm)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, tiles=tiles, anchors=anchors, directions=directions, members=members, permutation=perm, source=np.asarray(name), seed=np.int64(cfg.seed))
    return {"source": name, "cached": False, "cache": str(destination), "cache_sha256": sha256_file(destination), "members_sha256": digest_array(members)}


def phase_prepare(cfg: argparse.Namespace) -> None:
    if cfg.device != "cuda" or not torch.cuda.is_available(): raise RuntimeError("P3 requires local CUDA")
    train, evaluation = source_names(cfg)
    for name in train + evaluation:
        if not (cfg.targets / name).is_file(): raise FileNotFoundError(cfg.targets / name)
    device = torch.device("cuda"); rank96._set_deterministic_runtime(cfg.seed, device); models = rank96.load_models(config(), device)
    records = [build_one(name, i, cfg, models, device) for i, name in enumerate(train + evaluation)]
    report = {"experiment":"P3_CDCS", "gate":"G1_prepare_FIT_only", "decision":"ready_for_G1_train", "train_sources":train, "eval_sources":evaluation, "records":records,
              "split_sha256":sha256_file(cfg.split), "ranker_sha256":sha256_file(config().ranker_checkpoint), "affinity_primary_sha256":sha256_file(config().affinity_primary_checkpoint), "affinity_secondary_sha256":sha256_file(config().affinity_secondary_checkpoint),
              "CAL_target_opened":False, "DEV_targets_opened":False, "test_accessed":False, "layouts_assembled":False, "restorer_used":False}
    cfg.work.mkdir(parents=True, exist_ok=True); (cfg.work / "p3_g1_prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps({k: report[k] for k in ("experiment","gate","decision")}, indent=2), flush=True)


def fetch(cache: dict[str, np.lib.npyio.NpzFile], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tiles = cache["tiles"]; anchors = cache["anchors"]; directions = cache["directions"]; members = cache["members"]
    bands = np.stack([pair_band(tiles, int(anchors[i]), int(candidate), int(directions[i])) for i in indices for candidate in members[i]], axis=0)
    return bands, members[indices]


def l1_choice(bands: np.ndarray, query_count: int) -> np.ndarray:
    x = bands.reshape(query_count, K, 3, 20, 4).astype(np.float32)
    # Pair bands concatenate two two-pixel strips. Mean absolute seam mismatch is lower-is-better.
    l1 = np.abs(x[:, :, :, :, 1] - x[:, :, :, :, 2]).mean(axis=(2, 3))
    return np.argmin(l1, axis=1)


def evaluate(model: nn.Module, caches: list[dict[str, np.lib.npyio.NpzFile]], rng: np.random.Generator, queries_per_source: int, device: torch.device) -> tuple[float, float, int]:
    model.eval(); cdcs_correct = 0; l1_correct = 0; total = 0
    with torch.no_grad():
        for cache in caches:
            count = cache["anchors"].shape[0]
            ids = rng.choice(count, size=queries_per_source, replace=False)
            bands, _ = fetch(cache, ids)
            logits = model(torch.from_numpy(bands).float().to(device)).reshape(-1, K)
            cdcs_correct += int((logits.argmax(dim=1).cpu().numpy() == 0).sum())
            l1_correct += int((l1_choice(bands, len(ids)) == 0).sum())
            total += len(ids)
    return cdcs_correct / total, l1_correct / total, total


def phase_train(cfg: argparse.Namespace) -> None:
    if cfg.device != "cuda" or not torch.cuda.is_available(): raise RuntimeError("P3 requires local CUDA")
    if cfg.steps != 2000 or cfg.train_sources != 96 or cfg.eval_sources != 32 or cfg.queries_per_step != 12 or cfg.eval_queries_per_source != 64 or cfg.temperature != 0.10: raise ValueError("P3 G1 fixed configuration contract violated")
    train_names, eval_names = source_names(cfg)
    absent = [name for name in train_names + eval_names if not cache_path(cfg.work, name).is_file()]
    if absent: raise RuntimeError(f"cache incomplete: {len(absent)}")
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); device = torch.device("cuda")
    train_caches = [dict(np.load(cache_path(cfg.work, n), allow_pickle=False)) for n in train_names]
    eval_caches = [dict(np.load(cache_path(cfg.work, n), allow_pickle=False)) for n in eval_names]
    model = BoundaryBuddyNet(width=cfg.width).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(cfg.seed + 17); losses: list[float] = []
    model.train()
    for step in range(cfg.steps):
        cache = train_caches[int(rng.integers(0, len(train_caches)))]; count = cache["anchors"].shape[0]
        ids = rng.choice(count, size=cfg.queries_per_step, replace=False); bands, _ = fetch(cache, ids)
        logits = model(torch.from_numpy(bands).float().to(device)).reshape(cfg.queries_per_step, K)
        loss = F.cross_entropy(logits / cfg.temperature, torch.zeros(cfg.queries_per_step, dtype=torch.long, device=device))
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 250 == 0: print(f"step={step + 1} loss={np.mean(losses[-100:]):.6f}", flush=True)
    cdcs_top1, l1_top1, total = evaluate(model, eval_caches, np.random.default_rng(cfg.seed + 29), cfg.eval_queries_per_source, device)
    checkpoint = cfg.work / "p3_g1_cdcs_capacity.pt"; cfg.work.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict":model.state_dict(), "seed":cfg.seed, "steps":cfg.steps, "width":cfg.width, "temperature":cfg.temperature}, checkpoint)
    initial = float(np.mean(losses[:100])); final = float(np.mean(losses[-100:])); passed = bool(cdcs_top1 >= l1_top1 + 0.05 and final < initial)
    report = {"experiment":"P3_CDCS", "gate":"G1_FIT_capacity", "steps":cfg.steps, "train_sources":cfg.train_sources, "eval_sources":cfg.eval_sources, "heldout_queries":total,
              "loss_first_100":initial, "loss_last_100":final, "CDCS_top1":cdcs_top1, "L1_top1":l1_top1, "top1_delta_pp":100*(cdcs_top1-l1_top1), "pass_criteria":"CDCS >= L1 + 5.0pp and final_loss < first_loss", "passes_G1":passed,
              "decision":"pass_to_full_FIT_training" if passed else "reject_P3_before_full_training", "checkpoint":str(checkpoint), "checkpoint_sha256":sha256_file(checkpoint), "split_sha256":sha256_file(cfg.split),
              "CAL_target_opened":False, "DEV_targets_opened":False, "test_accessed":False, "layouts_assembled":False, "restorer_used":False}
    (cfg.work / "p3_g1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    cfg = args()
    if cfg.phase == "prepare": phase_prepare(cfg)
    else: phase_train(cfg)


if __name__ == "__main__":
    main()
