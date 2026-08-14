"""R9: FIT-only raw input-bag adaptation of the R8 full-pair compatibility CNN.

R9 consumes raw train *input* mosaics plus pre-existing frozen graph-cache
permutations.  It never opens targets.  The cache filename contract is
image_0014_k64.npz -> img_000014.png and split membership is checked against the
pinned source-disjoint manifest before any labels are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from train_r8_holistic_pair import (
    GRID, NFRAG, IGNORE, HolisticPairNet, all_direct_targets,
    backward_sampled_loss, dense_scores, sampled_loss, sampled_pair_lists,
)

CACHE = Path(r"E:\pazzle_work\edge_confidence\full_graph_cache")
INPUTS = Path(r"E:\pazzle_data\train\inputs")
SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
R8_CKPT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R8_holistic_full_pair\g1_capacity_resume1500_retry1\r8_last.pt")
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R9_raw_bag_full_pair_adaptation")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tiles(path: Path) -> Tensor:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (480, 480):
            raise ValueError(f"expected strict RGB 480x480 raw mosaic, got {path}")
        rgb = np.asarray(image, dtype=np.uint8)
    tiles = rgb.reshape(GRID, 20, GRID, 20, 3).transpose(0, 2, 1, 3, 4).reshape(NFRAG, 20, 20, 3)
    return torch.from_numpy(np.ascontiguousarray(tiles)).permute(0, 3, 1, 2).float().div_(255.0)


def source_for_cache(cache_name: str) -> str:
    if not (cache_name.startswith("image_") and cache_name.endswith("_k64.npz")):
        raise ValueError(f"invalid frozen graph cache name: {cache_name}")
    index = int(cache_name.removeprefix("image_").removesuffix("_k64.npz"))
    return f"img_{index:06d}.png"


def load_membership(path: Path) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))["splits"]
    membership = {name: role for role, names in payload.items() for name in names}
    if len(membership) != sum(len(names) for names in payload.values()):
        raise RuntimeError("duplicate source in pinned split")
    return membership


class RawCacheBags:
    def __init__(self, cache_dir: Path, input_dir: Path, split_path: Path, role: str) -> None:
        membership = load_membership(split_path)
        items: List[Tuple[str, Tensor, Tensor]] = []
        all_roles: Dict[str, int] = {}
        for cache_path in sorted(cache_dir.glob("image_*_k64.npz")):
            name = source_for_cache(cache_path.name)
            source_role = membership.get(name)
            if source_role is None:
                raise RuntimeError(f"cached raw source absent from pinned split: {name}")
            all_roles[source_role] = all_roles.get(source_role, 0) + 1
            if source_role != role:
                continue
            input_path = input_dir / name
            if not input_path.is_file():
                raise FileNotFoundError(f"raw input absent for cache {cache_path.name}: {input_path}")
            with np.load(cache_path, allow_pickle=False) as z:
                perm = np.asarray(z["permutation"], dtype=np.int64)
                candidates = np.asarray(z["candidate_ids"], dtype=np.int64)
            if perm.shape != (NFRAG,) or sorted(perm.tolist()) != list(range(NFRAG)):
                raise RuntimeError(f"malformed frozen cache permutation: {cache_path}")
            if candidates.shape != (NFRAG, 128):
                raise RuntimeError(f"malformed frozen candidates: {cache_path}")
            items.append((name, load_tiles(input_path), torch.from_numpy(perm.copy()).long()))
        expected = {"fit": 17, "cal": 1, "dev": 2}
        if role in expected and len(items) != expected[role]:
            raise RuntimeError(f"expected {expected[role]} cached {role} raw bags, got {len(items)}")
        self.role, self.items, self.all_roles = role, items, all_roles

    def __len__(self) -> int:
        return len(self.items)

    def get(self, index: int, device: torch.device) -> Tuple[str, Tensor, Tensor]:
        name, tiles, perm = self.items[index % len(self.items)]
        return name, tiles.to(device, non_blocking=True), perm.to(device, non_blocking=True)


def member_coverage(scores: Tensor, perm: Tensor, k: int = 128) -> float:
    """Undirected membership coverage after max over four raw directional scores."""
    pooled = scores.max(dim=0).values
    candidate = pooled.topk(k=k, dim=-1).indices
    targets = all_direct_targets(perm)
    valid = targets.ne(IGNORE)
    covered = torch.zeros((NFRAG,), dtype=torch.long, device=scores.device)
    total = 0
    for direction in range(4):
        truth = targets[direction]
        mask = valid[direction]
        hit = candidate.eq(truth[:, None]).any(dim=-1) & mask
        covered += hit.long()
        total += int(mask.sum().item())
    return float(covered.sum().item() / max(1, total))


@torch.inference_mode()
def evaluate_raw(model: HolisticPairNet, bags: RawCacheBags, device: torch.device, pair_chunk: int) -> Dict[str, float]:
    model.eval()
    totals = {1: 0, 5: 0, 20: 0, 96: 0, 128: 0}
    total_edges = 0
    coverage_values: List[float] = []
    for index in range(len(bags)):
        _, tiles, perm = bags.get(index, device)
        scores = dense_scores(model, tiles, pair_chunk=pair_chunk)
        targets = all_direct_targets(perm)
        valid = targets.ne(IGNORE)
        for k in totals:
            guesses = scores.topk(k=k, dim=-1).indices
            totals[k] += int((guesses.eq(targets.unsqueeze(-1)).any(dim=-1) & valid).sum().item())
        total_edges += int(valid.sum().item())
        coverage_values.append(member_coverage(scores, perm, 128))
    result = {"bags": float(len(bags)), "valid_directed_edges": float(total_edges), "member_coverage_at_128": float(np.mean(coverage_values))}
    result.update({f"recall_at_{k}": value / max(1, total_edges) for k, value in totals.items()})
    return result


def raw_smoke(model: HolisticPairNet, bags: RawCacheBags, device: torch.device, seed: int) -> Dict[str, object]:
    if bags.role != "fit" or len(bags) != 17 or bags.all_roles.get("dev", 0) != 2 or bags.all_roles.get("cal", 0) != 1:
        raise RuntimeError("R9 G0 raw provenance contract failed")
    name, tiles, perm = bags.get(0, device)
    rng = random.Random(seed)
    anchors, directions, candidates, positive, stats = sampled_pair_lists(perm, anchors_per_board=4, negatives=15, rng=rng)
    loss, sampled_stats = sampled_loss(model, tiles, perm, anchors_per_board=4, negatives=15, rng=random.Random(seed + 1))
    targets = all_direct_targets(perm)
    bad_self = 0
    bad_direct = 0
    for row in range(candidates.shape[0]):
        anchor = int(anchors[row].item())
        forbidden = set(int(x) for x in targets[:, anchor][targets[:, anchor].ne(IGNORE)].tolist()) | {anchor}
        values = [int(x) for x in candidates[row, 1:].tolist()]
        bad_self += sum(value == anchor for value in values)
        bad_direct += sum(value in forbidden for value in values)
    if not torch.isfinite(loss) or bad_self or bad_direct:
        raise RuntimeError("R9 G0 label/negative invariant failed")
    return {"passed": True, "raw_source": name, "loss": float(loss.item()), "negative_checks": {"self_negatives": bad_self, "direct_neighbour_negatives": bad_direct}, "sampled": stats, "pair_tensors": sampled_stats["pair_tensors"], "label_source": "frozen_graph_cache_permutation_only", "targets_opened": False, "fit_cache_count": len(bags)}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--anchors-per-board", type=int, default=96)
    p.add_argument("--negatives", type=int, default=15)
    p.add_argument("--row-microbatch", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--pair-chunk", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir", type=Path, default=CACHE)
    p.add_argument("--input-dir", type=Path, default=INPUTS)
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--r8-checkpoint", type=Path, default=R8_CKPT)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--seed", type=int, default=20260814)
    return p.parse_args()


def main() -> None:
    cfg = args()
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cfg.work.mkdir(parents=True, exist_ok=True)
    report_path = cfg.report or cfg.work / "r9_report.json"
    fit = RawCacheBags(cfg.cache_dir, cfg.input_dir, cfg.split, "fit")
    checkpoint = torch.load(cfg.r8_checkpoint, map_location=device, weights_only=False)
    model = HolisticPairNet(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model"])
    smoke = raw_smoke(model, fit, device, cfg.seed + 17)
    smoke_mode = device.type == "cpu" and cfg.steps == 1
    provenance = {"cache_dir": str(cfg.cache_dir), "input_dir": str(cfg.input_dir), "split": str(cfg.split), "r8_checkpoint": str(cfg.r8_checkpoint), "r8_sha256": digest(cfg.r8_checkpoint), "fit_count": len(fit), "cached_role_counts": fit.all_roles, "targets_opened": False, "orientation": "fixed_no_rotations"}
    if smoke_mode:
        result = {"experiment": "R9_raw_bag_full_pair_adaptation", "gate": "G0_smoke", "smoke": smoke, "provenance": provenance}
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.steps, eta_min=cfg.lr * 0.1)
    rng = random.Random(cfg.seed + 701)
    history: List[Dict[str, object]] = []
    started = time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        values: List[float] = []
        rows, pairs = 0, 0
        for _ in range(cfg.batch_size):
            _, tiles, perm = fit.get(rng.randrange(len(fit)), device)
            value, stats = backward_sampled_loss(model, tiles, perm, anchors_per_board=cfg.anchors_per_board, negatives=cfg.negatives, row_microbatch=cfg.row_microbatch, rng=rng)
            values.append(value)
            rows += int(stats["rows"])
            pairs += int(stats["pair_tensors"])
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(cfg.batch_size)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item())
        optimizer.step()
        scheduler.step()
        if step == 1 or step % cfg.eval_every == 0 or step == cfg.steps:
            row = {"step": step, "train_loss": float(np.mean(values)), "grad_norm": grad_norm, "sampled_rows": rows, "pair_tensors": pairs, "lr": float(optimizer.param_groups[0]["lr"]), "elapsed_s": round(time.time() - started, 2)}
            history.append(row)
            print(json.dumps(row), flush=True)
            torch.save({"model": model.state_dict(), "architecture": checkpoint["architecture"], "step": step, "args": vars(cfg), "provenance": provenance, "row": row}, cfg.work / "r9_last.pt")
    cal = RawCacheBags(cfg.cache_dir, cfg.input_dir, cfg.split, "cal")
    metrics = evaluate_raw(model, cal, device, cfg.pair_chunk)
    passes = metrics["recall_at_20"] >= 0.20 and metrics["member_coverage_at_128"] >= 0.50
    result = {"experiment": "R9_raw_bag_full_pair_adaptation", "gate": "G1_raw_CAL", "smoke": smoke, "history": history, "raw_cal": metrics, "gate_thresholds": {"recall_at_20": 0.20, "member_coverage_at_128": 0.50}, "passes_G1": passes, "decision": "advance_to_R9_G2" if passes else "reject_R9_before_DEV", "provenance": provenance, "artifact": str(cfg.work / "r9_last.pt")}
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
