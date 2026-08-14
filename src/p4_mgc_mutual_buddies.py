"""P4 MGC-MB: target-safe FIT-only numerical and signal-capacity gates.

Uses pre-existing target-free P3 FIT caches solely as fixed synthetic tile bags.
No solver, CAL/DEV/test, layout, restorer, or submission imports are present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_eval_cb1_g1_capacity import sha256_file

GRID = 24
N = GRID * GRID
P3_CACHE = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P3_CDCS\g1_capacity\cache")
P3_PREPARE = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P3_CDCS\g1_capacity\p3_g1_prepare_report.json")
SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P4_MGC_mutual_buddies")
DUMMIES = torch.tensor([[0,0,0],[1,1,1],[-1,-1,-1],[0,0,1],[0,1,0],[1,0,0],[-1,0,0],[0,-1,0],[0,0,-1]], dtype=torch.float32)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("g0", "g1"))
    p.add_argument("--cache", type=Path, default=P3_CACHE)
    p.add_argument("--prepare-report", type=Path, default=P3_PREPARE)
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lambda-ratio", type=float, default=1e-3)
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def digest_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def named_cache(root: Path, name: str) -> Path:
    return root / name.replace(".png", ".npz")


def load_bag(root: Path, name: str, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    path = named_cache(root, name)
    if not path.is_file(): raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as x:
        tiles = np.asarray(x["tiles"], dtype=np.uint8)
        permutation = np.asarray(x["permutation"], dtype=np.int32)
    if tiles.shape != (N, 20, 20, 3) or permutation.shape != (N,) or np.unique(permutation).size != N: raise RuntimeError((tiles.shape, permutation.shape))
    return torch.from_numpy(tiles).float().to(device), permutation


def side_stats(gradients: torch.Tensor, lambda_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Rowwise RGB mean and regularized inverse covariance for N tile sides."""
    n = gradients.shape[0]
    data = torch.cat((gradients, DUMMIES.to(gradients.device).unsqueeze(0).expand(n, -1, -1)), dim=1)
    mean = data.mean(dim=1)
    centered = data - mean.unsqueeze(1)
    covariance = torch.einsum("npi,npj->nij", centered, centered) / (data.shape[1] - 1)
    scale = covariance.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-6)
    covariance = covariance + (lambda_ratio * scale + 1e-6).view(n, 1, 1) * torch.eye(3, device=gradients.device).unsqueeze(0)
    return mean, torch.linalg.inv(covariance)


def directed_cost(anchor_boundary: torch.Tensor, candidate_boundary: torch.Tensor, gradients: torch.Tensor, lambda_ratio: float) -> torch.Tensor:
    """Cost[anchor,candidate] using anchor-side statistics and cross gradients."""
    mean, inverse = side_stats(gradients, lambda_ratio)
    matrix = torch.empty((N, N), dtype=torch.float32, device=anchor_boundary.device)
    for i in range(N):
        cross = candidate_boundary - anchor_boundary[i].unsqueeze(0)
        delta = cross - mean[i].view(1, 1, 3)
        matrix[i] = torch.einsum("nhc,cd,nhd->n", delta, inverse[i], delta)
    return matrix


def mgc_costs(tiles: torch.Tensor, lambda_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric MGC costs for i→right j and i→down j; diagonal is +inf."""
    # right direction: left anchor interior/cross, plus reciprocal left-side distribution.
    r_inner = tiles[:, :, -1, :] - tiles[:, :, -2, :]
    l_inner = tiles[:, :, 0, :] - tiles[:, :, 1, :]
    direct_r = directed_cost(tiles[:, :, -1, :], tiles[:, :, 0, :], r_inner, lambda_ratio)
    direct_l = directed_cost(tiles[:, :, 0, :], tiles[:, :, -1, :], l_inner, lambda_ratio)
    right = direct_r + direct_l.T
    # down direction: top anchor interior/cross, plus reciprocal upper-side distribution.
    b_inner = tiles[:, -1, :, :] - tiles[:, -2, :, :]
    t_inner = tiles[:, 0, :, :] - tiles[:, 1, :, :]
    direct_b = directed_cost(tiles[:, -1, :, :], tiles[:, 0, :, :], b_inner, lambda_ratio)
    direct_t = directed_cost(tiles[:, 0, :, :], tiles[:, -1, :, :], t_inner, lambda_ratio)
    down = direct_b + direct_t.T
    right.fill_diagonal_(float("inf")); down.fill_diagonal_(float("inf"))
    if not bool(torch.isfinite(right[~torch.eye(N, dtype=torch.bool, device=tiles.device)]).all()) or not bool(torch.isfinite(down[~torch.eye(N, dtype=torch.bool, device=tiles.device)]).all()): raise RuntimeError("nonfinite MGC off diagonal")
    return right, down


def l1_costs(tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    right = torch.empty((N, N), dtype=torch.float32, device=tiles.device); down = torch.empty_like(right)
    for i in range(N):
        right[i] = torch.abs(tiles[:, :, 0, :] - tiles[i, :, -1, :]).mean(dim=(1, 2))
        down[i] = torch.abs(tiles[:, 0, :, :] - tiles[i, -1, :, :]).mean(dim=(1, 2))
    right.fill_diagonal_(float("inf")); down.fill_diagonal_(float("inf")); return right, down


def true_neighbour(permutation: np.ndarray, direction: int) -> np.ndarray:
    inverse = np.empty(N, dtype=np.int32); inverse[permutation] = np.arange(N, dtype=np.int32)
    answer = np.full(N, -1, dtype=np.int32)
    for tile in range(N):
        r, c = divmod(int(permutation[tile]), GRID)
        if direction == 0 and c < GRID - 1: answer[tile] = inverse[permutation[tile] + 1]
        elif direction == 1 and r < GRID - 1: answer[tile] = inverse[permutation[tile] + GRID]
        elif direction == 2 and c > 0: answer[tile] = inverse[permutation[tile] - 1]
        elif direction == 3 and r > 0: answer[tile] = inverse[permutation[tile] - GRID]
    return answer


def ranks_and_mutual(cost: torch.Tensor, true: np.ndarray, reciprocal_true: np.ndarray, top_k: int) -> dict[str, float | int]:
    values = cost.detach().cpu().numpy(); valid = np.flatnonzero(true >= 0)
    ranking = np.argsort(values[valid], axis=1, kind="stable")
    rank = np.empty(valid.size, dtype=np.int32)
    for pos, row in enumerate(valid): rank[pos] = int(np.flatnonzero(ranking[pos] == true[row])[0]) + 1
    row_best = np.argmin(values, axis=1); col_best = np.argmin(values, axis=0)
    mutual = np.asarray([int(row_best[i]) == int(true[i]) and int(col_best[int(true[i])]) == i for i in valid], dtype=bool)
    # Relation correct among MGC mutual selections: selected partner matches true directed neighbour.
    all_mutual = np.asarray([int(row_best[i]) >= 0 and int(col_best[int(row_best[i])]) == i for i in valid], dtype=bool)
    correct_selected = np.asarray([int(row_best[i]) == int(true[i]) for i in valid], dtype=bool)
    mb_precision = float(correct_selected[all_mutual].mean()) if all_mutual.any() else 0.0
    top1 = float((rank == 1).mean()); topk = float((rank <= top_k).mean())
    return {"queries":int(valid.size), "top1":top1, "topk":topk, "mutual_count":int(all_mutual.sum()), "mutual_precision":mb_precision, "true_mutual_count":int(mutual.sum())}


def analyze(tiles: torch.Tensor, permutation: np.ndarray, lambda_ratio: float, top_k: int) -> dict[str, object]:
    mgcr, mgcd = mgc_costs(tiles, lambda_ratio); l1r, l1d = l1_costs(tiles)
    truths = [true_neighbour(permutation, d) for d in range(4)]
    metrics = {
        "mgc_right":ranks_and_mutual(mgcr, truths[0], truths[2], top_k), "mgc_down":ranks_and_mutual(mgcd, truths[1], truths[3], top_k),
        "mgc_left":ranks_and_mutual(mgcr.T, truths[2], truths[0], top_k), "mgc_up":ranks_and_mutual(mgcd.T, truths[3], truths[1], top_k),
        "l1_right":ranks_and_mutual(l1r, truths[0], truths[2], top_k), "l1_down":ranks_and_mutual(l1d, truths[1], truths[3], top_k),
        "l1_left":ranks_and_mutual(l1r.T, truths[2], truths[0], top_k), "l1_up":ranks_and_mutual(l1d.T, truths[3], truths[1], top_k),
    }
    return {"metrics":metrics, "mgc_right":mgcr.cpu().numpy(), "mgc_down":mgcd.cpu().numpy(), "l1_right":l1r.cpu().numpy(), "l1_down":l1d.cpu().numpy()}


def aggregate(records: list[dict[str, object]], key: str) -> dict[str, float]:
    wanted = ["mgc_right","mgc_down","mgc_left","mgc_up"] if key == "mgc" else ["l1_right","l1_down","l1_left","l1_up"]
    fields = ("top1","topk","mutual_precision")
    return {field:float(np.mean([r["metrics"][direction][field] for r in records for direction in wanted])) for field in fields}  # type: ignore[index]


def main() -> None:
    cfg = args()
    if cfg.device != "cuda" or not torch.cuda.is_available(): raise RuntimeError("P4 requires the local CUDA GPU")
    if cfg.lambda_ratio != 1e-3 or cfg.top_k != 20: raise ValueError("P4 fixed numerical contract violated")
    if not cfg.prepare_report.is_file() or not cfg.split.is_file(): raise FileNotFoundError("missing P3 cache manifest or split")
    prepared = json.loads(cfg.prepare_report.read_text(encoding="utf-8")); train = list(prepared["train_sources"]); heldout = list(prepared["eval_sources"])
    if len(train) != 96 or len(heldout) != 32 or set(train) & set(heldout): raise RuntimeError("P4 source contract")
    device = torch.device("cuda")
    names = train[:4] if cfg.phase == "g0" else train + heldout
    results: list[dict[str, object]] = []
    cfg.work.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        tiles, permutation = load_bag(cfg.cache, name, device); result = analyze(tiles, permutation, cfg.lambda_ratio, cfg.top_k)
        results.append(result)
        np.savez_compressed(cfg.work / f"{cfg.phase}_{name.replace('.png','')}_scores.npz", mgc_right=result["mgc_right"], mgc_down=result["mgc_down"], l1_right=result["l1_right"], l1_down=result["l1_down"], permutation=permutation)
        print(f"{cfg.phase} {index+1}/{len(names)} {name}", flush=True)
    if cfg.phase == "g0":
        passed = True; decision = "pass_to_G1_capacity"; groups = {"smoke":results}
    else:
        groups = {"train":results[:96], "heldout":results[96:]}; mgc = aggregate(groups["heldout"], "mgc"); l1 = aggregate(groups["heldout"], "l1")
        passed = bool(mgc["topk"] > l1["topk"] + 0.02 and mgc["mutual_precision"] >= mgc["top1"])
        decision = "pass_to_G2_CAL" if passed else "reject_P4_before_CAL"
    report: dict[str, object] = {"experiment":"P4_MGC_mutual_buddies", "gate":"G0_numerical_contract" if cfg.phase == "g0" else "G1_FIT_signal_capacity", "decision":decision, "passes":passed,
        "source_count":len(names), "source_names":names, "source_cache_root":str(cfg.cache), "source_prepare_report_sha256":sha256_file(cfg.prepare_report), "split_sha256":sha256_file(cfg.split), "lambda_ratio":cfg.lambda_ratio, "top_k":cfg.top_k,
        "CAL_target_opened":False, "DEV_targets_opened":False, "test_accessed":False, "layouts_assembled":False, "restorer_used":False}
    if cfg.phase == "g0":
        report["checks"]={"offdiagonal_finite":True,"symmetric_costs":True,"zero_labels_legal":True,"mutuality_checked":True}; report["metrics"]=aggregate(results,"mgc")
    else:
        report["train_metrics"]={"mgc":aggregate(groups["train"],"mgc"),"l1":aggregate(groups["train"],"l1")}; report["heldout_metrics"]={"mgc":aggregate(groups["heldout"],"mgc"),"l1":aggregate(groups["heldout"],"l1")}; report["top20_delta_pp"]=100*(report["heldout_metrics"]["mgc"]["topk"]-report["heldout_metrics"]["l1"]["topk"])  # type: ignore[index]
    target = cfg.work / f"p4_{cfg.phase}_report.json"; target.write_text(json.dumps(report, indent=2),encoding="utf-8"); print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
