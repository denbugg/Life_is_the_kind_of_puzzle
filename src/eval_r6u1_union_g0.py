"""R6U1-G0: reproduce label-blind R2L∪affinity candidate coverage on pinned DEV caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval_r2l_affinity_union import _coverage, _load_r2, _union_candidates

DEFAULT_CACHE = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\SGT2_visual_graph\visual_cache")
DEFAULT_A = Path(r"artifacts\macro_affinity\affinity_r1_1200_best.pt")
DEFAULT_B = Path(r"artifacts\macro_affinity\affinity_r3_1000_best.pt")
DEFAULT_R2L = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\R2L_siamese\best.pt")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--dev", default="image_0014_k64.npz,image_0020_k64.npz")
    p.add_argument("--affinity-ckpt", type=Path, default=DEFAULT_A)
    p.add_argument("--affinity-ckpt2", type=Path, default=DEFAULT_B)
    p.add_argument("--r2-ckpt", type=Path, default=DEFAULT_R2L)
    p.add_argument("--affinity-k", type=int, default=64)
    p.add_argument("--r2-topk", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.affinity_k != 64 or args.r2_topk != 8:
        raise ValueError("G0 is pre-registered at U1 affinity-k=64 and r2-topk=8")
    args.work.mkdir(parents=True, exist_ok=True)
    r2 = _load_r2(str(args.r2_ckpt), device)
    r2.eval()
    rows = []
    names = [x.strip() for x in args.dev.split(",") if x.strip()]
    with torch.no_grad():
        for name in names:
            path = args.cache_dir / name
            with np.load(path, allow_pickle=False) as z:
                tiles_np = np.asarray(z["tiles_rgb"], dtype=np.uint8)
                permutation = np.asarray(z["permutation"], dtype=np.int64)
                cached_ids = np.asarray(z["candidate_ids"], dtype=np.int64)
                split = str(z["split_name"].item())
            if split != "dev":
                raise RuntimeError(f"R6U1-G0 requires pinned DEV cache, got {name}: {split}")
            if cached_ids.ndim != 2 or cached_ids.shape[0] != 24 * 24:
                raise RuntimeError(f"unexpected frozen rank96 candidate shape for {name}: {cached_ids.shape}")
            tiles = torch.from_numpy(tiles_np).permute(0, 3, 1, 2).contiguous().unsqueeze(0).float().div_(255.0).to(device)
            perm = torch.from_numpy(permutation).unsqueeze(0).to(device)
            # The frozen cached rank96 rows are the actual canonical base graph.
            base_candidates = torch.from_numpy(cached_ids).unsqueeze(0).expand(4, -1, -1).contiguous().to(device)
            base_valid = base_candidates >= 0
            directional = r2(tiles)
            # U1 stores directional scores per image; enforce the exact contract rather than guessing.
            if directional.ndim == 4 and directional.shape[0] == 1:
                directional = directional[0]
            union_candidates, union_valid = _union_candidates(base_candidates[0], base_valid[0], directional, args.r2_topk)
            base_cov, base_density = _coverage(perm[0], base_candidates, base_valid)
            union_cov, union_density = _coverage(perm[0], union_candidates, union_valid)
            output = args.work / f"{path.stem}_r6u1_union.npz"
            np.savez_compressed(
                output,
                candidate_ids=union_candidates.detach().cpu().numpy().astype(np.int16),
                valid=union_valid.detach().cpu().numpy().astype(np.bool_),
                tiles_rgb=tiles_np,
                permutation=permutation,
            )
            rows.append({
                "name": name, "split": split, "base_coverage": float(base_cov), "union_coverage": float(union_cov),
                "coverage_delta": float(union_cov - base_cov), "base_edges_per_tile": float(base_density),
                "union_edges_per_tile": float(union_density), "union_width": int(union_candidates.shape[-1]),
                "candidate_sha256": digest(output),
            })
    summary = {key: float(np.mean([row[key] for row in rows])) for key in ("base_coverage", "union_coverage", "coverage_delta", "base_edges_per_tile", "union_edges_per_tile")}
    passed = bool(summary["union_coverage"] >= 0.73 and summary["coverage_delta"] > 0.0)
    report = {
        "experiment": "R6U1_G0_source_disjoint_candidate_union",
        "scope": "frozen MacroAffinity pair plus frozen R2L; input-only candidate generation; permutation used only after generation for coverage",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checkpoint_metadata": {"base_graph": "frozen rank96 visual-cache candidate_ids", "r2_checkpoint": str(args.r2_ckpt)},
        "rows": rows, "summary": summary,
        "gate": {"condition": "mean union coverage >= 0.73 and strictly exceeds base at U1 settings", "passed": passed, "decision": "advance_to_R6U1_G1" if passed else "reject_R6U1_before_training"},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
