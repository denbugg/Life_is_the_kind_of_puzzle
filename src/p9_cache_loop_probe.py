"""P9 G0c: structural loop coverage probe over frozen P8/rank96 cache records.

This probe intentionally loads only anchors, directions, members, and baseline
scores.  It does not read labels, permutations, tiles, image files, targets,
CAL, DEV, or test.  Its purpose is feasibility: quantify sparse valid 2×2 loop
support on actual canonical directional candidate lists before decoder G1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from p9_directed_loop_reweight import reweight_directed_2x2_loops

ROOT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813")
CACHE = ROOT / "P8_context_candidate_graph" / "g0_g1_capacity" / "cache"
OUT = ROOT / "P9_loop_decoder" / "g0_structural"
SENTINEL = -1.0e9
N = 576


def one(path: Path, loop_k: int, lam: float) -> dict:
    with np.load(path, allow_pickle=False) as z:
        required = {"anchors", "directions", "members", "baseline"}
        missing = required - set(z.files)
        if missing:
            raise RuntimeError(f"{path} missing {missing}")
        a = np.asarray(z["anchors"], dtype=np.int64)
        d = np.asarray(z["directions"], dtype=np.int64)
        m = np.asarray(z["members"], dtype=np.int64)
        s = np.asarray(z["baseline"], dtype=np.float64)
    out, report = reweight_directed_2x2_loops(
        a, d, m, s, n_tiles=N, loop_k=loop_k, lambda_value=lam, sentinel=SENTINEL
    )
    mask = s > SENTINEL / 2.0
    changed = mask & (out != s)
    return {
        "cache": path.name,
        "queries": int(a.size),
        "candidate_k": int(m.shape[1]),
        "usable_edges": int(mask.sum()),
        "changed_edges": int(changed.sum()),
        "changed_edge_rate": float(changed.sum() / max(mask.sum(), 1)),
        "mean_abs_delta_on_changed": float(np.abs(out[changed] - s[changed]).mean()) if changed.any() else 0.0,
        "loop_report": report.__dict__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=1)
    parser.add_argument("--loop-k", type=int, default=8)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.25)
    args = parser.parse_args()
    paths = sorted(CACHE.glob("*.npz"))[: args.sources]
    if len(paths) != args.sources:
        raise RuntimeError(f"Requested {args.sources} caches, found {len(paths)}")
    rows = [one(p, args.loop_k, args.lam) for p in paths]
    totals = {
        "source_count": len(rows),
        "usable_edges": int(sum(x["usable_edges"] for x in rows)),
        "changed_edges": int(sum(x["changed_edges"] for x in rows)),
        "accepted_loops": int(sum(x["loop_report"]["accepted_loops"] for x in rows)),
        "supported_edges": int(sum(x["loop_report"]["supported_horizontal_edges"] + x["loop_report"]["supported_vertical_edges"] for x in rows)),
    }
    totals["changed_edge_rate"] = float(totals["changed_edges"] / max(totals["usable_edges"], 1))
    result = {
        "experiment": "P9_loop_decoder",
        "gate": "G0c_frozen_graph_loop_coverage",
        "decision": "PASS" if totals["accepted_loops"] > 0 and totals["supported_edges"] > 0 else "REJECT_no_actual_loop_support",
        "loop_k": args.loop_k,
        "lambda": args.lam,
        "totals": totals,
        "rows": rows,
        "labels_read": False,
        "permutations_read": False,
        "tiles_read": False,
        "CAL_target_opened": False,
        "DEV_targets_opened": False,
        "test_accessed": False,
        "layouts_assembled": False,
        "restorer_used": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"p9_g0c_loop_coverage_sources{args.sources}_k{args.loop_k}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("experiment", "gate", "decision", "loop_k", "lambda", "totals")}, indent=2))


if __name__ == "__main__":
    main()
