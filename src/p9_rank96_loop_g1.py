"""ORBIT-24 P9: rank96-only loop-consistent canonical-decoder harness.

Pre-registered contract (see P8_RESULT_P9_PREREGISTRATION.md):
- fixed orientation, 24×24, valid bijection required;
- pinned 128/32 source-disjoint FIT split;
- frozen rank96 width 64 graph and score path;
- directed 2×2 loop support with loop_k=8;
- lambda grid {0,.05,.10,.20,.40}, selected on the 128 train sources by mean
  absolute placement accuracy with lower-lambda deterministic tie-break;
- one held-FIT evaluation only, before CAL/DEV/test.

`prepare` reads FIT targets solely to construct corrupted input tiles and their
known source permutation, as P8 did. `evaluate` consumes cached frozen graph
scores/permutations and invokes no image/target path. The P8 learned scorer,
labels, and candidate-order-leaked cache labels are not imported or read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import infer_rank96 as rank96
import p3_g1_cdcs_capacity as p3
from eval_candidate_rank import score_full_graph
from p9_directed_loop_reweight import directed_to_dense_rd, reweight_directed_2x2_loops
from solve_buddies import solve_buddies_from_scores
from train_eval_cb1_g1_capacity import distort_frags, load_rgb, to_frags
from train_offset_pose import mine_affinity_candidates

GRID = 24
N = GRID * GRID
FIT_TARGETS = Path(r"E:\pazzle_data\train\targets")
SPLIT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json")
WORK = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P9_loop_decoder\g1_rank96_only")
SENTINEL = -1.0e9
# Canonical rank96 mine call uses 64 per directional proposal; the returned
# anchor-indexed union is 128 wide, as verified in P8 and the P9 G0d smoke.
MINING_K = 64
CANDIDATE_K = 128
LOOP_K = 8
LAMBDAS = (0.00, 0.05, 0.10, 0.20, 0.40)
SEED = 20260820


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "smoke", "evaluate"))
    p.add_argument("--targets", type=Path, default=FIT_TARGETS)
    p.add_argument("--split", type=Path, default=SPLIT)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=160, help="prepare only; prefix of pinned 128+32 sources")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def pinned_sources(split: Path, targets: Path) -> tuple[list[str], list[str]]:
    manifest = json.loads(split.read_text(encoding="utf-8"))
    fit = list(manifest["splits"]["fit"])
    train, held = fit[:128], fit[128:160]
    if len(train) != 128 or len(held) != 32 or set(train) & set(held):
        raise RuntimeError("P9 fixed 128/32 FIT split contract violated")
    for name in train + held:
        if not (targets / name).is_file():
            raise FileNotFoundError(targets / name)
    return train, held


def neighbor(slot: int, direction: int) -> int | None:
    row, col = divmod(int(slot), GRID)
    if direction == 0:
        return None if col == GRID - 1 else slot + 1
    if direction == 1:
        return None if row == GRID - 1 else slot + GRID
    if direction == 2:
        return None if col == 0 else slot - 1
    if direction == 3:
        return None if row == 0 else slot - GRID
    raise ValueError(direction)


def cache_path(work: Path, source: str) -> Path:
    return work / "cache" / source.replace(".png", ".npz")


def build_query_view(candidates: np.ndarray, scores: np.ndarray, permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert canonical `[4,N,K]` rank96 graph to complete internal directional queries.

    No label is derived and no true neighbour is injected.  Boundary queries are
    omitted because they do not belong to board adjacencies.
    """
    if candidates.shape != (N, CANDIDATE_K) or scores.shape != (4, N, CANDIDATE_K):
        raise RuntimeError(f"Unexpected canonical graph shapes: {candidates.shape}, {scores.shape}")
    anchors: list[int] = []
    directions: list[int] = []
    members: list[np.ndarray] = []
    baseline: list[np.ndarray] = []
    for tile in range(N):
        slot = int(permutation[tile])
        for direction in range(4):
            if neighbor(slot, direction) is None:
                continue
            anchors.append(tile)
            directions.append(direction)
            members.append(candidates[tile].copy())
            baseline.append(scores[direction, tile].copy())
    a = np.asarray(anchors, dtype=np.int32)
    d = np.asarray(directions, dtype=np.int8)
    m = np.stack(members).astype(np.int32, copy=False)
    s = np.stack(baseline).astype(np.float32, copy=False)
    expected = 4 * GRID * (GRID - 1)
    if a.shape != (expected,) or m.shape != (expected, CANDIDATE_K) or s.shape != m.shape:
        raise RuntimeError(f"P9 directional query contract violated: {a.shape}, {m.shape}, {s.shape}")
    return a, d, m, s


def canonical_dense_rd(candidates: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact rank96 dense semantics: listwise softmax, scatter-add duplicates, R/L and D/U fusion."""
    candidate_t = torch.from_numpy(np.asarray(candidates, dtype=np.int64))
    score_t = torch.from_numpy(np.asarray(scores, dtype=np.float32))
    finite = torch.isfinite(score_t)
    safe = score_t.masked_fill(~finite, -torch.inf)
    safe = torch.where(finite.any(dim=-1, keepdim=True), safe, torch.zeros_like(safe))
    probabilities = torch.softmax(safe, dim=-1).masked_fill(~finite, 0.0)

    def direction_matrix(direction: int) -> torch.Tensor:
        out = torch.zeros((N, N), dtype=probabilities.dtype)
        out.scatter_add_(1, candidate_t, probabilities[direction])
        out.fill_diagonal_(0.0)
        return out

    right = 0.5 * (direction_matrix(0) + direction_matrix(2).transpose(0, 1))
    down = 0.5 * (direction_matrix(1) + direction_matrix(3).transpose(0, 1))
    right.fill_diagonal_(0.0)
    down.fill_diagonal_(0.0)
    return right.numpy(), down.numpy()


def prepare_one(args: argparse.Namespace, source: str, index: int, models: object, device: torch.device) -> dict:
    dst = cache_path(args.work, source)
    if dst.is_file():
        with np.load(dst, allow_pickle=False) as z:
            required = {"anchors", "directions", "members", "baseline", "candidates", "scores", "permutation", "source"}
            if required.issubset(z.files) and z["members"].shape == (4 * GRID * (GRID - 1), CANDIDATE_K):
                return {"source": source, "cached": True, "cache": str(dst), "sha256": sha256_file(dst)}
    clean = load_rgb(args.targets / source)
    fragments = distort_frags(to_frags(clean), np.random.default_rng(SEED * 1009 + index))
    permutation = np.random.default_rng(SEED * 2029 + index).permutation(N).astype(np.int32)
    tiles = fragments[permutation]
    tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).contiguous().float().to(device)
    with torch.no_grad():
        candidates, valid = mine_affinity_candidates(
            models.affinity_primary,
            tensor.unsqueeze(0),
            candidate_k=MINING_K,
            device=device,
            affinity_secondary=models.affinity_secondary,
        )
        full_scores = score_full_graph(models.ranker, tensor, candidates[0], valid[0], pair_batch=4096, device=device)
    candidate_np = candidates[0].detach().cpu().numpy().astype(np.int32, copy=False)
    score_np = full_scores.detach().cpu().numpy().astype(np.float32, copy=False)
    anchors, directions, members, baseline = build_query_view(candidate_np, score_np, permutation)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        anchors=anchors,
        directions=directions,
        members=members,
        baseline=baseline,
        candidates=candidate_np,
        scores=score_np,
        permutation=permutation,
        source=np.array(source),
        seed=np.array(SEED, dtype=np.int64),
    )
    return {"source": source, "cached": False, "cache": str(dst), "sha256": sha256_file(dst)}


def prepare(args: argparse.Namespace) -> None:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("P9 graph preparation requires local interactive CUDA")
    train, held = pinned_sources(args.split, args.targets)
    sources = train + held
    if not 1 <= args.limit <= len(sources):
        raise ValueError(f"limit must be [1,{len(sources)}]")
    device = torch.device("cuda")
    models = rank96.load_models(p3.config(), device)
    rows = []
    for index, source in enumerate(sources[: args.limit]):
        print(f"p9_cache_start source={source} index={index}", flush=True)
        row = prepare_one(args, source, index, models, device)
        rows.append(row)
        print(f"p9_cache_done source={source} cached={row['cached']}", flush=True)
    report = {
        "experiment": "P9_loop_decoder",
        "gate": "G0d_rank96_cache_prepare" if args.limit < 160 else "G1_rank96_cache_prepare",
        "decision": "PASS",
        "prepared_sources": len(rows),
        "source_rows": rows,
        "split_sha256": sha256_file(args.split),
        "candidate_k": CANDIDATE_K,
        "loop_k": LOOP_K,
        "P8_labels_imported": False,
        "CAL_target_opened": False,
        "DEV_targets_opened": False,
        "test_accessed": False,
        "layouts_assembled": False,
        "restorer_used": False,
    }
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / f"p9_prepare_limit{args.limit}_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("experiment", "gate", "decision", "prepared_sources", "candidate_k", "loop_k")}, indent=2))


def load_cache(args: argparse.Namespace, source: str) -> dict[str, np.ndarray]:
    path = cache_path(args.work, source)
    if not path.is_file():
        raise FileNotFoundError(f"Missing P9 rank96 cache {path}")
    with np.load(path, allow_pickle=False) as z:
        required = {"anchors", "directions", "members", "baseline", "candidates", "scores", "permutation", "source"}
        if not required.issubset(z.files):
            raise RuntimeError(f"Invalid P9 cache {path}: {z.files}")
        if str(z["source"].item()) != source:
            raise RuntimeError(f"Source/cache mismatch: {source} vs {z['source'].item()}")
        return {k: z[k].copy() for k in required if k != "source"}


def precompute_loop_delta(cache: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, object]]:
    """Compute lambda=1 normalized loop delta once; later lambdas only scale it."""
    unit, report = reweight_directed_2x2_loops(
        cache["anchors"],
        cache["directions"],
        cache["members"],
        cache["baseline"],
        n_tiles=N,
        loop_k=LOOP_K,
        lambda_value=1.0,
        sentinel=SENTINEL,
    )
    delta = unit - cache["baseline"].astype(np.float64, copy=False)
    return delta, report.__dict__


def score_cache(
    cache: dict[str, np.ndarray],
    lam: float,
    *,
    loop_delta: np.ndarray | None = None,
    loop_report: dict[str, object] | None = None,
) -> dict[str, object]:
    anchors = cache["anchors"]
    directions = cache["directions"]
    members = cache["members"]
    base = cache["baseline"]
    candidates = cache["candidates"]
    full_scores = cache["scores"]
    permutation = cache["permutation"]
    if loop_delta is None:
        loop_delta, loop_report = precompute_loop_delta(cache)
    if loop_delta.shape != base.shape:
        raise RuntimeError(f"P9 loop delta shape mismatch: {loop_delta.shape} vs {base.shape}")
    adjusted = base.astype(np.float64, copy=False) + float(lam) * loop_delta
    # `baseline` rows are direct views of `full_scores[direction, anchor]`.
    # Write only existing query scores back; duplicates remain canonical and
    # lambda=0 therefore follows rank96 dense softmax/scatter-add exactly.
    adjusted_full = full_scores.astype(np.float32, copy=True)
    for q in range(anchors.size):
        adjusted_full[int(directions[q]), int(anchors[q])] = adjusted[q].astype(np.float32, copy=False)
    r, d = canonical_dense_rd(candidates, adjusted_full)
    board, objective = solve_buddies_from_scores(r, d, max_edges=96, min_margin=0.0, repair_passes=2)
    flat = np.asarray(board, dtype=np.int64).reshape(-1)
    valid = flat.shape == (N,) and np.array_equal(np.sort(flat), np.arange(N))
    if not valid:
        raise RuntimeError("P9 canonical decoder produced an invalid permutation")
    accuracy = float(np.mean(permutation[flat] == np.arange(N, dtype=np.int32)))
    return {
        "accuracy": accuracy,
        "objective": float(objective),
        "valid_bijection": True,
        "loop_report": loop_report,
    }


def smoke(args: argparse.Namespace) -> None:
    train, held = pinned_sources(args.split, args.targets)
    source = train[0]
    cache = load_cache(args, source)
    loop_delta, loop_report = precompute_loop_delta(cache)
    rows = {str(lam): score_cache(cache, lam, loop_delta=loop_delta, loop_report=loop_report) for lam in LAMBDAS}
    result = {
        "experiment": "P9_loop_decoder",
        "gate": "G0d_one_source_rank96_to_decoder_smoke",
        "decision": "PASS",
        "source": source,
        "lambda_results": rows,
        "P8_labels_imported": False,
        "CAL_target_opened": False,
        "DEV_targets_opened": False,
        "test_accessed": False,
        "layouts_assembled": False,
        "restorer_used": False,
    }
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "p9_g0d_smoke_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    train, held = pinned_sources(args.split, args.targets)
    all_sources = train + held
    # Evaluation observes only graph/permutation caches. No images or target path is read after this point.
    cache = {source: load_cache(args, source) for source in all_sources}
    per_lambda: dict[float, dict[str, list[float] | int]] = {
        lam: {"train": [], "held": [], "invalid": 0} for lam in LAMBDAS
    }
    train_set = set(train)
    for source in all_sources:
        split_name = "train" if source in train_set else "held"
        loop_delta, loop_report = precompute_loop_delta(cache[source])
        for lam in LAMBDAS:
            row = score_cache(cache[source], lam, loop_delta=loop_delta, loop_report=loop_report)
            if not row["valid_bijection"]:
                per_lambda[lam]["invalid"] = int(per_lambda[lam]["invalid"]) + 1
            per_lambda[lam][split_name].append(float(row["accuracy"]))
    summary: dict[str, dict[str, object]] = {}
    for lam in LAMBDAS:
        train_acc = np.asarray(per_lambda[lam]["train"], dtype=np.float64)
        held_acc = np.asarray(per_lambda[lam]["held"], dtype=np.float64)
        summary[f"{lam:.2f}"] = {
            "train_mean_accuracy": float(train_acc.mean()),
            "held_mean_accuracy": float(held_acc.mean()),
            "train_count": int(train_acc.size),
            "held_count": int(held_acc.size),
            "invalid_decodes": int(per_lambda[lam]["invalid"]),
        }
    # Deterministic lower-lambda tie-break over exact pre-registered grid.
    selected = max(LAMBDAS, key=lambda lam: (summary[f"{lam:.2f}"]["train_mean_accuracy"], -lam))
    baseline = summary["0.00"]
    chosen = summary[f"{selected:.2f}"]
    held_delta_pp = 100.0 * (chosen["held_mean_accuracy"] - baseline["held_mean_accuracy"])
    passes = bool(selected != 0.0 and held_delta_pp >= 3.0 and chosen["invalid_decodes"] == 0)
    result = {
        "experiment": "P9_loop_decoder",
        "gate": "G1_rank96_only_held_FIT_decoder",
        "lambda_grid": list(LAMBDAS),
        "selection_split": "FIT train 128 only",
        "held_split": "FIT held 32 only",
        "selected_lambda": selected,
        "baseline_lambda": 0.0,
        "held_delta_pp_vs_rank96": held_delta_pp,
        "per_lambda": summary,
        "passes_G1": passes,
        "decision": "PASS_open_CAL" if passes else "REJECT_before_CAL",
        "P8_labels_imported": False,
        "CAL_target_opened": False,
        "DEV_targets_opened": False,
        "test_accessed": False,
        "layouts_assembled": False,
        "restorer_used": False,
    }
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "p9_g1_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    args = parse_args()
    if args.phase == "prepare":
        prepare(args)
    elif args.phase == "smoke":
        smoke(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
