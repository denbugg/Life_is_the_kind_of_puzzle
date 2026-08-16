"""P12 SLC-24 — sparse loop-consensus solver refiner.

Pre-registered in P12_PRE_REGISTRATION.md before this source file was created.
The harness intentionally operates only on frozen rank96 candidate graph evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from candidate_rank import DOWN, LEFT, RIGHT, UP, NUM_DIRECTIONS
from eval_candidate_rank import load_ranker
from eval_seeded_qap import dense_rd
from solve_buddies import solve_buddies_from_scores
from train_offset_pose import load_frozen_affinity, mine_affinity_candidates

GRID = 24
N_TILES = GRID * GRID
TILE_SIDE = 20
CANDIDATE_K = 64
CANDIDATE_WIDTH = 128
TOP_LOOP = 12
SOLVER_MAX_EDGES = 96
SEED = 20260816
LAMBDAS = (0.00, 0.05, 0.10, 0.20, 0.40, 0.80)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(array: np.ndarray) -> bytes:
    value = np.ascontiguousarray(array)
    return str(value.dtype).encode() + b"\0" + repr(value.shape).encode() + b"\0" + value.tobytes()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(canonical_bytes(array)).hexdigest()


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def source_cache_path(cache_dir: Path, source: str) -> Path:
    return cache_dir / f"{Path(source).stem}.npz"


def load_tiles(cache_dir: Path, source: str, device: torch.device) -> torch.Tensor:
    path = source_cache_path(cache_dir, source)
    with np.load(path, allow_pickle=False) as raw:
        values = raw["tiles_uint8"].copy()
    if values.shape != (N_TILES, TILE_SIDE, TILE_SIDE, 3):
        raise RuntimeError(f"unexpected tiles shape in {path}: {values.shape}")
    # Frozen rank96 accepts its canonical [0,1] RGB tensor, never P10/P11's [-1,1] encoder range.
    return torch.from_numpy(values).permute(0, 3, 1, 2).float().div_(255.0).to(device)


def load_labels(cache_dir: Path, source: str) -> tuple[np.ndarray, np.ndarray]:
    path = source_cache_path(cache_dir, source)
    with np.load(path, allow_pickle=False) as raw:
        target = raw["target_tile_to_slot"].astype(np.int32, copy=True)
        initial = raw["initial_tile_to_slot"].astype(np.int32, copy=True)
    expected = np.arange(N_TILES, dtype=np.int32)
    if not np.array_equal(np.sort(target), expected) or not np.array_equal(np.sort(initial), expected):
        raise RuntimeError(f"non-permutation labels in {path}")
    return target, initial


def source_lists(prepare_report: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(prepare_report.read_text(encoding="utf-8"))
    train = [str(x) for x in payload["train_sources"]]
    held = [str(x) for x in payload["held_sources"]]
    if len(train) != 128 or len(held) != 32 or set(train).intersection(held):
        raise RuntimeError("P12 requires the exact locked P10/P11 source-disjoint 128/32 partition")
    return train, held


def deterministic_row_permutation(source: str, direction: int, anchor: int, width: int) -> np.ndarray:
    # Every sort key depends solely on publicly derivable graph metadata, never labels or target pixels.
    keyed = []
    for pos in range(width):
        text = f"P12-candidate-order-v1|{SEED}|{source}|{direction}|{anchor}|{pos}".encode("utf-8")
        keyed.append((hashlib.blake2b(text, digest_size=16).digest(), pos))
    return np.asarray([pos for _, pos in sorted(keyed)], dtype=np.int64)


def permute_candidate_axes(candidates: np.ndarray, valid: np.ndarray, scores: np.ndarray, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if candidates.shape != (N_TILES, CANDIDATE_WIDTH) or valid.shape != candidates.shape or scores.shape != (NUM_DIRECTIONS, N_TILES, CANDIDATE_WIDTH):
        raise RuntimeError(f"unexpected canonical candidate/score shapes {candidates.shape} {valid.shape} {scores.shape}")
    out_c = np.empty_like(candidates)
    out_v = np.empty_like(valid)
    out_s = np.empty_like(scores)
    for anchor in range(N_TILES):
        order = deterministic_row_permutation(source, 0, anchor, CANDIDATE_WIDTH)
        out_c[anchor] = candidates[anchor, order]
        out_v[anchor] = valid[anchor, order]
        out_s[:, anchor] = scores[:, anchor, order]
    return out_c, out_v, out_s


def standardize_rows(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full_like(scores, -np.inf, dtype=np.float32)
    for direction in range(NUM_DIRECTIONS):
        for anchor in range(N_TILES):
            mask = valid[anchor] & np.isfinite(scores[direction, anchor])
            vals = scores[direction, anchor, mask]
            if vals.size == 0:
                continue
            std = float(vals.std())
            if not math.isfinite(std) or std < 1e-6:
                std = 1.0
            out[direction, anchor, mask] = (vals - float(vals.mean())) / std
    return out


def physical_directed_matrices(candidates: np.ndarray, z: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.full((N_TILES, N_TILES), -np.inf, dtype=np.float32)
    down = np.full((N_TILES, N_TILES), -np.inf, dtype=np.float32)
    for direction in range(NUM_DIRECTIONS):
        for anchor in range(N_TILES):
            mask = valid[anchor] & np.isfinite(z[direction, anchor])
            for target, value in zip(candidates[anchor, mask], z[direction, anchor, mask], strict=True):
                target = int(target)
                if target == anchor:
                    continue
                if direction == RIGHT:
                    right[anchor, target] = max(right[anchor, target], float(value))
                elif direction == LEFT:
                    right[target, anchor] = max(right[target, anchor], float(value))
                elif direction == DOWN:
                    down[anchor, target] = max(down[anchor, target], float(value))
                elif direction == UP:
                    down[target, anchor] = max(down[target, anchor], float(value))
    np.fill_diagonal(right, -np.inf)
    np.fill_diagonal(down, -np.inf)
    return right, down


def top_targets(matrix: np.ndarray, anchor: int, k: int = TOP_LOOP) -> np.ndarray:
    row = matrix[anchor]
    finite = np.flatnonzero(np.isfinite(row))
    if finite.size == 0:
        return finite
    selected = finite[np.argsort(-row[finite], kind="stable")[:k]]
    return selected.astype(np.int32, copy=False)


def loop_support(right: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return physical right/down 2x2 completion support; absent completion is 0."""
    loop_r = np.zeros_like(right, dtype=np.float32)
    loop_d = np.zeros_like(down, dtype=np.float32)
    top_r = [top_targets(right, i) for i in range(N_TILES)]
    top_d = [top_targets(down, i) for i in range(N_TILES)]
    for i in range(N_TILES):
        # i -> j (right), i -> k (down), j -> l (down), k -> l (right).
        for j in top_r[i]:
            best = -np.inf
            for k in top_d[i]:
                for l in top_d[int(j)]:
                    edge = right[int(k), int(l)]
                    if np.isfinite(edge):
                        best = max(best, float(down[i, int(k)] + down[int(j), int(l)] + edge))
            if np.isfinite(best):
                loop_r[i, int(j)] = best
        for k in top_d[i]:
            best = -np.inf
            for j in top_r[i]:
                for l in top_r[int(k)]:
                    edge = down[int(j), int(l)]
                    if np.isfinite(edge):
                        best = max(best, float(right[i, int(j)] + right[int(k), int(l)] + edge))
            if np.isfinite(best):
                loop_d[i, int(k)] = best
    return loop_r, loop_d


def refine_scores(candidates: np.ndarray, scores: np.ndarray, valid: np.ndarray, lam: float) -> tuple[np.ndarray, dict[str, float]]:
    z = standardize_rows(scores, valid)
    physical_r, physical_d = physical_directed_matrices(candidates, z, valid)
    support_r, support_d = loop_support(physical_r, physical_d)
    refined = z.copy()
    for direction in range(NUM_DIRECTIONS):
        for anchor in range(N_TILES):
            mask = valid[anchor] & np.isfinite(z[direction, anchor])
            targets = candidates[anchor, mask].astype(np.int32, copy=False)
            if direction == RIGHT:
                support = support_r[anchor, targets]
            elif direction == LEFT:
                support = support_r[targets, anchor]
            elif direction == DOWN:
                support = support_d[anchor, targets]
            else:
                support = support_d[targets, anchor]
            refined[direction, anchor, mask] = z[direction, anchor, mask] + float(lam) * support
    metrics = {
        "finite_raw_fraction": float(np.isfinite(z[:, valid]).mean()),
        "right_loop_nonzero_fraction": float((support_r[np.isfinite(physical_r)] != 0.0).mean()) if np.isfinite(physical_r).any() else 0.0,
        "down_loop_nonzero_fraction": float((support_d[np.isfinite(physical_d)] != 0.0).mean()) if np.isfinite(physical_d).any() else 0.0,
    }
    return refined, metrics


@torch.inference_mode()
def score_full_graph_fp32(model, tiles: torch.Tensor, candidates: torch.Tensor, valid: torch.Tensor, *, pair_batch: int, device: torch.device) -> torch.Tensor:
    """Canonical sparse ranker traversal with autocast intentionally disabled."""
    if tiles.shape != (N_TILES, 3, TILE_SIDE, TILE_SIDE):
        raise ValueError(f"unexpected ranker tiles shape {tuple(tiles.shape)}")
    if candidates.shape != (N_TILES, CANDIDATE_WIDTH) or valid.shape != candidates.shape:
        raise ValueError(f"unexpected frozen candidate tensor shapes {tuple(candidates.shape)} {tuple(valid.shape)}")
    if pair_batch < 1:
        raise ValueError("pair_batch must be positive")
    scores = torch.full((NUM_DIRECTIONS, N_TILES, CANDIDATE_WIDTH), -torch.inf, dtype=torch.float32, device=device)
    anchors = torch.arange(N_TILES, device=device).view(N_TILES, 1).expand_as(candidates)
    for direction in range(NUM_DIRECTIONS):
        mask = valid
        anchor_flat = anchors[mask]
        target_flat = candidates[mask]
        chunks: list[torch.Tensor] = []
        for start in range(0, int(anchor_flat.numel()), pair_batch):
            stop = min(start + pair_batch, int(anchor_flat.numel()))
            count = stop - start
            source_batch = tiles[anchor_flat[start:stop]]
            target_batch = tiles[target_flat[start:stop]]
            direction_ids = torch.full((count,), direction, dtype=torch.long, device=device)
            if count < pair_batch:
                pad = pair_batch - count
                source_batch = torch.cat((source_batch, source_batch[-1:].expand(pad, -1, -1, -1)), dim=0)
                target_batch = torch.cat((target_batch, target_batch[-1:].expand(pad, -1, -1, -1)), dim=0)
                direction_ids = torch.cat((direction_ids, direction_ids[-1:].expand(pad)), dim=0)
            # No torch.autocast / AMP context: this is deliberately FP32 end-to-end.
            logits = model(source_batch, target_batch, direction_ids).reshape(-1).float()[:count]
            chunks.append(logits)
        scores[direction][mask] = torch.cat(chunks, dim=0)
    return scores


def score_file(score_dir: Path, source: str) -> Path:
    return score_dir / f"{Path(source).stem}.npz"


def save_score(score_dir: Path, source: str, candidates: np.ndarray, valid: np.ndarray, scores: np.ndarray) -> Path:
    score_dir.mkdir(parents=True, exist_ok=True)
    target = score_file(score_dir, source)
    np.savez_compressed(
        target,
        source=np.asarray(source),
        candidates=candidates.astype(np.int16, copy=False),
        valid=valid.astype(np.bool_, copy=False),
        scores=scores.astype(np.float32, copy=False),
        candidate_sha256=np.asarray(array_sha(candidates.astype(np.int16, copy=False))),
        valid_sha256=np.asarray(array_sha(valid)),
        score_sha256=np.asarray(array_sha(scores)),
        candidate_order_protocol=np.asarray("P12-candidate-order-v1"),
    )
    return target


def load_score(score_dir: Path, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = score_file(score_dir, source)
    with np.load(path, allow_pickle=False) as raw:
        candidates = raw["candidates"].astype(np.int32, copy=True)
        valid = raw["valid"].astype(bool, copy=True)
        scores = raw["scores"].astype(np.float32, copy=True)
        stored_source = str(raw["source"].item())
        if stored_source != source:
            raise RuntimeError(f"score-cache source mismatch: {stored_source} != {source}")
        if str(raw["candidate_sha256"].item()) != array_sha(candidates.astype(np.int16)):
            raise RuntimeError(f"candidate SHA mismatch in {path}")
    if candidates.shape != (N_TILES, CANDIDATE_WIDTH) or valid.shape != candidates.shape or scores.shape != (NUM_DIRECTIONS, N_TILES, CANDIDATE_WIDTH):
        raise RuntimeError(f"score cache tensor shape mismatch in {path}: {candidates.shape} {valid.shape} {scores.shape}")
    if np.any((candidates < 0) | (candidates >= N_TILES)):
        raise RuntimeError(f"candidate IDs out of range in {path}")
    if not np.isfinite(scores[:, valid]).all() or not np.isneginf(scores[:, ~valid]).all():
        raise RuntimeError(f"invalid frozen score mask encoding in {path}")
    return candidates, valid, scores


def board_to_tile_slot(board: np.ndarray) -> np.ndarray:
    flat = np.asarray(board, dtype=np.int32).reshape(-1)
    expected = np.arange(N_TILES, dtype=np.int32)
    if flat.shape != (N_TILES,) or not np.array_equal(np.sort(flat), expected):
        raise RuntimeError("canonical buddies decoder did not return a 576-tile bijective board")
    result = np.empty(N_TILES, dtype=np.int32)
    result[flat] = expected
    return result


def solve_refined(candidates: np.ndarray, scores: np.ndarray, valid: np.ndarray, lam: float) -> tuple[np.ndarray, dict[str, float]]:
    refined, metrics = refine_scores(candidates, scores, valid, lam)
    tensor_c = torch.from_numpy(candidates).long()
    tensor_s = torch.from_numpy(refined).float()
    right, down = dense_rd(tensor_c, tensor_s)
    board, objective = solve_buddies_from_scores(
        right.detach().cpu().numpy(), down.detach().cpu().numpy(), max_edges=SOLVER_MAX_EDGES, min_margin=0.0, repair_passes=0
    )
    metrics["objective"] = float(objective)
    return board_to_tile_slot(board), metrics


def g0a(args: argparse.Namespace) -> None:
    # Construct a 2x2 configuration with one true loop and a score-matched false edge.
    right = np.full((N_TILES, N_TILES), -np.inf, np.float32)
    down = np.full((N_TILES, N_TILES), -np.inf, np.float32)
    right[0, 1] = right[2, 3] = 1.0
    down[0, 2] = down[1, 3] = 1.0
    right[0, 4] = 1.0  # score-matched nonloop
    loop_r, loop_d = loop_support(right, down)
    # Candidate-order invariance is verified through an explicit all-direction cache tensor.
    candidates = np.broadcast_to(np.arange(CANDIDATE_WIDTH, dtype=np.int32), (N_TILES, CANDIDATE_WIDTH)).copy()
    valid = np.ones_like(candidates, dtype=bool)
    scores = np.zeros((NUM_DIRECTIONS, N_TILES, CANDIDATE_WIDTH), dtype=np.float32)
    scores[:, :, 0] = 1.0
    shuffled_c, shuffled_v, shuffled_s = permute_candidate_axes(candidates, valid, scores, "synthetic.png")
    first, _ = refine_scores(candidates, scores, valid, 0.4)
    second, _ = refine_scores(shuffled_c, shuffled_s, shuffled_v, 0.4)
    # Compare value multiset per row: refinement is identity/order invariant after target-ID reindexing.
    order_ok = True
    for d in range(NUM_DIRECTIONS):
        for i in range(N_TILES):
            a = dict(zip(candidates[i].tolist(), first[d, i].tolist(), strict=True))
            b = dict(zip(shuffled_c[i].tolist(), second[d, i].tolist(), strict=True))
            if a != b:
                order_ok = False
                break
    report = {
        "experiment": "P12_loop_consensus",
        "gate": "G0a_synthetic_loop_contracts",
        "true_complete_loop_support": float(loop_r[0, 1]),
        "score_matched_nonloop_support": float(loop_r[0, 4]),
        "down_complete_loop_support": float(loop_d[0, 2]),
        "missing_edge_support": float(loop_r[5, 6]),
        "candidate_order_invariant": order_ok,
        "candidate_order_labels_used": False,
        "p8_labels_imported": False,
        "amp_used": False,
    }
    report["passes_G0a"] = bool(report["true_complete_loop_support"] > report["score_matched_nonloop_support"] and report["down_complete_loop_support"] > 0.0 and report["missing_edge_support"] == 0.0 and order_ok)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p12_g0a_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if not report["passes_G0a"]:
        raise RuntimeError("P12 G0a contract failed")


def load_frozen_models(device: torch.device) -> SimpleNamespace:
    # Reproduce only the three checkpoint loads from the frozen production profile.
    # `InferenceConfig` additionally describes submission I/O, so it is deliberately
    # not instantiated by this solver-only harness.
    repo = Path(r"C:\\Users\\pasha\\Documents\\GitHub\\pazzle_will_be_killed")
    ranker_path = repo / "artifacts" / "candidate_rank" / "rank_v2w64_best.pt"
    primary_path = repo / "artifacts" / "macro_affinity" / "affinity_r1_1200_best.pt"
    secondary_path = repo / "artifacts" / "macro_affinity" / "affinity_r3_1000_best.pt"
    for name, path in (("ranker", ranker_path), ("affinity_primary", primary_path), ("affinity_secondary", secondary_path)):
        if not path.is_file():
            raise FileNotFoundError(f"missing canonical P12 frozen {name} checkpoint: {path}")
    ranker, _ = load_ranker(str(ranker_path), device)
    primary, _, _ = load_frozen_affinity(str(primary_path), device)
    secondary, _, _ = load_frozen_affinity(str(secondary_path), device)
    ranker.eval()
    primary.eval()
    secondary.eval()
    return SimpleNamespace(ranker=ranker, affinity_primary=primary, affinity_secondary=secondary)


def score_source(models, cache_dir: Path, score_dir: Path, source: str, device: torch.device, pair_batch: int) -> Path:
    tiles = load_tiles(cache_dir, source, device)
    with torch.inference_mode():
        candidates_batched, valid_batched = mine_affinity_candidates(
            models.affinity_primary, tiles.unsqueeze(0), candidate_k=CANDIDATE_K, device=device, affinity_secondary=models.affinity_secondary
        )
        if candidates_batched.shape[0] != 1 or valid_batched.shape != candidates_batched.shape:
            raise RuntimeError(f"unexpected frozen rank96 batched candidate shape {tuple(candidates_batched.shape)}")
        candidates = candidates_batched[0]
        valid = valid_batched[0]
        raw_scores = score_full_graph_fp32(models.ranker, tiles, candidates, valid, pair_batch=pair_batch, device=device)
    c = candidates.detach().cpu().numpy().astype(np.int32, copy=False)
    v = valid.detach().cpu().numpy().astype(bool, copy=False)
    s = raw_scores.detach().cpu().numpy().astype(np.float32, copy=False)
    c, v, s = permute_candidate_axes(c, v, s, source)
    return save_score(score_dir, source, c, v, s)


def g0b(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("P12 G0b must run through interactive CUDA session")
    seed_all()
    train, _ = source_lists(args.prepare_report)
    source = args.source or train[0]
    device = torch.device("cuda")
    models = load_frozen_models(device)
    score_path = score_source(models, args.cache_dir, args.score_dir, source, device, args.pair_batch)
    candidates, valid, scores = load_score(args.score_dir, source)
    # A second deterministic candidate-order realization must be exact after ID reindexing.
    twice_c, twice_v, twice_s = permute_candidate_axes(candidates, valid, scores, source + "|audit")
    original, original_metrics = refine_scores(candidates, scores, valid, 0.40)
    shuffled, shuffled_metrics = refine_scores(twice_c, twice_s, twice_v, 0.40)
    same = True
    max_order_delta = 0.0
    order_tolerance = 1.0e-5
    for d in range(NUM_DIRECTIONS):
        for i in range(N_TILES):
            # The canonical union retains invalid duplicate slots; they are not graph identities.
            left_mask = valid[i]
            right_mask = twice_v[i]
            left = dict(zip(candidates[i, left_mask].tolist(), original[d, i, left_mask].tolist(), strict=True))
            right = dict(zip(twice_c[i, right_mask].tolist(), shuffled[d, i, right_mask].tolist(), strict=True))
            if left.keys() != right.keys():
                same = False
                continue
            delta = max((abs(float(left[key]) - float(right[key])) for key in left), default=0.0)
            max_order_delta = max(max_order_delta, delta)
            # Candidate permutation can change FP32 reduction order only. The semantic result
            # must agree within a predeclared numerical tolerance, not necessarily bitwise.
            if delta > order_tolerance:
                same = False
    report = {
        "experiment": "P12_loop_consensus",
        "gate": "G0b_one_fit_score_cache",
        "source": source,
        "score_file": str(score_path),
        "candidate_shape": list(candidates.shape),
        "score_shape": list(scores.shape),
        "valid_fraction": float(valid.mean()),
        "valid_ids_in_range": bool(np.all((candidates[valid] >= 0) & (candidates[valid] < N_TILES))),
        "candidate_order_invariant": same,
        "candidate_order_max_abs_delta": max_order_delta,
        "candidate_order_tolerance": order_tolerance,
        "raw_finite_only_at_valid": bool(np.isfinite(scores[:, valid]).all() and np.isneginf(scores[:, ~valid]).all()),
        "loop_metrics": original_metrics,
        "second_order_loop_metrics": shuffled_metrics,
        "target_labels_loaded": False,
        "cal_target_opened": False,
        "dev_targets_opened": False,
        "test_accessed": False,
        "p8_labels_imported": False,
        "p10_final_checkpoint_imported": False,
        "p11_final_checkpoint_imported": False,
        "amp_used": False,
    }
    report["passes_G0b"] = bool(report["candidate_shape"] == [N_TILES, CANDIDATE_WIDTH] and report["score_shape"] == [NUM_DIRECTIONS, N_TILES, CANDIDATE_WIDTH] and report["valid_ids_in_range"] and report["candidate_order_invariant"] and report["raw_finite_only_at_valid"])
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p12_g0b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if not report["passes_G0b"]:
        raise RuntimeError("P12 G0b contract failed")


def prepare(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("P12 prepare must run through interactive CUDA session")
    seed_all()
    train, held = source_lists(args.prepare_report)
    device = torch.device("cuda")
    models = load_frozen_models(device)
    rows = []
    for index, source in enumerate(train + held):
        output = score_file(args.score_dir, source)
        if output.exists() and args.resume:
            candidates, valid, scores = load_score(args.score_dir, source)
            cached = True
        else:
            output = score_source(models, args.cache_dir, args.score_dir, source, device, args.pair_batch)
            candidates, valid, scores = load_score(args.score_dir, source)
            cached = False
        rows.append({"source": source, "index": index, "cache": str(output), "cached": cached, "candidate_sha256": array_sha(candidates.astype(np.int16)), "valid_sha256": array_sha(valid), "score_sha256": array_sha(scores), "valid_fraction": float(valid.mean())})
        print(json.dumps({"prepared": index + 1, "total": 160, "source": source, "cached": cached}), flush=True)
    report = {
        "experiment": "P12_loop_consensus",
        "gate": "G1_prepare_frozen_rank96_graph",
        "source_count": len(rows),
        "train_sources": train,
        "held_sources": held,
        "rows": rows,
        "candidate_k_per_affinity": CANDIDATE_K,
        "candidate_storage_width": CANDIDATE_WIDTH,
        "solver_max_edges": SOLVER_MAX_EDGES,
        "targets_opened": "none; cached tiles only",
        "cal_target_opened": False,
        "dev_targets_opened": False,
        "test_accessed": False,
        "p8_labels_imported": False,
        "p10_final_checkpoint_imported": False,
        "p11_final_checkpoint_imported": False,
        "rank96_mining_invoked": True,
        "rank96_ranker_invoked": True,
        "amp_used": False,
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p12_prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def accuracy_for_sources(cache_dir: Path, score_dir: Path, sources: list[str], lam: float) -> tuple[float, int, dict[str, float]]:
    values: list[float] = []
    invalid = 0
    loop_terms: list[dict[str, float]] = []
    for source in sources:
        candidates, valid, scores = load_score(score_dir, source)
        target, _ = load_labels(cache_dir, source)
        try:
            pred, info = solve_refined(candidates, scores, valid, lam)
            values.append(float(np.mean(pred == target)))
            loop_terms.append(info)
        except Exception:
            invalid += 1
            values.append(0.0)
    aggregate = {key: float(np.mean([row[key] for row in loop_terms])) for key in loop_terms[0]} if loop_terms else {}
    return float(np.mean(values)), invalid, aggregate


def train_eval(args: argparse.Namespace) -> None:
    train, held = source_lists(args.prepare_report)
    if not (args.work_dir / "p12_prepare_report.json").is_file():
        raise RuntimeError("P12 train_eval requires completed frozen score-cache prepare report")
    for source in train + held:
        if not score_file(args.score_dir, source).is_file():
            raise RuntimeError(f"missing frozen P12 score cache for {source}")
    train_grid = []
    for lam in LAMBDAS:
        accuracy, invalid, info = accuracy_for_sources(args.cache_dir, args.score_dir, train, lam)
        train_grid.append({"lambda": lam, "train_accuracy": accuracy, "invalid_decodes": invalid, "loop_info": info})
        print(json.dumps(train_grid[-1]), flush=True)
    eligible = [row for row in train_grid if row["invalid_decodes"] == 0]
    if not eligible:
        raise RuntimeError("all P12 calibration lambda values produced invalid decodes")
    selected = sorted(eligible, key=lambda row: (-float(row["train_accuracy"]), float(row["lambda"])))[0]
    held_accuracy, invalid, held_info = accuracy_for_sources(args.cache_dir, args.score_dir, held, float(selected["lambda"]))
    baseline = 0.0018988715277777778
    report = {
        "experiment": "P12_loop_consensus",
        "gate": "G1_calibrate128_held32",
        "selected_by": "precommitted FIT-train lambda grid; lower lambda breaks ties; held evaluated once after selection",
        "lambda_grid": train_grid,
        "selected_lambda": selected["lambda"],
        "selected_train_accuracy": selected["train_accuracy"],
        "baseline_held_accuracy": baseline,
        "refined_held_accuracy": held_accuracy,
        "held_delta_pp_vs_rank96": 100.0 * (held_accuracy - baseline),
        "held_loop_info": held_info,
        "invalid_decodes": invalid,
        "passes_G1": bool(invalid == 0 and held_accuracy >= baseline + 0.03),
        "decision": "PASS_to_CAL" if invalid == 0 and held_accuracy >= baseline + 0.03 else "REJECT_before_CAL",
        "targets_opened": "cached_FIT_labels_after_frozen_score_cache",
        "cal_target_opened": False,
        "dev_targets_opened": False,
        "test_accessed": False,
        "p8_labels_imported": False,
        "p10_final_checkpoint_imported": False,
        "p11_final_checkpoint_imported": False,
        "rank96_mining_invoked": False,
        "rank96_ranker_invoked": False,
        "amp_used": False,
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p12_g1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="P12 Sparse Loop-Consensus Edge Refiner")
    parser.add_argument("phase", choices=("g0a", "g0b", "prepare", "train_eval"))
    parser.add_argument("--cache-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache"))
    parser.add_argument("--prepare-report", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json"))
    parser.add_argument("--work-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus"))
    parser.add_argument("--score-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P12_loop_consensus\score_cache"))
    parser.add_argument("--pair-batch", type=int, default=4096)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "g0a":
        g0a(args)
    elif args.phase == "g0b":
        g0b(args)
    elif args.phase == "prepare":
        prepare(args)
    else:
        train_eval(args)


if __name__ == "__main__":
    main()
