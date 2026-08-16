"""P13 CPGS-24 — Component-Pose Global Synchronization.

Pre-registered in P13_PRE_REGISTRATION.md before this source file was created.
This harness consumes only frozen P12 rank96 score-cache artifacts and, during
G1 only, cached FIT labels from the pre-existing P10 cache.  It never mines a
new graph, reads target PNGs, or accesses CAL/DEV/test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from candidate_rank import DOWN, LEFT, RIGHT, UP, NUM_DIRECTIONS

GRID = 24
N_TILES = GRID * GRID
CANDIDATE_WIDTH = 128
SEED = 20260816
THRESHOLDS = (0.00, 0.05, 0.10, 0.20)
HUBER_DELTA = 1.0
IRLS_ITERS = 8


@dataclass(frozen=True)
class Constraint:
    src: int
    dst: int
    dx: float
    dy: float
    weight: float


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def canonical_bytes(array: np.ndarray) -> bytes:
    value = np.ascontiguousarray(array)
    return (
        str(value.dtype).encode("utf-8")
        + b"\0"
        + repr(value.shape).encode("utf-8")
        + b"\0"
        + value.tobytes(order="C")
    )


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(canonical_bytes(array)).hexdigest()


def json_scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(-1)[0])


def cache_path(cache_dir: Path, source: str) -> Path:
    return cache_dir / f"{Path(source).stem}.npz"


def load_score_cache(cache_dir: Path, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = cache_path(cache_dir, source)
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen score cache for {source}: {path}")
    with np.load(path, allow_pickle=False) as raw:
        # Validate exact persisted dtypes before any solver-side conversion.
        candidates_raw = raw["candidates"].copy()
        valid_raw = raw["valid"].copy()
        scores_raw = raw["scores"].copy()
        recorded_source = json_scalar(raw["source"])
        expected_c = json_scalar(raw["candidate_sha256"])
        expected_v = json_scalar(raw["valid_sha256"])
        expected_s = json_scalar(raw["score_sha256"])
    if recorded_source != source:
        raise RuntimeError(f"cache source mismatch: expected {source}, got {recorded_source}")
    if candidates_raw.shape != (N_TILES, CANDIDATE_WIDTH):
        raise RuntimeError(f"unexpected candidate shape {candidates_raw.shape}")
    if valid_raw.shape != candidates_raw.shape:
        raise RuntimeError(f"unexpected valid shape {valid_raw.shape}")
    if scores_raw.shape != (NUM_DIRECTIONS, N_TILES, CANDIDATE_WIDTH):
        raise RuntimeError(f"unexpected score shape {scores_raw.shape}")
    if array_sha(candidates_raw) != expected_c or array_sha(valid_raw) != expected_v or array_sha(scores_raw) != expected_s:
        raise RuntimeError("frozen score-cache SHA contract failed")
    candidates = candidates_raw.astype(np.int32, copy=False)
    valid = valid_raw.astype(bool, copy=False)
    scores = scores_raw.astype(np.float32, copy=False)
    if np.any(valid & ((candidates < 0) | (candidates >= N_TILES))):
        raise RuntimeError("valid candidate identity outside [0,575]")
    return candidates, valid, scores


def source_lists(prepare_report: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(prepare_report.read_text(encoding="utf-8"))
    train = [str(x) for x in payload["train_sources"]]
    held = [str(x) for x in payload["held_sources"]]
    if len(train) != 128 or len(held) != 32 or set(train).intersection(held):
        raise RuntimeError("P13 requires exact locked 128/32 source-disjoint split")
    return train, held


def load_cached_labels(label_cache: Path, source: str) -> np.ndarray:
    """Load only already-frozen FIT labels.  Never used by G0a/G0b."""
    path = cache_path(label_cache, source)
    with np.load(path, allow_pickle=False) as raw:
        target = raw["target_tile_to_slot"].copy().astype(np.int64, copy=False)
    if target.shape != (N_TILES,) or np.unique(target).size != N_TILES:
        raise RuntimeError(f"invalid cached FIT label permutation: {path}")
    return target


def sigmoid(x: float) -> float:
    if x >= 0:
        return float(1.0 / (1.0 + math.exp(-min(x, 50.0))))
    exp = math.exp(max(x, -50.0))
    return float(exp / (1.0 + exp))


def standardize_rows(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per anchor/direction normalized frozen scores; invalid positions stay -inf."""
    out = np.full(scores.shape, -np.inf, dtype=np.float64)
    for direction in range(NUM_DIRECTIONS):
        for anchor in range(N_TILES):
            mask = valid[anchor] & np.isfinite(scores[direction, anchor])
            values = scores[direction, anchor, mask].astype(np.float64, copy=False)
            if values.size == 0:
                continue
            mean = float(values.mean())
            std = max(float(values.std()), 1e-6)
            out[direction, anchor, mask] = (values - mean) / std
    return out


def reverse_direction(direction: int) -> int:
    mapping = {RIGHT: LEFT, LEFT: RIGHT, DOWN: UP, UP: DOWN}
    return mapping[direction]


def find_candidate_index(candidates: np.ndarray, valid: np.ndarray, anchor: int, target: int) -> int | None:
    hits = np.flatnonzero(valid[anchor] & (candidates[anchor] == int(target)))
    if hits.size == 0:
        return None
    # Valid duplicate IDs would make frozen cache malformed; keep a deterministic audit.
    if hits.size != 1:
        raise RuntimeError(f"duplicate valid candidate identity anchor={anchor} target={target}")
    return int(hits[0])


def deterministic_row_permutation(source: str, anchor: int, width: int) -> np.ndarray:
    keyed: list[tuple[bytes, int]] = []
    for pos in range(width):
        text = f"P13-candidate-order-v1|{SEED}|{source}|{anchor}|{pos}".encode("utf-8")
        keyed.append((hashlib.blake2b(text, digest_size=16).digest(), pos))
    return np.asarray([pos for _, pos in sorted(keyed)], dtype=np.int64)


def permute_candidate_axes(source: str, candidates: np.ndarray, valid: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_c = np.empty_like(candidates)
    out_v = np.empty_like(valid)
    out_s = np.empty_like(scores)
    for anchor in range(N_TILES):
        order = deterministic_row_permutation(source, anchor, CANDIDATE_WIDTH)
        out_c[anchor] = candidates[anchor, order]
        out_v[anchor] = valid[anchor, order]
        out_s[:, anchor] = scores[:, anchor, order]
    return out_c, out_v, out_s


def select_constraints(candidates: np.ndarray, valid: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[list[Constraint], dict[str, float]]:
    """Choose one high-confidence RIGHT and DOWN relative-pose observation per tile.

    Candidate enumeration is explicitly sorted by tile ID, not storage position, so
    a bijection-preserving row permutation cannot affect the selected graph.
    """
    z = standardize_rows(scores, valid)
    constraints: list[Constraint] = []
    primary_conf: list[float] = []
    reciprocal_present = 0
    for direction, dx, dy in ((RIGHT, 1.0, 0.0), (DOWN, 0.0, 1.0)):
        reverse = reverse_direction(direction)
        for src in range(N_TILES):
            choices: list[tuple[float, int, float]] = []
            mask = valid[src] & np.isfinite(z[direction, src])
            candidate_ids = sorted(int(x) for x in candidates[src, mask])
            finite_values = z[direction, src, mask]
            top_two = np.sort(finite_values)[-2:]
            second = float(top_two[0]) if top_two.size == 2 else float(top_two[-1] - 1.0)
            for dst in candidate_ids:
                idx = find_candidate_index(candidates, valid, src, dst)
                assert idx is not None
                own = float(z[direction, src, idx])
                rev_idx = find_candidate_index(candidates, valid, dst, src)
                if rev_idx is not None and np.isfinite(z[reverse, dst, rev_idx]):
                    reciprocal = sigmoid(float(z[reverse, dst, rev_idx]))
                    reciprocal_present += 1
                else:
                    reciprocal = 0.0
                # Margin prevents uniformly ambiguous rows from dominating.
                margin = sigmoid(own - second)
                confidence = (0.65 * sigmoid(own) + 0.35 * reciprocal) * (0.5 + 0.5 * margin)
                choices.append((confidence, dst, margin))
            if not choices:
                continue
            # Fixed tile-ID tie-break makes candidate-storage order irrelevant.
            confidence, dst, margin = max(choices, key=lambda row: (row[0], -row[1]))
            if confidence >= threshold:
                constraints.append(Constraint(src, dst, dx, dy, float(confidence)))
                primary_conf.append(float(confidence))
    info = {
        "constraint_count": float(len(constraints)),
        "mean_selected_weight": float(np.mean(primary_conf)) if primary_conf else 0.0,
        "reciprocal_hits": float(reciprocal_present),
    }
    return constraints, info


def connected_components(constraints: list[Constraint]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(N_TILES)]
    for edge in constraints:
        adj[edge.src].append(edge.dst)
        adj[edge.dst].append(edge.src)
    seen = np.zeros(N_TILES, dtype=bool)
    components: list[list[int]] = []
    for root in range(N_TILES):
        if seen[root]:
            continue
        stack = [root]
        seen[root] = True
        comp: list[int] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in adj[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        components.append(sorted(comp))
    return components


def solve_component_pose(constraints: list[Constraint], components: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """IRLS translation synchronization with a deterministic anchor per component."""
    coords = np.zeros((N_TILES, 2), dtype=np.float64)
    if not constraints:
        return coords, np.zeros(0, dtype=np.float64)
    effective = np.asarray([max(edge.weight, 1e-6) for edge in constraints], dtype=np.float64)
    component_of = np.empty(N_TILES, dtype=np.int32)
    for idx, comp in enumerate(components):
        component_of[comp] = idx
    for _ in range(IRLS_ITERS):
        rows: list[np.ndarray] = []
        bx: list[float] = []
        by: list[float] = []
        for edge_index, edge in enumerate(constraints):
            row = np.zeros(N_TILES, dtype=np.float64)
            weight = math.sqrt(max(effective[edge_index], 1e-10))
            row[edge.src] = -weight
            row[edge.dst] = weight
            rows.append(row)
            bx.append(weight * edge.dx)
            by.append(weight * edge.dy)
        # Gauge-fix each component locally. Singletons are valid components too.
        for comp in components:
            row = np.zeros(N_TILES, dtype=np.float64)
            row[comp[0]] = 1000.0
            rows.append(row)
            bx.append(0.0)
            by.append(0.0)
        design = np.vstack(rows)
        coords[:, 0] = np.linalg.lstsq(design, np.asarray(bx), rcond=None)[0]
        coords[:, 1] = np.linalg.lstsq(design, np.asarray(by), rcond=None)[0]
        residuals = np.asarray(
            [
                math.hypot((coords[e.dst, 0] - coords[e.src, 0]) - e.dx, (coords[e.dst, 1] - coords[e.src, 1]) - e.dy)
                for e in constraints
            ],
            dtype=np.float64,
        )
        robust = np.minimum(1.0, HUBER_DELTA / np.maximum(residuals, 1e-9))
        effective = np.asarray([edge.weight for edge in constraints], dtype=np.float64) * robust
    # Deterministic non-overlap spacing only for disconnected components; fully connected
    # graphs retain their learned relative global coordinate frame unchanged.
    for component_index, comp in enumerate(components):
        if len(components) <= 1:
            break
        offset = np.asarray([float(component_index * (GRID + 2)), 0.0])
        coords[comp] += offset - coords[comp].mean(axis=0, keepdims=True)
    return coords, effective


def hungarian_project(coords: np.ndarray) -> np.ndarray:
    if coords.shape != (N_TILES, 2) or not np.isfinite(coords).all():
        raise RuntimeError("non-finite continuous pose")
    lattice = np.asarray([(float(slot % GRID), float(slot // GRID)) for slot in range(N_TILES)], dtype=np.float64)
    centered = coords - coords.mean(axis=0, keepdims=True)
    lattice_centered = lattice - lattice.mean(axis=0, keepdims=True)
    denom = float(np.sum(centered * centered))
    numer = float(np.sum(lattice_centered * lattice_centered))
    scale = math.sqrt(numer / max(denom, 1e-9))
    aligned = centered * scale + lattice.mean(axis=0, keepdims=True)
    cost = np.sum((aligned[:, None, :] - lattice[None, :, :]) ** 2, axis=2)
    rows, cols = linear_sum_assignment(cost)
    assignment = np.full(N_TILES, -1, dtype=np.int64)
    assignment[rows] = cols
    if np.any(assignment < 0) or np.unique(assignment).size != N_TILES:
        raise RuntimeError("Hungarian projection failed strict bijection")
    return assignment


def decode_graph(candidates: np.ndarray, valid: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[np.ndarray, dict[str, float]]:
    constraints, info = select_constraints(candidates, valid, scores, threshold)
    components = connected_components(constraints)
    coords, effective = solve_component_pose(constraints, components)
    assignment = hungarian_project(coords)
    info.update(
        {
            "component_count": float(len(components)),
            "largest_component": float(max(map(len, components))),
            "min_effective_weight": float(effective.min()) if effective.size else 0.0,
            "mean_effective_weight": float(effective.mean()) if effective.size else 0.0,
            "finite_coordinate_fraction": float(np.isfinite(coords).mean()),
        }
    )
    return assignment, info


def make_synthetic_graph(corrupt: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.full((N_TILES, CANDIDATE_WIDTH), -1, dtype=np.int32)
    valid = np.zeros_like(candidates, dtype=bool)
    scores = np.full((NUM_DIRECTIONS, N_TILES, CANDIDATE_WIDTH), -8.0, dtype=np.float32)
    target = np.arange(N_TILES, dtype=np.int64)
    rng = np.random.default_rng(20260816)
    for tile in range(N_TILES):
        x, y = tile % GRID, tile // GRID
        neighbor_for_direction = {
            RIGHT: tile + 1 if x + 1 < GRID else None,
            LEFT: tile - 1 if x > 0 else None,
            DOWN: tile + GRID if y + 1 < GRID else None,
            UP: tile - GRID if y > 0 else None,
        }
        ids = [value for value in neighbor_for_direction.values() if value is not None]
        while len(ids) < 12:
            candidate = int(rng.integers(0, N_TILES))
            if candidate != tile and candidate not in ids:
                ids.append(candidate)
        ids = sorted(ids)
        candidates[tile, : len(ids)] = ids
        valid[tile, : len(ids)] = True
        for direction, neighbor in neighbor_for_direction.items():
            if neighbor is not None:
                idx = ids.index(neighbor)
                scores[direction, tile, idx] = 12.0
        # Deterministic distractor values keep each row numerically non-degenerate.
        for pos, candidate in enumerate(ids):
            for direction in range(NUM_DIRECTIONS):
                if scores[direction, tile, pos] < 0:
                    scores[direction, tile, pos] = -2.0 - 0.01 * float((tile + candidate + direction) % 19)
    if corrupt:
        # Deliberately replace one right bridge with a geometrically impossible but high score.
        src, bad = 12 * GRID + 11, 23 * GRID + 23
        idx = int(np.flatnonzero(candidates[src] == bad)[0]) if np.any(candidates[src] == bad) else 11
        candidates[src, idx] = bad
        valid[src, idx] = True
        scores[RIGHT, src, idx] = 20.0
    return candidates, valid, scores, target


def run_g0a(args: argparse.Namespace) -> None:
    seed_all()
    c, v, s, expected = make_synthetic_graph(corrupt=False)
    decoded, clean_info = decode_graph(c, v, s, threshold=0.00)
    clean_exact = bool(np.array_equal(decoded, expected))
    pc, pv, ps = permute_candidate_axes("synthetic_clean", c, v, s)
    permuted, _ = decode_graph(pc, pv, ps, threshold=0.00)
    order_invariant = bool(np.array_equal(decoded, permuted))
    # A robust Huber factor must reduce the influence of a gross translation residual.
    small_factor = min(1.0, HUBER_DELTA / 0.10)
    gross_factor = min(1.0, HUBER_DELTA / 10.0)
    cc, cv, cs, _ = make_synthetic_graph(corrupt=True)
    _, corrupt_info = decode_graph(cc, cv, cs, threshold=0.00)
    report = {
        "experiment": "P13_CPGS-24",
        "gate": "G0a_synthetic_relative_translation",
        "clean_exact_bijection": clean_exact,
        "candidate_order_invariant": order_invariant,
        "small_residual_huber_factor": small_factor,
        "gross_residual_huber_factor": gross_factor,
        "corruption_downweighted_by_formula": bool(gross_factor < small_factor),
        "clean_info": clean_info,
        "corrupt_info": corrupt_info,
        "passes_G0a": bool(clean_exact and order_invariant and gross_factor < small_factor),
        "targets_opened": False,
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
    (args.work_dir / "p13_g0a_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passes_G0a"]:
        raise SystemExit("P13 G0a FAILED")


def run_g0b(args: argparse.Namespace) -> None:
    seed_all()
    candidates, valid, scores = load_score_cache(args.score_dir, args.source)
    decoded, info = decode_graph(candidates, valid, scores, threshold=args.threshold)
    pc, pv, ps = permute_candidate_axes(args.source, candidates, valid, scores)
    permuted, permuted_info = decode_graph(pc, pv, ps, threshold=args.threshold)
    report = {
        "experiment": "P13_CPGS-24",
        "gate": "G0b_one_FIT_frozen_cache",
        "source": args.source,
        "threshold": args.threshold,
        "strict_bijection": bool(np.unique(decoded).size == N_TILES and decoded.min() >= 0 and decoded.max() < N_TILES),
        "finite_pose": bool(info["finite_coordinate_fraction"] == 1.0),
        "deterministic_candidate_order_invariant": bool(np.array_equal(decoded, permuted)),
        "decode_info": info,
        "permuted_decode_info": permuted_info,
        "passes_G0b": False,
        "targets_opened": False,
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
    report["passes_G0b"] = bool(report["strict_bijection"] and report["finite_pose"] and report["deterministic_candidate_order_invariant"])
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p13_g0b_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passes_G0b"]:
        raise SystemExit("P13 G0b FAILED")


def accuracy_for_sources(score_dir: Path, label_cache: Path, sources: list[str], threshold: float) -> tuple[float, int, dict[str, float]]:
    values: list[float] = []
    invalid = 0
    rows: list[dict[str, float]] = []
    for source in sources:
        candidates, valid, scores = load_score_cache(score_dir, source)
        decoded, info = decode_graph(candidates, valid, scores, threshold=threshold)
        target = load_cached_labels(label_cache, source)
        if np.unique(decoded).size != N_TILES:
            invalid += 1
            continue
        values.append(float(np.mean(decoded == target)))
        rows.append(info)
    if len(values) != len(sources):
        invalid += len(sources) - len(values)
    aggregate = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {}
    return float(np.mean(values)) if values else 0.0, invalid, aggregate


def run_train_eval(args: argparse.Namespace) -> None:
    seed_all()
    train_sources, held_sources = source_lists(args.prepare_report)
    grid = []
    for threshold in THRESHOLDS:
        accuracy, invalid, info = accuracy_for_sources(args.score_dir, args.label_cache, train_sources, threshold)
        row = {"threshold": threshold, "train_accuracy": accuracy, "invalid_decodes": invalid, "pose_info": info}
        grid.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    selected = max(grid, key=lambda row: (row["train_accuracy"], -row["threshold"]))
    held_accuracy, invalid, held_info = accuracy_for_sources(args.score_dir, args.label_cache, held_sources, float(selected["threshold"]))
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    baseline = float(baseline_report["baseline_held_accuracy"])
    report = {
        "experiment": "P13_component_pose_synchronization",
        "gate": "G1_calibrate128_held32",
        "selected_by": "precommitted FIT-train threshold grid; lower threshold breaks ties; held evaluated once after selection",
        "threshold_grid": grid,
        "selected_threshold": selected["threshold"],
        "selected_train_accuracy": selected["train_accuracy"],
        "baseline_held_accuracy": baseline,
        "refined_held_accuracy": held_accuracy,
        "held_delta_pp_vs_rank96": 100.0 * (held_accuracy - baseline),
        "held_pose_info": held_info,
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
    (args.work_dir / "p13_g1_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="P13 CPGS-24 frozen relative-translation synchronization")
    parser.add_argument("--phase", choices=("g0a", "g0b", "train_eval"), required=True)
    parser.add_argument("--work-dir", type=Path, default=Path(r"E:\\pazzle_work\\pazzle_fixed_orientation_20260813\\P13_component_pose"))
    parser.add_argument("--score-dir", type=Path, default=Path(r"E:\\pazzle_work\\pazzle_fixed_orientation_20260813\\P12_loop_consensus\\score_cache"))
    parser.add_argument("--label-cache", type=Path, default=Path(r"E:\\pazzle_work\\pazzle_fixed_orientation_20260813\\P10_sinkhorn_refiner\\g1\\cache"))
    parser.add_argument("--prepare-report", type=Path, default=Path(r"E:\\pazzle_work\\pazzle_fixed_orientation_20260813\\P10_sinkhorn_refiner\\g1\\p10_g1_prepare_report.json"))
    parser.add_argument("--baseline-report", type=Path, default=Path(r"E:\\pazzle_work\\pazzle_fixed_orientation_20260813\\P12_loop_consensus\\p12_g1_report.json"))
    parser.add_argument("--source", type=str, default="img_000025.png")
    parser.add_argument("--threshold", type=float, default=0.00)
    args = parser.parse_args()
    if args.phase == "g0a":
        run_g0a(args)
    elif args.phase == "g0b":
        run_g0b(args)
    else:
        run_train_eval(args)


if __name__ == "__main__":
    main()
