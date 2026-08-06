"""Calibrate and confirm the corrected buddies edge budget on raw caches.

This CPU-only evaluator changes exactly one solver parameter: ``max_edges``.
The raw CandidateSeamRanker cache is replayed against the exact synthetic
corruption bytes from validation group ``10:12``.  Calibration is permanently
restricted to cache ids 10..17 and sweeps the predeclared budgets.  Confirmation
is permanently restricted to ids 18..21, loads one frozen budget, and compares
it pairwise with budget 512 without any sweep.

Permutation contract:
    cache/sample ``permutation[input_tile] = clean_cell``;
    solver ``placement[cell] = input_tile``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim

from canvas_data import CanvasDataset
from config import NFRAG, SEED, WORK_ROOT
from eval_seeded_qap import dense_rd
from imgio import assemble, train_val_split
from placement_metrics import neighbour_accuracy, placement_accuracy
from solve_buddies import solve_buddies_from_scores


SCHEMA_VERSION = 1
EXPERIMENT = "raw_buddies_solve_ssim_budget"
REPLAY_GROUP = (10, 12)
REPLAY_SEED = int(SEED)
DATASET_SEED = int(SEED + 400_000)
CALIBRATION_IDS = tuple(range(10, 18))
CONFIRMATION_IDS = tuple(range(18, 22))
BUDGETS = (64, 96, 128, 192, 256, 384, 512, 768, 900)
BASELINE_BUDGET = 512
REPAIR_PASSES = 0
MIN_MARGIN = 0.0
CACHE_TAG = "k64"


@dataclass(frozen=True)
class RawScene:
    image_id: int
    validation_name: str
    cache_path: Path
    cache_sha256: str
    candidate_ids: np.ndarray
    base_scores: np.ndarray
    permutation: np.ndarray
    tiles_uint8: np.ndarray
    target_uint8: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _uint8_tiles(sample: Mapping[str, torch.Tensor]) -> np.ndarray:
    tiles = sample["tiles"].permute(0, 2, 3, 1).numpy()
    return np.rint(tiles * 255.0).clip(0, 255).astype(np.uint8)


def _uint8_target(sample: Mapping[str, torch.Tensor]) -> np.ndarray:
    target = sample["clean"].permute(1, 2, 0).numpy()
    return np.rint(target * 255.0).clip(0, 255).astype(np.uint8)


def replay_group_10_12() -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, str]]:
    """Replay all 12 members, including RNG draws preceding confirmation ids."""

    start, count = REPLAY_GROUP
    _, validation_names = train_val_split()
    if start + count > len(validation_names):
        raise RuntimeError("fixed replay group exceeds the validation pool")
    random.seed(REPLAY_SEED)
    np.random.seed(REPLAY_SEED)
    torch.manual_seed(REPLAY_SEED)
    dataset = CanvasDataset(
        validation_names[start : start + count],
        real_prob=0.0,
        seed=DATASET_SEED,
    )
    samples: dict[int, dict[str, torch.Tensor]] = {}
    names: dict[int, str] = {}
    for local in range(count):
        # Do not skip calibration members during confirmation: __getitem__
        # consumes one global NumPy draw before constructing its local RNG.
        sample = dataset[local]
        absolute = start + local
        if not bool(sample["has_perm"]):
            raise RuntimeError("fixed synthetic replay unexpectedly returned a real sample")
        samples[absolute] = sample
        names[absolute] = validation_names[absolute]
    return samples, names


def _validate_raw_cache(
    path: Path,
    replayed_permutation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        required = {"candidate_ids", "candidate_scores", "permutation"}
        missing = sorted(required - set(stored.files))
        if missing:
            raise RuntimeError(f"{path} is missing fields {missing}")
        candidates = stored["candidate_ids"].astype(np.int64)
        flat_scores = stored["candidate_scores"].astype(np.float32)
        permutation = stored["permutation"].astype(np.int64)
        if "anchors" in stored.files and "directions" in stored.files:
            expected_anchors = np.repeat(np.arange(NFRAG, dtype=np.int64), 4)
            expected_directions = np.tile(np.arange(4, dtype=np.int64), NFRAG)
            if not np.array_equal(stored["anchors"], expected_anchors):
                raise RuntimeError(f"{path} rows are not anchor-major")
            if not np.array_equal(stored["directions"], expected_directions):
                raise RuntimeError(f"{path} directions are not UP,DOWN,LEFT,RIGHT")
    if candidates.ndim != 2 or candidates.shape[0] != NFRAG:
        raise RuntimeError(f"unexpected candidate_ids shape in {path}: {candidates.shape}")
    if flat_scores.shape != (NFRAG * 4, candidates.shape[1]):
        raise RuntimeError(f"unexpected candidate_scores shape in {path}: {flat_scores.shape}")
    if permutation.shape != (NFRAG,) or not np.array_equal(
        np.sort(permutation), np.arange(NFRAG, dtype=np.int64)
    ):
        raise RuntimeError(f"{path} permutation is not a tile->cell bijection")
    if not np.array_equal(permutation, replayed_permutation):
        raise RuntimeError(
            f"{path} permutation differs from fixed replay group {REPLAY_GROUP[0]}:{REPLAY_GROUP[1]}"
        )
    if np.any(candidates < 0) or np.any(candidates >= NFRAG):
        raise RuntimeError(f"{path} candidate id lies outside the tile bag")
    scores = flat_scores.reshape(NFRAG, 4, -1).transpose(1, 0, 2).copy()
    finite = np.isfinite(scores)
    if not bool(finite.any(axis=-1).all()):
        raise RuntimeError(f"{path} contains a fully invalid directional row")
    return candidates, scores, permutation


def load_raw_scenes(cache_dir: Path, image_ids: Sequence[int]) -> list[RawScene]:
    requested = tuple(map(int, image_ids))
    allowed = set(CALIBRATION_IDS) | set(CONFIRMATION_IDS)
    if len(requested) != len(set(requested)) or not set(requested) <= allowed:
        raise ValueError("image ids must be unique members of the fixed replay group")
    samples, names = replay_group_10_12()
    scenes = []
    for image_id in requested:
        cache_path = cache_dir / f"image_{image_id:04d}_{CACHE_TAG}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        sample = samples[image_id]
        replayed_permutation = sample["perm"].numpy().astype(np.int64)
        candidates, scores, permutation = _validate_raw_cache(
            cache_path, replayed_permutation
        )
        scenes.append(
            RawScene(
                image_id=image_id,
                validation_name=names[image_id],
                cache_path=cache_path.resolve(),
                cache_sha256=sha256_file(cache_path),
                candidate_ids=candidates,
                base_scores=scores,
                permutation=permutation,
                tiles_uint8=_uint8_tiles(sample),
                target_uint8=_uint8_target(sample),
            )
        )
        print(json.dumps({"loaded_image": image_id, "cache": str(cache_path)}), flush=True)
    return scenes


def scene_provenance(scene: RawScene) -> dict[str, Any]:
    return {
        "image": scene.image_id,
        "validation_name": scene.validation_name,
        "cache": str(scene.cache_path),
        "cache_sha256": scene.cache_sha256,
        "candidate_ids_sha256": array_sha256(scene.candidate_ids),
        "candidate_scores_sha256": array_sha256(scene.base_scores),
        "permutation_sha256": array_sha256(scene.permutation),
        "tiles_sha256": array_sha256(scene.tiles_uint8),
        "target_sha256": array_sha256(scene.target_uint8),
    }


def code_provenance() -> dict[str, str]:
    source_dir = Path(__file__).resolve().parent
    paths = {
        "eval_buddies_ssim_budget.py": Path(__file__).resolve(),
        "solve_buddies.py": source_dir / "solve_buddies.py",
        "eval_seeded_qap.py": source_dir / "eval_seeded_qap.py",
        "canvas_data.py": source_dir / "canvas_data.py",
        "distort.py": source_dir / "distort.py",
        "imgio.py": source_dir / "imgio.py",
        "placement_metrics.py": source_dir / "placement_metrics.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing provenance code files: " + ", ".join(missing))
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def dense_matrices(scene: RawScene) -> tuple[np.ndarray, np.ndarray]:
    """Convert the frozen raw rows once; budgets reuse identical matrices."""

    right_t, down_t = dense_rd(
        torch.from_numpy(scene.candidate_ids).long(),
        torch.from_numpy(np.ascontiguousarray(scene.base_scores)).float(),
    )
    right = np.ascontiguousarray(right_t.numpy(), dtype=np.float32)
    down = np.ascontiguousarray(down_t.numpy(), dtype=np.float32)
    if right.shape != (NFRAG, NFRAG) or down.shape != right.shape:
        raise AssertionError("dense_rd returned an invalid shape")
    if not np.isfinite(right).all() or not np.isfinite(down).all():
        raise AssertionError("dense_rd returned non-finite scores")
    if np.any(right < 0.0) or np.any(down < 0.0):
        raise AssertionError("dense_rd returned negative probabilities")
    if np.any(np.diag(right) != 0.0) or np.any(np.diag(down) != 0.0):
        raise AssertionError("dense_rd diagonal must be zero")
    return right, down


def evaluate_budget(
    scene: RawScene,
    right: np.ndarray,
    down: np.ndarray,
    budget: int,
) -> dict[str, Any]:
    if budget not in BUDGETS:
        raise ValueError("budget is outside the predeclared sweep")
    start = time.perf_counter()
    board, objective = solve_buddies_from_scores(
        right,
        down,
        max_edges=int(budget),
        min_margin=MIN_MARGIN,
        repair_passes=REPAIR_PASSES,
    )
    seconds = time.perf_counter() - start
    if board.shape != (NFRAG,) or not np.array_equal(
        np.sort(board), np.arange(NFRAG, dtype=np.int64)
    ):
        raise AssertionError("buddies solver did not return a strict tile permutation")
    truth_board = np.argsort(scene.permutation)
    placement = placement_accuracy(board, truth_board)[0]
    neighbour, right_accuracy, down_accuracy = neighbour_accuracy(board, truth_board)
    solved = assemble(scene.tiles_uint8, board)
    solve_only_ssim = sk_ssim(
        scene.target_uint8,
        solved,
        channel_axis=2,
        data_range=255,
    )
    return {
        "image": scene.image_id,
        "budget": int(budget),
        "solve_only_ssim": float(solve_only_ssim),
        "neighbour": float(neighbour),
        "right": float(right_accuracy),
        "down": float(down_accuracy),
        "placement": float(placement),
        "objective": float(objective),
        "solver_seconds": float(seconds),
        "board_sha256": array_sha256(board.astype(np.int64, copy=False)),
    }


def summarize_budget(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize an empty budget row set")
    return {
        "images": float(len(rows)),
        **{
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in (
                "solve_only_ssim",
                "neighbour",
                "right",
                "down",
                "placement",
                "objective",
                "solver_seconds",
            )
        },
    }


def select_budget(summaries: Mapping[int, Mapping[str, float]]) -> int:
    """Primary mean SSIM, then mean neighbour, then smaller edge budget."""

    if set(map(int, summaries)) != set(BUDGETS):
        raise ValueError("calibration summaries do not cover the exact budget grid")
    for budget, row in summaries.items():
        if not np.isfinite(float(row["solve_only_ssim"])) or not np.isfinite(
            float(row["neighbour"])
        ):
            raise ValueError(f"budget {budget} has a non-finite selection metric")
    return int(
        max(
            summaries,
            key=lambda budget: (
                float(summaries[budget]["solve_only_ssim"]),
                float(summaries[budget]["neighbour"]),
                -int(budget),
            ),
        )
    )


def paired_confirmation_summary(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(candidate_rows) != len(baseline_rows) or not candidate_rows:
        raise ValueError("paired confirmation rows must be non-empty and aligned")
    candidate_by_image = {int(row["image"]): row for row in candidate_rows}
    baseline_by_image = {int(row["image"]): row for row in baseline_rows}
    if set(candidate_by_image) != set(baseline_by_image):
        raise ValueError("candidate and baseline confirmation image ids differ")
    per_image = []
    for image in sorted(candidate_by_image):
        candidate = candidate_by_image[image]
        baseline = baseline_by_image[image]
        per_image.append(
            {
                "image": image,
                "solve_only_ssim": float(candidate["solve_only_ssim"])
                - float(baseline["solve_only_ssim"]),
                "neighbour": float(candidate["neighbour"])
                - float(baseline["neighbour"]),
                "placement": float(candidate["placement"])
                - float(baseline["placement"]),
            }
        )
    return {
        "mean_solve_only_ssim": float(
            np.mean([row["solve_only_ssim"] for row in per_image])
        ),
        "mean_neighbour": float(np.mean([row["neighbour"] for row in per_image])),
        "mean_placement": float(np.mean([row["placement"] for row in per_image])),
        "positive_ssim_scenes": int(sum(row["solve_only_ssim"] > 0.0 for row in per_image)),
        "per_image": per_image,
    }


def _fixed_contract() -> dict[str, Any]:
    return {
        "replay_group": f"{REPLAY_GROUP[0]}:{REPLAY_GROUP[1]}",
        "replay_seed": REPLAY_SEED,
        "dataset_seed": DATASET_SEED,
        "cache_tag": CACHE_TAG,
        "budgets": list(BUDGETS),
        "baseline_budget": BASELINE_BUDGET,
        "repair_passes": REPAIR_PASSES,
        "min_margin": MIN_MARGIN,
        "orientation": "fixed",
        "score_source": "raw_cached_candidate_seam_ranker",
    }


def validate_frozen_config(
    frozen: Mapping[str, Any],
    current_code: Mapping[str, str],
) -> int:
    if int(frozen.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError("frozen config schema is incompatible")
    if frozen.get("experiment") != EXPERIMENT or frozen.get("status") != "frozen":
        raise RuntimeError("confirmation requires a successfully frozen calibration")
    if tuple(frozen.get("calibration_ids", ())) != CALIBRATION_IDS:
        raise RuntimeError("frozen calibration ids differ from 10..17")
    if tuple(frozen.get("confirmation_ids_reserved", ())) != CONFIRMATION_IDS:
        raise RuntimeError("frozen confirmation ids differ from reserved 18..21")
    if frozen.get("contract") != _fixed_contract():
        raise RuntimeError("frozen solver/replay contract differs from current constants")
    selected = frozen.get("selected_budget")
    if not isinstance(selected, int) or selected not in BUDGETS:
        raise RuntimeError("frozen selected budget is invalid")
    if dict(frozen.get("code_provenance", {})) != dict(current_code):
        raise RuntimeError("solver/replay code hashes differ from frozen calibration")
    if not isinstance(frozen.get("calibration_scene_provenance_digest"), str):
        raise RuntimeError("frozen config lacks calibration provenance digest")
    if not isinstance(frozen.get("calibration_report_sha256"), str):
        raise RuntimeError("frozen config lacks calibration report hash")
    return int(selected)


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    code_hashes = code_provenance()
    scenes = load_raw_scenes(args.cache_dir, CALIBRATION_IDS)
    provenance = [scene_provenance(scene) for scene in scenes]
    per_budget: dict[int, list[dict[str, Any]]] = {budget: [] for budget in BUDGETS}
    dense_seconds = []
    for scene in scenes:
        start = time.perf_counter()
        right, down = dense_matrices(scene)
        dense_seconds.append(time.perf_counter() - start)
        for budget in BUDGETS:
            row = evaluate_budget(scene, right, down, budget)
            per_budget[budget].append(row)
            print(json.dumps(row), flush=True)
    summaries = {budget: summarize_budget(per_budget[budget]) for budget in BUDGETS}
    selected = select_budget(summaries)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "phase": "calibration",
        "status": "frozen",
        "calibration_ids": list(CALIBRATION_IDS),
        "confirmation_ids_reserved": list(CONFIRMATION_IDS),
        "contract": _fixed_contract(),
        "cache_dir": str(args.cache_dir.resolve()),
        "scene_provenance": provenance,
        "scene_provenance_digest": canonical_digest(provenance),
        "code_provenance": code_hashes,
        "mean_dense_rd_seconds": float(np.mean(dense_seconds)),
        "grid": {str(budget): summaries[budget] for budget in BUDGETS},
        "grid_per_image": {str(budget): per_budget[budget] for budget in BUDGETS},
        "selected_budget": selected,
        "selected_metrics": summaries[selected],
        "selection_order": ["mean_solve_only_ssim", "mean_neighbour", "smaller_budget"],
    }
    write_json(args.report, report)
    report_hash = sha256_file(args.report)
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "frozen",
        "calibration_ids": list(CALIBRATION_IDS),
        "confirmation_ids_reserved": list(CONFIRMATION_IDS),
        "contract": _fixed_contract(),
        "selected_budget": selected,
        "selected_calibration_metrics": summaries[selected],
        "calibration_report": str(args.report.resolve()),
        "calibration_report_sha256": report_hash,
        "calibration_scene_provenance_digest": canonical_digest(provenance),
        "code_provenance": code_hashes,
    }
    write_json(args.frozen_config, frozen)
    print(
        json.dumps(
            {
                "status": "frozen",
                "selected_budget": selected,
                "report": str(args.report),
                "frozen_config": str(args.frozen_config),
            }
        ),
        flush=True,
    )
    return report


def run_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    frozen_path = args.frozen_config.resolve()
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    current_code = code_provenance()
    selected = validate_frozen_config(frozen, current_code)
    if set(CALIBRATION_IDS) & set(CONFIRMATION_IDS):
        raise AssertionError("hard-coded calibration and confirmation ids overlap")
    calibration_report = Path(str(frozen["calibration_report"]))
    if not calibration_report.is_file() or sha256_file(calibration_report) != frozen[
        "calibration_report_sha256"
    ]:
        raise RuntimeError("frozen calibration report is missing or its hash changed")

    scenes = load_raw_scenes(args.cache_dir, CONFIRMATION_IDS)
    provenance = [scene_provenance(scene) for scene in scenes]
    candidate_rows = []
    baseline_rows = []
    dense_seconds = []
    for scene in scenes:
        start = time.perf_counter()
        right, down = dense_matrices(scene)
        dense_seconds.append(time.perf_counter() - start)
        candidate = evaluate_budget(scene, right, down, selected)
        if selected == BASELINE_BUDGET:
            baseline = dict(candidate)
        else:
            baseline = evaluate_budget(scene, right, down, BASELINE_BUDGET)
        candidate_rows.append(candidate)
        baseline_rows.append(baseline)
        print(
            json.dumps(
                {
                    "image": scene.image_id,
                    "selected": candidate,
                    "baseline_512": baseline,
                }
            ),
            flush=True,
        )
    candidate_summary = summarize_budget(candidate_rows)
    baseline_summary = summarize_budget(baseline_rows)
    paired = paired_confirmation_summary(candidate_rows, baseline_rows)
    checks = {
        "calibration_confirmation_disjoint": True,
        "raw_cache_only": True,
        "fixed_solver_contract": True,
        "paired_mean_solve_only_ssim_positive": paired["mean_solve_only_ssim"] > 0.0,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "phase": "confirmation",
        "status": "pass" if all(checks.values()) else "fail_closed",
        "frozen_config": str(frozen_path),
        "frozen_config_sha256": sha256_file(frozen_path),
        "calibration_ids": list(CALIBRATION_IDS),
        "confirmation_ids": list(CONFIRMATION_IDS),
        "contract": _fixed_contract(),
        "selected_budget": selected,
        "baseline_budget": BASELINE_BUDGET,
        "cache_dir": str(args.cache_dir.resolve()),
        "scene_provenance": provenance,
        "scene_provenance_digest": canonical_digest(provenance),
        "code_provenance": current_code,
        "mean_dense_rd_seconds": float(np.mean(dense_seconds)),
        "candidate": candidate_summary,
        "baseline_512": baseline_summary,
        "paired_delta": paired,
        "checks": checks,
        "candidate_per_image": candidate_rows,
        "baseline_512_per_image": baseline_rows,
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    calibrate = subparsers.add_parser(
        "calibrate", help="sweep fixed budgets on ids 10..17 and freeze one"
    )
    calibrate.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    calibrate.add_argument("--report", type=Path, required=True)
    calibrate.add_argument("--frozen-config", type=Path, required=True)

    confirm = subparsers.add_parser(
        "confirm", help="compare one frozen budget with 512 on ids 18..21"
    )
    confirm.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    confirm.add_argument("--frozen-config", type=Path, required=True)
    confirm.add_argument("--report", type=Path, required=True)
    return parser


def _failure_payload(phase: str, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "phase": phase,
        "status": "fail_closed",
        "contract": _fixed_contract(),
        "error": f"{type(error).__name__}: {error}",
    }


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.phase == "calibrate":
            run_calibration(args)
        elif args.phase == "confirm":
            report = run_confirmation(args)
            if report["status"] != "pass":
                raise SystemExit(2)
        else:  # pragma: no cover
            raise AssertionError(args.phase)
    except SystemExit:
        raise
    except Exception as error:
        failure = _failure_payload(args.phase, error)
        write_json(args.report, failure)
        if args.phase == "calibrate":
            write_json(
                args.frozen_config,
                {
                    **failure,
                    "calibration_ids": list(CALIBRATION_IDS),
                    "confirmation_ids_reserved": list(CONFIRMATION_IDS),
                    "selected_budget": None,
                },
            )
        print(json.dumps(failure, indent=2), flush=True)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
