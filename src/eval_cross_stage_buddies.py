"""Localize I21 solver transfer by crossing component and packing scores.

The old cache split is deliberately hard-coded:

* calibration: validation cache IDs 10..17;
* confirmation: validation cache IDs 18..21;
* deterministic corruption replay group: 10:12.

Calibration evaluates exactly four structural variants and freezes a non-base
variant only when its *paired mean solve-only SSIM* is strictly above the
raw->raw baseline.  Confirmation evaluates only raw->raw and that one frozen
choice; there is no confirmation sweep or fallback selection.

The score stages are separated without changing any raw logit:

``raw_to_raw``
    raw ranker scores construct components and pack/fill them;
``raw_components_i21_pack``
    raw constructs components, I21 fusion packs/fills them;
``i21_components_raw_pack``
    I21 constructs components, raw packs/fills them;
``i21_to_i21``
    I21 performs both stages.

Examples::

    python src/eval_cross_stage_buddies.py calibrate \
      --report E:/pazzle_work/gates/cross_stage_calibration.json \
      --frozen-config E:/pazzle_work/gates/cross_stage_frozen.json
    python src/eval_cross_stage_buddies.py confirm \
      --frozen-config E:/pazzle_work/gates/cross_stage_frozen.json \
      --report E:/pazzle_work/gates/cross_stage_confirmation.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim

from config import NFRAG, SEED, WORK_ROOT
from eval_calibrated_buddies import component_metrics
from eval_rank_transplant import (
    CachedScene,
    _edge_metrics,
    _load_spatial,
    load_cached_scenes,
    scene_provenance,
)
from eval_seeded_qap import dense_rd
from imgio import assemble
from placement_metrics import neighbour_accuracy, placement_accuracy
from rank_transplant import assert_disjoint_phases, row_zscore
from solve_buddies import (
    build_buddies_components,
    solve_components_from_scores,
)


SCHEMA_VERSION = 1
EXPERIMENT = "cross_stage_buddies_i21_transfer"

CALIBRATION_IDS: tuple[int, ...] = tuple(range(10, 18))
CONFIRMATION_IDS: tuple[int, ...] = tuple(range(18, 22))
REPLAY_GROUPS: tuple[tuple[int, int], ...] = ((10, 12),)
REPLAY_GROUP_TEXT = "10:12"
REPLAY_SEED = SEED

I21_ALPHA = 1.25
BUDGET = 512
REPAIR_PASSES = 0
CACHE_TAG = "k64"
BASELINE_VARIANT = "raw_to_raw"

MEAN_METRICS: tuple[str, ...] = (
    "candidate_recall",
    "edge_r1",
    "packing_edge_r1",
    "placement",
    "neighbour",
    "right",
    "down",
    "solve_only_ssim",
    "objective",
    "component_count",
    "largest_component",
    "largest_pure_component",
    "nontrivial_tile_coverage",
    "translation_aligned_accuracy",
    "internal_edge_precision",
    "internal_edges",
)
PRIMARY_METRICS: tuple[str, ...] = (
    "solve_only_ssim",
    "neighbour",
    "placement",
    "edge_r1",
    "packing_edge_r1",
)


@dataclass(frozen=True)
class CrossStageVariant:
    name: str
    component_source: str
    packing_source: str


VARIANTS: tuple[CrossStageVariant, ...] = (
    CrossStageVariant("raw_to_raw", "raw", "raw"),
    CrossStageVariant("raw_components_i21_pack", "raw", "i21"),
    CrossStageVariant("i21_components_raw_pack", "i21", "raw"),
    CrossStageVariant("i21_to_i21", "i21", "i21"),
)
VARIANT_BY_NAME: dict[str, CrossStageVariant] = {variant.name: variant for variant in VARIANTS}


@dataclass(frozen=True)
class PreparedScene:
    scene: CachedScene
    scores: Mapping[str, np.ndarray]
    dense: Mapping[str, tuple[np.ndarray, np.ndarray]]
    components: Mapping[str, list[dict[int, tuple[int, int]]]]
    edge: Mapping[str, Mapping[str, float]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a report/config and refuse ambiguous overwrite."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite experiment artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite experiment artifact: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _code_provenance() -> dict[str, dict[str, str]]:
    source = Path(__file__).resolve().parent
    paths = {
        "eval_cross_stage_buddies": Path(__file__).resolve(),
        "eval_rank_transplant": source / "eval_rank_transplant.py",
        "rank_transplant": source / "rank_transplant.py",
        "solve_buddies": source / "solve_buddies.py",
        "eval_seeded_qap": source / "eval_seeded_qap.py",
        "positional_ddpm": source / "positional_ddpm.py",
    }
    result: dict[str, dict[str, str]] = {}
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required code file is missing: {path}")
        result[role] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    return result


def i21_scores(
    raw_scores: np.ndarray,
    spatial_scores: np.ndarray,
    *,
    alpha: float = I21_ALPHA,
) -> np.ndarray:
    """Exact row-z I21 fusion used as a score matrix, without logit mutation."""
    raw = np.asarray(raw_scores)
    spatial = np.asarray(spatial_scores)
    if raw.ndim != 3 or raw.shape[0] != 4 or spatial.shape != raw.shape:
        raise ValueError("raw and spatial scores must align as (4,N,K)")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    valid = np.isfinite(raw)
    if np.any(valid & ~np.isfinite(spatial)):
        raise ValueError("spatial scores must be finite at every raw-valid slot")
    raw_z = row_zscore(raw, valid)
    spatial_z = row_zscore(spatial, valid)
    fused = np.full(raw.shape, -np.inf, dtype=np.float32)
    fused[valid] = raw_z[valid] + np.float32(alpha) * spatial_z[valid]
    return fused


def _dense_matrices(candidates: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right_t, down_t = dense_rd(
        torch.from_numpy(np.ascontiguousarray(candidates)).long(),
        torch.from_numpy(np.ascontiguousarray(scores)).float(),
    )
    right = np.ascontiguousarray(right_t.numpy(), dtype=np.float32)
    down = np.ascontiguousarray(down_t.numpy(), dtype=np.float32)
    if right.shape != (NFRAG, NFRAG) or down.shape != right.shape:
        raise AssertionError("dense_rd returned an invalid matrix shape")
    if not np.isfinite(right).all() or not np.isfinite(down).all():
        raise AssertionError("dense_rd returned non-finite values")
    if np.any(right < 0.0) or np.any(down < 0.0):
        raise AssertionError("dense_rd returned negative probabilities")
    if np.any(np.diag(right) != 0.0) or np.any(np.diag(down) != 0.0):
        raise AssertionError("dense_rd diagonal must be zero")
    return right, down


def prepare_scene(scene: CachedScene) -> PreparedScene:
    raw = np.array(scene.base_scores, dtype=np.float32, copy=True, order="C")
    fused = i21_scores(raw, scene.spatial_scores)
    scores = {"raw": raw, "i21": fused}
    dense = {source: _dense_matrices(scene.candidates, values) for source, values in scores.items()}
    components = {
        source: build_buddies_components(
            right,
            down,
            max_edges=BUDGET,
            min_margin=0.0,
        )
        for source, (right, down) in dense.items()
    }
    edge = {source: _edge_metrics(scene, values) for source, values in scores.items()}
    return PreparedScene(scene=scene, scores=scores, dense=dense, components=components, edge=edge)


def cross_stage_inputs(
    prepared: PreparedScene,
    variant_name: str,
) -> tuple[list[dict[int, tuple[int, int]]], np.ndarray, np.ndarray]:
    """Resolve component geometry and packing matrices for one fixed variant."""
    try:
        variant = VARIANT_BY_NAME[variant_name]
    except KeyError as exc:
        raise ValueError(f"unknown cross-stage variant {variant_name!r}") from exc
    right, down = prepared.dense[variant.packing_source]
    return prepared.components[variant.component_source], right, down


def evaluate_variant(prepared: PreparedScene, variant_name: str) -> dict[str, float | str]:
    variant = VARIANT_BY_NAME[variant_name]
    components, packing_right, packing_down = cross_stage_inputs(prepared, variant_name)
    board, objective = solve_components_from_scores(
        packing_right,
        packing_down,
        components,
        repair_passes=REPAIR_PASSES,
        restarts=1,
    )
    if board.shape != (NFRAG,) or not np.array_equal(np.sort(board), np.arange(NFRAG)):
        raise AssertionError("cross-stage buddies did not return a tile permutation")
    scene = prepared.scene
    target_board = np.argsort(scene.permutation)
    placement = placement_accuracy(board, target_board)[0]
    neighbour, right_acc, down_acc = neighbour_accuracy(board, target_board)
    solved = assemble(scene.tiles_uint8, board)
    solve_ssim = sk_ssim(scene.target_uint8, solved, channel_axis=2, data_range=255)
    component_edge = prepared.edge[variant.component_source]
    packing_edge = prepared.edge[variant.packing_source]
    return {
        "image": float(scene.image_id),
        "variant": variant.name,
        "component_source": variant.component_source,
        "packing_source": variant.packing_source,
        "candidate_recall": float(component_edge["candidate_recall"]),
        "edge_r1": float(component_edge["edge_r1"]),
        "packing_edge_r1": float(packing_edge["edge_r1"]),
        "placement": float(placement),
        "neighbour": float(neighbour),
        "right": float(right_acc),
        "down": float(down_acc),
        "solve_only_ssim": float(solve_ssim),
        "objective": float(objective),
        **component_metrics(components, scene.permutation),
    }


def summarize_rows(rows: Sequence[Mapping[str, float | str]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize zero cross-stage rows")
    return {
        **{
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in MEAN_METRICS
        },
        "images": float(len(rows)),
    }


def paired_deltas(
    candidate_rows: Sequence[Mapping[str, float | str]],
    baseline_rows: Sequence[Mapping[str, float | str]],
) -> dict[str, dict[str, Any]]:
    candidate_by_image = {int(float(row["image"])): row for row in candidate_rows}
    baseline_by_image = {int(float(row["image"])): row for row in baseline_rows}
    if candidate_by_image.keys() != baseline_by_image.keys() or len(candidate_by_image) != len(candidate_rows):
        raise ValueError("paired rows must contain exactly the same unique image IDs")
    result: dict[str, dict[str, Any]] = {}
    for metric in PRIMARY_METRICS:
        per_image = {
            str(image): float(candidate_by_image[image][metric]) - float(baseline_by_image[image][metric])
            for image in sorted(candidate_by_image)
        }
        values = np.asarray(list(per_image.values()), dtype=np.float64)
        result[metric] = {
            "mean_delta": float(values.mean()),
            "median_delta": float(np.median(values)),
            "wins": int(np.sum(values > 0.0)),
            "ties": int(np.sum(values == 0.0)),
            "losses": int(np.sum(values < 0.0)),
            "per_image": per_image,
        }
    return result


def select_calibration_variant(
    grid: Mapping[str, Mapping[str, Any]],
) -> tuple[str, Mapping[str, Any]] | None:
    """Freeze only a non-base variant with strictly positive paired mean SSIM."""
    order = {variant.name: index for index, variant in enumerate(VARIANTS)}
    eligible = [
        (name, row)
        for name, row in grid.items()
        if name != BASELINE_VARIANT
        and float(row["paired_delta"]["solve_only_ssim"]["mean_delta"]) > 0.0
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            float(item[1]["paired_delta"]["solve_only_ssim"]["mean_delta"]),
            float(item[1]["paired_delta"]["neighbour"]["mean_delta"]),
            float(item[1]["paired_delta"]["placement"]["mean_delta"]),
            float(item[1]["paired_delta"]["edge_r1"]["mean_delta"]),
            -order[item[0]],
        ),
    )


def _load_phase_scenes(
    image_ids: Sequence[int],
    *,
    cache_dir: Path,
    spatial_path: Path,
    device_text: str,
) -> tuple[list[CachedScene], str, int]:
    expected = CALIBRATION_IDS if tuple(image_ids) == CALIBRATION_IDS else CONFIRMATION_IDS
    if tuple(image_ids) != expected:
        raise ValueError("phase IDs differ from the hard-coded calibration/confirmation split")
    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(REPLAY_SEED)
    np.random.seed(REPLAY_SEED)
    torch.manual_seed(REPLAY_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(REPLAY_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    spatial_path = spatial_path.resolve()
    model, payload = _load_spatial(spatial_path, device)
    scenes = load_cached_scenes(
        cache_dir=cache_dir.resolve(),
        cache_tag=CACHE_TAG,
        image_ids=image_ids,
        groups=REPLAY_GROUPS,
        replay_seed=REPLAY_SEED,
        spatial_model=model,
        device=device,
    )
    return scenes, _sha256(spatial_path), int(payload.get("step", -1))


def _evaluate_grid(prepared: Sequence[PreparedScene]) -> dict[str, dict[str, Any]]:
    rows_by_variant = {
        variant.name: [evaluate_variant(scene, variant.name) for scene in prepared]
        for variant in VARIANTS
    }
    baseline_rows = rows_by_variant[BASELINE_VARIANT]
    return {
        variant.name: {
            "configuration": asdict(variant),
            "metrics": summarize_rows(rows_by_variant[variant.name]),
            "paired_delta": paired_deltas(rows_by_variant[variant.name], baseline_rows),
            "per_image": rows_by_variant[variant.name],
        }
        for variant in VARIANTS
    }


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    assert_disjoint_phases(CALIBRATION_IDS, CONFIRMATION_IDS)
    scenes, spatial_hash, spatial_step = _load_phase_scenes(
        CALIBRATION_IDS,
        cache_dir=Path(args.cache_dir),
        spatial_path=Path(args.spatial),
        device_text=args.device,
    )
    prepared = [prepare_scene(scene) for scene in scenes]
    grid = _evaluate_grid(prepared)
    selected = select_calibration_variant(grid)
    status = "frozen" if selected is not None else "no_positive_ssim_transfer"
    selected_name = selected[0] if selected is not None else None
    selected_row = selected[1] if selected is not None else None
    code = _code_provenance()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "phase": "calibration",
        "status": status,
        "calibration_images": list(CALIBRATION_IDS),
        "reserved_confirmation_images": list(CONFIRMATION_IDS),
        "replay_group": REPLAY_GROUP_TEXT,
        "replay_seed": REPLAY_SEED,
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "cache_tag": CACHE_TAG,
        "spatial_checkpoint": str(Path(args.spatial).resolve()),
        "spatial_sha256": spatial_hash,
        "spatial_step": spatial_step,
        "scene_provenance": scene_provenance(scenes),
        "i21_alpha": I21_ALPHA,
        "budget": BUDGET,
        "repair_passes": REPAIR_PASSES,
        "selection_rule": "nonbaseline paired mean solve_only_ssim delta > 0; maximize that delta",
        "grid": grid,
        "selected_variant": selected_name,
        "selected": selected_row,
        "code": code,
    }
    _write_new_json(Path(args.report), report)
    frozen: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": status,
        "calibration_images": list(CALIBRATION_IDS),
        "confirmation_images": list(CONFIRMATION_IDS),
        "replay_group": REPLAY_GROUP_TEXT,
        "replay_seed": REPLAY_SEED,
        "cache_tag": CACHE_TAG,
        "spatial_checkpoint": str(Path(args.spatial).resolve()),
        "spatial_sha256": spatial_hash,
        "spatial_step": spatial_step,
        "i21_alpha": I21_ALPHA,
        "budget": BUDGET,
        "repair_passes": REPAIR_PASSES,
        "calibration_report": str(Path(args.report).resolve()),
        "selection": (
            {
                "variant": selected_name,
                "configuration": asdict(VARIANT_BY_NAME[selected_name]),
                "calibration_metrics": selected_row["metrics"],
                "calibration_paired_delta": selected_row["paired_delta"],
            }
            if selected_name is not None and selected_row is not None
            else None
        ),
        "code": code,
    }
    _write_new_json(Path(args.frozen_config), frozen)
    print(
        json.dumps(
            {
                "status": status,
                "selected_variant": selected_name,
                "report": str(Path(args.report).resolve()),
                "frozen_config": str(Path(args.frozen_config).resolve()),
            }
        ),
        flush=True,
    )
    return report


def validate_frozen_config(frozen: Mapping[str, Any]) -> CrossStageVariant:
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "frozen",
        "calibration_images": list(CALIBRATION_IDS),
        "confirmation_images": list(CONFIRMATION_IDS),
        "replay_group": REPLAY_GROUP_TEXT,
        "replay_seed": REPLAY_SEED,
        "cache_tag": CACHE_TAG,
        "i21_alpha": I21_ALPHA,
        "budget": BUDGET,
        "repair_passes": REPAIR_PASSES,
    }
    for key, expected in fixed.items():
        if frozen.get(key) != expected:
            raise RuntimeError(f"frozen cross-stage config changed fixed field {key!r}")
    selection = frozen.get("selection")
    if not isinstance(selection, Mapping):
        raise RuntimeError("confirmation requires a positive-SSIM frozen calibration choice")
    name = selection.get("variant")
    if name == BASELINE_VARIANT or name not in VARIANT_BY_NAME:
        raise RuntimeError("frozen selection must be one nonbaseline cross-stage variant")
    variant = VARIANT_BY_NAME[str(name)]
    if selection.get("configuration") != asdict(variant):
        raise RuntimeError("frozen variant configuration differs from its canonical stage mapping")
    recorded_code = frozen.get("code")
    if recorded_code != _code_provenance():
        raise RuntimeError("cross-stage evaluator dependency hashes changed since calibration")
    assert_disjoint_phases(frozen["calibration_images"], frozen["confirmation_images"])
    return variant


def run_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.frozen_config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"frozen config does not exist: {config_path}")
    frozen = json.loads(config_path.read_text(encoding="utf-8"))
    variant = validate_frozen_config(frozen)
    scenes, spatial_hash, spatial_step = _load_phase_scenes(
        CONFIRMATION_IDS,
        cache_dir=Path(args.cache_dir),
        spatial_path=Path(args.spatial),
        device_text=args.device,
    )
    if spatial_hash != frozen.get("spatial_sha256"):
        raise RuntimeError("confirmation spatial checkpoint hash differs from calibration")
    if spatial_step != frozen.get("spatial_step"):
        raise RuntimeError("confirmation spatial checkpoint step differs from calibration")
    prepared = [prepare_scene(scene) for scene in scenes]
    # Intentionally no loop over VARIANTS in confirmation.
    baseline_rows = [evaluate_variant(scene, BASELINE_VARIANT) for scene in prepared]
    candidate_rows = [evaluate_variant(scene, variant.name) for scene in prepared]
    baseline = summarize_rows(baseline_rows)
    candidate = summarize_rows(candidate_rows)
    delta = paired_deltas(candidate_rows, baseline_rows)
    checks = {"paired_solve_only_ssim": delta["solve_only_ssim"]["mean_delta"] > 0.0}
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "phase": "confirmation",
        "status": "pass" if all(checks.values()) else "fail",
        "frozen_config": str(config_path),
        "frozen_config_sha256": _sha256(config_path),
        "calibration_images": list(CALIBRATION_IDS),
        "confirmation_images": list(CONFIRMATION_IDS),
        "replay_group": REPLAY_GROUP_TEXT,
        "replay_seed": REPLAY_SEED,
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "cache_tag": CACHE_TAG,
        "spatial_checkpoint": str(Path(args.spatial).resolve()),
        "spatial_sha256": spatial_hash,
        "spatial_step": spatial_step,
        "scene_provenance": scene_provenance(scenes),
        "selection": dict(frozen["selection"]),
        "i21_alpha": I21_ALPHA,
        "budget": BUDGET,
        "repair_passes": REPAIR_PASSES,
        "baseline": baseline,
        "candidate": candidate,
        "paired_delta": delta,
        "checks": checks,
        "baseline_per_image": baseline_rows,
        "candidate_per_image": candidate_rows,
        "code": _code_provenance(),
    }
    _write_new_json(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument(
        "--spatial",
        type=Path,
        default=Path(WORK_ROOT) / "positional_ddpm" / "positional_ddpm_train_latest.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    phases = parser.add_subparsers(dest="phase", required=True)
    calibrate = phases.add_parser("calibrate", help="evaluate four fixed stage crossings and freeze one")
    _add_common_arguments(calibrate)
    calibrate.add_argument("--frozen-config", type=Path, required=True)
    confirm = phases.add_parser("confirm", help="evaluate only baseline plus the frozen crossing")
    _add_common_arguments(confirm)
    confirm.add_argument("--frozen-config", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "calibrate":
        run_calibration(args)
    elif args.phase == "confirm":
        run_confirmation(args)
    else:  # pragma: no cover - argparse enforces the phase.
        raise AssertionError(f"unsupported phase {args.phase}")


if __name__ == "__main__":
    main()
