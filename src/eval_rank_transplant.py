"""Calibrate and confirm confidence-gated reciprocal rank transplantation.

This evaluator deliberately has two separate phases:

``calibrate``
    Sweeps donor alpha/top-M only on declared calibration cache ids and writes
    one frozen configuration.

``confirm``
    Loads exactly one frozen configuration, rejects image overlap, and never
    contains a best-configuration selection path.

The base is the cached CandidateSeamRanker row logits.  Spatial logits are
sampled on the same cached candidate ids, trusted reciprocal pairs are selected
by their weaker top-two margin, and existing raw base logits are swapped before
the unchanged ``dense_rd`` and corrected buddies solver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim

from canvas_data import CanvasDataset
from config import GRID, NFRAG, SEED, WORK_ROOT
from eval_calibrated_buddies import component_metrics
from eval_seeded_qap import dense_rd
from imgio import assemble, train_val_split
from placement_metrics import neighbour_accuracy, placement_accuracy
from positional_ddpm import PositionalDDPM
from rank_transplant import (
    ReciprocalPair,
    assert_disjoint_phases,
    confidence_gated_rank_transplant,
    fused_donor_scores,
    row_predictions,
    validate_candidate_rows,
)
from solve_buddies import build_buddies_components, solve_buddies_from_scores


SCHEMA_VERSION = 1
MEAN_METRICS: tuple[str, ...] = (
    "candidate_recall",
    "edge_r1",
    "placement",
    "neighbour",
    "right",
    "down",
    "solve_only_ssim",
    "objective",
    "eligible_pairs",
    "trusted_pairs",
    "changed_rows",
    "beneficial_rows",
    "harmful_rows",
    "neutral_changed_rows",
    "trusted_confidence_min",
    "trusted_confidence_mean",
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
)


@dataclass(frozen=True)
class CachedScene:
    image_id: int
    cache_path: Path
    cache_sha256: str
    candidates: np.ndarray
    base_scores: np.ndarray
    spatial_scores: np.ndarray
    permutation: np.ndarray
    tiles_uint8: np.ndarray
    target_uint8: np.ndarray


def _parse_ints(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("image/top-M lists must be non-empty and duplicate-free")
    return values


def _parse_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(not np.isfinite(value) for value in values):
        raise ValueError("alpha lists must contain finite values")
    return values


def _parse_groups(text: str) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    covered: set[int] = set()
    for item in text.split(","):
        start_text, count_text = item.strip().split(":", maxsplit=1)
        start, count = int(start_text), int(count_text)
        if start < 0 or count < 1:
            raise ValueError("groups must be START:positive-COUNT")
        indices = set(range(start, start + count))
        if covered & indices:
            raise ValueError("replay groups may not overlap")
        covered.update(indices)
        groups.append((start, count))
    if not groups:
        raise ValueError("at least one replay group is required")
    return groups


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _recreate_samples(
    image_ids: Sequence[int],
    groups: Sequence[tuple[int, int]],
    *,
    replay_seed: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """Replay groups exactly, including global RNG draws for skipped members."""
    requested = {int(value) for value in image_ids}
    _, validation_names = train_val_split()
    samples: dict[int, dict[str, torch.Tensor]] = {}
    for start, count in groups:
        if start + count > len(validation_names):
            raise ValueError(f"group {start}:{count} exceeds validation pool")
        np.random.seed(replay_seed)
        torch.manual_seed(replay_seed)
        dataset = CanvasDataset(
            validation_names[start : start + count],
            real_prob=0.0,
            seed=replay_seed + 400_000,
        )
        for local in range(count):
            # Calling every preceding member is part of the replay contract:
            # CanvasDataset consumes one global NumPy draw per __getitem__.
            sample = dataset[local]
            absolute = start + local
            if absolute in requested:
                samples[absolute] = sample
    missing = sorted(requested - samples.keys())
    if missing:
        raise ValueError(f"requested cache ids are not covered by --groups: {missing}")
    return samples


def _load_spatial(path: Path, device: torch.device) -> tuple[PositionalDDPM, Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"spatial checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model_args" not in payload or "model" not in payload:
        raise RuntimeError(f"unrecognized PositionalDDPM checkpoint: {path}")
    model = PositionalDDPM(**dict(payload["model_args"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def _uint8_tiles(sample: Mapping[str, torch.Tensor]) -> np.ndarray:
    tiles = sample["tiles"].permute(0, 2, 3, 1).numpy()
    return np.rint(tiles * 255.0).clip(0, 255).astype(np.uint8)


def _uint8_target(sample: Mapping[str, torch.Tensor]) -> np.ndarray:
    target = sample["clean"].permute(1, 2, 0).numpy()
    return np.rint(target * 255.0).clip(0, 255).astype(np.uint8)


@torch.inference_mode()
def _spatial_on_candidates(
    model: PositionalDDPM,
    sample: Mapping[str, torch.Tensor],
    candidates: np.ndarray,
    base_scores: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    tiles = sample["tiles"].unsqueeze(0).to(device)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        features = model.encode_tiles(tiles)
        full = model.directional_edge_scores(features)[0].float().cpu().numpy()
    anchors = np.arange(candidates.shape[0])[:, None]
    spatial = np.empty_like(base_scores, dtype=np.float32)
    for direction in range(4):
        spatial[direction] = full[direction][anchors, candidates]
    spatial[~np.isfinite(base_scores)] = -np.inf
    return spatial


def load_cached_scenes(
    *,
    cache_dir: Path,
    cache_tag: str,
    image_ids: Sequence[int],
    groups: Sequence[tuple[int, int]],
    replay_seed: int,
    spatial_model: PositionalDDPM,
    device: torch.device,
) -> list[CachedScene]:
    samples = _recreate_samples(image_ids, groups, replay_seed=replay_seed)
    scenes: list[CachedScene] = []
    for image_id in image_ids:
        cache_path = cache_dir / f"image_{image_id:04d}_{cache_tag}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"missing full-graph cache: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as stored:
            required = {"candidate_ids", "candidate_scores", "permutation"}
            missing = sorted(required - set(stored.files))
            if missing:
                raise RuntimeError(f"{cache_path} is missing fields {missing}")
            candidates = stored["candidate_ids"].astype(np.int64)
            permutation = stored["permutation"].astype(np.int64)
            flat_scores = stored["candidate_scores"].astype(np.float32)
        if candidates.shape[0] != NFRAG or permutation.shape != (NFRAG,):
            raise RuntimeError(f"unexpected puzzle shape in {cache_path}")
        if flat_scores.shape != (NFRAG * 4, candidates.shape[1]):
            raise RuntimeError(f"unexpected candidate_scores shape in {cache_path}: {flat_scores.shape}")
        base_scores = flat_scores.reshape(NFRAG, 4, -1).transpose(1, 0, 2).copy()
        validate_candidate_rows(candidates, base_scores)
        sample = samples[int(image_id)]
        replayed = sample["perm"].numpy().astype(np.int64)
        if not np.array_equal(permutation, replayed):
            raise RuntimeError(
                f"cache {image_id} permutation does not match deterministic replay; "
                "check --groups and --replay-seed"
            )
        spatial = _spatial_on_candidates(
            spatial_model,
            sample,
            candidates,
            base_scores,
            device=device,
        )
        scenes.append(
            CachedScene(
                image_id=int(image_id),
                cache_path=cache_path,
                cache_sha256=_sha256(cache_path),
                candidates=candidates,
                base_scores=base_scores,
                spatial_scores=spatial,
                permutation=permutation,
                tiles_uint8=_uint8_tiles(sample),
                target_uint8=_uint8_target(sample),
            )
        )
        print(json.dumps({"loaded_image": int(image_id), "cache": str(cache_path)}), flush=True)
    return scenes


def scene_provenance(scenes: Sequence[CachedScene]) -> list[dict[str, Any]]:
    """Record the exact cache, corruption bytes, target bytes, and permutation."""
    return [
        {
            "image": scene.image_id,
            "cache": str(scene.cache_path.resolve()),
            "cache_sha256": scene.cache_sha256,
            "tiles_sha256": _array_sha256(scene.tiles_uint8),
            "target_sha256": _array_sha256(scene.target_uint8),
            "permutation_sha256": _array_sha256(scene.permutation),
        }
        for scene in scenes
    ]


def _true_directional_targets(permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(permutation, dtype=np.int64)
    if cells.shape != (NFRAG,) or not np.array_equal(np.sort(cells), np.arange(NFRAG)):
        raise ValueError("permutation must be tile->cell and bijective")
    rows, columns = cells // GRID, cells % GRID
    exists = np.stack(
        (rows > 0, rows < GRID - 1, columns > 0, columns < GRID - 1),
        axis=1,
    )
    inverse = np.empty_like(cells)
    inverse[cells] = np.arange(NFRAG)
    target_cells = cells[:, None] + np.asarray((-GRID, GRID, -1, 1))[None]
    targets = inverse[np.clip(target_cells, 0, NFRAG - 1)]
    targets[~exists] = -1
    return targets, exists


def _edge_metrics(scene: CachedScene, scores: np.ndarray) -> dict[str, float]:
    predicted, _, _ = row_predictions(scene.candidates, scores)
    truth, exists = _true_directional_targets(scene.permutation)
    valid = np.isfinite(scores).transpose(1, 0, 2)
    present = (
        valid
        & (scene.candidates[:, None, :] == truth[:, :, None])
        & exists[:, :, None]
    ).any(axis=-1)
    exact = predicted.transpose(1, 0) == truth
    return {
        "candidate_recall": float(present[exists].mean()),
        "edge_r1": float(exact[exists].mean()),
    }


def _solver_metrics(
    scene: CachedScene,
    scores: np.ndarray,
    *,
    budget: int,
    repair_passes: int,
) -> dict[str, float]:
    right_t, down_t = dense_rd(
        torch.from_numpy(scene.candidates).long(),
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
    board, objective = solve_buddies_from_scores(
        right,
        down,
        max_edges=budget,
        min_margin=0.0,
        repair_passes=repair_passes,
    )
    if board.shape != (NFRAG,) or not np.array_equal(np.sort(board), np.arange(NFRAG)):
        raise AssertionError("buddies solver did not return a tile permutation")
    target_board = np.argsort(scene.permutation)
    placement = placement_accuracy(board, target_board)[0]
    neighbour, right_acc, down_acc = neighbour_accuracy(board, target_board)
    solved = assemble(scene.tiles_uint8, board)
    solve_ssim = sk_ssim(
        scene.target_uint8,
        solved,
        channel_axis=2,
        data_range=255,
    )
    components = build_buddies_components(
        right,
        down,
        max_edges=budget,
        min_margin=0.0,
    )
    return {
        "placement": float(placement),
        "neighbour": float(neighbour),
        "right": float(right_acc),
        "down": float(down_acc),
        "solve_only_ssim": float(solve_ssim),
        "objective": float(objective),
        **component_metrics(components, scene.permutation),
    }


def evaluate_baseline(
    scene: CachedScene,
    *,
    budget: int,
    repair_passes: int,
) -> dict[str, float]:
    return {
        "image": float(scene.image_id),
        **_edge_metrics(scene, scene.base_scores),
        **_solver_metrics(
            scene,
            scene.base_scores,
            budget=budget,
            repair_passes=repair_passes,
        ),
    }


def _pair_is_exact(pair: ReciprocalPair, truth: np.ndarray) -> bool:
    return int(truth[pair.anchor, pair.direction]) == pair.target


def evaluate_configuration(
    scene: CachedScene,
    *,
    alpha: float,
    top_m: int,
    min_confidence: float | None,
    budget: int,
    repair_passes: int,
) -> dict[str, float]:
    donor = fused_donor_scores(
        scene.base_scores,
        scene.spatial_scores,
        alpha=alpha,
    )
    transplant = confidence_gated_rank_transplant(
        scene.candidates,
        scene.base_scores,
        donor,
        top_m=top_m,
        min_confidence=min_confidence,
        verify=True,
    )
    if not np.array_equal(
        np.sort(scene.base_scores, axis=-1),
        np.sort(transplant.scores, axis=-1),
    ):
        raise AssertionError("transplanted rows failed exact multiset preservation")
    truth, exists = _true_directional_targets(scene.permutation)
    trusted_correct = sum(_pair_is_exact(pair, truth) for pair in transplant.selected_pairs)
    confidences = [pair.confidence for pair in transplant.selected_pairs]

    base_target, _, _ = row_predictions(scene.candidates, scene.base_scores)
    new_target, _, _ = row_predictions(scene.candidates, transplant.scores)
    truth_d_n = truth.transpose(1, 0)
    exists_d_n = exists.transpose(1, 0)
    changed = (base_target != new_target) & exists_d_n
    base_correct = base_target == truth_d_n
    new_correct = new_target == truth_d_n
    beneficial = changed & ~base_correct & new_correct
    harmful = changed & base_correct & ~new_correct
    neutral = changed & (base_correct == new_correct)
    return {
        "image": float(scene.image_id),
        "alpha": float(alpha),
        "top_m": float(top_m),
        "eligible_pairs": float(len(transplant.eligible_pairs)),
        "trusted_pairs": float(len(transplant.selected_pairs)),
        "trusted_correct": float(trusted_correct),
        "changed_rows": float(transplant.changed_row_count),
        "beneficial_rows": float(beneficial.sum()),
        "harmful_rows": float(harmful.sum()),
        "neutral_changed_rows": float(neutral.sum()),
        "trusted_confidence_min": float(min(confidences)) if confidences else 0.0,
        "trusted_confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
        **_edge_metrics(scene, transplant.scores),
        **_solver_metrics(
            scene,
            transplant.scores,
            budget=budget,
            repair_passes=repair_passes,
        ),
    }


def summarize_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize zero rows")
    summary = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in MEAN_METRICS
        if key in rows[0]
    }
    trusted_pairs = float(sum(float(row.get("trusted_pairs", 0.0)) for row in rows))
    trusted_correct = float(sum(float(row.get("trusted_correct", 0.0)) for row in rows))
    summary.update(
        {
            "images": float(len(rows)),
            "trusted_pairs_total": trusted_pairs,
            "trusted_correct_total": trusted_correct,
            "trusted_pair_precision": trusted_correct / trusted_pairs if trusted_pairs else 0.0,
        }
    )
    return summary


def paired_delta(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
) -> dict[str, float]:
    return {
        metric: float(candidate[metric]) - float(baseline[metric])
        for metric in PRIMARY_METRICS
    }


def select_calibration_configuration(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    minimum_trusted_precision: float,
) -> tuple[str, Mapping[str, Any]] | None:
    """Select on calibration only: SSIM, neighbour, edge, then smaller top-M."""
    if not 0.0 <= minimum_trusted_precision <= 1.0:
        raise ValueError("minimum_trusted_precision must lie in [0,1]")
    eligible = [
        (key, value)
        for key, value in summaries.items()
        if float(value["metrics"]["trusted_pairs_total"]) > 0.0
        and float(value["metrics"]["trusted_pair_precision"]) >= minimum_trusted_precision
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            float(item[1]["delta"]["solve_only_ssim"]),
            float(item[1]["delta"]["neighbour"]),
            float(item[1]["delta"]["edge_r1"]),
            -int(item[1]["top_m"]),
            -float(item[1]["alpha"]),
        ),
    )


def _load_scenes_from_args(args: argparse.Namespace) -> tuple[list[CachedScene], str, int]:
    image_ids = _parse_ints(args.images)
    groups = _parse_groups(args.groups)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.replay_seed)
    np.random.seed(args.replay_seed)
    torch.manual_seed(args.replay_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.replay_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    checkpoint = Path(args.spatial).resolve()
    model, payload = _load_spatial(checkpoint, device)
    checkpoint_hash = _sha256(checkpoint)
    scenes = load_cached_scenes(
        cache_dir=Path(args.cache_dir),
        cache_tag=args.cache_tag,
        image_ids=image_ids,
        groups=groups,
        replay_seed=args.replay_seed,
        spatial_model=model,
        device=device,
    )
    return scenes, checkpoint_hash, int(payload.get("step", -1))


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    scenes, spatial_hash, spatial_step = _load_scenes_from_args(args)
    image_ids = [scene.image_id for scene in scenes]
    alphas = _parse_floats(args.alphas)
    top_values = _parse_ints(args.top_ms)
    if any(value < 0 for value in top_values):
        raise ValueError("top-M values must be non-negative")
    baseline_rows = [
        evaluate_baseline(scene, budget=args.budget, repair_passes=args.repair_passes)
        for scene in scenes
    ]
    baseline = summarize_rows(baseline_rows)
    grid: dict[str, dict[str, Any]] = {}
    per_image: dict[str, list[dict[str, float]]] = {}
    for alpha in alphas:
        for top_m in top_values:
            key = f"alpha={alpha:g}:top_m={top_m}"
            rows = [
                evaluate_configuration(
                    scene,
                    alpha=alpha,
                    top_m=top_m,
                    min_confidence=args.min_confidence,
                    budget=args.budget,
                    repair_passes=args.repair_passes,
                )
                for scene in scenes
            ]
            metrics = summarize_rows(rows)
            grid[key] = {
                "alpha": float(alpha),
                "top_m": int(top_m),
                "metrics": metrics,
                "delta": paired_delta(metrics, baseline),
            }
            per_image[key] = rows
            print(json.dumps({"configuration": key, **grid[key]}), flush=True)
    selected = select_calibration_configuration(
        grid,
        minimum_trusted_precision=args.minimum_trusted_precision,
    )
    status = "frozen" if selected is not None else "failed_precision_gate"
    selected_key = selected[0] if selected is not None else None
    selected_value = dict(selected[1]) if selected is not None else None
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "confidence_gated_reciprocal_rank_transplant",
        "phase": "calibration",
        "status": status,
        "images": image_ids,
        "groups": args.groups,
        "replay_seed": int(args.replay_seed),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "cache_tag": args.cache_tag,
        "spatial_checkpoint": str(Path(args.spatial).resolve()),
        "spatial_sha256": spatial_hash,
        "spatial_step": spatial_step,
        "scene_provenance": scene_provenance(scenes),
        "budget": int(args.budget),
        "repair_passes": int(args.repair_passes),
        "minimum_trusted_precision": float(args.minimum_trusted_precision),
        "min_confidence": args.min_confidence,
        "baseline": baseline,
        "baseline_per_image": baseline_rows,
        "grid": grid,
        "grid_per_image": per_image,
        "selected_key": selected_key,
        "selected": selected_value,
    }
    _write_json(Path(args.report), report)
    frozen: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": report["experiment"],
        "status": status,
        "calibration_images": image_ids,
        "calibration_report": str(Path(args.report).resolve()),
        "spatial_checkpoint": report["spatial_checkpoint"],
        "spatial_sha256": spatial_hash,
        "spatial_step": spatial_step,
        "budget": int(args.budget),
        "repair_passes": int(args.repair_passes),
        "minimum_trusted_precision": float(args.minimum_trusted_precision),
        "min_confidence": args.min_confidence,
        "selection": (
            {
                "key": selected_key,
                "alpha": float(selected_value["alpha"]),
                "top_m": int(selected_value["top_m"]),
                "calibration_metrics": selected_value["metrics"],
                "calibration_delta": selected_value["delta"],
            }
            if selected_value is not None
            else None
        ),
    }
    _write_json(Path(args.frozen_config), frozen)
    print(json.dumps({"report": str(args.report), "frozen_config": str(args.frozen_config), "status": status}), flush=True)
    return report


def run_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.frozen_config)
    frozen = json.loads(config_path.read_text(encoding="utf-8"))
    if int(frozen.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError("frozen configuration schema is incompatible")
    if frozen.get("status") != "frozen" or not isinstance(frozen.get("selection"), Mapping):
        raise RuntimeError("confirmation requires a calibration config with status=frozen")
    confirmation_ids = _parse_ints(args.images)
    calibration_ids = [int(value) for value in frozen.get("calibration_images", ())]
    assert_disjoint_phases(calibration_ids, confirmation_ids)
    scenes, spatial_hash, spatial_step = _load_scenes_from_args(args)
    if spatial_hash != frozen.get("spatial_sha256"):
        raise RuntimeError("confirmation spatial checkpoint hash differs from frozen calibration")
    if spatial_step != int(frozen.get("spatial_step", -1)):
        raise RuntimeError("confirmation spatial checkpoint step differs from frozen calibration")
    selection = frozen["selection"]
    alpha = float(selection["alpha"])
    top_m = int(selection["top_m"])
    budget = int(frozen["budget"])
    repair_passes = int(frozen["repair_passes"])
    min_confidence = frozen.get("min_confidence")
    baseline_rows = [
        evaluate_baseline(scene, budget=budget, repair_passes=repair_passes)
        for scene in scenes
    ]
    candidate_rows = [
        evaluate_configuration(
            scene,
            alpha=alpha,
            top_m=top_m,
            min_confidence=min_confidence,
            budget=budget,
            repair_passes=repair_passes,
        )
        for scene in scenes
    ]
    baseline = summarize_rows(baseline_rows)
    candidate = summarize_rows(candidate_rows)
    delta = paired_delta(candidate, baseline)
    minimum_precision = float(frozen["minimum_trusted_precision"])
    checks = {
        "trusted_pair_precision": candidate["trusted_pair_precision"] >= minimum_precision,
        "paired_solve_only_ssim": delta["solve_only_ssim"] > 0.0,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": frozen["experiment"],
        "phase": "confirmation",
        "status": "pass" if all(checks.values()) else "fail",
        "frozen_config": str(config_path.resolve()),
        "calibration_images": calibration_ids,
        "confirmation_images": confirmation_ids,
        "groups": args.groups,
        "replay_seed": int(args.replay_seed),
        "spatial_checkpoint": str(Path(args.spatial).resolve()),
        "spatial_sha256": spatial_hash,
        "spatial_step": spatial_step,
        "scene_provenance": scene_provenance(scenes),
        "selection": dict(selection),
        "budget": budget,
        "repair_passes": repair_passes,
        "minimum_trusted_precision": minimum_precision,
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "checks": checks,
        "baseline_per_image": baseline_rows,
        "candidate_per_image": candidate_rows,
    }
    _write_json(Path(args.report), report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def _add_replay_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--images", required=True, help="comma-separated absolute validation cache ids")
    parser.add_argument("--groups", required=True, help="cache replay groups as START:COUNT,...")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument("--cache-tag", default="k64")
    parser.add_argument("--replay-seed", type=int, default=SEED)
    parser.add_argument(
        "--spatial",
        type=Path,
        default=Path(WORK_ROOT) / "positional_ddpm" / "positional_ddpm_train_latest.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    calibrate = subparsers.add_parser("calibrate", help="sweep calibration caches and freeze one config")
    _add_replay_arguments(calibrate)
    calibrate.add_argument("--alphas", default="0.5,0.75,1,1.25")
    calibrate.add_argument("--top-ms", default="8,16,32,64")
    calibrate.add_argument("--min-confidence", type=float, default=None)
    calibrate.add_argument("--minimum-trusted-precision", type=float, default=0.85)
    calibrate.add_argument("--budget", type=int, default=512)
    calibrate.add_argument("--repair-passes", type=int, default=0)
    calibrate.add_argument("--frozen-config", type=Path, required=True)

    confirm = subparsers.add_parser("confirm", help="evaluate exactly one frozen config")
    _add_replay_arguments(confirm)
    confirm.add_argument("--frozen-config", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "calibrate":
        if args.budget < 1 or args.repair_passes < 0:
            raise ValueError("budget must be positive and repair-passes non-negative")
        run_calibration(args)
    elif args.phase == "confirm":
        run_confirmation(args)
    else:  # pragma: no cover - argparse makes this unreachable.
        raise AssertionError(f"unsupported phase {args.phase}")


if __name__ == "__main__":
    main()
