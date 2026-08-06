"""Two-stage E3 gate for classical seam evidence and rank transplantation.

The split and replay protocol are intentionally hard-coded:

* calibration: validation cache ids 10..17;
* confirmation: validation cache ids 18..21;
* replay group: 10:12 with seed 1234.

``calibrate`` computes/caches all label-free classical variants, reports raw
and donor edge/precision-coverage diagnostics, and walks one predeclared
variant order.  It freezes the first variant that strictly improves both
all-true edge R@1 and trusted reciprocal-pair precision at fixed top-M=32.
There is no metric argmax.

``confirm`` accepts only that frozen variant.  It contains no variant or
threshold sweep, applies exact-multiset raw-logit rank transplantation, then
runs the unchanged ``dense_rd`` and corrected buddies baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim

from canvas_data import CanvasDataset
from classical_seam_rank import (
    PREDECLARED_VARIANT_ORDER,
    VARIANT_NAMES,
    compute_classical_candidate_scores,
    variant_index,
)
from config import GRID, NFRAG, SEED, WORK_ROOT
from eval_seeded_qap import dense_rd
from imgio import assemble, train_val_split
from placement_metrics import neighbour_accuracy, placement_accuracy
from rank_transplant import (
    confidence_gated_rank_transplant,
    reciprocal_physical_pairs,
    row_predictions,
    row_zscore,
    select_trusted_pairs,
    validate_candidate_rows,
)
from solve_buddies import solve_buddies_from_scores


SCHEMA_VERSION = 1
CALIBRATION_IMAGE_IDS: tuple[int, ...] = tuple(range(10, 18))
CONFIRMATION_IMAGE_IDS: tuple[int, ...] = tuple(range(18, 22))
REPLAY_GROUP_START = 10
REPLAY_GROUP_COUNT = 12
REPLAY_SEED = SEED
REPLAY_DATASET_SEED = REPLAY_SEED + 400_000
BASE_BUDGET = 512
REPAIR_PASSES = 0
TRUSTED_TOP_M = 32
ROW_COVERAGES: tuple[float, ...] = (0.01, 0.02, 0.04, 0.08, 0.16)
PAIR_TOP_MS: tuple[int, ...] = (8, 16, 32, 64)
MGC_RIDGE = 0.05


@dataclass(frozen=True)
class ClassicalScene:
    image_id: int
    cache_path: Path
    cache_sha256: str
    feature_path: Path
    feature_sha256: str
    candidates: np.ndarray
    base_scores: np.ndarray
    classical_scores: np.ndarray
    permutation: np.ndarray
    tiles_uint8: np.ndarray
    target_uint8: np.ndarray


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


def _replay_samples(image_ids: Sequence[int]) -> dict[int, dict[str, torch.Tensor]]:
    requested = {int(value) for value in image_ids}
    allowed = set(range(REPLAY_GROUP_START, REPLAY_GROUP_START + REPLAY_GROUP_COUNT))
    if not requested or not requested <= allowed:
        raise ValueError("requested ids must be inside the hard-coded replay group 10:12")
    _, validation_names = train_val_split()
    names = validation_names[
        REPLAY_GROUP_START : REPLAY_GROUP_START + REPLAY_GROUP_COUNT
    ]
    np.random.seed(REPLAY_SEED)
    torch.manual_seed(REPLAY_SEED)
    dataset = CanvasDataset(
        names,
        real_prob=0.0,
        seed=REPLAY_DATASET_SEED,
    )
    samples: dict[int, dict[str, torch.Tensor]] = {}
    last_local = max(requested) - REPLAY_GROUP_START
    for local in range(last_local + 1):
        # Every earlier access consumes the global NumPy draw that was present
        # when full_graph_cache was built.  Skipping it breaks byte replay.
        sample = dataset[local]
        absolute = REPLAY_GROUP_START + local
        if absolute in requested:
            samples[absolute] = sample
    if samples.keys() != requested:
        raise AssertionError("hard-coded replay did not cover requested ids")
    return samples


def _uint8_tiles(sample: Mapping[str, torch.Tensor]) -> np.ndarray:
    values = sample["tiles"].permute(0, 2, 3, 1).numpy()
    return np.rint(values * 255.0).clip(0, 255).astype(np.uint8)


def _uint8_target(sample: Mapping[str, torch.Tensor]) -> np.ndarray:
    values = sample["clean"].permute(1, 2, 0).numpy()
    return np.rint(values * 255.0).clip(0, 255).astype(np.uint8)


def _feature_metadata(
    *,
    image_id: int,
    cache_sha256: str,
    tiles: np.ndarray,
    candidates: np.ndarray,
    valid: np.ndarray,
) -> dict[str, str]:
    return {
        "schema_version": str(SCHEMA_VERSION),
        "image_id": str(image_id),
        "source_cache_sha256": cache_sha256,
        "tiles_sha256": _array_sha256(tiles),
        "candidates_sha256": _array_sha256(candidates),
        "valid_sha256": _array_sha256(valid.astype(np.uint8)),
        "mgc_ridge": repr(MGC_RIDGE),
    }


def _load_or_compute_features(
    *,
    feature_path: Path,
    metadata: Mapping[str, str],
    tiles: np.ndarray,
    candidates: np.ndarray,
    valid: np.ndarray,
    force: bool,
) -> np.ndarray:
    if feature_path.is_file() and not force:
        with np.load(feature_path, allow_pickle=False) as stored:
            if "scores" not in stored.files or "variant_names" not in stored.files:
                raise RuntimeError(f"incomplete classical feature cache: {feature_path}")
            cached_names = tuple(str(value) for value in stored["variant_names"].tolist())
            if cached_names != VARIANT_NAMES:
                raise RuntimeError(f"variant schema mismatch in {feature_path}")
            for key, expected in metadata.items():
                if key not in stored.files or str(stored[key].item()) != expected:
                    raise RuntimeError(
                        f"feature provenance mismatch for {key} in {feature_path}; "
                        "use --force-recompute-features explicitly"
                    )
            scores = stored["scores"].astype(np.float32)
        expected_shape = (len(VARIANT_NAMES), 4, *candidates.shape)
        if scores.shape != expected_shape:
            raise RuntimeError(f"invalid feature score shape in {feature_path}: {scores.shape}")
        if not np.array_equal(np.isfinite(scores), np.broadcast_to(valid[None], scores.shape)):
            raise RuntimeError(f"feature finite mask mismatch in {feature_path}")
        return scores

    scores = compute_classical_candidate_scores(
        tiles,
        candidates,
        valid,
        mgc_ridge=MGC_RIDGE,
    )
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    fields: dict[str, np.ndarray] = {
        "scores": scores,
        "variant_names": np.asarray(VARIANT_NAMES),
    }
    fields.update({key: np.asarray(value) for key, value in metadata.items()})
    np.savez_compressed(feature_path, **fields)
    return scores


def load_scenes(
    image_ids: Sequence[int],
    *,
    cache_dir: Path,
    feature_cache_dir: Path,
    force_recompute_features: bool,
) -> list[ClassicalScene]:
    samples = _replay_samples(image_ids)
    scenes: list[ClassicalScene] = []
    for image_id in image_ids:
        cache_path = cache_dir / f"image_{image_id:04d}_k64.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"missing candidate cache: {cache_path}")
        cache_hash = _sha256(cache_path)
        with np.load(cache_path, allow_pickle=False) as stored:
            required = {"candidate_ids", "candidate_scores", "permutation"}
            missing = sorted(required - set(stored.files))
            if missing:
                raise RuntimeError(f"{cache_path} is missing {missing}")
            candidates = stored["candidate_ids"].astype(np.int64)
            flat_scores = stored["candidate_scores"].astype(np.float32)
            permutation = stored["permutation"].astype(np.int64)
        if candidates.shape[0] != NFRAG or permutation.shape != (NFRAG,):
            raise RuntimeError(f"unexpected puzzle shape in {cache_path}")
        if flat_scores.shape != (NFRAG * 4, candidates.shape[1]):
            raise RuntimeError(f"unexpected candidate score shape in {cache_path}")
        base_scores = flat_scores.reshape(NFRAG, 4, -1).transpose(1, 0, 2).copy()
        validate_candidate_rows(candidates, base_scores)
        sample = samples[int(image_id)]
        replayed = sample["perm"].numpy().astype(np.int64)
        if not np.array_equal(replayed, permutation):
            raise RuntimeError(f"cache {image_id} failed exact replay; split/seed is fixed")
        tiles = _uint8_tiles(sample)
        target = _uint8_target(sample)
        valid = np.isfinite(base_scores)
        feature_path = feature_cache_dir / f"image_{image_id:04d}_classical_v1.npz"
        metadata = _feature_metadata(
            image_id=int(image_id),
            cache_sha256=cache_hash,
            tiles=tiles,
            candidates=candidates,
            valid=valid,
        )
        classical = _load_or_compute_features(
            feature_path=feature_path,
            metadata=metadata,
            tiles=tiles,
            candidates=candidates,
            valid=valid,
            force=force_recompute_features,
        )
        scenes.append(
            ClassicalScene(
                image_id=int(image_id),
                cache_path=cache_path,
                cache_sha256=cache_hash,
                feature_path=feature_path,
                feature_sha256=_sha256(feature_path),
                candidates=candidates,
                base_scores=base_scores,
                classical_scores=classical,
                permutation=permutation,
                tiles_uint8=tiles,
                target_uint8=target,
            )
        )
        print(
            json.dumps(
                {
                    "loaded_image": int(image_id),
                    "feature_cache": str(feature_path),
                }
            ),
            flush=True,
        )
    return scenes


def scene_provenance(scenes: Sequence[ClassicalScene]) -> list[dict[str, Any]]:
    return [
        {
            "image": scene.image_id,
            "candidate_cache": str(scene.cache_path.resolve()),
            "candidate_cache_sha256": scene.cache_sha256,
            "feature_cache": str(scene.feature_path.resolve()),
            "feature_cache_sha256": scene.feature_sha256,
            "tiles_sha256": _array_sha256(scene.tiles_uint8),
            "target_sha256": _array_sha256(scene.target_uint8),
            "permutation_sha256": _array_sha256(scene.permutation),
        }
        for scene in scenes
    ]


def _true_targets(permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(permutation, dtype=np.int64)
    if cells.shape != (NFRAG,) or not np.array_equal(np.sort(cells), np.arange(NFRAG)):
        raise ValueError("permutation must be a tile->cell bijection")
    rows, columns = cells // GRID, cells % GRID
    exists = np.stack(
        (rows > 0, rows < GRID - 1, columns > 0, columns < GRID - 1),
        axis=1,
    )
    inverse = np.empty_like(cells)
    inverse[cells] = np.arange(NFRAG)
    cells_by_direction = cells[:, None] + np.asarray((-GRID, GRID, -1, 1))[None]
    targets = inverse[np.clip(cells_by_direction, 0, NFRAG - 1)]
    targets[~exists] = -1
    return targets, exists


def fixed_raw_classical_donor(base_scores: np.ndarray, classical_scores: np.ndarray) -> np.ndarray:
    """Fixed equal row-z fusion; there is deliberately no fitted alpha."""
    base = np.asarray(base_scores)
    classical = np.asarray(classical_scores)
    if classical.shape != base.shape:
        raise ValueError("classical score shape must match base rows")
    valid = np.isfinite(base)
    if not np.array_equal(np.isfinite(classical), valid):
        raise ValueError("classical and base finite masks must match exactly")
    fused = np.full(base.shape, -np.inf, dtype=np.float32)
    base_z = row_zscore(base, valid)
    classical_z = row_zscore(classical, valid)
    fused[valid] = base_z[valid] + classical_z[valid]
    return row_zscore(fused, valid)


def _edge_summary(
    scenes: Sequence[ClassicalScene],
    scores_by_scene: Sequence[np.ndarray],
) -> dict[str, float]:
    exact_total = 0
    present_total = 0
    physical_total = 0
    for scene, scores in zip(scenes, scores_by_scene):
        predicted, _, _ = row_predictions(scene.candidates, scores)
        truth, exists = _true_targets(scene.permutation)
        valid = np.isfinite(scores).transpose(1, 0, 2)
        present = (
            valid
            & (scene.candidates[:, None, :] == truth[:, :, None])
            & exists[:, :, None]
        ).any(axis=-1)
        exact_total += int(((predicted.transpose(1, 0) == truth) & exists).sum())
        present_total += int(present.sum())
        physical_total += int(exists.sum())
    return {
        "all_true_edge_r1": exact_total / physical_total,
        "candidate_recall": present_total / physical_total,
        "exact_rows": float(exact_total),
        "physical_rows": float(physical_total),
    }


def _row_precision_curve(
    scenes: Sequence[ClassicalScene],
    scores_by_scene: Sequence[np.ndarray],
) -> dict[str, dict[str, float]]:
    totals = {coverage: [0, 0, 0] for coverage in ROW_COVERAGES}
    for scene, scores in zip(scenes, scores_by_scene):
        predicted, _, margin = row_predictions(scene.candidates, scores)
        truth, exists = _true_targets(scene.permutation)
        labels = (predicted.transpose(1, 0) == truth)[exists]
        confidence = margin.transpose(1, 0)[exists]
        order = np.argsort(-confidence, kind="mergesort")
        for coverage in ROW_COVERAGES:
            count = min(len(order), max(1, int(np.ceil(len(order) * coverage))))
            selected = labels[order[:count]]
            totals[coverage][0] += int(selected.sum())
            totals[coverage][1] += count
            totals[coverage][2] += len(order)
    return {
        f"{coverage:.2f}": {
            "precision": correct / selected if selected else 0.0,
            "accepted": float(selected),
            "rows": float(rows),
            "coverage": selected / rows if rows else 0.0,
        }
        for coverage, (correct, selected, rows) in totals.items()
    }


def _pair_precision_curve(
    scenes: Sequence[ClassicalScene],
    scores_by_scene: Sequence[np.ndarray],
    *,
    require_changed: bool,
    top_ms: Sequence[int] = PAIR_TOP_MS,
) -> dict[str, dict[str, float]]:
    totals = {int(top_m): [0, 0] for top_m in top_ms}
    physical_pairs_per_scene = 2 * GRID * (GRID - 1)
    for scene, scores in zip(scenes, scores_by_scene):
        pairs = reciprocal_physical_pairs(
            scene.candidates,
            scores,
            base_scores=scene.base_scores if require_changed else None,
            require_changed=require_changed,
        )
        truth, _ = _true_targets(scene.permutation)
        for top_m in totals:
            selected = select_trusted_pairs(pairs, top_m=top_m)
            totals[top_m][0] += sum(
                int(truth[pair.anchor, pair.direction]) == pair.target
                for pair in selected
            )
            totals[top_m][1] += len(selected)
    return {
        str(top_m): {
            "precision": correct / selected if selected else 0.0,
            "accepted_pairs": float(selected),
            "physical_pair_coverage": selected / (physical_pairs_per_scene * len(scenes)),
            "requested_pairs_per_scene": float(top_m),
        }
        for top_m, (correct, selected) in totals.items()
    }


def stage1_diagnostics(
    scenes: Sequence[ClassicalScene],
    *,
    include_all_variants: bool,
    selected_variant: str | None = None,
    include_curves: bool = True,
) -> dict[str, Any]:
    raw_scores = [row_zscore(scene.base_scores) for scene in scenes]
    raw_edge = _edge_summary(scenes, raw_scores)
    top_ms = PAIR_TOP_MS if include_curves else (TRUSTED_TOP_M,)
    raw_pair_curve = _pair_precision_curve(
        scenes,
        raw_scores,
        require_changed=False,
        top_ms=top_ms,
    )
    raw = {
        **raw_edge,
        "trusted": raw_pair_curve[str(TRUSTED_TOP_M)],
    }
    if include_curves:
        raw["row_precision_coverage"] = _row_precision_curve(scenes, raw_scores)
        raw["reciprocal_pair_precision_coverage"] = raw_pair_curve
    names = PREDECLARED_VARIANT_ORDER if include_all_variants else (str(selected_variant),)
    variants: dict[str, Any] = {}
    for name in names:
        if name not in VARIANT_NAMES:
            raise ValueError(f"unknown selected variant {name!r}")
        index = variant_index(name)
        classical_only = [row_zscore(scene.classical_scores[index]) for scene in scenes]
        donors = [
            fixed_raw_classical_donor(scene.base_scores, scene.classical_scores[index])
            for scene in scenes
        ]
        pair_curve = _pair_precision_curve(
            scenes,
            donors,
            require_changed=True,
            top_ms=top_ms,
        )
        variants[name] = {
            "classical_standalone": _edge_summary(scenes, classical_only),
            "fixed_raw_plus_classical": _edge_summary(scenes, donors),
            "trusted": pair_curve[str(TRUSTED_TOP_M)],
        }
        if include_curves:
            variants[name]["row_precision_coverage"] = _row_precision_curve(scenes, donors)
            variants[name]["reciprocal_pair_precision_coverage"] = pair_curve
    return {"raw": raw, "variants": variants}


def choose_first_qualifying_variant(stage1: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]] | tuple[None, list[dict[str, Any]]]:
    """Walk the scientific prior order; never choose by metric argmax."""
    raw = stage1["raw"]
    raw_r1 = float(raw["all_true_edge_r1"])
    raw_precision = float(raw["trusted"]["precision"])
    expected_pairs = TRUSTED_TOP_M * len(CALIBRATION_IMAGE_IDS)
    audit: list[dict[str, Any]] = []
    for name in PREDECLARED_VARIANT_ORDER:
        candidate = stage1["variants"][name]
        donor = candidate["fixed_raw_plus_classical"]
        trusted = candidate["trusted"]
        checks = {
            "all_true_edge_r1_improves": float(donor["all_true_edge_r1"]) > raw_r1,
            "trusted_precision_improves": float(trusted["precision"]) > raw_precision,
            "fixed_trusted_coverage_available": int(trusted["accepted_pairs"]) == expected_pairs,
        }
        audit.append(
            {
                "variant": name,
                "checks": checks,
                "qualifies": all(checks.values()),
            }
        )
        if all(checks.values()):
            return name, audit
    return None, audit


def _solver_metrics(scene: ClassicalScene, scores: np.ndarray) -> dict[str, float]:
    right_t, down_t = dense_rd(
        torch.from_numpy(scene.candidates).long(),
        torch.from_numpy(np.ascontiguousarray(scores)).float(),
    )
    right = np.ascontiguousarray(right_t.numpy(), dtype=np.float32)
    down = np.ascontiguousarray(down_t.numpy(), dtype=np.float32)
    if right.shape != (NFRAG, NFRAG) or down.shape != right.shape:
        raise AssertionError("dense_rd returned an invalid shape")
    if not np.isfinite(right).all() or not np.isfinite(down).all():
        raise AssertionError("dense_rd returned non-finite values")
    board, objective = solve_buddies_from_scores(
        right,
        down,
        max_edges=BASE_BUDGET,
        min_margin=0.0,
        repair_passes=REPAIR_PASSES,
    )
    if board.shape != (NFRAG,) or not np.array_equal(np.sort(board), np.arange(NFRAG)):
        raise AssertionError("solver output is not a tile permutation")
    truth_board = np.argsort(scene.permutation)
    placement = placement_accuracy(board, truth_board)[0]
    neighbour, right_acc, down_acc = neighbour_accuracy(board, truth_board)
    solved = assemble(scene.tiles_uint8, board)
    solve_ssim = sk_ssim(
        scene.target_uint8,
        solved,
        channel_axis=2,
        data_range=255,
    )
    return {
        "placement": float(placement),
        "neighbour": float(neighbour),
        "right": float(right_acc),
        "down": float(down_acc),
        "solve_only_ssim": float(solve_ssim),
        "objective": float(objective),
    }


def stage2_transplant(
    scenes: Sequence[ClassicalScene],
    variant: str,
) -> dict[str, Any]:
    index = variant_index(variant)
    baseline_rows: list[dict[str, float]] = []
    candidate_rows: list[dict[str, float]] = []
    for scene in scenes:
        donor = fixed_raw_classical_donor(
            scene.base_scores,
            scene.classical_scores[index],
        )
        transplant = confidence_gated_rank_transplant(
            scene.candidates,
            scene.base_scores,
            donor,
            top_m=TRUSTED_TOP_M,
            verify=True,
        )
        if not np.array_equal(
            np.sort(scene.base_scores, axis=-1),
            np.sort(transplant.scores, axis=-1),
        ):
            raise AssertionError("classical transplant changed a raw row value multiset")
        truth, exists = _true_targets(scene.permutation)
        base_target, _, _ = row_predictions(scene.candidates, scene.base_scores)
        new_target, _, _ = row_predictions(scene.candidates, transplant.scores)
        base_correct = base_target.transpose(1, 0) == truth
        new_correct = new_target.transpose(1, 0) == truth
        changed = (base_target.transpose(1, 0) != new_target.transpose(1, 0)) & exists
        trusted_correct = sum(
            int(truth[pair.anchor, pair.direction]) == pair.target
            for pair in transplant.selected_pairs
        )
        baseline_rows.append(
            {
                "image": float(scene.image_id),
                **_solver_metrics(scene, scene.base_scores),
            }
        )
        candidate_rows.append(
            {
                "image": float(scene.image_id),
                "eligible_pairs": float(len(transplant.eligible_pairs)),
                "trusted_pairs": float(len(transplant.selected_pairs)),
                "trusted_correct": float(trusted_correct),
                "changed_rows": float(transplant.changed_row_count),
                "beneficial_rows": float((changed & ~base_correct & new_correct).sum()),
                "harmful_rows": float((changed & base_correct & ~new_correct).sum()),
                **_solver_metrics(scene, transplant.scores),
            }
        )

    def summarize(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
        metrics = ("placement", "neighbour", "right", "down", "solve_only_ssim", "objective")
        result = {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in metrics
        }
        trusted = sum(float(row.get("trusted_pairs", 0.0)) for row in rows)
        correct = sum(float(row.get("trusted_correct", 0.0)) for row in rows)
        result.update(
            {
                "images": float(len(rows)),
                "trusted_pairs_total": float(trusted),
                "trusted_pair_precision": correct / trusted if trusted else 0.0,
                "changed_rows_mean": float(np.mean([row.get("changed_rows", 0.0) for row in rows])),
                "beneficial_rows_total": float(sum(row.get("beneficial_rows", 0.0) for row in rows)),
                "harmful_rows_total": float(sum(row.get("harmful_rows", 0.0) for row in rows)),
            }
        )
        return result

    baseline = summarize(baseline_rows)
    candidate = summarize(candidate_rows)
    delta = {
        key: candidate[key] - baseline[key]
        for key in ("solve_only_ssim", "neighbour", "placement")
    }
    return {
        "variant": variant,
        "top_m": TRUSTED_TOP_M,
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "baseline_per_image": baseline_rows,
        "candidate_per_image": candidate_rows,
    }


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    scenes = load_scenes(
        CALIBRATION_IMAGE_IDS,
        cache_dir=args.cache_dir,
        feature_cache_dir=args.feature_cache_dir,
        force_recompute_features=args.force_recompute_features,
    )
    stage1 = stage1_diagnostics(scenes, include_all_variants=True)
    selected, selection_audit = choose_first_qualifying_variant(stage1)
    stage2 = stage2_transplant(scenes, selected) if selected is not None else None
    status = "frozen" if selected is not None else "failed_no_candidate"
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "multi_depth_classical_rank_transplant",
        "phase": "calibration",
        "status": status,
        "calibration_images": list(CALIBRATION_IMAGE_IDS),
        "confirmation_images_reserved": list(CONFIRMATION_IMAGE_IDS),
        "replay": {
            "group": f"{REPLAY_GROUP_START}:{REPLAY_GROUP_COUNT}",
            "seed": REPLAY_SEED,
            "dataset_seed": REPLAY_DATASET_SEED,
        },
        "fixed_contract": {
            "variant_order": list(PREDECLARED_VARIANT_ORDER),
            "fusion": "row_z(raw_candidate_logits) + row_z(classical), then row_z",
            "trusted_top_m_per_image": TRUSTED_TOP_M,
            "row_coverages": list(ROW_COVERAGES),
            "pair_top_ms": list(PAIR_TOP_MS),
            "budget": BASE_BUDGET,
            "repair_passes": REPAIR_PASSES,
            "mgc_ridge": MGC_RIDGE,
            "orientation": "fixed; no rotations or reflections",
        },
        "scene_provenance": scene_provenance(scenes),
        "stage1": stage1,
        "selection_audit": selection_audit,
        "selected_variant": selected,
        "stage2": stage2,
    }
    _write_json(args.report, report)
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "experiment": report["experiment"],
        "status": status,
        "calibration_images": list(CALIBRATION_IMAGE_IDS),
        "confirmation_images": list(CONFIRMATION_IMAGE_IDS),
        "variant_order": list(PREDECLARED_VARIANT_ORDER),
        "selected_variant": selected,
        "fusion": report["fixed_contract"]["fusion"],
        "trusted_top_m_per_image": TRUSTED_TOP_M,
        "budget": BASE_BUDGET,
        "repair_passes": REPAIR_PASSES,
        "mgc_ridge": MGC_RIDGE,
        "calibration_report": str(args.report.resolve()),
        "calibration_provenance": report["scene_provenance"],
    }
    _write_json(args.frozen_config, frozen)
    print(
        json.dumps(
            {
                "status": status,
                "selected_variant": selected,
                "report": str(args.report),
                "frozen_config": str(args.frozen_config),
            }
        ),
        flush=True,
    )
    return report


def run_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    if int(frozen.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError("frozen classical config schema mismatch")
    if frozen.get("status") != "frozen" or not frozen.get("selected_variant"):
        raise RuntimeError("fail closed: calibration produced no approved classical variant")
    if tuple(frozen.get("calibration_images", ())) != CALIBRATION_IMAGE_IDS:
        raise RuntimeError("frozen calibration split differs from hard-coded ids 10..17")
    if tuple(frozen.get("confirmation_images", ())) != CONFIRMATION_IMAGE_IDS:
        raise RuntimeError("frozen confirmation split differs from hard-coded ids 18..21")
    if tuple(frozen.get("variant_order", ())) != PREDECLARED_VARIANT_ORDER:
        raise RuntimeError("predeclared variant ordering changed after calibration")
    if int(frozen.get("trusted_top_m_per_image", -1)) != TRUSTED_TOP_M:
        raise RuntimeError("trusted top-M changed after calibration")
    if int(frozen.get("budget", -1)) != BASE_BUDGET or int(frozen.get("repair_passes", -1)) != REPAIR_PASSES:
        raise RuntimeError("solver contract changed after calibration")
    if float(frozen.get("mgc_ridge", -1.0)) != MGC_RIDGE:
        raise RuntimeError("classical feature contract changed after calibration")
    selected = str(frozen["selected_variant"])
    if selected not in PREDECLARED_VARIANT_ORDER:
        raise RuntimeError("frozen selected variant is unknown")

    scenes = load_scenes(
        CONFIRMATION_IMAGE_IDS,
        cache_dir=args.cache_dir,
        feature_cache_dir=args.feature_cache_dir,
        force_recompute_features=args.force_recompute_features,
    )
    # Confirmation computes diagnostics for exactly one frozen variant.
    stage1 = stage1_diagnostics(
        scenes,
        include_all_variants=False,
        selected_variant=selected,
        include_curves=False,
    )
    stage2 = stage2_transplant(scenes, selected)
    raw = stage1["raw"]
    candidate = stage1["variants"][selected]
    checks = {
        "all_true_edge_r1_improves": (
            candidate["fixed_raw_plus_classical"]["all_true_edge_r1"]
            > raw["all_true_edge_r1"]
        ),
        "trusted_precision_improves": (
            candidate["trusted"]["precision"] > raw["trusted"]["precision"]
        ),
        "fixed_trusted_coverage_available": (
            int(candidate["trusted"]["accepted_pairs"])
            == TRUSTED_TOP_M * len(CONFIRMATION_IMAGE_IDS)
        ),
        "solve_only_ssim_improves": stage2["delta"]["solve_only_ssim"] > 0.0,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": frozen["experiment"],
        "phase": "confirmation",
        "status": "pass" if all(checks.values()) else "fail",
        "frozen_config": str(args.frozen_config.resolve()),
        "calibration_images": list(CALIBRATION_IMAGE_IDS),
        "confirmation_images": list(CONFIRMATION_IMAGE_IDS),
        "selected_variant": selected,
        "scene_provenance": scene_provenance(scenes),
        "stage1": stage1,
        "stage2": stage2,
        "checks": checks,
    }
    _write_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "edge_confidence" / "full_graph_cache",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=Path(WORK_ROOT) / "classical_seam_rank" / "features",
    )
    parser.add_argument("--force-recompute-features", action="store_true")
    parser.add_argument("--report", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    calibrate = phases.add_parser("calibrate", help="run fixed stage-1 calibration and freeze <=1 variant")
    _common(calibrate)
    calibrate.add_argument("--frozen-config", type=Path, required=True)
    confirm = phases.add_parser("confirm", help="run the one frozen variant on ids 18..21")
    _common(confirm)
    confirm.add_argument("--frozen-config", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.phase == "calibrate":
        run_calibration(args)
    elif args.phase == "confirm":
        run_confirmation(args)
    else:  # pragma: no cover
        raise AssertionError(f"unsupported phase {args.phase}")


if __name__ == "__main__":
    main()
