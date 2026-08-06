"""Retrieve public source photographs from *shuffled, degraded* PAZZLE inputs.

This is the test-time half of public-source forensics.  It never reads a test
target (none exists): a source candidate is ranked only from a
permutation-invariant fingerprint of the dirty 576-tile bag.

The fingerprint is a compact sliced-Wasserstein-style summary of normalized
5x5 tile appearance descriptors plus tile colour/spread quantiles.  Tile order
is discarded before comparison.  A second, explicit ``recover``/Hungarian
check can validate the top public candidate and recover its permutation.

All caches and reports live on E: by default.  An unverified public photo is
streamed in memory only; its original bytes are saved under ``found_test``
only after the robust assignment check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from scipy.optimize import linear_sum_assignment

from imgio import from_frags, load, to_frags
from source_retrieval import (
    DEFAULT_ROOT,
    SourceRecord,
    _decode_image,
    _download_original,
    _safe_stem,
    centre_square,
    load_tbank_manifest,
    preview_url,
)


GRID = 24
TILES = GRID * GRID
FINGERPRINT_SIZE = 120  # 24 source cells x 5 pixels per cell.
PROJECTIONS = 16
RANK_SAMPLES = 48
COLOUR_SAMPLES = 32
RANDOM = np.random.default_rng(20_260_711)
PROJECTION = RANDOM.normal(size=(75, PROJECTIONS)).astype(np.float32)
PROJECTION /= np.linalg.norm(PROJECTION, axis=0, keepdims=True)
RANK_INDEX = np.linspace(0, TILES - 1, RANK_SAMPLES).round().astype(np.int64)
COLOUR_INDEX = np.linspace(0, TILES - 1, COLOUR_SAMPLES).round().astype(np.int64)


def _paths(root: Path) -> dict[str, Path]:
    index = root / "index"
    reports = root / "matches"
    images = root / "images" / "found_test"
    for directory in (index, reports, images):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "index": index / "tbank_bag_fingerprint_index.npz",
        "calibration": index / "tbank_bag_calibration.npz",
        "reports": reports,
        "images": images,
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _five_by_five_descriptors(frags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized texture descriptors and raw tile mean/spread.

    ``frags`` is RGB ``(576,H,W,3)``.  Source previews use exactly 5x5 cells;
    dirty test tiles are average-pooled from 20x20 to the identical 5x5
    representation.  Per-tile z-normalization removes the random brightness
    and contrast nuisance before texture comparison.
    """
    if frags.ndim != 4 or frags.shape[0] != TILES or frags.shape[-1] != 3:
        raise ValueError(f"expected ({TILES},H,W,3) tiles, got {frags.shape}")
    height, width = frags.shape[1:3]
    if height % 5 or width % 5:
        raise ValueError("tile dimensions must be divisible by 5")
    pool_h, pool_w = height // 5, width // 5
    pooled = frags.astype(np.float32).reshape(TILES, 5, pool_h, 5, pool_w, 3).mean(axis=(2, 4))
    flat = pooled.reshape(TILES, -1)
    normalized = (flat - flat.mean(axis=1, keepdims=True)) / (flat.std(axis=1, keepdims=True) + 1.0e-5)
    means = frags.astype(np.float32).mean(axis=(1, 2))
    spreads = frags.astype(np.float32).std(axis=(1, 2))
    return normalized, np.concatenate((means, spreads), axis=1)


def _source_frags_from_bgr(image: np.ndarray) -> np.ndarray:
    canonical_bgr = centre_square(image, FINGERPRINT_SIZE)
    canonical_rgb = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2RGB)
    return canonical_rgb.reshape(GRID, 5, GRID, 5, 3).transpose(0, 2, 1, 3, 4).reshape(TILES, 5, 5, 3)


def bag_fingerprint(frags: np.ndarray) -> np.ndarray:
    """Make a fixed-length permutation-invariant descriptor of a tile bag."""
    texture, colour = _five_by_five_descriptors(frags)
    # Sorted one-dimensional projections approximate set transport without
    # allocating a 576x576 pair matrix.  They retain much more instance-level
    # information than a global colour histogram while ignoring shuffle order.
    projected = np.sort(texture @ PROJECTION, axis=0)
    texture_part = projected[RANK_INDEX].T.reshape(-1)
    colour_sorted = np.sort(colour, axis=0)
    colour_part = colour_sorted[COLOUR_INDEX].T.reshape(-1)
    return np.concatenate((texture_part, colour_part)).astype(np.float32)


def _fetch_source_fingerprint(record: SourceRecord, timeout: float) -> tuple[int, np.ndarray | None]:
    try:
        response = requests.get(
            preview_url(record.url),
            timeout=timeout,
            headers={"User-Agent": "PAZZLE-bag-source-retrieval/1.0"},
        )
        response.raise_for_status()
        return record.record_id, bag_fingerprint(_source_frags_from_bgr(_decode_image(response.content)))
    except (requests.RequestException, ValueError, cv2.error):
        return record.record_id, None


def _save_bag_index(
    path: Path,
    *,
    records: list[SourceRecord],
    values: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Atomically save a resumable feature cache (never public image bytes)."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            record_ids=np.arange(len(records), dtype=np.int32),
            fingerprints=values,
            valid=valid,
            projections=PROJECTION,
            rank_index=RANK_INDEX,
            colour_index=COLOUR_INDEX,
        )
    temporary.replace(path)


def build_tbank_bag_index(root: Path, *, workers: int = 10, timeout: float = 30.0) -> Path:
    """Stream T-Bank previews and cache only their compact set fingerprints."""
    if workers < 1:
        raise ValueError("workers must be positive")
    records = load_tbank_manifest(root)
    paths = _paths(root)
    index_path = paths["index"]
    expected_dimension = PROJECTIONS * RANK_SAMPLES + 6 * COLOUR_SAMPLES
    partial_path = index_path.with_suffix(".partial.npz")
    if index_path.exists():
        with np.load(index_path, allow_pickle=False) as prior:
            ids = prior["record_ids"]
            fp = prior["fingerprints"]
            if np.array_equal(ids, np.arange(len(records), dtype=np.int32)) and fp.shape == (len(records), expected_dimension):
                print(f"reusing bag fingerprint index {index_path}", flush=True)
                return index_path

    values = np.zeros((len(records), expected_dimension), dtype=np.float16)
    valid = np.zeros(len(records), dtype=bool)
    if partial_path.exists():
        try:
            with np.load(partial_path, allow_pickle=False) as partial:
                ids = partial["record_ids"]
                fp = partial["fingerprints"]
                saved_valid = partial["valid"].astype(bool, copy=False)
                saved_projection = partial["projections"].astype(np.float32, copy=False)
                if (
                    np.array_equal(ids, np.arange(len(records), dtype=np.int32))
                    and fp.shape == values.shape
                    and saved_valid.shape == valid.shape
                    and np.allclose(saved_projection, PROJECTION)
                ):
                    values[:] = fp
                    valid[:] = saved_valid
                    print(f"resuming bag fingerprint index: {int(valid.sum())}/{len(records)} already cached", flush=True)
        except (OSError, KeyError, ValueError):
            print(f"ignoring unreadable partial bag index {partial_path}", flush=True)
    pending = [record for record in records if not valid[record.record_id]]
    if not pending:
        _save_bag_index(index_path, records=records, values=values, valid=valid)
        partial_path.unlink(missing_ok=True)
        print(f"saved bag fingerprint index {index_path}; valid={int(valid.sum())}/{len(records)}", flush=True)
        return index_path
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_fetch_source_fingerprint, record, timeout) for record in pending]
        for completed, future in enumerate(as_completed(futures), start=1):
            record_id, fingerprint = future.result()
            if fingerprint is not None:
                values[record_id] = fingerprint.astype(np.float16)
                valid[record_id] = True
            if completed % 100 == 0 or completed == len(pending):
                _save_bag_index(partial_path, records=records, values=values, valid=valid)
                print(
                    f"bag source index {completed}/{len(pending)} pending; valid={int(valid.sum())}/{len(records)} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    _save_bag_index(index_path, records=records, values=values, valid=valid)
    partial_path.unlink(missing_ok=True)
    print(f"saved bag fingerprint index {index_path}; valid={int(valid.sum())}/{len(records)}", flush=True)
    return index_path


def _load_bag_index(root: Path) -> tuple[list[SourceRecord], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    records = load_tbank_manifest(root)
    path = _paths(root)["index"]
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run index-tbank first")
    with np.load(path, allow_pickle=False) as index:
        ids = index["record_ids"].astype(np.int64, copy=False)
        values = index["fingerprints"].astype(np.float32, copy=False)
        valid = index["valid"].astype(bool, copy=False)
        projections = index["projections"].astype(np.float32, copy=False)
    if not np.array_equal(ids, np.arange(len(records), dtype=np.int64)):
        raise RuntimeError("bag index does not align with T-Bank manifest")
    if not np.allclose(projections, PROJECTION):
        raise RuntimeError("bag index was built with a different fingerprint projection")
    # Robust feature scaling is fit to the source catalogue only.  This is
    # fixed before test queries and does not use hidden target information.
    source = values[valid]
    center = np.median(source, axis=0)
    scale = np.quantile(source, 0.75, axis=0) - np.quantile(source, 0.25, axis=0)
    scale = np.maximum(scale, 0.05)
    return records, values, valid, center, scale


def rank_bag(fingerprint: np.ndarray, source_values: np.ndarray, valid: np.ndarray, center: np.ndarray, scale: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Rank source records using only an unordered dirty-bag fingerprint."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    ids = np.flatnonzero(valid)
    query = (fingerprint - center) / scale
    candidates = (source_values[ids] - center) / scale
    distance = np.mean(np.square(candidates - query[None, :]), axis=1)
    order = np.argsort(distance, kind="stable")[: min(top_k, len(ids))]
    return ids[order], distance[order]


def _fit_calibration(source: np.ndarray, dirty: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a diagonal public-source -> dirty-input calibration.

    The task applies independent photometric corruption to every tile.  Its
    effect on sorted bag statistics is systematic but quite different from a
    simple source-catalogue z-score.  A per-feature affine transfer is fit
    only on independently source-verified *train* examples, then its residual
    scale becomes a diagonal Mahalanobis distance at test time.
    """
    if source.ndim != 2 or source.shape != dirty.shape or source.shape[0] < 2:
        raise ValueError(f"need matching (N,D) source/dirty arrays, got {source.shape} and {dirty.shape}")
    source_mean = source.mean(axis=0)
    dirty_mean = dirty.mean(axis=0)
    source_centered = source - source_mean
    dirty_centered = dirty - dirty_mean
    source_var = np.mean(np.square(source_centered), axis=0)
    covariance = np.mean(source_centered * dirty_centered, axis=0)
    ridge = max(1.0e-6, 1.0e-3 * float(np.median(source_var)))
    slope = covariance / (source_var + ridge)
    intercept = dirty_mean - slope * source_mean
    residual = dirty - (source * slope + intercept)
    residual_scale = np.sqrt(np.mean(np.square(residual), axis=0) + 1.0e-6)
    # A handful of nearly deterministic order-statistic dimensions otherwise
    # dominate due to float16 cache quantisation rather than useful evidence.
    residual_scale = np.maximum(residual_scale, np.quantile(residual_scale, 0.05))
    return slope.astype(np.float32), intercept.astype(np.float32), residual_scale.astype(np.float32)


def rank_calibrated(
    fingerprint: np.ndarray,
    source_values: np.ndarray,
    valid: np.ndarray,
    slope: np.ndarray,
    intercept: np.ndarray,
    residual_scale: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank public candidates after the train-only source->dirty calibration."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if fingerprint.shape != slope.shape or slope.shape != intercept.shape or slope.shape != residual_scale.shape:
        raise ValueError("fingerprint and calibration dimensions do not agree")
    ids = np.flatnonzero(valid)
    predicted_dirty = source_values[ids] * slope[None, :] + intercept[None, :]
    distance = np.mean(np.square((predicted_dirty - fingerprint[None, :]) / residual_scale[None, :]), axis=1)
    order = np.argsort(distance, kind="stable")[: min(top_k, len(ids))]
    return ids[order], distance[order]


def _verified_tbank_queries(
    root: Path,
    report_path: Path,
    records: list[SourceRecord],
    valid: np.ndarray,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load dirty train queries plus source identities proved in a prior audit."""
    source_id_by_url = {record.url: record.record_id for record in records}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row.get("accepted")]
    if limit is not None:
        rows = rows[:limit]
    input_root = Path(os.environ.get("PAZZLE_DATA", r"E:/pazzle_data")) / "train" / "inputs"
    samples: list[dict[str, Any]] = []
    for row in rows:
        source_id = source_id_by_url.get(row["accepted"]["url"])
        if source_id is None or not valid[source_id]:
            continue
        input_path = input_root / row["target"]
        if not input_path.exists():
            continue
        samples.append(
            {
                "target": row["target"],
                "source_id": int(source_id),
                "event_slug": records[source_id].event_slug,
                "fingerprint": _input_fingerprint(input_path),
            }
        )
    if not samples:
        raise RuntimeError("no usable source-verified dirty train inputs")
    return samples


def _event_group_folds(samples: list[dict[str, Any]], count: int) -> np.ndarray:
    """Split by event, so validation never sees another photo from its event."""
    if count < 2:
        raise ValueError("at least two folds are required")
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(str(sample["event_slug"]), []).append(index)
    loads = [0] * count
    assignment: dict[str, int] = {}
    for group, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        fold = min(range(count), key=lambda candidate: (loads[candidate], candidate))
        assignment[group] = fold
        loads[fold] += len(members)
    return np.asarray([assignment[str(sample["event_slug"])] for sample in samples], dtype=np.int8)


def _rank_summary(ranks: list[int]) -> dict[str, float]:
    value = np.asarray(ranks, dtype=np.int64)
    if value.size == 0:
        raise RuntimeError("no calibration validation rows were ranked")
    return {
        "queries": int(value.size),
        "r1": float(np.mean(value == 1)),
        "r5": float(np.mean(value <= 5)),
        "r20": float(np.mean(value <= 20)),
        "r50": float(np.mean(value <= 50)),
        "median_rank": float(np.median(value)),
        "mean_rank": float(value.mean()),
    }


def validate_calibration(
    source_values: np.ndarray,
    valid: np.ndarray,
    samples: list[dict[str, Any]],
    *,
    folds: int = 5,
    top_k: int = 50,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Event-grouped honest validation of test-time source retrieval."""
    fold = _event_group_folds(samples, folds)
    sample_source = np.asarray([source_values[int(sample["source_id"])] for sample in samples], dtype=np.float32)
    sample_dirty = np.asarray([sample["fingerprint"] for sample in samples], dtype=np.float32)
    ranks: list[int] = []
    details: list[dict[str, Any]] = []
    for held_out in range(folds):
        train_mask = fold != held_out
        test_mask = fold == held_out
        if train_mask.sum() < 2 or not test_mask.any():
            continue
        slope, intercept, residual_scale = _fit_calibration(sample_source[train_mask], sample_dirty[train_mask])
        for sample, fingerprint in zip(np.asarray(samples, dtype=object)[test_mask], sample_dirty[test_mask]):
            candidate_ids, distances = rank_calibrated(
                fingerprint,
                source_values,
                valid,
                slope,
                intercept,
                residual_scale,
                top_k=int(valid.sum()),
            )
            location = np.flatnonzero(candidate_ids == int(sample["source_id"]))
            rank = int(location[0] + 1) if location.size else int(valid.sum()) + 1
            ranks.append(rank)
            details.append(
                {
                    "target": str(sample["target"]),
                    "event_slug": str(sample["event_slug"]),
                    "truth_record_id": int(sample["source_id"]),
                    "fold": int(held_out),
                    "rank": rank,
                    "top": candidate_ids[:top_k].astype(int).tolist(),
                    "top_distance": distances[:top_k].astype(float).tolist(),
                }
            )
    return _rank_summary(ranks), details


def fit_tbank_calibration(root: Path, *, report_path: Path, folds: int = 5) -> Path:
    """Fit final test-time calibration and report event-held-out accuracy."""
    records, values, valid, _center, _scale = _load_bag_index(root)
    samples = _verified_tbank_queries(root, report_path, records, valid)
    source = np.asarray([values[int(sample["source_id"])] for sample in samples], dtype=np.float32)
    dirty = np.asarray([sample["fingerprint"] for sample in samples], dtype=np.float32)
    slope, intercept, residual_scale = _fit_calibration(source, dirty)
    summary, details = validate_calibration(values, valid, samples, folds=folds)
    paths = _paths(root)
    calibration_path = paths["calibration"]
    temporary = calibration_path.with_suffix(calibration_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            version=np.asarray([1], dtype=np.int32),
            source_records=np.asarray([len(records)], dtype=np.int32),
            feature_dimension=np.asarray([values.shape[1]], dtype=np.int32),
            fitted_rows=np.asarray([len(samples)], dtype=np.int32),
            slope=slope,
            intercept=intercept,
            residual_scale=residual_scale,
            projections=PROJECTION,
        )
    temporary.replace(calibration_path)
    report_path_out = paths["reports"] / "tbank_bag_calibration_validation.json"
    _write_json(
        report_path_out,
        {
            "method": "event_grouped_cross_validation",
            "query": "dirty_shuffled_train_input_only",
            "source_labels": "previously_geometrically_verified_train_sources",
            "folds": folds,
            "summary": summary,
            "rows": details,
        },
    )
    print(f"saved calibrated bag retriever {calibration_path}; validation={summary}", flush=True)
    print(f"saved event-held-out validation {report_path_out}", flush=True)
    return calibration_path


def load_transfer_calibration(root: Path, feature_dimension: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the train-fitted source->dirty transfer for another public catalogue."""
    path = _paths(root)["calibration"]
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run calibrate-tbank first")
    with np.load(path, allow_pickle=False) as calibration:
        slope = calibration["slope"].astype(np.float32, copy=False)
        intercept = calibration["intercept"].astype(np.float32, copy=False)
        residual_scale = calibration["residual_scale"].astype(np.float32, copy=False)
        saved_dimension = int(calibration["feature_dimension"][0])
        projections = calibration["projections"].astype(np.float32, copy=False)
    if saved_dimension != feature_dimension:
        raise RuntimeError("bag calibration does not align with the source fingerprint dimension")
    if slope.shape != (feature_dimension,) or intercept.shape != slope.shape or residual_scale.shape != slope.shape:
        raise RuntimeError("bag calibration arrays have invalid dimensions")
    if not np.allclose(projections, PROJECTION):
        raise RuntimeError("bag calibration was built with a different fingerprint projection")
    return slope, intercept, residual_scale


def _load_calibration(root: Path, feature_dimension: int, record_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = _paths(root)["calibration"]
    with np.load(path, allow_pickle=False) as calibration:
        source_records = int(calibration["source_records"][0])
    if source_records != record_count:
        raise RuntimeError("bag calibration does not align with the current source index")
    return load_transfer_calibration(root, feature_dimension)


def _input_fingerprint(path: Path) -> np.ndarray:
    return bag_fingerprint(to_frags(load(str(path))))


def _highres_descriptor(frags: np.ndarray, size: int = 10) -> np.ndarray:
    """A finer photometric-invariant tile view used only to verify a shortlist."""
    if frags.shape != (TILES, 20, 20, 3):
        raise ValueError(f"expected 576 20x20 RGB fragments, got {frags.shape}")
    if 20 % size:
        raise ValueError("verification descriptor size must divide 20")
    block = 20 // size
    pooled = frags.astype(np.float32).reshape(TILES, size, block, size, block, 3).mean(axis=(2, 4))
    flat = pooled.reshape(TILES, -1)
    return (flat - flat.mean(axis=1, keepdims=True)) / (flat.std(axis=1, keepdims=True) + 1.0e-5)


def verify_source_candidate(input_path: Path, original_bgr: np.ndarray) -> tuple[dict[str, float | int | bool], np.ndarray]:
    """Verify a public original without ever reading a test target.

    A bag score only proposes candidates.  This routine independently assigns
    all 576 dirty tiles to the candidate's fixed grid using a higher-resolution
    descriptor, reconstructs the dirty frame, then counts SIFT matches that
    land at the *same pixel location* in the candidate.  Ordinary visually
    similar photos can obtain a high Hungarian score through arbitrary tile
    swaps; they do not produce spatially aligned SIFT evidence.
    """
    clean_bgr = centre_square(original_bgr, 480)
    clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
    dirty_rgb = load(str(input_path))
    dirty_frags = to_frags(dirty_rgb)
    clean_frags = to_frags(clean_rgb)
    dirty_descriptor = _highres_descriptor(dirty_frags)
    clean_descriptor = _highres_descriptor(clean_frags)
    dimension = dirty_descriptor.shape[1]
    cost = 2.0 * dimension - 2.0 * (dirty_descriptor @ clean_descriptor.T)
    row, column = linear_sum_assignment(cost)
    inverse = np.empty(TILES, dtype=np.int32)
    inverse[column] = row
    assigned_corr = (dirty_descriptor[row] * clean_descriptor[column]).sum(axis=1) / dimension
    reconstructed = from_frags(dirty_frags[inverse])

    sift = cv2.SIFT_create(nfeatures=1200)
    keypoints_clean, descriptors_clean = sift.detectAndCompute(
        cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2GRAY), None
    )
    keypoints_reconstructed, descriptors_reconstructed = sift.detectAndCompute(
        cv2.cvtColor(reconstructed, cv2.COLOR_RGB2GRAY), None
    )
    good: list[Any] = []
    aligned = 0
    ransac_inliers = 0
    median_displacement = float("inf")
    if descriptors_clean is not None and descriptors_reconstructed is not None:
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_clean, descriptors_reconstructed, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
        if good:
            points_clean = np.float32([keypoints_clean[match.queryIdx].pt for match in good])
            points_reconstructed = np.float32([keypoints_reconstructed[match.trainIdx].pt for match in good])
            displacement = np.linalg.norm(points_clean - points_reconstructed, axis=1)
            aligned = int(np.sum(displacement < 5.0))
            median_displacement = float(np.median(displacement))
            if len(good) >= 4:
                _homography, mask = cv2.findHomography(points_clean, points_reconstructed, cv2.RANSAC, 4.0)
                if mask is not None:
                    ransac_inliers = int(mask.sum())
    identity_fraction = float(aligned / len(good)) if good else 0.0
    accepted = bool(aligned >= 5 and identity_fraction >= 0.35)
    metrics: dict[str, float | int | bool] = {
        "assignment_mean_correlation": float(assigned_corr.mean()),
        "assignment_q10_correlation": float(np.quantile(assigned_corr, 0.10)),
        "sift_good_matches": int(len(good)),
        "sift_identity_matches": aligned,
        "sift_identity_fraction": identity_fraction,
        "sift_ransac_inliers": ransac_inliers,
        "sift_median_displacement": median_displacement,
        "accepted": accepted,
    }
    return metrics, clean_bgr


def _verify_candidate_task(
    input_path: Path,
    record: SourceRecord,
    *,
    timeout: float,
) -> tuple[dict[str, float | int | bool | str], bytes | None, np.ndarray | None]:
    """Download one candidate in memory and retain bytes only if verified."""
    try:
        raw, original_bgr = _download_original(record, timeout)
        metrics, clean_bgr = verify_source_candidate(input_path, original_bgr)
        payload: dict[str, float | int | bool | str] = {
            "record_id": int(record.record_id),
            "url": record.url,
            "event_slug": record.event_slug,
            **metrics,
        }
        return payload, raw if metrics["accepted"] else None, clean_bgr if metrics["accepted"] else None
    except (requests.RequestException, ValueError, cv2.error) as error:
        return {
            "record_id": int(record.record_id),
            "url": record.url,
            "event_slug": record.event_slug,
            "accepted": False,
            "error": f"{type(error).__name__}: {error}",
        }, None, None


def verify_test_candidates(
    root: Path,
    *,
    candidate_report: Path,
    top_n: int = 1,
    start_rank: int = 1,
    previous_report: Path | None = None,
    workers: int = 4,
    timeout: float = 30.0,
    limit: int | None = None,
) -> Path:
    """Apply high-precision source verification to ranked test candidates.

    Unverified originals exist only in worker memory.  The raw source and its
    480px centre crop are written under ``E:/.../found_test`` solely after the
    spatial verification gate accepts them.
    """
    if top_n < 1 or start_rank < 1 or workers < 1:
        raise ValueError("top_n, start_rank and workers must be positive")
    records = load_tbank_manifest(root)
    payload = json.loads(candidate_report.read_text(encoding="utf-8"))
    rows = list(payload["rows"])
    if previous_report is not None:
        prior = json.loads(previous_report.read_text(encoding="utf-8"))
        prior_accepted = {str(row["test"]) for row in prior["rows"] if row.get("accepted") is not None}
        rows = [row for row in rows if str(row["test"]) not in prior_accepted]
    if limit is not None:
        rows = rows[:limit]
    test_root = Path(os.environ.get("PAZZLE_DATA", r"E:/pazzle_data")) / "test"
    result_rows: list[dict[str, Any]] = [
        {"test": str(row["test"]), "attempts": [], "accepted": None} for row in rows
    ]
    futures: dict[Any, tuple[int, int, float]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for row_index, row in enumerate(rows):
            input_path = test_root / row["test"]
            begin = start_rank - 1
            for offset, candidate in enumerate(row["candidates"][begin : begin + top_n]):
                rank = start_rank + offset
                record_id = int(candidate["record_id"])
                if record_id < 0 or record_id >= len(records):
                    continue
                future = executor.submit(_verify_candidate_task, input_path, records[record_id], timeout=timeout)
                futures[future] = (row_index, rank, float(candidate["distance"]))
        completed = 0
        accepted_payloads: dict[int, list[tuple[int, dict[str, Any], bytes, np.ndarray]]] = {}
        for future in as_completed(futures):
            row_index, rank, bag_distance = futures[future]
            attempt, raw, clean_bgr = future.result()
            attempt["rank"] = rank
            attempt["bag_distance"] = bag_distance
            result_rows[row_index]["attempts"].append(attempt)
            if raw is not None and clean_bgr is not None:
                accepted_payloads.setdefault(row_index, []).append((rank, attempt, raw, clean_bgr))
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                print(f"verified {completed}/{len(futures)} public candidates", flush=True)
    found_directory = _paths(root)["images"]
    for row_index, candidates in accepted_payloads.items():
        rank, attempt, raw, clean_bgr = min(candidates, key=lambda item: item[0])
        test_name = Path(result_rows[row_index]["test"])
        stem = f"{test_name.stem}__{_safe_stem(str(attempt['event_slug']))}__{attempt['record_id']}"
        original_path = found_directory / f"{stem}.jpg"
        clean_path = found_directory / f"{stem}__centre480.png"
        original_path.write_bytes(raw)
        ok, encoded = cv2.imencode(".png", clean_bgr)
        if not ok:
            raise RuntimeError(f"could not encode verified clean crop for {test_name.name}")
        clean_path.write_bytes(encoded.tobytes())
        result_rows[row_index]["accepted"] = {
            **attempt,
            "saved_original": str(original_path),
            "saved_clean": str(clean_path),
        }
    for result in result_rows:
        result["attempts"].sort(key=lambda item: int(item["rank"]))
    suffix = "" if start_rank == 1 and previous_report is None else f"_rank{start_rank}"
    destination = _paths(root)["reports"] / f"tbank_test_verified_sources{suffix}.json"
    _write_json(
        destination,
        {
            "query": "dirty_shuffled_test_input_only",
            "candidate_source": str(candidate_report),
            "top_n": top_n,
            "start_rank": start_rank,
            "previous_report": str(previous_report) if previous_report is not None else None,
            "verification": "10x10 Hungarian tile assignment plus spatially aligned SIFT gate",
            "accepted": sum(result["accepted"] is not None for result in result_rows),
            "rows": result_rows,
        },
    )
    print(f"saved verified test-source report {destination}; accepted={sum(result['accepted'] is not None for result in result_rows)}", flush=True)
    return destination


def benchmark_train(
    root: Path,
    *,
    report_path: Path,
    limit: int | None = None,
    top_k: int = 50,
    folds: int = 5,
) -> dict[str, float]:
    """Honest event-held-out validation for dirty input -> public source."""
    records, values, valid, _center, _scale = _load_bag_index(root)
    samples = _verified_tbank_queries(root, report_path, records, valid, limit=limit)
    summary, details = validate_calibration(values, valid, samples, folds=folds, top_k=top_k)
    destination = _paths(root)["reports"] / "tbank_bag_benchmark.json"
    _write_json(
        destination,
        {
            "method": "event_grouped_cross_validation",
            "query": "dirty_shuffled_train_input_only",
            "summary": summary,
            "rows": details,
        },
    )
    print(f"saved bag benchmark {destination}: {summary}", flush=True)
    return summary


def rank_test(root: Path, *, top_k: int = 50, limit: int | None = None) -> Path:
    """Rank public-source candidates for test inputs, without downloading originals."""
    records, values, valid, _center, _scale = _load_bag_index(root)
    slope, intercept, residual_scale = _load_calibration(root, values.shape[1], len(records))
    test_root = Path(os.environ.get("PAZZLE_DATA", r"E:/pazzle_data")) / "test"
    names = sorted(test_root.glob("*.png"))
    if limit is not None:
        names = names[:limit]
    rows: list[dict[str, Any]] = []
    for number, path in enumerate(names, start=1):
        ids, distances = rank_calibrated(
            _input_fingerprint(path),
            values,
            valid,
            slope,
            intercept,
            residual_scale,
            top_k=top_k,
        )
        candidates = [
            {
                "record_id": int(record_id),
                "distance": float(distance),
                "url": records[int(record_id)].url,
                "event_slug": records[int(record_id)].event_slug,
            }
            for record_id, distance in zip(ids, distances)
        ]
        rows.append({"test": path.name, "candidates": candidates})
        if number % 50 == 0 or number == len(names):
            print(f"ranked {number}/{len(names)} test bags", flush=True)
    destination = _paths(root)["reports"] / "tbank_test_bag_candidates.json"
    _write_json(
        destination,
        {
            "source": "dirty_shuffled_bag_only_with_train_calibrated_statistics",
            "top_k": top_k,
            "rows": rows,
        },
    )
    print(f"saved test candidate report {destination}", flush=True)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index-tbank")
    index.add_argument("--workers", type=int, default=10)
    index.add_argument("--timeout", type=float, default=30.0)
    calibration = sub.add_parser("calibrate-tbank")
    calibration.add_argument("--report", type=Path, default=DEFAULT_ROOT / "matches" / "clean_targets_matches.json")
    calibration.add_argument("--folds", type=int, default=5)
    benchmark = sub.add_parser("benchmark-train")
    benchmark.add_argument("--report", type=Path, default=DEFAULT_ROOT / "matches" / "clean_targets_matches.json")
    benchmark.add_argument("--limit", type=int, default=None)
    benchmark.add_argument("--top-k", type=int, default=50)
    benchmark.add_argument("--folds", type=int, default=5)
    test = sub.add_parser("rank-test")
    test.add_argument("--top-k", type=int, default=50)
    test.add_argument("--limit", type=int, default=None)
    verify = sub.add_parser("verify-test")
    verify.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_ROOT / "matches" / "tbank_test_bag_candidates.json",
    )
    verify.add_argument("--top-n", type=int, default=1)
    verify.add_argument("--start-rank", type=int, default=1)
    verify.add_argument("--previous-report", type=Path, default=None)
    verify.add_argument("--workers", type=int, default=4)
    verify.add_argument("--timeout", type=float, default=30.0)
    verify.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    if args.command == "index-tbank":
        build_tbank_bag_index(root, workers=args.workers, timeout=args.timeout)
    elif args.command == "calibrate-tbank":
        fit_tbank_calibration(root, report_path=args.report, folds=args.folds)
    elif args.command == "benchmark-train":
        benchmark_train(root, report_path=args.report, limit=args.limit, top_k=args.top_k, folds=args.folds)
    elif args.command == "rank-test":
        rank_test(root, top_k=args.top_k, limit=args.limit)
    elif args.command == "verify-test":
        verify_test_candidates(
            root,
            candidate_report=args.candidates,
            top_n=args.top_n,
            start_rank=args.start_rank,
            previous_report=args.previous_report,
            workers=args.workers,
            timeout=args.timeout,
            limit=args.limit,
        )
    else:
        raise AssertionError(f"unexpected command {args.command!r}")


if __name__ == "__main__":
    main()
