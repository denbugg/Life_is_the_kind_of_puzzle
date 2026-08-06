"""Public-source retrieval for Central University/static event galleries.

This is deliberately separate from :mod:`bag_source_retrieval`: T-Bank has a
special historical API/proxy, whereas CU/Tilda sources are direct public image
URLs.  The statistical transfer is nevertheless the same and is learned only
from geometrically verified T-Bank train sources.  Static photos are streamed
into a compact E:-resident fingerprint index; an actual source image is saved
only after a test-time spatial verification gate accepts it.

Example::

    python src/static_bag_source_retrieval.py index --tag central_static ^
        --catalogue central_university --catalogue central_events ^
        --catalogue central_event_galleries_july11
    python src/static_bag_source_retrieval.py benchmark-train --tag central_static
    python src/static_bag_source_retrieval.py rank-test --tag central_static --top-k 20
    python src/static_bag_source_retrieval.py verify-test --tag central_static --top-n 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from bag_source_retrieval import (
    PROJECTION,
    _input_fingerprint,
    _source_frags_from_bgr,
    bag_fingerprint,
    load_transfer_calibration,
    rank_calibrated,
    verify_source_candidate,
)
from source_retrieval import DEFAULT_ROOT, SourceRecord, _decode_image, _download_original, _safe_stem


def _paths(root: Path, tag: str) -> dict[str, Path]:
    safe = _safe_stem(tag)
    manifests = root / "manifests"
    index = root / "index"
    reports = root / "matches"
    images = root / "images" / "found_test"
    for directory in (manifests, index, reports, images):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "manifest": manifests / f"static_bag_{safe}_sources.json",
        "index": index / f"static_bag_{safe}_fingerprint_index.npz",
        "partial": index / f"static_bag_{safe}_fingerprint_index.partial.npz",
        "reports": reports,
        "images": images,
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _source_digest(records: list[SourceRecord]) -> str:
    raw = "\n".join(record.url for record in records).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_static_catalogues(root: Path, catalogues: list[str]) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    seen: set[str] = set()
    for catalogue in catalogues:
        manifest = root / "manifests" / f"{_safe_stem(catalogue)}_images.json"
        if not manifest.exists():
            raise FileNotFoundError(f"missing static source manifest {manifest}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for row in payload.get("records", []):
            url = str(row["url"])
            if url in seen:
                continue
            seen.add(url)
            event_slug = f"{_safe_stem(catalogue)}__{_safe_stem(str(row.get('event_slug', 'source')))}"
            records.append(
                SourceRecord(
                    record_id=len(records),
                    url=url,
                    event_slug=event_slug,
                    event_title=str(row.get("event_title", catalogue)),
                    event_id=str(row.get("event_id", catalogue)),
                )
            )
    if not records:
        raise RuntimeError("the requested static catalogues contain no image URLs")
    return records


def _write_combined_manifest(root: Path, tag: str, catalogues: list[str]) -> list[SourceRecord]:
    records = _read_static_catalogues(root, catalogues)
    destination = _paths(root, tag)["manifest"]
    _atomic_json(
        destination,
        {
            "source": "combined_public_static_catalogues",
            "tag": tag,
            "catalogues": catalogues,
            "image_count": len(records),
            "digest": _source_digest(records),
            "records": [asdict(record) for record in records],
        },
    )
    return records


def _load_combined_manifest(root: Path, tag: str) -> list[SourceRecord]:
    path = _paths(root, tag)["manifest"]
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run index first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [SourceRecord(**row) for row in payload["records"]]
    if [record.record_id for record in records] != list(range(len(records))):
        raise RuntimeError("static source manifest has non-contiguous record IDs")
    return records


def _fetch_fingerprint(record: SourceRecord, timeout: float) -> tuple[int, np.ndarray | None]:
    try:
        response = requests.get(
            record.url,
            timeout=timeout,
            headers={"User-Agent": "PAZZLE-static-bag-retrieval/1.0"},
        )
        response.raise_for_status()
        image = _decode_image(response.content)
        return record.record_id, bag_fingerprint(_source_frags_from_bgr(image))
    except (requests.RequestException, ValueError, cv2.error):
        return record.record_id, None


def _save_index(path: Path, records: list[SourceRecord], fingerprints: np.ndarray, valid: np.ndarray) -> None:
    _atomic_npz(
        path,
        record_ids=np.arange(len(records), dtype=np.int32),
        fingerprints=fingerprints,
        valid=valid,
        source_digest=np.asarray([_source_digest(records)]),
        projections=PROJECTION,
    )


def build_index(
    root: Path,
    *,
    tag: str,
    catalogues: list[str],
    workers: int = 4,
    timeout: float = 20.0,
    reuse_tags: list[str] | None = None,
) -> Path:
    """Build a resumable numeric-only index from direct public static URLs."""
    if workers < 1:
        raise ValueError("workers must be positive")
    records = _write_combined_manifest(root, tag, catalogues)
    paths = _paths(root, tag)
    index_path = paths["index"]
    dimension = 960
    digest = _source_digest(records)
    if index_path.exists():
        with np.load(index_path, allow_pickle=False) as existing:
            aligned = (
                np.array_equal(existing["record_ids"], np.arange(len(records), dtype=np.int32))
                and existing["fingerprints"].shape == (len(records), dimension)
                and str(existing["source_digest"][0]) == digest
                and np.allclose(existing["projections"].astype(np.float32, copy=False), PROJECTION)
            )
            if aligned and existing["valid"].astype(bool, copy=False).all():
                print(f"reusing static bag index {index_path}", flush=True)
                return index_path
    fingerprints = np.zeros((len(records), dimension), dtype=np.float16)
    valid = np.zeros(len(records), dtype=bool)
    # Exact URL reuse is safe: all static indices use the same direct-image
    # centre-crop transform and projection.  It avoids re-fetching already
    # fingerprinted public bytes while still keeping only numeric features.
    for reuse_tag in reuse_tags or []:
        try:
            prior_records, prior_values, prior_valid = _load_index(root, reuse_tag)
            prior_by_url = {
                record.url: prior_values[record.record_id]
                for record in prior_records
                if prior_valid[record.record_id]
            }
            reused = 0
            for record in records:
                value = prior_by_url.get(record.url)
                if value is not None:
                    fingerprints[record.record_id] = value.astype(np.float16)
                    valid[record.record_id] = True
                    reused += 1
            print(f"reused {reused} static fingerprints from tag {reuse_tag}", flush=True)
        except (FileNotFoundError, RuntimeError, OSError, KeyError, ValueError):
            print(f"could not reuse static tag {reuse_tag}; indexing its URLs directly", flush=True)
    if index_path.exists():
        try:
            with np.load(index_path, allow_pickle=False) as existing:
                if (
                    np.array_equal(existing["record_ids"], np.arange(len(records), dtype=np.int32))
                    and existing["fingerprints"].shape == fingerprints.shape
                    and str(existing["source_digest"][0]) == digest
                    and np.allclose(existing["projections"].astype(np.float32, copy=False), PROJECTION)
                ):
                    fingerprints[:] = existing["fingerprints"]
                    valid[:] = existing["valid"].astype(bool, copy=False)
                    print(f"retrying {len(records) - int(valid.sum())} previously unavailable static URLs", flush=True)
        except (OSError, KeyError, ValueError):
            pass
    partial_path = paths["partial"]
    if partial_path.exists():
        try:
            with np.load(partial_path, allow_pickle=False) as partial:
                if (
                    np.array_equal(partial["record_ids"], np.arange(len(records), dtype=np.int32))
                    and partial["fingerprints"].shape == fingerprints.shape
                    and str(partial["source_digest"][0]) == digest
                    and np.allclose(partial["projections"].astype(np.float32, copy=False), PROJECTION)
                ):
                    fingerprints[:] = partial["fingerprints"]
                    valid[:] = partial["valid"].astype(bool, copy=False)
                    print(f"resuming static index: {int(valid.sum())}/{len(records)} already cached", flush=True)
        except (OSError, KeyError, ValueError):
            print(f"ignoring unreadable partial index {partial_path}", flush=True)
    pending = [record for record in records if not valid[record.record_id]]
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_fetch_fingerprint, record, timeout) for record in pending]
            for completed, future in enumerate(as_completed(futures), start=1):
                record_id, fingerprint = future.result()
                if fingerprint is not None:
                    fingerprints[record_id] = fingerprint.astype(np.float16)
                    valid[record_id] = True
                if completed % 100 == 0 or completed == len(pending):
                    _save_index(partial_path, records, fingerprints, valid)
                    print(
                        f"static bag index {completed}/{len(pending)} pending; valid={int(valid.sum())}/{len(records)} "
                        f"elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
    finally:
        _save_index(partial_path, records, fingerprints, valid)
    _save_index(index_path, records, fingerprints, valid)
    partial_path.unlink(missing_ok=True)
    print(f"saved static bag index {index_path}; valid={int(valid.sum())}/{len(records)}", flush=True)
    return index_path


def _load_index(root: Path, tag: str) -> tuple[list[SourceRecord], np.ndarray, np.ndarray]:
    records = _load_combined_manifest(root, tag)
    path = _paths(root, tag)["index"]
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run index first")
    with np.load(path, allow_pickle=False) as index:
        ids = index["record_ids"].astype(np.int64, copy=False)
        fingerprints = index["fingerprints"].astype(np.float32, copy=False)
        valid = index["valid"].astype(bool, copy=False)
        saved_digest = str(index["source_digest"][0])
        saved_projection = index["projections"].astype(np.float32, copy=False)
    if not np.array_equal(ids, np.arange(len(records), dtype=np.int64)) or fingerprints.shape != (len(records), 960):
        raise RuntimeError("static bag index does not align with its manifest")
    if saved_digest != _source_digest(records):
        raise RuntimeError("static bag index was built for a different source manifest")
    if not np.allclose(saved_projection, PROJECTION):
        raise RuntimeError("static bag index was built with a different fingerprint projection")
    return records, fingerprints, valid


def _rank_summary(ranks: list[int]) -> dict[str, float]:
    value = np.asarray(ranks, dtype=np.int64)
    if not value.size:
        raise RuntimeError("no static source-verified train rows belong to this catalogue")
    return {
        "queries": int(value.size),
        "r1": float(np.mean(value == 1)),
        "r5": float(np.mean(value <= 5)),
        "r20": float(np.mean(value <= 20)),
        "r50": float(np.mean(value <= 50)),
        "median_rank": float(np.median(value)),
        "mean_rank": float(value.mean()),
    }


def benchmark_train(root: Path, *, tag: str, reports: list[Path], top_k: int = 50) -> Path:
    """Validate T-Bank-trained transfer on independently verified CU/static rows."""
    records, values, valid = _load_index(root, tag)
    slope, intercept, residual_scale = load_transfer_calibration(root, values.shape[1])
    source_id_by_url = {record.url: record.record_id for record in records}
    input_root = Path(os.environ.get("PAZZLE_DATA", r"E:/pazzle_data")) / "train" / "inputs"
    # A few public galleries expose the same photograph through URL aliases.
    # Either alias is an equivalent clean answer, so validate against the set.
    truth_by_target: dict[str, set[int]] = {}
    for report_path in reports:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            if not row.get("accepted"):
                continue
            source_id = source_id_by_url.get(row["accepted"]["url"])
            if source_id is not None and valid[source_id]:
                truth_by_target.setdefault(str(row["target"]), set()).add(int(source_id))
    details: list[dict[str, Any]] = []
    ranks: list[int] = []
    for target, source_ids in sorted(truth_by_target.items()):
        candidates, distance = rank_calibrated(
            _input_fingerprint(input_root / target),
            values,
            valid,
            slope,
            intercept,
            residual_scale,
            top_k=int(valid.sum()),
        )
        locations = [int(location[0] + 1) for source_id in source_ids if (location := np.flatnonzero(candidates == source_id)).size]
        rank = min(locations) if locations else int(valid.sum()) + 1
        ranks.append(rank)
        details.append(
            {
                "target": target,
                "truth_record_ids": sorted(source_ids),
                "rank": rank,
                "top": candidates[:top_k].astype(int).tolist(),
                "top_distance": distance[:top_k].astype(float).tolist(),
            }
        )
    destination = _paths(root, tag)["reports"] / f"static_bag_{_safe_stem(tag)}_benchmark.json"
    _atomic_json(
        destination,
        {
            "method": "T-Bank-train-calibrated transfer to independent static catalogue",
            "summary": _rank_summary(ranks),
            "rows": details,
        },
    )
    print(f"saved static bag benchmark {destination}: {_rank_summary(ranks)}", flush=True)
    return destination


def rank_test(root: Path, *, tag: str, top_k: int = 20, limit: int | None = None) -> Path:
    records, values, valid = _load_index(root, tag)
    slope, intercept, residual_scale = load_transfer_calibration(root, values.shape[1])
    test_root = Path(os.environ.get("PAZZLE_DATA", r"E:/pazzle_data")) / "test"
    names = sorted(test_root.glob("*.png"))
    if limit is not None:
        names = names[:limit]
    rows: list[dict[str, Any]] = []
    for number, path in enumerate(names, start=1):
        ids, distances = rank_calibrated(
            _input_fingerprint(path), values, valid, slope, intercept, residual_scale, top_k=top_k
        )
        rows.append(
            {
                "test": path.name,
                "candidates": [
                    {
                        "record_id": int(record_id),
                        "distance": float(distance),
                        "url": records[int(record_id)].url,
                        "event_slug": records[int(record_id)].event_slug,
                    }
                    for record_id, distance in zip(ids, distances)
                ],
            }
        )
        if number % 50 == 0 or number == len(names):
            print(f"ranked {number}/{len(names)} static test bags", flush=True)
    destination = _paths(root, tag)["reports"] / f"static_bag_{_safe_stem(tag)}_test_candidates.json"
    _atomic_json(
        destination,
        {
            "query": "dirty_shuffled_test_input_only",
            "source": "direct_public_static_catalogues_with_T-Bank-train-calibrated_transfer",
            "top_k": top_k,
            "rows": rows,
        },
    )
    print(f"saved static test candidate report {destination}", flush=True)
    return destination


def _verify_task(input_path: Path, record: SourceRecord, timeout: float) -> tuple[dict[str, Any], bytes | None, np.ndarray | None]:
    try:
        raw, original_bgr = _download_original(record, timeout)
        metrics, clean_bgr = verify_source_candidate(input_path, original_bgr)
        attempt: dict[str, Any] = {
            "record_id": int(record.record_id),
            "url": record.url,
            "event_slug": record.event_slug,
            **metrics,
        }
        return attempt, raw if metrics["accepted"] else None, clean_bgr if metrics["accepted"] else None
    except (requests.RequestException, ValueError, cv2.error) as error:
        return {
            "record_id": int(record.record_id),
            "url": record.url,
            "event_slug": record.event_slug,
            "accepted": False,
            "error": f"{type(error).__name__}: {error}",
        }, None, None


def verify_test(
    root: Path,
    *,
    tag: str,
    candidate_report: Path | None,
    top_n: int = 1,
    start_rank: int = 1,
    previous_report: Path | None = None,
    catalogue_prefix: str | None = None,
    workers: int = 4,
    timeout: float = 20.0,
    limit: int | None = None,
) -> Path:
    if top_n < 1 or start_rank < 1 or workers < 1:
        raise ValueError("top_n, start_rank and workers must be positive")
    records, _values, _valid = _load_index(root, tag)
    paths = _paths(root, tag)
    if candidate_report is None:
        candidate_report = paths["reports"] / f"static_bag_{_safe_stem(tag)}_test_candidates.json"
    payload = json.loads(candidate_report.read_text(encoding="utf-8"))
    rows = list(payload["rows"])
    if previous_report is not None:
        prior = json.loads(previous_report.read_text(encoding="utf-8"))
        accepted_tests = {str(row["test"]) for row in prior["rows"] if row.get("accepted") is not None}
        rows = [row for row in rows if str(row["test"]) not in accepted_tests]
    if catalogue_prefix is not None:
        begin = start_rank - 1
        rows = [
            row
            for row in rows
            if any(
                str(candidate.get("event_slug", "")).startswith(catalogue_prefix)
                for candidate in row["candidates"][begin : begin + top_n]
            )
        ]
    if limit is not None:
        rows = rows[:limit]
    test_root = Path(os.environ.get("PAZZLE_DATA", r"E:/pazzle_data")) / "test"
    result_rows: list[dict[str, Any]] = [
        {"test": str(row["test"]), "attempts": [], "accepted": None} for row in rows
    ]
    futures: dict[Any, tuple[int, int, float]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for row_index, row in enumerate(rows):
            begin = start_rank - 1
            for offset, candidate in enumerate(row["candidates"][begin : begin + top_n]):
                rank = start_rank + offset
                record_id = int(candidate["record_id"])
                if 0 <= record_id < len(records):
                    future = executor.submit(_verify_task, test_root / row["test"], records[record_id], timeout)
                    futures[future] = (row_index, rank, float(candidate["distance"]))
        accepted_payloads: dict[int, list[tuple[int, dict[str, Any], bytes, np.ndarray]]] = {}
        for completed, future in enumerate(as_completed(futures), start=1):
            row_index, rank, bag_distance = futures[future]
            attempt, raw, clean_bgr = future.result()
            attempt["rank"] = rank
            attempt["bag_distance"] = bag_distance
            result_rows[row_index]["attempts"].append(attempt)
            if raw is not None and clean_bgr is not None:
                accepted_payloads.setdefault(row_index, []).append((rank, attempt, raw, clean_bgr))
            if completed % 25 == 0 or completed == len(futures):
                print(f"verified {completed}/{len(futures)} static public candidates", flush=True)
    for row_index, candidates in accepted_payloads.items():
        _rank, attempt, raw, clean_bgr = min(candidates, key=lambda item: item[0])
        test_name = Path(result_rows[row_index]["test"])
        stem = f"static_{_safe_stem(tag)}__{test_name.stem}__{_safe_stem(str(attempt['event_slug']))}__{attempt['record_id']}"
        original_path = paths["images"] / f"{stem}.jpg"
        clean_path = paths["images"] / f"{stem}__centre480.png"
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
    for row in result_rows:
        row["attempts"].sort(key=lambda item: int(item["rank"]))
    suffix = "" if start_rank == 1 and previous_report is None else f"_rank{start_rank}"
    destination = paths["reports"] / f"static_bag_{_safe_stem(tag)}_test_verified_sources{suffix}.json"
    _atomic_json(
        destination,
        {
            "query": "dirty_shuffled_test_input_only",
            "candidate_source": str(candidate_report),
            "top_n": top_n,
            "start_rank": start_rank,
            "previous_report": str(previous_report) if previous_report is not None else None,
            "catalogue_prefix": catalogue_prefix,
            "verification": "10x10 Hungarian tile assignment plus spatially aligned SIFT gate",
            "accepted": sum(row["accepted"] is not None for row in result_rows),
            "rows": result_rows,
        },
    )
    print(f"saved static verified-source report {destination}; accepted={sum(row['accepted'] is not None for row in result_rows)}", flush=True)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index")
    index.add_argument("--tag", required=True)
    index.add_argument("--catalogue", action="append", required=True)
    index.add_argument("--workers", type=int, default=4)
    index.add_argument("--timeout", type=float, default=20.0)
    index.add_argument("--reuse-tag", action="append", default=None)
    benchmark = sub.add_parser("benchmark-train")
    benchmark.add_argument("--tag", required=True)
    benchmark.add_argument("--report", action="append", type=Path, default=None)
    benchmark.add_argument("--top-k", type=int, default=50)
    rank = sub.add_parser("rank-test")
    rank.add_argument("--tag", required=True)
    rank.add_argument("--top-k", type=int, default=20)
    rank.add_argument("--limit", type=int, default=None)
    verify = sub.add_parser("verify-test")
    verify.add_argument("--tag", required=True)
    verify.add_argument("--candidates", type=Path, default=None)
    verify.add_argument("--top-n", type=int, default=1)
    verify.add_argument("--start-rank", type=int, default=1)
    verify.add_argument("--previous-report", type=Path, default=None)
    verify.add_argument("--catalogue-prefix", default=None, help="only verify ranked candidates with this event_slug prefix")
    verify.add_argument("--workers", type=int, default=4)
    verify.add_argument("--timeout", type=float, default=20.0)
    verify.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    if args.command == "index":
        build_index(
            root,
            tag=args.tag,
            catalogues=args.catalogue,
            workers=args.workers,
            timeout=args.timeout,
            reuse_tags=args.reuse_tag,
        )
    elif args.command == "benchmark-train":
        reports = args.report or [
            root / "matches" / "central_university_clean_matches.json",
            root / "matches" / "central_events_clean_matches.json",
        ]
        benchmark_train(root, tag=args.tag, reports=reports, top_k=args.top_k)
    elif args.command == "rank-test":
        rank_test(root, tag=args.tag, top_k=args.top_k, limit=args.limit)
    elif args.command == "verify-test":
        verify_test(
            root,
            tag=args.tag,
            candidate_report=args.candidates,
            top_n=args.top_n,
            start_rank=args.start_rank,
            previous_report=args.previous_report,
            catalogue_prefix=args.catalogue_prefix,
            workers=args.workers,
            timeout=args.timeout,
            limit=args.limit,
        )
    else:
        raise AssertionError(f"unexpected command {args.command!r}")


if __name__ == "__main__":
    main()
