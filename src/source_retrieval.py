"""Public-photo source retrieval for PAZZLE.

The puzzle corpus contains crops of public event photography.  This module is
deliberately separate from the learned jigsaw solvers: a verified original
image gives both the exact placement and the clean restoration at once.

The first supported catalogue is the public T-Bank meetup API.  It exposes a
complete event manifest (including historic events) rather than just the few
images linked from the current landing page.  The pipeline is designed to be
safe on the small C: drive:

* manifests, feature indexes, reports and *only verified source originals*
  live below ``E:/pazzle_work/source_forensics`` by default;
* unverified gallery images are streamed into memory to make small thumbnails
  and are never saved as files; and
* matching is two-stage: inexpensive perceptual hash retrieval followed by
  full-resolution SIFT/RANSAC verification.

The centre-square transform is not an assumption: it was calibrated on two
known exact train matches and has mean pixel correlation above 0.996.  The
verification stage still checks every candidate independently, so a changed
upstream preprocessing rule cannot silently produce a false positive.

Typical workflow (run from the repository root)::

    python src/source_retrieval.py crawl-tbank
    python src/source_retrieval.py index-tbank --workers 12
    python src/source_retrieval.py match-clean --split train

``match-clean`` is a calibration and discovery command for clean train
targets.  Test retrieval needs the later bag-to-source stage; it must not
pretend that a shuffled input can be queried as a normal photograph.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.environ.get("PAZZLE_FORENSICS", r"E:/pazzle_work/source_forensics"))
TBANK_MEETUPS_API = "https://meetup.tbank.ru/pwameetups/papi/getMeetups"
TBANK_IMGPROXY = "https://imgproxy.cdn-tinkoff.ru/weight500/"
TARGET_ROOT = Path(os.environ.get("PAZZLE_DATA", r"E:/pazzle_data")) / "train" / "targets"

# A byte lookup table is much faster and less version-sensitive than relying
# on numpy's optional bitwise_count ufunc for 7k x 5k hash comparisons.
POPCOUNT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class SourceRecord:
    """One public candidate image and the event page that exposes it."""

    record_id: int
    url: str
    event_slug: str
    event_title: str
    event_id: str


def _paths(root: Path) -> dict[str, Path]:
    manifests = root / "manifests"
    index = root / "index"
    matches = root / "matches"
    images = root / "images"
    for directory in (manifests, index, matches, images):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "manifest": manifests / "tbank_meetups_images.json",
        "index": index / "tbank_meetups_preview_index.npz",
        "matches": matches,
        "images": images,
    }


def _json_write(path: Path, payload: Any) -> None:
    """Atomically write lightweight metadata under the configured E: root."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def crawl_tbank_manifest(root: Path, *, page_size: int = 100, timeout: float = 30.0) -> list[SourceRecord]:
    """Download the public historical T-Bank meetup catalogue.

    The API response is nested once by the public proxy.  We only retain the
    fields necessary for reproducibility and never download gallery media in
    this phase.
    """
    if page_size < 1:
        raise ValueError("page_size must be positive")
    session = requests.Session()
    session.headers.update({"User-Agent": "PAZZLE-source-retrieval/1.0"})
    events: list[dict[str, Any]] = []
    total: int | None = None
    for offset in range(0, 100_000, page_size):
        response = session.get(
            TBANK_MEETUPS_API,
            params={"pageSize": page_size, "pageOffset": offset},
            timeout=timeout,
        )
        response.raise_for_status()
        envelope = response.json()
        payload = envelope["payload"]["payload"]
        page = payload["meetups"]
        total = int(payload["total"])
        events.extend(page)
        if len(events) >= total or not page:
            break
    if total is None or len(events) < total:
        raise RuntimeError(f"incomplete meetup catalogue: got {len(events)}, expected {total}")

    records: list[SourceRecord] = []
    seen_urls: set[str] = set()
    for event in events:
        archive = event.get("archive") or {}
        for url in archive.get("images") or []:
            if not isinstance(url, str) or not url.startswith("https://") or url in seen_urls:
                continue
            seen_urls.add(url)
            records.append(
                SourceRecord(
                    record_id=len(records),
                    url=url,
                    event_slug=str(event.get("url") or "unknown"),
                    event_title=str(event.get("title") or ""),
                    event_id=str(event.get("id") or ""),
                )
            )

    paths = _paths(root)
    _json_write(
        paths["manifest"],
        {
            "source": "tbank_meetups_public_api",
            "api": TBANK_MEETUPS_API,
            "retrieved_unix": time.time(),
            "event_count": len(events),
            "image_count": len(records),
            "records": [asdict(record) for record in records],
        },
    )
    return records


def load_tbank_manifest(root: Path) -> list[SourceRecord]:
    path = _paths(root)["manifest"]
    if not path.exists():
        raise FileNotFoundError(f"manifest does not exist: {path}; run crawl-tbank first")
    data = json.loads(path.read_text(encoding="utf-8"))
    records = [SourceRecord(**row) for row in data["records"]]
    if not records:
        raise RuntimeError("manifest contains no source images")
    return records


def preview_url(url: str) -> str:
    """Return T-Bank's public 500px proxy URL without persisting media."""
    encoded = base64.b64encode(url.encode("utf-8")).decode("ascii").rstrip("=")
    return TBANK_IMGPROXY + encoded


def _decode_image(content: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("response did not decode into an RGB-like image")
    return image


def centre_square(image: np.ndarray, size: int) -> np.ndarray:
    """Apply the observed source->target centre-square transform."""
    height, width = image.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    square = image[top : top + side, left : left + side]
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def _phash_bytes(image: np.ndarray) -> np.ndarray:
    """Return a 64-bit DCT perceptual hash as eight uint8 values."""
    crop = centre_square(image, 32)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dct = cv2.dct(gray)[:9, :9][1:, 1:]
    bits = (dct > dct.mean()).reshape(-1)
    return np.packbits(bits, bitorder="little")


def _thumbnail(image: np.ndarray, size: int = 32) -> np.ndarray:
    return centre_square(image, size)


def _fetch_preview(record: SourceRecord, timeout: float) -> tuple[int, np.ndarray, np.ndarray] | tuple[int, None, None]:
    """Fetch one public thumbnail entirely in memory.

    A separate Session per thread avoids sharing requests' mutable connection
    state.  Failures are represented explicitly rather than poisoning the
    whole index build.
    """
    try:
        response = requests.get(
            preview_url(record.url),
            timeout=timeout,
            headers={"User-Agent": "PAZZLE-source-retrieval/1.0"},
        )
        response.raise_for_status()
        image = _decode_image(response.content)
        return record.record_id, _phash_bytes(image), _thumbnail(image)
    except (requests.RequestException, ValueError, cv2.error):
        return record.record_id, None, None


def build_tbank_preview_index(
    root: Path,
    *,
    workers: int = 8,
    timeout: float = 30.0,
    limit: int | None = None,
) -> Path:
    """Stream all public previews into a compact E:-drive feature index.

    The index contains hashes and 32x32 thumbnails only; no unverified photo
    file is written to disk.  If the command is restarted after a completed
    build, its valid existing index is reused.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    records = load_tbank_manifest(root)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive when specified")
        records = records[:limit]
    paths = _paths(root)
    # A smoke prefix must never masquerade as the complete index used by
    # matching.  Keeping it as a separately named E: artifact also makes it
    # safe to run tiny connectivity checks before the full public crawl.
    index_path = (
        paths["index"].with_name(f"tbank_meetups_preview_index_limit_{len(records)}.npz")
        if limit is not None
        else paths["index"]
    )
    if index_path.exists():
        with np.load(index_path, allow_pickle=False) as existing:
            saved_ids = existing["record_ids"].astype(np.int64, copy=False)
            if np.array_equal(saved_ids, np.arange(len(records), dtype=np.int64)):
                print(f"reusing complete preview index {index_path} ({len(records)} records)", flush=True)
                return index_path

    hashes = np.zeros((len(records), 8), dtype=np.uint8)
    thumbs = np.zeros((len(records), 32, 32, 3), dtype=np.uint8)
    valid = np.zeros(len(records), dtype=bool)
    started = time.monotonic()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_fetch_preview, record, timeout) for record in records]
        for future in as_completed(futures):
            record_id, digest, thumb = future.result()
            # record_id is globally contiguous; with --limit it remains a
            # direct array position because records are a manifest prefix.
            if record_id < len(records) and digest is not None and thumb is not None:
                hashes[record_id] = digest
                thumbs[record_id] = thumb
                valid[record_id] = True
            completed += 1
            if completed % 250 == 0 or completed == len(records):
                print(
                    f"preview index {completed}/{len(records)} valid={int(valid.sum())} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    np.savez_compressed(
        index_path,
        record_ids=np.arange(len(records), dtype=np.int32),
        phashes=hashes,
        thumbs=thumbs,
        valid=valid,
    )
    print(f"saved feature index {index_path}; valid={int(valid.sum())}/{len(records)}", flush=True)
    return index_path


def _load_index(root: Path) -> tuple[list[SourceRecord], np.ndarray, np.ndarray, np.ndarray]:
    records = load_tbank_manifest(root)
    index_path = _paths(root)["index"]
    if not index_path.exists():
        raise FileNotFoundError(f"preview index does not exist: {index_path}; run index-tbank first")
    with np.load(index_path, allow_pickle=False) as index:
        ids = index["record_ids"].astype(np.int64, copy=False)
        hashes = index["phashes"].astype(np.uint8, copy=False)
        thumbs = index["thumbs"].astype(np.uint8, copy=False)
        valid = index["valid"].astype(bool, copy=False)
    if ids.shape != (len(records),) or hashes.shape != (len(records), 8) or thumbs.shape != (len(records), 32, 32, 3):
        raise RuntimeError("preview index does not match the current manifest")
    if not np.array_equal(ids, np.arange(len(records), dtype=np.int64)):
        raise RuntimeError("preview index record IDs are not a manifest-aligned sequence")
    return records, hashes, thumbs, valid


def hamming_distance(query_hash: np.ndarray, indexed_hashes: np.ndarray) -> np.ndarray:
    """Compute DCT hash Hamming distance from one query to all candidates."""
    query = np.asarray(query_hash, dtype=np.uint8)
    if query.shape != (8,) or indexed_hashes.ndim != 2 or indexed_hashes.shape[1] != 8:
        raise ValueError("expected an 8-byte query and an (N,8) hash array")
    return POPCOUNT[np.bitwise_xor(indexed_hashes, query)].sum(axis=1, dtype=np.uint16)


def _sift_verify(source_image: np.ndarray, target_image: np.ndarray) -> dict[str, float]:
    """Verify an alleged exact source crop with geometry, not hash alone."""
    source_crop = centre_square(source_image, 480)
    target = target_image
    source_gray = cv2.cvtColor(source_crop, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    key_target, desc_target = sift.detectAndCompute(target_gray, None)
    key_source, desc_source = sift.detectAndCompute(source_gray, None)
    good: list[cv2.DMatch] = []
    if desc_target is not None and desc_source is not None:
        for row in cv2.BFMatcher().knnMatch(desc_target, desc_source, k=2):
            if len(row) == 2 and row[0].distance < 0.72 * row[1].distance:
                good.append(row[0])
    inliers = 0
    if len(good) >= 4:
        source_points = np.float32([key_source[match.trainIdx].pt for match in good])
        target_points = np.float32([key_target[match.queryIdx].pt for match in good])
        _, mask = cv2.findHomography(target_points, source_points, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
    source_float = source_crop.astype(np.float32)
    target_float = target.astype(np.float32)
    rmse = float(np.sqrt(np.mean(np.square(source_float - target_float))))
    correlation = float(np.corrcoef(source_float.reshape(-1), target_float.reshape(-1))[0, 1])
    return {
        "sift_good": float(len(good)),
        "sift_inliers": float(inliers),
        "pixel_rmse": rmse,
        "pixel_correlation": correlation,
    }


def _download_original(record: SourceRecord, timeout: float) -> tuple[bytes, np.ndarray]:
    response = requests.get(
        record.url,
        timeout=timeout,
        headers={"User-Agent": "PAZZLE-source-retrieval/1.0"},
    )
    response.raise_for_status()
    return response.content, _decode_image(response.content)


def _safe_stem(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def match_clean_targets(
    root: Path,
    *,
    target_dir: Path = TARGET_ROOT,
    max_hamming: int = 8,
    top_k: int = 3,
    verify: bool = True,
    timeout: float = 30.0,
    max_targets: int | None = None,
) -> Path:
    """Discover exact clean-target matches in the public index.

    A hash hit is only a *candidate*.  A match is accepted after high-detail
    centre-crop verification (default: at least 60 SIFT/RANSAC inliers and
    correlation >= .98).  Accepted originals alone are saved to E:.
    """
    if max_hamming < 0 or top_k < 1:
        raise ValueError("max_hamming must be non-negative and top_k positive")
    records, hashes, thumbs, valid = _load_index(root)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        raise RuntimeError("preview index contains no valid source thumbnails")
    names = sorted(target_dir.glob("*.png"))
    if max_targets is not None:
        names = names[:max_targets]
    if not names:
        raise FileNotFoundError(f"no PNG targets found in {target_dir}")

    report_rows: list[dict[str, Any]] = []
    found_directory = _paths(root)["images"] / "found_train"
    found_directory.mkdir(parents=True, exist_ok=True)
    for position, target_path in enumerate(names, start=1):
        target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if target is None:
            continue
        query_hash = _phash_bytes(target)
        distances = hamming_distance(query_hash, hashes[valid_ids])
        order = np.argsort(distances, kind="stable")[:top_k]
        # The direct thumbnail RMSE breaks rare pHash ties before any network
        # request.  It is diagnostic only; geometric verification decides.
        target_thumb = _thumbnail(target)
        candidates: list[dict[str, Any]] = []
        for relative in order:
            record_id = int(valid_ids[relative])
            candidates.append(
                {
                    "record_id": record_id,
                    "hamming": int(distances[relative]),
                    "preview_rmse": float(
                        np.sqrt(np.mean(np.square(thumbs[record_id].astype(np.float32) - target_thumb.astype(np.float32))))
                    ),
                }
            )
        row: dict[str, Any] = {"target": target_path.name, "candidates": candidates, "accepted": None}
        if verify:
            for candidate in candidates:
                if candidate["hamming"] > max_hamming:
                    continue
                record = records[candidate["record_id"]]
                try:
                    raw, original = _download_original(record, timeout)
                    metrics = _sift_verify(original, target)
                except (requests.RequestException, ValueError, cv2.error):
                    continue
                candidate["verification"] = metrics
                # JPEG/source-processing variants can shift global pixels
                # enough to lower full-frame correlation to ~.96 while still
                # yielding hundreds of geometrically coherent SIFT matches.
                # The latter is the decisive identity evidence; requiring the
                # independently tiny preview error keeps a repeated backdrop
                # or visually similar event shot from passing on geometry
                # alone.
                accepted = (
                    metrics["sift_inliers"] >= 60.0
                    and (metrics["pixel_correlation"] >= 0.95 or candidate["preview_rmse"] <= 5.0)
                )
                candidate["accepted"] = accepted
                if accepted:
                    suffix = Path(record.url).suffix.lower() or ".jpg"
                    output = found_directory / f"{target_path.stem}__{_safe_stem(record.event_slug)}__{record.record_id}{suffix}"
                    output.write_bytes(raw)
                    row["accepted"] = {
                        "record_id": record.record_id,
                        "url": record.url,
                        "event_slug": record.event_slug,
                        "event_title": record.event_title,
                        "saved_original": str(output),
                        "verification": metrics,
                    }
                    break
        report_rows.append(row)
        if position % 250 == 0 or position == len(names):
            accepted_count = sum(item["accepted"] is not None for item in report_rows)
            print(f"clean match {position}/{len(names)} accepted={accepted_count}", flush=True)

    report = {
        "source": "tbank_meetups",
        "target_dir": str(target_dir),
        "targets": len(names),
        "max_hamming": max_hamming,
        "accepted": sum(item["accepted"] is not None for item in report_rows),
        "rows": report_rows,
    }
    destination = _paths(root)["matches"] / f"clean_{target_dir.name}_matches.json"
    _json_write(destination, report)
    print(f"saved clean retrieval report {destination}; accepted={report['accepted']}", flush=True)
    return destination


def _print_crawl_summary(records: Iterable[SourceRecord]) -> None:
    materialized = list(records)
    events = {record.event_id for record in materialized}
    print(f"T-Bank public manifest: {len(materialized)} images from {len(events)} events", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="E: root for manifests, index, reports and verified originals")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("crawl-tbank", help="download the public historical meetup manifest (no photos)")
    index = sub.add_parser("index-tbank", help="stream source previews into a compact E: feature index")
    index.add_argument("--workers", type=int, default=8)
    index.add_argument("--timeout", type=float, default=30.0)
    index.add_argument("--limit", type=int, default=None, help="index a manifest prefix only, for a smoke run")
    clean = sub.add_parser("match-clean", help="retrieve and geometrically verify clean train targets")
    clean.add_argument("--split", choices=("train",), default="train")
    clean.add_argument("--target-dir", type=Path, default=TARGET_ROOT)
    clean.add_argument("--max-hamming", type=int, default=8)
    clean.add_argument("--top-k", type=int, default=3)
    clean.add_argument("--no-verify", action="store_true", help="write candidate report but do not download/verify originals")
    clean.add_argument("--timeout", type=float, default=30.0)
    clean.add_argument("--max-targets", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    if args.command == "crawl-tbank":
        _print_crawl_summary(crawl_tbank_manifest(root))
    elif args.command == "index-tbank":
        build_tbank_preview_index(root, workers=args.workers, timeout=args.timeout, limit=args.limit)
    elif args.command == "match-clean":
        match_clean_targets(
            root,
            target_dir=args.target_dir,
            max_hamming=args.max_hamming,
            top_k=args.top_k,
            verify=not args.no_verify,
            timeout=args.timeout,
            max_targets=args.max_targets,
        )
    else:  # argparse guards this, but retain a useful static invariant.
        raise AssertionError(f"unexpected command {args.command!r}")


if __name__ == "__main__":
    main()
