"""Crawl public sitemap-backed photo sources and retrieve PAZZLE originals.

This companion to :mod:`source_retrieval` handles public Central University
and T-Education pages whose image lists live in HTML/static assets rather than
the T-Bank meetup API.  It follows the same storage rule: feature manifests
and reports go to ``E:/pazzle_work/source_forensics``; an actual image file is
written only after a geometric exact-match verification succeeds.

Examples::

    python src/static_source_retrieval.py crawl-sitemap --name central_university \
        --sitemap https://cu.ru/sitemap.xml
    python src/static_source_retrieval.py index --name central_university --workers 12
    python src/static_source_retrieval.py match-clean --name central_university
"""

from __future__ import annotations

import argparse
import html as html_module
import json
import os
import re
import time
import xml.etree.ElementTree as element_tree
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup

from source_retrieval import (
    DEFAULT_ROOT,
    POPCOUNT,
    SourceRecord,
    TARGET_ROOT,
    _decode_image,
    _download_original,
    _phash_bytes,
    _safe_stem,
    _sift_verify,
    _thumbnail,
    hamming_distance,
)


# HTML is often a mixture of img/srcset/CSS/serialized JSON.  Extracting URLs
# rather than assuming one framework keeps this useful for both Angular and
# older Tilda event pages.
IMAGE_URL = re.compile(
    r"(?:(?:https?:)?//|/)[^\"'\\()<>\s]+?\.(?:jpe?g|png|webp)(?:\?[^\"'\\()<>\s]*)?",
    re.IGNORECASE,
)
HREF = re.compile(r"href=[\"']([^\"'#]+)", re.IGNORECASE)


def _catalogue_paths(root: Path, name: str) -> dict[str, Path]:
    safe = _safe_stem(name)
    manifests = root / "manifests"
    index = root / "index"
    matches = root / "matches"
    images = root / "images"
    for directory in (manifests, index, matches, images):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "manifest": manifests / f"{safe}_images.json",
        "index": index / f"{safe}_preview_index.npz",
        "matches": matches / f"{safe}_clean_matches.json",
        "images": images / "found_train",
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sitemap_pages(url: str, timeout: float) -> list[str]:
    # Some Tilda-hosted sitemaps (e.g. prodcontest.com) return 403 for a
    # non-browser User-Agent on /sitemap.xml specifically, even though the
    # same host serves ordinary pages fine to the plain UA below.
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()
    root = element_tree.fromstring(response.content)
    pages = [element.text for element in root.iter() if element.tag.endswith("loc") and element.text]
    # A sitemap can in principle contain image loc tags too.  Only keep HTML
    # page URLs here; direct image URLs will be extracted below.
    return [page for page in dict.fromkeys(pages) if not re.search(r"\.(?:jpe?g|png|webp)(?:\?|$)", page, re.I)]


def _fetch_page(url: str, timeout: float) -> tuple[str, str]:
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "PAZZLE-source-retrieval/1.0"})
        response.raise_for_status()
        return url, response.text
    except requests.RequestException:
        return url, ""


def crawl_sitemap(root: Path, *, name: str, sitemap: str, workers: int = 8, timeout: float = 30.0) -> list[SourceRecord]:
    """Build an E:-resident manifest from all public image URLs in a sitemap."""
    if workers < 1:
        raise ValueError("workers must be positive")
    pages = _sitemap_pages(sitemap, timeout)
    records: list[SourceRecord] = []
    seen: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_fetch_page, page, timeout) for page in pages]
        for future in as_completed(futures):
            page, html = future.result()
            for candidate in IMAGE_URL.findall(html):
                image_url = urljoin(page, candidate.replace("\\/", "/"))
                # Ignore icons and data-like malformed candidates while keeping
                # all public image CDNs, including static.tildacdn.com.
                if not image_url.startswith("http") or image_url in seen:
                    continue
                seen.add(image_url)
                parsed = urlparse(page)
                slug = _safe_stem(parsed.path.strip("/") or "home")
                records.append(
                    SourceRecord(
                        record_id=len(records),
                        url=image_url,
                        event_slug=slug,
                        event_title=page,
                        event_id=page,
                    )
                )
    paths = _catalogue_paths(root, name)
    _write_json(
        paths["manifest"],
        {
            "source": "public_sitemap_html_images",
            "catalogue": name,
            "sitemap": sitemap,
            "retrieved_unix": time.time(),
            "page_count": len(pages),
            "image_count": len(records),
            "records": [asdict(record) for record in records],
        },
    )
    print(f"saved {name} manifest: pages={len(pages)} images={len(records)} -> {paths['manifest']}", flush=True)
    return records


def crawl_seed_pages(
    root: Path,
    *,
    name: str,
    seeds: list[str],
    max_pages: int = 250,
    workers: int = 8,
    timeout: float = 30.0,
) -> list[SourceRecord]:
    """Crawl a bounded public event catalogue when no usable sitemap exists.

    It stays on the hosts explicitly provided in ``seeds`` and follows only
    same-host HTML links.  This is intentionally a small, transparent crawl
    for historic Central University event pages, not a broad web spider.
    """
    if not seeds or max_pages < 1 or workers < 1:
        raise ValueError("seeds, max_pages and workers must all be non-empty/positive")
    hosts = {urlparse(seed).netloc for seed in seeds}
    pending = list(dict.fromkeys(seeds))
    visited: set[str] = set()
    page_html: list[tuple[str, str]] = []
    while pending and len(visited) < max_pages:
        batch: list[str] = []
        while pending and len(visited) + len(batch) < max_pages and len(batch) < max(1, workers * 3):
            page = pending.pop(0)
            if page not in visited:
                visited.add(page)
                batch.append(page)
        if not batch:
            continue
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for page, html in executor.map(lambda value: _fetch_page(value, timeout), batch):
                if not html:
                    continue
                page_html.append((page, html))
                for href in HREF.findall(html):
                    child = urljoin(page, href.replace("\\/", "/"))
                    parsed = urlparse(child)
                    if (
                        parsed.scheme in {"http", "https"}
                        and parsed.netloc in hosts
                        and not re.search(r"\.(?:jpe?g|png|webp|css|js|pdf|zip)(?:\?|$)", parsed.path, re.I)
                        and child not in visited
                        and child not in pending
                    ):
                        pending.append(child)
    records: list[SourceRecord] = []
    seen_images: set[str] = set()
    for page, html in page_html:
        slug = _safe_stem(urlparse(page).path.strip("/") or "home")
        for candidate in IMAGE_URL.findall(html):
            image_url = urljoin(page, candidate.replace("\\/", "/"))
            if not image_url.startswith("http") or image_url in seen_images:
                continue
            seen_images.add(image_url)
            records.append(
                SourceRecord(
                    record_id=len(records),
                    url=image_url,
                    event_slug=slug,
                    event_title=page,
                    event_id=page,
                )
            )
    paths = _catalogue_paths(root, name)
    _write_json(
        paths["manifest"],
        {
            "source": "bounded_public_seed_page_crawl",
            "catalogue": name,
            "seeds": seeds,
            "page_count": len(page_html),
            "visited_count": len(visited),
            "image_count": len(records),
            "records": [asdict(record) for record in records],
        },
    )
    print(f"saved {name} seed crawl: pages={len(page_html)} images={len(records)} -> {paths['manifest']}", flush=True)
    return records


def _tilda_strings(value: Any) -> Iterator[str]:
    """Yield every textual leaf from a decoded Tilda API response."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tilda_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _tilda_strings(child)


def _fetch_tilda_post(feeduid: str, post: dict[str, Any], timeout: float) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        response = requests.get(
            "https://feeds.tildaapi.com/api/getpost/",
            params={"feeduid": feeduid, "postuid": post["uid"]},
            timeout=timeout,
            headers={"User-Agent": "PAZZLE-source-retrieval/1.0"},
        )
        response.raise_for_status()
        payload = response.json().get("post")
        return post, payload if isinstance(payload, dict) else None
    except (requests.RequestException, ValueError, KeyError):
        return post, None


def crawl_tilda_feed(
    root: Path,
    *,
    name: str,
    feeduid: str,
    workers: int = 8,
    timeout: float = 30.0,
) -> list[SourceRecord]:
    """Enumerate an official finite Tilda news feed without a broad web crawl.

    ``cu.ru/news`` exposes historic articles through this public API while the
    normal sitemap omits much of the dynamically loaded archive.  We fetch the
    finite feed pages and each official post payload, extract direct image URLs
    into an E:-resident manifest, and do not download image bytes here.
    """
    if not feeduid or workers < 1:
        raise ValueError("feeduid and workers must be non-empty/positive")
    posts: list[dict[str, Any]] = []
    slice_number = 1
    while True:
        response = requests.get(
            "https://feeds.tildaapi.com/api/getfeed/",
            params={"feeduid": feeduid, "slice": slice_number},
            timeout=timeout,
            headers={"User-Agent": "PAZZLE-source-retrieval/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        page_posts = [post for post in payload.get("posts", []) if isinstance(post, dict) and post.get("uid")]
        posts.extend(page_posts)
        next_slice = payload.get("nextslice")
        if not next_slice or not page_posts:
            break
        slice_number = int(next_slice)
    unique_posts = list({str(post["uid"]): post for post in posts}.values())
    records: list[SourceRecord] = []
    seen_images: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_fetch_tilda_post, feeduid, post, timeout) for post in unique_posts]
        for completed, future in enumerate(as_completed(futures), start=1):
            summary, post = future.result()
            if post is None:
                continue
            page = str(post.get("url") or summary.get("url") or "https://cu.ru/news")
            uid = str(post.get("uid") or summary["uid"])
            title = str(post.get("title") or summary.get("title") or page)
            for text in _tilda_strings(post):
                for candidate in IMAGE_URL.findall(text):
                    image_url = urljoin(page, candidate.replace("\\/", "/"))
                    if not image_url.startswith("http") or image_url in seen_images:
                        continue
                    seen_images.add(image_url)
                    records.append(
                        SourceRecord(
                            record_id=len(records),
                            url=image_url,
                            event_slug=f"tilda_news_{_safe_stem(uid)}",
                            event_title=title,
                            event_id=uid,
                        )
                    )
            if completed % 25 == 0 or completed == len(unique_posts):
                print(f"{name} Tilda posts {completed}/{len(unique_posts)} images={len(records)}", flush=True)
    paths = _catalogue_paths(root, name)
    _write_json(
        paths["manifest"],
        {
            "source": "official_tilda_news_feed_api",
            "catalogue": name,
            "feeduid": feeduid,
            "post_count": len(unique_posts),
            "image_count": len(records),
            "records": [asdict(record) for record in records],
        },
    )
    print(f"saved {name} Tilda feed manifest: posts={len(unique_posts)} images={len(records)} -> {paths['manifest']}", flush=True)
    return records


def _telegram_manifest_payload(
    *,
    name: str,
    channel: str,
    records: list[SourceRecord],
    pages: int,
    next_before: int | None,
    complete: bool,
    search_query: str | None,
) -> dict[str, Any]:
    return {
        "source": "official_public_telegram_channel",
        "catalogue": name,
        "channel": channel,
        "page_count": pages,
        "next_before": next_before,
        "complete": complete,
        "search_query": search_query,
        "image_count": len(records),
        "records": [asdict(record) for record in records],
    }


def crawl_telegram_channel(
    root: Path,
    *,
    name: str,
    channel: str,
    max_pages: int = 250,
    checkpoint_pages: int = 25,
    timeout: float = 30.0,
    start_before: int | None = None,
    search_query: str | None = None,
) -> list[SourceRecord]:
    """Boundedly enumerate photos from one explicit public Telegram channel.

    This uses Telegram's public ``t.me/s/<channel>`` HTML view only, without a
    login, private API, or browser workaround.  It follows the channel's own
    monotonic ``before`` cursor and stores just an E:-resident URL manifest.
    A partial manifest is checkpointed so an intentionally bounded crawl can
    be resumed later.
    """
    if not channel or max_pages < 1 or checkpoint_pages < 1 or (start_before is not None and start_before < 1):
        raise ValueError("channel, max_pages and checkpoint_pages must be positive/non-empty")
    channel = channel.lstrip("@")
    search_query = search_query.strip() if search_query else None
    paths = _catalogue_paths(root, name)
    records: list[SourceRecord] = []
    seen_images: set[str] = set()
    seen_before: set[int] = set()
    pages = 0
    before: int | None = start_before
    if paths["manifest"].exists():
        try:
            saved = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            if (
                saved.get("source") == "official_public_telegram_channel"
                and saved.get("channel") == channel
                and saved.get("search_query") == search_query
            ):
                records = [SourceRecord(**row) for row in saved.get("records", [])]
                seen_images = {record.url for record in records}
                pages = int(saved.get("page_count", 0))
                before = saved.get("next_before")
                if saved.get("complete"):
                    print(f"reusing complete {name} Telegram manifest ({len(records)} images)", flush=True)
                    return records
                if before is not None:
                    before = int(before)
                print(f"resuming {name} Telegram crawl from before={before}; cached images={len(records)}", flush=True)
        except (OSError, ValueError, KeyError):
            print(f"ignoring unreadable Telegram manifest {paths['manifest']}", flush=True)
            records, seen_images, pages, before = [], set(), 0, start_before
    started = time.monotonic()
    completed = False
    for local_page in range(1, max_pages + 1):
        request_url = f"https://t.me/s/{channel}"
        params: dict[str, Any] = {}
        if before is not None:
            params["before"] = before
        if search_query is not None:
            params["q"] = search_query
        try:
            response = requests.get(
                request_url,
                params=params or None,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; PAZZLE-source-retrieval/1.0)"},
            )
            response.raise_for_status()
        except requests.RequestException as error:
            print(f"{name} Telegram request failed before={before}: {type(error).__name__}", flush=True)
            break
        soup = BeautifulSoup(response.text, "html.parser")
        message_ids: list[int] = []
        for wrapper in soup.select(".tgme_widget_message_wrap"):
            message = wrapper.select_one("[data-post]")
            if message is None:
                continue
            post = str(message.get("data-post", ""))
            match = re.fullmatch(r"([^/]+)/([0-9]+)", post)
            if match is None or match.group(1).lower() != channel.lower():
                continue
            message_id = int(match.group(2))
            message_ids.append(message_id)
            for photo in wrapper.select("a.tgme_widget_message_photo_wrap"):
                style = str(photo.get("style", ""))
                url_match = re.search(r"background-image\s*:\s*url\(['\"]?([^'\")]+)", style, re.IGNORECASE)
                if url_match is None:
                    continue
                image_url = html_module.unescape(url_match.group(1)).replace("\\/", "/")
                if image_url.startswith("//"):
                    image_url = "https:" + image_url
                if not image_url.startswith("http") or image_url in seen_images:
                    continue
                seen_images.add(image_url)
                records.append(
                    SourceRecord(
                        record_id=len(records),
                        url=image_url,
                        event_slug=f"telegram_{_safe_stem(channel)}_{message_id}",
                        event_title=f"https://t.me/{channel}/{message_id}",
                        event_id=str(message_id),
                    )
                )
        if not message_ids:
            completed = True
            break
        next_before = min(message_ids)
        if next_before in seen_before or (before is not None and next_before >= before):
            completed = True
            break
        seen_before.add(next_before)
        before = next_before
        pages += 1
        if local_page % checkpoint_pages == 0 or local_page == max_pages:
            _write_json(
                paths["manifest"],
                _telegram_manifest_payload(
                    name=name,
                    channel=channel,
                    records=records,
                    pages=pages,
                    next_before=before,
                    complete=False,
                    search_query=search_query,
                ),
            )
            print(
                f"{name} Telegram pages +{local_page}/{max_pages}; total_pages={pages} "
                f"images={len(records)} next_before={before} elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    _write_json(
        paths["manifest"],
        _telegram_manifest_payload(
            name=name,
            channel=channel,
            records=records,
            pages=pages,
            next_before=before,
            complete=completed,
            search_query=search_query,
        ),
    )
    print(f"saved {name} Telegram manifest: pages={pages} images={len(records)} complete={completed} -> {paths['manifest']}", flush=True)
    return records


def _load_manifest(root: Path, name: str) -> list[SourceRecord]:
    path = _catalogue_paths(root, name)["manifest"]
    if not path.exists():
        raise FileNotFoundError(f"missing {name} manifest: {path}; run crawl-sitemap first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [SourceRecord(**record) for record in payload["records"]]


def _fetch_preview(record: SourceRecord, timeout: float) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    """Download and reduce one source image without saving it to disk."""
    try:
        response = requests.get(record.url, timeout=timeout, headers={"User-Agent": "PAZZLE-source-retrieval/1.0"})
        response.raise_for_status()
        image = _decode_image(response.content)
        return record.record_id, _phash_bytes(image), _thumbnail(image)
    except (requests.RequestException, ValueError, cv2.error):
        return record.record_id, None, None


def _save_preview_index(
    path: Path,
    records: list[SourceRecord],
    hashes: np.ndarray,
    thumbnails: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Atomically checkpoint compact numerical previews, never image bytes."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            record_ids=np.arange(len(records), dtype=np.int32),
            phashes=hashes,
            thumbs=thumbnails,
            valid=valid,
        )
    temporary.replace(path)


def build_index(root: Path, *, name: str, workers: int = 8, timeout: float = 30.0) -> Path:
    records = _load_manifest(root, name)
    if not records:
        raise RuntimeError(f"{name} manifest has no images")
    paths = _catalogue_paths(root, name)
    index_path = paths["index"]
    partial_path = index_path.with_suffix(".partial.npz")
    hashes = np.zeros((len(records), 8), dtype=np.uint8)
    thumbnails = np.zeros((len(records), 32, 32, 3), dtype=np.uint8)
    valid = np.zeros(len(records), dtype=bool)
    if index_path.exists():
        with np.load(index_path, allow_pickle=False) as prior:
            ids = prior["record_ids"]
            aligned = (
                np.array_equal(ids, np.arange(len(records), dtype=np.int32))
                and prior["phashes"].shape == hashes.shape
                and prior["thumbs"].shape == thumbnails.shape
            )
            if aligned and prior["valid"].astype(bool, copy=False).all():
                print(f"reusing complete index {index_path}", flush=True)
                return index_path
            if aligned:
                hashes[:] = prior["phashes"]
                thumbnails[:] = prior["thumbs"]
                valid[:] = prior["valid"].astype(bool, copy=False)
                print(f"retrying {len(records) - int(valid.sum())} previously unavailable {name} previews", flush=True)
    if partial_path.exists():
        try:
            with np.load(partial_path, allow_pickle=False) as partial:
                if (
                    np.array_equal(partial["record_ids"], np.arange(len(records), dtype=np.int32))
                    and partial["phashes"].shape == hashes.shape
                    and partial["thumbs"].shape == thumbnails.shape
                ):
                    hashes[:] = partial["phashes"]
                    thumbnails[:] = partial["thumbs"]
                    valid[:] = partial["valid"].astype(bool, copy=False)
                    print(f"resuming {name} preview index: {int(valid.sum())}/{len(records)} already cached", flush=True)
        except (OSError, KeyError, ValueError):
            print(f"ignoring unreadable partial index {partial_path}", flush=True)
    pending = [record for record in records if not valid[record.record_id]]
    if not pending:
        _save_preview_index(index_path, records, hashes, thumbnails, valid)
        partial_path.unlink(missing_ok=True)
        print(f"saved {name} feature index -> {index_path}", flush=True)
        return index_path
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_fetch_preview, record, timeout) for record in pending]
            for completed, future in enumerate(as_completed(futures), start=1):
                record_id, digest, thumbnail = future.result()
                if digest is not None and thumbnail is not None:
                    hashes[record_id] = digest
                    thumbnails[record_id] = thumbnail
                    valid[record_id] = True
                if completed % 100 == 0 or completed == len(pending):
                    _save_preview_index(partial_path, records, hashes, thumbnails, valid)
                    print(
                        f"{name} previews {completed}/{len(pending)} pending; valid={int(valid.sum())}/{len(records)} "
                        f"elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
    finally:
        _save_preview_index(partial_path, records, hashes, thumbnails, valid)
    _save_preview_index(index_path, records, hashes, thumbnails, valid)
    partial_path.unlink(missing_ok=True)
    print(f"saved {name} feature index -> {index_path}", flush=True)
    return index_path


def _load_index(root: Path, name: str) -> tuple[list[SourceRecord], np.ndarray, np.ndarray, np.ndarray]:
    records = _load_manifest(root, name)
    path = _catalogue_paths(root, name)["index"]
    if not path.exists():
        raise FileNotFoundError(f"missing {name} index: {path}; run index first")
    with np.load(path, allow_pickle=False) as index:
        ids = index["record_ids"].astype(np.int64, copy=False)
        hashes = index["phashes"].astype(np.uint8, copy=False)
        thumbnails = index["thumbs"].astype(np.uint8, copy=False)
        valid = index["valid"].astype(bool, copy=False)
    if not np.array_equal(ids, np.arange(len(records), dtype=np.int64)):
        raise RuntimeError("index records do not align with its manifest")
    return records, hashes, thumbnails, valid


def match_clean(
    root: Path,
    *,
    name: str,
    target_dir: Path,
    max_hamming: int = 8,
    top_k: int = 3,
    verify: bool = True,
    timeout: float = 30.0,
) -> Path:
    """Hash-retrieve, then SIFT-verify exact train matches for one catalogue."""
    records, hashes, thumbnails, valid = _load_index(root, name)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        raise RuntimeError(f"{name} index contains no valid images")
    targets = sorted(target_dir.glob("*.png"))
    paths = _catalogue_paths(root, name)
    paths["images"].mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for number, target_path in enumerate(targets, start=1):
        target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if target is None:
            continue
        distances = hamming_distance(_phash_bytes(target), hashes[valid_ids])
        ranking = np.argsort(distances, kind="stable")[:top_k]
        target_thumbnail = _thumbnail(target)
        candidates: list[dict[str, Any]] = []
        for rank in ranking:
            record_id = int(valid_ids[rank])
            candidates.append(
                {
                    "record_id": record_id,
                    "hamming": int(distances[rank]),
                    "preview_rmse": float(
                        np.sqrt(np.mean(np.square(thumbnails[record_id].astype(np.float32) - target_thumbnail.astype(np.float32))))
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
                accepted = metrics["sift_inliers"] >= 60.0 and (
                    metrics["pixel_correlation"] >= 0.95 or candidate["preview_rmse"] <= 5.0
                )
                candidate["verification"] = metrics
                candidate["accepted"] = accepted
                if accepted:
                    suffix = Path(record.url).suffix.lower() or ".jpg"
                    output = paths["images"] / f"{target_path.stem}__{_safe_stem(name)}__{record.record_id}{suffix}"
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
        rows.append(row)
        if number % 250 == 0 or number == len(targets):
            print(f"{name} match {number}/{len(targets)} accepted={sum(r['accepted'] is not None for r in rows)}", flush=True)
    report = {
        "source": name,
        "target_dir": str(target_dir),
        "accepted": sum(row["accepted"] is not None for row in rows),
        "max_hamming": max_hamming,
        "rows": rows,
    }
    _write_json(paths["matches"], report)
    print(f"saved {name} report -> {paths['matches']}; accepted={report['accepted']}", flush=True)
    return paths["matches"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    subcommands = parser.add_subparsers(dest="command", required=True)
    crawl = subcommands.add_parser("crawl-sitemap")
    crawl.add_argument("--name", required=True)
    crawl.add_argument("--sitemap", required=True)
    crawl.add_argument("--workers", type=int, default=12)
    crawl.add_argument("--timeout", type=float, default=30.0)
    seed = subcommands.add_parser("crawl-seeds")
    seed.add_argument("--name", required=True)
    seed.add_argument("--seed", action="append", required=True, help="public event-catalogue page; may be supplied more than once")
    seed.add_argument("--max-pages", type=int, default=250)
    seed.add_argument("--workers", type=int, default=12)
    seed.add_argument("--timeout", type=float, default=30.0)
    feed = subcommands.add_parser("crawl-tilda-feed")
    feed.add_argument("--name", required=True)
    feed.add_argument("--feeduid", required=True)
    feed.add_argument("--workers", type=int, default=12)
    feed.add_argument("--timeout", type=float, default=30.0)
    telegram = subcommands.add_parser("crawl-telegram")
    telegram.add_argument("--name", required=True)
    telegram.add_argument("--channel", required=True, help="explicit public t.me channel name, without @")
    telegram.add_argument("--max-pages", type=int, default=250)
    telegram.add_argument("--checkpoint-pages", type=int, default=25)
    telegram.add_argument("--timeout", type=float, default=30.0)
    telegram.add_argument("--start-before", type=int, default=None, help="optional public Telegram before cursor for a bounded historical crawl")
    telegram.add_argument("--query", dest="search_query", default=None, help="optional public Telegram channel search query")
    index = subcommands.add_parser("index")
    index.add_argument("--name", required=True)
    index.add_argument("--workers", type=int, default=12)
    index.add_argument("--timeout", type=float, default=30.0)
    match = subcommands.add_parser("match-clean")
    match.add_argument("--name", required=True)
    match.add_argument("--target-dir", type=Path, default=TARGET_ROOT)
    match.add_argument("--max-hamming", type=int, default=8)
    match.add_argument("--top-k", type=int, default=3)
    match.add_argument("--no-verify", action="store_true")
    match.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    if args.command == "crawl-sitemap":
        crawl_sitemap(root, name=args.name, sitemap=args.sitemap, workers=args.workers, timeout=args.timeout)
    elif args.command == "crawl-seeds":
        crawl_seed_pages(
            root,
            name=args.name,
            seeds=args.seed,
            max_pages=args.max_pages,
            workers=args.workers,
            timeout=args.timeout,
        )
    elif args.command == "crawl-tilda-feed":
        crawl_tilda_feed(root, name=args.name, feeduid=args.feeduid, workers=args.workers, timeout=args.timeout)
    elif args.command == "crawl-telegram":
        crawl_telegram_channel(
            root,
            name=args.name,
            channel=args.channel,
            max_pages=args.max_pages,
            checkpoint_pages=args.checkpoint_pages,
            timeout=args.timeout,
            start_before=args.start_before,
            search_query=args.search_query,
        )
    elif args.command == "index":
        build_index(root, name=args.name, workers=args.workers, timeout=args.timeout)
    elif args.command == "match-clean":
        match_clean(
            root,
            name=args.name,
            target_dir=args.target_dir,
            max_hamming=args.max_hamming,
            top_k=args.top_k,
            verify=not args.no_verify,
            timeout=args.timeout,
        )
    else:
        raise AssertionError(f"unexpected command {args.command!r}")


if __name__ == "__main__":
    main()
