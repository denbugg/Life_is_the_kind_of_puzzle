"""Crawl a public Telegram channel's web preview for PAZZLE source retrieval.

``t.me/s/<channel>`` is Telegram's official unauthenticated preview used for
embedding/SEO: it server-renders recent posts (including inline photo URLs)
without any API token or login. Pagination walks backward in time via
``?before=<message_id>``. This produces a manifest in the exact same
``SourceRecord`` schema as :mod:`static_source_retrieval`, so the existing
``index`` / ``match-clean`` (train) and :mod:`static_bag_source_retrieval`
``index`` / ``rank-test`` / ``verify-test`` (test) commands work unchanged.

Only a channel's own recent-to-oldest post history is reachable this way; it
cannot search across Telegram or read private/restricted channels.

Example::

    python src/telegram_source_retrieval.py crawl --name kod_zheltyi_telegram \
        --channel kod_zheltyi --max-messages 3000
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests

from source_retrieval import DEFAULT_ROOT, SourceRecord, _safe_stem


PHOTO_URL = re.compile(r"background-image:url\('([^']+)'\)")
POST_ID = re.compile(r'data-post="[^"]+/(\d+)"')
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _catalogue_paths(root: Path, name: str) -> dict[str, Path]:
    safe = _safe_stem(name)
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    return {"manifest": manifests / f"{safe}_images.json"}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fetch_page(channel: str, before: int | None, timeout: float) -> str:
    url = f"https://t.me/s/{channel}"
    params = {"before": before} if before else None
    response = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.text


def crawl_channel(
    root: Path,
    *,
    name: str,
    channel: str,
    max_messages: int = 3000,
    max_pages: int = 60,
    timeout: float = 20.0,
    sleep_seconds: float = 0.4,
) -> list[SourceRecord]:
    """Walk one public channel's preview backward in time; no login involved."""
    if max_messages < 1 or max_pages < 1:
        raise ValueError("max_messages and max_pages must be positive")
    seen_urls: set[str] = set()
    records: list[SourceRecord] = []
    before: int | None = None
    oldest_seen = None
    for page in range(max_pages):
        html = _fetch_page(channel, before, timeout)
        photo_urls = PHOTO_URL.findall(html)
        post_ids = [int(value) for value in POST_ID.findall(html)]
        new_count = 0
        for url in photo_urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            new_count += 1
            records.append(
                SourceRecord(
                    record_id=len(records),
                    url=url,
                    event_slug=f"telegram_{_safe_stem(channel)}",
                    event_title=f"t.me/{channel}",
                    event_id=channel,
                )
            )
        print(
            f"{name}: page {page + 1} before={before} posts_seen={len(post_ids)} "
            f"new_images={new_count} total_images={len(records)}",
            flush=True,
        )
        if not post_ids or len(records) >= max_messages:
            break
        page_oldest = min(post_ids)
        if oldest_seen is not None and page_oldest >= oldest_seen:
            break  # the channel stopped returning older pages; avoid an infinite loop
        oldest_seen = page_oldest
        before = page_oldest
        time.sleep(sleep_seconds)

    paths = _catalogue_paths(root, name)
    _write_json(
        paths["manifest"],
        {
            "source": "public_telegram_channel_preview",
            "catalogue": name,
            "channel": channel,
            "pages_fetched": page + 1,
            "image_count": len(records),
            "records": [asdict(record) for record in records],
        },
    )
    print(f"saved {name} manifest: channel=t.me/{channel} images={len(records)} -> {paths['manifest']}", flush=True)
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    crawl = sub.add_parser("crawl")
    crawl.add_argument("--name", required=True, help="catalogue tag used for the saved manifest")
    crawl.add_argument("--channel", required=True, help="public Telegram channel username, no @")
    crawl.add_argument("--max-messages", "--max_messages", dest="max_messages", type=int, default=3000)
    crawl.add_argument("--max-pages", "--max_pages", dest="max_pages", type=int, default=60)
    crawl.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    if args.command == "crawl":
        crawl_channel(
            root, name=args.name, channel=args.channel,
            max_messages=args.max_messages, max_pages=args.max_pages, timeout=args.timeout,
        )
    else:
        raise AssertionError(f"unexpected command {args.command!r}")


if __name__ == "__main__":
    main()
