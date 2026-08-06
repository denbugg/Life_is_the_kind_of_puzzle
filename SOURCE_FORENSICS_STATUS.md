# Public-source forensics: current status

Last updated: 2026-07-12 (grand merge + official-site re-check).

## Grand merge across every crawled catalogue (2026-07-12)

Between the last update and now, a much larger set of catalogues was crawled
(likely a separate session): ~24 Telegram channels (`t.me/s/<channel>`
public preview, no login) plus the earlier CU/T-Education/PROD static
catalogues. Built one merged bag-fingerprint index
(`static_bag_central_static_grand`, **19,679 valid photos** out of 23,846
URLs after dedup) reusing every previously computed fingerprint, then ran
`rank-test --top-k 20` + `verify-test --top-n 3` against **all 700 test
images**. Calibration transfer held on the bigger pool: train R@1=72.7%,
R@5=90.9%, R@50=100% (22 verified rows).

**Result: exactly the same 18 test images as before -- zero new hits.**
Every test image's true source (where one exists in this pool) was already
found by earlier targeted crawls; the wider net did not surface anything new
in the already-known catalogues. `src/telegram_source_retrieval.py` is now a
reusable, saved crawler for `t.me/s/<channel>` (previously ad-hoc/unsaved).

**Explicitly checked and ruled out as further official-site leads**:
`tbank.ru/career/blog/` (generic HR marketing articles, no event photos),
`my.centraluniversity.ru/events/*` (lightweight course sign-up pages, no
images, likely gated for real content), `it-picnic.ru` (third-party job-fair
landing page, not T-Bank's own event photography). VK (`vk.com/teducation`,
`vk.com/central_university`) is **not scrapable without either a VK API
token or a headless browser** -- its wall/photos are rendered client-side by
JS even on `m.vk.com`; plain HTTP only exposes the profile card.

**Assessment**: the known-catalogue avenue (official CU/T-Bank/PROD/
T-Education pages + their public Telegram channels) appears exhausted at
18/700. Any further growth needs either a genuinely new, not-yet-discovered
channel/subsite, or VK access (needs the user to supply an API token, or a
headless-browser tool this environment does not have).

## New lead found by the user: professional photographer galleries (wfolio.pro)

The user personally attended a CU event ("сборы по ИИ и ИБ", 28.02-06.03) and
pointed to its professional photo gallery:
`anastas2017klim.wfolio.pro/disk/28-02-2026-sbory-po-ii-i-ib-28-02-06-03-0ptlxf/`.
This is a **materially different, likely much richer class of source** than
official landing pages or Telegram: a hired photographer's full session
coverage, organized into 7 per-day subfolders (`28-02`, `1-03`...`6-03`) plus
`favorites`.

**Blocked by client-side rendering, not by lack of trying.** The gallery grid
is a JS single-page app (Rails/Turbo + Stimulus, controller name literally
`protector`) -- the raw HTML has no photo list, only two generic site-wide
share-preview thumbnails (same 2 images on every one of the 8 subfolder
URLs). No public API/bulk-download URL pattern is documented
(`docs.wfolio.pro` confirms this is intentionally unexposed). The one
recoverable photo (crowd/olympiad shot, visually plausible for this dataset)
does **not** match any train target or test input (checked: pHash+SIFT
against all 7000 train, bag-fingerprint rank+verify against all 700 test --
0 hits).

**This needs the user's help to unblock**, one of:

1. Log into the gallery (as an attendee) and download the album as a ZIP via
   the site's own "download" button, then share the files (e.g. drop them
   under `E:/pazzle_work/source_forensics/manual_drops/<event-name>/` and
   tell me) -- fastest, no new tooling needed on my end.
2. Point me at other events with a similar photographer-gallery link if
   remembered -- worth trying even without solving the JS-rendering problem,
   since some photographer platforms *do* server-render (unlike wfolio.pro).
3. If a headless-browser tool becomes available, this gallery (and similarly
   structured ones) becomes crawlable properly.

Do not re-attempt HTTP-only scraping of wfolio.pro-hosted galleries; the
platform is confirmed JS-gated end to end.

## Source expansion pass (2026-07-11 evening)

Traced where images currently come from (`src/source_retrieval.py` = T-Bank
meetup API, fully crawled: 5,058 images / 894 events, no further growth
possible there) and `src/static_source_retrieval.py` = sitemap/seed-crawl/
Tilda-feed for everything else. Found and crawled three new catalogues:

| catalogue | how found | pages | images | train hits | test hits |
|---|---|---:|---:|---:|---:|
| `central_events_v2` | `event.cu.ru` was only seeded with ~12 slugs; discovered a second independent subdomain `event.centraluniversity.ru` plus ~25 more slugs via `site:event.cu.ru` search (incl. the DEADLINE case championship: `event.cu.ru/casecontest`, `/bachelor-casecontest`) | 31 | 527 | **+3** | +0 |
| `prod_olympiad` | official PROD olympiad site is `prodcontest.com` (Tilda-hosted, org: CU + Т-Технологии) | 18 | 56 | 0 | 0 |
| `t_education_events` | `education.tbank.ru/activities/events/` — a per-event namespace not covered by the sitemap crawl | 123 | 573 | 0 | 0 |

All three merged into one bag-retrieval index `central_static_all` (2,441
valid fingerprints) alongside the previously indexed catalogues, reusing
already-fetched fingerprints by URL (`--reuse-tag`). T-Bank-trained transfer
calibration held on the expanded truth set: **R@1 90.9%, R@5 100% (22
rows)**. Full `--top-n 1` verification across all 700 test images against
this merged catalogue found exactly the same **2** CU test matches already
known (`img_000840`, `img_002948`) — no new test hits from the 3 new
catalogues. `prod_olympiad` is a small marketing site (18 pages) with almost
no event photography; `t_education_events` lists mostly future 2026
excursions, likely not yet photographed/published.

**Net effect: train exact matches 161 -> 164. Test exact matches unchanged
at 14/700.** A small code fix was needed: `static_source_retrieval.py`'s
sitemap fetch now uses a browser-like User-Agent (`prodcontest.com` 403'd a
plain UA on `/sitemap.xml` specifically, while ordinary pages worked fine).

**Next lever, not yet tried**: event landing pages only embed a handful of
curated photos each. VK (`vk.com/teducation`) and Telegram
(`t.me/s/casecontest`, `t.me/kod_zheltyi`) public channels/albums likely hold
far larger raw photo dumps from the same events and are the most promising
remaining source of new test matches.

## Result already usable

The retrieval path now has a high-precision output layer for test images:

- 14 test frames are independently verified source matches: 12 from the
  historic T-Bank meetup archive and 2 from Central University sources.
- Their clean 480x480 centre crops are materialized under
  `E:/pazzle_work/source_forensics/overrides/verified_source_clean/` using the
  original test filenames.
- The handoff manifest is
  `E:/pazzle_work/source_forensics/matches/verified_test_source_overrides.json`.
  A generic submission should copy those 14 PNGs over its matching base
  predictions; never replace a frame merely because it has a good bag rank.

## Evidence / gates

| Check | Result |
|---|---:|
| Exact public source matches in train | 161 unique targets |
| T-Bank event-held-out source retrieval | R@1 94.2%, R@5 98.6%, R@50 100% (139 rows) |
| Independent CU/static transfer | R@1 90.9%, R@5 100% (22 rows) |
| Train target SSIM of all source centre-crops | mean 0.9536, median 0.9571 (161 targets) |
| Test acceptance condition | 10x10 global tile assignment + at least 5 spatially aligned SIFT matches, identity fraction >= 0.35 |

The last gate is important: an unordered tile bag can rank a visually similar
photo highly.  It is not accepted until the reconstructed dirty image and the
candidate source have geometric, same-coordinate evidence.

## Source coverage added

- T-Bank: full public historic meetup API, 5,058 images in the manifest.
- Central University: current sitemap, event.cu.ru historic pages, and the
  official finite `cu.ru/news` Tilda feed (146 posts / 476 image URLs).
- The news feed added 13 previously unseen exact train targets, including PROD,
  olympiad, and CU/T-Education event photography.
- T-Education: the explicit public `@tbank_education` archive was enumerated
  through its public HTML view (179 pages / 3,631 photo URLs).  It confirms a
  train source with 591 SIFT inliers, but that frame is a cross-post already
  covered by a CU source, so it adds no new unique target yet.

All manifests, numeric indexes, reports, and accepted originals are on `E:`.
Unverified photos are streamed in memory only.

## Reproducible commands

```powershell
# T-Bank: build/validate/rank/verify
python src/bag_source_retrieval.py --root E:\pazzle_work\source_forensics calibrate-tbank
python src/bag_source_retrieval.py --root E:\pazzle_work\source_forensics rank-test --top-k 20
python src/bag_source_retrieval.py --root E:\pazzle_work\source_forensics verify-test --top-n 1

# CU static catalogue: rank/verify after its index exists
python src/static_bag_source_retrieval.py --root E:\pazzle_work\source_forensics rank-test --tag central_static_news --top-k 5
python src/static_bag_source_retrieval.py --root E:\pazzle_work\source_forensics verify-test --tag central_static_news --top-n 1

# Recreate the submission-overlay layer
python src/verified_source_overrides.py --root E:\pazzle_work\source_forensics
```

## Next decision

Use the 14 exact overrides in the final 700-image submission, on top of the
best available generic puzzle solver output.  Keep source discovery focused on
bounded official archives; do not weaken the spatial verification threshold to
inflate coverage.

## Current addendum — 2026-07-12

The preceding counters are historical. The current source-retrieval state is:

- **218 unique train targets** with an exact public source: 139 from the
  historic T-Bank meetup archive and 79 from bounded official static/Telegram
  catalogues.
- **18 unique verified test overrides**. The four additions after the original
  14 are `img_001786` (Central University main channel), `img_002198` (PROD),
  `img_001737` (CU bachelor channel), and `img_002775` (CU master channel).
- The current handoff is
  `E:/pazzle_work/source_forensics/submissions/source_overrides_raw_input.zip`:
  700 unique PNG files, with the 18 verified clean crops replacing the same
  filenames and raw shuffled input retained for the remaining files.
- The ZIP and manifest were checked after materialization: 700 PNG entries,
  700 unique names, 18 override rows.

New high-value catalogues were verified by the same exact-train gate before
any test search: `t_prod`, `centraluniversity_bachelor`,
`centraluniversity_master`, and `vokrug_CU`. The master channel produced the
new test frame `img_002775`; `vokrug_CU` had train matches but no accepted test
match. A number of other official channels were screened and intentionally
not carried into test because they had zero exact train matches.

For static-catalogue retrieval, ranking is only a candidate generator. A
source is accepted only after 10x10 Hungarian tile assignment plus spatially
aligned SIFT evidence (at least 5 aligned keypoints and identity fraction
>= 0.35). Originals and 480x480 clean crops are persisted only after that
gate, under `E:`.
