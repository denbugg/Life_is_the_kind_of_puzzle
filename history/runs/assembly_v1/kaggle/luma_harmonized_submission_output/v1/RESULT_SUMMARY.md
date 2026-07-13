# Luma-harmonized submission v1

## Decision

Promoted. The bounded luminance gain was frozen after development calibration,
passed a source-disjoint 32-image confirmation on both corruption panels, and
was rendered over the existing frozen 700-image QAP submission layouts.

## Leakage-safe evidence

- Calibration: 16 whole sources from `assembly_cal[32:48]`, 32 records.
- Calibration source-macro SSIM delta: `+0.0014314713394617206`.
- Calibration paired bootstrap 95% CI: `[+0.0011056400062492845, +0.0017491168549248098]`.
- Confirmation: 32 whole sources from `assembly_incremental_gate[0:32]`, 64 records.
- Confirmation source-macro SSIM delta: `+0.0017219010426128445`.
- Confirmation paired bootstrap 95% CI: `[+0.0014445107715963434, +0.0020091469002556464]`.
- Confirmation wins: `32/32` sources and `32/32` records on each panel.
- Target-referenced seam error improved on both panels.
- Calibration report SHA-256: `9593b8809d2b0e6a0f928e7b0cf41e47f4406e5dd3d8151ea6b75a521c65bbee`.
- Confirmation report SHA-256: `aeac52ebdde35581a974ac863ccb9f7af22f4c4153b667ccb81f68c884b158c1`.

## Submission artifact

- Archive: `submission.zip`.
- SHA-256: `099d1c5fe69cda8519a4f19750cb3a481ac87999c294a35e19691a849d4c6096`.
- Bytes: `206206965`.
- Members: `700` flat RGB PNG names.
- Shards: `350 + 350` rendered concurrently on two Tesla T4 GPUs.
- Preflight replay: byte-identical to the first full shard image.
- Frozen method SHA-256: `f77c000f9afc52edfed236a021b5c2e0366d049199a59026fdab43075cd6f121`.
- Kaggle run marked `safe_for_submission=true`.

The archive has not yet been leaderboard-scored.
