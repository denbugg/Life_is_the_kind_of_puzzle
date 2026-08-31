# SocketMatcher + retained k5 edge-ranker: fixed rank fusion

## Verdict

The old pairwise ranker contains a **small complementary top-1 signal**, but a
50/50 rank fusion is not strong enough to replace the SocketMatcher control.
On one fixed, source-fresh train-24 panel the fusion improved all three global
means versus SocketMatcher decoder144:

- direct placement: `0.1013% -> 0.1302%` (`14 -> 18` recovered-reference
  exact tiles over 24 boards);
- adjacency: `7.7408% -> 8.1069%` (`+0.3661 pp`);
- raw SSIM: `0.108348 -> 0.108756` (`+0.000408`).

These gains are too small for promotion. A 20,000-draw paired board bootstrap
with fixed seed `20260830` gives intervals that all cross zero: direct delta
95% CI `[-0.0579,+0.1157] pp`, adjacency
`[-0.0189,+0.7473] pp`, raw SSIM `[-0.002205,+0.002936]`. Fusion also reduced
SocketMatcher's wider neighbour recall: pooled R@5 `26.50% -> 24.24%`, R@16
`42.80% -> 38.84%`, R@32 `54.58% -> 50.15%`. Do not tune a weight on this now
opened panel and do not treat this arm as submission-ready.

## Why the k5 checkpoint was reused

Two ranker checkpoints were available. The broader k16/train256 model had the
highest historical local R@1, but failed its original end-to-end gate and was
explicitly closed for integration. The retained raw-k5/train64 checkpoint
passed its local gate, reproduced its geometry gain, and remained documented
as a reusable layout auxiliary. This experiment therefore reused exactly:

- SocketMatcher d32 v2 SHA-256
  `7ccb14042e50432bf450018d4ebb32b78866d3755d8387cb1534f67155fd1c19`;
- raw-k5 edge-ranker SHA-256
  `d18ff864c63170d5fcdb868d672a60515d10ac600afa2ed0424000921ecbb21a`.

All semantic hashes declared by the ranker contract matched the current
source. The ranker still provides full `576 x 576` scores: untouched pairs use
its frozen bilateral base and only its deterministic raw/tile-z/bilateral/gray
union top-5 receives neural residuals. No missing score was invented.

## Frozen protocol

The panel consists of 24 manifest-train records at SocketMatcher selector ranks
`2560:2586`, excluding preregistered ranks `2573` and `2583`. Rank 2573 belongs
to an available ranker training lineage; rank 2583 had already appeared in a
repository experiment report. Before target access, the runner verified zero
overlap with both selected checkpoint lineages and every prior report in
`outputs/`.

The fixed roster had exactly three arms:

1. original SocketMatcher partial-OT assignment + decoder144;
2. k5 ranker's dense scores, inverse-normal row-rank calibration, analytic
   unmatched-socket logits, exact-capacity partial OT + decoder144;
3. 50/50 inverse-normal row-rank fusion, 50/50 socket/ranker border evidence,
   exact-capacity partial OT + the same decoder144.

There was no learned or target-calibrated fusion weight. Every decoder used
144 component edges per axis, 144 swap edges per axis, and at most 24 exact
delta swaps. All 72 layouts were strict permutations of the original 576 dirty
tiles. The complete layouts and assembled-image hashes were persisted before
the first clean target was opened.

Prediction commitment:
`outputs/socket-ranker-fusion/fresh-train24-ranks2560-2585-k5/prediction-commitment.json`,
SHA-256 `dac1cb90b9d709738aaab48044ed27f26ae7af8342e3887d6bd74a9a6839fb89`.

Authoritative report:
`outputs/socket-ranker-fusion/fresh-train24-ranks2560-2585-k5/report.json`,
SHA-256 `0a0ffda7aa87ba648cea21629e357df8cdf712ebf03c23d0002c7b5bb172c348`.

## Results

| Fixed decoder144 arm | Exact tiles / 24 | Direct | Translation-aligned | Adjacency | Raw SSIM |
|---|---:|---:|---:|---:|---:|
| SocketMatcher | 14 | 0.1013% | 1.3383% | 7.7408% | 0.108348 |
| k5 ranker | 11 | 0.0796% | 1.1936% | 5.7669% | **0.110832** |
| Equal-rank fusion | **18** | **0.1302%** | **1.3961%** | **8.1069%** | 0.108756 |

| Assignment real block | Pooled R@1 | R@5 | R@16 | R@32 |
|---|---:|---:|---:|---:|
| SocketMatcher | 11.062% | **26.502%** | **42.799%** | **54.578%** |
| k5 ranker | 9.281% | 20.188% | 30.699% | 40.025% |
| Equal-rank fusion | **11.092%** | 24.238% | 38.836% | 50.155% |

The fusion's R@1 edge gain is only `+0.030 pp`, while higher-K coverage falls.
That pattern and the sparse exact counts explain the cautious verdict: the
ranker can break a few SocketMatcher top-1 ties, but equal global weighting
dilutes the stronger socket candidate distribution.

## Reproduction

```bash
.venv/bin/python scripts/evaluate_socket_ranker_fusion.py --device cpu
```

Runtime was 277.7 seconds for dirty-only freeze and 3.1 seconds for
target-assisted diagnostics. Tests:

```bash
.venv/bin/ruff check src/aiijc_puzzle/socket_ranker_fusion.py \
  scripts/evaluate_socket_ranker_fusion.py tests/test_socket_ranker_fusion.py
.venv/bin/pytest -q tests/test_socket_ranker_fusion.py \
  tests/test_socket_decoder.py tests/test_socket_matcher.py
```

The final source passed Ruff and 23 related tests. The only warnings were PyTorch's
existing nested-tensor advisory for norm-first Transformer layers.
