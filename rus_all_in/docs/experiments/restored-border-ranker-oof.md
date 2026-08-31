# E20 restored BorderRanker: source-disjoint bounded resolution

## Verdict

**Keep the restored-descriptor union as candidate-supply evidence; reject the
tested residual BorderRanker.** On 16 source-disjoint exact-synthetic 24×24
boards the restored top-32 emitter added `+2.9778 pp` right and `+2.7627 pp`
down coverage to frozen d64 top-32. The learned cross-ranker did not convert
that supply into better ordering:

| Local exact metric, 17,664 directed edges | frozen d64 OT | restored cross-ranker | delta |
|---|---:|---:|---:|
| pooled R@1 | `17.9178%` | `17.8442%` | `−0.0736 pp` |
| pooled R@5 | `35.8016%` | `35.8356%` | `+0.0340 pp` |

Native reciprocal precision rose from `30.0268%` to `33.9055%`, but only while
coverage contracted from `46.5127%` to `33.3107%`. At exactly matched
`33.3107%` coverage, d64 reached `37.1516%`, so the candidate lost
`−3.2461 pp`. The predeclared local gate failed. No global decoder, layout,
calibration, holdout, competition test or production integration was opened.

## Historical E20 audit

The historical source was inspected directly at commit `a877065944` in the
fetched `pazzle_will_be_killed` repository. Its intended path was raw E14
top-32 union restored-descriptor top-32, followed by a restored BorderRanker
and a fixed `.25` robust-z bonus. Only the candidate-coverage prerequisite was
actually executed: `+5.0951 pp` right and `+4.6535 pp` down, with the latter
missing the fixed `+5 pp` gate. The ranker was never called and no layout was
produced.

The exact historical run cannot be reproduced from the fetched repository or
current workspace because all three ignored binary artifacts are absent:

| Missing artifact | Expected SHA-256 |
|---|---|
| `real_fragment_restorer_best.pt` | `6fcc7de2cf8063b4f2f45d4b96b8999d5eb9c29a071ff2c0031d2703c70d6695` |
| `restored_border_ranker_best.pt` | `8eb7b7e106c0333b9a099f88894eac7b1081555643d3828e479aaf4e56137be1` |
| 69,387,927-byte restored sidecar | `65c04742aeaa1fb51934fd70951052a46443f09dd60c798b484f66aca29e5cab` |

The committed historical metrics report candidate-32 validation R@1/R@5
`38.6836/68.8516%`, but the original corpus manifest is absent, source overlap
cannot be disproved, and these shortlist metrics are not comparable to
all-576 d64 retrieval.

## Materially new legal substitute

The bounded local runner avoids the missing and provenance-ambiguous binaries:

1. independently corrupt and shuffle clean manifest-train targets with a known
   exact permutation;
2. obtain raw top-32 from frozen d64 partial OT;
3. run official externally pretrained colour DRUNet at sigma 40 independently
   on each upright 20×20 tile;
4. add top-32 nearest normalised restored border descriptors;
5. train a 73,889-parameter seven-channel seam CNN plus eight raw/restored rank
   features to predict a residual over the d64 score;
6. evaluate only the final checkpoint on 16 different source-held-out boards.

This differs from the already rejected DRUNet and DualNAF matcher experiments.
Those used a direct restored bilateral score or fixed 50/50 averaging. Here the
restored view expands candidate supply and a learned cross-ranker sees both the
restored seam and raw/restored rank agreement. The last layer starts at zero,
so the untrained model exactly preserves d64 ordering.

Restored pixels are never output. The frozen prediction artifact contains only
candidate identities, union membership and reciprocal evidence. Any later
decoder would have to assemble every original dirty tile exactly once, but the
failed local gate forbids that follow-up here.

### DRUNet small-tile limitation

The official colour DRUNet is not geometrically well matched to an independent
20×20 fragment. This runner reflect-pads each tile only to 24×24, while the
network has three stride-2 encoder stages, so its spatial pyramid is
`24→12→6→3`. Full-resolution and intermediate skip connections mean the tile
does not literally collapse to a 3×3 representation, but the deepest context
used to decide denoising is only 3×3 and exact boundary phase can be smoothed or
aliased. The measured pattern is consistent with that limitation: the restored
descriptor finds extra top-32 candidates, yet neither top-1 ordering nor
matched-coverage precision improves.

Therefore this result rejects the tested deep downsampling U-Net view, not all
denoise-before-match approaches. A materially new follow-up may use a shallow
full-resolution residual/NAF network or explicit border-strip model with no
spatial downsampling, trained for neighbour retrieval/boundary preservation
rather than another generic clean-pixel objective. Raw d64 evidence must remain
an explicit parallel view and the same local gate must precede any decoder.

## Frozen protocol and gate

- Training: 256 manifest-`train` sources, 400 dynamic exact-synthetic updates,
  32 eligible union rows per update.
- Evaluation: 16 different sources × one exact synthetic draw.
- Full d64 checkpoint lineage and declared prior exact/model reports excluded:
  3,000 unique filenames.
- Train/eval source digests:
  `1eedd50927b6032b1f3cdaec84d9cfe11422a0feee1f161b0e2fbff1d557ebfd` /
  `c637b982ef575dfcead86d6a40ca88c286ba03de8d3a4b2ef32eacbaf7f15e36`.
- Supply gate: right and down top-32 union coverage must each improve by at
  least `+3 pp`.
- Ranking gate: pooled R@1 must improve by at least `+0.5 pp` and R@5 must not
  regress.
- Alternative precision gate: matched reciprocal precision `+5 pp` at at least
  `3%` coverage.
- Decoder eligibility required `supply AND (ranking OR precision)`.

Supply narrowly missed its threshold in both directions; ranking and matched
precision also failed materially. Thresholds are not relaxed post hoc.

## Resource result

The one-update/full-eval device smoke justified MPS rather than artificial
loading:

| Work | CPU | MPS | speed-up |
|---|---:|---:|---:|
| one complete training update | `5.434 s` | `1.755 s` | `3.10×` |
| 68,178-pair full cross-rank eval | `18.202 s` | `0.954 s` | `19.08×` |
| DRUNet per board | `3.854 s` | `0.266 s` | `14.51×` |

The substantive MPS run took `229.756 s`, or `0.5744 s/update`. Mean eval-board
components were d64 `0.0454 s`, DRUNet `0.2616 s`, CPU descriptor/union supply
`0.1437 s`, and cross-ranker `0.7903 s`. MPS backward emitted the recorded
non-deterministic indexed-update warning; source selection, corruptions and
shuffle seeds remain fixed, but bitwise checkpoint reproducibility is not
claimed.

## Artifacts

- authoritative report:
  `outputs/restored-border-ranker/pilot-train256-s400-eval16-mps/report.json`,
  SHA-256 `5b58ca446a541b3a33fcda7243cee7bec7b4cba68e463889b6f3f0d1f37e47be`;
- rejected checkpoint: `restored_border_ranker.pt`, SHA-256
  `f828d632afa3a4fabf887b84eec947e8d29c0129699e106abbb811e71ca81cbd`;
- frozen candidate artifact: `frozen_local_predictions.npz`, SHA-256
  `5415dfa101c56dc4c203f0913d506895b234d51cfefbd06c555e9b250a863c47`;
- implementation: `src/aiijc_puzzle/restored_border_ranker.py` and
  `scripts/run_restored_border_ranker_oof.py`.

Do not continue this checkpoint, tune its residual strength on the opened
panel, or run a global decoder. Reuse only the independent restored-descriptor
emitter inside a materially different multi-view/context-aware ranker, or
replace DRUNet with the explicitly new no-downsampling boundary model described
above, with a new source panel and a newly declared gate.
