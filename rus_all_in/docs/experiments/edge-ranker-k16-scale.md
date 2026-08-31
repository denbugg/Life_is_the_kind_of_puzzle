# Candidate-k16 / train256 pairwise edge ranker

## Decision

**Reject as tested. Do not run the preregistered confirmation, holdout, test, or
production integration.** The larger raw pairwise ranker improved local exact
edge retrieval and true adjacency on every calibration board, but it did not
reach useful absolute layout quality and materially reduced the compliant final
SSIM endpoint.

This is the terminal result for the hypothesis “make the existing raw pairwise
ranker broader and larger”: `candidate-k=16`, 256 training boards, four epochs,
128 rows per board, width 32 and hidden width 64. No hyperparameter or blend was
tuned after seeing calibration targets.

## Preregistration and leakage boundary

Before training or opening calibration targets, the complete experiment was
written to `configs/edge_ranker_k16_scale_preregistered_v1.json`, SHA-256
`22d81c542b5cd598fde6cdd6fadb7847ea974ef68c7f5774e336e9fc5b5ab422`.
It fixed:

- first 256 shared-selector `train` records, roster digest
  `4e407402ad5c81fd1698c65a22da5c5b8d12ea886608eed337051473e54348dc`;
- primary calibration records `228:252`, roster digest
  `d36f91a3f83718a28b295700f7f0bb7e1a8374f1e820f774e51d132ee793b103`;
- one disjoint confirmation `252:276` only if every primary gate passed;
- unchanged analytic views `raw,tile_z,bilateral,gray`, raw model channels,
  exact-neighbour CE plus the existing trusted clean-continuity teacher;
- identical `solve_buddies(max_edges=96)` for control and learned layouts;
- original upright 20x20 dirty tiles, each used exactly once;
- frozen post-assembly RGB seam offsets -> bounded luminance gains -> one
  coloured NLM pass at `h=20`.

The existing `scripts/run_edge_ranker.py` trainer froze all 24 dirty-only score
matrices, layouts and images before its first target read. The exact-tail runner
then independently repeated phase one, wrote a content-addressed prediction
commitment, read it back, and only then opened targets for labels, metrics and a
manual sheet. All 48 raw audits (24 boards x 2 arms) passed exact declared
reassembly, 576 unique indices, tile-multiset equality and raw-pixel
preservation. Holdout and test were not opened.

## Frozen gate

Every condition was required:

1. learned mean adjacency `>= 0.08`;
2. paired adjacency-gain 95% bootstrap CI lower bound `> 0`;
3. mean final SSIM delta `>= -0.002`;
4. paired final-SSIM 95% CI lower bound `>= -0.006`;
5. mean translation-aligned placement delta `>= 0`.

The same 20,000-replicate paired percentile bootstrap and seed `20260830` were
fixed before target access.

## Training and candidate diagnostics

The unchanged trainer completed within the 25-minute infrastructure cap:
board preparation `334.82 s`, four training epochs `410.70 s`, inference freeze
`55.08 s`, and its target diagnostics `3.18 s`. Exact CE decreased
`2.4801 -> 2.1981 -> 2.0855 -> 2.0277`; teacher CE was
`3.3752 -> 3.4893 -> 3.5284 -> 3.5486`.

On the primary calibration panel, broadening the same four dirty-only emitters
worked as intended:

| Diagnostic | k5 / bilateral control | k16 learned | Delta |
|---|---:|---:|---:|
| Exact neighbour in candidate union, all legal rows | 0.298611 | 0.473279 | +17.467 pp |
| Exact neighbour in candidate union, trusted rows | 0.497441 | 0.673770 | +17.633 pp |
| All pooled exact R@1 | 0.067369 | 0.112923 | +4.555 pp |
| Trusted pooled exact R@1 | 0.137574 | 0.259062 | +12.149 pp |

Thus neither candidate coverage nor local R@1 collapsed. The failure is the
conversion from stronger local edges to a globally useful image.

## Exact compliant-tail result

| Mean over calibration 228:252 | Bilateral buddies96 | k16 ranker buddies96 | Delta |
|---|---:|---:|---:|
| Adjacency | 0.032684 | 0.062689 | **+0.030005** |
| Right adjacency | 0.031175 | 0.063406 | +0.032231 |
| Down adjacency | 0.034194 | 0.061972 | +0.027778 |
| Direct placement | 0.001664 | 0.001447 | -0.000217 |
| Translation-aligned placement | 0.009404 | 0.011574 | +0.002170 |
| Raw SSIM | 0.109956 | 0.103507 | **-0.006449** |
| RGB+luma SSIM | 0.114721 | 0.109600 | -0.005121 |
| RGB+luma+NLM h20x1 SSIM | **0.247168** | 0.237782 | **-0.009386** |

Adjacency improved on 24/24 boards; mean gain `+0.030005`, preregistered 95% CI
`[+0.027325,+0.032646]`. Translation-aligned placement also passed its weak
non-decrease condition. The other three gate conditions failed:

- learned absolute adjacency `0.062689 < 0.08`;
- mean final SSIM delta `-0.009386 < -0.002`;
- final SSIM CI lower bound `-0.016116 < -0.006` (full CI
  `[-0.016116,-0.002948]`, 8 wins / 16 losses).

Manual inspection agrees with the numbers. The learned arm makes some larger
colour/texture components, but neither arm reconstructs a globally readable
scene; the k16 arm is often less faithful in large colour regions. This is not a
manual-review-quality puzzle solution.

Because the primary gate failed, the script rejects confirmation mode before
any record at offset 252 can be decoded. There was no blending, post-hoc score
scaling, retraining, holdout access, test access, or production change.

## Reproduction and artifacts

```bash
.venv/bin/python scripts/run_edge_ranker.py \
  --output-dir outputs/edge-ranker/scale-raw-k16-train256-cal24-offset228 \
  --train-limit 256 --eval-limit 24 --eval-offset 228 --epochs 4 \
  --rows-per-board 128 --batch-rows 24 --pair-batch 1024 \
  --candidate-k 16 --view-mode raw --width 32 --hidden 64 --device mps

.venv/bin/python scripts/run_edge_ranker_k16_tail.py --mode primary --device mps
```

Hashes:

- checkpoint `edge_ranker.pt`:
  `939305342a9551806cd1896aae3020950ab0c883f5d0f52063989d7269f1b7e7`;
- trainer `report.json`:
  `1a95fb25ecb49e832be3cf38af0c980c77ee8e87c668849ed169bf495157a74f`;
- exact-tail `report.json`:
  `6fe6790f470c3e39a28d3c5c050feac1cd08623b7db3677d4f2102d7028ddad9`;
- `prediction-commitment.json`:
  `30daaf1a0761a1c72d58390ae6091bf6c943ae9e59a4fda1e92384eb0d3d9eda`;
- `manual-layout-sheet.png`:
  `b054a1f626af5f8fc8760d262b7d73d8687b272fc32392587a44d0499655b7df`.

The exact-tail freeze took `87.21 s`; target-assisted metrics and sheet took
`5.09 s` on MPS.
