# Foundation semantic component placement: preregistration stop

Status: **STOPPED before target decode or model fitting** (2026-08-30).

## Question

The bounded proposal was to use an official frozen DINOv2 ViT-S/14 only as an
absolute-position cost for layout. Rendering would remain a strict permutation
of the 576 upright 20×20 input tiles, followed by the same frozen restoration
tail as the buddies96 control.

The proposed candidate was deliberately narrower than a direct 576-class
head:

1. fit one orthogonal dirty-tile → contextual-target feature bridge on 16
   manifest-train boards;
2. average contextual target features into a 6×6 train-population field;
3. at inference, score only legal rigid translations of bilateral buddies96
   components with at least two tiles;
4. add the standardized component score at fixed weight 0.25 during packing;
5. assign exactly zero semantic score to singleton components and to leftover
   filling;
6. freeze predictions for eight disjoint manifest-train dev boards before
   decoding their targets.

In symbols, the new evidence would still have been additive isolated-tile
evidence:

`S(C, Δ) = mean[i in C] cosine(W f_dirty(i), field[bin(pos_C(i) + Δ)])`.

The component geometry restricts legal translations, but the feature extractor
does not jointly observe component pixels and the field is not conditioned on
the inference board.

## Mandatory prior-work audit

The audit found that this is a weaker repackaging of an already rejected
information family, rather than the requested materially new mechanism:

- P35 trained continuous row/column prediction over frozen DINO features.
  Train MAE was 4.215 slots and exact accuracy 0.6529%, while source-disjoint
  selection degraded to MAE 6.569 and exact accuracy 0.2387%. This is recorded
  in `docs/prior-research/cb1-orbit-r-p.md`.
- P32 combined mean-tile DINO features with a set Transformer. Train top-20 was
  13.8636% and placement 1.2080%; source-disjoint selection fell to top-20
  3.2878% and placement 0.1682%, explicitly diagnosed as source memorization.
- The Russian branch tested a stronger-context **DINO 4×4** positional probe:
  36 rigid 80×80 blocks, jointly encoded CLS+mean-patch DINO descriptors, then
  a set head over 36 coarse cells. On 512 fit / 64 development / exact-8 it
  reached dev cell accuracy 0.044705 versus 0.027778 chance (gate 0.10),
  Manhattan-distance reduction 0.078 (gate 0.25), and exact-8 accuracy 0.0625
  with distance reduction 0.0576. Its wrong-position subset worsened by
  0.002626, while training token accuracy reached 0.240: a clear fit/transfer
  gap. Historical source paths are
  `origin/таска-говно:history/runs/assembly_v1/research/global_layout_prior_audit_20260712.md`
  and
  `origin/таска-говно:history/runs/assembly_v1/dino_superblock_probe_output/v1/ANALYSIS.md`;
  the consolidated “do not repeat the same probe” verdict is also in
  `docs/prior-research/legacy-and-agent-branches.md`.
- M234/M235 separately found that positional signal increases with patch size,
  but the components actually available to the solver are too small. The
  indexed conclusions are in
  `docs/prior-research/generated/m-experiments.md`.
- The knowledge base therefore marks DINO absolute/global heads as
  reject-as-tested and preserves DINO only as a possible candidate generator,
  where P29 had increased candidate coverage without converting to layout.

Summing isolated-tile absolute scores over a component does not introduce new
inference information. It is still a DINO population-position unary, with a
rigid-component voting constraint. The earlier 4×4 probe jointly exposed DINO
to more pixels than most buddies96 components and still remained near chance.
Under the assigned fail-fast rule (“if the route duplicates prior work, stop
instead of running”), opening even the 16 fit targets was not justified.

Evidence file hashes at the decision point:

| File | SHA-256 |
|---|---|
| `docs/prior-research/knowledge-base.md` | `27250e29462f0406a3fb4af08d7b7adbc2cc2b5f438307e827c2f1536049367f` |
| `docs/prior-research/cb1-orbit-r-p.md` | `f9e8ab6c614567e1035c30ec1425bda6245d383a6e04817793ea510610ffc27a` |
| `docs/prior-research/legacy-and-agent-branches.md` | `b263609dac4b2cddc5cc2744d34a64ffb4cc6342a470920c4079e04698398705` |
| `docs/prior-research/generated/m-experiments.md` | `ac1dbe0524216d7006cea9700f125a164be51662d6fd5ca122fef558e5c5226b` |

## Access and mutation audit

- No fit or dev target PNG was decoded by this experiment.
- No calibration, holdout, or competition-test target was accessed.
- No prediction, fitted head, layout, restored image, or score report was
  produced.
- No production code, config, output, or submission was changed.
- The incomplete experimental runner/config/module/tests were removed after
  the stop decision so they cannot be mistaken for a validated runnable.
- Only manifest metadata was inspected to construct a possible 16/8
  train-only split; it was never executed.

## Pretrained asset provenance

The prerequisite itself was available and license-clear, so availability was
not the stop reason. The official assets were retained for possible future
non-duplicative research:

| Asset | SHA-256 |
|---|---|
| `dinov2_vits14_pretrain.pth` | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |
| `LICENSE` | `600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7` |
| `MODEL_CARD.md` | `70ca59606bee0a5fbb1baec80e7e29a93cd7cfbe26ca1910c52a852c4aab09d0` |

Checkpoint URL:
`https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth`.
The downloaded model card identifies ViT-S/14 and the retained official
license is Apache-2.0. A local load probe through timm 1.0.29 found 22,056,192
inference parameters after removing the checkpoint-only mask token.
The historical 4×4 probe records the same official checkpoint hash. Its
DINO state-dict hash was
`105f6c60aae15fee9c29f86dccefbcbdc1443fc40f1d0e9a5850513be1e34dbf`
and trained-head hash was
`a79887bc9aa7bc4cd067f9d8d75399caa7ad6dcd20f2dc2b156ded466ce8859c`;
the trained-head blob is absent from Git, so that experiment is evidence, not
a reusable runnable.

## What would actually be different

A future semantic arm must add inference-visible information missing here, for
example a model that jointly encodes an actual multi-tile island plus a
candidate island and predicts their relative relation, or a whole-dirty-board
conditioned field. It must not merely aggregate independent tile→absolute
position votes. Given the negative 4×4 result and the small size of current
components, even that route should first pass a cheaper representation-signal
probe before any end-to-end run.
