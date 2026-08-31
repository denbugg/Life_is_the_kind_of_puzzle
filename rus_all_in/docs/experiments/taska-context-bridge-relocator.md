# TASKA: weak context bridge rigid relocator

Статус: **local32 OOF negative; held32/fresh32 не открывались**. Confirmed
six-arm selective+unique-fullres fusion остаётся pair leader. Test, production,
submission, matcher, denoiser, pixels и solver supply не затрагивались.

## Distinctness / no-repeat audit

Это не повтор:

- `taska-monotone-components` менял initial component placement внутри
  pre-solver raw tail и сохранял построенные компоненты; здесь intervention
  строго post-tail над confirmed final layout;
- `taska-component-relation-anchor` выбирал любой nontrivial component только
  из cross-component relation votes и проверял лишь vote-implied shifts;
  здесь only an OOF `p<0.5` graph bridge is cut, movable leaf cannot contain a
  single `p>=0.5` core tile, and every core-safe board translation is evaluated
  by unchanged raw dense cost.

До регистрации target-free feasibility с frozen final local model дала в
среднем 17.625 weak bridges и 0.9375 eligible weak-only cut sides per board;
strict cost-improving move был на 4/32. Поэтому это не был near-certain no-op
и получил ровно один preregistered OOF test.

## Fixed rule

1. Existing 27-feature context logistic is fit only on local32; local board
   receives OOF model by `GroupKFold(8, source_filename)`.
2. Input graph — only already realised focal-positive selected-supply edges,
   preserved by the confirmed tail. Classifier is not used to reorder edges.
3. A weak bridge is a `p < 0.5` edge whose removal disconnects its endpoints.
   A cut side is eligible only when it has at least two tiles and contains no
   endpoint from any `p >= 0.5` edge.
4. The high-confidence core positions are immutable. Every nonzero rigid
   in-board translation of an eligible subtree which does not enter core cells
   is considered; existing bijective local fill moves only non-core displaced
   tiles. Internal geometry survives.
5. At most one strictly lower original all-1104-bond raw-cost move wins;
   deterministic ties are lower bridge `p`, larger subtree, frozen edge order,
   row-major shift. Otherwise output is control bit-for-bit.

No weights, confidence threshold, component size, seam gain, tail, feature
or model variants were swept. Signed config:
[taska_context_bridge_relocator_v1.json](../../configs/taska_context_bridge_relocator_v1.json),
SHA-256 `55329ee531f247d6de78d865633155f4c6823c35793f02daf5ec3d0bccdf70aa`.

## Result

| Panel | Changed | Control pairs / exact | Candidate pairs / exact | Δ pairs | Δ exact |
|---|---:|---:|---:|---:|---:|
| local32 OOF | 1/32 | 326.78125 / 5.93750 | 326.75000 / 5.93750 | -0.03125 | 0.00000 |
| held32 | not opened | — | skipped by local pair gate | — | — |
| fresh32 | not opened | — | skipped by local gate | — | — |

Local pair result is one loss and 31 ties (source CI95 `[-0.09375, 0]`); exact
is 32 ties. The registered local pair gate `>=0` therefore closed the branch.
The raw-cost decrease from this particular weak cut did not translate into a
true puzzle contact improvement.

Do not retry the weak-bridge definition, `p=0.5` core split, exhaustive
raw-cost relocation or its nearby thresholds/tie-breaks on these opened
panels. A later component route needs a materially different signal that can
identify a globally correct gauge without relying on raw seam descent.

Report: `outputs/taska-context-bridge-relocator/fixed-v1/report.json`.

## Verification

- target-free smoke passed and frozen input artifacts predate labels;
- `ruff check`, `py_compile`, and `pytest -q
  tests/test_taska_context_bridge_relocator.py` (`2 passed`);
- output is always a strict original upright 576-tile permutation; core cells
  are asserted unchanged whenever a candidate move is made.
