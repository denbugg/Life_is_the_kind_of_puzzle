# TASKA: realised-edge context protector

Статус: **закрыто отрицательным local32 OOF; held32/fresh32 не открывались**.
Confirmed six-arm selective+unique-fullres fusion остаётся pair leader.
Competition test, production, submission, matcher, denoiser, supply и pixels не
затрагивались.

## Что проверялось

Это не новый matcher и не новая генерация рёбер. Для уже выбранного six-arm
**pre-tail** layout берутся исключительно supplied edges, которые одновременно:

- имеют frozen focal logit `>= 0`;
- уже реально являются направленным соседством выбранного layout.

Для каждого такого ребра frozen target-free table содержит 27 ровно
зарегистрированных признаков: focal/raw seam rank+margin, axis, membership в
current/selective-new/unique-fullres, выбранный arm, agreement шести pre-tail
layouts, локальную компонентную структуру и oriented supply conflicts. One
`StandardScaler + LogisticRegression(C=1, lbfgs, max_iter=1000, random_state=0)`
без class weights оставляет natural `p >= 0.5` edges. Только эти endpoints
передаются неизменному raw non-adjacent `tail96`; никакого строения новых edge,
изменения seams или изображения нет.

Local prediction — честный `GroupKFold(8)` по `source_filename`: модель
validation source обучается лишь на остальных source groups. Before any label
reconstruction были записаны feature rows, pre-tail layouts, raw cost matrices
и mechanical replay confirmed control. Local model только после OOF freeze был
бы подписан для held; gate не прошёл.

Signed preregistration: [taska_context_protector_v1.json](../../configs/taska_context_protector_v1.json),
SHA-256 `1d08add9ae29a836298f4f5e15c8e03d0aee8c66b43445f8d903635dddbcf805`.

## Result

| Panel | Control pairs / exact | Candidate pairs / exact | Δ pairs | Δ exact | W/T/L pairs |
|---|---:|---:|---:|---:|---:|
| local32 OOF | 326.78125 / 5.93750 | 326.40625 / 5.84375 | -0.37500 | -0.09375 | 10/7/15 |
| held32 | not opened | skipped by local gate | — | — | — |
| fresh32 | not opened | skipped by local gate | — | — | — |

On local32 the filter kept `264.125` realised edges per board. Pair source CI95
was `[-1.6875, +0.90625]`; exact CI95 `[-0.34375, +0.15625]`. Fixed local gates
were pair `>= 0` and exact `>= -1`, so the -0.375 pair mean correctly prevented
held construction/scoring.

Report: `outputs/taska-context-protector/fixed-v1/report.json`.

## No-repeat ledger

Do not retry this exact context feature roster, `p=0.5`, unweighted linear
logistic head, selected-realised-edge-only corpus, or tail96 interface on the
opened panels. This is distinct from the earlier unique-fullres **pre-solver**
calibrator (only new fullres supply) and incidence-GNN/hard-edge ordering, but
its direct post-tail freezing formulation is now negative. A later edge
selection direction needs a materially different inference signal or a
different solver intervention, not threshold/model/feature tuning here.

## Verification

- `ruff check` passed for module, runner and tests;
- `py_compile` passed;
- `pytest -q tests/test_taska_context_protector.py` — `2 passed`;
- target-free smoke rebuilt and bitwise replayed the frozen control before any
  reference was reconstructed;
- all candidate outputs are strict original upright 576-tile permutations.
