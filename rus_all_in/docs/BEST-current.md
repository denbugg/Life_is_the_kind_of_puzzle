# BEST current

Срез: 2026-08-31.

- **Official leaderboard best остаётся `0.2762279116935955`.** Новый solver не
  запускался на competition test, не собирал ZIP и не менял default/submission.
- **Текущий подтверждённый pair-solver:** relation-level HGB selector одного из
  шести whole post-tail TASKA layouts. Formal source-disjoint `16×2`:
  `332.219→338.063` satisfied pairs/board, delta **`+5.844`**, source-CI95
  **`[+3.000,+9.126]`**, case W/T/L **`13/19/0`**. Exact delta `-0.156`.
- **Текущий подтверждённый matcher/verifier:** signed joint reciprocal tri-v2.
  На source-disjoint DEV32 он улучшил raw pooled R@1/R@5 на
  **`+0.7133/+1.1690 pp`**, а precision фиксированной reciprocal-head 5% — на
  **`+10.3448 pp`** (`77.7478→88.0927%`). Все девять directional/pooled
  bootstrap-CI положительны; это retrieval-result, а не готовый layout gain.
- **Следующий scale-переход:** target-free FIT256×2 cache закончен `512/512`
  (`3.342 GiB`, все SHA проверены). Reserved DEV64 остаётся неоткрытым;
  обучение и consumer freeze выполняются отдельным подписанным переходом.
- **Legal output contract:** strict permutation всех 576 original upright tile
  IDs; restored view используется только matcher-ом; denoised pixels не
  выводятся.
- **Production-ready layout CLI:** `uv run aiijc-taska-relation-selector
  tiles.npy --output-layout layout.npy --diagnostics-json receipt.json`. Он
  SHA-gate-ит frozen six-arm parent, model/config/report/evidence и не меняет
  прежние `aiijc-taska-best-pair-fusion` / `aiijc-taska-best-pair` fallbacks.
- Fixed all-edge synthesis из тех же HGB scores отвергнут уже на local32
  (`-127.25` pairs); whole-arm selector остаётся current pair leader.
- Same-case bridge из joint-head в relation arms также отвергнут: он изменил
  `10/32` layouts и ухудшил pairs `353.344→348.406` (`-4.938`), несмотря на
  `+0.094` exact. Не использовать и не подбирать nearby thresholds.
- **Head-only rebuild также закрыт.** Единственная заранее frozen
  FIT64-попытка использовать 58 reciprocal-head edges как основной
  каркас дала `349.484→69.063` pairs, delta **`-280.422`**, case/source
  W/T/L `0/0/64` и `0/0/32`; exact delta `-1.781`, Manhattan benefit
  `-0.584`. Все gates fail. Это закрывает только sparse-head-first
  пересборку: сигнал остаётся кандидатом для локальных constraints/repair
  поверх сильного control.
- **Frozen evidence:** model
  `ec4eca99243cdc6be20104d789b9e5d5598b79fa0d1b7e69bc37314375ad8c6b`,
  confirmation config
  `3d903eb595d1c0d152a8b53c7c9fa578b5b012227eeb03ab629a7dd24d5ce4e9`,
  report `d260872251077e1515251b6c7afc316af25df75045c8119112dff4f36c68ea23`.
- **Descriptive distance bridge на том же frozen formal roster:** selector
  одновременно дал mean Manhattan `14.9034→14.7269`, radius2
  `4.0907%→5.3331%` и clean/dirty/h20 SSIM
  `+.00148/+.00115/+.00085`; radius0/exact снизился на `0.0271 pp`. Это
  post-hoc consistency evidence, не новый gate или leaderboard claim.
- **Exact headline `12.875` не является устойчивым leader:** median `1`,
  `24/32` boards имеют `<=1` exact tile, один sample создаёт `83.93%`
  positive mass. Дальше обязательны median/quartiles/W-T-L/source bootstrap,
  max-contribution и leave-largest-positive вместе с mean.

Полное описание: [TASKA relation-level truth selector](experiments/taska-relation-truth-selector.md).
