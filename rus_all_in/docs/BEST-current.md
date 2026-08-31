# BEST current

Срез: 2026-08-31.

- **Official leaderboard best остаётся `0.2762279116935955`.** Новый solver не
  запускался на competition test, не собирал ZIP и не менял default/submission.
- **Текущий подтверждённый pair-solver:** relation-level HGB selector одного из
  шести whole post-tail TASKA layouts. Formal source-disjoint `16×2`:
  `332.219→338.063` satisfied pairs/board, delta **`+5.844`**, source-CI95
  **`[+3.000,+9.126]`**, case W/T/L **`13/19/0`**. Exact delta `-0.156`.
- **Legal output contract:** strict permutation всех 576 original upright tile
  IDs; restored view используется только matcher-ом; denoised pixels не
  выводятся.
- **Production-ready layout CLI:** `uv run aiijc-taska-relation-selector
  tiles.npy --output-layout layout.npy --diagnostics-json receipt.json`. Он
  SHA-gate-ит frozen six-arm parent, model/config/report/evidence и не меняет
  прежние `aiijc-taska-best-pair-fusion` / `aiijc-taska-best-pair` fallbacks.
- Fixed all-edge synthesis из тех же HGB scores отвергнут уже на local32
  (`-127.25` pairs); whole-arm selector остаётся current pair leader.
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

Полное описание: [TASKA relation-level truth selector](experiments/taska-relation-truth-selector.md).
