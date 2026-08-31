# Покрытие всех веток `pazzle_will_be_killed`

[К сводному индексу](README.md) ·
[Машинный инвентарь](generated/branch-inventory.md)

Аудит выполнен по remote refs без checkout и без изменения исследовательского
репозитория. На момент среза Git содержит **27 remote refs: 26 именованных
веток + symbolic `origin/HEAD`**, и **498 commit objects** во всех историях.
Каждая именованная ветка приведена ниже, включая aliases, поглощённые histories
и orphan roots.

`Уник.` в машинной таблице означает commits, достижимые только из одного tip
среди всех 26 tips. Поэтому у `Taska-govna` ноль: её snapshot-коммит изучен, но
он также входит в более поздние CB1/M/V histories. Это не означает пустую
ветку.

## Legacy и ранние agent-ветки

Подробности: [legacy-and-agent-branches.md](legacy-and-agent-branches.md).

| Ветка | Tip / место в графе | Что в ней проверено | Итог |
|---|---|---|---|
| `origin/pasha883` | `94166506b0`; 7 commits | Первый full pipeline: recovered permutation, CompatNet/PairwiseNet, greedy+SA, RestoreNet/NLM, диагностика seam signal. | Oracle solver исправен, learned compatibility слаба: placement около chance, solve SSIM ≈0.106. NLM полезен только после хорошей раскладки. |
| `origin/MAESTRO` | тот же tip/tree `94166506b0` | Ничего сверх `pasha883`. | **Точный alias**, не независимый эксперимент. |
| `origin/Taska-govna` | `d28136151f`; snapshot +1 commit от pasha, затем вошёл в поздние ветки | Архив ORBIT/Rank96 до E21: generator forensics, learned/classical matchers, global/pose/component solvers, rendering/restoration. | Frozen Rank96 external 0.216198 включает 18 exact test overrides, поэтому не является чистым generic score. Основной solver signal остаётся слабым. |
| `origin/agent/research-restorer-rl-pipeline` | `f4e72db849`; +5 commits от pasha | Residual restorer, PPO swaps с guard, 5-class relation model, graph-greedy solver v2, packaging/tests. | Tile-level restoration metrics улучшены, assembly gain не доказан; RL даёт маленький 4-image gain; relation domain-adapt plateau около accuracy 0.52. Финальный graph solver без законченного validation/LB. |
| `origin/agent/ssim-scorer` | `f40e77baf9`; независимый orphan commit | Browser SSIM calculator и 18 reference PNG. | Формула полезна как forensic UI, но PNG — чистые test answers и те же 18 overrides. **Запрещено использовать для selection/validation.** |
| `origin/таска-говно` | `d6a82f82ce`; отдельный монолитный +1 commit, 1533 files | TileNAF, C1/HBT, directional QAP, harmonization/luma и большой bundle исторических artifacts. | Exact RGB LB 0.216784; luma заявлен округлённо ≈0.218. Candidate graph содержит 72.98% true edges, production adjacency лишь ≈6%: verifier/selection, а не renderer — узкое место. |

## Быстрые E-ветки

Подробности и все 28 достижимых commit objects:
[e-series.md](e-series.md).

| Ветка | Tip | Эксперимент | Итог |
|---|---:|---|---|
| `origin/autoresearch/e1-margin` | `c2c4f9675f` | Reciprocal-margin confidence bonus в SA. | **Reject:** положительный seed0, отрицательный alt seed; signal seed-unstable. |
| `origin/autoresearch/e2-score-fusion` | `63c14562ce` | Raw MGC+SSD `alpha=.2` вместе с learned scores. | Standalone joint gate не пройден; cue положителен и позже вошёл в E14. |
| `origin/autoresearch/e3-cache-multistart` | `72a9c3b747` | Cython exact SA hot loop; план equal-wall-clock multistart. | **Keep engineering:** exact layouts, 3.40× speed. Multistart не завершён, не считать reject. |
| `origin/autoresearch/e4-bestbuddy` | `44a874afe7` | Reciprocal component initializer. | **Reject:** adjacency выросла, robust/mean SSIM упали. |
| `origin/autoresearch/e11-relaxation` | `4d677494a6` | Sparse relaxation labeling. | Standalone не выиграл; solver mechanics переиспользованы в E14. Ветвь полностью содержится в E14-fusion history, поэтому global unique=0. |
| `origin/autoresearch/e12-cpsat` | `581c8f7fcf` | Sparse CP-SAT 4×4 repair top-16. | **Reject:** proxy/objective mismatch, target metrics ухудшены. |
| `origin/autoresearch/e13-border-encoder` | `a6058147df` | Corruption-aware boundary CNN package. | **Не измерено:** local checks есть, training/Kaggle run не состоялся. |
| `origin/autoresearch/e14-fusion-relaxation` | `2087f8d402` | E2 classical fusion → E11 relaxation/Hungarian на frozen raw-score cache. | **Лучший offline E-layout:** robust SSIM +0.001123, mean +0.001207, adjacency +0.01713, 3.43× быстрее paired SA. |
| `origin/autoresearch/e14-kaggle-port` | `2fd08f5b52` | Self-contained production port E14. | Packaging/parity есть, но production использует другой EdgeMatcher на restored tiles; offline win не переносится автоматически. |
| `origin/autoresearch/e15-no-gray-multiplex` | `77496e476d` | Raw/guarded-restorer score multiplex. | Positive smoke, strict gain gate не пройден; возможен train/eval overlap. **Promising, но unconfirmed.** |
| `origin/autoresearch/e18-nlm-polish` | `0d8a5260e2` | Full-image NLM h=9 после fixed layout + gray-cell raw fallback; bundle. | Offline E18b: mean +0.06782, robust +0.06523, 128/128 wins. Remote chain проиграла named control −0.006963 и timeout на 189/700; готового submission нет. |
| `origin/autoresearch/e19-nlm-dual-view` | `0e58675e09` | Per-tile NLM как второй classical edge view. | **Reject:** почти нулевой SSIM gain, adjacency ниже, runtime 2.4×. |
| `origin/autoresearch/e20-restored-ranker-verifier` | `a877065944` | Union E14 top-32 + restored descriptors, затем план BorderRanker. | Coverage +5.10/+4.65 pp; один direction чуть не прошёл gate. Ranker/layout вообще не запускался — не считать провалом scorer-а. |
| `origin/autoresearch/fast-score-gen1` | `c57c126ab1` | Интегрированная E18b chain, отчёт E1–E20 и remote follow-up. | Новых model metrics нет. Фиксирует champion chain и её проблемы: score mismatch, ложный fallback, двойной NLM/лишний legacy path. |

## Длинная CB1 / ORBIT / R / P-линия

| Ветка | Tip / охват | Что изучено | Итог |
|---|---|---|---|
| `origin/autoresearch/pazzle-fixed-orientation-cb1` | `9bd8db1dce`; 346 commits total, +339 от pasha, 338 после `Taska-govna` | Gap-free ledger ORBIT/R/P1–P39: candidate retrieval, relation/cross scorers, source retrieval, R5 restoration, solvers, DINO, raw/set Transformers, masked pretraining. | Официальный S1 rank96→R5→NLM = **0.237485**. SA2 source retrieval — самый сильный route при наличии source. Большинство solver/global heads закрыто; P8 invalid из-за leakage. |

Полный 338-commit map, протоколы и P1–P39:
[cb1-orbit-r-p.md](cb1-orbit-r-p.md).

## M-линия

| Ветка | Tip / охват | Что изучено | Итог |
|---|---|---|---|
| `origin/autoresearch/pazzle-fixed-orientation-20260813` | `6fb563c4b7`; 240 commits total, +233 от pasha; 431 records M1–M420 | Restoration, MGC/learned matcher, LP/GA/BP/relaxation, islands, view ensembles, selectors, energy/RL, shipping path и многочисленные controls/corrections. | M420 выявил content/identity mismatch: 8-board clean oracle-like nearest-twin layout дал 0.4236 при placement 0. Это не deployable/bijective result; exact-index diagnostics требуют constrained пересчёта. |

Подробности: [m-series.md](m-series.md) и
[generated/m-experiments.md](generated/m-experiments.md).

## Post-audit M421–M479 и V31–V33

| Ветка | Tip / охват | Что изучено | Итог |
|---|---|---|---|
| `origin/TASKA-GOVNO-EBANOE` | `ae9d231ad4`; +1 commit поверх M420; 71 records M421–M479 | Centred 2×2 square, raw-tail ordering, adaptive cutoff, verifier focal/top-k hinge, global permutation decoders, rendered assignment и plug-and-play denoiser. | **Unmatched must-replay:** M450/M455 adjacency `0.2702–0.2714` и placement около `4.4–6.0` tile/board численно сильнее current workspace, но protocol не matched и external weights/caches не в Git. M467 absolute top-k hinge закрыт. |
| `origin/codex/autoresearch-puzzle-v32-noise` | `ccd5b6a676`; V31 ancestry +1 commit | Exact synthetic corruption, paired clean/noisy score cache, 1.00M spatial board critic. | Только smoke/interim; финального pair/exact result нет. Candidate cache включает clean-target-derived boards, поэтому не production target-free comparator. |
| `origin/codex/autoresearch-puzzle-v33-transformer` | `7191d11a59`; launch `8d978b0`, result `7191d11` | 3.11M/8.77M whole-board Transformer selectors на V32 candidate cache. | T-M OOF `0.3143464` vs `0.3134581`, но locked `0.3716033 < 0.3776042`; T-MC тоже ниже. Reject; цифры adjacency нельзя выдавать за legal solver result из-за clean-assisted candidate pool. |

Точные metrics, code refs и legality caveats:
[m421-m479-and-v32-v33.md](m421-m479-and-v32-v33.md).

## V10–V30 neural/contour-линия

| Ветка | Tip / охват | Что изучено | Итог |
|---|---|---|---|
| `origin/codex/contour-normalization` | `1a714e115d`; 182 commits total, +175 от pasha; 8 commits после общего M/V base `5e36b3b` | V10/V18 dense Transformer, V22 cross-reranker, V23 bi-encoder, V25–V28 union/multimodal retrieval, V29 portfolio, V30 graph unaries + LNS. | V28 — лучший retrieval этой линии (top-32 51.45%); V30 — лучший global composite 0.11106. Direct placement ≈0.20%, translation-aligned ≈2.13%; terminal split исчерпан. |

Подробности: [v-series.md](v-series.md).

## Alias и containment map

```text
origin/HEAD ───────────────→ origin/pasha883 = origin/MAESTRO
                                      │
                                      ├── agent/research-restorer-rl-pipeline
                                      ├── Taska-govna
                                      │      ├── cb1: ORBIT/R/P1–P39
                                      │      └── M/V common history ... 5e36b3b
                                      │             ├── M1–M420
                                      │             │      └── TASKA M421–M479
                                      │             └── V10–V30 → V31–V33
                                      └── таска-говно (один отдельный bundle commit)

agent/ssim-scorer ───────── independent orphan root
E1…E20 / fast-score ─────── separate common E root, branching/cherry-picks
```

Эта схема упрощена до исследовательских семейств. Точные reachable/exclusive
counts и SHA являются authoritative в
[машинном инвентаре](generated/branch-inventory.md).

## Проверка полноты

- перечислены все 26 значения `refs/remotes/origin/*`, кроме отдельно
  отмеченного symbolic `origin/HEAD`;
- exact aliases не выданы за независимые эксперименты;
- поглощённые commits E11/Taska всё равно разобраны в своих семейных отчётах;
- orphan `agent/ssim-scorer` изучен отдельно;
- для CB1 есть gap-free карта 338 commits;
- для E-семейства есть карта 28 reachable commit objects/cherry-pick pairs;
- для legacy набора есть карта 15 commits;
- для M есть автоматически проверяемые 431 journal records с blame commit
  и ручной appendix для 71 поздней записи M421–M479;
- для V проверены V10–V33, включая reports и code-path
  clean-assisted candidate pool V32/V33.
