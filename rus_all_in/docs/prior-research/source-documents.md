# Идеи из PDF и их фактическая судьба

[К сводному индексу](README.md)

Проверены текст и визуальный render всех страниц четырёх PDF, достижимых из
`origin/codex/contour-normalization`:

- `docs/pazzle_strategy.pdf` — 4 страницы;
- `docs/pazzle_alt_ideas.pdf` — 3 страницы;
- `NEW APPROACH.pdf` — 12 страниц;
- `Modular ML Pipeline for 576-Piece Jigsaw Reassembly and Image Restoration_ Technical Design and Implementation Spec.pdf` — 13 страниц.

Первые два документа уже есть в раннем `origin/pasha883`; последние два
появляются в поздней истории. Это roadmap/specification, а не evidence:
заявленные ожидаемые 0.6–0.8 SSIM или 0.9 neighbour accuracy не являются
измеренными результатами.

## `pazzle_strategy.pdf`

Ранний диагноз:

- shuffled identity ≈0.08–0.11 SSIM;
- perfect placement degraded ≈0.44;
- perfect placement + NLM ≈0.57;
- текущий learned RestoreNet около 0.4385;
- pairwise v1 R@1 около 0.20, best-buddy precision около 0.48;
- bottleneck — placement/compatibility, затем безопасный NLM tail.

Предлагались wider seam scorer, real/synthetic hard negatives,
best-buddy/loops и NLM. Поздняя проверка:

- **подтверждено:** oracle/clean solver работает, NLM после layout полезен,
  простой learned seam scorer действительно недостаточен;
- **уточнено:** raw MGC не «мертв» вообще — M17/M18 почти решает clean puzzle,
  но dirty corruption убивает его;
- **уточнено:** placement важен для exact assembly, однако M420 показал, что
  SSIM допускает unconstrained content substitutions и не требует exact
  identity; bijection там не подтверждена;
- **реализовано:** learned matchers, hard negatives, loops, LP/SA/GA/LNS,
  restorers и NLM многократно проверены в ORBIT/P/M/E/V.

## `pazzle_alt_ideas.pdf`

| Идея PDF | Где дошла до эксперимента | Итог |
|---|---|---|
| A1 foundation features (DINO/CLIP/SAM) | P29/P30/P32/P35 | Отдельно измерялся DINO: candidate coverage +8.116 pp, но fusion/absolute heads не конвертировали; source memorization. CLIP и SAM как отдельные arms не проверялись. |
| A2 low-frequency/coarse-to-fine | M134–M142, M161–M177, M387–M391 | Oracle coarse map имеет payoff; предсказание/anchoring из bag недостаточно. |
| A3 joint photometric calibration | ORBIT PN, M8/M38/M129–M132/M179/M185/M199, E15 | Pre-match normalization в основном вредна; post-layout seam offset levelling полезен. |
| B1 Sinkhorn / set-to-grid Transformer | ORBIT PGA1, P5/P10/P11, M116/M403, V29 | Direct/global variants провалены; log-Sinkhorn полезен лишь как cost calibration; M403 four-seed gain нулевой. |
| B2 spectral layout | M293–M299 | Oracle mechanism работает, real signal слаб; spectral/diffusion averaging закрыты в проверенной форме. |
| B3 row/column decomposition | M250/M330–M335; V30 unary heads | Exact row-order objective предпочитает неверный optimum; V30 weak row/col unaries полезны только внутри joint LNS. |
| C1 learned global assembly energy | ORBIT GC1, M202/M337–M341 | Простые critics провалены; M340 впервые ранжировал truth выше forged optima, но rounds переобучились и не дали generator/reranker conversion. |
| C2 RL/backtracking | agent PPO, M411–M418 | PPO/DAgger механизмы работают локально; peripheral growth/value STOP/UNDO не двигают shipping block/placement. |

Иными словами, почти ни одна идея из PDF уже не является «непробованной».
Возвращаться можно только с тем изменением target/evidence, которое появилось
после M420.

## `NEW APPROACH.pdf`

Документ предлагает retrieve-then-rerank:

1. siamese/bi-encoder формирует shortlist;
2. early-fusion/JigsawNet cross-scorer оценивает пары;
3. MGC, loop consistency, Hungarian/global solver собирают board;
4. NAFNet/Restormer восстанавливает целое изображение;
5. опционально solve → restore → re-solve и x8 test-time augmentation.

Фактическая проверка:

- retrieval→rerank — **живая архитектурная схема**: V22, V25–V28 и M419
  показывают маленькие/средние gains, а V28 расширяет top-32;
- pairwise cross-scorer — R8 переносится с synthetic на raw плохо; P24/P25
  упёрлись в memory/runtime; V22 — лучший положительный пример на frozen
  shortlist;
- MGC/loops/Hungarian — многократно проверены; они не спасают отсутствующий или
  неверно ранжированный candidate signal;
- full-image restoration — полезна после фиксированного layout (S1/E18b), но
  не решает placement;
- solve→restore→re-solve — M21/M37/M42/M209/M292 не подтвердили bootstrap;
- x8/TTA — M37 дал почти нейтральный результат, не самостоятельный lever.

Заявления документа о neighbour accuracy порядка 0.9 и restorer SSIM 0.6–0.8
не достигнуты на честном dirty full-bag protocol.

## `Modular ML Pipeline…Spec.pdf`

Спецификация описывает модульную цепочку:

`synthetic degrader → fragment denoiser → directional compatibility → global
solver → full-frame NAFNet/Restormer → MS-SSIM+L1/x8`.

Полезные инженерные принципы:

- воспроизводить generator и permutation labels;
- разделять restoration, matching, solver и rendering;
- измерять каждый модуль oracle/ablation gates;
- сохранять exact permutation и bijection constraint;
- комбинировать pixel fidelity и perceptual/SSIM objective.

Поздние результаты существенно ограничили оптимистичные предпосылки:

- recovered permutation labels могут быть шумными/biased; P8 показал, насколько
  легко получить leakage через candidate order;
- «restoration easy» неверно для seam signal: residual correlated и сильнее на
  border, а меньший MSE часто ухудшает matching;
- generic global solver не компенсирует objective, optimum которого не truth;
- full-frame NLM/restoration — сильный output tail, но quality зависит от
  layout и production protocol;
- 576-piece scale действительно намного сложнее small-puzzle literature.

Спецификация остаётся хорошей картой модульных interfaces, но не планом
непроверенных моделей и не источником численных claims.

## Команды для доступа без checkout

```bash
git -C /path/to/pazzle_will_be_killed show \
  'origin/codex/contour-normalization:docs/pazzle_strategy.pdf' > /tmp/pazzle_strategy.pdf

git -C /path/to/pazzle_will_be_killed show \
  'origin/codex/contour-normalization:NEW APPROACH.pdf' > /tmp/new_approach.pdf
```

Для бинарных PDF нужен `git show ref:path`; обычный поиск по рабочему tree
увидит только документы текущей checked-out ветки.
