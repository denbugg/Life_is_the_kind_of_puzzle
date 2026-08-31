# Предыдущие исследования: сводный индекс

Дата среза: 2026-08-31. Источник:
`https://github.com/fenix0501/pazzle_will_be_killed.git`, локальный fetch
`/Users/rusyalain/Documents/GitHub/pazzle_will_be_killed`.

Аудит покрывает **все 26 именованных remote-веток**, symbolic `origin/HEAD`,
**498 commit objects**, все branch tips и основные code/report/model artifacts.
Для самой длинной M-серии построен полный индекс **431 journal records**
(M1–M420, M144 отсутствует в самом источнике, corrections сохранены отдельно).
Поздние 71 запись M421–M479 и ветки V32/V33 разобраны в
[отдельном appendix](m421-m479-and-v32-v33.md), чтобы не менять
воспроизводимый индекс старого M-tip.
Research-репозиторий изучался read-only: checkout, history и рабочие файлы не
менялись.

## Главный вывод

Нельзя начинать новый цикл с «ещё одного seam scorer/denoiser/solver». Эти
семейства уже многократно проверены. Поздняя находка M420 — официальный SSIM
платит за пиксельное содержание, а не за identity фрагмента — теперь независимо
подтверждена на общем frozen protocol и при строгой one-to-one биекции. На
holdout-48 clean Hungarian derangement имеет SSIM `0.533053` при placement 0 и
без duplicate use; recovered dirty render после colored NLM `h=9` имеет
`0.383725`. Оба числа остаются target-assisted diagnostics, а не deployable
score.

Запланированный после аудита цикл выполнен: frozen split создан; M420 проверен с
bijection control; pixel-tail roster сравнен на calibration-48/holdout-48;
analytic content-aware candidate supply измерен; position-aware listwise verifier
обучен и масштабирован. Подробный authoritative итог находится в
[реестре новых экспериментов](../experiments/README.md).

Практический итог цикла:

1. colored NLM `h=9` продвигается как default output tail с paired control на
   фактической раскладке solver-а;
2. content-aware one-to-one diagnostics и labels сохраняются, но target-free
   substitute recovery всё ещё не решён;
3. analytic union даёт достаточный shortlist, но не fixed-budget/global win;
4. глобальная content-multipositive формулировка verifier-а закрыта после
   calibration scale-up: `all` exact `+1.079 pp`, но content≤20 `−3.378 pp`
   против ensemble и `−4.789 pp` против bilateral; strict trusted exact/content
   `+3.246/+3.266 pp`, но content почти совпадает с exact после исключения
   low-margin twins и не доказывает content slack;
5. fresh scale-up holdout и decoder не запускались, потому что gate провален.

## Самые сильные подтверждённые точки

| Результат | Что именно доказано | Ограничение |
|---|---|---|
| **S1: 0.237485 official SSIM** | Сильнейший generic platform anchor: `rank96 → R5 → canonical NLM`, 700 test images; зафиксированный runner не задаёт override directory. | Checkpoints/ZIP не в Git. Старый score 0.216198 для сравнения включал 18 exact overrides, поэтому delta не является чистым paired baseline. |
| **Frozen colored NLM `h=9`** | На paired holdout-48 raw 0.430621→0.557442, gain `+0.126821`, 95% CI `[+0.114821,+0.138820]`, 48/48 wins. | Fixed-layout pixel-tail, а не solver score; layouts target-assisted. |
| **M420 при строгой биекции** | На holdout-48 clean derangement 0.533053, dirty+NLM 0.383725, placement/reuse 0. Metric slack не исчезает при one-to-one. | Не recovery method: target выбирает substitutes и помогает alignment. |
| **Analytic candidate supply** | На trusted holdout union@32 exact/content≤20: right 0.7719/0.7970, down 0.7931/0.8182. | Target-assisted labels и easy-half trusted subset; фактический budget около 78–79, не 32. |
| **Scaled listwise verifier** | На calibration-24 all exact `+1.079 pp`; strict trusted exact/content `+3.246/+3.266 pp`. | All content регрессировал на `−3.378 pp` vs ensemble и `−4.789 pp` vs bilateral: gate fail, formulation reject-as-tested, fresh holdout не открыт. После strict confidence content почти совпадает с exact и не доказывает content slack. |
| **SA2 source retrieval R@1 94.24%** | Strict held verification: 100% true accept / 0% wrong accept; SA1 tile agreement 84.79% при правильном source. | Покрытие corpus и разрешённость внешних sources. |
| **E14 offline layout** | На frozen raw DirectionalTransformer cache full128: robust SSIM +0.001123, adjacency +0.01713, 3.43× быстрее SA. | Full128 включает tune32; на untouched96 robust gain +0.000631. Kaggle port использует другой EdgeMatcher/restored domain; production win не доказан. |
| **E18b offline pixels** | Full-image NLM h=9 + gray guard: mean +0.06782, 128/128 wins. | h выбран на связанных data; remote chain −0.006963 и timeout 189/700, fallback реализован неверно. |
| **V28 retrieval** | Top-1/5/32 = 15.73/29.20/51.45%, MRR 23.02% на 11 fresh scenes. | Exact-index target; upstream weights/caches отсутствуют. |
| **V30 global solver, partial** | Adjacency 10.57%, direct placement 0.150→0.197%, composite 0.11106. | Final 15 caches уже были просмотрены; translation-aligned placement 2.18→2.13%, proxy не SSIM, edge calibrator rejected. Это comparator, не готовый следующий default. |
| **TASKA M450/M455 solver — unmatched must-replay** | Historical held adjacency `0.2702–0.2714` (около `298–300/1104` bonds), placement около `4.4–6.0` exact tiles/board; поздний journal reference для shipping adjacency `0.2890`. | Другой split/pipeline и недостающие external checkpoints/caches; не current champion до matched source-disjoint replay. |
| **M17/M18 clean control** | MGC chain собирает clean puzzle с placement 0.9965. | Dirty corruption разрушает boundary signal; не production result. |

## Навигация

| Документ | Для чего открывать |
|---|---|
| [branches.md](branches.md) | Одна строка на каждую из 26 веток, tip SHA, lineage и итог. |
| [knowledge-base.md](knowledge-base.md) | Поисковая матрица «идея → где проверяли → verdict → условие возврата» и приоритеты. |
| [layout-sorter-ledger.md](layout-sorter-ledger.md) | Authoritative cross-series таблица методов сортировки: direct exact, split/leakage, solver, artifacts и открытые блоки. |
| [../experiments/README.md](../experiments/README.md) | Выполненные после аудита frozen-panel эксперименты, точные числа, confidence и решения. |
| [legacy-and-agent-branches.md](legacy-and-agent-branches.md) | pasha/MAESTRO, оба архива, ранний restorer/RL agent и leaky SSIM scorer; 6 refs / 15 commits. |
| [cb1-orbit-r-p.md](cb1-orbit-r-p.md) | Полный ORBIT/R/P1–P39 audit и gap-free map 338 commits после archive base. |
| [e-series.md](e-series.md) | E1–E20/fast-score: 14 refs, 28 commit objects, offline/production caveats. |
| [m-series.md](m-series.md) | Интерпретация M1–M420, corrections, closed routes и M420 reframe. |
| [m421-m479-and-v32-v33.md](m421-m479-and-v32-v33.md) | Post-audit appendix: TASKA M421–M479, top-k hinge M467, V32/V33 и strict-legality caveats. |
| [v-series.md](v-series.md) | V10–V30 retrieval/global solver, matched tables, artifacts и split caveats. |
| [source-documents.md](source-documents.md) | Четыре PDF-roadmap-а и crosswalk их идей с фактическими экспериментами. |
| [generated/branch-inventory.md](generated/branch-inventory.md) | Воспроизводимый список remote refs, tips, total/exclusive commits и tree sizes. |
| [generated/m-experiments.md](generated/m-experiments.md) | Все 431 M-records: title, source verdict, current-line blame commit и строка. |

## Что точно не повторять без нового механизма

- обычные raw seam MSE/MGC/phase/derivative variants как единственный scorer;
- ещё один pixel-L1 tile denoiser ради matching или denoise→match→repeat;
- простой score averaging нескольких почти одинаковых models;
- best-buddy/reciprocity/2×2 cycles как полный assembler;
- solver-only rescue прежних score matrices: greedy, LP, BP, CP-SAT,
  Sinkhorn, spectral/diffusion, SA/LNS уже имеют activation/proxy failures;
- direct set-to-grid, absolute-position head или coarse colour map из bag;
- peripheral component/island growth, RL STOP/UNDO или новый fixed merge rule;
- per-board quality policy по texture/summary features;
- P8 artifacts или candidate-slot-as-rank assumption;
- E18b gray guard ради contest SSIM без отдельного safety-требования;
- ту же global content-multipositive verifier formulation на большем числе boards;
- absolute k-th-threshold top-k hinge: M467 уже рухнул с `.6420` до `.4417`
  и не вернул baseline за четыре эпохи;
- подбор по 18 известным test references из `agent/ssim-scorer`.

Полные границы этих verdicts и resource-only stops находятся в
[knowledge-base.md](knowledge-base.md). «Не повторять» не означает, что класс
математически невозможен; нужен новый target, evidence source или существенно
другой protocol.

## Главные риски воспроизводимости

- Во многих ветках нет `.pt/.npz/.zip` artifacts: checkpoints и score caches
  были на `E:\pazzle_work`, Kaggle или `/home/kva`.
- Split manifests иногда упомянуты только hash/path и не закоммичены.
- Метрики разных серий используют разные boards, caches, renderers, seeds и
  даже exact/translation-aligned definitions. Межсерийный ranking по числам
  без matched rerun недопустим.
- Historical Rank96 0.216198 включает 18 exact source overrides. Эти же 18
  clean test PNG лежат в orphan SSIM-scorer; использовать их для selection
  нельзя.
- E14 offline и Kaggle production score sources различаются.
- V29 — 3-fold OOF на 15 caches; V30 имеет отдельные train/validation heads,
  но final evaluation на тех же 15 ранее просмотренных caches. После V28 нового
  terminal split нет.
- M/P журналы содержат self-corrections; headline без соседнего `CORRECTION`
  часто неверен.

## Как искать и обновлять

Поиск идеи:

```bash
rg -ni 'sinkhorn|spectral|cross.?attention|denois|source retrieval' \
  docs/prior-research
```

После нового fetch машинные индексы пересобираются без checkout:

```bash
uv run python scripts/build_prior_research_index.py \
  --research-repo /path/to/pazzle_will_be_killed
```

Скрипт обновляет только:

- `docs/prior-research/generated/branch-inventory.md`;
- `docs/prior-research/generated/m-experiments.md`.

Если tip изменился, сначала смотреть новые exclusive commits, затем дополнять
ручные семейные отчёты. Машинный diff не заменяет чтение protocol/results.

Новые frozen эксперименты этим скриптом не пересобираются. Для verifier scale-up
authoritative файл —
`outputs/content-verifier/scale128-calibration24-final.json`; предварительный
`scale128-calibration24.json` оставлен для provenance, но его trusted-content
headline использовал менее строгую `trusted_query` семантику.

## Минимальный gate для следующего эксперимента

1. Зафиксировать source-disjoint split, cache/checkpoint hashes, seed и renderer.
2. Назвать target: exact-index diagnostic, content-equivalent diagnostic или
   официальный full-image SSIM.
3. Сделать leakage/control probe до training.
4. Проверить cheap retrieval/coverage gate.
5. Только затем запускать solver и paired end-to-end SSIM.
6. Считать single-seed gain ниже noise floor незавершённой гипотезой.
7. Сохранять code + JSON + artifact hashes; resource stop помечать отдельно от
   scientific reject.
