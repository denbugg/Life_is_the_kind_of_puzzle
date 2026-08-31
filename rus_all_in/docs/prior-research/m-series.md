# Глубокий аудит M-серии (M1–M420)

[К сводному индексу](README.md) · [Полный реестр 431 записей](generated/m-experiments.md)

Источник — `origin/autoresearch/pazzle-fixed-orientation-20260813` на
`6fb563c4b7`. Ветка содержит 240 коммитов, из них 233 после ранней базы
`origin/pasha883`; общий ствол до `5e36b3b` разделяется с V-веткой, последние
66 коммитов принадлежат только этой линии. Основной журнал —
`autoresearch-runs/pazzle-mgc-restoration-20260818/EXPERIMENTS.md`.

В журнале 431 уникальная именованная запись: M1–M420, кроме отсутствующего в
источнике M144, плюс `CORRECTION`, `RESULT`, `FINAL`, `GATES` и другие
повторные проверки. Полный поисковый индекс с commit и номером строки
генерируется автоматически; ниже — интерпретация, а не сокращённая замена
реестра.

## Итог линии

M-серия прошла путь от «восстановить границы и решить обычный пазл» до важной
смены самой целевой переменной:

1. На чистых фрагментах задача почти полностью решается: MGC + проверенный
   solver достигал placement 0.9965 (M17–M18). Значит, геометрическая цепочка
   принципиально работоспособна.
2. На реальном искажении соседство разрушается прежде всего в двухпиксельном
   внешнем кольце (M27–M31). Обычная реставрация, увеличение модели,
   post-processing, контекст и несколько видов seam loss не вернули нужный
   сигнал.
3. Learned matcher на raw-фрагментах поднял R@1 примерно с 0.056 до 0.240
   (M79), а корректная калибровка log-probability/Sinkhorn улучшила форму
   весов (M85–M87). Это улучшило локальный signal, но не создало связный
   правильный блок.
4. Множество detector/view/selector-ов увеличивало число правильных рёбер,
   purity или размер блока, но почти всегда не placement и не итоговый SSIM.
   До M420 рабочая модель мира свелась к bond-percolation knee около 450–500
   clustered correct bonds против примерно 348 доступных (M395, M407).
5. M420 показал, что эта модель мира измеряла **индекс фрагмента**, а SSIM
   измеряет **пиксельное содержание**. На восьми boards oracle-like раскладка
   из ближайших визуальных двойников имела placement 0 и SSIM 0.4236, тогда как
   случайные фрагменты — 0.2507. Выбор сделан по clean content в true cells;
   one-to-one/bijection constraint в журнале не подтверждён, поэтому это
   доказательство metric mismatch, а не deployable score или ceiling.

Последний пункт — не ещё одна маленькая абляция. Он отменяет право считать
старые index-based потолки потолками метрики, но constrained headroom пока
неизвестен. Следующий исследовательский цикл должен сначала воспроизвести
unconstrained и Hungarian/one-to-one варианты, затем ввести content-equivalent
labels и только потом заново оценивать retrieval, selector, solver и
«недостающие 100–150 рёбер».

## Хронология и устойчивые выводы

| Диапазон | Что проверялось | Что пережило поздние проверки |
|---|---|---|
| M1–M78 | MSE/ridge/MGC seams, photometry, tile restorers, кольцо, greedy/loop/LP/BP, context/EM, chroma, redundancy | На clean MGC силён; на dirty главный ущерб в border ring. Метрики restorer-а и seam proxy не гарантируют assembly. LP корректен на хорошем signal, но не спасает плохой. |
| M79–M147 | learned directional matcher, calibration, GA/relaxation/diffusion, joint heads, coarse field, output restoration | Learned raw matcher — реальный локальный прирост. Solver улучшает только режим выше activation knee. Coarse colour и простой global context не дают абсолютную раскладку. Считать M141 окончательным нельзя: его premise отменён M304. |
| M148–M224 | partial rendering, islands/components, multi-view agreement, restored view, packer, self-consistency, edge diversity | Частично правильный ответ может платить; agreement и независимые views дают более чистые рёбра; packer впервые двинул placement. Но growth/merge и абсолютное размещение остаются узкими местами. |
| M225–M313 | full shipping path, fill/render, border/frame cues, learned selector, placement search, oracle bounds, spectral/diffusion, restoration bounds | Border/frame prior — редкий настоящий absolute cue; seam levelling и аккуратный render полезны. Global averaging, spectral shrinkage, restore-input и простые selectors закрыты в измеренных формах. Несколько громких «information bounds» были отменены M304. |
| M314–M343 | oracle selection, precision/volume, seeded growth, row ordering, learned/global energies, frame prior | Candidate supply шире top-1; проблема была в выборе. Per-edge selector и consistency search не конвертировали. Energy, обученная против собственных ложных optimum, впервые предпочла truth, но rounds переобучились. Oracle outcomes были бимодальны. |
| M344–M385 | per-board controls, analytic views, contour signal, roster saturation, island purity, selector depth | Adaptive/per-board policies не выдержали whole-pipeline controls. Analytic filtered views обошли learned restorers как voters и были shipped. Contours переживают corruption на крупном масштабе, но не работают как fragment-scale adjacency. Чистые islands действительно можно растить, реальная сборка всё равно разваливается. |
| M386–M419 | metric/payoff, percolation, assignment, Sinkhorn, RL/DAgger policies, merge rules, chooser scaling | До M420: placement и крупный блок считались единственной валютой. Plain top-1 часто обгонял сложные edge sets; Sinkhorn gain исчез на четырёх seeds; peripheral growth закрыт. Max seam лучше mean при merge. Chooser на 2248 boards дал лишь +7 correct bonds и выбрал 4% доступного top-5 headroom. |
| M420 | pixel-twin substitution | Exact identity — metric-misaligned surrogate. Восьми-board clean oracle без подтверждённой bijection; content-aware constrained переоценка не завершена. |

## Числа, которые полезно помнить

| Проверка | Результат | Правильное чтение |
|---|---:|---|
| M1, clean seam MSE | R@1 0.788 | Локальная информация есть до corruption. |
| M17–M18, clean MGC chain | R@1 около 0.94–0.97; placement 0.9965 | Solver и orientation/origin pipeline принципиально исправны. |
| M18, clean+blur / noise4 / noise8 | placement 0.6389 / 0.4190 / 0.1782 | Assembly нелинейно падает при деградации edge signal. |
| M25, реальные restored tiles | placement 0.0009–0.0043; SSIM 0.152–0.181 | Рост seam proxy не конвертировался в сборку. |
| M26, best real restorer | R@1/R@5/R@20/R@64 = 0.159/0.328/0.512/0.680 | Старый candidate ceiling был index-based и после M420 требует пересчёта. |
| M28, oracle 2 px ring | R@1 0.774 | Для exact-neighbour задачи почти вся информация действительно в кольце. |
| M79, learned matcher raw | R@1 0.240, R@20 0.602 | Лучший ранний скачок дал matcher, а не denoiser. |
| M181, полный conformant arm | первый результат выше flat fill в том протоколе | Render/fill имеют смысл только вместе с честным end-to-end gate. |
| M248, closed-loop island merge | precision 0.938 | Высокая локальная precision достижима, но объём/связность ограничены. |
| M342, perfect selector | 29.2% boards > половины placement; 54.2% почти 0 | Среднее скрывает две разные моды отказа. |
| M403 FINAL | paired R@1 gain +0.0036 ± 0.0033 | Sinkhorn mixed loss не дал подтверждённого прироста. |
| M419, chooser 2248 boards | 331.3 → 338.5 correct bonds | Больше данных помогает, но тот же seam evidence почти исчерпан. |
| M420, nearest visual twin | SSIM 0.4236 при placement 0, 8 boards | Главная незавершённая диагностика; не recovery method и не one-to-one result. |

Числа из разных M-эпох не всегда прямо сопоставимы: менялись cache, scorer,
render, split, число boards и даже определение полезной цели. Использовать их
нужно вместе с ID и исходным protocol, а не как единую leaderboard-таблицу.

## Что не стоит повторять в прежнем виде

| Семейство | Закрывающие evidence | Вердикт в границах проверенного варианта |
|---|---|---|
| Простые MSE/ridge/MGC улучшения на dirty input | M3, M17, M35, M46, M59 | Аналитическая seam-функция не компенсирует потерянный signal. MGC остаётся хорошим clean-control. |
| Pixel-L1 / ring / larger restorer для matching | M6, M23–M25, M37–M40, M54–M55, M126, M209, M275, M279–M292, M301–M304 | Не возобновлять «ещё тот же denoiser». Restoration после layout для изображения — отдельный и живой вопрос. |
| Context restoration и EM bootstrap | M21, M42, M131, M152 | Контекст начинает помогать слишком поздно; неверные соседи портят границу. |
| Простое score fusion и ещё один похожий matcher | M39, M57, M124, M163, M169–M170, M186, M201, M208, M309 | Независимость важнее числа почти одинаковых моделей; обычное усреднение насыщено. |
| Solver-only rescue текущих costs | M44–M53, M64, M76, M111–M123, M242–M245, M293–M299, M360–M361 | Solver полезен выше knee; ниже него objective или evidence не поддерживают truth. |
| Greedy/component/island growth | M153, M180, M205–M207, M220–M223, M248–M251, M265–M273, M319–M329, M378–M381, M413–M418 | Механизмы иногда увеличивают block, но whole-pipeline placement не растёт. M415 max-contact — полезный diagnostic, не доказанный submission gain. |
| Coarse colour / absolute-coordinate prediction | M67–M70, M134–M142, M161–M162, M177, M196, M217, M228–M247, M387–M401 | Frame/border cue слаб, остальные поля не дают нужного anchoring. Не путать oracle payoff с predictability. |
| Spectral, diffusion distance, global smoothing | M293–M299 | Oracle mechanism есть, real signal слаб; три global-averaging варианта закрыты. |
| Hand-written selector / shallow context selector | M253–M260, M314–M321, M383–M385, M398–M410 | Candidate supply есть, но прежние признаки и objectives не выбирают достаточно хорошо. |
| Per-board policy по готовым summary | M344–M358 | Board texture одновременно двигает SSIM и выглядит как «качество сборки»; judges не выбирают выигрышный arm. |
| Sinkhorn как небольшой training add-on | M403, M403-CORRECTION, M403 FINAL | Четыре paired seeds дают результат, совместимый с нулём. |
| Ещё больше данных для того же five-seam chooser | M412, M419 | Overfit ушёл, но извлечено лишь около 4% shortlist headroom; нужен новый evidence, не масштабирование того же. |

«Закрыто» здесь означает «не повторять без нового механизма, данных, target или
протокола». Это не математическая невозможность семейства вообще.

## Ошибки, отозванные выводы и ловушки

- **M9:** ранний `place_acc` ломался из-за torus origin; для старых solver
  результатов нужен `fix_origin` или best cyclic shift.
- **M23/M40:** меньший residual sigma не означает лучший matching; сглаживание
  уничтожает discriminative border detail.
- **M68:** хороший score на confidently matched positions — selection bias, а
  не двукратное улучшение всего pipeline.
- **M125:** первая twin-tolerant target implementation разрушала модель; не
  переносить её код как подтверждение против идеи. M420 заново открывает
  content tolerance на уровне метрики.
- **M141 и M303:** выводы об information ceiling отменены M304 — premise был
  артефактом сопоставления labels.
- **M258:** corroboration gain не реплицировался.
- **M264:** degenerate Hungarian tie случайно передавал ответ; результаты
  отозваны.
- **M296 INTERIM:** unmatched control; финальный M306 отверг размер descriptor-а
  как lever и измерил заметный run-to-run noise floor.
- **M322:** повтор M222 с перезаписью script, не независимое подтверждение.
- **M339:** утверждение, что никакой global objective не может предпочесть
  truth, отменено M340. Практическая реализация M341 всё равно переобучилась.
- **M347, M388, M399, M405, M417:** headline менялся после правильного whole-
  pipeline или matched-control run. Всегда читать соседнюю `CORRECTION`.
- **M403:** single-seed +3% был ниже noise floor; четыре seeds дали ноль.
- **M386/M395/M407:** их identity/placement/bond interpretation полезна для
  exact assembly, но не является окончательной моделью SSIM после M420.

`NEXT.md` на tip ветки устарел: он всё ещё описывает chooser и Sinkhorn как
running и формулирует цель через exact-index placement. Финальные M403,
M419 и M420 появились позже и имеют приоритет.

## Что реально сохранено в Git

Полезные committed узлы:

- `src/distort.py`, `src/mgc.py`, `src/seam_embed.py`,
  `src/train_seam_embed.py`, `src/eval_seam_embed.py` — degrader, controls и
  learned matcher;
- `src/solve_loop.py`, `src/solve_lp.py`, `src/solve_ga.py`,
  `src/solve_relax.py`, `src/solve_anneal.py` — проверенные solver harnesses;
- `src/analytic_views.py`, `src/harvest_votes.py`, `src/consensus_islands.py`,
  `src/place_islands.py`, `src/border_prior.py`, `src/level_seams.py` — поздний
  multi-view/component pipeline;
- `src/infer_composed.py`, `src/infer_conformant.py` — end-to-end wiring;
- `src/choose5.py`, `src/train_choose5.py` — five-candidate chooser M412/M419;
- журналы `EXPERIMENTS.md`, `FINDINGS.md`, `PLAN.md`, `NEXT.md`.

Не всё воспроизводимо из одной ветки. Упомянутые поздно
`scratchpad/twin_slack.py`, `content_top1.py`, DAgger/value/merge-policy scripts
и их dumps на tip **не закоммичены**. Большинство checkpoints, score caches и
submission artifacts также лежали вне Git. Поэтому M420 — подтверждённый
диагностический результат из журнала, но его exact implementation/bijection
semantics нельзя перепроверить из tree, а незавершённый `content_top1` нельзя
выдавать за выполненный эксперимент.

## Открытые вопросы после M420

1. Ввести content-aware ground truth: tile/edge/cell считается допустимым по
   локальному SSIM или perceptual/RMSE tolerance, с отдельной проверкой, что
   surrogate коррелирует с full-image SSIM.
2. Пересчитать top-k retrieval, candidate coverage, correct bonds,
   percolation knee и placement payoff в новых labels. Старые цели 430,
   450–500 и 552 рёбер до этого нельзя использовать как design requirement.
3. Проверить unconstrained и Hungarian/one-to-one content-nearest layouts как
   диагностические baselines; M420 показал только clean-content oracle-like
   replacement в true cells, а не способ получить cells из input.
4. Переобучить/переоценить matcher и selector на equivalence sets, чтобы
   визуальный двойник не был hard negative. Это принципиально отличается от
   дефектной первой реализации M125.
5. Только после новой оценки решать, нужен ли V30-style global solver,
   five-candidate cross-attention или source retrieval. Все три должны быть
   measured end-to-end на одном split и renderer-е.
