# Handoff исследования solver-а — 2026-08-30

Этот документ фиксирует состояние проекта в момент, когда пользователь попросил
считать текущий рабочий цикл завершённым. Он сохраняет результаты, ограничения,
приоритеты и точку возобновления. Полный аудит прежних Git-веток остаётся в
[prior-research/README.md](prior-research/README.md), а подробные протоколы новых
запусков — в [experiments/README.md](experiments/README.md).

## Короткий статус

- Лучшим итоговым submission остаётся пользовательски подтверждённый
  `fixed-B standard + buddies96` со score **`0.2762279116935955`**.
- Новый combined arm `Union-v2 + historical h20` получил
  **`0.24201676406343967`** и отклонён как итоговый.
- Это не честное сравнение только solver-ов: одновременно изменились layout и
  restoration tail. Union-v2 нельзя объявлять хуже buddies96 без same-tail A/B.
- Лучший подтверждённый относительный matcher/layout arm сейчас — Union-v2:
  adjacency `14.419%`, exact `1.281/576` tile/board на frozen fresh64.
- Последняя oracle-диагностика показала, что candidate supply уже намного богаче,
  чем итоговая раскладка. Главный ближайший bottleneck — совместный выбор
  согласованных относительных смещений и глобальная упаковка компонентов.
- Текущий цикл закрыт; тяжёлых процессов после сохранения этого handoff нет.

## Что именно требуется решать

Нужно восстановить изображение `480×480` из `576` исходных upright-фрагментов
`20×20`, то есть получить строгую сетку `24×24`, а затем легально восстановить
качество изображения. Основная метрика соревнования — RGB SSIM.

Из письма организаторов и обсуждения с пользователем следует обязательный
контракт:

1. В layout каждый исходный фрагмент используется ровно один раз.
2. Фрагменты нельзя заменять, растягивать, поворачивать, деформировать или
   подменять одноцветными/синтетическими квадратами.
3. До и после solver-а разрешена обработка качества: denoise, deblur,
   harmonization и другие универсальные преобразования, если они не превращают
   решение в генерацию/подмену содержимого.
4. Для matcher-а можно использовать restored views и любые обученные признаки,
   но финальная сборка должна оставаться биекцией исходных tiles.
5. Диффузионная или generative модель допустима только как безопасный bounded
   restoration/feature extractor. Генерация нового содержимого вместо tiles —
   высокий manual-compliance риск и не входит в текущий план.
6. У фрагментов нет специальных пазловых выступов/узоров: сигнал приходится
   извлекать из обычных граничных пикселей, текстуры и глобальной сцены.

Подробный compliance gate: [submission-compliance.md](submission-compliance.md).

## Зафиксированные submission-артефакты

### Лучший, не заменять без лучшего официального результата

- pipeline: `fixed-B standard restoration + buddies96 layout`;
- official score, сообщённый пользователем: `0.2762279116935955`;
- ZIP: `outputs/compliant-fixed-b-standard-submission-v1/submission.zip`;
- SHA-256: `07298cd3e92e4420eacff9c797f9b9d189a67222c9d25955459dbde9795ef0ae`.

Этот ZIP остаётся отмечаемым как лучший. Layout buddies96 дал слабые offline
exact/adjacency показатели, но сильный pixel tail обеспечил лучший известный
leaderboard score. Нельзя автоматически переносить вывод с offline exact на
SSIM и наоборот.

### Последний проверенный submission, не отмечать лучшим

- pipeline: `Union-v2 layout + RGB/luma + single colored NLM h20`;
- official score, сообщённый пользователем: `0.24201676406343967`;
- ZIP: `outputs/union-v2-submission/submission-union-v2.zip`;
- SHA-256: `8866e060cae32d56277470f565779cd68826d9a766513e3e81eed2165f6d9725`;
- разница к лучшему: `-0.03421114763015583`, или около `-12.39%`.

700/700 PNG имеют RGB `480×480`, а сохранённые layouts прошли проверку строгой
перестановки исходных upright tiles. Однако независимый полный neural replay на
MPS нельзя честно назвать пройденным: недетерминированный `index_add_` меняет
малые float-значения, после чего discrete decoder иногда выбирает другой layout.
Правильная граница утверждения и production lineage описаны в
[union-v2-submission-production.md](union-v2-submission-production.md).

## Текущий solver baseline: Union-v2

Union-v2 объединяет raw d64 и full-resolution Twin candidate views, после чего
bidirectional row/incoming-column OT-reranker выбирает отношения. Frozen
source-disjoint fresh64 без retrain дал:

| Метрика | Raw d64 | Union-v2 | Delta |
|---|---:|---:|---:|
| exact tiles/board | `0.9375` | `1.28125` | `+0.34375` |
| adjacency | `13.6676%` | `14.4192%` | `+0.7515 pp` |
| correct fixed top144/board | — | — | `+5.265625` |

95% CI для exact пересёк ноль, но CI adjacency был положительным; все `128/128`
горизонтальные/вертикальные layouts были strict. Источники:

- config: `configs/raw_twin_union_reranker_v2_preregistered.json`, SHA-256
  `6741e92e832a630f1b83bde6edc8a341a348f52daa82313c40a8f32c7c1173d4`;
- checkpoint:
  `outputs/raw-twin-union-reranker/v2-fit256-s400-eval24/raw-twin-union-reranker-v2.pt`,
  SHA-256 `a5f882ab3c827e4e3779be3372c62d2a8fb9cd95d3558fd30cc566a9c3137f79`;
- fresh64 report:
  `outputs/raw-twin-union-reranker/frozen-v2-fresh64-draw0/report.json`, SHA-256
  `c4ae8cb6fff97cc5a2901f922273e0702db373e25eb03986dd8af089582d04f7`;
- полный отчёт: [raw-twin-union-reranker.md](experiments/raw-twin-union-reranker.md).

## Новая точная диагностика bottleneck-а

В конце цикла добавлены:

- `outputs/raw-twin-union-reranker/frozen-v2-fresh64-draw0/exact-bottleneck-oracle-v1.json`,
  SHA-256 `501ee494a805498126f4b8c4c4677f0f13b4b4a7ee02b21cc2f37048635b41d2`;
- воспроизводящий скрипт:
  `scripts/diagnose_union_v2_exact_bottlenecks.py`;
- script прошёл Ruff и точно воспроизвёл frozen current exact `1.28125`.

Диагностика target-assisted и является только oracle-анализом, не inference
кандидатом. Competition test и organizer target pixels не открывались.

Ключевые результаты fresh64:

| Диагностика | Результат |
|---|---:|
| current exact | `1.281/576` |
| лучший общий cyclic origin | `15.391/576` |
| independent translation oracle текущих компонентов | `419.594/576` |
| тот же oracle с D4 координатами | `422.594/576` |
| выигрыш D4 сверх translation | только `+3.000` tiles/board |
| internal-geometry loss | `156.406` tiles/board |
| relative component placement gap | `404.703` tiles/board |

Вывод: origin важен и способен вернуть около `+13.94` tiles/board, но главный
разрыв — неправильное относительное размещение множества компонентов. Пытаться
исправить только общий cyclic shift недостаточно.

Budget sweep нашёл полезное колено на `48` hard edges на ось:

- hard-edge precision `79.753%`;
- independent-translation oracle `554.828/576 = 96.324%`;
- pure nontrivial support `93.125` tiles/board;
- largest component mean `9.47`;
- около `482` компонентов остаются доступными для совместной синхронизации.

При этом oracle-filtered learned top-5 relations образуют giant component в
среднем из `161.17` tiles и дают `406.97` правильных edges среди `413.09` tiles
в nontrivial components. Значит, корректные отношения часто уже есть в supply;
текущий decoder плохо выбирает совместно согласованное подмножество.

## Иерархия целей и метрик

Пользователь подтвердил, что сейчас работа ведётся над solver-ом, а restoration
и leaderboard не должны отвлекать от локального сигнала. При возобновлении
использовать такую иерархию:

1. True-neighbor R@1/R@5 и candidate coverage.
2. Precision/coverage уверенных edges, correct top144 и adjacency.
3. Размер и purity translation-consistent components.
4. Exact absolute tiles/board — конечная главная solver-метрика.
5. Только после promotion solver-а — same-tail raw-layout SSIM и полный SSIM.

Если pair signal пока слаб, улучшать matcher и graph selection. Когда появятся
чистые крупные компоненты, переходить к их global synchronization/origin.
Семантические эвристики вроде «уверенное лицо/человека ближе к центру, ровный
фон ближе к краю» допустимы как слабый tie-breaker только после появления
чистого компонента; применять их к шумным одиночным tiles преждевременно.

## Главная точка возобновления

### 1. Top48-fragment robust 2D coordinate synchronization

Это теперь наиболее прямой и дешёвый high-signal эксперимент.

1. Сделать только top-48 projected edges на каждую ось необратимыми жёсткими
   constraints: они формируют маленькие высокоточные rigid fragments.
2. Все остальные raw32/twin32 Union candidates оставить обратимыми soft
   equations между translations двух fragments; агрегировать одинаковые
   component-pair displacement hypotheses.
3. Совместно решить обе координаты на `Z24²` через robust max-consensus и cycle
   consistency. Не делать greedy merge и не предсказывать absolute coordinate
   по изолированному tile.
4. Превратить synchronized offsets в tile-to-slot unary, затем применить global
   linear assignment и существующий bounded pair-energy polish.
5. Только общий origin выбирать frozen border5 scorer-ом.

Отличие от уже проваленных веток: это не absolute component head и не greedy
relation forest. Слабые edges остаются reversible до совместного решения циклов.

Первый bounded gate: одна заранее фиксированная реализация на source-disjoint64,
без sweep. Продолжать при exact `>= +0.25` tile/board и неотрицательной adjacency
относительно того же Union-v2 decoder144; иначе остановить формулировку.

### 2. Union-v3 с multimodal candidate supply

Если coordinate synchronization упирается в отсутствующие правильные
отношения, расширить успешный Union-v2, а не строить standalone scorer:

- raw + Twin оставить основой;
- добавить full-resolution restored/descriptor/contour/reliability views;
- выбирать views контекстным reranker-ом, не fixed score fusion;
- сначала провести дешёвый exact16 oracle-supply gate;
- обучать только если pooled true-edge coverage растёт хотя бы на `+2 pp` и
  доступных correct top144 становится хотя бы на `+2`/board больше Union-v2.

Полезный checkpoint full-resolution denoiser:
`outputs/fullres-boundary-denoiser/pilot-train32-s400-eval16-auto/fullres_boundary_denoiser.pt`,
SHA-256 `a6dfc3e264e97d93ad678f3ee97e070067357c2a6f6875e7b7432f880aa1492c`.
Standalone restored scorer ухудшал R@1/R@5, но raw∪restored top32 supply рос на
`+4.806 pp`, descriptor supply — ещё на `+2.072 pp`. Это auxiliary view, а не
replacement scorer.

### 3. Iterative verify → merge → re-encode

Альтернативный consumer хорошего graph signal:

- seed из высокоуверенных Union edges;
- collision/cycle checks;
- re-render/re-encode открытых границ собранного компонента;
- повторная оценка конкурирующих merges;
- итерация до стабилизации.

Gate на exact16: accepted-edge precision `>=0.90`, минимум `16` полезных
merges/board, медианный largest exact-pure component растёт хотя бы вдвое без
падения purity.

### 4. Component-support cyclic roll ranker

После улучшения components ранжировать все 576 cyclic rolls по сохраняемым
translation-consistent components, их purity/size, разрушенным уверенным edges
и border evidence. Учить listwise против exact-support каждого roll, а не
whole-image CNN или marginal frame classifier.

Gate exact16: dominant roll R@5 выше uniform, exact `>=+0.1` tile/board против
cyclic5, adjacency loss не хуже `-0.2 pp`.

### 5. Legal layout portfolio selector

Использовать только если oracle среди уже рассчитанных legal layouts даёт
`>=+1` exact tile/board к фиксированному лучшему layout. Candidate roster:
raw Socket, Union-v2, direct-hard, relation forest/rolls. Selector должен вернуть
не менее 40% oracle gain и `>=+0.25` exact tile/board на fresh gate.

### 6. Долгий путь при наличии более сильного compute

Corruption-invariant full-resolution side encoder или большой transformer имеет
смысл только с новым target/representation: ordered boundary positions,
reliability/keypoint selection, raw skip, board-level hard negatives и
многообразные легальные corruption augmentations. Capacity sweep сам по себе
уже проверялся и не является достаточной гипотезой. До global decoder-а модель
должна дать хотя бы `+3 pp` high-confidence precision или `+1` correct
top32/board.

## Что не повторять без новой информации

Полный evidence ledger находится в
[layout-sorter-ledger.md](prior-research/layout-sorter-ledger.md). Короткий
no-repeat список:

- standalone absolute-coordinate / Set-to-Grid / Hungarian heads;
- isolated-tile population, DINO, centre/background absolute priors;
- larger component absolute/shift head;
- marginal frame classifier и whole-layout cyclic-origin CNN;
- BorderPointer free-run/rescue;
- SocketPermutationFlow/diffusion поверх слабых scores;
- текущий Sparse BorderGraph-QAP;
- relation cap/bonus sweeps и fixed raw/restored score fusion;
- новый global solver на неизменённых слабых score matrices;
- GANzzle latent canvas as-is: прошлый G3 после 600 шагов дал placement top1
  лишь около `0.2–0.3%`, top20 `3.7%`;
- большой transformer/capacity sweep без новой задачи обучения;
- downsampling U-Net как default matcher на `20×20`: после нескольких stride
  уровней spatial phase слишком грубая. Предпочтителен stride-1 full-resolution
  путь с raw skip.

Отдельно: ветка `pasha883` не содержала solver с подтверждённым R@1 `0.4`.
Аудит нашёл local R@1 около `19.68%`, но его buddies96 adjacency/глобальная
конверсия были хуже; подробности в
[pasha883-pairwise-audit.md](experiments/pasha883-pairwise-audit.md).

## Правила следующего рабочего цикла

- Не пересчитывать неизменившиеся артефакты и не собирать один submission дважды.
- Сначала задавать чувствительный discovery gate, затем bounded exact pilot и
  только после него fresh confirmation.
- Положительный локальный сигнал не отбрасывать из-за слишком жёсткого раннего
  порога, но не путать candidate supply с готовым solver gain.
- CPU и GPU использовать параллельно для реально независимых полезных задач:
  например, GPU training/inference и CPU decoder/oracle/feature-cache. Не
  создавать искусственную нагрузку ради температуры или utilisation.
- На MPS не требовать bitwise replay от reduction-based decoder. Для нового
  production либо переносить grouped reductions/всю inference на CPU, либо
  сохранять и валидировать frozen layouts с полным provenance.
- Для честного leaderboard A/B менять только layout, оставляя один и тот же
  frozen pixel tail. Публиковать exact, raw-layout SSIM, same-tail SSIM и paired
  bootstrap.
- Общаться коротко и по фактическим новым результатам; не устраивать повторную
  «перепроверку», когда входы не менялись.

## Restoration остаётся отдельным треком

Пока solver исследуется по pair/adjacency/exact, лучший `0.2762279` submission
заморожен. Denoise можно использовать до matcher-а как дополнительный view и
после строгой сборки как bounded tail. Не выводить denoised/generated tile
вместо соответствующего исходного фрагмента без отдельной manual-compliance
проверки.

Последний official `0.2420` не позволяет сравнить layout-ы из-за разных tails.
Следующий честный end-to-end тест — buddies96 и новый solver через идентичный
frozen tail; до появления solver signal этот тест не приоритетен.

## Быстрое возобновление

1. Прочитать этот handoff, затем
   [raw-twin-union-reranker.md](experiments/raw-twin-union-reranker.md) и новый
   oracle JSON.
2. Реализовать единственный no-sweep top48 `Z24²` synchronization pilot.
3. Прогнать source-disjoint64 gate: exact delta `>=+0.25`, adjacency delta
   `>=0` против Union-v2 decoder144.
4. При pass — fresh confirmation; при fail — Union-v3 oracle-supply gate.
5. Не трогать competition test и лучший submission до promotion.

## Отложенное напоминание

Пользователь попросил после полного завершения всей работы над проектом
напомнить ему рассказать про **Veko.ai**. Сейчас ничего устанавливать или
настраивать не нужно. Текущий solver-цикл завершён, но общий проект и roadmap
ещё существуют, поэтому напоминание сохранено здесь и должно прозвучать только
при настоящем финальном закрытии проекта.
