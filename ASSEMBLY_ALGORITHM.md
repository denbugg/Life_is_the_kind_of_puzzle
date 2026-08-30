# Алгоритм сборки пазла — без реставрации

Дата фиксации: 2026-08-29.

## Решение в одном абзаце

Локальные стыки не дают достаточно рёбер для связной сборки всей доски: первые
примерно 150 рёбер почти чистые, но на объёме, нужном для перколяции, точность
падает примерно до 0.67, а одно ложное ребро «сваривает» два правильных острова
в неверном относительном сдвиге. Поэтому основной solver должен получать
**абсолютную систему координат** не из роста графа, а из условной гипотезы всей
фотографии. Модель по unordered bag из 576 тайлов генерирует несколько грубых
карт 96×96; для каждой карты Hungarian решает полную биекцию «тайл → клетка».
Только самый надёжный глобальный хвост seam-рёбер фиксируется как набор жёстких
маленьких островов. Beam выбирает их абсолютные сдвиги без пересечений, после
чего Hungarian размещает все остальные тайлы. Seam objective используется
только для выбора из небольшого числа уже полных гипотез, но не оптимизируется
по свободному пространству перестановок.

Реализация:

- `src/field_diffusion.py` — генеративная coarse-map модель;
- `src/train_field_diffusion.py` — обучение и честная held-out оценка;
- `src/field_solver.py` — Hungarian, rigid-island beam и выбор гипотез;
- `src/eval_field_solver.py` — oracle/tolerance gate;
- `src/infer_field_layout.py` — полный solve-only inference;
- `tests/test_field_solver.py` — инвариантность, биективность и геометрия.

## Что показали все поколения экспериментов

В репозитории проведено более 470 нумерованных экспериментов. Ниже не выборка
«удачных запусков», а карта выводов по всем семействам; точные строки и числа
сохранены в `autoresearch-runs/pazzle-mgc-restoration-20260818/EXPERIMENTS.md`.

### 1. PairwiseNet, MGC и локальные seam-модели

- Старый PairwiseNet имел `acc@48=0.477`, но полный solve давал placement около
  0.0015 и solve-SSIM около 0.106. Sampled candidate accuracy не была метрикой
  полного графа.
- На чистых тайлах MGC действительно решает задачу: placement 0.9965. На
  реальных повреждённых тайлах он падает практически до случайного уровня.
- Learned seam embed поднял R@1 примерно с 0.056 до 0.24–0.32, но этого
  недостаточно для связной доски.
- Вся полезная информация о соседстве живёт примерно в четырёх колонках у шва;
  чтение целого фрагмента хуже. Это около 80 RGB-пикселей при очень низком SNR.
- Capacity, width, depth, дополнительные views, hard negatives, listwise loss,
  chooser, joint verifier, MGC fusion, chroma и hand-crafted features дали
  малые улучшения там, где нужен скачок примерно на 0.30 precision.

Вывод: ещё один локальный edge-ranker не является новой алгоритмической веткой.

### 2. Greedy, SA, LP, BP, pose graph и острова

- Greedy/component growth работает на чистых оценках и каскадно ломается от
  ложного ребра.
- Старый `place_acc≈0` частично скрывал torus-origin bug: идеальная относительная
  раскладка могла быть циклически сдвинута и получить ноль. После исправления
  исхода основной недостаток всё равно остался.
- Weighted-L1 LP — лучший из проверенных глобальных solver-ов, когда edge
  precision около 0.9 и выше. На реальных score он остаётся у chance.
- Loopy BP, simulated annealing, path cover, row-first beam, merge by paths,
  square loops, component caps, consensus и hole filling не восстанавливают
  отсутствующую информацию.
- Критическая кривая M456: при фиксированных правильных рёбрах connected block
  примерно 350 при precision 1.00, 186 при 0.99, 65 при 0.95 и 25 при 0.90.
  На рабочем уровне кривая почти плоская.
- Самые сильные 127 пар имеют около 0.985 precision. Это хороший набор жёстких
  локальных ограничений, но плохое зерно для роста: он покрывает лишь небольшие
  разрозненные острова.

Вывод: надёжный остров — constraint, а не seed для добавления более слабых рёбер.

### 3. Абсолютное положение

- Generic scene prior, border prior, coarse colour field, learned coordinate,
  JPEG phase, lens cues и logits iterative assembler не дают пригодной полной
  системы координат.
- Правильное размещение крупнейшего компонента само по себе было оценено как
  примерно +0.0195 SSIM, но border cue редко видит нужную рамку.
- Итеративный assembler хорошо размещает тайл, когда соседи уже раскрыты, но не
  умеет начать с пустой доски — bootstrap wall.

Вывод: абсолютный сигнал должен зависеть от **этого мешка и этой фотографии**, а
не быть средним prior-ом по датасету.

### 4. Content assignment меняет постановку

- Если известна чистая фотография в разрешении всего 4×4 значения на тайл
  (96×96 на всю доску), обычный Hungarian по содержимому даёт около 0.429 SSIM.
- Другая фотография и средняя фотография дают chance: generic prior бесполезен.
- MSE-регрессия coarse field коллапсирует к почти постоянной карте. Это ожидаемый
  conditional mean, а assignment требует sharp posterior sample.
- M471 показал, что карта этой фотографии с ошибкой порядка 32 grey levels всё
  ещё размещает около 210 тайлов в точную клетку и может перейти 0.39 SSIM.

Вывод: оставшийся честный путь — bag-conditioned generative posterior, а не
детерминированная регрессия среднего изображения.

## Точный алгоритм

### Offline

1. Для каждой train-доски восстановить соответствие входных тайлов target
   только для формирования supervision. Split остаётся source-disjoint:
   последние 300 досок не участвуют в fitting.
2. Представить каждый тайл unordered token-ом: pooled 8×8 RGB плюс mean/std.
   Никакой positional encoding к токену добавлять нельзя.
3. Target — clean photograph 96×96, то есть 4×4 на каждую клетку.
4. Обучить conditional diffusion/flow sampler. Обязательный control — та же
   архитектура в режиме one-shot MSE regression.
5. Выбирать checkpoint по held-out **placement после Hungarian**, а не по
   diffusion loss и не по RMSE в одиночку.

### Inference одной доски

1. Разрезать вход 480×480 на 576 исходных повреждённых тайлов 20×20. Порядок
   остаётся входным; модель множества инвариантна к нему.
2. Сгенерировать `K=8..32` sharp coarse hypotheses 96×96.
3. Для каждой гипотезы построить матрицу `C[cell,tile]`:
   - `raw` squared distance — безопасный режим для неточной карты;
   - `zscore` — убирает независимые brightness/contrast и имеет намного более
     высокий ceiling, но включается только после held-out A/B;
   - `blend:X` — промежуточный валидируемый режим.
4. Если seam roster доступен, взять только глобально лучшие примерно 127
   направленных пар по fused score. Conflict-safe DSU превращает их в rigid
   islands. Нельзя добирать по одному лучшему ребру от каждого тайла.
5. Для каждого острова перечислить лучшие абсолютные сдвиги под `C`. Bounded
   beam выбирает непересекающиеся сдвиги всех островов.
6. Все оставшиеся клетки и тайлы назначить одним Hungarian. На выходе всегда
   ровно одна полная перестановка 576 элементов.
7. Повторить для всех coarse hypotheses. Выбрать одну из них по assignment
   cost; realised seam energy допустим только как слабый tie-break внутри этого
   конечного набора.
8. Собрать PNG простой перестановкой исходных тайлов. Никакого denoise,
   deblock, brightness levelling или подмены пикселей на этом этапе нет.

## Уже пройденные проверки реализации

`tests/test_field_solver.py` проверяет:

- permutation equivariance bag encoder-а;
- точное восстановление синтетической перестановки;
- полную биекцию 576↔576;
- инвариантность zscore assignment к affine photometry;
- соблюдение rigid-island геометрии;
- отклонение конфликтующих компонентов;
- выбор seam-score только между готовыми гипотезами.

Smoke run на реальном `field_cache.npz` и RTX 2070 прошёл. Нулевая модель дала
ожидаемые `1/576`, а oracle-карта на той же доске — 0.616 exact placement.

Oracle/tolerance gate на восьми held-out досках:

| ошибка coarse map | raw placement | zscore placement | лучший blend |
|---:|---:|---:|---:|
| 0 | 0.4507 | 0.8077 | 0.6871 (`blend:0.5`) |
| ~15.6 RMSE | 0.4089 | 0.6035 | 0.5612 |
| ~29.9 RMSE | 0.3459 | 0.4453 | 0.4366 |
| ~55.1 RMSE | 0.2220 | 0.2075 | 0.2454 |

Это tolerance diagnostic с iid noise, не оценка обученной модели. Ошибка
реального sampler-а коррелирована и может быть существенно вреднее при том же
RMSE. Поэтому единственный go/no-go — downstream placement.

Paired A/B жёсткого 127-edge tail на тех же восьми досках:

| coarse map / descriptor | без островов | с островами | delta | paired SE | знак |
|---|---:|---:|---:|---:|---:|
| oracle / raw | 0.4490 | 0.4763 | +0.0273 | 0.0094 | 6/2 |
| oracle / zscore | 0.8114 | 0.7951 | −0.0163 | 0.0108 | 3/4 |
| RMSE≈30 / raw | 0.3537 | 0.4091 | **+0.0553** | 0.0088 | 8/0 |
| RMSE≈30 / zscore | 0.4440 | 0.4703 | **+0.0263** | 0.0048 | 8/0 |

Острова помогают именно в реалистичном неточном режиме и должны отключаться,
когда absolute field уже почти совершенен: несколько ложных seam-рёбер вреднее
почти точного zscore assignment. Это делает качество поля частью routing rule,
а не просто метрикой checkpoint-а.

### G2 pilot: обычный image DDPM не прошёл

Одинаковый малый бюджет был дан regression control и DDPM, затем DDPM получил
расширенный pilot: 512 FIT-досок, 2560 optimizer steps, 8 held-out досок.

| arm | RMSE | map spread | best placement | bag delta | wrong-bag placement |
|---|---:|---:|---:|---:|---:|
| regression control | ~63 | 6–8 | 0.0026 | — | — |
| DDPM, ранний | ~138 | ~117 | 0.0030 | 3.72 | ~chance |
| DDPM, 2560 steps | ~95 | ~62 | 0.0024 | 11.66 | ~chance |

`bag delta` — RMSE между двумя samples с одинаковым starting noise, но correct
и wrong bag. Conditioning постепенно влияет на пиксели, однако не влияет на
раскладку: correct-bag и wrong-bag placement совпадают. End-to-end evaluation
50-step sampler-а: free placement 0.0017 / SSIM 0.0953, late-snapped placement
0.0015 / SSIM 0.0963, flat fill 0.3514.

**Вердикт:** масштабировать обычный DDPM запрещено. Он учит unconditional image
prior быстрее, чем зависимость конкретной фотографии от bag. Следующий вариант
должен на малом бюджете пройти дополнительный causal gate: при одинаковом noisy
state correct bag обязан давать лучший x0/assignment, чем wrong bag. Практически
это означает обучение преимущественно на high-noise timesteps, где `x_t` не
может сам раскрыть фотографию, плюс явный correct-vs-wrong-bag objective. Если
downstream placement снова не отрывается от wrong-bag control, вся generative
field ветка закрывается, а не масштабируется.

## Gates и stop conditions

1. **G0 correctness — PASS.** Все unit tests и smoke inference проходят;
   layout всегда bijective.
2. **G1 oracle/tolerance — PASS.** Интерфейс field→assignment имеет достаточный
   ceiling и выдерживает около 30 RMSE.
3. **G2 generative mechanism — FAIL для обычного DDPM.** Он не превзошёл
   regression control по held-out placement; падение loss/RMSE не конвертируется.
   Повтор возможен только с high-noise + wrong-bag causal objective и тем же
   малым gate, не как более крупный запуск прежней формулировки.
4. **G3 learned field.** Первый полезный рубеж — held-out placement ≥0.10;
   целевой рубеж сборки — ≥0.30. RMSE без placement не открывает следующий gate.
5. **G4 rigid islands — mechanism PASS, deployment pending.** На oracle/noisy
   fields paired A/B положителен в рабочем режиме, но включение на learned field
   всё равно требует отдельного paired gate. Precision из M449 не заменяет его.
6. **G5 posterior selection.** `K>1` должно побить fixed single-sample control.
   Если selector выбирает хуже постоянного sample/seed, оставить `K=1`.
7. До прохождения G3 не обучать restorer и не оценивать restoration variants.

## Команды

```powershell
# Проверка assignment ceiling
python src/eval_field_solver.py --boards 8 --noise 0 16 32 64 --modes raw zscore blend:0.25 blend:0.5

# Regression control
python src/train_field_diffusion.py --mode regress --epochs 60 --out field_regress.pt

# Conditional sampler
python src/train_field_diffusion.py --mode diffusion --epochs 60 --out field_diff.pt

# Solve-only held-out gate, без seam constraints
python src/infer_field_layout.py --checkpoint field_diff.pt --samples 8 --validate 24 --mode raw

# Только после предыдущего gate: A/B с чистым seam tail
python src/infer_field_layout.py --checkpoint field_diff.pt --samples 8 --validate 24 --mode raw `
  --matchers seam_embed_v3.pt seam_embed_local.pt seam_embed_wide.pt --edge-keep 127
```

## Что не делать до нового доказательства

- Не обучать очередной pairwise scorer на тех же seam pixels без нового
  источника информации. Допустимое исключение — listwise выбор между уже
  найденными top-k кандидатами: он решает другую задачу и проходит отдельный
  end-to-end gate.
- Не наращивать острова рёбрами ниже измеренного high-precision tail.
- Не оптимизировать глобально seam/TV objective по свободным перестановкам.
- Не считать adjacency заменой absolute placement: текущий pipeline имеет
  adjacency около 0.27 при placement около 0.01.
- Не подмешивать реставрацию в solver gate. Сначала должна двигаться раскладка.

## Дискретный curriculum и строгая реставрация (29 августа)

После G2 был добавлен второй, полностью дискретный путь:

1. `SinkhornAssembler` предсказывает двустохастическую перестановку, а не
   координаты или пиксели.
2. Два замороженных full-resolution seam-энкодера дают directed right/down
   матрицы по исходным 20×20 тайлам.
3. Из глобального score-tail строятся только conflict-safe rigid islands.
4. `island_field_decoder.py` размещает острова под абсолютным unary и завершает
   остаток одним Hungarian; свободной оптимизации seam objective нет.
5. `frame_classifier.py` учит bag-relative вероятность отсутствующего соседа.

Held-out результаты curriculum:

| размер | chance placement | base placement | islands placement | adjacency |
|---:|---:|---:|---:|---:|
| 6×6 | 0.0278 | 0.0673 | **0.2222** | **0.3768** |
| 12×12 | 0.0069 | 0.0113 | **0.0625** | **0.3278** |

На 12×12 frame classifier дал отдельный подтверждённый lift при неизменной
геометрии островов: `0.0308 → 0.0565` placement на 64 новых досках. На полной
24×24 доске этот prior не прошёл paired end-to-end gate внутри зрелого packer-а
и поэтому по умолчанию выключен. Базовый полный conformant pipeline
harvest/quad/component search с joint hinge-verifier на 16 held-out досках дал
placement `0.0135`, adjacency `0.2650`.

Отдельно проверен буквальный `row→column` decoder: 24 направленные цепочки по
24 тайла, затем упорядочивание цепочек по поперечным стыкам. В своей первой
оси он достигал примерно `0.30` правильных соседств, но целые строки нельзя
надёжно совместить по второй оси: итоговая adjacency `0.152`, а после QAP
repair — `0.139`. В production decoder не включён.

Joint verifier не меняет набор harvested-рёбер и не касается изображения. Он
видит обе стороны конкретного шва и шесть относительных score-признаков, после
чего только переупорядочивает рёбра перед conflict-safe сборкой компонентов.
На одинаковых 16 досках hinge-вариант дал `0.0099→0.0135` placement и
`0.261→0.265` adjacency; BCE вернул `0.0093/0.261`, а rank-ensemble hinge+focal
дал `0.0095/0.265` и был отклонён. Поэтому default — только `verify_hinge.pt`.

### Полный top-5 curriculum (30 августа)

Для всех 7 000 доступных досок построен воспроизводимый memmap-cache кандидатов
двух замороженных seam-моделей. В cache лежат только индексы и scores
повреждённых входных тайлов; clean pixels в него не записываются. Разделение
фиксировано заранее: 6 700 train и последние 300 validation. Проверка целостности:

- self-candidates: `0`;
- ошибочных boundary labels: `0`;
- recall истинного соседа: top-1 `0.30453`, top-5 `0.46694`;
- доступных истинных направленных связей: `515.51` на доску.

Listwise chooser для каждого anchor выбирает один из пяти кандидатов либо
воздерживается. Он не синтезирует пиксели и не ищет исходники; результатом
остаются только directed adjacency hypotheses. Обучение на 6 700 досках с
`none_weight=0` и отдельными checkpoint-ами по correct-bonds и precision@430
дало на фиксированных 96 validation-досках:

- matcher: `350.33` истинных связей/доску;
- chooser best-bonds: `357.84` (`+7.51`);
- chooser best precision@430: `0.65618`.

Оба checkpoint-а затем прошли одинаковый строгий end-to-end gate. На 64 полных
held-out досках best-bonds выиграл у best-precision по всем главным метрикам:

| checkpoint | placement | adjacency | strict SSIM |
|---|---:|---:|---:|
| best-precision | 0.0177 | 0.2870 | 0.2647 |
| **best-bonds** | **0.0251** | **0.2890** | **0.2720** |

Поэтому production-default — `choose5_full_none0_best_bonds.pt` вместе с
`verify_hinge.pt`, `sel_volume=430` и seam fill.

`sel_volume=430` дополнительно проверен парным end-to-end sweep на одинаковых
32 досках. Слишком плотный граф не соединяет пазл лучше, а сваривает правильные
острова ошибочными рёбрами:

| принятых рёбер | placement | adjacency | strict SSIM |
|---:|---:|---:|---:|
| 350 | 0.0202 | 0.283 | 0.2610 |
| 390 | 0.0180 | 0.288 | 0.2610 |
| **430** | **0.0323** | **0.284** | **0.2711** |
| 470 | 0.0164 | 0.279 | 0.2581 |
| 510 | 0.0048 | 0.266 | 0.2551 |

### Жёсткая граница допустимой реставрации

Финальный PNG разрешено строить только так:

`input tiles → bijective permutation → level correction → R5 → NLM → bilateral`.

Нельзя смешивать результат с coarse field, средним изображением, flat fill или
любой другой картой пикселей. Историческая формула `field + alpha*highpass`
считается **недопустимой** и не является результатом проекта. Все её SSIM
измерения отозваны. В `infer_composed.py` ветка такой композиции удалена:
параметров управления смешиванием больше нет, а field model в стандартном
`fill=seam` режиме даже не загружается. Если `fill=field` выбран для исследования
размещения, предсказанные значения участвуют только как unary-cost дискретной
перестановки и никогда не попадают в пиксели результата.

Текущий честный restoration gate на 64 полных held-out досках:

- placement: **`0.0251`**;
- adjacency: **`0.2890`**;
- SSIM после R5→NLM→bilateral, без alpha/field: **`0.2720`**.
