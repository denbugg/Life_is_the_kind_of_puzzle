# Новый подход: рабочий журнал исследований

Дата начала: 2026-07-12

## Цель

Решить раскладку 24x24 из 576 независимо испорченных фрагментов по самим
входным данным. Поиск исходных фотографий больше не считается основной
стратегией: найденные точные оригиналы остаются бонусными заменами, но не
ответом на задачу.

## Что уже исключено

- Сравнение сырых или восстановленных границ тайлов: потолок candidate recall
  и точности недостаточен для глобальной сборки.
- Direct pose / абсолютная клетка по отдельному фрагменту: постановка не
  переносится между разными сценами.
- Обычный глобальный Sinkhorn, dense pointer и генерация canvas из мешка:
  предыдущие реализации не вышли из режима усреднения/случайной привязки.
- Pose graph, RSCM/IRLS, SA и buddies: хорошие локальные seeds не образуют
  достаточно связного правильного графа.
- Seeded QAP: честный oracle из случайной инициализации не решил даже идеальный
  граф. При 48 шагах и 256 Sinkhorn-нормализациях: placement=0.2986,
  neighbour=0.6513, row/col max=0.9968, ds_error=0.0177 при обязательном
  oracle-пороге placement/neighbour >= 0.999. На test ветка не запускалась.

## Новая гипотеза A: инверсия генератора датасета

Искажения могут содержать информацию о порядке обработки тайлов. Если
brightness/contrast/noise/blur/JPEG применялись последовательно в исходном
row-major порядке, а shuffle выполнялся позже, то оценённый вектор параметров
искажения является потенциальным скрытым кодом исходной клетки. Дополнительно
сама перестановка может повторяться или воспроизводиться из номера изображения
и состояния PRNG.

Это принципиально не seam matching: используются только артефакты процедуры
генерации и известные пары train input/target.

### Первый forensic-гейт

1. Загрузить существующий `E:/pazzle_work/cache/perms.npz` и проверить точную
   семантику отображения.
2. Измерить повторяемость перестановок между train-изображениями и зависимость
   target cell от input slot / номера файла.
3. Для сопоставленных dirty-clean тайлов оценить affine brightness/contrast,
   шум, blur и JPEG/DCT-признаки.
4. Сделать group split по изображениям. Сравнить предсказание row/column/cell
   только по forensic-признакам с шансом: cell top-1 1/576, row/column 1/24.
5. Продолжать только при переносимом сигнале: row или column accuracy >= 0.10,
   либо cell R@25 >= 0.20. Иначе гипотезу закрыть без большого обучения.

## Параллельные кандидаты

Три независимых аудита проверяют подходы, которые используют глобальный
контекст всей сцены, но не повторяют закрытые absolute-slot и seam-ветки.
После их ответа здесь будет записана выбранная архитектура и её kill gate.

## Результаты быстрых forensic-проверок

### Перестановки

- В `perms.npz` находятся 7000 различных перестановок; полных повторов нет.
- Доля одинаковых отображений в соседних изображениях: 0.001723 при случайном
  ожидании 1/576 = 0.001736.
- Лучший lookup `input slot -> target cell` на всех train даёт 0.00346 top-1.
  Это лишь finite-sample переобучение частот и не является рабочим сигналом.
- Простые варианты `RandomState`, `default_rng` и `random.shuffle` с типичными
  seed не воспроизводят первые перестановки.

Вердикт: повторяющейся или тривиально seed-зависимой перестановки не найдено.
Гипотеза A остаётся открытой только для проверки параметров самих искажений.

Финальный oracle-feature gate на 120 реальных train-парах (96 train / 24
group-held-out) также провален:

| metric | result | chance | pass threshold |
|---|---:|---:|---:|
| row top-1 | 0.0479 | 0.0417 | 0.10 |
| column top-1 | 0.0407 | 0.0417 | 0.10 |
| cell top-1 | 0.00166 | 0.00174 | — |
| cell R@25 | 0.0502 | 0.0434 | 0.20 |

Даже oracle-оценки affine/noise/blur/JPEG параметров не содержат переносимого
позиционного сигнала. **Гипотеза A закрыта.** Полный JSON:
`E:/pazzle_work/gates/generator_forensics.json`.

### Имена test

700 test-имён являются разреженным подмножеством диапазона `000000..002999`,
и для каждого существует одноимённый train target. Однако строгая проверка
первых 20 пар дала 0/20 совпадений: 0 пространственно согласованных SIFT-точек
во всех случаях. Следовательно, имя test не указывает на одноимённый target.

### Метаданные PNG

У проверенных train/test PNG нет text/EXIF-полей. Временные метки относятся к
упаковке датасета и не кодируют раскладку.

## Выбранная ветка B: Frontier Inpainting Pointer

Основная новая постановка — последовательное заполнение дыр на границе уже
правильно собранной компоненты. Для каждой пустой клетки модель получает
относительное окно 5x5 из уже размещённых тайлов и mask-токенов, строит query
ожидаемого центрального фрагмента и выбирает его среди всех ещё не
использованных tile keys.

Почему это не повтор прежних веток:

- нет абсолютного класса клетки и попытки одним шагом решить перестановку;
- score кандидата зависит от текущего двумерного окружения и пересчитывается
  после каждого нового тайла;
- используются L-образные и более крупные 2D-контексты, а не один seam или
  цепочка A-B-C;
- исходные тайлы сохраняются, генеративная часть предсказывает embedding для
  pointer, а не рисует финальную фотографию;
- rollout стартует из существующих высокоточных seed-компонент, но не считает
  их статический pose-граф достаточным решением.

### Минимальная модель

- общий dirty-tile encoder для keys;
- окно 5x5, относительные координаты, mask/occupied tokens;
- 3 небольших transformer-блока, d около 160;
- query-to-all-remaining-keys listwise CE;
- дополнительное dirty-to-clean embedding alignment и value-head для
  обнаружения испорченного контекста;
- обучение с контекстами размера 2, 4, 8, 16 и подменой одного тайла в 10-20%
  примеров.

### Information gate до написания полного assembler

- минимум 512 held-out запросов;
- context 4: all-bag R@1 >= 0.30 и R@5 >= 0.60;
- context 8: all-bag R@1 >= 0.50;
- high-confidence precision >= 0.75 при coverage >= 0.05;
- при подмене/удалении 20% контекста падение R@1 не более 0.15.

Если gate пройден, первый bounded rollout обязан дать neighbour >= 0.20,
precision первых 100 добавленных рёбер >= 0.65 и SSIM не менее чем на 0.02
выше generic baseline. До прохождения information gate test не используется.

### Фактический information gate

Модель: 1,216,354 параметра, 1200 шагов, 24 query/step, exact synthetic
dirty tiles, 4 фиксированных held-out изображения, по 512 запросов на размер
контекста. Полный отчёт:
`E:/pazzle_work/gates/frontier_pointer_gate.json`.

| context | final R@1 | final R@5 | required R@1 | required R@5 |
|---:|---:|---:|---:|---:|
| 2 | 0.0059 | 0.0645 | — | — |
| 4 | **0.0234** | **0.0762** | 0.30 | 0.60 |
| 8 | **0.0215** | **0.0898** | 0.50 | — |

High-confidence coverage при p>=0.5 остался нулевым. Контекст даёт заметный
рост над случайным R@5, но разрыв до rollout-gate слишком велик. Увеличивать
обучение в десятки раз без изменения источника информации не оправдано.
**Ветка B закрыта до full assembler и не запускалась на test.**

## Новая гипотеза C: модульная координата из JPEG/resampling phase

До индивидуальной порчи исходная фотография, вероятно, уже несёт периодические
JPEG или resize-артефакты. Шаг тайла 20 px меняет фазу исходной решётки:

- 20 mod 8 = 4, поэтому JPEG 8x8 phase чередуется по parity row/column;
- 20 mod 16 = 4, поэтому chroma/MCU phase потенциально кодирует row/column
  modulo 4.

Вместо слабой 576-классовой absolute-pose задачи надо отдельно предсказывать
row/column residues modulo 2 и 4 по профилям высокочастотных разрывов внутри
тайла. Эти головы делят данные между всеми клетками с одинаковой фазой и могут
вытащить сигнал, который теряется в абсолютной классификации.

Первый gate будет train-only и group-held-out:

1. exposure-normalized horizontal/vertical derivative profiles;
2. blockiness scores для всех фаз 0..7 и 0..15, отдельно luma/chroma;
3. классификация row/column mod 2 и mod 4 на clean и на реально dirty тайлах;
4. рабочий сигнал: dirty mod2 accuracy >=0.60 или dirty mod4 >=0.35.

При успехе residue log-probabilities станут cell constraints в Hungarian и
будут объединены с оставшимися относительными seeds. При провале гипотеза
закрывается за один CPU-прогон.

### Фактический phase gate

Group split: 96 train / 24 held-out изображения, `conf>=0.70`, 228
content-minimal high-frequency признаков, независимые ExtraTrees для clean и
dirty. Полный отчёт: `E:/pazzle_work/gates/jpeg_phase_gate.json`.

| branch | row mod2 | col mod2 | row mod4 | col mod4 |
|---|---:|---:|---:|---:|
| dirty | 0.4985 | 0.5024 | 0.2484 | 0.2496 |
| clean | 0.4964 | 0.5053 | 0.2505 | 0.2533 |
| chance | 0.5000 | 0.5000 | 0.2500 | 0.2500 |

Даже clean originals не содержат общей переносимой JPEG/resampling phase.
Равномерные feature importances подтверждают отсутствие скрытого признака.
**Гипотеза C закрыта.**

## Следующая ветка D: learned global decoder мягкого oriented graph

Единственный подтверждённый общий сигнал после всех гейтов — candidate union
содержит примерно 67-69% истинных прямых соседей, хотя top-1 и reciprocal
срезы слишком редки. Новый solver должен получать весь ориентированный
взвешенный граф и учиться декодировать из него структуру 24x24, не сводя его
заранее к фиксированным ошибочным рёбрам.

Минимальный безопасный порядок:

1. Procedural oracle: случайно перенумерованный точный directed grid.
2. Procedural noisy graph с эмпирическими recall/false-candidate параметрами.
3. Только при прохождении первых двух — кэш нескольких настоящих ranker-графов
   на E и один held-out gate.

Задача первого этапа — проверить саму возможность learned grid decoding за
минуты, не тратя часы на извлечение реальных графов.

### Два независимых procedural decoder-а

1. **Directional GraphGRU.** Случайно перенумерованный grid, sparse R/D
   candidate lists, 24 общих message-passing раунда от границ, row/column
   heads и Hungarian. Обучение только на процедурных exact/noisy graphs;
   отсутствие node-ID и новая нумерация каждого batch исключают запоминание.
2. **Commuting Nilpotent Shift Decoder (CNSD).** Одновременно восстанавливает
   два частичных оператора-сдвига R/D. Ограничения: `RD≈DR`, 24 цепочки длины
   24, `R^24=D^24=0`, ровно 24 boundary входов/выходов и одна вершина на каждую
   пару boundary-distance coordinates. Это рассматривает grid как 2D
   error-correcting code и может алгебраически достраивать true edges,
   отсутствующие в candidate pool.

Оба варианта сначала проходят только procedural exact/noisy gate. Реальные
ranker-графы не извлекаются, пока хотя бы один decoder не докажет, что способен
стартовать с полностью случайной нумерации при recall около 0.67.

### GraphGRU procedural result

Конфигурация: width 48, 24 shared rounds, K=16, batch 2, 150 exact + 350
noisy curriculum steps. Отчёт:
`E:/pazzle_work/gates/graph_grid_decoder_gate.json`.

| graph | placement | neighbour | required neighbour |
|---|---:|---:|---:|
| exact | 0.1126 | 0.0796 | 0.995 |
| recall 0.67 | 0.00087 | 0.00159 | 0.15 |
| recall 0.50 | 0.00195 | 0.00181 | 0.08 |
| coherent decoy | 0.00152 | 0.00260 | 0.08 |
| shuffled control | 0.00304 | 0.00125 | <=0.02 |

При переходе к noisy graph row/column CE немедленно вернулся к `log(24) =
3.178` и остался там. Weighted message averaging размывает редкое true edge
среди K кандидатов вместо дискретного глобального выбора. **GraphGRU закрыт;
реальные ranker-графы не извлекались.**

### CNSD procedural result

Grid-4 algebra/gradient smoke прошёл, затем exact grid-24 был восстановлен
идеально за 120 шагов:

- placement = 1.000;
- neighbour = 1.000;
- commutation/occupancy/tile-mass = 0;
- Sinkhorn row error = 0.

Отчёт: `E:/pazzle_work/gates/cnsd_exact_gate.json`.

На корректно калиброванном noisy graph (`candidate_recall=0.6784`, conditional
R@1=0.2758) исходные 16 Sinkhorn rounds дали row error=1.0 и случайную
раскладку. Единственный corrective repeat с 128 rounds и T_end=0.20 улучшил
численную проекцию, но не решил задачу:

| metric | result | continue threshold |
|---|---:|---:|
| Sinkhorn row error | 0.0726 | <=0.02 |
| placement | 0.00694 | 0.25 |
| neighbour | 0.00543 | 0.25 |
| correct edges | 6 / 1104 | — |

Отчёт: `E:/pazzle_work/gates/cnsd_gate_corrective.json`.

Exact-алгебра верна, но rank-noisy objective не выбирает planted shifts и не
сходится к требуемой субперестановке. Дальнейший подбор temperature/lambda/seed
запрещён гейтом. **CNSD закрыт, real ranker graph и test не использовались.**

## Ветка E: факторизованный 2D-hole scoring

Frontier Pointer одновременно учил три сложные вещи одним listwise CE:
устойчивый dirty-tile embedding, модель двумерного продолжения сцены и поиск
среди 576 кандидатов. Новый тест разделяет их:

1. dirty→clean tile representation обучается прямым paired contrastive loss;
2. clean masked-halo model предсказывает embedding центрального тайла по
   2/4/8 соседям;
3. только после отдельных oracle-гейтов обе части соединяются: predicted clean
   query ранжирует dirty candidate keys.

Это проверит, был ли провал Pointer отсутствием 2D-информации или лишь
необученным bottleneck-представлением. До прохождения clean-halo и paired
alignment gates полный rollout не пишется.

### Отвергнутые альтернативы этой итерации

- Raw permutation diffusion из полностью случайной перестановки: при t=1
  первый reverse-step снова обязан решить провалившуюся one-shot absolute-slot
  задачу. Diffusion не создаёт нового сигнала.
- Graph-conditioned permutation diffusion оставлен резервом, но он зависит от
  того же candidate-графа с recall около 0.67 и требует 2-5 часов на честный
  MVP. Frontier gate дешевле и проверяет ранее не использованный 2D-контекст.
- Глобальный VAE/latent canvas + OT: свежий oracle latent остаётся около
  случайного уровня; lossy canvas не годится как якорь для EM.

## Реализация ветки E, этап 1: paired alignment (dirty -> clean identity)

Вопрос, который ни одна прежняя ветка не проверяла напрямую: после ОДНОГО
независимого применения деградации (affine + шум sigma 40-55 + blur + JPEG
35-50) содержимое тайла всё ещё указывает, из какого именно чистого патча
оно получено — среди ~576 визуально похожих кандидатов той же фотографии?
Это не seam-matching (сравнение с соседями) и не абсолютная позиция — это
проверка того, выживает ли идентичность тайла как таковая.

Реализация: `src/eval_paired_alignment.py`. Два независимых `TileEncoder`
(dirty/clean), symmetric InfoNCE (CLIP-style) на точных synthetic парах
`(dirty_i, clean_i)`; позиция никогда не подаётся на вход, поэтому shuffle
не имеет значения для самого гейта. Held-out метрика — ранг истинной пары
среди тайлов той же картинки (`same_image_*`, практический пул ~576) и среди
объединённого пула нескольких картинок (`pooled_*`, диагностика).

Смоук (idempotent-инвариантность ранга, конечные градиенты, perfect-similarity
R@1=1.0) проходит на CPU. 20-шаговый preflight на GPU уже дал сильный сигнал:

| step | same_image clean->dirty R@1 | R@5 | median rank | (шанс R@1=0.17%, R@5=0.87%) |
|---:|---:|---:|---:|---|
| 20 | **16.2%** | **38.5%** | 9 | ~94x chance at R@1 |

Гейт (`--steps 1500 --bs 4 --tiles-per-image 192 --embed-dim 128`, held-out
каждые 300 шагов, 8 фиксированных held-out картинок) запущен; порог
продолжения: `same_image_clean_to_dirty_r1 >= 0.05` И `r5 >= 0.15` (в ~29x и
~17x выше шанса соответственно — сознательно скромная планка, означающая
лишь "сигнал есть", а не "assembly решена"). Полный отчёт:
`E:/pazzle_work/gates/paired_alignment_gate.json`.

Если гейт пройден — следующий шаг: этап 2 (clean masked-halo модель,
предсказывающая embedding центрального чистого тайла по 2/4/8 известным
соседям, тренируется только на чистых картинках) и затем этап 3
(комбинация: predicted-clean embedding из контекста ранжирует реальные dirty
кандидаты через уже обученный dirty encoder).

### Фактический stage-1 gate: ПРОЙДЕН с большим запасом

Полный прогон (1500 шагов, `E:/pazzle_work/gates/paired_alignment_gate.json`),
чекпоинт `E:/pazzle_work/ckpt/paired_alignment_best.pt`:

| metric | step 20 | step 300 | step 1500 (финал) | требование |
|---|---:|---:|---:|---:|
| same_image clean->dirty R@1 | 0.162 | 0.710 | **0.759** | >= 0.05 |
| same_image clean->dirty R@5 | 0.385 | 0.847 | **0.873** | >= 0.15 |
| same_image dirty->clean R@1 | — | 0.719 | 0.772 | — |
| pooled (8 картинок, 4608 кандидатов) clean->dirty R@1 | — | — | 0.724 | диагностика |
| median rank (same-image) | — | 1 | 1 | — |

Кривая вышла на плато к шагу ~900 (0.750) и держится до 1500 (0.759). Даже
кросс-картиночный pooled R@1=72.4% (шанс ~0.02% на 4608 кандидатов)
подтверждает: сигнал общего содержания, а не переобучение на 8 held-out
картинках. **Это первый по-настоящему сильный результат за весь день
исследований** — идентичность тайла массово переживает независимую
деградацию (affine+noise sigma 40-55+blur+JPEG 35-50).

## Этап 2: clean masked-halo context (реализовано, гейт запущен)

`src/eval_halo_context.py`. Замороженный `clean_encoder` из этапа 1 даёт keys
для всех 576 тайлов картинки; обучается только spatial-reasoning transformer
(relative position embedding по 25 слотам окна 5x5, mask-token для
неизвестных, query head). Вопрос: определяет ли **чисто чистый** 2D-контекст
(без какой-либо деградации где-либо) идентичность конкретного замаскированного
центрального тайла среди всех 576 той же картинки — при context sizes 2/4/8
случайно выбранных соседей окна.

Data-free смоук проверяет: маскировку контекстных тайлов из кандидатов,
доступность истинного центра, конечные градиенты через frozen keys, отсутствие
утечки центра в context_indices. Технический preflight (20 шагов) на GPU
чист, 0.71 с/шаг.

Порог продолжения к этапу 3: `context4 R@1>=0.30` И `context4 R@5>=0.60` И
`context8 R@1>=0.50` — те же ориентиры, что были заданы для полного Frontier
Pointer (ветка B, провалившийся: context4 R@1=2.3%, context8 R@1=2.2%).
Ключевое отличие постановки: здесь исключены ВСЕ источники деградации — если
и эта чисто чистая версия провалится на похожих числах, диагноз будет другим:
не слабый dirty-encoder (уже опровергнуто этапом 1), а то, что локальное 2D
фотографическое содержание само по себе не определяет конкретный
недостающий тайл (повторяющиеся небо/стены/текстуры).

### Фактический stage-2 gate: ПРОВАЛЕН, плато рано и стабильно

Полный прогон (1500 шагов, `E:/pazzle_work/gates/halo_context_gate.json`):

| context size | step 300 R@1 | step 1500 R@1 (финал) | финал R@5 | median rank | требование R@1 |
|---:|---:|---:|---:|---:|---:|
| 2 соседа | 0.024 | 0.022 | 0.092 | 82 | — |
| 4 соседа | 0.027 | **0.029** | **0.108** | 51 | 0.30 |
| 8 соседей | 0.035 | **0.054** | 0.186 | 34 | 0.50 |

Плато наступило уже к шагу ~300 и не сдвинулось за оставшиеся 1200 шагов.
Числа почти идентичны провалившемуся Frontier Pointer (ветка B: context4
R@1=2.3%/R@5=7.6%, context8 R@1=2.2%/R@5=9.0%) — **несмотря на то, что здесь
энкодер заведомо силён** (тот же `clean_encoder`, что в stage-1 дал R@1=76%),
данные полностью чистые (без единой деградации) и позиции истинные.

**Диагноз теперь однозначен**, чего не было видно на Frontier Pointer: провал
не в представлении тайла (stage 1 доказал обратное) и не в шуме — а в том,
что **2-8 ближайших соседей 20x20-патча в обычной фотографии просто не
содержат достаточно информации, чтобы отличить конкретный недостающий патч
среди 576 кандидатов той же сцены**. Локальная фотографическая непрерывность
(небо, стены, ткань, боке) слишком однородна на этом масштабе фрагмента.
Это причина, по которой ЛЮБОЙ seam/локально-контекстный подход (все ветки
A-B, весь seam/affinity-граф от предыдущих дней, включая наиболее удачный
listwise ranker с p~0.95 но покрытием лишь ~0.03-0.08) упирается в один и
тот же потолок: сама степень локальной похожести соседних патчей физически
недостаточна для этого разрешения тайла (20x20 из 480x480, 576 штук).

**Ветка E закрыта** (этап 3 не имеет смысла без пройденного этапа 2; test не
использовался). Из неё остаётся ценный актив: контрастный `clean_encoder`/
`dirty_encoder` из этапа 1 (R@1=76% при поиске СВОЕГО тайла среди 576) —
он не решает assembly "с нуля", но является сильным кандидатом для
повышения точности/recall существующего пайплайна поиска оригиналов
(`source_forensics`): вместо грубого bag-fingerprint + SIFT можно матчить
каждый dirty-тайл напрямую в этом embedding-пространстве против тайлов
кандидатов-фотографий, получая как обнаружение совпадения, так и прямое
per-tile соответствие (без отдельного Hungarian/SIFT шага).

## Ветка F: радиальный композиционный/оптический макро-приор

Гипотеза: раз локальное СОДЕРЖАНИЕ не определяет тайл (ветка E), возможно
физическое/оптическое ИЛИ композиционное положение в кадре (не зависящее от
сцены) определяет хотя бы грубый радиус от центра — subject-centered
композиция типичной репортажной съёмки, глубина резкости (объект в фокусе
в центре, фон размыт по краям), виньетирование и хроматическая аберрация
линзы (растут от центра к краю независимо от содержимого).

`src/eval_macro_composition_gate.py`: 8 ручных признаков (яркость, sharpness/
Laplacian-энергия, хроматическая аберрация R/B, насыщенность, доля
skin-tone пикселей) → ExtraTrees, group-held-out (96 train/24 val картинок),
фиксированные (не зависящие от контента) метки: радиальный квартиль (4-way,
шанс 25%) и бинарная граница/интерьер (шанс 50%), отдельные ветки dirty/clean.

**Результат: провал даже на ЧИСТЫХ изображениях.**

| branch | quartile acc | quartile chance | border bal_acc | border chance |
|---|---:|---:|---:|---:|
| dirty | 0.271 | 0.25 | 0.511 | 0.50 |
| clean | 0.275 | 0.25 | 0.525 | 0.50 |

Пилотный прогон на 5 картинках намекал на слабый сигнал (clean quartile
31.6%), но это оказался шум малой выборки — на статистически значимых 24
held-out картинках эффект исчезает даже без какой-либо деградации. Вывод:
митап-фотография (часто групповые/панорамные кадры, экраны, сцена) не имеет
устойчивого across-photo "объект в центре" смещения. **Ветка F закрыта.**

## Внешняя проверка: та же стена уже задокументирована в литературе

Поиск подтвердил независимо: "Benchmarking Content-Based Puzzle Solvers on
Corrupted Jigsaw Puzzles" (arXiv 2507.07828) тестирует ровно такой же класс
деградаций (noise/blur/JPEG/photometric shifts, erosion) на современных
солверах, включая diffusion-based методы, и заключает: **"content-based
solvers struggle dramatically... at severe corruption levels... solver
performance approaches near-total failure regardless of algorithmic
sophistication"** и **"no existing method maintains robust performance under
heavy, multi-type corruption."** Это независимо подтверждает наш вывод после
~7 архитектур за 2 дня: это не пробел в нашем инжиниринге, а
задокументированный фундаментальный предел content-based сборки при такой
степени деградации.

Отдельно изучен самый релевантный положительный метод — JPDVT (arXiv
2404.07292, diffusion-based jigsaw solver): диффузия **непрерывного
позиционного embedding** (не дискретного класса), conditioned на content
embeddings всех кусочков сразу (joint self-attention), затем greedy
nearest-neighbor + dedup декодирование. Механически отличается от всех
закрытых веток (не sparse-graph как CNSD/GraphGRU, не local-context как
Frontier/halo, не absolute-slot classification). Но их лучшие результаты —
на 9-150 кусках при **лёгкой** деградации (missing pieces/erosion 7-33%, без
noise/blur/JPEG в основных числах) с piece-level acc 54-83%; ни разу не
тестировалось на 576 кусках с sigma 40-55 шумом + blur + JPEG q35-50
одновременно. Комбинируя это с их же benchmark-статьёй (выше), нет оснований
считать, что диффузионный декодер обойдёт уже задокументированный потолок
именно на нашем уровне деградации — реализация была бы дорогим повтором
уже закрытого класса подходов (dense pointer/GraphGRU уже проверили
"joint whole-bag reasoning" в других формах и оба уперлись в тот же барьер).

## Текущий статус после веток A-F

Пройдены оба обязательных вопроса ветки E по отдельности (выживает ли
идентичность тайла - ДА, определяет ли локальный 2D-контекст недостающий
тайл - НЕТ), что впервые чисто разделило две конкурирующие гипотезы провала
предыдущих подходов. Ветка F закрыла последний непроверенный класс
content-independent сигналов (композиция/оптика кадра) — тоже провал даже на
чистых картинках. Все дешёвые и среднедорогие гипотезы geometry-from-content
(A: генератор, B: frontier context, C: JPEG-фаза, D: graph decoder, E: paired
alignment + halo context, F: радиальная композиция/оптика) исчерпаны
отрицательным или предельным результатом, и внешняя литература (arXiv
2507.07828) независимо подтверждает: это задокументированный предел класса
методов, а не пробел в нашем инжиниринге.

**Рекомендация**: assembly "с нуля" эта задача на данном уровне деградации
(576 кусков, noise sigma 40-55 + blur + JPEG q35-50, независимо на кусок) не
решается предложенными и литературными методами. Продуктивный путь дальше —
не искать восьмую архитектуру в том же content-based семействе, а
использовать два реально работающих актива по-другому:

1. **Тайл-идентичность (этап E1, R@1=76%) как усилитель внешнего поиска
   оригиналов** — не для сборки с нуля, а для матчинга dirty-тайлов теста
   против кандидатов-фотографий из web-каталога напрямую в этом
   embedding-пространстве (точнее и без отдельного SIFT/Hungarian шага).
2. **Частичный/regional успех вместо all-or-nothing**: SSIM — поэлементная
   метрика; даже неполная, но правильно собранная и корректно
   ПОЗИЦИОНИРОВАННАЯ (не обязательно у истинных координат, а внутренне
   консистентная) компонента из high-precision seeds (p~0.95, RSCM) может
   поднять частичный SSIM локально, если её потом трактовать не как "решение
   пазла", а как reference patch для NLM/restoration поверх наивной раскладки.
   Это ещё не проверено количественно и было бы следующим честным гейтом,
   если ветка "assembly с нуля" официально закрывается.

## Повторная строгая проверка: восстановим ли PRNG-сид перестановки?

Пользователь потребовал настоящее решение, не поиск test-датасета в сети —
это ортогонально к внешнему поиску: если сид shuffle-перестановки выводим из
имени/номера файла, это даёт 100% точную раскладку для всех 700 test БЕЗ
единого ML-компонента. Прошлая проверка в этом журнале была краткой ("простые
варианты RandomState/default_rng/random.shuffle с типичными seed не
воспроизводят"); повторил строже: `src/check_permutation_seed.py` берёт 5
train-картинок с максимальной recover-confidence (mean_conf>=0.97) как
надёжный ground truth и перебирает 11 схем вывода сида (номер файла,
номер×константа, sorted index, SEED+номер, md5/sha256/python hash(name)) x 7
PRNG API (`default_rng`/`RandomState` × `.permutation`/`.shuffle`,
`random.Random` × `.shuffle`/`.sample`) — итого ~77 комбинаций на картинку.

**Результат: везде уровень шума** (лучшее совпадение 0.70% при шансе 0.17%,
т.е. просто статистическая случайность на 77×576 сравнений). **Гипотеза о
угадываемом сиде окончательно закрыта** — не просто "несколько типичных
seed не сработали", а систематический перебор реалистичных схем вывода.

## Ветка G: иерархическая группировка по 4x4 макро-блокам

Ключевой факт проекта с самого начала (macro_oracle, эта же кодовая база):
**если тайлы правильно разбить на группы по 16 (блок 4x4), существующий
scorer уже решает каждый блок с placement≈0.68, neighbour≈0.72.** Узкое
место все две недели было не "решить локально", а "сгруппировать" —
все affinity-графы (r=1/r=3 энкодеры, listwise ranker) целились в точную
adjacency (recall потолок ~50-67% для *непосредственных* соседей) и не
довели дело до рабочей группировки по 16.

Ветка E этой сессии доказала: правильно поставленный corruption-invariant
contrastive objective (paired alignment) восстанавливает сигнал там, где
affinity-based similarity проваливается (R@1=76% против affinity-графа
~50-67%). Ветка G применяет тот же приём не к идентичности отдельного тайла,
а к более грубому и контентно насыщенному вопросу: **какому из 36 блоков 4x4
(80x80px) принадлежит этот отдельный грязный тайл?** Блок в 16 раз крупнее
одного тайла — гораздо менее однородный контентно, чем окно 2-8 соседей из
закрытой ветки E2.

`src/eval_block_identity.py`: `TileToBlockEncoder` (dirty 20x20 tile) и
`BlockEncoder` (clean 80x80 macro-block) в общем embedding-пространстве,
CE-классификация тайла против 36 истинных блоков той же картинки (точные
synthetic метки, никакой permutation cache). Смоук проверяет геометрию
блоков (`to_macro_blocks` побитово согласуется с `imgio.to_frags`),
градиенты, идеальный ранг. GPU preflight (20 шагов) уже дал R@1=9.7%,
R@5=37.3% при шансе 2.78%/13.9% (~3.5x/2.7x). Гейт намеренно нестрогий
(`same_image_r1 >= 0.15`) — решающая проверка не в точности одного тайла, а
в том, восстанавливает ли **capacitated joint-assignment по 16 тайлов на
блок** (агрегация 16 шумных "голосов") чистые группы; даже скромный
per-tile R@1 может дать почти идеальную группировку после голосования по
16 независимым тайлам одного блока (биномиальная концентрация). Полный
1500-шаговый прогон запущен, отчёт: `E:/pazzle_work/gates/block_identity_gate.json`.

### Фактический stage G1 (oracle vs true clean blocks): ПРОЙДЕН уверенно

Финал 1500 шагов: `same_image_r1=0.2465`, `r5=0.6410`, median rank **3 из 36**
(шанс R@1=2.78%, R@5=13.9% → ~8.9x/4.6x). Кривая всё ещё росла на шаге 1500
(не вышла на раннее плато, в отличие от веток B/E2/F) — сильнейший
content-based сигнал за всё исследование.

### Stage G2: реалистичная (без чистого референса) капацитированная кластеризация

На тесте у нас нет чистых блоков для сравнения — только грязный мешок.
`src/eval_block_group_assignment.py` замораживает обученный `tile_encoder`
и кластеризует ТОЛЬКО 576 грязных embedding'ов через alternating balanced
k-means (Hungarian на раунд, 36 кластеров по строго 16), без единого
обращения к чистым данным — реалистичный inference-сценарий.

| | mean purity | perfect blocks (of 36) | null baseline purity |
|---|---:|---:|---:|
| 8 held-out картинок | **0.246** | **0/36 во всех 8** | 0.139 |

Сигнал реальный (~1.6-1.9x null для каждой картинки), но **гейт
(`purity>=0.30` И `>=3x null`) не пройден** — ни одного полностью чистого
блока из 16 тайлов не найдено ни в одной из 8 картинок. Null-baseline сам
по себе оказался выше наивного ожидания (13.9%, не ~2.8%) из-за эффекта
"выбираем лучшее из 36! паросочетаний" при итоговом сопоставлении найденных
кластеров с истинными блоками — важная calibration-деталь для будущих
подобных метрик.

### Stage G3: прямой tile-to-tile same-block sanity check

Быстрая доп.проверка без кластеризации вообще: у какой доли из 576 грязных
тайлов ближайший (по cosine similarity) ДРУГОЙ грязный тайл действительно
из того же истинного блока?

**Raw top-1 same-block rate: 22.2% (шанс 15/575=2.61%, ~8.5x).** Это сильнее,
чем ожидалось от G2 — сигнал определённо есть на уровне пар. Но попытка
выделить high-confidence подмножество (по margin = top1_sim − mean_sim,
та же логика, что подняла RSCM с 44% до 95% на seam-рёбрах) сработала
гораздо слабее здесь:

| top-N% по margin | same-block precision |
|---|---:|
| 5% | 0.361 |
| 10% | 0.394 |
| 30% | 0.325 |
| 100% (весь пул) | 0.222 |

Confidence-фильтрация поднимает точность лишь в ~1.6-1.8x (22%→39%), а не в
5-10x, как это было для seam-рёбер в RSCM. Уверенное подмножество, готовое
к скармливанию в macro_oracle solver (там нужна точность groups, близкая к
0.8-0.9+), пока не выделяется.

### Итог ветки G

Это **самый сильный content-based сигнал за всё двухдневное исследование**
(A-G) — принципиально иной канал (макро-block identity через corruption-
invariant contrastive alignment), не seam, не локальный контекст, не
абсолютная позиция. Oracle-версия (G1) разгромно проходит; но обе
инференс-реалистичные декодировки (G2 balanced clustering, G3 confidence-
filtered nearest-neighbor) дают лишь ~1.6-1.9x null/шанс — недостаточно для
чистых, готовых к сборке групп. **Ветка G не закрыта категорически (сигнал
не на уровне шума, как C/D/F), но и не даёт готового решения** при текущей
ёмкости/декодировании. Незавершённые, но обоснованные варианты продолжения:
(a) больше ёмкости/шагов для tile_encoder, аналогично тому как ёмкость
подняла старый candidate_rank с 163k до 646k параметров; (b) более сильный
decoder кластеризации (например, learned/differentiable capacitated
assignment вместо frozen-embedding k-means); (c) отдельный dirty-dirty
"same-block" объектив, обучаемый напрямую (а не через clean-block proxy),
аналогично тому, как stage E1 работал через прямое InfoNCE, а не через
transformer-context.

## Ветка G4: прямой dirty-dirty same-block Siamese — реализована, гейт не пройден

Проверен незавершённый вариант (c) выше. Реализация:

- `src/block_siamese.py`: общий Siamese-энкодер грязного тайла, direct
  sibling supervised-contrastive loss, retrieval-метрики, multi-start
  balanced spherical k-means с точной ёмкостью 16;
- `src/train_block_siamese.py`: две независимые challenge-деградации,
  balanced sampling по всем 36 блокам, held-out gate, checkpoint/report;
- инициализация из прошедшего G1 `block_identity_best.pt`, но positive —
  только ДРУГОЙ dirty-тайл того же блока. Две деградации одного и того же
  source-тайла исключаются и из positive set, и из denominator, чтобы модель
  не подменяла same-block identity более лёгкой tile-self-identity.

Первый AMP-preflight обнаружил `grad=inf` и пропуск optimizer-step. Поэтому
финальный протокол по умолчанию FP32; AMP доступен только явным `--amp`, а
scheduler двигается только после реально применённого optimizer update.
FP32 smoke проверил конечные градиенты и exact 36/36 recovery на разделимом
procedural примере.

Основной run: 1200 шагов, 1 image/step, 8 tiles/block, две независимые
деградации, held-out каждые 200 шагов. Лучший checkpoint по purity — step
1000. Расширенный финальный eval: 8 фиксированных held-out изображений,
balanced clustering 20 итераций x 8 restart.

| metric | старый G2/G3 proxy | G4 direct Siamese |
|---|---:|---:|
| top-1 другой тайл того же блока | 0.222 | **0.2203** |
| matched balanced-cluster purity | 0.2457 | **0.2533** |
| reciprocal precision | — | **0.2610** |
| perfect 16-tile blocks / image | 0 | **0** |
| near-perfect (>=14/16) / image | 0 | **0** |

Purity выросла лишь на `+0.0076` absolute, retrieval top-1 не вырос вообще,
а ни одной пригодной для macro local solver группы не появилось. Заранее
заданный gate (`top1>=0.40`, `purity>=0.35`, `perfect blocks>=1/image`) не
пройден. **Standalone ветка G4 закрыта:** direct metric learning подтвердил
реальный same-block сигнал, но не превратил его в декодируемые чистые группы.
Запуск local solver/end-to-end SSIM после этого был бы невалиден, поскольку
его измеренный placement≈0.68 относится только к oracle-чистым группам.

Артефакты:

- checkpoint: `E:/pazzle_work/ckpt/block_siamese_best.pt`;
- train gate: `E:/pazzle_work/gates/block_siamese_gate.json`;
- extended eval: `E:/pazzle_work/gates/block_siamese_eval8.json`;
- log: `E:/pazzle_work/logs/block_siamese_g4_1200.log`.

## Branch H: whole-board neural energy and bounded TV residual

### H1: GlobalStructuralCritic -- gate failed

Implemented `src/train_global_critic.py` around the previously untested
`GlobalStructuralCritic`.  Every positive/negative set contains the exact same
576 independently degraded tiles; only their order changes.  The negative
families cover adjacent and nearby swaps, 3x3 shuffles, 2x2/4x4/6x6 block
swaps, and a full random permutation.  This prevents tile/content identity
shortcuts and directly tests whether a network can score a proposed board.

The 505,290-parameter FP32 run completed 1200 steps.  Best held-out local
accuracy was only **0.5391** (step 1000), while the deliberately simple
tile-mean TV baseline scored **0.9330** on the same small gate.  The learned
score fluctuated around chance and did not transfer between source images.
The gate failed, so critic-guided discrete repair was intentionally not run.

Artifacts:

- checkpoint: `E:/pazzle_work/ckpt/global_critic_best.pt`;
- report: `E:/pazzle_work/gates/global_critic_gate.json`;
- log: `E:/pazzle_work/logs/global_critic_1200.log`.

### H2: bounded neural residual over TV-hard local negatives -- safe but too weak

Implemented `src/train_tv_residual_critic.py`.  It mines the lowest-TV-margin
local corruptions from a pool of 48 candidates, then trains

`hybrid = 1000 * TV + 0.3 * tanh(neural_score)`.

The learned term is therefore bounded: it can only alter close TV decisions
and cannot overturn a sufficiently confident TV margin.  The held-out gate
uses 8 images and 32 negatives per each of 4 local families (1024 decisions),
and separately counts corrected TV failures and broken TV successes.

Best checkpoint (step 300):

| metric | value |
|---|---:|
| TV accuracy | 0.8145 |
| hybrid accuracy | **0.8203** |
| absolute lift | **+0.0059** |
| TV failures corrected | **6 / 190** |
| TV successes broken | **0** |

The safety mechanism worked, and the residual signal was directionally useful,
but the predeclared gate (`lift >= 0.02`, correction rate `>= 0.15`, break rate
`<= 0.05`) failed.  Post-hoc bound sweeps from 0.3 through 1.5 did not reveal
a hidden large gain: corrections and breakages both rose slightly, while lift
stayed around +0.004 to +0.005.  This rules out "the coefficient was merely
too conservative" as the main explanation.

Artifacts:

- checkpoint: `E:/pazzle_work/ckpt/tv_residual_critic_best.pt`;
- report: `E:/pazzle_work/gates/tv_residual_critic_gate.json`;
- log: `E:/pazzle_work/logs/tv_residual_600.log`.

Conclusion: a full-board CNN energy does not generalize, while the bounded
residual provides a small, safe improvement but is not yet strong enough to
drive search.  The next justified neural experiment is an additive
orientation-shared seam energy trained only on TV-hard seams, not another
global board CNN or RL policy.

## Branch I1: clean-structure auxiliary seam ranker -- gate failed

The ranked roadmap after branches A-H is in `NEXT_EXPERIMENTS.md`.  Its first
experiment followed the inpaint-then-classify idea from eroded-boundary puzzle
work, but adapted it to this dataset: a shared oriented-pair encoder ranks the
old frozen hard candidate list and simultaneously reconstructs clean
luminance plus horizontal/vertical gradient fields from the independently
degraded pair.

Implementation:

- `src/structural_seam.py`: canonical structural channels, clean target,
  multi-task ranker, seam-weighted reconstruction loss, GPU smoke;
- `src/train_structural_seam.py`: exact old candidate graph and held-out rank
  metrics, clean positive-pair auxiliary supervision, checkpoint transfer.

The scratch 216k model improved slowly but remained below the old ranker:
best R@1 `0.1973`, R@5 `0.4245`, reciprocal precision `0.5112`.

The decisive comparison transferred the old width-64 ranker into the new
854k model.  The transfer was numerically exact (`max_abs_diff=0.0`), with the
three new structural input channels initially zeroed.  After 300 low-LR
fine-tuning steps:

| metric | old width-64 ranker | best structural fine-tune |
|---|---:|---:|
| conditional R@1 | 0.2715 | **0.2721** |
| conditional R@5 | **0.5078** | 0.4948 |
| all-true R@1 proxy | 0.1852 | **0.1858** |
| reciprocal exact precision | **0.5846** | 0.5389 |

The R@1 change is only `+0.0006`, while graph-usable reciprocal precision
regresses materially.  The auxiliary reconstruction loss did learn, but did
not expose a new transferable adjacency signal.  Gate thresholds
(`R@1>=0.35`, `R@5>=0.60`, reciprocal precision `>=0.65`, all-true proxy
`>=0.24`) all failed.  Branch I1 is closed.

Artifacts:

- checkpoints: `E:/pazzle_work/ckpt/structural_seam*_best.pt`;
- reports: `E:/pazzle_work/gates/structural_seam_gate.json` and
  `E:/pazzle_work/gates/structural_seam_ft_gate.json`;
- logs: `E:/pazzle_work/logs/structural_seam_600.log` and
  `E:/pazzle_work/logs/structural_seam_ft300.log`.

## Branch I2: label-free per-puzzle adaptation -- gate failed

`src/eval_test_time_adaptation.py` adapts one puzzle bag without accepting
permutation labels in either pseudo-edge selection or the optimizer.  It
selects high-margin reciprocal edges (and supports exact 2x2 prediction loops),
then optimizes either the ranking MLP or normalization affine parameters using
photometric augmentation consistency, soft pseudo-label CE, distillation, and
a trust region.  Labels are exposed only for final paired metrics.

The selector itself worked: a one-image full probe selected 96 pseudo edges at
96.9% exact precision.  Nevertheless, the flexible head adapter reduced R@1
by `0.0156`; a normalization-only adapter reduced it by `0.0078`.  The final
four-image conservative gate produced:

- mean pseudo-edge precision `0.872` (range `0.714..1.000`);
- candidate R@1 `0.2520 -> 0.2520`;
- reciprocal exact precision delta `+0.0042`.

This is far below the required `+0.05/+0.08`.  Clean seeds do not teach the
model how to repair uncertain rows within one bag: flexible adaptation
overfits them and conservative adaptation is effectively a no-op.  I2 is
closed as an adaptation method.  Its strict pseudo-edge selector is retained
for the future consensus-island experiment.

Artifact: `E:/pazzle_work/gates/test_time_adaptation_gate.json`.

## Branch I3: 4x4 relative-coordinate flow -- gate failed

The JPDVT/PuzzleFlow-inspired bounded prototype is implemented in
`src/relative_flow.py` and `src/train_relative_flow.py`.  It has no token
position embeddings, passes a numerical permutation-equivariance test, starts
from Gaussian 2-D coordinates, predicts a conditional velocity field, and
uses Hungarian assignment only at the endpoint.

The model can memorize a fixed degraded batch perfectly (100% placement at
step 200), so neither the implementation nor decoder is the blocker.  Across
64 frozen puzzles from 32 unseen images, however, the best of 1600 steps was
only `0.0723` placement and `0.1061` neighbour accuracy.  Coordinate RMSE
improved while identity-to-coordinate association stayed random.  This is
scene memorization/conditional-mean collapse, so the planned 8x8 curriculum is
cancelled.

Artifacts:

- `E:/pazzle_work/gates/relative_flow_4x4_gate.json`;
- `E:/pazzle_work/relative_flow/relative_flow_4x4_best.pt`;
- `E:/pazzle_work/logs/relative_flow_4x4_1600.log`.

## Branch I5: consensus islands -- margin-only gate failed

`src/consensus_islands.py` turns reciprocal directional predictions into
translation-free coordinate components, greedily rejects inconsistent
geometry and occupied-coordinate collisions, and measures component purity
only after construction. `src/eval_consensus_islands.py` scores the complete
directional graph and sweeps label-free confidence thresholds.

The first three-image top-16-per-affinity gate found meaningful but
uncalibrated local signal. At quantile 0.80 it covered 15.1% of tiles in pure
nontrivial islands, but exact edge precision was only 74.1%. At quantile 0.93
precision rose only to 84.6% while coverage fell to 8.7%. The best pure
components averaged roughly five tiles, and no 2x2 prediction loop survived.

This rejects margin quantiles as the freezing criterion. A stricter
independent-graph agreement variant is the remaining bounded I5 test.

The independent-agreement test failed as well. Across three scenes, its
strictest tested regime (quantile 0.90) averaged 84.8% exact edge precision,
7.2% pure nontrivial coverage, and a largest pure island of only three tiles.
Agreement helped the easiest scene but did not protect against the hardest
scene. I5 is closed; reports are
`E:/pazzle_work/gates/consensus_islands_gate.json` and
`E:/pazzle_work/gates/dual_consensus_islands_gate.json`.

## Branch I4: posterior seam marginalization -- gate failed

A stochastic boundary generator was trained around the frozen deterministic
matching denoiser. Best-of-four supervision produced diverse clean hypotheses
and a strong oracle improvement (`0.0634 -> 0.0433` held-out boundary L1).
Thus this experiment did not merely reproduce the deterministic restorer.

Nevertheless, label-free log-mean-exp marginalization improved candidate R@5
by `0.0313` and NLL by `0.0291`, but changed R@1 by exactly zero and improved
Brier by only 0.32% relative. Raw-score residual fusion and analytic Gaussian
edge overlap each produced only `+0.0104` R@1 while worsening calibration.

The missing information is now specific: multiple plausible clean edges can
be generated, but an isolated pair cannot select the scene-consistent
hypothesis. More posterior samples will not solve that selection problem.

Artifacts:

- `src/posterior_edge.py`, `src/train_posterior_edge.py`,
  `src/eval_posterior_seam.py`;
- `E:/pazzle_work/posterior_edge/posterior_edge_best.pt`;
- `E:/pazzle_work/gates/posterior_seam_analytic_gate.json`.

## Branch I6: balanced discrete partition flow -- gate failed

The refiner is exactly equivariant to tile order and anonymous group-label
permutations, trains on capacity-preserving corruption, and uses a 576x576
Hungarian decode after every denoising stage. A learned identity prior makes
the procedure residual and prevents it from destroying the starting
partition.

On artificial corruptions the model eventually improved assignment accuracy
by about 0.5--0.8 percentage points. On the actual block-Siamese balanced
clustering, however, every one of four refinement iterations reproduced the
same `0.2387` purity, with zero perfect or near-perfect groups. Training was
stopped at the checkpointed step 400 no-change gate. The mechanics of discrete
capacity flow work, but a roughly 25%-pure seed does not contain enough
coherent group identity for this refiner.

Artifacts:

- `src/balanced_partition_flow.py`,
  `src/train_balanced_partition_flow.py`;
- `E:/pazzle_work/balanced_partition_flow/best.pt`;
- `E:/pazzle_work/gates/balanced_partition_flow_gate.json`.

The next bounded experiment should target the clearest remaining measurable
failure: confidence calibration across scenes. Existing reciprocal edges
occasionally reach >90% precision, but raw margins do not predict which scene
is reliable. A scene-conditioned correctness calibrator can be trained on
whole-image splits and gated directly on precision/coverage before any new
assembly work.

## Branch I7: scene-conditioned edge confidence -- sparse success

The implementation is in `src/edge_confidence.py` and
`src/train_edge_confidence.py`. It uses image-disjoint fit/calibration/held-out
splits and fixes one probability threshold using calibration scenes only.
Inference features include forward/reverse score shape, reciprocity, agreement
and rank in two affinity graphs, direction, tile texture/photometric statistics,
and within-puzzle standardized ranks. No permutation-derived feature is
available to the model.

An initial K=8 run exposed a useful guardrail: 11.1% total top-edge accuracy
makes a 90%-precision/15%-coverage gate mathematically impossible because it
would require 13.5% exact-edge coverage. The final run therefore restored the
ranker's intended K=64 candidate graph.

On 40/10/10 whole-image splits:

- held-out positive rate: `0.1914`;
- fixed-threshold precision: `0.8974`;
- fixed-threshold coverage: `0.0305`;
- precision at 2% coverage: `0.9615`;
- precision at 5% coverage: `0.8906`;
- precision at 15% coverage: `0.6354`.

The strict I7 gate failed on coverage and worst-image acceptance, but this is
the first learned confidence transformation that materially beats raw
cross-scene margin calibration.

## Branch I8-I10: calibrated islands and growth -- partial success, gates failed

`src/eval_confident_islands.py` scores all 2304 directed rows, caches their
features/candidate logits, and applies the fixed I7 threshold before any
permutation is consulted. Three full held-out graphs averaged:

- exact seed-edge precision `0.9433`;
- pure nontrivial tile coverage `0.1244`;
- largest pure component `5.33`;
- translation-aligned tile accuracy `0.9959`.

The sparse confidence signal is therefore real and geometrically useful.
However, three expansion rules did not convert it into broad coverage:

| expansion | best edge precision | pure coverage | largest pure component |
|---|---:|---:|---:|
| single weak edge from seed, p>=0.95 | 0.9135 | 0.1302 | 8.33 |
| reciprocal component shift, p>=0.90 | 0.9331 | 0.1354 | 5.67 |
| top-k alternative shift consensus | 0.9433 | 0.1244 | 5.33 |

Single-edge growth can create larger correct islands but also contaminates
other components, so total pure coverage barely changes. Reciprocal and
alternative agreement are safer but mostly restate already accepted seed
geometry. The correct next abstraction is not another greedy edge rule:
calibrated islands should become soft supernodes inside a global assignment
objective, with uncertain evidence allowed to vote without being frozen.

Artifacts:

- `E:/pazzle_work/edge_confidence/best.pt`;
- `E:/pazzle_work/gates/edge_confidence_gate.json`;
- `E:/pazzle_work/gates/alternative_consensus_gate.json`;
- `E:/pazzle_work/edge_confidence/full_graph_cache/`.

## Branch I11: corrected discrete global solver -- breakthrough

The planned soft QAP route was rejected by an oracle test: perfect directional
relations still produced only 34.7% placement and 69.7% neighbour accuracy
after a stronger optimization schedule. Hungarian rounding did not fix the
underlying local optimum.

Forensics of `src/solve_buddies.py` found that its edge miner applied board-edge
validity to arbitrary shuffled tile IDs. Removing this invalid assumption and
using the candidate-ranker's full K=64 directional distributions changed the
global result materially:

- mean neighbour accuracy on six synthetic held-out scenes: `0.1647`;
- scene-name paired check on the two legacy report scenes: `0.1803`;
- old two-scene PairwiseNet buddies report: `0.1386`.

The latter comparison is scene-paired but not corruption-paired, because the
new evaluation samples fresh generator degradation. The six-scene number is
therefore the main result. It exceeds the 16% breakthrough gate and is about
19% relatively above the recorded legacy mean. Calibrated bonuses add only
`+0.00045`, showing that the next bottleneck is absolute packing of already
useful components, not discovery of additional high-confidence local edges.

Implementation and reports:

- `src/eval_calibrated_buddies.py`;
- corrected `src/solve_buddies.py`;
- `E:/pazzle_work/gates/calibrated_buddies_gate_6img.json`;
- `E:/pazzle_work/gates/candidate_buddies_paired_0_1.json`.
