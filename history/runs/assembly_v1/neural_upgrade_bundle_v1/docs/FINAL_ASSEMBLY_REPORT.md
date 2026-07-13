# Tile assembly: итоговый отчёт по экспериментам

Дата: 2026-07-11  
Проект: `/Users/rusyalain/Documents/test`  
Статус документа: **финальный; test700 inference завершён и локально
перепроверен по полному provenance contract**.

## 1. Итог в одном абзаце

Лучший подтверждённый input-only tile solver на фиксированном real16 —
**boundary-QAP** с SSIM `0.18281991502795386`. Он улучшает soft-cycle seed
`0.16543114001888390` на `+0.01738877500906996`, выигрывает на всех `16/16`
источниках, а paired source-bootstrap 95% CI для улучшения равен
`[+0.012173, +0.023101]`. Это реальный и воспроизводимый выигрыш, поэтому именно
этот конфиг выбран для финального test700 inference. При этом добавочный
boundary term сам по себе не доказан лучше обычного QAP: преимущество над QAP
25x2 без boundary составляет всего `+0.00049028689908426`, CI
`[-0.001814, +0.002633]` пересекает ноль.

Все проверенные поверх QAP маршруты — RL, LNS, annealing, line continuation,
CP-SAT, context reorganization, learned 2x2 hyperedges, MAE population search и
frozen-DINOv2 superblock positioning — не прошли заранее заданные gates. LaMa
занимает отдельную категорию: три bounded-запуска закончились до получения
метрики из-за legacy-checkpoint compatibility, поэтому это
**infrastructure-inconclusive**, а не доказанный научный провал метода.

По имеющимся validation-данным ожидать SSIM `>=0.3` от текущего пайплайна не
обосновано. Лучший валидный input-only результат находится около `0.183`, а
даже target-only oracle по различным проверенным candidate pools остаётся ниже
`0.194`. Это не обещание точного leaderboard score, а честный вывод из текущих
whole-source validation panels.

## 2. Задача и метрика

- Вход: PNG `480x480`, разбитый на сетку `24x24`, то есть `576` независимо
  испорченных тайлов по `20x20` пикселей.
- Тайлы случайно переставлены; к каждому независимо применяются шум, blur,
  JPEG, сдвиги яркости и контраста.
- Train: `7000` пар input/target; test: `700` input без target.
- Нужно одновременно восстановить перестановку и качество пикселей.
- Основная метрика: средний RGB SSIM,
  `skimage.metrics.structural_similarity(channel_axis=2, data_range=255)` с
  остальными параметрами по умолчанию (`win_size=7`).
- Требуемый submission: ровно `700` RGB PNG `480x480`, исходные имена, файлы в
  корне ZIP.

Этот отчёт посвящён именно assembly-части. Denoiser зафиксирован как upstream
компонент и не дообучается по target метрике assembly_cal.

## 3. Протокол оценки и защита от leakage

### 3.1. Основные панели

| Панель | Назначение | Ground truth | Правило использования |
|---|---|---|---|
| `primary_kornia` exact | Контролируемая синтетическая порча, истинная перестановка известна | Да | retrieval, adjacency, absolute position и reconstructed-image SSIM |
| `independent_libjpeg` exact | Независимый corruption engine | Да | проверка переноса, а не подгонки под Kornia |
| `assembly_cal` real16 | Фиксированные 16 целых реальных source images | Target открывается только после freeze layouts | Главный быстрый end-to-end gate |
| `assembly_cal` real64 | Более широкий фиксированный transfer panel | Аналогичный input-only freeze | Использовался в до-QAP истории для отбраковки real16 overfit |
| test700 | Финальный inference | Нет | Только выбранный заранее fixed pipeline |

Все обучающие и проверочные разбиения выполняются по целым source images, а не
по тайлам. Для real16 layout predictor не принимает target; в отчётах
зафиксированы `predictor_accepts_target=false`, `pseudo_mapping_used=false` и
`target_opened_after_layouts_frozen=true`.

Фиксированные real16 source IDs:

`img_003877.png`, `img_005080.png`, `img_004383.png`, `img_006582.png`,
`img_004810.png`, `img_006306.png`, `img_005844.png`, `img_004191.png`,
`img_001281.png`, `img_003971.png`, `img_006070.png`, `img_005710.png`,
`img_005224.png`, `img_006489.png`, `img_002514.png`, `img_004878.png`.

Фиксированные exact8 source IDs, использованные поздними global gates:

`img_003571.png`, `img_006833.png`, `img_000878.png`, `img_002134.png`,
`img_003560.png`, `img_000963.png`, `img_000534.png`, `img_002060.png`.

Разбиения поздних learned gates:

| Gate | Train | Validation/development | Exact transfer | Real |
|---|---:|---:|---:|---:|
| Context reorganization | `24` whole sources | `4` whole sources | exact8 | real16 |
| 2x2 hyperedge verifier | `64` whole sources | `8` whole sources | exact8 | real16 |
| DINOv2 superblock head | `512` whole sources | development64 | exact8 | real16 **не открыт** после kill-gate |
| Frozen MAE energy/search | Frozen pretrained model, без task training | input-only candidates на frozen real16 | — | targets присоединены только после freeze energies/layouts |

Во всех трёх trainable gates пересечения train/validation/exact/real source IDs
проверены как пустые.

### 3.2. Что можно и нельзя сравнивать напрямую

- Числа real4, real16 и real64 получены на разных размерах панелей. Они нужны
  для gate/transfer, но разницу между ними нельзя интерпретировать как чистый
  эффект метода.
- Exact adjacency/position и end-to-end SSIM измеряют разные свойства. Высокая
  локальная adjacency не гарантирует правильного глобального сдвига фрагмента.
- Target-only oracle применяется только как диагностический верхний предел
  candidate pool и никогда не является допустимым solver/selectors.

## 4. Вывод по denoise

**Denoise не является причиной низкого assembly score; в среднем он помогает и
пикселям, и извлечению соседей. Главный bottleneck — перестановка.**

Подтверждения:

1. Зафиксированный TileNAF checkpoint имеет validation tile SSIM
   `0.808280007`; на paired Kornia — `0.822673948`, на independent libjpeg —
   `0.819282899`. Ordered-image SSIM равен соответственно `0.723804582`,
   `0.716454916` и `0.713109433`. Эти числа относятся к restoration при
   известном правильном порядке, а не к итоговой сборке пазла.
2. Для одной и той же frozen boundary-QAP layout denoised render даёт
   `0.182819915`, raw render — `0.110247459`. Это прямое разделение качества
   перестановки и качества пикселей.
3. На real16 identity layout после denoise имеет `0.132569035`, тогда как raw
   identity render — `0.085231351`.
4. У HBT retrieval denoised RGB+Sobel даёт validation R1 `0.223845`, raw
   RGB+Sobel — `0.179008`. Sobel помогает только как добавочный канал к RGB;
   Sobel-only (`0.034279` denoised, `0.015002` raw) и binary edges
   (`~0.0075`) почти уничтожают сигнал.

Исторически seam-trained TileNAF давал небольшой renderer-only выигрыш на
frozen real64 layouts: `0.192371973` против `0.191869870`, delta
`+0.000502103`, wins `82.8%`, 95% CI `[+0.000343096, +0.000669386]`. Это
подтверждает, что post-layout restoration может немного помочь, но эффект
слишком мал, чтобы решить проблему порядка, и текущий QAP test job не следует
смешивать с этим старым real64 экспериментом.

## 5. История baseline до QAP

### 5.1. Solver sanity check

На одном clean shuffle weighted-L1 PBC восстановил SSIM `0.961719589` и
adjacency `0.941123188`. Следовательно, grid optimizer способен собрать пазл,
когда compatibility matrix качественная. Разрыв до real SSIM около `0.18-0.19`
показывает, что главный источник ошибки — оценка совместимости 20px границ под
независимой порчей, а не валидность перестановки как таковая.

### 5.2. Классические и learned local scorers

| Ветка | Лучший подтверждённый сигнал | Перенос на real | Решение |
|---|---:|---:|---|
| Classical C1 / PBC / MGC / tone / Lab fusion | clean sanity до `0.961720` SSIM | C1 real16 `0.174216024`; real64 `0.191869870` | Исторический сильный baseline |
| L0 seam-pair CNN | validation R1 `0.152231` | Ниже classical | Закрыта |
| L1 pooled side embedding | validation R1 `0.219486`, R32 `0.698299` | best real16 `0.172663036` | Не promoted |
| L1-v2 sequence | validation R1 `0.203167` | Ниже L1 | Закрыта |
| T0 absolute tile-position context | exact position accuracy `0.002658` | L1+T0 real16 `0.172996919` | Слишком слабый global prior |
| X0 reranker | R1 `0.200153`, candidate recall `0.761096` | L1+X0 real16 `0.171149818` | Не promoted |
| L1+X0+T0 | real16 `0.175864160` | best real64 learned combo `0.188669392`, ниже C1 `0.191869870` | real16 overfit; закрыта |
| Real pseudo-label L1 | exact R1 около `0.194` против base `0.219` | best real16 `0.170358834` | Self-confirming degradation |
| Direct Sobel/binary masks | Очень низкий edge-only R1 | best edge-fusion real16 `0.159580086` | Закрыта |
| HBT denoised RGB+Sobel | validation R1 `0.223845`, R32 `0.703889` | best real16 `0.172113739` | Лучший learned retrieval, но без real gain |
| HBT denoised RGB-only | validation R1 `0.215636` | best real16 `0.172337894` | Не promoted |
| G0 residual global matcher | R1 `0.217165` против frozen HBT `0.224072` | real16 не открыт после kill-gate | Scientific gate failed |

Этот этап дал важный отрицательный результат: улучшение synthetic neighbor R1
до `~0.224` не переносится автоматически в end-to-end real SSIM. Поэтому
последующие эксперименты всегда сравнивались с одной и той же frozen real16
панелью и authoritative QAP baseline.

## 6. Promoted QAP configuration

Зафиксированный test candidate:

- scoring/restoration checkpoint:
  `runs/denoise_v2/release/selected_tilenaf_synth_50k.pt`;
- checkpoint SHA-256:
  `77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734`;
- side-embedding checkpoint:
  `runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt`;
- side-embedding SHA-256:
  `c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787`;
- initial layout: soft-cycle, score `l1`, top-k `8`;
- QAP directional score: `l1w4` / denoised C1+L1w4 rank fusion;
- iterations `25`, restarts `2`;
- initial weight `0.75`, boundary weight `0.05`;
- noise scale `1.0`, noisy components `3`;
- refine swaps `8`, weak cells `32` in the final job config;
- no line score, no CP-SAT, no learned context/hyperedge/MAE/DINO selector.

На real16:

- soft-cycle seed: `0.16543114001888390`;
- boundary-QAP: `0.18281991502795386`;
- delta: `+0.01738877500906996`;
- wins: `16/16`;
- paired bootstrap 95% CI: `[+0.012173, +0.023101]`.

Обычный QAP 25x2 без boundary даёт `0.18232962812886960`; это статистически
неотличимый tie с boundary-QAP. Boundary-вариант выбран как фиксированный
validation winner, а не как доказательство сильного border prior.

## 7. Полная таблица end-to-end экспериментов на real16

SSIM в таблице — denoised-render SSIM на тех же 16 whole-source images. Delta
считается относительно authoritative boundary-QAP `0.18281991502795386`.

Статусы далее используются строго: **scientific fail** означает, что run и
оценка завершились штатно, но gate не пройден; **infrastructure-inconclusive**
означает, что целевая метрика вообще не была получена; **gated out** означает,
что зависимый дорогой эксперимент намеренно не запускался после провала
заранее заданного prerequisite.

| Ветка / фиксированный вариант | Real16 SSIM | Delta vs QAP | Итог |
|---|---:|---:|---|
| Identity layout | `0.132569035` | `-0.050250880` | Reference only |
| Classical denoised C1 fusion | `0.174216024` | `-0.008603891` | Historical baseline, не final |
| Direct edge fusion | `0.159580086` | `-0.023239829` | Scientific fail |
| L1 best component variant | `0.172663036` | `-0.010156879` | Не promoted |
| L1 + T0 best | `0.172996919` | `-0.009822996` | Не promoted |
| L1 + X0 best | `0.171149818` | `-0.011670097` | Не promoted |
| L1 + X0 + T0 best | `0.175864160` | `-0.006955755` | real64 transfer failed |
| Real-pseudo L1 best | `0.170358834` | `-0.012461081` | Scientific fail |
| HBT D320 denoised RGB+Sobel best | `0.172113739` | `-0.010706176` | Не promoted |
| HBT D320 denoised RGB-only best | `0.172337894` | `-0.010482021` | Не promoted |
| Seam-trained denoiser as scorer, best real16 variant | `0.174211464` | `-0.008608451` | Scorer not promoted |
| Soft-cycle L1 k8 seed | `0.165431140` | `-0.017388775` | Baseline for QAP |
| QAP, component L1-fusion q50 init | `0.180438955` | `-0.002380960` | Ниже fixed seed choice |
| QAP, denoised C1 component init | `0.177493708` | `-0.005326207` | Ниже fixed seed choice |
| Cross-QAP, component cross-L1w4 q50 init | `0.175050910` | `-0.007769006` | Ниже fixed seed choice |
| QAP L1w4, 25 iter, 2 restarts | `0.182329628` | `-0.000490287` | Tie, viable fallback |
| Cross-L1w4 QAP, 25 iter, 2 restarts | `0.182305312` | `-0.000514603` | Tie, no gain |
| Heavy QAP, 40 iter, 4 restarts | `0.181305114` | `-0.001514801` | Больше compute, хуже score |
| **Boundary-QAP, 25 iter, 2 restarts, w=0.05** | **`0.182819915`** | **`0.000000000`** | **Selected / promoted** |
| Multi-phase RL top-k 4 | `0.168510764` | `-0.014309151` | Scientific fail; CI ниже нуля |
| Multi-phase RL top-k 8 | `0.170995557` | `-0.011824358` | Scientific fail; CI ниже нуля |
| Multi-phase RL top-k 16 | `0.172828236` | `-0.009991679` | Scientific fail; CI ниже нуля |
| LNS subset 64 | `0.171237437` | `-0.011582478` | Scientific fail; CI ниже нуля |
| LNS subset 192 | `0.169914329` | `-0.012905586` | Scientific fail; CI ниже нуля |
| Cross-view soft-cycle | `0.175156078` | `-0.007663837` | Scientific fail; CI ниже нуля |
| Simulated annealing, 20k evaluations | `0.170495328` | `-0.012324587` | Scientific fail; CI ниже нуля |
| Learned context reorganization | `0.182819915` | `+0.000000000` | Scientific fail: layout unchanged |
| Learned 2x2 hyperedge anchors | `0.161223830` | `-0.021596085` | Scientific fail |
| Frozen-MAE selection over old mixed pool | `0.183549762` | `+0.000729847` | Correlation gate pass, end-to-end gain insufficient |
| MAE-guided 192-candidate population search | `0.182006832` | `-0.000813083` | Scientific fail; 95% CI below zero |

### 7.1. Paired confidence intervals for approximate solvers

| Route | Wins vs QAP | Paired bootstrap 95% CI |
|---|---:|---:|
| RL top-k 4 | `0/16` | `[-0.018596558, -0.010352519]` |
| RL top-k 8 | `0/16` | `[-0.015937366, -0.008409346]` |
| RL top-k 16 | `1/16` | `[-0.013361092, -0.006399882]` |
| LNS subset 64 | `4/16` | `[-0.023503856, -0.000166610]` |
| LNS subset 192 | `3/16` | `[-0.025238241, -0.002377245]` |
| Cross-view soft-cycle | `6/16` | `[-0.013833023, -0.001543782]` |
| Annealing 20k | `4/16` | `[-0.023718128, -0.001704383]` |
| MAE population search | `4/16` | `[-0.001512262, -0.000221584]` |

Ни один из этих интервалов не пересекает положительный эффект. Увеличивать те
же RL phases, LNS subset, annealing budget или MAE population не обосновано.

## 8. Эксперименты без полноценного real16 score

| Ветка | Проверенный протокол | Результат | Классификация |
|---|---|---|---|
| Faithful multi-phase RL | real4 | SSIM `0.141865475` против QAP `0.183733363` | Scientific fail |
| Particle beam p16/k4 | real4 | SSIM `0.157449129` | Scientific fail |
| Line-continuation fusion | real4 | soft-cycle `0.167554`, QAP `0.170975` против base QAP `0.183733` | Scientific fail |
| CP-SAT after base QAP | real4 | `0.183733`, layout без изменений | Scientific fail / no added value |
| CP-SAT after line-QAP | real4 | `0.168478`, хуже line-QAP `0.170975` | Scientific fail |
| Frozen-DINOv2 4x4 superblock head | development64 + exact8; real16 target не открыт | Не прошёл kill-gate | Scientific fail |
| Fragment positional diffusion | Не запускался по predeclared dependency gate | DINO prerequisite failed | Correctly gated out, не экспериментальный fail |
| GANzzle-style latent retrieval | Не запускался по prerequisite gate | DINO retrieval signal недостаточен | Correctly gated out, не экспериментальный fail |
| LaMa masked-consistency | Три bounded infrastructure attempts | Ни одной correlation metric; target не открыт | **Infrastructure-inconclusive** |

## 9. Почему закрыта каждая крупная ветка

### 9.1. Больше оптимизации поверх той же pairwise energy

QAP дал полезный global improvement, но дополнительные iterations/restarts уже
не помогают: 40x4 (`0.181305114`) хуже 25x2 (`0.182329628`). Target-only oracle
по четырём fixed QAP settings всего `0.185735`. Следовательно, проблема не в
недостатке итераций, а в качестве/неполноте pairwise energy.

RL, LNS и annealing создают другие локальные конфигурации, но все real16 CIs
ниже нуля. Даже target-only oracle по QAP плюс лучшим RL/LNS/cross/anneal
кандидатам равен только `0.188504012`, то есть всего `+0.005684097` над QAP.

### 9.2. Line continuation и CP-SAT

Самостоятельный raw+denoised line score имеет R1 лишь `5.84%` на primary и
`6.25%` на independent. Добавление line score снижает C1+HBT R1 с `16.12%` до
`14.76%` primary и с `14.54%` до `13.95%` independent. CP-SAT не может
восстановить информацию, которой нет в top-k pair graph: на base QAP он вернул
тот же layout, а на line-QAP ухудшил результат.

### 9.3. Learned context reorganization

Run полностью исправен: 2xT4, training/checkpoint/evaluation/leakage audits
завершены, baseline воспроизведён точно. Но exact8 wrong positions остались
`4597 -> 4597`, real16 `0.182819915 -> 0.182819915`; все layouts без изменений.
Training loss упал `5.70 -> 5.10`, но current-neighbour features оказались
self-confirming для локально связных, глобально смещённых QAP fragments. Это
научный нулевой результат, а не инфраструктурный сбой.

### 9.4. Learned 2x2 hyperedge verifier

Validation AP всего `0.01593`. При откалиброванном threshold `0.9283` verifier
принял `344` anchors, из них правильны только `6`: precision `0.01744`, coverage
`0.29861`. Exact adjacency упала `0.06103 -> 0.03442`, real16 SSIM —
`0.182819915 -> 0.161223830`. Primary и independent engines деградировали в
одном направлении. Threshold retuning не решает конфликт precision/coverage;
ветка научно закрыта.

### 9.5. Frozen MAE energy и population search

Первый mixed-pool gate выглядел сильным: mean per-source Spearman `0.651787`,
micro pairwise accuracy `0.752354` на `2124` парах. Но pool включал очевидно
слабые component layouts, и MAE в основном отделял их от QAP. End-to-end выбор
дал только `+0.000729847`: `6/16` wins, `10/16` losses, median delta
`-0.002817067`. Положительное среднее создавалось несколькими крупными
выигрышами и не было достаточно устойчивым для promotion.

Специальный falsification run проверил `192` seam-guarded кандидата на источник
(`3072` layouts). В competitive pool MAE почти случайный: mean Spearman
`0.057418`, micro pairwise accuracy `0.520181` на `262180` парах. Выбранный
score `0.182006832` статистически хуже QAP, а target-only oracle лишь
`0.188939313`. Поэтому generic MAE/IQA reranking для этой candidate family
закрыт. Версии 1 и 2 MAE-search были инфраструктурными сбоями; научный вывод
основан только на успешной версии 3.

## 10. DINOv2 superblock probe: точные development metrics

Frozen-DINOv2 4x4 superblock run завершился штатно на 2xT4. Обучалась только
маленькая set-to-position head на `512` whole sources; development — `64`
непересекающихся источника, затем exact8. Real16 target не открывался после
провала development kill-gate.

### Development64

| Метрика | QAP blocks | DINO assignment | Gate |
|---|---:|---:|---:|
| Coarse-cell accuracy | `0.028645833` | `0.044704861` | `>=0.10` |
| Mean coarse Manhattan | `3.816840278` | `3.519097222` | — |
| Aggregate Manhattan reduction | — | `0.078007733` | `>=0.25` |

Chance coarse-cell accuracy равна `1/36 = 0.027777778`. DINO выше chance, но
сигнал слишком слабый для безопасного перемещения блоков.

### Exact8 transfer

| Метрика | QAP blocks | DINO assignment |
|---|---:|---:|
| Coarse-cell accuracy | `0.052083333` | `0.062500000` |
| Mean coarse Manhattan | `3.736111111` | `3.520833333` |
| Aggregate Manhattan reduction | — | `0.057621` |
| Mean per-source wrong-position reduction | — | `-0.002626` |

Training token accuracy выросла до `0.2400`, а development assignment accuracy
осталась `0.0447`: это scene overfit. Поэтому positional diffusion и
GANzzle-style retrieval, которые зависели от transferable fragment-position
signal, не запускались. Их отсутствие — соблюдение kill-gate, а не скрытая
незавершённая успешная ветка.

## 11. LaMa: почему результат именно infrastructure-inconclusive

Нельзя писать «LaMa не работает»: ни одной LaMa correlation metric не было
получено, и ни один real target не открывался. Три bounded attempts завершились
до freeze Phase-A energy artifact:

1. Kaggle показал два code roots; первоначальный runner выбрал не тот. После
   этого добавлен contract-based selector, локальный manifest test прошёл.
2. Xet object hash был ошибочно принят за SHA-256 скачанного ZIP. Pinning был
   исправлен: revision/Xet identity и archive SHA-256 записываются отдельно.
3. Проверенный архив скачался, все 16 inputs были denoised, но legacy pickle
   checkpoint ссылается на `pytorch_lightning.callbacks`. Минимальный inference
   shim не является полной старой Lightning installation, поэтому `torch.load`
   остановился до inference.

Расширять arbitrary pickle shims или ломать современный Kaggle PyTorch старым
Lightning stack было признано неоправданным compatibility/security риском для
слабомотивированного no-reference gate. Научный дизайн сохранён, но метод не
оценён.

Зафиксированные pins:

- LaMa source commit:
  `786f5936b27fb3dacd2b1ad799e4de968ea697e7`;
- source archive SHA-256:
  `6759af2b68f942c32c52ecfed42d46b414cb1a8c1960a7b1167b88d40828deb7`;
- Big-LaMa mirror revision:
  `05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9`;
- downloaded Big-LaMa ZIP SHA-256:
  `f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6`;
- Xet object hash:
  `b2a4ef7f88e28fb6c15f0be152d7265a770b54a719774df975847430fa92a283`.

## 12. Почему текущие данные поддерживают ожидание ниже 0.3

Это вывод из нескольких независимых потолков:

| Candidate family | Лучший допустимый input-only | Target-only oracle | Интерпретация |
|---|---:|---:|---|
| Four fixed QAP variants | `0.182819915` | `0.185735` | Больше restarts той же energy не решит задачу |
| QAP + RL/LNS/cross/anneal | `0.182819915` | `0.188504012` | Даже идеальный selector далеко от `0.3` |
| Old mixed MAE pool | `0.183549762` | `0.193139685` | Более высокий oracle создают слабые/разные layouts, но MAE gain мал |
| 192-candidate MAE mutation pool | `0.182819915` (retain QAP) | `0.188939313` | Competitive reranking почти случаен |

Максимум допустимого selection здесь `0.183549762`; даже недопустимый
target-oracle максимум `0.193139685`. Поэтому значение `>=0.3` потребовало бы
не тонкой настройки текущего solver, а качественно нового сигнала, которого ни
один проведённый gate не обнаружил.

Ограничение вывода: real16 — малая validation panel, test distribution может
отличаться. Поэтому корректная формулировка — «`>=0.3` не поддерживается
экспериментами», а не «математически невозможно».

## 13. Финальный test700 artifact

Kaggle job `pasha883/vsos-final-qap-submission-t4x2` завершился со статусом
`COMPLETE`. Он применил один frozen boundary-QAP config к `700` test images на
двух T4, предварительно решил один image end-to-end, затем собрал два shard по
`350` images и fail-closed проверил archive/report/hash contracts. После
скачивания весь provenance contract был независимо повторён локально.

| Поле | Значение |
|---|---|
| Kaggle kernel final status | `KernelWorkerStatus.COMPLETE`, подтверждён CLI в `2026-07-11 06:56 MSK` |
| Kernel/runtime version | code dataset `pasha883/vsos-solver-rework-night-code` v7, contract `17` files; Python `3.12.13`, PyTorch `2.10.0+cu128`, `2 x Tesla T4`, capability `7.5` |
| Total wall time | `4923.541259 s` (`82 min 03.54 s`) |
| One-image preflight | `24.245045 s`; archive SHA `5e34d25e449ee34ea1d164b577e7f90aba0a1a636834fd1708e9d966b398f9bd`; replay PNG и layout byte-identical |
| Shard 0 count/hash | GPU 0, `350`, `4807.713717 s`, SHA `6ff51c568743a4dae80185f24b93f20accd5e99b335a154e1f1c02580b50c679` |
| Shard 1 count/hash | GPU 1, `350`, `4799.747990 s`, SHA `4da04b30d4a9e18160c55f511502d60b93df7d2c38593894b1ea87872a74a2e5` |
| Final archive path | `runs/assembly_v1/kaggle/final_qap_submission_output/v1/submission.zip` |
| Final archive SHA-256 | `1eeae828dd893198c07ac502d29aa5eeebd54bf6b818293d3b7e3f67ecb59607`; `204691118` bytes |
| Root member count | Ровно `700`, имена уникальны, вложенных путей/директорий нет |
| RGB/480x480/decode/CRC/name validation | Все `700/700` прошли `ZipFile.testzip`, полный PIL decode, PNG/RGB/`480x480`, per-member bytes/SHA и manifest-set comparison |
| Deterministic report/hash manifest | Report SHA `0fd2a7bba4543ab437f2e6df278e9fc72c1548c34d28579724e607b8fbfbb97f`; manifest SHA `c36a941e2740dac70d69678eda46052333c111c803802f9bfc31d1afe26844e0`; deterministic/operational hash manifests и `SHA256SUMS.txt` совпали |
| Leaderboard SSIM, если загружен | Не доступен: submission подготовлен, но в competition leaderboard не отправлялся |

Подготовленный job directory:
`runs/assembly_v1/kaggle/final_qap_submission_job`.

Предыдущий архив
`runs/assembly_v1/submission/classical_confirmed/submission.zip` с SHA-256
`79b0ad3275f22bfe5fa7d071e6d30c13c750e3a7b02aabe0ae70c700a9342bed`
является валидированным **историческим classical fallback**, а не результатом
текущего boundary-QAP test700 job.

## 14. Reproducibility: authoritative paths и hashes

| Артефакт | Путь | SHA-256 / статус |
|---|---|---|
| Selected TileNAF denoiser | `runs/denoise_v2/release/selected_tilenaf_synth_50k.pt` | `77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734` |
| HBT side embedding | `runs/assembly_v1/kaggle/edge2vec_gradient_gpu/hbt_d320_denoised_rgb_sobel.pt` | `c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787` |
| Boundary-QAP real16 report | `runs/assembly_v1/kaggle/qap_tuning_night_output/v2/qap_l1w4_boundary_real16.json` | `cc1b694b1501ba9b02e5618ad838e155ae40af7990bbbf4542b281fc21adec60` |
| Ordinary QAP real16 | `runs/assembly_v1/kaggle/qap_tuning_night_output/v2/qap_l1w4_multiseed_real16.json` | `2c486b433cd93654cb9eee7a155b0605da094dc5d4d47ef5b65bb99fcde75117` |
| Cross QAP real16 | `runs/assembly_v1/kaggle/qap_tuning_night_output/v2/qap_cross_multiseed_real16.json` | `1aa5df6d6cca404dd91a5075495822eee6f9e6a92a5f7cb189657630de2d9fd7` |
| Heavy QAP real16 | `runs/assembly_v1/kaggle/qap_tuning_night_output/v2/qap_l1w4_heavy_real16.json` | `2a2871922f171c99829bc0254b3c3ee20a27fe4739611c63658e55579f4e5520` |
| Solver rework matrix | `runs/assembly_v1/kaggle/solver_rework_night_job_download/v6/ANALYSIS.md` | Все 11 experiments внутри run complete; per-report hashes внутри |
| Context reorg checkpoint | `runs/assembly_v1/kaggle/context_reorg_gate_output/v1/` | `6911f28ea964e4ffc9582051c6ab7c329018282e9bf5ea4af5ad76ded47fde99` |
| Context gate exact/real payloads | тот же directory | exact `f66ce5cc500b25a85cc97a20ee76cd45482282d6dc7fd0f25a8caaee6113c962`; real `135bc02caa2901ed8413303bc76cc50b99560041e9769ab1267883bb94ae8178` |
| Hyperedge checkpoint | `runs/assembly_v1/kaggle/hyperedge_gate_output/v1/` | `ee23d6388e93f4e7581bc6184c82a24fd5ca8f8dd755a5a0d1e67e8523d2ebf3` |
| Hyperedge gate report | тот же directory | `e72405bee4baee26ff12170afb9141f5064c5dc5f3dbd48acfcb2ff09751f33c` |
| MAE energy frozen artifact | `runs/assembly_v1/kaggle/mae_energy_gate_output/v2/mae_energy_frozen.json` | `d4c33fca72b2e1480cd030f97897502719f6f4d74fd83ed217a699ddd0e1e39b` |
| MAE search frozen artifact | `runs/assembly_v1/kaggle/mae_search_gate_output/v3/mae_search_frozen.json` | `3ea6c18c61efeb9e02b444bdfcdd304d7aa3897015a0a996cfc0367440d75c14` |
| DINO authoritative compact QAP-reference manifest | `runs/assembly_v1/kaggle/dino_superblock_probe_output/v1/dino_superblock_code/reference/qap_l1w4_boundary_real16_manifest.json` | `92233fc5343aac3049ce0327417b645998bf477c6db91a4a852659312949ced6` |
| DINO probe report | тот же directory | `0d1e95b7ff5635642907936c26b1f4055decebc645c7c9d8c4aad816b0969555` |
| Submission builder | `scripts/build_assembly_submission.py` | Проверяется final job code contract |
| Global-placement research shortlist | `runs/assembly_v1/research/global_placement_shortlist_20260711.md` | Predeclared gates и stop rules для DINO/LaMa/diffusion/GANzzle |

Authoritative analysis files:

- `runs/assembly_v1/kaggle/qap_tuning_night_output/v2/ANALYSIS.md`;
- `runs/assembly_v1/kaggle/solver_rework_night_job_download/v6/ANALYSIS.md`;
- `runs/assembly_v1/kaggle/line_cpsat_gate_output/v1/ANALYSIS.md`;
- `runs/assembly_v1/kaggle/context_reorg_gate_output/v1/ANALYSIS.md`;
- `runs/assembly_v1/kaggle/hyperedge_gate_output/v1/ANALYSIS.md`;
- `runs/assembly_v1/kaggle/mae_energy_gate_output/v2/ANALYSIS.md`;
- `runs/assembly_v1/kaggle/mae_search_gate_output/v3/ANALYSIS.md`;
- `runs/assembly_v1/kaggle/dino_superblock_probe_output/v1/ANALYSIS.md`;
- `runs/assembly_v1/kaggle/lama_consistency_gate_output/v3/ANALYSIS.md`.

Версия `qap_tuning_night_output/v1` — incomplete download и не должна
использоваться; authoritative QAP output — только `v2`.

## 15. Следующие идеи, поддержанные текущими данными

1. **Fixed test700 boundary-QAP artifact завершён и полностью валидирован.**
   Дальнейший шаг возможен только как загрузка готового ZIP в competition; это
   отдельное внешнее действие, не выполнявшееся автоматически.
2. Если research будет продолжен после submission, менять нужно прежде всего
   **compatibility signal под corruption**, а не optimizer. Clean sanity
   (`0.961720` SSIM, `0.941123` adjacency) показывает, что solver работает при
   хорошем score matrix; QAP oracle и negative CIs показывают потолок текущей
   pairwise energy.
3. LaMa можно считать только незавершённой проверкой, а не перспективным
   подтверждённым направлением. Возвращаться к ней имеет смысл лишь в
   изолированном совместимом legacy environment или после безопасной конверсии
   checkpoint; текущие данные не оправдывают риск менять рабочую Kaggle env.
4. Positional diffusion и GANzzle-style retrieval сейчас не поддержаны
   prerequisite evidence: DINO superblock probe не достиг даже первого gate.
   Полный 576-node запуск без нового fragment-position signal будет расходом
   GPU без экспериментального основания.
5. Generic MAE/NR-IQA reranking и дополнительные мутации того же типа закрыты:
   competitive ranking около chance, а candidate oracle слишком низкий.

## 16. Self-review: ограничения и потенциальные несогласованности

- **Real16 vs real64:** более высокий old classical real64 score
  (`0.191869870`) нельзя напрямую сравнивать с boundary-QAP real16
  (`0.182819915`), потому что наборы источников различаются. Нужна отдельная
  QAP real64 evaluation, если потребуется утверждать transfer magnitude.
- **Boundary term:** итоговый boundary config лучший по mean, но его
  `+0.000490` против ordinary QAP статистически не подтверждён. Надёжно
  подтверждён QAP как семейство против soft-cycle, а не конкретно border prior.
- **MAE:** aggregate gate pass и population-search fail не противоречат друг
  другу: первый pool содержал явно слабые layouts, второй проверял только
  competitive QAP-near layouts.
- **LaMa:** отсутствие метрики нельзя включать в число scientific failures или
  использовать как доказательство бесполезности inpainting consistency.
- **DINO:** development fail корректно остановил real16 target access; поэтому
  у DINO нет end-to-end real16 SSIM и его нельзя вставлять в real16 ranking.
- **Historical archive:** старый `classical_confirmed/submission.zip` валиден,
  но не содержит новый boundary-QAP inference. Не выдавать его за текущий
  final artifact.
- **Final test700:** новый ZIP, его hash, runtime и валидность 700 PNG
  подтверждены скачанными authoritative outputs и повторной локальной
  проверкой. Leaderboard score по-прежнему неизвестен, поскольку submission не
  отправлялся в competition.
- **Runtime definitions:** DINO analysis указывает total wrapper time
  `905.38 s`, тогда как внутренний report field `seconds` относится к более
  узкой фазе (`821.45 s`). В итоговой презентации использовать одно явно
  названное определение.

## 17. Финальное заключение

Boundary-QAP — честно выбранный и статистически поддержанный final solver, но
не высокоточный решатель задачи. Он систематически улучшает soft-cycle, однако
собирает главным образом локальные fragments и не восстанавливает глобальную
семантику достаточно хорошо. Denoiser работает заметно лучше raw pixels и не
является причиной провала; bottleneck — совместимость и абсолютное размещение
тайлов. Ни один из реализованных global/approximate/learned post-solvers не дал
подтверждённого улучшения. До появления нового corruption-robust global signal
ожидание SSIM `>=0.3` не поддерживается.
