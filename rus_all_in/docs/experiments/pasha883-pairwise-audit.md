# Honest audit архивного Pasha883 PairwiseNet C64

## Проверка заявления `R@1 ≈ 0.4`

Исторический ref `origin/pasha883` (он же `origin/MAESTRO`) указывает на commit
`94166506b092bc46a03dfd338e85006726bd9097`. Ни в коде, ни в сохранённых
артефактах этой ветки нет валидного all-576 `R@1 ≈ 0.4`. Рядом встречаются три
разные величины, которые легко перепутать:

| Число | Что это в действительности | Почему это не all-576 R@1 |
|---:|---|---|
| `0.4765625` | checkpoint `val`, в логе ошибочно названный `acc@48` | `evaluate(..., M=32, nA=48)`: 32 sampled candidates на anchor, а 48 — число anchors; random negatives допускают self, повторы и дополнительные true neighbours |
| `≈0.48` | precision только среди взаимных top-1, то есть best buddies | условная precision на маленьком выбранном подмножестве, не recall всех 1 104 directed true edges |
| `0.43–0.50` | SSIM при teacher/oracle perfect placement; рядом также упомянут leaderboard `0.40` | это качество уже правильно собранного изображения, не метрика matcher/solver |

Сама ветка подтверждает различие в `docs/pazzle_strategy.html`: sampled
`acc@48=0.477`, полный true-neighbour `R@1≈0.20`, best-buddy precision `≈0.48`,
а end-to-end arrangement почти случайный. Наш frozen full-pool audit даёт
точную ближайшую оценку: `R@1=18.0707%`, `R@5=34.8732%` на четырёх boards.
Эти boards входят в historical last-300 validation roster, на котором
checkpoint выбирался каждые 500 steps; reference получен через target-assisted
`perms.npz`. Поэтому даже `18.07%` — exposed diagnostic, а не fresh estimate.

### Код, законность и воспроизводимость

- Checkpoint соответствует не branch-tip class, а `PairwiseNet(C=64)` из
  commit `bf5084c`: ordered pair исходных `20×20` tiles склеивается в
  `(3,20,40)`, down direction использует score-time transpose, затем CNN с
  global average pooling выдаёт один logit. В output transpose не попадает.
- `solve.py::_greedy` ставит каждый индекс ровно один раз через `used`, а
  `_anneal` меняет только две позиции местами. `assemble(frags, order)` берёт
  исходные upright fragments без rotation, warp или tile synthesis. Таким
  образом, solver удовлетворяет strict original-upright permutation, хотя в
  historical runner нет отдельного runtime permutation assertion.
- Inference воспроизводим через архивный `pair_best.pt` SHA-256
  `4686aadbf36b26d45edc5f87646a6f6e01d10e004fee784b8f3af6d0ea7e4639`
  и наш strict C64 loader. Checkpoint не был закоммичен в git: исторически он
  жил во внешнем Kaggle resume dataset. Branch-tip C96/seam-aware class с ним
  несовместим. Exact retraining не воспроизводим: worker augmentations создают
  unseeded `np.random.default_rng()`, включён cuDNN benchmark, а checkpoint не
  содержит optimizer/scheduler/scaler state.
- По own logs global solver даёт `place_acc≈0.0015`, `SSIM≈0.106`; на нашем
  buddies96 audit direct placement `0.0868%`. Oracle-perfect score claim
  относится к non-deployable teacher matrix и не превращает real scorer в
  сильный solver.

### Apples-to-apples и переносимый результат

На тех же четырёх dirty boards и том же recovered reference Pasha C64 имеет
`R@1=18.0707%`, raw d64 `14.7192%`, d64 partial-OT `16.5082%`. Один заранее
фиксированный 50/50 rank fusion поднял local `R@1` до `19.6784%`, но ухудшил
decoder adjacency `6.8614%→5.6159%`; старый global solver и этот неизменённый
fusion повторять не следует. Direct hard-edge head нельзя честно назвать
all-576 R@1 scorer: он только переупорядочивает уже выбранные hard edges. Его
source-disjoint fresh64 результат на собственной apples metric —
`+1.469` correct top288 edges/board и `+0.263 pp` adjacency.

Единственный реально переносимый компонент — frozen C64 pair logits (или их
precomputed matrices) как независимый evidence/candidate channel для d64 и
board-conditioned selector. Они дают complementary local signal, но не
заменяют d64, не оправдывают заявление `R@1≈0.4` и требуют нового
source-disjoint gate. Не переносим старые greedy+SA/buddies placement,
target-assisted cache или уже провалившийся fixed 50/50 fusion.

## Вердикт

Архивный `pair_best.pt` действительно содержит сильный local edge signal, а не
только красивую sampled-validation цифру. На последних четырёх boards
`img_006996.png..img_006999.png` full all-576 pooled retrieval равен:

- R@1 `18.0707%`;
- R@5 `34.8732%`;
- R@25 `58.4918%`;
- median rank `18.125`.

Это примерно на `+0.3057 pp` выше SocketMatcher d64 OT R@1 `17.7649%` из
source-disjoint exact-synthetic 16-source × 2-draw отчёта. Но сравнение не
является победой Pasha883: модели проверены на разных panels и corruptions, а
Pasha883 reference target-assisted и все четыре sources участвовали в
historical model selection. Разница одного R@1 на четырёх exposed boards
слишком мала для выбора default.

Последующий строго matched diagnostic снял различие panels: на этих же четырёх
boards frozen Socket d64 partial-OT получил pooled R@1/R@5/R@25
`16.5082/34.0353/58.6957%`, то есть Pasha сохраняет преимущество R@1
`+1.5625 pp`. Единственный заранее фиксированный 50/50 rank-percentile fusion
поднял local retrieval до `19.6784/37.2056/60.6431%`, но buddies96 adjacency
ухудшилась `6.8614%→5.6159%`. Это полезный local complementarity diagnostic,
но не основание менять default или открывать новый exact panel. Важно:
matched boards отсутствуют и в Socket train1024, и в его eval32, тогда как для
Pasha они model-selection-exposed; сравнение совпадает по pixels/reference, но
не по model exposure.

Глобальная конверсия остаётся слабой. Pasha scores через неизменный buddies96
дали direct placement `0.0868%`, то есть хуже bilateral `0.1302%`; adjacency
выросла `3.0571%→6.8614%`, но raw SSIM почти не изменилась:
`0.099593→0.099685`. Следовательно, checkpoint полезен как дополнительный
local scorer/fusion candidate, но сам по себе не решает absolute layout.

## Что именно загружено

Checkpoint:
`artifacts/prior-pasha883/pair_best.pt`, SHA-256
`4686aadbf36b26d45edc5f87646a6f6e01d10e004fee784b8f3af6d0ea7e4639`.

В нём 1 953 025 параметров и architecture из historical commit `bf5084c`:
`PairwiseNet(C=64)` с global average pooling. Более поздний branch-tip source
описывал другую C96/seam-aware class, поэтому loader воспроизводит C64 явно и
делает только strict `state_dict` load; partial/silent load запрещён.

Checkpoint сохранён на step 6500 с `val=0.4765625`. Historical log называет это
`acc@48`, но код `train_pair.py::evaluate(model, ..., M=32, nA=48)` использует
**32 candidates и 48 anchors**. Более того, 31 random candidate мог включить
self, повтор, true neighbour или второй допустимый neighbour. Поэтому это не
R@1 среди 48 и тем более не all-576 R@1. В новом report поле намеренно названо
`sampled_validation_accuracy_at_32_mislabeled_acc_at_48`.

## Определение новой метрики

Для каждой board используются recovered `inv[position] = input_tile` из
`artifacts/prior-pasha883/perms.npz` (SHA-256
`690a433da1ade5ce3f61885ccd89cc8705eadbb38288307c4c62d73ff3bf4b12`).

Для horizontal и vertical оси берутся ровно 552 anchor tiles, у которых есть
правый или нижний физический сосед. Для каждого anchor модель оценивает все 576
input tiles; только self score маскируется. Rank равен единице плюс число logits,
строго превышающих true-neighbour logit, то есть точные ties получают общий
лучший rank. Pooled metrics объединяют 552 + 552 = 1 104 queries на board.

Vertical pair обрабатывается historical способом: оба 20×20 tile
транспонируются, затем конкатенируются в `(3,20,40)`. Это сохраняет exact
training contract.

Все восемь `576×576` dirty-only score matrices были сохранены до чтения `inv`,
`conf` и target PNG. Артефакты:
`outputs/pasha883-pairwise-audit/last4-full576/scores-6996.npz` …
`scores-6999.npz`. Суммарный model inference занял 282.45 секунды на MPS.

## Source/model-selection exposure

Historical split — последние 300 сортированных train filenames,
indices `6700:7000`. `pair_best.pt` выбирался repeated sampled validation на
этом же roster каждые 500 steps. Наши indices `6996–6999` поэтому гарантированно
source-exposed. Это сознательный diagnostic старого artifact, не fresh holdout,
не confirmation и не основание для confidence interval.

Reference тоже не organizer ground truth. `perms.npz` восстановлен через
target-assisted Hungarian по dirty/clean descriptors. На четырёх boards mean
confidence равен `0.797–0.910`, но доля positions с confidence `<0.5` достигает
`11.11%`; ambiguous labels могут смещать retrieval и placement.

## Результаты

| Index | Pooled R@1 | R@5 | R@25 | Buddies direct | Buddies adjacency |
|---:|---:|---:|---:|---:|---:|
| 6996 | 13.4964% | 28.6232% | 50.3623% | 0.1736% | 4.7101% |
| 6997 | 21.9203% | 40.9420% | 66.5761% | 0.1736% | 7.9710% |
| 6998 | 25.1812% | 44.9275% | 71.1051% | 0% | 8.8768% |
| 6999 | 11.6848% | 25.0000% | 45.9239% | 0% | 5.8877% |
| **Mean** | **18.0707%** | **34.8732%** | **58.4918%** | **0.0868%** | **6.8614%** |

Axis means близки: right R@1 `17.9801%`, down R@1 `18.1612%`. Это полезная
проверка, что transpose contract не разрушил vertical direction.

Authoritative report:
[report.json](../../outputs/pasha883-pairwise-audit/last4-full576/report.json), SHA-256
`c91e992a388a1ee3d2820b3a313f154a388cc6b619e4f828a28862d9d26296fb`.
Он содержит per-board scores, bilateral control, geometry, raw SSIM, exact
artifact hashes и полное определение exposure.

## Matched Socket d64 и единственный 50/50 fusion

После Pasha freeze на **тех же** `6996..6999` запущен frozen
`v2-d64-train1024-s1600-r400-dev32`, checkpoint SHA-256
`0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670`.
Filename для каждой board читался только из пяти-полевого dirty-only Pasha
artifact; recovered `inv`, Pasha report и targets не открывались, пока на диск
не были записаны все Socket matrices и fusion layouts.

Явная проверка checkpoint lineage дала для каждой из четырёх boards
`socket_train1024=false`, `socket_eval32=false` и
`socket_any_checkpoint_exposure=false`. Для Pasha все четыре имеют
`historical_validation_and_model_selection=true`. Поэтому Socket arm здесь
source-disjoint относительно собственного checkpoint, но Pasha и fusion всё
равно source-exposed, а recovered reference target-assisted; это exploratory
diagnostic, не общий holdout.

Определение retrieval полностью совпадает с Pasha audit: 552 right и 552 down
queries на board, все 576 candidates, self masked, rank по числу строго лучших
scores. `Socket raw` — исходная cosine-logit matrix, `Socket partial-OT` — её
реальный `576×576` блок после partial Sinkhorn; dustbin в candidate pool не
входит.

| Scorer, mean 4 boards | Right R@1/5/25 | Down R@1/5/25 | Pooled R@1/5/25 |
|---|---:|---:|---:|
| Pasha raw | 17.9801 / 34.3750 / 57.2917% | 18.1612 / 35.3714 / 59.6920% | 18.0707 / 34.8732 / 58.4918% |
| Socket raw | 13.8134 / 30.4348 / 54.8007% | 15.6250 / 34.5109 / 58.9221% | 14.7192 / 32.4728 / 56.8614% |
| Socket partial-OT | 15.4891 / 31.6123 / 56.6123% | 17.5272 / 36.4583 / 60.7790% | 16.5082 / 34.0353 / 58.6957% |
| Pasha + Socket-OT rank50 | **18.4330 / 35.5525 / 58.9221%** | **20.9239 / 38.8587 / 62.3641%** | **19.6784 / 37.2056 / 60.6431%** |

Fusion был ровно один и зафиксирован до reference scoring. Для каждой строки и
каждой модели отдельно 575 non-self candidates получают stable descending
ordinal percentile: лучший `1`, худший `0`, self `−1`. Затем берётся
`0.5 × Pasha raw percentile + 0.5 × Socket partial-OT percentile`; никаких
весов, top-k или board-specific параметров не перебиралось. Из fused scores
один раз строится неизменный ORBIT `buddies96`.

| Global mean | Pasha buddies96 | Rank50 fusion buddies96 | Fusion delta |
|---|---:|---:|---:|
| Direct placement | 0.0868% | **0.1302%** | +0.0434 pp |
| Row accuracy | **3.6024%** | 3.3420% | −0.2604 pp |
| Column accuracy | 3.6458% | **5.8594%** | +2.2135 pp |
| Translation-aligned | **1.0851%** | 1.0417% | −0.0434 pp |
| Adjacency | **6.8614%** | 5.6159% | −1.2455 pp |
| Raw SSIM | 0.099685 | **0.101277** | +0.001591 |

Local complementarity реальна на этой маленькой exposed панели, но текущий
generic decoder её не сохраняет: primary geometry indicators aligned и
adjacency снижаются. Direct и raw SSIM gains составляют буквально один tile
scale и четыре Pasha-exposed/recovered-reference observations, поэтому
статистически или как promotion evidence не интерпретируются. Ни default, ни
production меняются; новый exact-synthetic panel из-за этого diagnostic не
открывается.

Matched report:
[report.json](../../outputs/socket-matcher/v2-d64-vs-pasha883-matched-last4/report.json),
SHA-256
`56f4dfe928ff0943f806df0283f9e5cc84d0dcff5e553bd9a32f103425303bbb`.
Dirty-only lineage находится в соседнем `dirty_freeze.json`; четыре
`scores-6996.npz..scores-6999.npz` содержат Socket raw/OT, ровно одну fusion и
её frozen layout. Pasha local metrics воспроизвелись с max absolute error `0`.

## Решение

- Сохранить exact C64 loader и frozen full-pair matrices как reusable local
  evidence.
- Не заменять SocketMatcher d64: observed R@1 difference мала и protocol не
  matched по model exposure; matched-pixels diagnostic также не дал global
  conversion gain.
- 50/50 Pasha + Socket-OT rank fusion считать проверенной local-only идеей:
  retrieval вырос, но buddies96 adjacency и aligned ухудшились. Не повторять
  тот же generic fusion/decoder и не открывать из-за него fresh panel.
- Не цитировать checkpoint `val=.47656` как `R@1@48`; это sampled 32-way
  accuracy с несовершенным negative sampler.
