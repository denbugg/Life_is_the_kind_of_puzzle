# SocketMatcher v2 d32: border heads, OT-mass decoder и centre-prior

## Вердикт

`SocketMatcher v2 d32` устойчиво улучшает поиск соседей и геометрию
компонент, но всё ещё не восстанавливает абсолютную раскладку. На двух
source-disjoint real train-dev панелях по 24 изображения fixed
component/QAP decoder поднял adjacency с `3.44%` до `7.76–7.97%`, а
translation-aligned placement — с `0.93–0.95%` до `1.43–1.46%`. Exact
synthetic проверка независимо подтвердила adjacency `4.05% -> 8.40%`.

Однако direct placement остаётся порядка `0.20–0.33%` при chance
`1/576 = 0.1736%`, а raw SSIM decoder-а ниже bilateral control. Поэтому
текущий checkpoint и decoder — **перспективная research-база, не готовый
submission**.

Слабый `texture-centrality` prior с весом `0.05` улучшил часть метрик на первой
real панели, но полностью не подтвердился на второй. Он остаётся выключенным
по умолчанию. Гладкие/однотонные tiles не объявляются рамкой и не ставятся туда
жёстко.

## Что обучено

Authoritative checkpoint:
`outputs/socket-matcher/v2-border-train512-s300-r100-dev24/socket_matcher.pt`,
SHA-256 `7ccb14042e50432bf450018d4ebb32b78866d3755d8387cb1534f67155fd1c19`.

Модель имеет 150 280 параметров: `d=32`, 4 attention heads, один whole-board
Transformer layer, один SocketGNN layer и 10 Sinkhorn iterations. Она
warm-started из scale checkpoint v1 и получила interleaved 300 exact synthetic
и 100 low-weight real steps на lineage из 512 manifest-train sources. Real loss
имел вес `0.25`, four-side border loss — `0.5`. CPU training занял 542 секунды.

Главное отличие от v1 — четыре per-socket border head: right-out, left-in,
bottom-out и top-in. Partial OT по-прежнему имеет точную массу 552 внутренних
пар и 24 непарных socket-а на каждую сторону оси. Входной shuffled tile index
не кодируется позиционно.

Training artifact:
[report.json](../../outputs/socket-matcher/v2-border-train512-s300-r100-dev24/report.json),
SHA-256 `d9f1522a2befc76161b7f5ef25887138db0562ca4db30d1ff778c8fc6413b26f`.
Его top-level поле `experiment` по ошибке осталось со строкой `v1`; фактический
`contract.architecture`, checkpoint и 150 280 параметров соответствуют v2.

## Важное разделение: старый row-renormalized report и новые reruns

Первый training report был записан до исправления conversion-а scores для
layout solver. Его local `R@K`, exact-top-24 border metrics, training history и
checkpoint корректны. Но его global `*_buddies96` и `*_relax_*` arms получили
real-real OT block после повторной row normalization. Эта операция условно
нормировала вероятность только среди 576 внутренних кандидатов и стирала
выученную моделью относительную массу `real match` против `dustbin/border`.

Поэтому global таблицу из первого report нельзя смешивать с результатами
текущего solver path. В двух более поздних checkpoint-only reruns score

`S_ij = log P_OT(i -> j) + log(576 + 24)`

передаётся без повторной row normalization. Decoder получает полный
`577 x 577` log assignment и жёстко проектирует его в ровно 552 пары и 24
непарных socket-а. Именно эти OT-mass-preserving reruns ниже являются
authoritative real-panel сравнением solver-ов. Оба сделали ноль training steps,
то есть проверяют те же веса на новых source-disjoint панелях.

## Local retrieval и border signal

| Панель | Bilateral R@1 | Socket raw R@1 | Socket OT R@1 | Bilateral R@32 | Socket OT R@32 |
|---|---:|---:|---:|---:|---:|
| Fresh-24, offset 1536 | 6.884% | 9.854% | **11.130%** | 40.882% | **55.631%** |
| Confirm-24, offset 2048 | 6.544% | 10.243% | **11.500%** | 39.074% | **54.978%** |

То есть на независимой confirmation панели выигрыш Socket OT составил
`+4.956 pp` по R@1 и `+15.904 pp` по R@32. Это воспроизводит главный вывод v1:
board-conditioned matcher действительно добавляет local signal; bottleneck
находится позже, в absolute anchoring и упаковке компонент.

Four-side exact-top-24 recall у самих border heads равен `6.03%` на fresh и
`7.42%` на confirmation против random expectation `4.17%`. После partial OT он
равен `10.11%` на обеих панелях. Это настоящий per-socket diagnostic, в отличие
от ошибочной aggregate-dustbin метрики v1, но сигнал пока слаб и неоднороден.
Learned-border relaxation не превосходит analytic-border relaxation
устойчиво, поэтому border head ещё не является надёжным absolute anchor.

## OT-mass-preserving global results

Fresh artifact:
[report.json](../../outputs/socket-matcher/v2-checkpoint-fresh-dev24-decoder-prior005/report.json),
SHA-256 `1b2ddc65bdbe3529d0f52446e8832c791d787bd3ee3a7f684b68937c9eb9cd40`.

| Fresh-24 arm | Direct | Translation-aligned | Adjacency | Raw SSIM |
|---|---:|---:|---:|---:|
| Bilateral buddies96 | 0.1230% | 0.9332% | 3.4383% | **0.115953** |
| Socket OT buddies96 | 0.1953% | 1.1574% | 5.8575% | 0.110186 |
| Fused OT buddies96 | 0.1085% | 1.2370% | 5.9330% | 0.113508 |
| Fused relaxation, analytic border | 0.2170% | 1.1140% | 6.5708% | 0.105449 |
| Fused relaxation, learned border | **0.2821%** | 1.0995% | 6.4727% | 0.106827 |
| Socket OT decoder144, no prior | 0.2025% | 1.4251% | **7.7597%** | 0.107319 |
| Decoder144 + texture-centre 0.05 | 0.2459% | **1.4540%** | 7.7446% | 0.108380 |

Confirmation artifact:
[report.json](../../outputs/socket-matcher/v2-checkpoint-confirm-dev24-decoder-prior005/report.json),
SHA-256 `aba24f50af794033d0ef23851c9210a72b55494937c7bae5777e519e426ab79c`.

| Confirm-24 arm | Direct | Translation-aligned | Adjacency | Raw SSIM |
|---|---:|---:|---:|---:|
| Bilateral buddies96 | 0.2170% | 0.9549% | 3.4420% | **0.111025** |
| Socket OT buddies96 | 0.1447% | 1.1936% | 6.2538% | 0.107511 |
| Fused OT buddies96 | 0.2170% | 1.2297% | 6.2840% | 0.108186 |
| Fused relaxation, analytic border | 0.2098% | 1.0706% | 6.6048% | 0.103523 |
| Fused relaxation, learned border | 0.1953% | 1.0851% | 6.6689% | 0.103183 |
| Socket OT decoder144, no prior | **0.3328%** | **1.4612%** | **7.9672%** | 0.105217 |
| Decoder144 + texture-centre 0.05 | 0.2025% | 1.4034% | 7.9597% | 0.104774 |

Base decoder144 — единственный новый conversion arm, который воспроизвёл
сильный geometry gain на обеих real панелях:

- adjacency gain к bilateral: `+4.321 pp` fresh и `+4.525 pp` confirmation;
- translation-aligned gain: `+0.492 pp` и `+0.506 pp`;
- direct gain: `+0.080 pp` и `+0.116 pp`.

Последняя метрика всё ещё слишком мала, а SSIM падает на `0.00863` и `0.00581`.
Это подтверждает decoder как research primitive, а не готовую финальную
раскладку.

## Проверка идеи «уверенное содержимое в центр, однотонное — фон»

`texture-centrality-v1` использует только dirty pixels и работает на уровне
целой rigid component. Выше-медианные по локальной структуре tiles получают
плавное притяжение к центру; tiles около и ниже медианы имеют нулевой unary.
Таким образом, однотонный tile не объявляется фоном и не фиксируется на рамке —
он просто не получает центрального бонуса.

На fresh-24 prior `0.05` относительно base decoder дал `+0.0434 pp` direct,
`+0.0289 pp` translation-aligned и `+0.001061` raw SSIM, но adjacency был на
`0.0151 pp` ниже. На неизменённой confirmation-24 он проиграл base decoder по
всем четырём метрикам: direct `-0.1302 pp`, translation-aligned `-0.0579 pp`,
adjacency `-0.0075 pp`, raw SSIM `-0.000444`.

Вердикт: observed first-panel gain был selection noise. Prior остаётся
доступным ablation hook, но `component_prior_weight=0.0` — обязательный default.
Повторять isolated-tile face/centre hard placement или автоматически отправлять
гладкие tiles на границу не нужно.

## Exact synthetic confirmation

Отдельный fail-closed evaluator использовал восемь clean manifest-train
источников вне 512-source checkpoint lineage, одну независимую official-like
corruption draw и точную inverse-shuffle label. Он также использует
OT-mass-preserving conversion. Полный protocol и таблицы находятся в
[отчёте exact synthetic](socket-matcher-exact-synthetic.md); artifact:
[report.json](../../outputs/socket-matcher/exact-synthetic-v2-source8-draw1/report.json),
SHA-256 `9358828611135909905c0074f39d4a67e93ef244657e3f2f4f44807330069df3`.

Ключевые числа:

- fused local R@1: `8.37% -> 13.72%`, R@32: `52.03% -> 64.78%`;
- Socket OT buddies adjacency: `4.05% -> 5.96%`;
- fused relaxation adjacency: `7.53%`;
- base component/QAP decoder144 adjacency: `8.40%`,
  translation-aligned placement `1.50%`;
- decoder direct placement: `0.217%`, всё ещё около chance.

Exact panel устраняет ambiguity recovered real labels и подтверждает matcher +
decoder signal. Восемь sources и одна draw не измеряют variance, поэтому это не
promotion evidence.

## Зафиксированное решение

- Сохранять v2 checkpoint как лучший текущий contextual local matcher.
- Сохранять `socket_ot_decoder144` без semantic prior как основной research
  conversion arm: он воспроизвёл adjacency/translation gain на трёх панелях.
- Не цитировать global arms первого training report как результаты текущего
  OT-mass-preserving solver; local и border diagnostics оттуда остаются
  корректными.
- Не включать texture-centre prior, learned-border relaxation или этот d32
  pipeline в submission сейчас.
- Следующий полезный scale test — более ёмкий matcher и больше interleaved exact
  synthetic supervision при фиксированном decoder144, с multi-draw exact
  source-disjoint evaluation. Главный gate — direct placement и final SSIM, а
  не ещё один небольшой рост local R@K.
