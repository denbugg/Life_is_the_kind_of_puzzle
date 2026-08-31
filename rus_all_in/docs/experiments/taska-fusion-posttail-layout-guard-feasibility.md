# TASKA learned post-tail fusion guard: feasibility stop

Дата фиксации: 2026-08-31. Статус: **stop до fit/evaluation из-за
недостаточного minority class**.

## Вопрос и аудит повторов

Проверялась узкая идея: после двух уже frozen legal final layouts —
`selective-target500` и `selective + unique-fullres fusion`, оба после своего
focal-gated tail96 — один маленький target-free classifier должен выбрать
целый layout, не смешивая tiles.

Перед экспериментом проверены ближайшие selector-ветки:

- exact portfolio proxies выбирали focal/pair-leader по frozen focal-logit или
  structural-border objective и не перенесли exact;
- direct rank-delta component selector использовал фиксированную component
  geometry, а на fresh64 проиграл always-rank по exact;
- fixed multistart выбирал минимум original all-bond cost среди 16 layouts и
  ухудшил pairs/exact;
- self-satisfied pair selector улучшил pairs, но резко ухудшил exact.

Точного дубликата не найдено: новый guard впервые обучался бы прямо на том,
какой из **двух final post-tail layouts** имеет больше satisfied pairs, используя
только уже frozen board-level признаки. Но дальше сработал отдельный
preregistered feasibility gate.

## Oracle ceiling

По существующим frozen scored rows посчитан `max(selective final, fusion
final)` без обучения selector-а:

| Panel | Selective pairs | Fusion pairs | Oracle pairs | Oracle − fusion | Selective / fusion / tie cases |
|---|---:|---:|---:|---:|---:|
| local32 | `323.62500` | `326.78125` | `327.46875` | **`+0.68750`** | `1 / 8 / 23` |
| held32 | `343.09375` | `345.31250` | `345.78125` | `+0.46875` | `3 / 5 / 24` |
| fresh32 | `354.09375` | `355.62500` | `355.62500` | `0.00000` | `0 / 5 / 27` |
| formal32 | `330.03125` | `333.12500` | `333.31250` | `+0.18750` | `2 / 7 / 23` |

Заранее заданный eligibility rule требовал хотя бы `+0.5` пары/board на одном
fit/eval aggregate. Local32 с `+0.6875` его проходит, поэтому идея не была
закрыта как заведомо бессодержательная.

## Почему model fit запрещён

Фиксированный fit должен был использовать только local32+held32 (`64` rows),
label — победитель по final post-tail satisfied pairs; ties не являются
информативными non-tie labels. Получилось:

- selective wins: **`4`**;
- fusion wins: **`13`**;
- ties: **`47`**.

Selective-win rows принадлежат лишь трём sources; одна local row несёт сразу
`+22` пары, а две held rows приходятся на два draws одного source. Gate требовал
не меньше `8` non-tie rows **каждого** класса. Значит, даже
`class_weight=balanced` не исправил бы отсутствие разнообразия: logistic guard
в основном учился бы отделять три source-specific исключения. По протоколу
эксперимент остановлен до `StandardScaler + LogisticRegression` fit, без
threshold/C/feature sweep и без оценки на fresh/formal.

Это не проблема отсутствующих признаков. Frozen rows содержат все заранее
разрешённые поля: pre-tail original costs и margins, final tail costs,
unique-fullres count, focal-kept/protected deltas и combined-arm choice.

## Решение и no-repeat boundary

Статус: **`feasibility-stop-insufficient-minority-class`**.

- Weco steps `107/108` не логировались: фактической candidate evaluation не
  было.
- Production/default, matcher, denoiser, pixels и submission не менялись.
- Fresh32 и formal32 после oracle-а нельзя задним числом превращать в fit.
- Не форсировать модель, не oversample-ить четыре exceptions и не подбирать
  threshold на уже открытых rows.
- Возвращаться к learned guard можно только с новым заранее подписанным
  source-disjoint **fit** roster, где до открытия evaluation будет хотя бы
  `8/8` non-tie rows обоих классов. До этого confirmed fusion остаётся
  безусловным pair-oriented control.

Machine-readable feasibility report:
`outputs/taska-fusion-posttail-layout-guard/feasibility-v1/report.json`.

