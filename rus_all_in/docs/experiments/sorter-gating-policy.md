# Tile-sorter gating: discovery отдельно от promotion

Дата изменения: 2026-08-30. Пользователь отдельно потребовал повысить
чувствительность к положительному сигналу и не закрывать потенциально сильные
направления слишком строгим ранним порогом.

## Почему правило изменено

Ранние эксперименты иногда использовали почти production-level требования уже
на local gate. Это смешивало два разных вопроса:

1. есть ли в механизме новая информация и стоит ли тратить следующий небольшой
   эксперимент;
2. достаточно ли доказательств, чтобы заменить default или собрать submission.

Например, component-relation head дал source-disjoint R@1 `+3.366 п.п.` и R@5
`+4.934 п.п.`, но не прошёл изначально строгий top-32 confidence gate. Такой
результат нельзя promote, но также нельзя считать бесперспективным и закрывать.

## Три уровня решения

### D1 — discovery / продолжить исследование

Здесь нужен чувствительный, дешёвый gate. Направление сохраняется и получает
следующий bounded шаг, если на matched source-disjoint данных есть хотя бы один
релевантный положительный сигнал без крупной встречной регрессии. Типичные
ориентиры, фиксируемые до конкретной оценки:

- pooled local R@1 `+0.25 п.п.` при неотрицательном R@5;
- matched reciprocal precision `+1 п.п.`;
- top-32 candidate coverage `+1 п.п.` хотя бы в одном направлении без потери
  больше `0.5 п.п.` в другом;
- global development exact `+0.1` tile/board или устойчивое улучшение
  row/column/component evidence, которое новый mechanism прямо должен давать.

Это не универсальное логическое OR для любого отчёта: до запуска выбирается
метрика, причинно связанная с гипотезой. CI выше нуля на D1 не обязателен.
Положительный supply-only результат можно сохранить для другого ranker/context
model, даже если текущая ranking head провалилась.

### D2 — маленький decoder/exact pilot

Если D1 показывает новую информацию, разрешён небольшой заранее зарезервированный
exact-synthetic decoder panel. Для него не требуется заранее доказанный
production effect или положительная CI. Обязательны:

- source-disjoint known-shuffle truth;
- frozen candidate и decoder configuration до scoring;
- строгая перестановка всех 576 original upright tiles;
- primary `correct_tile_count`/board, adjacency и runtime;
- отсутствие competition-test access.

D2 нужен именно потому, что local R@k и confidence могут не предсказывать
конверсию в absolute positions. Небольшой exact pilot дешевле, чем ошибочно
закрыть новый relational signal.

### D3 — confirmation / default promotion

Только здесь остаются строгие требования. Они preregistered для конкретного
кандидата; текущий типичный ориентир:

- mean exact gain не меньше `+0.5` tile/board;
- source-clustered 95% CI lower bound выше нуля;
- adjacency loss не больше `0.2 п.п.` или отдельный заранее обоснованный
  geometry trade-off;
- strict permutation/compliance audits;
- legal post-layout full-cycle SSIM/runtime report;
- новый source-disjoint confirmation panel без tuning на нём.

Лишь D3 может менять default/submission. Descriptive D1/D2 gains остаются
research evidence.

## Когда всё же останавливаем рано

Низкий discovery threshold не означает продолжать любой run. Ветка закрывается,
если signal явно отрицательный или механизм причинно не работает, например:

- E13 R@1 `18.65→6.88%`;
- transpose continuation exact `−0.328` tile/board;
- restored ranker matched precision `−3.246 п.п.` при flat R@1, хотя его
  положительный candidate-supply emitter сохранён отдельно.

Пороги нельзя ослаблять постфактум на уже увиденном panel. Изменение этого
документа применяется к новым continuation experiments и фиксируется до
доступа к их confirm/exact labels.
