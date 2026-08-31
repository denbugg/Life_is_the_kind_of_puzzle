# Exact mean tail-risk audit

Дата: 2026-08-31. Статус: **обязательная корректировка интерпретации;
12.875 exact не является устойчивым solver-result**.

Проверен Weco exact headline `12.875 tiles/board` из fixed TASKA six-arm +
Socket cyclic-border5 на уже открытом local32. Арифметика корректна, но
распределение крайне тяжёлохвостое.

| Metric | Six-arm control | Cyclic-origin candidate |
|---|---:|---:|
| mean exact | 5.9375 | 12.8750 |
| median exact | 1.0 | 1.0 |
| Q25 / Q75 | 0 / 2 | 0 / 1.25 |
| zero-exact boards | 13/32 | 14/32 |
| boards with <=1 exact | 22/32 | 24/32 |
| maximum | 74 | 256 |
| mean after removing maximum | 3.742 | 5.032 |

Paired delta был `+6.9375` mean, но `0` median и W/T/L `4/22/6`.
Крупнейший `+256` sample дал `83.93%` всей положительной массы; после удаления
только этого positive outlier mean delta становится `-1.097`. Он также входит
в Socket-checkpoint lineage. На lineage-disjoint26 mean candidate exact равен
`5.577`, median всё ещё `1`, W/T/L `3/19/4`; крупнейший `+42` даёт `85.71%`
positive mass, а leave-largest-positive mean delta остаётся лишь `+0.04`.

Вывод: rare global-origin successes настоящие и полезны как mechanism signal,
но headline mean нельзя называть общим качеством solver-а. Этот endpoint уже
провалил pair floor и теперь дополнительно помечен как exact-tail fragile.

С этого момента каждый exact отчёт обязан содержать как минимум:

- mean, median, Q25/Q75, zero fraction и `<=1` fraction;
- W/T/L paired delta и source-group bootstrap CI;
- maximum, долю positive mass крупнейшего source и leave-largest-positive mean;
- отдельно parent-lineage-disjoint срез, если checkpoint ancestry пересекается;
- pairs, absolute Manhattan и radius2 на тех же frozen layouts.

Mean остаётся важной ожидаемой метрикой: для global-origin решения естественно
бывают all-or-nothing. Но outlier-dominated mean разрешает только продолжение
исследования; promotion требует повторяемости по независимым sources/draws и
неотрицательной геометрии.

Source report:
`outputs/taska-socket-cyclic-origin-transfer/local32-v1/report.json`.
