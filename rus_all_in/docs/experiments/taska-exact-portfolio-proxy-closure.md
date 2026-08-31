# TASKA exact-oriented portfolio proxies: bounded closure

Дата фиксации: 2026-08-31.

## Вопрос и жёсткая граница поиска

На held300 focal top-5 был лучшим exact-arm (`4.00000`
tile/board), а three-arm all-bond portfolio + protected-tail — лучшим
pair-arm (`337.03125` pairs, но только `3.15625` exact). Проверено,
может ли простой target-free selector сохранить exact-выигрыш
focal без отрицательной pair-дельты.

Поиск заранее ограничен двумя мотивированными правилами на
уже открытом opened32. Из них до held300 фиксировалось ровно
одно — лучшее по opened exact без дополнительного threshold/weight
sweep. Ни одно правило не принимает target, exact permutation,
filename, source coordinate или canonical tile id.

## Четыре frozen legal arm

- `raw`: исходный TASKA raw-priority layout;
- `calibrated`: frozen train256 logistic edge-priority layout;
- `focal`: recovered verifier с exact training contract top-5;
- `pair-leader`: minimum original TASKA all-bond cost среди
  `raw/calibrated/focal`, затем fixed 24-swap protected-tail polish.

Каждый arm и каждый selected output — строгая перестановка всех
576 исходных upright 20×20 tiles. Пиксели, matcher costs, candidate
membership и frozen raw solver не менялись.

## Proxy 1: focal realised-logit objective

Для focal и pair-leader суммировался frozen focal logit всех
harvested edges, которые layout реально ставит в запрошенном
направлении. Выбирался больший score; tie сохранял pair-leader.

На opened32 правило выбрало focal/pair-leader `13/19` раз и дало:

- `336.75000` pairs, recall `0.305027174`;
- `4.34375` exact tiles.

Это хуже pair-leader и по pairs (`338.68750`), и по exact (`4.65625`).
Proxy закрыт на opened32 и не переносился на held300.

## Proxy 2: audited structural-border score

Для каждого dirty tile bag заново строился уже audited
`structural_border_unary`: SHA-locked TASKA v3+local raw logits, slack
Sinkhorn `slack=6`, `20` iterations, без target и content labels. Score layout
— сумма unary для фактически занятых border positions. Выбирался
больший score; tie сохранял pair-leader.

Это правило было лучшим из двух по opened exact и поэтому
фиксировалось до held300.

| Panel | Focal / leader choices | Selected pairs | Recall | Selected exact | Pair-leader pairs / exact | Focal pairs / exact |
|---|---:|---:|---:|---:|---:|---:|
| opened32 | 15 / 17 | 336.31250 | 0.304630888 | 4.46875 | 338.68750 / 4.65625 | 335.50000 / 4.34375 |
| held300 unchanged | 16 / 16 | 335.25000 | 0.303668478 | 3.46875 | 337.03125 / 3.15625 | 332.53125 / 4.00000 |

На held300 proxy остался pair-positive относительно raw
(`329.62500`) и focal (`332.53125`) и вернул `+0.31250` exact от
pair-leader. Но он не сохранил полные `4.00000` focal exact, а на
opened32 exact-дельта от pair-leader была отрицательной
(`-0.18750`). Знак exact не перенёсся.

## Verdict

`bounded-closure; no production selector`.

Оба простых target-free proxy проверены, но ни один не дал
переносимого exact gain против pair-leader. Не следует свипать
logit thresholds, border weights, margins или combinations на этих уже
opened panels. До fresh source-disjoint gate сохранятся два раздельных
arm: focal top-5 как exact-oriented diagnostic и three-arm+tail как pair
leader.
