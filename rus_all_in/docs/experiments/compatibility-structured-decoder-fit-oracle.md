# Compatibility-aware structured decoder: FIT oracle screen

Дата: 2026-08-31. Текущий статус: **FIT capacity gate не пройден; ветка
останавливается до structured-model training**. Это только organizer-train FIT
capacity audit. Он не является solver result и никогда не может быть production
policy.

## Почему это не повтор прежних component/consensus веток

Перед формулировкой action space проверены ближайшие реальные попытки:

| Ветка | Что уже проверили | Почему не повторяем |
|---|---|---|
| HGB-ranked all-edge union | Принудительное добавление примерно `3992` unique edges/board обрушило pairs `326.750→199.500`, delta `-127.250` | Нельзя превращать independent relation score в безусловный pooled tail |
| Unique-fullres translation consensus | `8` сигналов на `4/32` boards, `6/8` true, но whole-arm selector всегда скрыл затронутый arm; итоговый delta ровно `0` | Простое priority promotion внутри шести готовых arms закрыто |
| Dense top-8 reciprocal translation consensus | Частый supply, но precision лишь `10.858%` FIT / `11.612%` local | Rigid agreement broad top-8 не является clean anchor head |
| Majority component arm | Consensus-tail потерял `18.469` pairs и `2.844` exact/board | Agreement коррелированных layouts закрепляет общий wrong core |
| Cross-arm component anchor | `-8.375` pairs и `-0.250` exact/board | Один large-component move без явной incremental pair utility небезопасен |
| Component relation anchor | Переносимый маленький pair gain `+0.375/+0.313/+0.156`, но exact `+0.063/0/0` | Single-move seam veto полезен как primitive, но не конвертирует origin/exact |
| Joint component pose | Conditional R@5 signal, но packed layout дал `-66.469` pairs | Absolute/component repacking без pair-preserving action objective закрыт |

Новый screen materially distinct: он начинает с неизменного confirmed
relation-selector layout, принимает только уже замороженный reciprocal top-5%
head, имеет явный `STOP`, проверяет strict feasible edit и запрещает каждый
incremental pair loss. Никакого all-edge tail, threshold/top-k sweep, нового
matcher score или absolute pose prediction здесь нет.

## Fail-closed input contract

Consumer принимает только три артефакта новой joint reciprocal ветки:

- `fit/frozen-target-free-reciprocal-heads.npz`;
- matching metadata schema
  `aiijc-joint-reciprocal-target-free-fit-heads-v1`;
- pre-score freeze, который байтово фиксирует первые два файла до label score.

Для каждого из exact `32 sources × 2 draws = 64` immutable FIT cases нужны
ровно `ceil(.05×576)=29` learned reciprocal edges на каждую axis. Source и
target identities обязаны быть уникальны внутри axis; `target_slots`, truth,
reference, correctness и label arrays запрещены. Case id, source, draw, dirty
SHA, cache SHA и union digest должны совпасть с исходным tri-emitter FIT cache.
Consumer также проверяет metadata protocol SHA, endpoint/runner/module
provenance, dtype/shape/rank order всех head arrays, достаточный reciprocal
count и точный NPZ field inventory: schema не может молча потерять или добавить
поля.
Старый opened `local16` не является fallback и не читается.

До появления и проверки этого exact archive runner умеет только вернуть
`blocked-missing-fixed-fit-head`; он не запускает старый checkpoint, matcher,
solver или labels.

## Зафиксированный первый action space

После доступности head отдельный preregistration связывает его SHA-256, полный
ordered FIT roster, неизменные confirmed six-arm solver bytes и следующий
единственный oracle contract до reference scoring:

1. Control — целая strict relation-selector layout, не пустая board.
2. Proposal roster — только fixed-head edges, которые после freeze оказались
   exact true и ещё не реализованы control-ом.
3. Oracle-correct relations, уже реализованные текущим state, образуют rigid
   components. Atomic feasible edit переносит целиком либо source-, либо
   target-component на edge-implied integer shift; непосредственно вытесненные
   tiles bijective заполняют освободившиеся клетки.
4. Edit отвергается при out-of-frame span, internal/identity collision, нарушении strict
   permutation, отсутствии net supplied-true-edge gain или incremental pair
   delta `<0`.
5. Oracle tie-break фиксирован: true-edge gain, pair delta, exact delta,
   Manhattan improvement, radius2 gain, frozen confidence, меньший component,
   target-before-source и current grid positions/shift. Последний tie-break
   сохраняет tile-id relabel equivariance.
6. `STOP` всегда допустим; поиск заканчивается, когда safe edit отсутствует.

Это target-assisted **capacity ceiling**, а не deployable decoder. Truth
используется одновременно для фильтра true proposals и utility tie-break;
поэтому даже успешный screen только разрешит следующий source-disjoint model,
но ничего не промоутит.

## Gate и диагностики

Один gate, без nearby variants:

- mean compatible missing true-edge supply `>=8` на board;
- mean actually realised supplied-true-edge gain pair-safe ceiling-а `>=8`;
- pair delta неотрицателен на каждой board.

Companion diagnostics: exact tiles, absolute mean Manhattan и absolute radius2
recall. Все control/ceiling layouts проверяются как перестановки всех `576`
original upright tile identities ровно по одному разу. Pixel output отсутствует.

## Реализация

- pure oracle accounting:
  `src/aiijc_puzzle/structured_decoder_fit_oracle.py`;
- staged runner:
  `scripts/audit_structured_decoder_fit_oracle.py`;
- focused tests:
  `tests/test_structured_decoder_fit_oracle.py`.

Ruff и `8` focused tests проходят, включая зафиксированный `6×6`
compatible/incompatible graph, forced `STOP` и tile-id relabel equivariance.

## Результат signed FIT-only audit

Strict target-free freezer не материализовал `target_slots`; structured config
был подписан до FIT-head label score. Сам fixed 5% head оказался очень чистым:
`3341/3712 = 90.005%` pooled precision (`87.985%` right, `92.026%` down).
Однако это не новый supply: confirmed control уже реализовал почти все его
истинные связи.

| Метрика, mean/board | Control / supply | Pair-safe ceiling / delta |
|---|---:|---:|
| selected true head edges | `52.203` | — |
| already realised true head edges | `51.031` | — |
| missing true head edges | `1.172` | — |
| compatible missing headroom | `1.047` | gate `>=8`: **fail** |
| realised supplied-edge gain | — | `+0.891`, gate `>=8`: **fail** |
| satisfied pairs | `349.484` | `350.859`, delta `+1.375` |
| exact tiles | `2.672` | `2.672`, delta `0.000` |
| absolute Manhattan | `15.508247` | `15.507975`, delta `-0.000271` |
| radius2 recall | `0.023166` | `0.023193`, delta `+0.000027` |

Pair safety выдержана на всех boards: pair W/T/L = `29/35/0`, minimum delta
`0`, maximum `+8`. Но exact W/T/L = `0/64/0`; mean, median, Q25 и Q75 exact
delta равны нулю. В absolute ceiling `37/64` boards имеют zero exact и `51/64`
имеют exact `<=1`. Положительного exact tail нет вообще, поэтому
max-positive share и leave-largest-positive mean также нулевые.

Вывод: clean reciprocal verifier полезен как уже согласованный pair signal, но
на confirmed six-arm control он почти полностью избыточен. Первый
compatibility-aware decoder не имеет preregistered `+8` action-space capacity,
поэтому structured model/threshold/beam sweep не открывается. Если вернуться к
направлению позже, нужен source-disjoint head с **новыми** истинными рёбрами или
action family, способный безопасно исправлять уже неверно реализованный control,
а не ещё один consumer тех же contacts.

Финальные SHA-256:

- structured config: `6985fe5548af48adc9ee8628179259b4863dbed8a37f9ccebbd993067d49b7e1`;
- fixed head archive: `f2a60ccfc5d8d09d25e4648bf7226e1364b3bd2575275be6bb6a7b9ac562257c`;
- controls archive: `f567be7e1c9e2b5fa4cb64d896c709ebd0fa1595fa52b4f6d40106c2a6c228d0`;
- primary report: `bc3ad541df586911d14d90d74d599e2528085c9c62d2e71c66bdafe0bdb88621`;
- read-only exact-tail audit: `311770853a91a882419f779114256289122c79a542ea85837f71a90cbba01042`.

Ни `local16`, ни `terminal16`, ни competition test/submission не открывались;
output pixels отсутствуют, все `64` control/ceiling layouts — strict
перестановки `576` original upright tiles. Weco не использовался.
