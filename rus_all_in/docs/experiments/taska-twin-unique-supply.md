# TASKA FullResolutionTwin unique candidate supply

Дата: 2026-08-31. Статус: **held gate не пройден; fresh не открыт**.

## Вопрос и fixed contract

Эксперимент проверил не уже отрицательный standalone Twin ranker, а только его
дополнительный candidate supply поверх frozen
`selective-target500 + unique-fullres` fusion. До target scoring был
зафиксирован один итоговый contract:

1. frozen FullResolutionTwin напрямую номинирует свой row-top32;
2. остаются только рёбра из первых 144 confidence-sorted hard-projection edges
   на каждую ось frozen Union-v2;
3. удаляются все рёбра, уже присутствующие в parent combined union;
4. recovered `train_exact_top5` focal verifier принимает только `logit >= 0`;
5. принятые рёбра добавляются одним седьмым
   `twin_unique_union_focal` arm;
6. selector по-прежнему минимизирует исходную TASKA стоимость всех 1,104
   связей, после чего применяется unchanged focal-gated tail96.

Raw dense costs, первые шесть layout arms, parent final control и все
гиперпараметры оставались frozen. Twin/Socket использовались только как
matcher views; результат всегда был строгой перестановкой исходных upright
20x20 tiles.

Механический one-case smoke сначала обнаружил, что более узкая формулировка
«доказуемо Twin-only относительно Socket raw32/raw-hard» даёт ноль кандидатов.
Reference тогда не восстанавливался. До любого target scoring preregistration
была один раз исправлена на прямую проверку фактического Twin top32 membership;
top144 budget, focal threshold, parent exclusion, selector и tail не менялись.
Финальный target-free smoke дал 134 proposals и 7 accepted edges, после чего
contract больше не менялся.

## Результат

| Panel | Parent pairs / exact | Twin candidate pairs / exact | Pair delta, source CI95 | Exact delta, source CI95 |
|---|---:|---:|---:|---:|
| local32 | `326.781 / 5.938` | `328.406 / 7.656` | **`+1.625 [0.375,3.125]`** | `+1.719 [-0.688,5.813]` |
| held32 | `345.313 / 1.906` | `345.531 / 1.813` | `+0.219 [-1.781,2.625]` | `-0.094 [-0.469,0.188]` |

Local pair W/T/L был `5/27/0`, и local gate `delta >= 0` прошёл. На held
получен только `+0.21875` pairs/board при заранее заданном gate `+0.5`; W/T/L
`1/29/2`. Поэтому fresh32 корректно не открывался. Положительный local exact
создаётся редкими boards, interval пересекает ноль и на held знак меняется.

## Supply diagnostic

| Panel | Proposed / board | Proposed precision | Accepted / board | Accepted true / board | Accepted precision |
|---|---:|---:|---:|---:|---:|
| local32 | `114.188` | `9.06%` | `5.313` | `2.438` | `45.88%` |
| held32 | `114.938` | `9.68%` | `5.344` | `2.188` | `40.94%` |

Focal gate действительно отбрасывает почти весь очень слабый Twin supply и
поднимает precision примерно в 4–5 раз. Однако оставшиеся 5.3 edges/board всё
ещё недостаточно стабильны: новый arm выиграл selector на `5/32` local cases и
`3/32` held cases, но held pair transfer оказался ниже gate.

## Решение и no-repeat

Ветку сохранить как **полезный negative/diagnostic**, но не включать в текущий
pair leader, production или submission. На открытых local/held panels не
ослаблять Twin rank/budget, focal threshold, parent-exclusion, selector roster
или tail budget и не повторять standalone Twin averaging/ranking. Новая попытка
имеет смысл только с materially новым precision mechanism и новым
source-disjoint panel; простой top-k/threshold sweep уже не авторизован.

## Freeze и артефакты

Для каждого panel сначала записаны candidate layouts/edge identities,
target-free metadata и `pre-score-freeze.json`; references восстановлены только
после SHA audit. Competition test, postprocess и rendered/restored pixel output
не использовались.

- preregistration:
  `configs/taska_twin_unique_supply_preregistered_v1.json`, SHA
  `05ac95769646a569573dafdccb4082e5ba33da063f7742597f5cee8bfbb0df53`;
- report:
  `outputs/taska-twin-unique-supply/fixed-v1/report.json`, SHA
  `3dd619c93561d0bb9971d92c40c84d82c104d098217a8a8eb1290f0edafd71d8`;
- local archive / metadata / freeze:
  `adf8e1d4...e48e / a4399716...8204 / 6d84250a...b79d`;
- held archive / metadata / freeze:
  `978f57c3...ebc1 / a820ccf6...362e / 76bf6ca1...ca7d`;
- module / runner SHA:
  `60200f76...538 / 27b9d0d7...94a`.

Запуск:

```bash
.venv/bin/python scripts/run_taska_twin_unique_supply.py --device cpu
```

Mechanical tests: `tests/test_taska_twin_unique_supply.py`. Weco Observe:
steps `99/100`, parent branch `97 -> 99 -> 100`; step101 намеренно не
логировался, потому что fresh не запускался.

