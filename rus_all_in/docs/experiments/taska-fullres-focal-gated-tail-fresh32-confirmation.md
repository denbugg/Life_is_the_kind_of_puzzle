# TASKA fullres union + focal-gated tail96: disjoint fresh32 confirmation

## Итог

Одна заранее зарегистрированная end-to-end панель подтвердила фиксированную
композицию fullres candidate supply и focal-gated tail96. На 16 новых sources
с двумя независимыми draws каждый итоговый combo получил **356.3125**
правильных соседних пар на board против **348.40625** у same-pass текущего
four-arm+all-edge-tail96 control:

- pair delta **+7.90625**;
- source-cluster bootstrap CI95 **[+3.53125,+12.96875]**;
- case W/T/L `24/0/8`, source W/T/L `13/0/3`;
- adjacency recall `0.315585371→0.322746830`, то есть **+0.71615 pp**.

Заранее заданный gate требовал mean delta `>=+2.0` и CI95 lower `>=0`.
Обе границы пройдены с запасом, статус report — **confirmed**.

Это подтверждение pair-oriented solver-рецепта, а не официальный submission и
не exact-oriented promotion. Production pipeline этим запуском не менялся;
официальный best SSIM `0.2762279116935955` остаётся без изменений.

## Фиксированные arms и ablations

Все три arms использовали один и тот же target-350 TASKA matcher pass, один
current harvest и исходные dense raw seam costs:

1. four-arm raw/logistic/focal-top5/nonlinear selector + all-edge tail96;
2. тот же portfolio с пятым `fullres_union_focal` arm + all-edge tail96;
3. тот же самый five-arm pre-tail winner + focal-logit-zero protected tail96.

Fullres denoiser использовался только для matcher view. Новое ребро должно
было отсутствовать в current harvest, получить support `>=3/4` от фиксированных
restored v3/local × two-orientation scorers и recovered
`train_exact_top5` focal logit `>=0`. Для финального tail protection применена
та же естественная граница `logit>=0`, `max_swaps=96` и
`minimum_gain=1e-9`. Threshold, support, orientation, budget, arm или roster
не перебирались.

| Arm | Pairs/board | Adjacency recall | Exact tiles/board |
|---|---:|---:|---:|
| four-arm + all-edge tail96 | 348.40625 | 0.315585371 | 8.00000 |
| five-arm fullres + all-edge tail96 | 353.84375 | 0.320510643 | 8.00000 |
| five-arm fullres + focal-gated tail96 | **356.31250** | **0.322746830** | 8.00000 |

Обе части композиции перенеслись независимо:

- fullres − control: **+5.4375** pairs, CI95 `[+1.25,+10.6875]`;
- focal-gated combo − fullres: **+2.46875** pairs, CI95
  `[+0.34375,+4.6875]`.

Mean exact одинаков и равен `8.0` для всех arms. Paired exact delta для combo
против control равен `0.0`, CI95 `[-0.34375,+0.375]`; exact не использовался
для gate и не даёт основания объявлять exact improvement.

## Почему появился прирост

Current harvest содержал в среднем `376.31` ребра, `273.72` из них истинные:
precision `72.74%`, candidate recall `24.793%`. Широкий restored pool давал
ещё `233.66` отсутствующих proposals/board с precision всего `17.48%`.
Пересечение с frozen focal gate оставило `32.53` ребра при precision
**62.34%**, из них **20.28** истинных новых соседств на board. Union recall
вырос до `26.630%`, то есть на `+1.837 pp`.

Fullres arm выиграл raw-cost selector на `8/32` случаях. Отдельный focal gate
затем уменьшил среднее число protected tiles `387.31→309.88` и увеличил
использованные tail swaps `84.91→94.06`. Поэтому второй gain — активная
дополнительная свобода tail, а не случайный no-op.

## Независимость панели и freeze

Roster был подписан до inference в
`configs/taska_fullres_focal_gated_tail_fresh32_confirmation_v1.json`. Он
выбран SHA-ranking из `img_006400..img_006699` после исключения:

- всех sources из подписанного tail192 reservation и его signed dependencies;
- всех явно встречавшихся sources из отдельно подписанной fullres lineage.

Получились 16 sources × draws `0/1`; source-order digest
`3120b719d7cbf496f5505e0459ecdf597a98637c841c8ef843eb11945adf6c1a`,
case digest
`999667ad4ee95b35cc76537a7e1b99f3f3d96b2dc92c8c42eca80baeef0ac745`.

Все costs, candidate edges, logits и три strict layouts были записаны в NPZ,
а полный artifact manifest — в pre-score freeze до восстановления exact
synthetic references. Bootstrap unit — source с двумя draws, `20 000`
resamples. Панель открывалась ровно один раз.

## Легальность

- Каждый output — строгая permutation всех 576 исходных upright 20×20 tiles.
- Restored pixels применялись только matcher-ом и не рендерились в результат.
- Targets/labels не участвовали в candidate inference и появились только при
  offline scoring после freeze.
- Нет rotations, warps, replacements, constant tiles или postprocessing.
- Competition test не открывался.

## Воспроизведение и артефакты

```bash
.venv/bin/python \
  scripts/run_taska_fullres_focal_gated_tail_fresh32_confirmation.py \
  --device mps --allow-nondeterministic-mps
```

- report SHA-256:
  `6f02f83407eb15f42c696d1f458b658465d94bfd5c3087135cfc6cbcd548ecf3`;
- target-free NPZ / metadata / pre-score-freeze SHA-256:
  `ba1dbe99c76b00a4497f2658636394a4c5ad0268062f08fc721ef208897c03c3`,
  `b74dc4e8f115c62aad32093b1aefcb5c76721669e2753b9bf8e14e9786b6aedc`,
  `ac711ca1c56da57b62e681464657443b7e452f8982901c98b7ae241ee182ee10`;
- preregistration SHA-256:
  `742a427ba5cc844afedb7107c39bd0ba9cbe246fad4daf0fd6ae959cf4846794`;
- runner SHA-256:
  `957209c92f7057d34eef93c1240fa06d18f1093f54cbe2907e9bc54dbfca000b`;
- frozen raw solver остался
  `97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.

Runtime target-free части — `146.28 s`, полный runtime — `146.99 s` на MPS.
Weco Observe: pair и exact step `94`, parent step `88`; pair run дополнительно
содержит adjacency recall.
