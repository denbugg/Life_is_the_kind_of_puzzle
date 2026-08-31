# Content-aware listwise verifier

Статус: **финальный scale-up завершён; formulation закрыта как глобальный
verifier**. Checkpoint оставлен только как исследовательский exact-edge
auxiliary. Decoder, placement и full-image SSIM не запускались.

## Чем это отличается от M419

M412/M419 сравнивали пять узких seam patches и классические scalar scores.
Даже на 2,248 досках M419 поднял число правильных bonds лишь с 331.3 до 338.5
и извлёк около 4% shortlist headroom. Поэтому ещё одно масштабирование
seam-only chooser закрыто предыдущими экспериментами.

Здесь каждый pair строится из полных 20x20 dirty tiles. После convolutional
stem anchor и candidate представлены общим 5x5 spatial grid, совместно проходят
pair Transformer, а второй permutation-equivariant Transformer сравнивает весь
shortlist из примерно 14 кандидатов. Loss listwise и multi-positive: допустимы
content twins с clean RGB RMSE <= 20.

## Две фазы и исправление pilot

### Pilot: 32 train / 12 calibration

Pilot показал небольшой устойчивый exact signal, включая старый holdout 0:12.
Однако последующий audit нашёл, что в первой реализации stem tokens не имели
spatial positional encoding. Pair Transformer видел spatial patches как
множество, а не как 5x5 grid. Поэтому pilot нельзя считать проверкой задуманного
spatial cross-attention. Старые `calibration32x12.pt/json`,
`calibration12-confirm.json` и `holdout12.json` сохранены только как история.

### Final frozen scale-up: 128 train / fresh calibration 12:36

До финального запуска были зафиксированы ровно следующие correctness fixes:

1. общий learned positional embedding `25 x D` для anchor/candidate 5x5 grids;
2. у каждого кандидата хранится target-assisted mapping margin;
3. RMSE<=20 training positive разрешён только при candidate mapping margin не
   ниже board median; exact neighbour остаётся positive на trusted query;
4. shared manifest selector получил `eval_offset`, поэтому calibration 12:36
   не пересекается с pilot 0:12;
5. архитектура и roster после этого не менялись: union top-5, ordered views
   `raw, tile_z, bilateral, gray`, dim 32, один pair и один listwise layer,
   seed 20260829, LR 3e-4.

Frozen run: первые 128 manifest-train досок, 5 эпох, до 192 rows/board,
batch 64; evaluation — 24 свежие calibration-доски с offset 12.

После получения результата audit только ужесточил отчётность: добавил
right/down gates, strongest-classical comparator и strict content label
validity. Это не меняло checkpoint или predictions и не могло превратить уже
проваленный all-content gate в успех.

## Leakage boundary и trusted scopes

- Candidate emitters и модель видят только corrupted shuffled pixels и
  classical costs, вычисленные из них.
- Clean target используется только для recovered mapping, margins, exact и
  RMSE labels на train/eval.
- `trusted_query` требует высоких margins у anchor и true neighbour, но ещё не
  проверяет mapping выбранного non-exact candidate.
- Strict `trusted` дополнительно считает content hit только если mapping
  выбранного candidate тоже выше board median. Exact semantics не меняется.
- Оба trusted scope target-assisted и **недоступны на test inference**. Решение
  о deployability должно опираться на `all` и direction-stratified `all`.

Manifest protocol digest:
`2a9e3b74f7defa8c00846a05eb598fd263fd16c2787c70e77d3b7a4b585bfbf4`.
Shared subset namespace — `aiijc-puzzle-experiments-v1`, seed — `20260829`.

## Команды

Final training:

```bash
uv run python scripts/run_content_verifier.py \
  --train-limit 128 --eval-limit 24 --eval-offset 12 \
  --eval-split calibration --epochs 5 --rows-per-board 192 \
  --batch-size 64 \
  --output outputs/content-verifier/scale128-calibration24.json --run
```

Поскольку strict checkpoint contract был добавлен audit-ом после обучения,
исходный checkpoint связан с его immutable report отдельным воспроизводимым
шагом:

```bash
uv run python scripts/run_content_verifier.py \
  --checkpoint-in outputs/content-verifier/scale128-calibration24.pt \
  --bind-checkpoint-report outputs/content-verifier/scale128-calibration24.json \
  --output outputs/content-verifier/scale128-calibration24-bound.pt
```

Authoritative checkpoint-only calibration re-evaluation:

```bash
uv run python scripts/run_content_verifier.py \
  --train-limit 128 --eval-limit 24 --eval-offset 12 \
  --eval-split calibration --epochs 5 --rows-per-board 192 \
  --batch-size 64 \
  --checkpoint-in outputs/content-verifier/scale128-calibration24-bound.pt \
  --output outputs/content-verifier/scale128-calibration24-final.json --run
```

## Final calibration results

`Ensemble` — фактически strongest exact classical baseline. Для all-row
content strongest baseline — bilateral. Числа объединяют right/down.

| Scope | Pool oracle exact / <=20 | Strongest exact baseline | Verifier exact | Strongest <=20 baseline | Verifier <=20 |
|---|---:|---:|---:|---:|---:|
| all, 26,496 rows | .30450 / .61945 | ensemble .08405 | **.09484** | bilateral **.21675** | .16886 |
| trusted_query, 9,613 | .49381 / .52731 | ensemble .16686 | **.19931** | ensemble .17476 | **.20566** |
| strict trusted, 9,613 | .49381 / .50723 | ensemble .16686 | **.19931** | ensemble .17029 | **.20295** |

All-row exact вырос на **+1.079 п.п.**, 23/24 board wins, normal lower-95
`+0.820 п.п.`. Но all-row content RMSE<=20 упал на **-4.789 п.п.** против
bilateral, 1/24 wins и 23/24 losses.

Strict trusted exact вырос на `+3.246 п.п.`, а strict content на `+3.266 п.п.`
против strongest ensemble. Эти два прироста почти совпадают: у verifier
`.19931 exact` и лишь `.20295 content<=20`. Следовательно, trusted result не
подтверждает отдельный content-twin mechanism; почти весь выигрыш — exact
identity signal на target-assisted high-confidence labels.

### Direction split

| Direction | Exact: ensemble -> verifier | Delta | <=20: bilateral -> verifier | Delta |
|---|---:|---:|---:|---:|
| right, 13,248 rows | .08205 -> .09360 | **+1.155 п.п.** | .21052 -> .17323 | **-3.729 п.п.** |
| down, 13,248 rows | .08605 -> .09609 | **+1.004 п.п.** | .22298 -> .16448 | **-5.850 п.п.** |

Exact improvement переносится в обе стороны. Content degradation также
воспроизводится в обе стороны и особенно велик по вертикали.

## Frozen gate

| Check | Threshold | Delta | Result |
|---|---:|---:|---|
| all exact vs strongest | >= +0.5 п.п. | +1.079 | pass |
| all content<=20 vs strongest | >= 0 | -4.789 | **FAIL** |
| strict trusted exact vs strongest | >= +1.0 п.п. | +3.246 | pass |
| strict trusted content<=20 vs strongest | >= +1.0 п.п. | +3.266 | pass |
| right all exact | >= 0 | +1.155 | pass |
| right all content<=20 | >= 0 | -3.729 | **FAIL** |
| down all exact | >= 0 | +1.004 | pass |
| down all content<=20 | >= 0 | -5.850 | **FAIL** |

Итог gate: **FAIL**. Поэтому fresh holdout 12:36 не открывался. Старый pilot
holdout 0:12 не использовался для final selection.

## Runtime

- Apple MPS, PyTorch 2.13.0, 30,626 parameters.
- 26,812 eligible trusted training rows; 24,060 sampled rows/epoch.
- Loss: `2.08893 -> 1.88566 -> 1.82302 -> 1.79000 -> 1.76539`.
- Полный scale run: 284.38 s; train preparation 160.82 s; пять эпох 88.13 s;
  calibration preparation 29.95 s; inference 4.41 s.
- Final contracted eval-only run: 34.77 s.

## Checkpoint contract и provenance

Contract до чтения eval images валидирует architecture, exact ordered views,
candidate k, label policy/threshold, protocol digest, namespace/seed, stored
train filenames и recomputed selection digest. Негативный тест с изменённым
порядком views останавливается до подготовки данных.

Bound checkpoint SHA-256:
`e1c908b4ab84310054176fba2a79599ad7541c1a750bbce87238403a4e913a74`.

Current evaluator hashes внутри contract:

- `content_verifier.py`: `dc11011457bdf3cf45136d619e77df62d6cb87f3d6e05184eab6b9fa07523ee0`;
- `candidate_supply.py`: `2b0fe09bdcac736b34b2f260a0456f519e6dc6cdb4568e79e3a6b8679884dcfc`;
- `protocol.py`: `c97fcb5e6eb07abe6a91480c525f63b72a73bfb0aec7df4d51971e82b2f481be`;
- `run_content_verifier.py`: `4069d7cf71895a511a2dfffb003a8a22476586b2540457e821ef03a6413777f6`.

Legacy training report сохранил training-time hash только для
`content_verifier.py`:
`619ca6a265427c2fdce3beca694646f7554bdb1ecd615936860039bc16851e99`.
Для остальных трёх training-time hashes исходный report их не записывал;
contract честно маркирует их как `not-recorded-by-legacy-training-report`, не
выдавая bind-time hashes за training provenance.

## Решение

Content-multipositive formulation **закрыта как глобальный verifier**. Она
систематически выбирает exact identity лучше, но платит за это большим падением
inference-relevant all-row content recall, хотя pool содержит огромный content
headroom. Запуск decoder-а после такого gate не оправдан.

Checkpoint можно сохранять только как exact-edge research auxiliary для
будущего отдельного objective. Нельзя подменять им global classical scorer,
заявлять placement/SSIM gain или продолжать tuning на уже открытых panels.

## Проверки и artifacts

- `ruff format` и `ruff check` проходят.
- Unit tests проверяют candidate-list permutation equivariance, spatial grid,
  variable-length padding, finite gradients при no-positive rows,
  confidence-aware training positives и strict trusted content semantics.
- `outputs/content-verifier/scale128-calibration24.json` — исходный training
  report и timing;
- `outputs/content-verifier/scale128-calibration24-bound.pt` — authoritative
  contracted checkpoint;
- `outputs/content-verifier/scale128-calibration24-final.json` — authoritative
  audited calibration report с scopes, directions, paired deltas и gate.

Generated outputs игнорируются Git; все существенные digests и filenames
записаны внутри JSON/checkpoint contract.
