# TASKA unique-fullres accepted-edge calibrator

Дата фиксации: 2026-08-31.

## Вердикт

Один заранее зафиксированный `StandardScaler -> LogisticRegression(C=1)`
успешно повысил precision уже принятых unique-fullres edges, но **не перенёс
pair-выигрыш с held32 на fresh32**. На held candidate дал `+1.84375`
пары/board, после чего неизменный preregistered gate разрешил fresh. На fresh
знак сменился на `-0.6875`; формальная confirmation-панель не открывалась.

| Panel | Confirmed fusion pairs / exact | Calibrated pairs / exact | Pair delta, source CI95 | Exact delta, source CI95 |
|---|---:|---:|---:|---:|
| held32 | `345.3125 / 1.90625` | `347.15625 / 1.625` | `+1.84375 [-2.0,6.125]` | `-0.28125 [-0.75,0.09375]` |
| fresh32 | `355.625 / 0.9375` | `354.9375 / 1.0` | `-0.6875 [-2.0,0.46875]` | `+0.0625 [0.0,0.1875]` |

Fresh pair W/T/L равен `2/26/4`. Направление закрыто без feature, `C`,
class-weight или threshold sweep. Подтверждённый unfiltered
selective-plus-unique-fullres fusion остаётся pair leader.

## Почему это не повтор прошлых экспериментов

Перед запуском были проверены fullres union voter, confirmed selective-fullres
fusion, focal-feature stacker и fullres relation fusion. Focal-feature stacker
обучал score для **всего current harvest** и менял его ordering. Relation
fusion работал с component-direction queries и другим decoder bridge. Этот
эксперимент единственный раз обучает бинарный фильтр только на уже принятых,
отсечённых от selective overlap **unique-fullres** edges, а затем оставляет
подтверждённый global consumer неизменным.

## Fixed contract

Fit использовал только уже открытую local32: 375 unique-fullres edges из 31
source, 169 positive (`45.07%`). До held scoring были зафиксированы:

- признаки в точном порядке: recovered focal logit, restored support `3/4`
  или `4/4`, исходная directional raw seam cost, outgoing/incoming raw rank,
  outgoing/incoming margin from best и axis flag;
- rank как доля строго меньших costs среди 575 non-self значений;
- `StandardScaler` и unweighted `LogisticRegression(C=1, max_iter=1000,
  random_state=0, solver=lbfgs)`;
- natural decision threshold `0.5`;
- exact confirmed unfiltered fusion final layout как control;
- held pair gate `delta >= 0`, затем ровно один unchanged fresh replay.

Config SHA-256:
`079238829e55f3719592d6c061b040aedf72dc4d1c68f9454681195546532e26`.
Matcher и denoiser не запускались повторно.

## Edge-level результат

Классификатор действительно переносит edge-level precision, но выбрасывает
примерно 41% правильного unique supply:

| Panel | Unfiltered edges / precision | Kept edges / precision | True-edge retention |
|---|---:|---:|---:|
| local fit | `375 / 45.07%` | `147 / 68.03%` | `59.17%` |
| held32 | `474 / 52.53%` | `225 / 64.44%` | `58.23%` |
| fresh32 | `406 / 47.54%` | `164 / 69.51%` | `59.07%` |

Это важный отрицательный результат: higher edge precision сама по себе не
гарантирует лучший rigid global layout. Подтверждённому solver полезен более
широкий complementary supply, а ошибка consumer/selector доминирует над
локальной чистотой фильтра. Не повторять nearby probability thresholds,
class weights или subsets этих восьми признаков на открытых panels.

## Freeze и legality

Held и fresh candidate layouts, probabilities, masks и provenance были
записаны target-free и SHA-frozen до восстановления references. Все control и
candidate layouts — строгие перестановки `0..575` исходных upright fragments.
Restored pixels использовались только историческим matcher-родителем; rotations,
warps, replacement tiles, postprocess и competition test отсутствуют.

Artifacts:

- report: `outputs/taska-fullres-unique-edge-calibrator/fixed-v1/report.json`,
  SHA-256 `91bfad9817ab9fe82337a6f76201f1544c8209eff1b56dd4a5a9d7bdd7c6cb00`;
- calibrator: `outputs/taska-fullres-unique-edge-calibrator/fixed-v1/calibrator.npz`,
  SHA-256 `dc44859b5ef43324a88d214a74c6681983d4236ffaccde119516991a0ec6149e`;
- held/fresh target-free archives: SHA-256
  `68a236456ab6584e281b265602ca7ac3120d9bf5c9e2038bb1ab08afc6f4e6b4` /
  `b930f4a2bcc7a71137e15b5ba9b9590c1b10d604f39bdf316a64b35a3132fa68`;
- module: `src/aiijc_puzzle/taska_fullres_unique_edge_calibrator.py`;
- runner: `scripts/run_taska_fullres_unique_edge_calibrator.py`.

Weco Observe pair+exact: held step `104` branched from confirmed fusion step
`102`; unchanged fresh step `105` branched from `104`.
