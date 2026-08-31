# TASKA focal verifier: fixed current-harvest fine-tune

Дата фиксации: 2026-08-31.

## Итог

Один заранее зафиксированный fine-tune recovered focal verifier на текущем
распределении harvested edges корректно обучился, но не прошёл отдельный local
gate. На 32 disjoint organizer-train sources он потерял `0.53125` satisfied
pairs и `0.0625` exact tiles относительно recovered top5 checkpoint. Поэтому
unchanged held32 по preregistration не запускался, arm закрыт и не является
кандидатом на promotion или submission.

Это полезный отрицательный результат: уменьшение supervised ranking loss само
по себе не гарантирует улучшение жадного rigid-component consumer. Повторять
тот же two-epoch all-pairs ranking fine-tune на этом roster не нужно.

## Заранее зафиксированный протокол

- Config: `configs/taska_focal_current_finetune_v1.json`, SHA-256
  `94dce4a73410f9d40cf52b136af27116ddabf79dd5948c68307839e1e7bc6a23`.
- Источник roster: неизменный train256 roster, SHA-256
  `e940944865f0a4f93e6f6a9782c33c2da1566ffb8ef1253e88bec369d30c630c`.
- Train: первые 96 sources, draw0, order digest
  `7b3ff55d8e73097fccfe2aeae45528c13734c7793a9fd0f8ee1dfdf4893cd7fe`.
- Local gate: следующие 32 sources, draw0, order digest
  `f516f12e8943580ab62e17cd6d4064dc519aa20df6485bf5bca34030beaa2bc3`.
- Train и gate не пересекаются между собой, а также с opened32, held300 и
  protected-tail fresh32 rosters.
- Все 128 cases используют только organizer-train и тот же synthetic seed
  `1267233517`; competition test не открывался.
- Candidate membership строилась текущим frozen TASKA v3+local matcher по
  raw/median/bilateral views, `quad_weight=0`, target-derived boundary masks нет.
- Init: audited `verify_pair_best.pt`, SHA-256
  `3bcc89a12e7b539304484b441688b4b9fb1c3711e918befed9cdef7c17f776e7`.
- Feature contract: exact historical training mode `train_exact_top5`.
- Loss: средний `softplus(-(positive_logit-negative_logit))` по всем true/false
  harvested-edge парам внутри каждого board.
- Единственные training settings: AdamW, lr `3e-5`, weight decay `0.01`, ровно
  2 epochs, board batch size 1, clip norm 1, seed `2026083103`.
- Learned raw-score `prior` был заморожен точно на
  `1.1238815784454346`; residual CNN/MLP обучались.
- Checkpoint выбирался только после финальной эпохи: epoch sweep и best-epoch
  selection отсутствуют.
- Gate comparator — recovered top5 на тех же 32 cases. Условие запуска held:
  `mean(finetuned_pairs - recovered_pairs) >= 0`.

## Обучение

Текущий matcher выдал `36022` harvested edges на 96 train boards, из них
`24581` истинных (`68.24%`). Boardwise ranking loss снизился:

| Epoch | Mean all-pairs logistic ranking loss |
|---:|---:|
| 1 | 0.2377800 |
| 2 | 0.2075207 |

То есть оптимизация не была no-op: модель лучше разделила train labels по
своему loss, а отрицательный gate отражает именно несовпадение surrogate с
конечным solver consumer / переносом, а не отсутствие обучения.

## Local gate

Все три arm сохранили frozen candidate membership, original TASKA costs для
placement/Hungarian fill и выдали строгую перестановку всех 576 upright tiles.
Target-free layouts/logits были записаны до построения exact references.

| Arm | Pairs / board | Adjacency recall | Exact tiles / board |
|---|---:|---:|---:|
| Raw TASKA | **310.09375** | **0.2808820** | 1.3750 |
| Recovered focal top5 | 308.71875 | 0.2796365 | **2.0625** |
| Fine-tuned focal | 308.18750 | 0.2791553 | 2.0000 |

Fine-tuned versus recovered:

- pair delta: `-0.53125`;
- exact delta: `-0.0625`;
- source W/T/L по pairs: `14 / 1 / 17`.

Fine-tuned versus raw pair delta: `-1.90625`. Gate не пройден; held32 имеет
статус `skipped_by_preregistered_nonnegative_local_gate`.

Raw сильнее focal на этом конкретном local panel по pairs, но это не отменяет
ранее перенесённый pair/exact сигнал recovered top5 на других panels: здесь
решался только вопрос, улучшает ли фиксированный fine-tune recovered arm.

## Artifacts и проверка

- Supervised train archive:
  `outputs/taska-focal-current-finetune/v1/training-harvest.npz`, SHA-256
  `5ee7b100eb213076fc1acbcace1c6d22e17bea99b88266c5c255cd94c85a17a1`.
- Fine-tuned checkpoint:
  `outputs/taska-focal-current-finetune/v1/taska-focal-current-finetuned.pt`,
  SHA-256
  `0ad2b0154b068d27bab4e3d8b8fdba4ad962291d53e1665676dad8833d091f6f`.
- Frozen local-gate predictions:
  `local-gate-target-free.npz`, SHA-256
  `723bae86920410ef64a15607037afeb4020766058967f25065aa0d0810c8351e`;
  metadata SHA-256
  `30d6dd6ebe0e4d492bf43fce8436494e93e165066cf08b27b74ce4470aaadd8e`.
- Report: `outputs/taska-focal-current-finetune/v1/report.json`, SHA-256
  `cd451fe06ee63a93775fee1595265556d74a95c4d2885c152555404917c82770`.
- Module SHA-256:
  `a8ed67f3a45cf3091ab56d6a7a4e0c53ffdb4a22b6eefcc75839e4c813ec7a74`.
- Runner SHA-256:
  `cf614ad0bf50c6aa5c63798ee40c4ae475c1123501be0c14404c26f6f959d9e0`.
- Tests SHA-256:
  `caab758c42630638281d5966fde06076e2999e29a7d5af8473a1a89184df7420`.
- 27 relevant tests passed; Ruff passed.
- Frozen raw solver остался byte-for-byte прежним, SHA-256
  `97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.

Первый gate attempt остановился до записи gate artifacts из-за различия
строковых представлений Torch devices `mps` и `mps:0`. 96-board train и
checkpoint уже были полностью завершены, поэтому они не пересчитывались:
device validation нормализована, затем runner продолжил с gate через явно
проверяемый `--resume-after-training`. Это не изменило model, roster, loss или
evaluation arm.

## Решение

Arm закрыт. Не запускать этот checkpoint на held/test и не продолжать теми же
hyperparameters. Если fine-tuning вернуться позже, нужен materially different
consumer-aligned objective, который учитывает конфликтующие rigid-component
решения, а не только попарный порядок true и false harvested edges.
