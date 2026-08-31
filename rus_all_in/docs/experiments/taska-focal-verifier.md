# TASKA focal seam verifier: artifact audit and frozen replay

Дата фиксации: 2026-08-31.

## Итог

В забытом локальном Kaggle-audit cache найден реально доступный historical
focal verifier. Он легален как target-free inference primitive: получает только
сырые dirty RGB-полосы двух фрагментов и шесть признаков из текущей строки
matcher cost matrix, не меняет candidate membership и выдаёт только приоритет
ребра. На двух уже открытых панелях знак pair delta переносится. Более точный
training contract `top5` лучше repository-tip `top8`, но confidence intervals
пересекают ноль, а held300 был model-selection-exposed исторически. Поэтому
результат — `promising transferred diagnostic`, не fresh promotion и не новый
submission default.

## Найденный checkpoint и provenance

- Постоянная копия:
  `artifacts/prior-taska/ckpt/verify_pair_best.pt`.
- Исходная recovered копия:
  `/private/tmp/aiijc-kaggle-audit.33E0I4/seam-verifier/verify_pair_best.pt`.
- Размер: `817615` bytes; постоянная копия read-only (`0444`).
- SHA-256:
  `3bcc89a12e7b539304484b441688b4b9fb1c3711e918befed9cdef7c17f776e7`.
- Источник binary: cached output Kaggle kernel
  `pasha883/vsos-pazzle-seam-verifier`. Binary не был tracked git blob или LFS
  object; unreachable git objects с ним также не найдены.
- Exact source provenance: repository
  `/Users/rusyalain/Documents/GitHub/pazzle_will_be_killed`, commit
  `ae9d231ad450cf4e66685498adfbf918003e3239`, файлы
  `src/verify_pair.py`, `src/train_verify.py`, `src/verify_edges.py`,
  `src/choose5.py`, `src/build_verify_topk_cache.py`.

Safe loader сначала проверяет size и SHA, затем использует только
`torch.load(..., weights_only=True, map_location="cpu")`, требует ровно ключи
`model,args`, exact args `ch=64, blocks=4, strip=4` и strict state-dict.
Архитектура содержит `200322` trainable parameters и `200838` state elements.

## Exact inference contract

`SeamVerifier` — joint CNN над одним швом:

- horizontal patch: последние четыре колонки source + первые четыре target;
- vertical patch: последние четыре строки source + первые четыре target, затем
  транспонирование, чтобы тот же encoder видел вертикальный шов как горизонтальный;
- patch shape: `3×20×8`, raw uint8 values переводятся в float без деления на 255;
- trunk: `Conv3×3→GELU`, четыре `Conv3×3→BN→GELU` blocks и дополнительный
  `Conv3×1` после каждого нечётного block;
- отдельно pool-ится средняя пара колонок шва и whole patch;
- MLP: `(2×64+6)→128→64→1`;
- final score: learned delta + `prior * features[:,0]`; больше — лучше.

Compatibility задаётся как `M=-cost`, diagonal принудительно равен `-1e9`.
Для edge `(source,target)` используются:

1. `M[source,target]/10`;
2. score minus row best;
3. число элементов строки строго выше score;
4. индикатор score == row best;
5. среднее top-K строки `/10`;
6. top1-minus-topK spread.

Есть два несовместимых исторических режима:

- `train_exact_top5`: K=5, буквально как в `train_verify.py` и training dump;
- `historical_tip_top8`: K=8, как в repository-tip `verify_edges.py`.

Inference не принимает target, exact permutation, `restore_labels`, filename,
canonical tile id или source-grid coordinate. `restore_labels.inv` использовался
только offline для supervised training labels. Candidate roster не расширяется
и не фильтруется. Logits меняют только порядок добавления edges в rigid component
builder; original cost matrices остаются неизменными в placement и Hungarian
fill. Все outputs — строгие перестановки 576 исходных upright tiles.

## Frozen replay

Все candidate logits/features/layouts были записаны и hash-rostered до создания
exact references. Канонические v2 runs используют постоянную workspace-копию
checkpoint. Первые v1 runs дали бит-в-бит те же candidate archives и метрики,
но ссылались на временный absolute checkpoint path; v2 заменяет их только для
долговечного provenance.

| Panel / mode | Raw pairs | Focal pairs | Pair delta, clustered 95% CI | Raw exact | Focal exact | Exact delta, CI |
|---|---:|---:|---:|---:|---:|---:|
| opened32 / train-exact top5 | 334.71875 | 335.50000 | +0.78125 `[-1.15625, 2.65625]` | 4.46875 | 4.34375 | -0.125 `[-1.78125, 1.15625]` |
| held300 / train-exact top5 | 329.62500 | **332.53125** | **+2.90625** `[-3.34375, 11.1875]` | 2.90625 | **4.00000** | **+1.09375** `[-0.625, 3.53125]` |
| held300 / historical-tip top8 | 329.62500 | 331.81250 | +2.18750 `[-4.28125, 11.1875]` | 2.90625 | 3.96875 | +1.06250 `[-0.71875, 3.375]` |

Root independently measured historical top8 on opened32 before this port:
337.03125 pairs versus 334.71875 (`+2.3125`, CI `[-0.907, 6.0]`) and
3.75 exact versus 4.46875 (`-0.71875`). That result is stored in
`outputs/weco-observe/solver-step-25-historical-focal-verifier-opened-pair-signal.md`.

Canonical artifacts:

- `outputs/taska-focal-verifier/opened32-train-exact-top5-cpu-v2/`
  - frozen NPZ `60243ab924da96d8bb49b072458c4710c65b8195b8d2c31eff1132b59ee56fd2`
  - metadata `8e6be1d0f4b2652b784141d7c53d7fb63394e8bda6af3b076a9fd5721f07c9d5`
  - pre-score freeze `718e274fcf9c19715a60b722be69c6d42b2e65855d19af545d700ab2e8d5a6c1`
  - report `4ac3ee90c2e03f44051435a2eee9d6a7baec771ab8c5b7d7f7478e7dfb54880a`
- `outputs/taska-focal-verifier/held300-train-exact-top5-cpu-v2/`
  - frozen NPZ `7d4ad494ab572d1ac3c94ab73a49b54e80b26baba489dfbd56f732a5c43394c5`
  - metadata `301ba535f04b63ff8da48a0a83b5f207521d4b57f1bdbb61ceb58dbee57daff2`
  - pre-score freeze `5faf81531665d7148cb797f904de28702263547f66bf2a092a58f1936aed5522`
  - report `b0700098b6a7e232a56cc268c3f2bc14ab9d1af3e5bd84304d5674b40a72a103`
- `outputs/taska-focal-verifier/held300-historical-tip-top8-cpu-v2/`
  - frozen NPZ `4d347f9dffed6b767a4aece09752649235d4c438968cb29293663abc1bdb095a`
  - metadata `6d0091230ea10edb77c7c2da31a7f7ce604006616997a5ad02b7201d8432a840`
  - pre-score freeze `12f6777e563ca4215e681ed80772a1434e012fba559b855e22d2ca5266541029`
  - report `abb2f16b150457c9bd23e9439f49a7023f207adb06acf3b81e8931ea9788bd34`

## Остальные найденные или закрытые artifacts

| Candidate | SHA-256 / architecture | Historical evidence | Verdict / legality |
|---|---|---|---|
| `verify_pair.pt` | `24f51201a6b49fa5034056547116ce7b45e48395b9c14fac9803de8f58c14d74`; tensors exactly equal `verify_pair_best.pt` | duplicate final epoch payload | Не хранить как отдельный arm; best-copy выше достаточна. |
| `verify_hinge.pt` | отсутствует; Kaggle kernel `pasha883/vsos-pazzle-verify-hinge` versions 1–3 had empty output / `CANCEL_ACKNOWLEDGED` | matched local M467 precision collapsed `0.6420→0.4417`, epoch4 `0.5817` | Closed: нет auditable checkpoint и historical metric отрицательная. |
| `choose5_big.pt` | `d7f95a8a5dd1b884c995082e66e5c8d5e2c8b8cf9c2980caa3a3be13369d1454`; 1,214,787 params | M438 init 333.7 correct bonds; trained epochs ~305–311 | Rejected. Historical wrapper также исключал rows по `tile_id%24` / `tile_id//24`, что не permutation-equivariant на raw bag. |
| `seam_big_big.pt` | `81f177aea5f6a4f04d3bf2cc1d9d3ae925e756c4640764df1c85f475e468ee9e`; ch192/blocks8/dim384/strip5 | R@1 `.28370`, R@20 `.62745`; matched solo/trio хуже v3/trio | Closed negative. |
| strip3/5/8/12/20 sweep | `3fd959…`, `1afac2…`, `e5903f…`, `823d84…`, `ae65ce…`; ch128/blocks6/dim256 | R@1 `.2425–.2663`, все ниже v3/local при equal budget | Closed negative; не повторять strip-width sweep. |
| `seam_embed_v3.pt` | `6f0917…`; global ch96/blocks6/dim192/strip3 | R@1 `.33650`, R@20 `.66636` | Текущий основной TASKA matcher. |
| `seam_embed_local.pt` | `593285…`; local ch96/blocks6/dim192/strip3 | R@1 `.33303`, R@20 `.66606` | Полезен в v3+local fusion. |
| `seam_embed_wide.pt` | `9faf63…`; local strip7 | R@1 `.33424`; v3+local precision `.496`, all-three `.491–.494` | Excluded из fusion. |

Ни tracked checkpoint blobs, ни LFS pointers, ни unreachable git objects, ни
другие verifier/chooser/seam binaries в Documents, Downloads, Desktop, common
caches, Trash и `/private/tmp` не найдены. В частности, `verify_hinge.pt`
закрыт как missing, а не как скрытый доступный artifact.

## Код и verification

- `src/aiijc_puzzle/taska_focal_verifier.py` SHA
  `daa4c41e19db51264d284dbfcc7baec7ba3d5d66e22d34aadffccb3849e030cd`.
- `scripts/run_taska_focal_verifier_replay.py` SHA
  `2835d6bd39a4358561b3c0cb1750e2c4664b91818ac267b52ca4028ae326e803`.
- `tests/test_taska_focal_verifier.py` SHA
  `d664a82da0b263bf94a3326cd3290dd6c74dfd1107ee18f56ae1fd24cb528e3e`.
- Frozen raw solver не изменён: SHA
  `97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486`.
- `24` relevant pytest tests passed: focal module, external-priority wrapper,
  raw-tail solver.
- Ruff passed для module, runner и focal tests.

Проверки покрывают SHA-before-deserialization, weights-only strict load,
architecture/state counts, golden checkpoint inference, literal horizontal и
vertical patch construction, literal top5/top8 feature formula,
bag-relabeling equivariance, immutable aligned logits/features и strict input
contracts. Каждый canonical v2 pre-score freeze также повторно валидирован
после run.

## Решение

Если focal verifier развивать дальше, фиксировать `train_exact_top5` как
предпочтительный контракт: он соответствует training distribution и дал более
сильный held pair/exact result, чем top8. Не подбирать K, threshold или blend на
уже открытых panels. Следующий допустимый сильный шаг — один заранее
зафиксированный replay top5 на действительно source/model-selection-fresh
panel или использование logits внутри materially different global consumer.
До этого raw remains production default, focal — promising optional ordering
arm без submission promotion.
