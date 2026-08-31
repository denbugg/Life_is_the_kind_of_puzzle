# Official DRUNet40 только как matcher-view

## Решение

**Train-only reject; calibration запрещена и не открывалась.** Независимый
tile-wise official DRUNet `sigma=40` дал локальный edge signal, но не дал
устойчивого end-to-end layout signal. Frozen selection на первых 8 train boards
выбрала fusion weight `.50`. На последних 8 она улучшила exact adjacency на
`+0.004642`, paired 95% CI `[+0.000709,+0.008576]`, но ухудшила
translation-aligned placement на `-0.002170`, CI
`[-0.004330,-0.000010]`. Final F SSIM gain `+0.006846` имел CI
`[-0.005396,+0.019088]` и только `4/8` wins. Frozen verification gate провален,
поэтому calibration, holdout, test и production не затронуты.

Это не противоречит прежнему положительному результату DRUNet как pixel tail:
здесь его pixels никогда не рендерят output. Проверялась другая гипотеза — может
ли denoised копия каждого dirty tile дать bilateral matcher более устойчивые
границы, после чего layout собирается только из исходных фрагментов.

## Legal inference boundary

Для каждого board использованы два target-free score views:

1. baseline bilateral directional scores на original dirty `20x20` tiles;
2. те же bilateral directional scores после official DRUNet40, применённого
   независимо к каждому dirty tile только для matcher view.

DRUNet получает reflect-padded `24x24` tile, результат сразу crop-ится обратно к
`20x20`. Right/down score matrices уже являются row-normalized E14 log
probabilities, поэтому frozen fusion вычисляется отдельно по направлениям как
`(1-w)*dirty + w*drunet` для `w=.25,.50,.75`. Pure DRUNet `w=1` оставлен только
как diagnostic. Каждый arm декодируется неизменённым
`solve_buddies(max_edges=96)`.

Каждая сборка содержит строгую биекцию всех 576 **original upright dirty
tiles**. DRUNet pixels не входят в raw, harmonized, h28, F или какой-либо
submission output. После layout применены exact frozen RGB offsets, bounded
luminance gains и два заранее фиксированных legal tails:

- independent single-pass colored NLM h28;
- F: exact h20-derived t40 mask, independent h28 у protected pixels и
  independent h40 в flat regions.

Targets и target-assisted recovered mapping использованы только после freeze для
метрик exact adjacency, translation/direct placement и RGB SSIM; в matcher или
decoder они не входят.

## Immutable train-only protocol

Config:
`configs/drunet_matcher_train_diagnostic_preregistered_v1.json`, SHA-256
`781c177c391b30e0a261c12ffd5723928d6f031eb468815279c4df4392cef62e`.
Файл заморожен read-only до target decode.

Общий selector `aiijc-puzzle-experiments-v1`, seed `20260829`, ranked train
`512:528`, count 16:

- full filename digest
  `d81693d8a929d9ac3a107a8ce53e186031096cbe02e33c4d5f53b496225c0e67`;
- filename+input roster digest
  `1ec2cfca38293d0997c27c597dfa8f7a016a5bd03734dab1ad9ea9db2e0b5802`;
- selection first8 digest
  `7329e8e7e46ac891013e3f53a38bfdc24d9017a8caa728b60e914f50611bfc67`;
- verification last8 digest
  `94cc609ca438289118315c5bd6cb59e4c401833ef30de5ff92a31a05a86316d3`.

Все 16 train targets были исторически открыты прежним DRUNet-tail experiment
(`outputs/pretrained-tile-denoiser/train-development-offset512-count16/report.json`,
SHA `7c64c966…`). Поэтому это reused development, не fresh panel и не unbiased
generalization estimate. Но текущий experiment всё равно заморозил все layouts и
predictions до своего target decode.

Official KAIR checkpoint
`artifacts/pretrained-denoisers/kair-fc1732f/drunet_color.pth`, sigma 40,
SHA-256
`479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4`.
Freeze commitment:

- file SHA
  `d606b4d852a7bda611092d3f8c3108fd5eef2c400db95310dcd82cdb841ee947`;
- payload SHA
  `44a2c8005d59143e3ccd82b9bb6ee0f504fd841bd5fba42a6800de55d7fde228`;
- source SHAs: module `f0ce1ebd…`, runner `26660814…`.

Сначала был открыт только first8 selection roster. Read-only decision выбрал
`.50` по maximum mean F среди `.25/.50/.75`, с frozen tie-break h28,
adjacency, translation и lower weight. Pure DRUNet не мог быть выбран. Лишь
после записи decision SHA
`443aedcc21af416b373b2d09e230357ac8776acb10f8e81d912e952c988f0136`
был открыт disjoint last8 verification roster. Selection target receipt SHA
`618a1cde…`, verification receipt SHA `565a7b31…`.

## Selection first8

Selection rule обязана выбрать лучший **только среди fusion arms**, даже если
он проигрывает baseline. Именно это произошло:

| Arm | Mean adjacency | Translation | h28 SSIM | F SSIM | F gain vs baseline |
|---|---:|---:|---:|---:|---:|
| dirty bilateral | 0.033741 | 0.009983 | 0.285199 | **0.287305** | — |
| fusion .25 | 0.038157 | 0.009766 | 0.281572 | 0.283724 | -0.003581 |
| **fusion .50, selected** | 0.038157 | **0.010200** | **0.283084** | **0.285183** | **-0.002122** |
| fusion .75 | **0.038610** | 0.009766 | 0.282175 | 0.284226 | -0.003080 |
| pure DRUNet diagnostic | 0.037817 | 0.009766 | 0.276878 | 0.278882 | -0.008423 |

Уже здесь ни один candidate не улучшил final F или h28 endpoint. Локальная
adjacency росла, но этот signal не преобразовывался в full-board score.

## Disjoint verification last8

Frozen selected `.50`:

| Metric | Dirty baseline | Fusion .50 | Gain | Paired 95% CI | W/T/L |
|---|---:|---:|---:|---:|---:|
| exact adjacency | 0.038383 | **0.043025** | **+0.004642** | `[+0.000709,+0.008576]` | 6/1/1 |
| right adjacency | 0.038043 | **0.043705** | **+0.005661** | `[+0.001827,+0.009496]` | 7/0/1 |
| down adjacency | 0.038723 | 0.042346 | +0.003623 | `[-0.003388,+0.010634]` | 5/1/2 |
| translation placement | **0.010851** | 0.008681 | **-0.002170** | `[-0.004330,-0.000010]` | 1/2/5 |
| direct placement | 0.001085 | 0.001736 | +0.000651 | `[-0.000429,+0.001731]` | 4/3/1 |
| h28 final SSIM | 0.251822 | 0.258662 | +0.006841 | `[-0.005452,+0.019133]` | 4/0/4 |
| F final SSIM | 0.254238 | 0.261084 | +0.006846 | `[-0.005396,+0.019088]` | 4/0/4 |

Gate требовал positive F CI lower, минимум 6/8 F wins, минимум 6/8 h28 wins,
positive adjacency и translation gains, nonnegative direct gain и permutation
integrity. Провалены четыре проверки: F CI, F wins, h28 wins и translation gain.
Permutation audit, adjacency, mean h28/F и direct conditions прошли.

Другие last8 arms не дают скрытого robust candidate:

- fusion `.25`: F `+0.005779`, только 3/8 wins, translation `-0.000868`;
- fusion `.75`: F `+0.003534`, 6/8 wins, translation `-0.001302`;
- pure DRUNet: F `+0.010516`, 6/8 wins, CI lower `-0.007335`, translation
  `-0.000868`.

То есть все DRUNet-containing views ухудшили translation mean. Post-hoc смена
weight или promotion pure diagnostic запрещены и всё равно не исправляют эту
проблему.

## Строгий вывод и правило «не повторять»

DRUNet40 matcher view действительно меняет edge ranking и может увеличить
локальную exact adjacency. Но выигрыш ориентирован главным образом на right
edges, не устойчив по down direction и не улучшает global placement. Final SSIM
различается по boards в обе стороны. Поэтому этот exact sigma40 bilateral fusion
route закрыт как `reject-as-tested`; локальную adjacency нельзя выдавать за
восстановление layout.

Не масштабировать `.25/.50/.75`, pure DRUNet view или новый calibration panel на
основании этих 16 boards. Повтор имеет смысл только при материально новом
mechanism, который заранее демонстрирует robust translation/full-layout gain на
train-only verification, а не только edge retrieval. Ни один calibration target
для этого направления не декодирован.

## Artifacts и QA

- source: `src/aiijc_puzzle/drunet_matcher.py`;
- runner: `scripts/run_drunet_matcher_train_diagnostic.py`;
- tests: `tests/test_drunet_matcher.py`;
- output: `outputs/drunet-matcher/train-offset512-count16/`;
- authoritative report SHA-256:
  `d899d4dfad32a6f1b4384aeb6951c5ba7e00184242fc44743957da5d5cd3e1e6`.

```bash
uv run python scripts/run_drunet_matcher_train_diagnostic.py --phase prepare
uv run python scripts/run_drunet_matcher_train_diagnostic.py --phase score
uv run ruff check src/aiijc_puzzle/drunet_matcher.py \
  scripts/run_drunet_matcher_train_diagnostic.py tests/test_drunet_matcher.py
uv run pytest tests/test_drunet_matcher.py
```

Prepare/score single-use и fail closed. Config, commitment, receipts, decision и
report read-only. Frozen edge-v1/v2 code/artifacts и production ZIP не менялись.
