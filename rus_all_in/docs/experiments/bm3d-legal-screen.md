# Legal BM3D restoration screen

Статус: **decisive reject-as-tested**. Ни один BM3D candidate не выиграл ни
одной из 24 досок против frozen NLM h20. Confirmation `252:276`, holdout и
competition test не открывались; production не менялся.

## Зачем и что проверено

До запуска поиск по `docs/`, `outputs/`, `configs/`, `scripts/`, `src/`,
`tests/`, `pyproject.toml` и `uv.lock` не нашёл прежней реализации BM3D. Screen
проверяет новую legal full-canvas restoration после одного и того же strict
layout и harmonizer:

```text
corresponding dirty board
  -> 576 upright tiles
  -> no-atlas bilateral buddies96 strict permutation
  -> raw permutation audit
  -> frozen RGB seam offsets -> bounded luma
  -> one of seven frozen restoration arms
```

Все outputs используют только pixels соответствующей доски после сборки. Нет
reference lookup, generation, templates, cross-board pixels, tile substitution,
rotation, resize или warp.

## Pinned dependency и лицензия

Использован официальный [PyPI `bm3d==4.0.3`](https://pypi.org/project/bm3d/4.0.3/),
wheel `bm3d-4.0.3-py3-none-any.whl`, SHA-256
`fc4dfc0de0cd810fcb6ad198e1d0c6f99cf19d41f2ec69ff867674cfb9f2a775`.
PyPI классифицирует пакет как free for non-commercial use; embedded LICENSE
разрешает informational non-commercial scope при сохранении notices и ссылке на
авторов. Эксперимент использует пакет только для non-commercial research.
Submission может содержать лишь полученные PNG, а не BM3D/BM4D code, binary или
модифицированное ПО.

Ephemeral runtime полностью pin-ился командой:

```bash
uv run \
  --with bm3d==4.0.3 --with bm4d==4.2.5 \
  --with numpy==2.4.6 --with scipy==1.17.1 --with PyWavelets==1.9.0 \
  python scripts/run_bm3d_legal_screen.py ...
```

Commitment хранит distribution `RECORD`, `METADATA` и `LICENSE` hashes. Для
BM3D LICENSE SHA-256 равен
`a399290de3726a9a351621ee6a7d3d259c10ebe5e2ec24b4979d0719c50f6d16`,
RECORD —
`848a783923c037a6b609bb4f0c20a23589681262226845911497bb08a7e8d8f0`.
Авторы package: Ymir Mäkinen, Lucio Azzari, Alessandro Foi; algorithm references
в PyPI metadata — Mäkinen/Azzari/Foi (2020) и Dabov/Foi/Katkovnik/Egiazarian
(2007).

## Frozen arms

Один `uint8` harmonized canvas переводился в `float64 RGB [0,1]`.
`bm3d.bm3d_rgb` запускался с profile `np`, opponent colorspace `opp` и обоими
default stages. Результат clip-ился в `[0,1]`, умножался на 255, округлялся
`numpy.rint` и возвращался в `uint8`.

| ID | Arm | Role |
|---|---|---|
| A | colored NLM `h20` | baseline |
| B | colored NLM `h28` | diagnostic safe reference |
| C | RGB BM3D `sigma=0.12` | candidate |
| D | RGB BM3D `sigma=0.16` | candidate |
| E | RGB BM3D `sigma=0.20` | candidate |
| F | RGB BM3D `sigma=0.16`, затем ровно один NLM `h10` | candidate |
| G | half-up `uint8` 50/50 blend BM3D `0.16` и NLM `h20` | candidate |

Config `configs/bm3d_legal_screen_preregistered_v1.json`, SHA-256
`934506c22420aba4aabe7d3c0ba786482a7c3d5ca700fa3866c6a55c44ea15d4`,
был зафиксирован до target decode.

## Данные и commitment

Primary: reused calibration ranked `276:300`, count 24, selection digest
`eda60f632677a6da07d263ef559ad4d5a99e38fbef072f45fbca50f842b49708`.
Planned confirmation: `252:276`, count 24, digest
`1bd59c6db73fa4af59e4304949c87512a3a4bb700845ad2bbe0ef677d48a27f1`.
Они взаимно непересекаются, но не untouched: historical calibration700 уже
скорил все calibration records; три confirmation names также встречались в
`novel-analog-layout/calibration24`. Это reused-calibration evidence, не
holdout/generalization claim.

Все 168 primary predictions и diagnostics были сохранены до первого target
decode. Commitment:

- SHA-256
  `f9bc9c41b2a97b4297fd94c5b6605b3f7669ee9d57b086755122700685fa8267`;
- frozen pixel-roster digest
  `4b6312694049de9e44303e931052c1b039796c56d5c8ddbef6cb37550156acda`;
- raw permutation audits 24/24 PASS;
- семь predictions distinct на каждой из 24 досок;
- target-blind inference runtime `262.96 s` на текущем Apple runtime.

## Primary result

Все сравнения paired против A, на тех же 24 layouts/targets.

| Arm | Mean SSIM | Delta vs A | Paired 95% CI | W/T/L |
|---|---:|---:|---:|---:|
| A NLM h20 | **0.253976** | — | — | — |
| B NLM h28 reference | **0.265573** | **+0.011597** | `[+0.010508,+0.012618]` | 24/0/0 |
| C BM3D .12 | 0.197066 | −0.056910 | `[−0.063568,−0.050213]` | 0/0/24 |
| D BM3D .16 | 0.212816 | −0.041160 | `[−0.045685,−0.036517]` | 0/0/24 |
| E BM3D .20 | 0.221649 | −0.032327 | `[−0.036132,−0.028464]` | 0/0/24 |
| F BM3D .16 -> NLM h10 | **0.243646** | **−0.010330** | `[−0.011821,−0.008800]` | **0/0/24** |
| G 50/50 BM3D .16 + NLM h20 | 0.235226 | −0.018750 | `[−0.020783,−0.016673]` | 0/0/24 |

Все candidates прошли target-free minimum detail/clipping bounds: BM3D не
схлопнул gradient или Laplacian. Напротив, pure BM3D сохраняет заметно больше
high-frequency energy, чем NLM h20. Но все C–G одновременно провалили absolute
`>=0.27`, positive-CI и 18/24 wins; каждый проиграл A на 24/24. Поэтому нет
passing winner и manual gate не может авторизовать confirmation.

Diagnostic review сравнил A с лучшим BM3D candidate F на всех 24 досках. Новых
severe hallucinations или geometry warp не видно; F заметно резче и более
блочный. Обе стороны остаются мозаичными из-за общего layout. Этот просмотр —
описание failure mode, не manual PASS решения.

## Решение

Не повторять RGB BM3D `sigma=.12/.16/.20`, tested `.16→NLM10` cascade или
50/50 blend с NLM20 на большем split. На этой постановке BM3D сохраняет noise и
block detail, которые SSIM penalizes относительно более сильного NLM. Даже safe
NLM h28 reference достиг только `0.265573 < 0.27` на этой панели.

Confirmation runner был вызван как fail-closed audit и отказал до создания
директории: `no primary candidate passed every quantitative gate`.

Artifacts:

- `outputs/bm3d-legal-screen/primary-calibration-offset276-count24/prediction-commitment.json`;
- `outputs/bm3d-legal-screen/primary-calibration-offset276-count24/report.json`,
  SHA-256
  `17739e5f2a872529cc5ade0e2420506e5f6450c904d9aa5c1526adb3e6b38313`;
- `outputs/bm3d-legal-screen/primary-calibration-offset276-count24/manual-review.json`,
  SHA-256
  `285cdae52c00c18501ace84a110e641c33b813ae8a0b4148afba052ff81a10ea`;
- `src/aiijc_puzzle/bm3d_screen.py`;
- `scripts/run_bm3d_legal_screen.py`;
- `tests/test_bm3d_screen.py`.
