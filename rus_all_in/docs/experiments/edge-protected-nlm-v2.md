# h28-safe / h40-flat edge-protected NLM v2

## Решение

**Stable legal tail improvement, but reject for frozen promotion gate; production
не менять.** Единственный candidate `F_h28safe_flat_h40_t40` прошёл numeric и
root manual gates на reused-calibration primary `60:120`: `0.280087`, gain к
global h28 `+0.002074`, paired 95% CI `[+0.001894,+0.002254]`, `60/60` wins,
severe manual artifacts `0`. На неизменённой confirmation `0:60` F снова
выиграл у h28 `60/60`, gain `+0.002220`, CI
`[+0.002042,+0.002397]`, и прошёл every safety bound, но абсолютный mean был
`0.266852<0.27`. Это единственный failed confirmation check, поэтому frozen
gate запрещает promotion и production integration.

Обе панели исторически target-exposed через
`outputs/legacy-upgrade/calibration700-champion/report.json` и другие workspace
reports. Это не fresh calibration, не holdout и не unbiased generalization
estimate. Primary, confirmation, development panels и объявленные concurrent
new-method panels взаимно не пересекаются, но это не отменяет историческое
exposure.

## Почему появился v2

Edge-protected v1 использовал `h20` у Sobel/grid mask и `h35/h40` только в flat
regions. Он прошёл primary `0.274239`, но провалил confirmation absolute gate
с `0.249786`. При этом global h28 diagnostic был существенно сильнее кандидата
на обеих панелях. v2 проверил ровно одну новую композицию:

- exact v1 `t40` soft mask по-прежнему строится из независимого h20;
- protected pixels теперь берутся из independent single-pass h28, а не h20;
- flat pixels берутся из independent single-pass h40;
- output равен
  `rint(mask*h28 + (1-mask)*h40)` без routing и повторного NLM.

Arm выбрали на уже открытых development records до preregistration:

- combined edge-v1 `120:168`: F `0.271490`, h28 `0.269263`, gain `+0.002227`,
  `48/48`;
- historical DualNAF-stack `636:668`: F `0.269138`, h28 `0.266820`, gain
  `+0.002318`, `32/32`;
- добавление старого DualNAF alpha `.125` дало лишь `+0.000565` к F и `17/32`,
  поэтому neural renderer заранее исключён из v2.

После freeze дополнительный development-only sweep на тех же уже открытых 48
нашёл более высокий `h50/t60` (`+0.003580` против h28). Он появился **после**
immutable preregistration и не изменил arm, threshold, code или gate v2.
`h50/t60` остаётся лишь future evidence; подменять им этот результат нельзя.

## Legal inference boundary

Для каждого board общий pipeline:

1. bilateral directional scores -> `solve_buddies(max_edges=96)`;
2. strict upright bijection всех 576 original `20x20` tiles;
3. frozen RGB seam offsets -> bounded luminance gains;
4. независимые proper colored-NLM single passes `h20`, `h28`, `h40` на одном
   harmonized canvas;
5. mask из h20: Sobel magnitude `>=40`, все fragment boundaries, `3x3`
   dilation, Gaussian `sigma=1`;
6. target-blind fixed h28/h40 blend.

Нет sequential/multi-pass NLM, global final `h>=30`, filename/per-board routing,
target/reference/cross-board pixels, tile substitution, rotation, warp или
geometry change. A/h20 используется как baseline и mask source, B/h28 — strong
control и safe-pixel source, F — единственный promotable candidate.

## Immutable protocol

Config:
`configs/edge_protected_nlm_h28safe_reused_calibration_preregistered_v2.json`,
SHA-256
`fcb48204015d240d400aec4e5e9d95f0564a5781e305b5b59cf23452da5ede0d`.
Shared selector namespace `aiijc-puzzle-experiments-v1`, seed `20260829`:

- primary ranked `60:120`, filename digest
  `9d37fb3e9a57c59e83e10794a812f1992d871ff61e98403c25aededcb2934e61`,
  filename+input digest
  `96dca03277355c1c1d0ab9fc5c31f95270c4c1ac4ebb871ef7a0b06b549e4b69`;
- confirmation ranked `0:60`, filename digest
  `f15d8258f51e9ac48b41aa0d3e05d8582659e437cdbd2eefa73859ae31ae3f9d`,
  filename+input digest
  `24ae62e80d73921c554e89d709c64a8959baf0aaee73a4a501f5f7a61d44de26`;
- overlap primary/confirmation, development `120:168`/`636:668` и concurrent
  new-method calibration ranges `168:264`: везде `0`.

До каждого target decode read-only NPZ сохраняли dirty input, layout, raw,
harmonized, independent h20/h28/h40, exact binary/soft mask и A/B/F predictions.
Commitment связывает exact roster, каждый array/file digest и все runtime source
hashes. Все strict permutation audits прошли.

| Panel | Commitment file | Payload | Target receipt |
|---|---|---|---|
| Primary | `ee8fed33aa2faf8584174de593bb100d456371c0a98bc296ccb4ba04209b8db1` | `35b8ad5f719625fde8046c5c8786fe40f8679372a0fd7795004a2d6603632e14` | `1c3879b83ca0f3dab35b3a6c5dff86abfbabf8905952a32637daef77b2b4a76e` |
| Confirmation | `6012d86f35f0bc8a43fdee84dfda1d4759b2bd65d5b941862786d03ad2fb75f3` | `badd70740e2906556544e2723573d30553c3b31e3c66b6d36d81d4575628c2d0` | `c575c628bdde7a0e09e9afef944dc5da2251a7b9322de10a667bae726fc84ad0` |

## Frozen gate

F должен одновременно пройти:

- mean RGB SSIM `>=0.27`;
- paired t CI lower `>0` против A и B;
- не менее `45/60` wins против каждого control;
- luma-gradient retention mean/min к A `>=0.80/0.70`;
- chroma-gradient retention mean/min `>=0.80/0.70`;
- luminance-Laplacian retention mean/min `>=0.72/0.60`;
- relative grid ratio mean/max `<=1.05/1.12`;
- protected fraction mean `[0.40,0.75]`, every board `[0.30,0.85]`;
- clipped-fraction increase `<=0.01`, distinct from A/B на 60/60;
- severe manual artifacts `0` на всех 60 full canvases и fixed zoom indices
  `0,19,39,59`.

Confirmation разрешалась только после numeric PASS и отдельного root manual PASS,
привязанного к exact report/commitment/sheet hashes. Менять F, mask или thresholds
между panels было запрещено.

## Primary `60:120`: numeric + manual PASS

| Arm | Mean RGB SSIM | Gain F vs arm | F paired 95% CI | F wins |
|---|---:|---:|---:|---:|
| A h20 | 0.267331 | +0.012755 | `[+0.011907,+0.013604]` | 60/60 |
| B h28 | 0.278013 | +0.002074 | `[+0.001894,+0.002254]` | 60/60 |
| **F h28-safe/h40-flat** | **0.280087** | — | — | — |

F safety к A:

- luma gradient mean/min `0.865082/0.794532`;
- chroma gradient mean/min `0.967557/0.917545`;
- Laplacian mean/min `0.741107/0.669663`;
- grid ratio mean/max `0.778458/0.837551`;
- protected fraction mean/min/max `0.526861/0.421168/0.662231`;
- clipped increase `0`, distinct from A и B `60/60`.

Root просмотрел шесть 10-board full-canvas pages и fixed zoom sheet. Manual
PASS: severe artifacts `0`, включая отсутствие нового smooth blob через
protected edge/grid, identifiable-edge erasure, halo/contour, ухудшения fragment
boundary и локального discontinuity между h28/h40 regions. Existing incoherent
mosaic layout остаётся отдельным manual-disqualification risk и не считается
новым denoise artifact. Bound review SHA-256:
`6ee3d1cfec637dbcada5a11c327250bb84baf6079692614e483ae4daef1ac140`.

Primary report SHA-256:
`0f81dd82fee9c71724311a178f91f7997c1ca3466540ae7be96f775481363139`.

## Confirmation `0:60`: absolute FAIL

| Arm | Mean RGB SSIM | Gain F vs arm | F paired 95% CI | F wins |
|---|---:|---:|---:|---:|
| A h20 | 0.253435 | +0.013417 | `[+0.012627,+0.014206]` | 60/60 |
| B h28 | 0.264632 | +0.002220 | `[+0.002042,+0.002397]` | 60/60 |
| **F unchanged** | **0.266852** | — | — | — |

Confirmation safety также прошёл полностью: luma mean/min
`0.865966/0.813688`, chroma `0.969216/0.922900`, Laplacian
`0.739465/0.699783`, grid mean/max `0.772749/0.839873`, protected fraction
mean/min/max `0.533210/0.365742/0.635885`, clipped increase `0`. Единственный
failed check — `candidate_mean_rgb_ssim_min`: `0.266852<0.27`. Поэтому
`provisional_winner=null`; confirmation manual PASS не создавался.

Confirmation report SHA-256:
`1f541d606a6b78727cb6013fe0bf9e5e3e092072b3bf994c3e0cda3170547252`.

## Строгий вывод

Descriptive pooling обеих preregistered reused panels (не новый gate) даёт 120
boards: A `0.260383`, B `0.271322`, F `0.273469`. F выиграл у B `120/120`, mean
gain `+0.002147`, paired 95% CI `[+0.002022,+0.002272]`; против A gain
`+0.013086`, CI `[+0.012512,+0.013660]`, `120/120`. Значит, fixed F — очень
устойчивое legal pixel-tail improvement над h28 для этого reused-calibration
roster.

Но F не решает layout: absolute score меняется вместе со сложностью panel, а
confirmation осталась ниже frozen `0.27`. Этот experiment нельзя выдавать за
promoted final solution. Он доказывает переносимый relative tail signal и
manual safety, но production по текущему gate остаётся неизменным.

## Artifacts и воспроизведение

- Primary sheets: overview SHA `57308b352ca8da8eb428a840049d90893ca77278bcd0997c793840c92c6ed5d2`,
  zoom SHA `e822afcc237cb46ca81a1959c0ae6420febe89b74ec113abbf23119c9cdce436`,
  page SHAs `a474c785…`, `d493d239…`, `c0a5b9f0…`, `75a2649e…`,
  `e0677765…`, `753d77c8…`.
- Confirmation sheets: overview SHA
  `ea4e1bad542e28c7e874bf54456a6b0b9764899600b61a1161536c55834a9627`,
  zoom SHA `a5ee5cc136ce6fbef267f2c2d23c3b7a18b5eebf17c0ed1b1d2ed8b7644b04c3`,
  page SHAs `9579c4ad…`, `da23ef55…`, `d8fd0465…`, `0bcc68b9…`,
  `cb47774a…`, `682ca9b0…`.

```bash
uv run python scripts/run_edge_protected_nlm_v2.py --mode primary --phase prepare
uv run python scripts/run_edge_protected_nlm_v2.py --mode primary --phase score
uv run python scripts/run_edge_protected_nlm_v2.py --mode confirmation --phase prepare
uv run python scripts/run_edge_protected_nlm_v2.py --mode confirmation --phase score
```

Все completed phases single-use и fail closed на rerun. Competition holdout,
test, frozen edge-v1 files, production code и production ZIP не менялись.
