# Joint pairwise edge ranker

## Цель и compliance-граница

Эта линия оптимизирует только реальную геометрию пазла. Каждый decoder output —
строгая перестановка всех 576 входных фрагментов в сетке 24×24. Сначала строится
`raw = assemble_tiles(input_tiles[layout])`, затем разрешён frozen coloured NLM
`h=9`. Constant, template, clean-target и public-source substitution запрещены.

Calibration target не участвует в candidate generation, neural score, solver или
restoration. Runner сначала для **всей** evaluation-панели фиксирует matrices,
layouts, raw/NLM images и SHA-256, и только затем открывает targets для labels,
SSIM и placement/adjacency diagnostics. Holdout и test не открываются.

## Почему это не повтор M419/P31/content verifier

- P31 был raw 8-pixel seam CNN и почти не изменил recall20.
- M419 сравнивал пять уже вырезанных seam patches внутри top-5 и извлёк около
  4% shortlist headroom.
- Первый content verifier видел full-tile patch sets и учился на content-RMSE
  multi-positive target; на scale128 он улучшил exact, но ухудшил all-row
  content≤20, поэтому его formulation закрыта.
- V22 был multiscale boundary cross-attention поверх внешнего V18 top-32;
  checkpoint и caches в Git отсутствуют.

Новый scorer — raw-domain joint cross-encoder proposed pair: он видит
гипотетический 20×40 join одновременно как full-resolution 8+8 seam и как
downsampled whole-tile collage. Exact recovered neighbour даёт listwise CE;
trusted candidate mappings дополнительно дают clean symmetric extrapolation
teacher. Training negatives — union top-5 четырёх inference-visible emitters
`raw, tile_z, bilateral, gray`. Это проверка ранее resource-stopped joint
raw-domain pair route, а не ещё один pooled bi-encoder.

Полный bilateral score существует для всех 576×576 пар. Neural residual меняет
только union shortlist; вне него matrix остаётся frozen bilateral. Decoder один
и тот же для control и learned: `solve_buddies(max_edges=96)`.

## Guarded pre-denoise ablation

Raw evidence обязательно в обеих руках:

- `raw`: RGB + per-tile normalised RGB;
- `dual`: те же raw channels + low-strength bilateral, его per-tile-normalised
  view и signed raw−bilateral residual.

Это не denoise-only и не NLM-per-tile E19: модель всегда может обратиться к raw,
а ablation использует одинаковые rows, labels, optimiser и gate.

## Frozen protocol и gates

Shared selector: namespace `aiijc-puzzle-experiments-v1`, seed `20260829`.
Plumbing-check использует отдельную calibration record offset 200. Gate-панель
начинается с calibration offset 48, то есть не пересекается с ранее открытыми
first-48 supply/verifier panels.

До smoke зафиксирован local-edge gate против artifact-free bilateral:

1. all pooled exact R@1 delta ≥ `+0.005`;
2. trusted-query pooled exact R@1 delta ≥ `+0.010`;
3. all/trusted right и down R@1 delta каждый ≥ `0`.

Smoke: train first 16 manifest-train, calibration records 48:52, 3 epochs,
128 trusted exact-present rows/board, union k=5. Raw и dual проходят независимо.
Scale разрешён только победителю, прошедшему smoke gate. Full-image raw/NLM SSIM
и placement/adjacency публикуются только после local gate; сравнение paired с
bilateral+buddies96 на тех же inputs и decoder.

## Smoke results

Команды отличались только `--view-mode raw|dual`:

```bash
.venv/bin/python scripts/run_edge_ranker.py \
  --output-dir outputs/edge-ranker/smoke-<arm>-train16-cal4 \
  --train-limit 16 --eval-limit 4 --eval-offset 48 --epochs 3 \
  --rows-per-board 128 --batch-rows 24 --pair-batch 1024 \
  --candidate-k 5 --view-mode <arm> --width 24 --hidden 48 --device mps
```

| arm | all R@1 delta | trusted R@1 delta | adjacency delta | raw SSIM delta | NLM9 SSIM delta | gate |
|---|---:|---:|---:|---:|---:|---|
| raw | +1.766 pp | +5.393 pp | +2.604 pp | −0.000425 | −0.001521 | pass |
| dual | +0.928 pp | +4.061 pp | +1.676 pp | +0.001910 | +0.000953 | pass |

Оба arm прошли заранее заданный local gate, но proxy разошлись: raw лучше по
edge/adjacency, dual — по SSIM на малой панели. Поэтому bounded scale обоих arm
был информативнее выбора по одному шумному endpoint. Это единственное
расширение roster; architecture, optimiser и остальные параметры не менялись.

## Scale results

Scale использует first 64 manifest-train и 12 новых calibration records 52:64,
не пересекающихся со smoke 48:52:

```bash
.venv/bin/python scripts/run_edge_ranker.py \
  --output-dir outputs/edge-ranker/scale-<arm>-train64-cal12 \
  --train-limit 64 --eval-limit 12 --eval-offset 52 --epochs 3 \
  --rows-per-board 128 --batch-rows 24 --pair-batch 1024 \
  --candidate-k 5 --view-mode <arm> --width 24 --hidden 48 --device mps
```

### Local exact-edge gate

| scope / direction | bilateral | raw learned | raw delta | dual learned | dual delta |
|---|---:|---:|---:|---:|---:|
| all pooled R@1 | 0.075408 | 0.106658 | **+3.125 pp** | 0.074502 | −0.091 pp |
| all pooled R@5 | 0.182518 | 0.199049 | +1.653 pp | 0.107111 | −7.541 pp |
| all right R@1 | 0.074426 | 0.105374 | +3.095 pp | 0.071407 | −0.302 pp |
| all down R@1 | 0.076389 | 0.107941 | +3.155 pp | 0.077597 | +0.121 pp |
| trusted pooled R@1 | 0.149623 | 0.237636 | **+8.801 pp** | 0.173303 | +2.368 pp |
| trusted pooled R@5 | 0.309933 | 0.399413 | +8.948 pp | 0.236169 | −7.376 pp |
| trusted right R@1 | 0.146519 | 0.240068 | +9.355 pp | 0.172149 | +2.563 pp |
| trusted down R@1 | 0.152612 | 0.235294 | +8.268 pp | 0.174414 | +2.180 pp |

Raw проходит все условия. Dual проваливает deployable all rows, right и R@5;
его full-board metrics по правилу не открывались. Guarded bilateral channels не
перенесли smoke gain и отклонены в этой formulation.

### Strict bijective decoder, raw arm only

На каждой доске baseline и learned используют один `buddies96`, после которого
`validate_layout` подтверждает 576 уникальных индексов. Images собраны из
исходных dirty fragments; NLM применяется только после assembly.

| metric, mean over calibration-12 | bilateral+buddies96 | raw ranker+buddies96 | delta |
|---|---:|---:|---:|
| raw SSIM | 0.115865 | 0.116101 | **+0.000235** |
| NLM9 SSIM | 0.212201 | 0.211710 | **−0.000491** |
| adjacency | 0.038496 | 0.063859 | **+2.536 pp** |
| right adjacency | 0.038798 | 0.063557 | +2.476 pp |
| down adjacency | 0.038194 | 0.064161 | +2.597 pp |
| direct placement | 0.001881 | 0.002170 | +0.029 pp |
| translation-aligned placement | 0.009838 | 0.012876 | +0.304 pp |

Scale runtime на MPS: board preparation `86.67 s`, training `51.98 s`, freeze
inference+decoder+NLM `24.29 s`, target-assisted evaluation `1.53 s`. Dual:
`89.79/51.48/23.77/0.66 s` соответственно. Train/calibration selection digests
raw/dual одинаковы: `c12899e6…f3755fedc` и `60f0c076…3f2e6d7c`.

Artifacts:

- raw report SHA-256 `a016ea96f5bb7d9a5ce4940d5a6c75288d6a0e234775dade114c777205c3af26`;
- raw checkpoint SHA-256 `d18ff864c63170d5fcdb868d672a60515d10ac600afa2ed0424000921ecbb21a`;
- dual report SHA-256 `baf0667e4428d2f4893f5123afdf505af8e44ffb05fe526d92bd1f0d1bb3089f`;
- dual checkpoint SHA-256 `57bfe005a2696ab7bc07379e2c40a7cad8c574a9f1fcce6f6111df38d56e6e51`.

## Verdict and limitations

Raw cross-encoder дал крупный, direction-balanced и scale-positive прирост
геометрии; это первый полезный runnable auxiliary этой новой линии. Однако
frozen NLM9 официальный endpoint немного ухудшился. Поэтому checkpoint нельзя
продвигать как end-to-end champion или автоматически менять текущий compliant
baseline: retain только как layout/edge auxiliary для отдельной restoration
проверки. Dual arm закрыт.

Ограничения: exact labels восстановлены target-assisted Hungarian и остаются
приближёнными; trusted scope недоступен при inference; один seed; scale всего 64
train / 12 calibration boards; scorer меняет только union top-5 и не может
восстановить отсутствующего кандидата; buddies96 слабо превращает adjacency в
absolute placement; NLM может маскировать небольшой raw-layout gain. Holdout и
test не открывались.

CPU plumbing-check на train1 и отдельной calibration record offset 200 прошёл
end-to-end, но не используется как quality evidence.

## Проверка manual-layout риска под frozen final tail

После выбора compliant final tail checkpoint raw-arm был повторно проверен **без
дообучения**. SHA-256 checkpoint `d18ff864…bb21a`; перед inference runner
проверил неизменность contract, train64 roster, validation protocol и semantic
hashes `edge_ranker/candidate_supply/legacy_upgrade/protocol`.

Панель — shared-selector calibration offset `204:228`, следующие 24 records
после contiguous learned-matcher панелей, закончившихся на offset 204. Она не
пересекается с train64 или прежней edge-ranker evaluation `52:64`. Формулировка
freshness относится к checkpoint и legal learned-matcher sequence: старый
quarantined calibration700 sweep по определению уже открывал весь calibration и
исключён из model-selection claims.

Обе arms используют один decoder `buddies96` и собирают **исходные upright
dirty tiles** строгой перестановкой. Один и тот же frozen tail применяется после
assembly: RGB seam offsets → bounded luminance gains → coloured NLM `h=20`, один
pass. Все 48 raw audits (24 boards × 2 arms) подтвердили 576 уникальных tiles,
точный declared reassembly, одинаковый tile multiset и сохранение raw pixels.
Все scores, layouts, audits, raw/harmonized/final images и hashes были записаны
в prediction commitment до первого чтения target.

| metric, mean over fresh calibration-24 | bilateral+buddies96 | raw ranker+buddies96 | delta |
|---|---:|---:|---:|
| all pooled exact edge R@1 | 0.076238 | 0.105601 | +2.936 pp |
| all pooled exact edge R@5 | 0.187311 | 0.200483 | +1.317 pp |
| adjacency | 0.038081 | 0.063783 | **+2.570 pp** |
| direct placement | 0.001157 | 0.002894 | +0.174 pp |
| translation-aligned placement | 0.009476 | 0.011791 | +0.231 pp |
| raw SSIM | 0.101145 | 0.100485 | −0.000660 |
| RGB+luma SSIM | 0.104752 | 0.104918 | +0.000166 |
| RGB+luma+NLM h20x1 SSIM | 0.219109 | 0.217611 | **−0.001498** |

Preregistered dual gate:

- adjacency delta CI95 lower `> 0`: **pass**, mean `+0.025702`, percentile
  bootstrap CI `[+0.021286, +0.030042]`, улучшение на 24/24 boards;
- final-tail SSIM noninferiority CI95 lower `>= −0.003`: **fail**, mean
  `−0.001498`, CI `[−0.005890, +0.002998]`, 10 wins / 14 losses.

Manual sheet подтверждает статистический вывод: ranker местами формирует более
крупные связные цвето-текстурные компоненты, но ни одна arm не восстанавливает
глобально читаемую сцену; локальный прирост geometry недостаточен и после NLM
остаётся риск заметно худшего endpoint. Поэтому checkpoint остаётся только
диагностическим layout auxiliary. Fresh confirmation, scale256 retraining и
production integration **не разрешены** этим результатом.

Воспроизведение (MPS runtime: `50.51 s` freeze + `3.54 s` target diagnostics):

```bash
.venv/bin/python scripts/run_edge_ranker_final_tail.py \
  --device mps --offset 204 --count 24 \
  --output-dir outputs/edge-ranker/manual-tail-raw-checkpoint-cal24-offset204
```

Artifacts:

- `outputs/edge-ranker/manual-tail-raw-checkpoint-cal24-offset204/report.json`,
  SHA-256 `e7c94a91…8caaee`;
- `prediction-commitment.json`, SHA-256 `fc601a93…ae7b8`;
- `manual-layout-sheet.png`, SHA-256 `aa43d20e…cb008`.

## Final broader-candidate scale attempt

The separately preregistered [candidate-k16 / train256 experiment](edge-ranker-k16-scale.md)
closed the one remaining scale/candidate-budget hypothesis. It expanded exact
candidate coverage from `0.298611` to `0.473279` and improved adjacency on
24/24 boards, but learned absolute adjacency was only `0.062689` and the frozen
RGB+luma+NLM h20x1 endpoint lost `-0.009386`, CI
`[-0.016116,-0.002948]`. Three of five promotion conditions failed, so the
disjoint confirmation, holdout, test and production integration were not run.
