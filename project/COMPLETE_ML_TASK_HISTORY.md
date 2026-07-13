# Полная история решения задачи restoration + tile assembly

Дата сборки отчёта: 2026-07-13  
Статус: финальный исследовательский отчёт; production зафиксирован на лучшем
пользовательском результате `0.218`.

## 1. Задача и критерий успеха

Нужно восстановить RGB-изображение `480x480`, составленное из `24x24 = 576`
перемешанных тайлов `20x20`. Каждый тайл независимо испорчен изменением
яркости и контраста, сильным шумом, Gaussian blur и JPEG-артефактами. Итоговая
метрика — средний RGB SSIM:

```python
skimage.metrics.structural_similarity(
    target,
    prediction,
    channel_axis=2,
    data_range=255,
)
```

Следовательно, решение обязано одновременно:

1. восстановить перестановку всех 576 тайлов;
2. убрать локальные искажения без разрушения границ;
3. выровнять независимые цветовые/яркостные сдвиги;
4. воспроизводимо сформировать 700 RGB PNG `480x480` в корне `submission.zip`.

Мы намеренно исключили два пути, которые давали высокий локальный или публичный
score, но не решали задачу: копирование train target по совпадающему имени и
одноцветные predictions, оптимизирующие особенности SSIM. Они зафиксированы как
leak/metric abuse и не входят ни в один продвигаемый артефакт.

## 2. Принципы честной валидации

- Все train/validation/audit разбиения выполнялись по целым source images, а не
  по тайлам одного изображения.
- Denoise V2 обучался на точных synthetic pairs: corruption применялся к уже
  вырезанному чистому тайлу, поэтому перестановка и Hungarian matching не
  участвовали в основном supervision.
- Основные assembly-проверки использовали две независимые corruption panels:
  `primary_kornia` и `independent_libjpeg`.
- Для actual-input calibration layout строился input-only; target открывался
  только после freeze всех layouts и их хешей.
- Поздние sealed gates физически разделяли input-only Phase A и label-only
  Phase B. Holdout не создавался до прохождения calibration gate.
- Candidate не продвигался по одному красивому примеру. Использовались paired
  deltas, source wins, panel-wise non-regression и bootstrap confidence bounds.
- Если несколько честных попыток не давали устойчивого сигнала, ветка
  закрывалась вместо постфактум ослабления gate.

Главные split/config sources:

- `configs/denoise_splits_seed20260710.json`;
- `configs/denoise_validation_quarantine_v1.json`;
- `src/puzzle_assembly/protocol.py`;
- `src/puzzle_assembly/panels.py`;
- `src/puzzle_assembly/metrics.py`.

## 3. Лучший наблюдавшийся результат

Финальный production pipeline состоит из:

1. tilewise TileNAF restoration и `0.5` renderer blend с seam-trained TileNAF;
2. C1/HBT compatibility rank fusion;
3. soft-cycle seed;
4. QAP `w4`, boundary weight `0.05`, 25 iterations, 2 restarts;
5. post-assembly RGB seam-graph harmonization;
6. bounded luminance-gain correction поверх RGB harmonization.

Предыдущий RGB-only submission:

- leaderboard: `0.2167844489529071`;
- archive:
  `runs/assembly_v1/harmonized_submission/local_full700_v1/submission.zip`.

До harmonization пользователь сообщал округлённый LB `0.203` для QAP render;
точное число в локальном provenance не сохранено. Таким образом, основной
публичный скачок дал именно renderer-side RGB harmonizer, а не новая
перестановка.

Добавочная luma-коррекция прошла две source-disjoint validation checks:

- calibration source-macro delta: `+0.001431471`;
- confirmation source-macro delta: `+0.001721901`;
- confirmation wins: `32/32`;
- обе corruption panels положительны, seam error не ухудшен.

Пользователь сообщил отображённый leaderboard score `0.218` для luma archive.
Это текущий лучший наблюдавшийся результат, показанный платформой с точностью
до трёх знаков. Exact RGB-only LB равен `0.2167844489529071`; exact luma LB и
exact delta неизвестны. Разность отображённых точек равна `+0.001215551`, а при
обычном округлении до трёх знаков консервативная нижняя граница delta была бы
`+0.000715551`. Последняя величина явно помечена как inference из правила
округления, а не значение, полученное с платформы.

Canonical luma archive:

- path:
  `runs/assembly_v1/kaggle/luma_harmonized_submission_output/v1/submission.zip`;
- SHA-256:
  `099d1c5fe69cda8519a4f19750cb3a481ac87999c294a35e19691a849d4c6096`;
- LB observation:
  `runs/assembly_v1/kaggle/luma_harmonized_submission_output/v1/leaderboard_observation.json`.

## 4. Главный научный вывод

Основной bottleneck — перестановка, а не denoise:

- при правильном порядке selected TileNAF даёт ordered-image SSIM около
  `0.713–0.724`;
- на одной frozen неправильной QAP layout raw render даёт около `0.11025`, а
  TileNAF render — `0.18282`;
- post-assembly RGB correction на full32 добавляет `+0.01286/+0.01311`, а на
  confirmation32 `+0.01175/+0.01181` для двух corruption panels;
- luma добавляет ещё около `+0.0017`;
- на clean shuffled tiles простой solver достигает SSIM `0.96172` и adjacency
  `0.94112`.

То есть restoration существенно помогает, а grid optimizer способен решать
задачу при хорошем compatibility matrix. Ошибка возникает в оценке истинного
соседства маленьких независимо испорченных тайлов.

Особенно важен candidate-graph oracle: union top-32 C1/HBT уже содержит в
среднем `72.98%` истинных рёбер, но production layout сохраняет лишь около `6%`
adjacency. Post-hoc true-edge filtering поднимает mean SSIM примерно до `0.6273`,
а truth-assisted component translation ceiling — до `0.7091`. Следовательно,
большой потенциальный gain находится в high-precision edge verification и
global consistency, а не в очередном увеличении числа QAP iterations.

## 5. История restoration / denoise

### 5.1. Legacy q90 pseudo-pair CNN

Первая рабочая ветка восстанавливала соответствия corrupted-to-clean через
descriptor costs + Hungarian assignment, затем учила residual CNN. Она дала
заметный прирост над raw/NLM, но supervision зависел от pseudo-matching и часть
validation sources была использована в старом model selection. Поэтому q90
сохранён только как внешний baseline, а не production model.

Evidence: архивный deprecated `DENOISE_PIPELINE.md`, authoritative описание
rollback и replacement — `DENOISE_V2.md` и
`runs/denoise_v2/denoise_v2_bundle_20260710/docs/DENOISE_PIPELINE_V1_DEPRECATED.md`.

### 5.2. Leakage-safe TileNAF synthetic training

TileNAF обучался на точных synthetic corruptions после crop `20x20`, без
неизвестной перестановки. Основной run: 50,000 updates, batch 256, 4 h 25 min на
P100. Selected checkpoint:

- `runs/denoise_v2/release/selected_tilenaf_synth_50k.pt`;
- SHA-256:
  `77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734`.

Fixed synthetic validation:

- raw tile SSIM `0.56302` -> restored `0.80828`;
- paired Kornia `0.52435` -> `0.82267`;
- independent libjpeg `0.51295` -> `0.81928`;
- boundary MAE `19.1869` -> `14.5694`.

На one-shot sealed 350-source real-pair gate selected TileNAF также устойчиво
победил raw, NLM и legacy q90. Primary source-macro tile SSIM:

- raw `0.67570`;
- NLM `0.72041`;
- legacy `0.77100`;
- selected TileNAF `0.81098`.

Decision: promoted.

Evidence: `DENOISE_V2.md` и
`runs/denoise_v2/denoise_v2_bundle_20260710/results/final_gate/selected_final_gate_report.json`.

### 5.3. CPU pre-finetune spending gate

Перед расходом GPU synthetic-50k сравнивался с raw, OpenCV NLM и legacy на 257
чистых calibration sources. TileNAF получил `0.8075045` против legacy
`0.7682083`: delta `+0.0392962`, bootstrap CI
`[+0.0369888,+0.0416223]`. Решение
`proceed_to_bounded_real_pair_finetune` разрешало ровно один conservative
fine-tune. Это diagnostic spending gate, а не отдельная model promotion.

Evidence:
`runs/denoise_v2/release_readback/20260710T074500Z/prefinetune_cpu_v3_current/prefinetune_calibration_report.json`.

### 5.4. Conservative real-pair fine-tune

Fine-tune исключил 93 quarantined sources и использовал 257 calibration sources;
350-source gate оставался sealed. Primary calibration gain составил `+0.00183`
с положительным bootstrap lower bound, но заранее требовалось `+0.003`. Gate не
открывался, weights были откатаны к synthetic EMA побитно.

Decision: rollback, threshold не ослаблялся.

Evidence:
`runs/denoise_v2/denoise_v2_bundle_20260710/results/real_finetune/real_finetune_result.json`.

### 5.5. Block5x5 и analytic hybrid

Варианты с контекстом 5x5 улучшали ordered-image SSIM примерно на
`+0.0041…+0.0055`, но немного снижали tile SSIM и ухудшали boundary MAE.
Analytic hybrid screen тоже не показал устойчивого совместного сигнала.

Decision: `stop_no_development_signal`; не продвигались.

Evidence:

- `runs/denoise_v2/kaggle_block5x5_readback/output_v2/block5x5_selection.json`;
- `runs/denoise_v2/block5x5_hybrid_development_v1.json`.

### 5.6. Renderer-only seam denoiser и post-assembly CK

Seam-trained TileNAF не был стабильным scorer, но на frozen layouts дал
renderer-only gain `+0.000502`, bootstrap 95% CI полностью выше нуля. Позднее
input-only RGB seam-graph harmonizer дал гораздо больший прирост на двух свежих
32-source panels: примерно `+0.0118…+0.0131`, `32/32` wins на каждой panel.
Bounded luma gain поверх RGB дал ещё `+0.001721901` на confirmation.

Decision: текущий promoted pixel-side pipeline — selected TileNAF + `0.5`
renderer-only seam-trained TileNAF blend + RGB CK + luma. Seam checkpoint не
promoted как compatibility scorer, но promoted как renderer asset канонического
submission. Термин rollback относится только к failed real-pair fine-tune.

Evidence:

- `runs/assembly_v1/postassembly_harmonizer/actual_qap_v1_full32_score_20260712T1520Z/report.json`;
- `runs/assembly_v1/postassembly_harmonizer/actual_qap_confirmation_v1_score_20260712T1535Z/report.json`;
- `runs/assembly_v1/kaggle/actual_qap_luma_confirmation_output/v1/actual_qap_luma_confirmation_report.json`.

### 5.7. Learned contextual post-assembly refiner

Contextual refiner смог улучшить изображения при заведомо правильном layout:
delta `+0.0068733/+0.0064815` на двух panels. Но на реальных frozen QAP layouts
он дал `-0.0011441/-0.0012744`; обе paired CI полностью отрицательны, face ROI
также ухудшился. Status: `smoke_gate_failed_stop_or_pivot`, checkpoint не
promoted (SHA prefix `88bddc0c`, suffix `f344e`).

Evidence:
`runs/assembly_v1/kaggle/contextual_refiner_smoke_output_v1/contextual_refiner_smoke_report.json`.

## 6. История assembly / solver

Статусы ниже означают:

- **PROMOTED** — вошло в production или текущий submission;
- **RETIRED** — получен честный отрицательный/недостаточный результат и тот же
  рецепт продолжать нельзя;
- **INCONCLUSIVE** — целевая научная метрика не получена;
- **GATED OUT** — зависимая ветка намеренно не запускалась после провала
  prerequisite;
- **DIAGNOSTIC ONLY** — target-assisted oracle или другой неинференсный анализ.

### 6.1. Ранний normal pipeline и classical baseline

Самая ранняя рабочая normal-ветка сочетала tile restorer, edge assembly,
SideEmbeddingNet и локальный Hungarian repair слабых клеток. Candidate
`side_all_repair16_a002` имел mixed-panel SSIM `0.19875` на 48 источниках;
известный честный публичный fallback имел LB `0.1901853834`. Позднее этот
пайплайн был полностью заменён leakage-safe Denoise V2 и новой assembly
валидацией. Metric-abuse score `0.3961` и same-name target copy были исключены.
Значение `0.19875` использовало legacy pseudo-GT/mixed validation и не
сопоставимо напрямую с поздними leakage-safe real16/real64 SSIM или LB.

Evidence: `FINAL_EXPERIMENTS_REPORT.md`.

Classical compatibility проверяла RGB/Lab/tone/PBC/MGC/C1, reciprocal
components, loops и confidence pruning. На clean shuffle weighted-L1 PBC дал
SSIM `0.961720`, adjacency `0.941123`; на corrupted actual inputs C1 дал
real64 `0.191869870`. Это был сильный historical baseline, позднее superseded
QAP.

Evidence:

- `runs/assembly_v1/development/clean_raw_1_g2_lp.json`;
- `runs/assembly_v1/real_cal/real_cal_64_selecteddenoise_classical.json`.

### 6.2. Local learned compatibility до QAP

| Ветка | Основной результат | Решение | Evidence |
|---|---|---|---|
| L0 seam-pair CNN | R1 `0.152231`, ниже classical | RETIRED | `runs/assembly_v1/kaggle/l0_gpu_full/` |
| L1 pooled side embedding | R1 `0.219486`, R32 `0.698299`, real16 `0.172663` | полезный feature, не promoted | `runs/assembly_v1/kaggle/l1_gpu_full/` |
| L1-v2 sequence embedding | R1 `0.203167`, хуже L1 | RETIRED | `runs/assembly_v1/kaggle/l1v2_gpu_full/` |
| T0 absolute-position Transformer | position accuracy `0.002658` | RETIRED | `runs/assembly_v1/kaggle/t0_gpu_full/` |
| X0 rank reranker | R1 `0.200153`, candidate recall `0.761096` | не promoted | `runs/assembly_v1/kaggle/x0_gpu_full/` |
| L1+X0+T0 | real16 `0.175864`, real64 `0.188669 < 0.191870` C1 | RETIRED как real16 overfit | `runs/assembly_v1/real_cal/real_cal_64_l1full_x0full_t0full.json` |
| Real pseudo-label L1 | exact R1 около `0.194` против base `0.219`; real16 `0.170359` | RETIRED, self-confirming degradation | `runs/assembly_v1/l1_real_pseudo/full_512x1.json` |
| G0 residual global matcher | R1 `0.217165` против frozen HBT `0.224072` | RETIRED на retrieval gate; target не открыт | `runs/assembly_v1/kaggle/g0_global_matcher_gpu/g0_global_matcher_512x2.json` |

Отдельный factorial проверил исходную идею «оставить только границы»:

| HBT input | Validation R1 |
|---|---:|
| denoised RGB+Sobel | `0.223845` |
| denoised RGB-only | `0.215636` |
| raw RGB+Sobel | `0.179008` |
| denoised Sobel-only | `0.034279` |
| raw Sobel-only | `0.015002` |
| binary edges | около `0.0075` |

Вывод: gradients полезны только дополнительным каналом к RGB; Sobel/binary-only
почти уничтожают сигнал. Denoise повышает retrieval, а не ломает его. HBT
RGB+Sobel сохранён как лучший learned feature, но сам по себе не победил C1 на
real layout.

Evidence: `runs/assembly_v1/kaggle/edge2vec_gradient_gpu/`.

### 6.3. Promoted directional QAP

Frozen production configuration:

- soft-cycle L1 top-k 8 seed;
- denoised C1+HBT rank fusion weight 4;
- QAP 25 iterations, 2 restarts;
- initial weight `0.75`, boundary weight `0.05`;
- noise scale `1.0`, 3 noisy components, 8 refine swaps.

На real16:

- soft-cycle seed `0.165431140`;
- ordinary QAP `0.182329628`;
- heavy QAP 40x4 `0.181305114`;
- boundary-QAP `0.182819915`;
- selected delta к seed `+0.017388775`, wins `16/16`, paired CI
  `[+0.012173,+0.023101]`.

Boundary term отдельно не доказан: преимущество над ordinary QAP всего
`+0.000490`, CI пересекает ноль. Но полный fixed configuration стал production
layout winner. RGB/luma harmonization — более поздний renderer поверх
неизменённой QAP permutation, а не часть layout solver.

Evidence: `runs/assembly_v1/kaggle/qap_tuning_night_output/v2/`.

### 6.4. Search и global optimization поверх той же энергии

| Ветка | Real16 / gate result | Решение |
|---|---:|---|
| Multi-phase RL top-k 4/8/16 | `0.16851 / 0.170996 / 0.172828` | RETIRED, paired CIs ниже нуля |
| LNS subset 64/192 | `0.171237 / 0.169914` | RETIRED |
| Cross-view soft-cycle | `0.175156` | RETIRED |
| Generic annealing 20k | `0.170495` | RETIRED |
| Protected annealing | w4 layout changed `0/8`; pure-HBT delta `-0.000041` | RETIRED |
| Line continuation | R1 `5.84–6.25%`, line-QAP real4 `0.170975 < 0.183733` | RETIRED |
| CP-SAT after base QAP | вернул тот же layout | RETIRED / no added value |

Target-only oracle по QAP+RL/LNS/cross/anneal pool достиг лишь `0.188504`.
Следовательно, увеличение search budget для той же pairwise energy не закрывает
разрыв.

Evidence:

- `runs/assembly_v1/FINAL_ASSEMBLY_REPORT.md`;
- `runs/assembly_v1/kaggle/anneal_upgrade_gate_output/v1/protected_anneal_exact_gate.json`;
- `runs/assembly_v1/kaggle/line_cpsat_gate_output/v1/ANALYSIS.md`.

### 6.5. Learned global/context branches

| Ветка | Ключевой результат | Решение | Evidence |
|---|---|---|---|
| Context reorganization | exact wrong positions `4597→4597`; real16 layout/SSIM без изменений | RETIRED scientific zero | `runs/assembly_v1/kaggle/context_reorg_gate_output/v1/context_reorg_gate_report.json` |
| 2x2 hyperedge verifier | AP `0.01593`, precision `6/344`, adjacency `0.06103→0.03442`, SSIM `0.18282→0.16122` | RETIRED | `runs/assembly_v1/kaggle/hyperedge_gate_output/v1/hyperedge_gate_report.json` |
| Frozen MAE energy | broad-pool Spearman `0.6518`, но gain лишь `+0.000730` | не promoted | `runs/assembly_v1/kaggle/mae_energy_gate_output/v2/` |
| MAE population search | competitive Spearman `0.0574`, pair accuracy `0.5202`, SSIM delta `-0.000813` | RETIRED | `runs/assembly_v1/kaggle/mae_search_gate_output/v3/` |
| DINOv2 4x4 superblock | development cell accuracy `0.0447`, Manhattan reduction `0.078` при gates `0.10/0.25` | RETIRED | `runs/assembly_v1/kaggle/dino_superblock_probe_output/v1/dino_superblock_probe_report.json` |
| LaMa masked consistency | три infrastructure attempts, correlation metric не получена, targets не открыты | INCONCLUSIVE, не scientific fail | `runs/assembly_v1/kaggle/lama_consistency_gate_output/v3/ANALYSIS.md` |

### 6.6. Neural upgrade matrix

| Ветка | Результат | Решение | Evidence |
|---|---|---|---|
| QAP w1 | fresh64 delta `+0.001375`, CI `[+0.000220,+0.002580]`, 35/64 wins | confirmed small gain, no promotion: gates `+0.005` и 40/64 не пройдены | `runs/assembly_v1/kaggle/qap_weight_confirmation_output/v3_verified/RESULT_SUMMARY.md` |
| ViT-Sinkhorn absolute assignment | selection `0.175902 < 0.201440`, holdout `0.196005 < 0.222600`, position accuracy около chance | RETIRED | `runs/assembly_v1/kaggle/vit_sinkhorn_pilot_output/v4_reports/` |
| Pair Transformer | R1 epoch1 `0.188406 < 0.199275` HBT, epoch2 `0.187047`, затем fail-closed nonfinite | RETIRED; дорогой QAP eval не открыт | `runs/assembly_v1/kaggle/pair_transformer_pilot_output/v2_failure/RESULT_SUMMARY.md` |
| Raw layout-energy Transformer | holdout AUC `0.9605`, но adjacency `-0.00286`, repair почти ноль | RETIRED | `runs/assembly_v1/kaggle/layout_energy_pilot_output/v1/RESULT_SUMMARY.md` |
| Layout-energy hybrid salvage | best `+0.000117`, learned-minus-equal-control CI пересекает ноль | RETIRED | `runs/assembly_v1/kaggle/layout_energy_hybrid_diagnostic_output/v3/RESULT_SUMMARY.md` |
| Positional diffusion | Kornia SSIM `-0.05978`, libjpeg `-0.06386`; adjacency около `-0.117` | RETIRED decisive fail | `runs/assembly_v1/kaggle/positional_diffusion_pilot_output/v2/RESULT_SUMMARY.md` |
| HBT continuation | R1 `0.223845→0.229676`, MRR `0.321852→0.328905`, но pre-gates не пройдены | RETIRED same-architecture continuation | `runs/assembly_v1/kaggle/hbt_continuation_output/v2_failure/RESULT_SUMMARY.md` |
| Dense all-pairs residual | R1 `0.182476→0.113423`, MRR `0.273801→0.204989` | RETIRED first gate | `runs/assembly_v1/kaggle/dense_pair_residual_pilot_output/verified_v1/` |

### 6.7. Candidate graph oracle и structured-solver chain

Candidate-graph oracle v4 — **DIAGNOSTIC ONLY**, но это самый важный ceiling:

- union recall `0.729789`;
- median true-edge largest component `545.5/576`;
- production-like QAP SSIM `0.193591`;
- truth-filtered oracle SSIM `0.627267`;
- adjacency `0.062358→0.385134`;
- truth-assisted translation ceiling SSIM `0.7091`, position accuracy `0.964`.

v1 был invalid из-за shape-descriptor bug, v2 потерял crash-safe push journal,
v3 был recovery-only; только exploratory v4 используется как ceiling, не как
формальный inference result (`safe_for_submission=false`). Incident evidence:

- `runs/assembly_v1/candidate_graph_oracle_v1_invalid_no_result/INCIDENT.md`;
- `runs/assembly_v1/candidate_graph_oracle_v2_stranded_phase_a_no_result/INCIDENT.md`.

Evidence:
`runs/assembly_v1/candidate_graph_oracle_v4_phase_b_output_exploratory_v2/EXPLORATORY_RESULT_SUMMARY.md`
и `candidate_graph_oracle_ceiling_report.json` в том же каталоге. Последний
содержит mean position accuracy `0.9636773`; summary отдельно документирует
SSIM `0.709094` и adjacency `0.939750`.

Последующие попытки превратить этот ceiling в input-only solver:

| Ветка | Результат | Решение |
|---|---|---|
| Full-union tabular HGB | AP `0.1742`, AUC `0.6801`; QAP SSIM delta `+0.000030` | не promoted |
| Higher-precision verifier p80 | AP `0.26865`, AUC `0.82586`, но mean component size `2.95`, rigid SSIM `0.19059 < QAP` | RETIRED standalone |
| HGB 4-cycle features | worst-panel AP delta `+0.000898 < +0.005` gate | RETIRED |
| HGB component sync | best macro SSIM `-0.002084` | RETIRED |
| HGB trust repair | adjacency около `+0.001`, macro SSIM `-0.0000175` | RETIRED |
| Robust translation sync + OT | macro SSIM около `-0.01421`, adjacency `-0.045…-0.053` | RETIRED |
| Parallel-tempered HGB anneal | best macro `+0.000008`, фактически no-op | RETIRED |
| D4 consensus | runtime около 41 s/source > predeclared 20 s; targets не открыты | RETIRED prerequisite |
| Independent-edge GNC-TLS | correctness fixture не восстановлен | RETIRED before calibration |
| Group-switch sync | synthetic rank-2 fixture: 16/16 wrong | RETIRED synthetic gate |
| Exact axis path cover | delta adjacency около нуля, ~40% rescue arcs, runtime до 130 s/source | RETIRED |
| LongSync-4 cycles | R1 `-0.0144/-0.0186`, AP `-0.0285/-0.0344`, 0/8 wins | RETIRED decisive fail |
| Dual LambdaRank | retrieval mixed; QAP SSIM `0.206636→0.201832`, 0/8 wins | RETIRED |
| ContinuationNet-0 | R1 `0.05967` против w4 `0.16950`; blend хуже обеих panels | RETIRED Calibration A |
| Binary edge verifier CNN | remote SIGKILL до valid report/checkpoint; local smoke не является model-selection evidence | INCONCLUSIVE infrastructure |
| GANzzle-style latent retrieval | не запускался после провала DINO prerequisite | GATED OUT |
| TileNAF latent edge embedding | против HBT R1 `-0.0099/-0.0108`; против W4 на отдельном holdout R1 `+0.00776/+0.00793`, но gate недобран | FAILED GATE / NOT PROMOTED |

Точные authoritative evidence paths для строк structured chain:

- full-union HGB: `runs/assembly_v1/full_union_tabular/v1/report.json` и `runs/assembly_v1/full_union_tabular/qap_v1_report.json`;
- verifier p80: `runs/assembly_v1/candidate_edge_verifier_v1_p80/report.json` и `runs/assembly_v1/candidate_edge_verifier_v1_p80/RESULT_SUMMARY.md`;
- HGB cycle: `runs/assembly_v1/kaggle/hgb_cycle_diagnostic_output/v1/hgb_cycle_diagnostic_report.json`;
- component sync: `runs/assembly_v1/kaggle/component_sync_calibration_output/v1/component_sync_calibration_report.json`;
- trust repair: `runs/assembly_v1/kaggle/hgb_trust_repair_calibration_output/v1/hgb_trust_repair_calibration_report.json`;
- RTS-OT: `runs/assembly_v1/kaggle/rts_ot_calibration_output/v1/rts_ot_calibration_report.json`;
- PT-HGB: `runs/assembly_v1/kaggle/pt_hgb_anneal_screen_output/v1/pt_hgb_anneal_screen_report.json`;
- D4: `runs/assembly_v1/kaggle/d4_consensus_gate_output/v3/d4_consensus_gate_report.json`;
- GNC-TLS: `runs/assembly_v1/kaggle/gnc_tls_sync_gate_output/RESULT_SUMMARY.md`;
- group-switch: `runs/assembly_v1/kaggle/group_switch_synth_gate_output/v3/group_switch_synth_wrapper.json`;
- path-cover: `runs/assembly_v1/kaggle/path_cover_gate_output/v1/RESULT_SUMMARY.md`;
- LongSync-4: `runs/assembly_v1/kaggle/longsync4_retrieval_output/v1/RESULT_SUMMARY.md`;
- dual LambdaRank: `runs/assembly_v1/kaggle/dual_lambdarank_retrieval_output/v5/RESULT_SUMMARY.md` и `runs/assembly_v1/kaggle/dual_lambdarank_qap_diagnostic_output/v3/RESULT_SUMMARY.md`;
- ContinuationNet-0: `runs/assembly_v1/kaggle/continuation_net0_gate_output/v1/RESULT_SUMMARY.md`;
- binary verifier: `runs/assembly_v1/kaggle/binary_edge_verifier_pilot_output/v3_failure/binary_edge_verifier_pilot_wrapper.json`;
- TileNAF latent edge: `runs/assembly_v1/kaggle/latent_edge_embedding_output/v3_complete/latent_edge_pilot/latent_edge_embedding_report.json` и `runs/assembly_v1/kaggle/latent_edge_embedding_output/v3_complete/production_anchor_holdout_208_224.json`.

Полный path/SHA inventory включён в machine-readable manifest итогового архива.

## 7. Counterfactual: почему фиксированные 4x4 блоки не решают задачу

На первых 16 whole sources `assembly_cal` был проведён read-only ad-hoc
counterfactual с чистыми target tiles и официальным SSIM. Внутри каждого блока
порядок был идеальным:

| Layout | Mean SSIM |
|---|---:|
| случайная перестановка отдельных tiles | 0.12147 |
| случайная перестановка идеальных 2x2 блоков | 0.15640 |
| случайная перестановка идеальных 4x4 блоков | 0.18982 |
| случайная перестановка идеальных 6x6 блоков | 0.22246 |
| циклический сдвиг всех 4x4 блоков вправо | 0.21599 |
| один swap двух 4x4 блоков, остальное идеально | 0.95099 |

Это объясняет, почему визуально правдоподобные цельные блоки в неверных
абсолютных позициях могут давать score около текущего LB. Coarse-to-fine полезен
только вместе с сильным absolute component placement.

Ограничение evidence: отдельный frozen report с seed, 16 source IDs и
name-list hash для этой диагностики не был сохранён. Поэтому таблица является
интерпретирующей ad-hoc диагностикой, не authoritative promotion result; в
machine-readable итоговую таблицу model decisions она не включается.

## 8. Staged masked-gap experiment

Последняя bounded hypothesis скрывала центральные 4 колонки канонической пары
тайлов, восстанавливала clean gap из masked raw + masked denoised views, а затем
сравнивала два одинаково инициализированных listwise ranker arms:

- candidate: predicted gap;
- direct control: zero/direct gap.

Обучение и dense scoring фиксированы заранее; candidate проверяется против
direct control и production w4 на `primary_kornia` и `independent_libjpeg`.
Calibration B и holdout были разделены на физически изолированные Phase A/Phase
B; secret permutation seeds хранились только в label-only archive. QAP не должен
был запускаться до прохождения retrieval/reconstruction gate.

До scientific run был выполнен engineering-only synthetic capacity benchmark:

- report: `runs/assembly_v1/kaggle/masked_gap_ddp_benchmark_output/v1/masked_gap_t4_ddp_selection_v2.json`;
- selected config: `w32_g3_r3`;
- projected runtime with safety factor: `3.497697 h`;
- peak reserved: `1,031,798,784` bytes/GPU;
- synthetic-only tensors, discarded weights, no target access и не scientific
  model training.

Stage 1 завершился на двух Tesla T4 `sm_75` и сохранил синхронизированный
checkpoint. Основные evidence:

- `runs/assembly_v1/kaggle/masked_gap_stage1_output/v4_required_readback/masked_gap_stage1_report.json`;
- `runs/assembly_v1/kaggle/masked_gap_stage1_output/v4_required_readback/training/training_report.json`;
- checkpoint SHA-256
  `79447dce4c5943abceb1ec166685a6724fb0b7c10446d20d4a1b11be74afdf48`;
- Stage-1 training/calibration-A selected blend MRR `0.2751373536`.

Первые три Phase-A попытки завершились fail-closed до scoring: новый Kaggle
mount layout, автоматическая распаковка ZIP и неприсоединённый временный
archive dataset. После перехода на уже распакованный input-only dataset с
hash-pinned manifest Phase A version 4 успешно посчитал dense scores на `2xT4`;
labels и puzzle dataset при этом не монтировались. Затем изолированный Phase B
version 1 открыл frozen label manifest и вынес terminal decision.

Calibration-B результат:

- generator reconstruction хуже copy и interpolation на обеих panels;
- candidate против equal-rank direct control: MRR delta `-0.00121115`;
- candidate blend против production `w4`: MRR `-0.0442041`, R1 `-0.0166440`,
  R5 `-0.0657835`;
- panel-wise retrieval deltas отрицательны на обеих panels;
- source-mean MRR wins: `0/4`;
- все шесть frozen conditions равны `false`;
- `passed=false`, `final_holdout=false`, `qap_run=false`.

Authoritative evidence:

- `runs/assembly_v1/kaggle/masked_gap_phasea_output/v4_required/masked_gap_phasea_stage_report.json`;
- `runs/assembly_v1/kaggle/masked_gap_phasea_output/v4_required/phase_a/phase_a_report.json`;
- `runs/assembly_v1/kaggle/masked_gap_phaseb_output/v1/phase_b_report.json`;
- Phase-B report SHA-256
  `0795759fd6eec4ff6757b7827257b905dbff7ee92506a9155284c4add9113d25`.

Ветка имеет итоговый статус **FAILED GATE / NOT PROMOTED**. По заранее
зафиксированному protocol holdout и QAP не запускались; production submission
остаётся без изменений.

## 9. TileNAF latent edge embedding

Идея «уйти от RGB-границы в пространство, которое нейросеть придумает сама»
была реализована как компактный directional embedding model. Frozen TileNAF
возвращает своё последнее `48x20x20` decoder representation до RGB-head. К нему
добавляются raw RGB, restored RGB, residual и Sobel features. Два
последовательных Transformer layers сохраняют все 20 позиций вдоль стороны и
выдают четыре пары нормированных query/key embeddings. Всего обучается
`1,682,785` параметров; TileNAF и HBT остаются frozen.

Главный objective был выровнен с реальным inference: listwise cross-entropy
внутри frozen HBT top-64, плюс слабый all-575 hard-negative auxiliary loss.
Каждый train source видел `primary_kornia` и `independent_libjpeg` corruption;
обучение шло два эпохи на `edge_train[4096:4352]`, строго по whole sources, на
двух T4 в fp32. Epoch-2 full-ranking R1/MRR достигли `0.158667/0.254240` против
`0.046044/0.097606` после первой эпохи, то есть модель действительно научилась
не тривиальному edge space.

На selection `assembly_incremental_gate[192:208]` лучший безопасный blend
`alpha=0.1` улучшал production W4, но проигрывал более сильному HBT comparator:

| Panel | Candidate R1 | HBT R1 | delta | Candidate MRR | HBT MRR | delta |
|---|---:|---:|---:|---:|---:|---:|
| primary_kornia | `0.202842` | `0.212749` | `-0.009907` | `0.301603` | `0.308261` | `-0.006658` |
| independent_libjpeg | `0.201936` | `0.212749` | `-0.010813` | `0.300132` | `0.307528` | `-0.007396` |

Поэтому формальный strongest-baseline gate завершился
`stop_selection_retrieval`. Поскольку signal против production W4 был
устойчивым, `alpha=0.1` заморозили и один раз проверили на отдельном untouched
`assembly_incremental_gate[208:224]`, без retune:

| Panel | R1 delta | MRR delta | R5 delta | R32 delta | R1 bootstrap lower | wins |
|---|---:|---:|---:|---:|---:|---:|
| primary_kornia | `+0.007756` | `+0.010650` | `+0.012908` | `+0.013757` | `+0.004472` | `13/16` |
| independent_libjpeg | `+0.007926` | `+0.012171` | `+0.017437` | `+0.016304` | `+0.004189` | `13/16` |

Результат положительный и статистически согласованный, но заранее заданные
условия требовали R1 delta не меньше `+0.008` и coverage не меньше `0.75` на
каждой panel. Наблюдались R1 `+0.007756/+0.007926` и coverage
`0.744452/0.744735`. Gate есть gate: итоговый статус
**FAILED GATE / NOT PROMOTED**, QAP/SSIM не запускались, production submission
не менялся.

Authoritative hashes:

- pilot report:
  `16279686ed179a5a0f9ceb8764ecc027500560f9d11962e4dca1727e8e14b05f`;
- latent checkpoint:
  `01a63b08e4f1dabb6e475b690ea6f4b57aa19f87e24d25c3d0f641972e634e9f`;
- production-anchor holdout report:
  `87321f9e75468a39e1198b62a91979dde70661694df147f2bb7174a93afc40e1`;
- untouched holdout source-name SHA-256:
  `1ea60e90616267e186cb0ccbcd09b33a9f1d994ed006a2aebdde22d6b393e974`.

## 10. Исследовательские источники и что из них было использовано

Ниже перечислены не просто найденные ссылки, а источники, которые повлияли на
конкретные проверенные ветки. Полный ранний research handoff с заметками
сохранён в `ASSEMBLY_RESEARCH.md` и `TILE_ASSEMBLY_HANDOFF.md`.

### 10.1. Restoration

| Источник | Перенесённая идея | Что произошло в этой задаче |
|---|---|---|
| DnCNN — <https://arxiv.org/abs/1608.03981> | residual image denoising | использован как историческая архитектурная отправная точка, но не как финальная модель |
| MPRNet — <https://arxiv.org/abs/2102.02808> | multi-stage restoration | рассмотрен, но для `20x20` tiles и лимита GPU оказался тяжелее необходимого |
| Restormer — <https://arxiv.org/abs/2111.09881> | transformer restoration | рассмотрен как крупная альтернатива; не получил приоритета перед точным TileNAF supervision |
| SwinIR — <https://arxiv.org/abs/2108.10257> | windowed restoration Transformer | рассмотрен для JPEG/noise restoration; не запускался как promoted route |
| FBCNN — <https://arxiv.org/abs/2109.14573> | blind JPEG artifact removal | повлиял на моделирование неизвестного JPEG quality, но отдельный FBCNN checkpoint не продвигался |
| NAFNet — <https://arxiv.org/abs/2204.04676> | простые nonlinear-activation-free restoration blocks | стал основой компактного TileNAF, который прошёл sealed gate и вошёл в production |

### 10.2. Classical и learned jigsaw compatibility

| Источник | Перенесённая идея | Проверенная ветка/вывод |
|---|---|---|
| Pomeranz et al., greedy square puzzle solver — <https://www.cs.bgu.ac.il/~ben-shahar/Publications/2011-Pomeranz_Shemesh_and_Ben_Shahar-A_Fully_Automated_Greedy_Square_Jigsaw_Puzzle_Solver.pdf> | directional compatibility, best-buddies, component growth | classical components и reciprocal-edge diagnostics; на noisy tiny tiles локальная точность недостаточна |
| Gallagher, MGC — <https://chenlab.ece.cornell.edu/people/Andy/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf> | Mahalanobis Gradient Compatibility | MGC/PBC/C1 baselines; clean puzzle почти решается, corrupted compatibility остаётся bottleneck |
| Paikin and Tal — <https://openaccess.thecvf.com/content_cvpr_2015/papers/Paikin_Solving_Multiple_Square_2015_CVPR_paper.pdf> | быстрый global growth при missing/mixed pieces | повлиял на component/loop solvers; не победил QAP energy |
| Son et al., Growing Consensus — <https://openaccess.thecvf.com/content_cvpr_2016/papers/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.pdf> | geometric consensus для очень маленьких pieces | loops, order-2 и growing-consensus attempts; слабые edges не дали устойчивого роста |
| Sholomon et al., genetic solver — <https://openaccess.thecvf.com/content_cvpr_2013/papers/Sholomon_A_Genetic_Algorithm-Based_2013_CVPR_paper.pdf> | segment-preserving global search | мотивировал population/global search; search по той же слабой energy был закрыт |
| Yu et al., linear programming — <https://arxiv.org/abs/1511.04472> | global relaxation после candidate pruning | LP/assignment-family reasoning; candidate recall оказался приемлемым, precision — нет |
| DNN-Buddies — <https://arxiv.org/abs/1711.08762> | learned narrow-boundary verifier | L0 seam CNN и binary verifier family; standalone gain не подтверждён |
| JigsawNet — <https://arxiv.org/abs/1809.04137> | learned compatibility + loop closure | pair scoring и cycle features; cycle additions дали слишком малый либо отрицательный сигнал |
| TEN — <https://arxiv.org/abs/2203.06488> | encode each side once, cheap all-pairs scoring | L1/HBT embedding family |
| Edge2Vec — <https://arxiv.org/abs/2211.07771> | hard-negative boundary embeddings | HBT RGB+Sobel checkpoint; полезный production feature, но не самостоятельный solver |

### 10.3. Global neural and generative assembly

| Источник | Перенесённая идея | Проверенная ветка/вывод |
|---|---|---|
| Gumbel-Sinkhorn — <https://arxiv.org/abs/1802.08665> | differentiable permutation relaxation | ViT-Sinkhorn absolute assignment; decisive negative transfer на 576 positions |
| Heck et al., vision-transformer jigsaw solver — <https://link.springer.com/article/10.1007/s10044-025-01484-z> | large ViT plus edge CNN and Sinkhorn/Hungarian | послужил верхней архитектурной точкой; доступный pilot был намного меньше и не дал signal |
| JPDVT — <https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Solving_Masked_Jigsaw_Puzzles_with_Diffusion_Vision_Transformers_CVPR_2024_paper.pdf> | positional diffusion for shuffled pieces | positional diffusion pilot; SSIM и adjacency сильно ухудшились |
| DiffAssemble — <https://openaccess.thecvf.com/content/CVPR2024/html/Scarpellini_DiffAssemble_A_Unified_Graph-Diffusion_Model_for_2D_and_3D_Reassembly_CVPR_2024_paper.html> | graph diffusion for reassembly | исследован как общий prior, но масштаб и supervision не совпали с текущим `24x24` regime |
| Masked Autoencoders — <https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html> | frozen masked-image plausibility energy | MAE energy/search: broad candidates различались, competitive candidates почти не ранжировались |
| Positional diffusion reference implementation — <https://github.com/IIT-PAVIS/Positional_Diffusion> | practical diffusion parameterization | использован при реализации pilot; branch retired по честному two-panel gate |
| Multi-phase RL square-puzzle solver — <https://github.com/BenVr/multi-phase-rl-for-square-puzzles> | staged reinforcement/global search | RL top-k 4/8/16; все варианты проиграли QAP на fixed validation |

Ни одна внешняя работа не демонстрировала готовый переносимый рецепт именно для
`576` independently corrupted `20x20` tiles при этом бюджете. Поэтому literature
использовалась для формулировки bounded hypotheses, а promotion всегда решала
только зафиксированная validation этой задачи.

## 11. Reproducibility и итоговый архив

### 11.1. Граница текущего delivery

Этот delivery закрывает ML research history, canonical code/config/tests,
promoted weights и пользовательский scored `submission.zip`. Он не выдаётся за
полный пакет для экспертной защиты конкурса: отдельный `solution.ipynb`, чистый
Google Colab replay и presentation в ходе этой работы не создавались и не
аудировались. Поэтому:

- `safe_for_submission_format` относится только к 700 PNG archive;
- `expert_submission_package_ready = false`;
- byte-identical end-to-end Colab replay финального archive не заявляется;
- обучение, seeds, split IDs и runtime evidence сохранены в canonical scripts,
  configs и decision reports, но не объединены в один Colab notebook.

Это явное ограничение важнее формальной галочки: исходные требования конкурса
нельзя считать выполненными без реального clean replay.

### 11.2. Frozen production record

- layout solver: QAP `w4`, boundary `0.05`, `25x2`, seed/config evidence в
  `qap_tuning_night_output/v2`;
- selected denoiser SHA-256:
  `77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734`;
- HBT SHA-256:
  `c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787`;
- promoted renderer-only seam checkpoint SHA-256:
  `f973c7e606a112020c527bb72277b82586df915edc829a22305e587b35aec1b9`;
- canonical luma archive SHA-256:
  `099d1c5fe69cda8519a4f19750cb3a481ac87999c294a35e19691a849d4c6096`;
- canonical archive validation: 700 unique root members, names
  `img_[0-9]{6}.png`, all PNG/RGB/`480x480`, CRC and full decode pass;
- decoded pixel-stream SHA-256:
  `b8fd646bc3cc2071853988cc36d13108c30c5aa7efc0097f6dc4ef91cbc4cc98`;
- principal seed: `20260710`; model/data/split/quarantine hashes сохранены в
  luma confirmation, TileNAF release audit и manifest;
- validation source IDs сохранены целиком внутри reports; confirmation RGB
  source-name-list SHA-256:
  `91d8ea6ea78b65eac69e863ce3ad98099d46328d00032f02736ebbe8c3fc9c8f`;
- promoted pipeline не использует filename/target leakage и не открывает test
  targets; candidate oracle помечен `safe_for_submission=false` и применяется
  только как diagnostic ceiling;
- credentials, Kaggle tokens, raw train/test, local env и external caches
  исключены policy и path allowlist.

SHA-256 самого outer delivery archive записывается в detached
`.verification.json` после атомарной упаковки: вложить собственный final hash в
архив без самоссылки невозможно.

### 11.3. Состав архива

Финальный delivery содержит:

- этот отчёт без pending sections;
- полный canonical snapshot `src/`, `scripts/`, `configs/`, `tests/`;
- environment и project rules;
- все decision-bearing JSON/Markdown/log reports;
- компактные существующие denoise/assembly/neural-upgrade bundles;
- текущий лучший luma `submission.zip`;
- promoted checkpoint assets;
- masked-gap artifacts независимо от pass/fail;
- TileNAF latent-edge code, Kaggle job, reports and terminal holdout evidence;
- machine-readable manifest с SHA-256 и byte size;
- hash-only references для больших duplicate NPZ/checkpoint/candidate dumps;
- archive member/CRC/decode/path verification report.

Raw `puzzle/train`, `puzzle/test`, локальная `.conda`, внешние model caches,
`__pycache__` и дубли Kaggle-mounted code в архив не включаются.
