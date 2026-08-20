# Puzzle assembly autoresearch report

Дата фиксации: 2026-08-20. Этот документ — единый индекс серии экспериментов
по ускорению и улучшению сборщика паззлов. Он отделяет проверенные локальные
результаты от exploratory-результатов и от фактической Kaggle-проверки.

## Короткий итог

- Лучший проверенный **layout solver**: E14 — fusion learned directional scores
  с raw MGC+SSD (`alpha=0.2`) и sparse multi-phase relaxation/Hungarian.
- Лучший проверенный **pixel output**: E18b — неизменный layout E14, full-image
  OpenCV colored NLM (`h=9`, `hColor=9`, windows `7/21`) и детерминированный
  возврат только вновь появившихся серых 20x20-ячеек к raw-пикселям.
- E14 на frozen full-128 улучшает robust SSIM на `+0.00112291`, mean SSIM на
  `+0.00120702`, adjacency на `+0.01713230` и работает в `3.43x` быстрее
  matched simulated-annealing baseline.
- E18b поверх E14 улучшает robust SSIM ещё на `+0.06522740`, mean SSIM на
  `+0.06781749`, выигрывает `128/128` изображений и не меняет layout/adjacency.
- Kaggle v2 подтвердил корректность self-contained packaging и no-gray path,
  но не завершился: validation E18b/E14 оказался хуже v5 на `-0.006963`, а
  test inference занял около `14 s/image`; run остановлен лимитом после
  `189/700`. Submission-файл не сформирован.

## Протокол проверки

- Frozen cache: `outputs/directional_student_holdout128.npz`.
- SHA-256: `74db2b62e9d5eafffae33117c7771512d823b0dcaa0095ef5807adb8e86a25df`.
- Содержимое: 128 grouped real-noisy cases, по 576 тайлов `20x20`, сетка
  `24x24`; target/truth доступны только evaluator после выбора layout.
- Tune/smoke: индексы `0..31`; untouched verification: `32..127`.
- Для seed-чувствительных методов использовался второй seed offset `1000003`.
- Promotion требовал валидной перестановки, конечных метрик, чистого лога и
  одновременной проверки SSIM/adjacency; proxy objective не считался победой.
- Метрики ниже — offline frozen-cache, а не официальный hidden leaderboard.

## Главные проверенные результаты

### E14: fusion + global relaxation

| metric | matched baseline | E14 | delta |
|---|---:|---:|---:|
| robust SSIM | 0.1003414429 | **0.1014643490** | **+0.0011229061** |
| mean SSIM | 0.1027065484 | **0.1039135704** | **+0.0012070220** |
| adjacency | 0.0855129076 | **0.1026452106** | **+0.0171323030** |
| runtime, full-128 | 428.7263 s | **125.0452 s** | **3.43x faster** |

Дополнительные gates: `67/128` SSIM wins, `116/128` adjacency wins,
`128/128` valid layouts. Untouched-96 deltas:
`+0.0006311362 / +0.0006457597 / +0.0165591033`
(robust/mean/adjacency). Два predeclared smoke-16 seeds имели одинаковый
положительный знак по всем трём метрикам.

### E18b: guarded full-image NLM

| metric | raw E14 | unguarded E18 | guarded E18b |
|---|---:|---:|---:|
| robust SSIM | 0.1014643490 | 0.1703372742 | **0.1666917489** |
| mean SSIM | 0.1039135704 | 0.1753758864 | **0.1717310628** |
| adjacency | 0.1026452106 | 0.1026452106 | **0.1026452106** |
| gray-cell total | 17,996 | 19,644 | **16,776** |
| images with gray excess | — | 97/128 | **0/128** |

E18b: `128/128` SSIM wins, `128/128` byte-identical layouts, no adjacency
change. Guard reverted 2,868 newly-gray cells и сохранил 94.90% mean и 94.71%
robust прироста unguarded NLM. End-to-end `142.2222 s`: E14 layout
`126.9480 s`, NLM `14.6209 s`, guard `0.6533 s`.

## Полный experiment ledger

| id | изменение | результат | статус | ветка / evidence |
|---|---|---|---|---|
| E0 | directional student baseline | full-128 robust `0.09981750`, mean `0.10218954`, adj `0.08694237` | baseline | `main` |
| E1 | reciprocal-margin bonus, beta `.5`, threshold `.5` | seed0 robust `+0.000755`, но alt robust `-0.002894`, mean `-0.002614`, adj `-0.001104` | DROP: seed-unstable | `autoresearch/e1-margin`, `c2c4f96` |
| E2 | raw MGC+SSD fusion, alpha `.2` | seed0 robust/mean `+0.005158/+0.005467`; alt SSIM остаётся положительным, но adj `-0.001330`, runtime `1.11975x` | DROP standalone | `autoresearch/e2-score-fusion`, `63c1456` |
| E3 | compiled exact SA hot loop, прежний NumPy RNG stream | `32/32` exact layouts/SSIM/adj, objective delta `0`, `3.404x` speedup | PASS efficiency | `autoresearch/e3-cache-multistart`, `72a9c3b` |
| E4 | reciprocal cycle-safe component initializer | robust `-0.001273`, mean `-0.001150`, adj `+0.017946`, runtime `1.0098x` | DROP: topology up, SSIM down | `autoresearch/e4-bestbuddy`, `44a874a` |
| E5 | capped hard-negative fine-tune | не запускался | QUEUED | plan only |
| E6 | clean/corrupt consistency | не запускался | QUEUED | plan only |
| E7 | asymmetric directional heads | не запускался | QUEUED | plan only |
| E8 | one-epoch low-LR continuation (`5e-6`) | restored macro-F1 `0.503714→0.500355`, adjacency `0.77126→0.69198`; epoch0 остался best | DROP | evidence in shared history/current branch |
| E9 | equal-wall-clock multistart | остановлен на `3/32` при structural pivot; метрика не заявлена | STOPPED | no claimed result |
| E10 | guarded source-aware + NLM | заменён более чистыми E15/E18 ablations | SUPERSEDED | plan only |
| E11 | sparse relaxation labeling + Hungarian | seed0 robust/mean/adj `+0.001144/+0.001580/+0.013644`, `5.33x` faster; alt robust/mean `-0.001437/-0.001038`, adj `+0.007756` | DROP standalone | `autoresearch/e11-relaxation`, `4d67749` |
| E12 | sparse weighted CP-SAT LNS | robust/mean `-0.000289/-0.000277`, adj `+0.000113`; sparse proxy `+82.114`, dense objective `-692.080` | DROP: proxy mismatch | `autoresearch/e12-cpsat`, `581c8f7` |
| E13 | corruption-aware border CNN + Sinkhorn/Hungarian diagnostics | local shape/loss/backward/corruption/assignment tests passed; remote training metric отсутствует | DESIGN/BLOCKED | `autoresearch/e13-border-encoder`, `a605814` |
| E14 | E2 fused scores → E11 relaxation | full-128 robust/mean/adj `+0.001123/+0.001207/+0.017132`, `3.43x` faster | **PASS layout** | `autoresearch/e14-fusion-relaxation`, `2087f8d`; Kaggle port `2fd08f5` |
| E15 | E14 raw graph + guarded-restorer classical multiplex | smoke16 robust `+0.001289`, mean `+0.001769`, adj `+0.006624`, runtime `1.461x`; robust gate требовал `+.002` | DROP/provisional | `autoresearch/e15-no-gray-multiplex`, `77496e4` |
| E16 | source-conditioned position prior | не запускался: leakage/source-disjoint gate не обеспечен | BLOCKED | plan only |
| E17 | dual-view trained hard-negative verifier | не запускался | QUEUED | plan only |
| E18 | full-image NLM after fixed E14 layout | robust `+0.068873`, mean `+0.071462`, `128/128` wins, но gray excess `97/128` | DROP safety | included in E18 branch evidence |
| E18b | E18 + newly-gray raw-cell fallback | robust `+0.065227`, mean `+0.067817`, zero layout change, zero gray-excess images | **PASS pixels** | `autoresearch/e18-nlm-polish`, `0d8a526`; champion branch `autoresearch/fast-score-gen1` |
| E19 | raw + per-tile NLM classical dual view | smoke16 robust `+0.000009`, mean `+0.000158`, adj `-0.000283`, runtime `2.403x` | DROP | `autoresearch/e19-nlm-dual-view`, `0e58675` |
| E20 | restored BorderRanker over top32 union | coverage gain right `+0.050951` PASS, down `+0.046535` FAIL vs `+.05`; layout stage не запускался | DROP/non-promotable | `autoresearch/e20-restored-ranker-verifier`, `a877065` |

## Механистические выводы

1. Raw classical boundary evidence даёт устойчивый SSIM-сигнал, но отдельно
   может ухудшить adjacency; global relaxation даёт сильный topology-сигнал,
   но отдельно seed-нестабилен по SSIM. E14 работает благодаря композиции этих
   двух взаимодополняющих ошибок.
2. Proxy objective нельзя считать достаточным: E12 улучшил sparse CP objective,
   но ухудшил SSIM и dense objective.
3. Best-buddy/component и relaxation методы часто сильно улучшают adjacency,
   но глобальная пиксельная ориентация/позиция остаётся отдельной проблемой.
4. NLM полезен как post-assembly image operation, но не как второй per-tile
   edge-score view: E19 почти не дал SSIM и ухудшил adjacency/runtime.
5. Любой denoiser должен проходить независимый gray-cell guard; большой SSIM
   E18 не отменяет его safety-failure.

## Kaggle v2: фактический remote run

- Private kernel: `phoenix0501/pazzle-e18b-guarded-nlm`.
- Self-contained entrypoint загрузился без import/package ошибок.
- E14 и E18b выполнялись; в логах `fallback_reason=none`, gray guard работал.
- Validation: `validation_mean_solver_ssim=0.180304`.
- Сравнение: `validation_mean_v5_baseline_ssim=0.187267`.
- Delta: `-0.006963`; relation guard отключил продвижение E18b/E14 для test.
- Test: около `13.8–14.2 s/image`, достигнуто `189/700`.
- Terminal status: `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` после лимита около
  3600 секунд; ожидалось ещё примерно 1ч58м на отметке 189/700.
- `kaggle kernels files` не вернул output-файлов; submission не создан.

Это означает, что offline E18b остаётся доказанным frozen-cache pixel winner,
но текущий Kaggle pipeline нельзя считать hidden-test winner. Следующий remote
вариант должен выбирать v5 по validation gate и укладываться примерно в
`<=5 s/image`, устранив двойной дорогой solver path.

## Ветки и воспроизводимость

Основная ветка результата: `autoresearch/fast-score-gen1`.

Отдельные ветки сохраняют код, raw JSON/logs и negative results без смешивания
с production champion: `autoresearch/e1-margin`, `e2-score-fusion`,
`e3-cache-multistart`, `e4-bestbuddy`, `e11-relaxation`, `e12-cpsat`,
`e13-border-encoder`, `e14-fusion-relaxation`, `e14-kaggle-port`,
`e15-no-gray-multiplex`, `e18-nlm-polish`, `e19-nlm-dual-view`,
`e20-restored-ranker-verifier`.

Ключевые raw evidence:

- `autoresearch-runs/e14-fusion-relaxation/results/full128_aggregate.json`
- `autoresearch-runs/e14-fusion-relaxation/results/untouched96_seed0.json`
- `autoresearch-runs/e18-nlm-polish/full128.json`
- `autoresearch-runs/e3-cache-multistart/e3_identity_smoke32.json`
- соответствующие `RESULTS.md`, evaluator scripts и test files в каждой ветке.

Не публикуются в Git: исходные датасеты, frozen cache, model checkpoints,
локальные virtualenv/node_modules и generated submission archives. Они
исключены `.gitignore`; в отчётах сохранены SHA/config/provenance, где это
необходимо для проверки.
