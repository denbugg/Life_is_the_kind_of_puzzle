# Manual compliance audit: iterative NLM h=10

> **SUPERSEDED FOR SUBMISSION SELECTION.** This audit established target
> independence and absence of a literal constant/template frame, but its
> `20-pass` recommendation was too permissive for the organizers' image-quality
> review. The cross-strength audit
> [nlm-strength-manual-safety.md](nlm-strength-manual-safety.md) freezes one
> colored NLM `h=20` pass, with `h=15 x1` fallback, and rejects every multi-pass
> tail and every `h>=30` setting from final submission.

## Binding and leakage boundary

Audit привязан к authoritative report
`outputs/restoration-r6/compliant-iterative-nlm-fresh-calibration24-passes20.json`,
layout `bilateral_buddies96_atlas_w0p03`, calibration offset 48, 24 boards.
Holdout/test не открывались.

Из source report whitelist-ятся только filename, input hash, frozen
`tile_at_position` и layout hash. Для всех 24 boards сначала из dirty input
строятся strict raw assembly и iterative NLM 1/5/10/15/20, фиксируются hashes и
diagnostics. Targets загружаются только после этого и используются только в
side-by-side contact sheets.

Raw permutation audit проходит `24/24`: 576 уникальных tile indices, без
missing/duplicate, exact reassembly, input/output tile multiset identical. NLM
запускается только после этого; он не меняет layout, не выполняет warp и не
подставляет constant/template pixels.

Representative set выбран без targets: шесть равномерных квантилей
input-gradient energy. Просмотрены full-board и deterministic center crops для
raw, 1x, 5x, 10x, 15x, 20x и target.

## Quantitative preservation and collapse diagnostics

| passes | official SSIM | phase max, px | global std min | tile-mean std min | range min | entropy min, bit | tile-mean corr mean | descriptor top1 mean | near-constant tiles mean / max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.272445 | 0.040 | 0.873 | 0.943 | 0.889 | 6.506 | 0.991 | 0.352 | 3.07% / 19.44% |
| 10 | 0.289291 | 0.095 | 0.838 | 0.916 | 0.864 | 6.410 | 0.981 | 0.230 | 4.34% / 28.82% |
| 15 | 0.298125 | 0.109 | 0.821 | 0.901 | 0.844 | 6.335 | 0.971 | 0.173 | 5.22% / 32.12% |
| 20 | 0.303459 | 0.120 | 0.811 | 0.885 | 0.824 | 6.302 | 0.962 | 0.144 | 5.92% / 33.68% |

Здесь variance/range указаны как ratio к strict raw assembly. На всех четырёх
pass counts все `24/24` boards проходят core non-collapse checks: phase shift
≤0.25 px, global std ratio ≥0.5, tile-mean std ratio ≥0.8, dynamic-range ratio
≥0.7 и entropy ≥4.5 bit.

Conservative raw-fidelity stress thresholds намеренно перестают проходить:
raw-output SSIM mean падает `0.473→0.412→0.380→0.359`, а same-position
high-frequency tile descriptor top1 `0.352→0.230→0.173→0.144` для
5/10/15/20. Это реальное стирание локальной текстуры, а не spatial motion:
phase max остаётся 0.120 px, tile-mean correlation на 20x всё ещё 0.962 mean,
и межтайловая variance сохраняется.

Один тёмный board `img_006499.png` превышает диагностический порог 25%
near-constant tiles начиная с 10x (`28.82/32.12/33.68%`). Он специально попал
в contact sheet как low-gradient tail. Визуально flattening следует исходным
тёмным областям и не распространяется на весь canvas; global std ratio на 20x
равен 1.001, entropy 6.302 bit. Поэтому это warning, но не constant-frame
collapse.

## Manual inspection

Во всех `6/6` representative boards на 5/10/15/20:

- нет сдвига, resize, crop, spatial warp или перестановки после raw assembly;
- сохраняются board-specific palette, крупные формы и tile-grid geometry;
- не появляется target-derived или template-like новая детализация;
- нет whole-frame constant/blank collapse.

5x уже превращает шумную текстуру в piecewise-smooth regions. 10x усиливает
этот эффект. 15x и 20x дополнительно округляют границы и удаляют мелкие детали,
но изменение 15→20 существенно меньше ранних 1→5 passes; изображения остаются
различимыми между boards и spatially tied к raw mosaic. Target column наглядно
показывает, что NLM не «восстанавливает» clean scene и не подменяет content.

## Historical recommendation (superseded)

The historical audit proposed **20 passes** as its non-collapse cap. Это был
максимальный фактически проверенный count: strict permutation `24/24`, core
numerical non-collapse `24/24`, visual non-collapse `6/6`.

Рекомендация не основана только на SSIM: 20x разрешён потому, что независимые
geometry/variance/range/entropy checks и ручной просмотр не находят collapse.
При этом 15/20 следует пометить как aggressive restoration: local identity
сильно снижается, а один dark-tail board имеет треть near-constant tiles. Если
правило/жюри трактует чрезмерное сглаживание консервативнее, fallback cap — 10;
5 — наиболее консервативный visual-preservation вариант. По текущей формальной
границе «strict bijective layout, затем restoration» оснований отклонять 20x
нет. Этот вывод больше нельзя использовать для submission: новый cross-strength
visual gate показал progressive fragment-quality collapse задолго до literal
constant-frame collapse.

Artifacts:

- `outputs/manual-compliance/iterative-nlm-h10-cal24-passes20/report.json`;
- `outputs/manual-compliance/iterative-nlm-h10-cal24-passes20/contact-sheet-full.png`;
- `outputs/manual-compliance/iterative-nlm-h10-cal24-passes20/contact-sheet-center-zoom.png`.
