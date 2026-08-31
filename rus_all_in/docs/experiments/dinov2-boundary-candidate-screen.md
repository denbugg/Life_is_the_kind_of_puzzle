# DINOv2 boundary candidate screen

Дата: 2026-08-31. Статус: **сильный complementary supply, direct gate fail;
сохранить только как emitter для нового verifier-а**.

## Что проверено

Официальный frozen DINOv2 ViT-S/14 использован только как matcher-view. Каждый
исходный dirty `20×20` tile bicubic-resize-ится до `98×98`; из `7×7` patch
tokens берутся две крайние полосы. Для right/down score усредняется cosine
соответствующих ordered tokens на противоположных сторонах. Модель не получает
position id, source id или target, не выбирает центр/лицо и не меняет output
pixels. Top-32 всегда содержит ids исходных upright tiles.

Это воспроизводит доказанно открытый кусок исторического P29: DINO как
candidate generator, а не отвергнутые DINO absolute heads или fixed score
fusion. Fixed contract был подписан до replay:
`configs/dinov2_boundary_candidate_screen_v1.json`, SHA-256
`77751ed387a5bd34da6202b23c250fb0e36e772b0ad0050ff290d6138a38c494`.
Support/size/band/top-k sweep не проводился.

Панель — уже открытый source-disjoint local16 из full-resolution adapter
protocol. Raw d64 candidates воспроизведены точно из frozen archive. DINO
candidates и metadata были записаны до exact scoring.

## Local16 retrieval

| matcher | pooled R@1 | R@5 | R@32 | mutual precision | mutual coverage |
|---|---:|---:|---:|---:|---:|
| raw d64 | 19.565% | 38.887% | 69.724% | 31.975% | 48.353% |
| DINOv2 boundary | 5.135% | 14.176% | 37.953% | 11.346% | 20.907% |

DINO не является replacement scorer. Но его ошибки сильно отличаются от raw:

| top-32 supply | exact-neighbour coverage | gain vs raw |
|---|---:|---:|
| raw | 69.724% | — |
| raw ∪ DINO | **75.385%** | **+5.661 pp** |

Пререгистрированный strong gate требовал одновременно union gain `>=+2 pp` и
либо nonnegative direct R@5 delta, либо mutual precision `>=25%`. Supply часть
пройдена с большим запасом, direct/precision часть провалена. Поэтому простой
DINO replacement, fixed blend или немедленный hard-edge consumer не строился.

## Тройной opened-panel diagnostic

После завершения screen те же frozen identities были механически объединены с
raw и adapter-step400 archive на том же уже открытом local16:

- raw ∪ adapter400: `72.611%` (`+2.887 pp`);
- raw ∪ DINO: `75.385%` (`+5.661 pp`);
- raw ∪ adapter400 ∪ DINO: `77.400%` (`+7.677 pp`);
- **raw ∪ adapter1600 ∪ DINO: `78.029%` (`+8.305 pp`)**;
- DINO сохраняет `+4.461 pp` unique coverage поверх raw+adapter1600;
- adapter1600 сохраняет `+2.644 pp` unique coverage поверх raw+DINO.

Это не promotion metric и не новый scored arm; diagnostic использует уже
opened local16. Он показывает, что pre-solver adapter и foundation descriptor
действительно комплементарны. Следующий materially different consumer должен
векторизованно обучаться на exact synthetic candidate union и выдавать
calibrated relation probabilities. Нельзя повторять P29 fixed logistic/rank
fusion или подавать DINO top-32 напрямую в rigid solver.

## Артефакты и проверки

- report:
  `outputs/dinov2-boundary-candidate-screen/opened-local16-v1/report.json`,
  SHA-256 `6e5d04814775ee5eff652d30b3fb33d73bdd6ddc9a0332760a3f5b3cff2c1b71`;
- frozen candidates:
  `frozen-target-free-candidates.npz`, SHA-256
  `48f4c3991a8e9f050d16bdb5e0d99f796bdebddc47935bfb80e6f04817c5a03a`;
- module: `src/aiijc_puzzle/dinov2_boundary_matcher.py`;
- runner: `scripts/run_dinov2_boundary_candidate_screen.py`;
- tests: `tests/test_dinov2_boundary_matcher.py` (`3 passed`), Ruff passed;
- per-board CPU runtime: about `1.04 s`, 16 boards about `20 s` total;
- competition test, production, submission and pixels untouched.

Weco Observe retrieval-only step `144`, parent pair/exact solver step `102`.
