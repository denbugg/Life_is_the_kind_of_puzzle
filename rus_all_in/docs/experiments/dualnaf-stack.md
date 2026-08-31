# Frozen DualNAF alpha=.125 + single-pass h28 stack

## Решение

**Reject as tested; confirmation не открывать, production не менять.**
Единственный заранее назначенный candidate `D` достиг `0.267429`, но не прошёл
absolute gate `0.27` и не доказал преимущество над pure-denoise control `B`:
mean gain составил только `+0.000608`, обе 95% CI пересекли ноль, wins `17/32`
при пороге `20/32`.

Это reused calibration, не fresh calibration и не holdout. Legacy report
`outputs/legacy-upgrade/calibration700-champion/report.json` уже содержит exact
filename/input/target hashes всех 700 текущих calibration records. Primary и
confirmation взаимно не пересекаются, но исторически exposed `32/32` каждая.

## Frozen protocol

Authoritative preregistration:
`configs/dualnaf_stack_reused_calibration_preregistered_v1.json`, SHA-256
`c4cf677227da645021e4d06874a29aae4c22db82d77c1e3e8af3825f20d0405a`.
Она была записана read-only до target decode и фиксировала:

- shared-ranked calibration primary `636:668`, NUL-join digest
  `652f57b31ba5642babdd9d48202d0b08f738b07258fb4a60689103def452197a`;
- gated confirmation `668:700`, digest
  `4dcd776a77c6effe966a276b44ab7de1710014165e46c83e51ac38b29d4bf7eb`;
- no-atlas bilateral buddies96 strict layout на original dirty tiles;
- frozen checkpoint
  `outputs/restoration-r6/compliant-r6-medium-train256-step2000-h10.pt`,
  SHA-256
  `331322460c8af87e5d4760b075726979f0574a23209889c1e95b6b90f2eac1a9`;
- independent per-tile renderer без cross-tile pixels/context;
- same-index blend
  `rint(0.875*original + 0.125*DualNAF)`, один глобальный alpha без routing;
- blend до target-blind RGB seam offsets и bounded luminance gains;
- ровно один proper colored-NLM pass `h20` или `h28`.

Четыре arms были фиксированы без post-hoc выбора:

| Arm | Alpha | NLM | Роль |
|---|---:|---:|---|
| A | 0 | h20 x1 | baseline |
| B | 0 | h28 x1 | denoise control |
| C | 0.125 | h20 x1 | bridge diagnostic |
| D | 0.125 | h28 x1 | единственный promotable candidate |

До текущего target decode сохранены 32 layouts, raw canvases, same-index
renders, pre-harmonizer blends, обе harmonized canvases и все четыре arms. Все
strict 576-tile permutation audits прошли. Commitment SHA-256:
`e4ec4f0eeeb94c6c652a257a48d6af66547d72bfe68db2460a4a760eb1a12f7e`.

## Preregistered gate

Для `D` одновременно требовались:

- mean RGB SSIM `>=0.27`;
- paired t и deterministic 20k-bootstrap CI lower `>0` против `A` и `B`;
- wins не менее `24/32` против A и `20/32` против B;
- dense-h28 safety bounds относительно A:
  mean/min within-tile gradient `>=0.80/0.70`, mean/min Laplacian
  `>=0.72/0.60`, mean/max relative grid ratio `<=1.05/1.12`;
- 32 distinct D outputs;
- explicit manual PASS на panel indices `0,10,21,31`, full canvas и center
  zoom.

Confirmation разрешалась только после прохождения всех numeric и manual gates.

## Primary result

| Arm | Mean RGB SSIM |
|---|---:|
| A: alpha0 + h20 | 0.255451 |
| B: alpha0 + h28 | 0.266820 |
| C: alpha.125 + h20 | 0.256635 |
| **D: alpha.125 + h28** | **0.267429** |

Сравнения D:

| Baseline | Mean gain | Paired t 95% CI | Bootstrap 95% CI | Wins/ties/losses | Gate |
|---|---:|---:|---:|---:|---|
| A | +0.011977 | `[+0.010583,+0.013371]` | `[+0.010751,+0.013361]` | 32/0/0 | PASS |
| B | +0.000608 | `[-0.000147,+0.001364]` | `[-0.000020,+0.001401]` | 17/0/15 | **FAIL** |

Absolute `0.267429<0.27` тоже **FAIL**. Поэтому nominal положительный mean gain
относительно B нельзя считать подтверждённым stacking win.

Safety D прошёл все bounds:

- within-tile gradient retention mean/min `0.86777/0.77225`;
- Laplacian retention mean/min `0.72679/0.63542`;
- relative grid ratio mean/max `0.77233/0.80998`;
- distinct outputs `32/32`.

Manual review также PASS по узкому safety-вопросу: на четырёх preregistered
boards D визуально почти неотличим от B; новых крупных smooth blobs, erasure
text/faces/object edges и ухудшения 20-pixel grid относительно B не видно. Это
не означает manual puzzle acceptance: все reconstructed canvases остаются
очевидными мозаиками и не восстанавливают читаемые сцены.

## Artifacts

- Report:
  `outputs/dualnaf-stack/c4cf677227da645021e4d06874a29aae4c22db82d77c1e3e8af3825f20d0405a/primary/report.json`,
  SHA-256
  `e3d6ab66b624e0223504e427e2a5f4bc1261ad15a904032671c50e35705a7301`.
- Target-open receipt SHA-256:
  `9e699c1d04ffd97f839cc826098601ff4ef8903f06829dbd8585673cb0c94bc8`.
- Manual review:
  `outputs/dualnaf-stack/c4cf677227da645021e4d06874a29aae4c22db82d77c1e3e8af3825f20d0405a/primary/manual-review.json`,
  SHA-256
  `cd3e17e9ae4e69d127fde463bb7425adb54df1a5f6da5386e6f371a97b9fd845`.
- Full sheet SHA-256:
  `2dd049b34471153eb9c3c014d28218b1c5277017c722e551d9eb53f1712f6ced`.
- Center-zoom sheet SHA-256:
  `38fa789632f25fa64af8996123a7eb5a89be8a89585a69c7a730cf4cc42ba75a`.

```bash
uv run python scripts/run_dualnaf_stack.py audit
uv run python scripts/run_dualnaf_stack.py prepare-primary --device mps
uv run python scripts/run_dualnaf_stack.py score-primary
uv run python scripts/run_dualnaf_stack.py review-primary \
  --verdict PASS --reason "<review notes>"
```

Команда `prepare-confirmation` проверена fail-closed: она отказывает из-за
failed primary numeric gate, а confirmation directory отсутствует.
