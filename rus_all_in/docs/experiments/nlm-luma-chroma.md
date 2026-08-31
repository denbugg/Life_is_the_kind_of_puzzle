# Раздельная сила colored NLM для яркости и цвета

Статус: **primary FAIL; confirmation не открывалась**.

## Гипотеза

Все найденные прежние вызовы `cv2.fastNlMeansDenoisingColored` использовали
одинаковую силу для яркостного и цветового каналов. Было заранее проверено, не
позволит ли более сильное подавление chroma-noise оставить яркостную геометрию
почти на уровне production `h=20`:

```text
bilateral buddies96 strict layout
  -> RGB seam offsets -> bounded luminance gains
  -> один colored-NLM pass с независимо заданными h и hColor
```

Нет multi-pass, `h/hColor >= 30`, генеративной модели, reference pixels,
cross-board pixels, warp, rotation или tile substitution. Raw assembly до
restoration проверяется как точная перестановка всех 576 upright input tiles.

## Протокол

- immutable preregistration:
  `configs/nlm_luma_chroma_reused_calibration_preregistered_v1.json`;
- config SHA-256:
  `38151503c12a39b3f5be7f2a19ad7d939796d33dc43416609810da47ef901108`;
- manifest SHA-256:
  `4781e370e092ad272c63e6d5165b25951aaf93fae5fde74c75d534a9e8efc9da`;
- shared ranked selector, calibration `468:504`, 36 boards;
- primary filename digest:
  `0fe1f7beb15995e98461f863856f74f1c96837732dfb935772c87643288b5000`;
- planned confirmation: `504:540`, 36 disjoint boards;
- historical freshness не заявляется: legacy calibration-700 уже открывал все
  filenames обеих панелей;
- все восемь predictions, layouts, audits, target-free structure metrics и их
  hashes для всех 36 boards записаны в commitment до первого target decode.

Заранее заданы control `(h=20,hColor=20)` и семь candidates:
`(20,24)`, `(20,28)`, `(22,20)`, `(22,24)`, `(22,28)`, `(24,20)`, `(24,28)`.
Gate требовал mean SSIM `>=0.27`, положительный lower bound парного 95% t-CI,
не меньше 24/36 wins и сохранение яркостных, цветовых и Laplacian деталей.

## Результат

| h | hColor | Mean SSIM | Gain vs 20/20 | CI95 lower | Wins | Решение |
|---:|---:|---:|---:|---:|---:|---|
| 20 | 20 | 0.247069 | — | — | — | control |
| 20 | 24 | 0.246967 | -0.000102 | -0.000142 | 3/36 | reject |
| 20 | 28 | 0.246869 | -0.000200 | -0.000269 | 2/36 | reject |
| 22 | 20 | 0.250401 | +0.003332 | +0.003099 | 36/36 | absolute fail |
| 22 | 24 | 0.250311 | +0.003242 | +0.003008 | 36/36 | absolute fail |
| 22 | 28 | 0.250218 | +0.003149 | +0.002908 | 36/36 | absolute fail |
| 24 | 20 | **0.253257** | **+0.006188** | **+0.005750** | **36/36** | absolute fail |
| 24 | 28 | 0.253092 | +0.006023 | +0.005582 | 36/36 | absolute fail |

При фиксированном `h` увеличение только `hColor` слабо, но последовательно
ухудшало SSIM. Весь положительный эффект обеспечивало обычное увеличение
яркостной силы `h`, уже независимо измеренное dense-tail sweep-ом. Ни один arm
не достиг `0.27`, поэтому selection winner отсутствует и confirmation
`504:540` fail-closed не открывалась.

## Артефакты

- commitment:
  `outputs/nlm-luma-chroma/primary-calibration-offset468-count36/prediction-commitment.json`,
  file SHA-256
  `fdf27a96212c15909bb3dc5e195ca20e9adb86b4f35f7c5c98122e1e3a3921d6`;
- report:
  `outputs/nlm-luma-chroma/primary-calibration-offset468-count36/report.json`,
  SHA-256
  `2a80faffffedb313141d14abc901b2d4f27c9b08d9466a6acea6a55fccac6153`;
- contact sheet:
  `outputs/nlm-luma-chroma/primary-calibration-offset468-count36/contact-sheet.png`,
  SHA-256
  `34dfa822dbb4b75671eb8374ede3b89577e5848e378120daeae4f665ebd6aaa1`;
- implementation: `src/aiijc_puzzle/nlm_luma_chroma.py` and
  `scripts/run_nlm_luma_chroma.py`;
- tests: `tests/test_nlm_luma_chroma.py` (`4 passed`), relevant Ruff check PASS.

Вердикт: **не повторять decoupled hColor sweep**. Если нужен допустимый более
сильный single-pass tail, использовать общий manual-safe ceiling `h=28`, а не
поднимать только chroma strength.
