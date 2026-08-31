# Scale256 joint evidence → frozen six-arm selector portfolio

Дата: 2026-08-31. Статус: **implementation reviewed locally; unsigned,
non-executable, no DEV64 data access**.

## Что именно фиксируется

Этот consumer не строит новый пазл. Он может вернуть только один из шести уже
frozen strict layouts (`raw`, `logistic`, `focal_top5`, `nonlinear`,
`selective_vote500_focal`, `combined_union_focal`). Каждый layout остаётся
перестановкой всех 576 исходных upright tile IDs. Пиксели, ориентация и состав
тайлов не меняются.

В архиве до reference access одновременно сохраняются четыре прозрачных
portfolio members:

1. `incumbent_keep` — точный HGB incumbent, то есть настоящий KEEP/control;
2. `fixed_head_comparator` — прежний fixed-5%-head rule без изменений;
3. `union_dense_dominance` — новый fixed rule A;
4. `source_normalized_dominance` — новый fixed rule B.

Фраза «три candidates» здесь означает три selector candidates над control:
старый comparator и два новых правила. Control хранится четвёртым выходом, а
не теряется или неявно подменяется. Никакой выбор лучшего member по DEV labels
не разрешён этим freeze.

## No-repeat anchor

Старый axiswise fixed-head consumer уже менял `10/32` boards и потерял
`-4.938` satisfied pairs/board. Поэтому он сохранён только как честный
comparator. Новые правила не являются sweep его head fraction, threshold или
tie-break и используют full frozen joint evidence.

Также не повторяется отвергнутый all-edge decoder/HGB-ranked union: ни одно
joint edge не добавляется в layout, и ни один layout не собирается заново.

## Rule A: union coverage + dense two-sided confidence

Для всех 1,104 realised relations каждого arm отдельно считаются по right и
down:

- число отношений, присутствующих в immutable joint union;
- сумма и среднее `learned_joint_confidence` на этих отношениях.

Switch допустим, только если arm относительно incumbent одновременно:

- не уменьшает union coverage ни на right, ни на down;
- не уменьшает mean dense two-sided confidence ни на right, ни на down;
- строго улучшает хотя бы один из этих четырёх показателей.

Неопределённое среднее при нулевом incumbent coverage означает fail-closed
KEEP. Fixed tie-break: total/min-axis coverage delta, total/min-axis confidence
delta, frozen expected-correct score, frozen arm order.

## Rule B: full per-source normalized evidence

Для каждой пары `(axis, source tile)` независимо нормализуются **все** valid
union candidates:

`tanh(((value - row_mean) / population_std) / 2)`.

Константная строка даёт нули. Отдельно нормализуются learned edge logit и dense
two-sided confidence, затем берётся их фиксированное среднее `0.5/0.5`. Если
realised arm edge отсутствует в union, каждый компонент получает
детерминированный source-local floor `minimum_valid_normalized - 1`. Поэтому
каждый из 552 right и 552 down edges вносит вклад; missing edge не исчезает из
среднего.

Switch требует axiswise nonregression combined sum отдельно на right и down и
хотя бы одного strict improvement. Tie-break: total delta, minimum axis delta,
fewest missing edges, frozen expected-correct score, frozen arm order.

## Tile-ID equivariance и диагностика

Арифметика использует только relation membership и значения joint slots.
Numeric tile ID не входит ни в score, ни в tie-break. Synthetic test применяет
произвольную биекцию ко всем sources, targets, union rows, heads и layouts:
arm indices остаются теми же, а каждый output становится ровно relabeled
версией прежнего output.

Target-free archive сохраняет для всех шести arms матрицы `[6,2]`:

- union coverage, confidence sums/means;
- source-normalized logit/confidence/combined sums;
- missing-edge counts;
- legacy fixed-head hits.

Это позволяет после freeze объяснить каждый switch без доступа к truth.

## Fail-closed sequencing

Файлы:

- module: `src/aiijc_puzzle/joint_relation_selector_portfolio.py`;
- freezer: `scripts/freeze_joint_relation_selector_scale256_portfolio.py`;
- blocked template:
  `configs/joint_relation_selector_scale256_portfolio_unsigned_template_v1.json`;
- tests: `tests/test_joint_relation_selector_portfolio.py`.

Checked-in template намеренно не имеет sidecar, связывает переданные root-agent
SHA-256 уже frozen target-free sibling triplets, но остаётся blocked с
`execution_authorized=false`. Эти NPZ не открывались при разработке portfolio.
Runner сначала
проверяет fixed DEV64 roster/rule commitment и останавливается на blocked
status **до** любого artifact lookup. В будущем reviewer должен создать новый
signed config, связать SHA-256:

- scale256 joint archive/metadata/pre-score-freeze (уже записаны в template);
- same-case relation roster archive/metadata/pre-score-freeze (уже записаны);
- legacy и portfolio implementation bytes.

Только после этого freezer проверит sibling hashes, schemas, no-label/no-pixel
flags, одинаковый case order и identity tuple
`case_id/source/draw/dirty_sha256`, затем эксклюзивно запишет portfolio archive,
metadata и pre-reference receipt. У runner нет score/reference mode.

## Научные ограничения

- Target-free nonregression не доказывает рост pairs/exact/Manhattan. Оба новых
  правила могут оставить incumbent на всех 64 boards — это честный no-op.
- Rule A сравнивает mean только среди mapped edges; coverage gate снижает, но не
  устраняет composition bias.
- Rule B намеренно убирает абсолютный scale между source rows. Он устойчивее к
  row calibration, но способен считать «наименее плохое» ребро положительным.
  Missing floor ограничивает этот риск, не доказывая truth.
- Frozen HGB expected score используется только поздним tie-break. Он не может
  сделать model-space-regressing arm eligible.
- DEV64 labels в будущем должны лишь прозрачно отчитаться по **каждому** member
  (`pairs`, adjusted pairs, exact, absolute Manhattan, radius-2); выбирать
  production member по этому одному panel нельзя без заранее отдельного gate.

На момент этой preregistration organizer pixels/labels, DEV64, local/terminal,
competition test, MPS и Weco не открывались и не запускались.
