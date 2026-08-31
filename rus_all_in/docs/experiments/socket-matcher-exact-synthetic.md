# SocketMatcher: source-disjoint exact synthetic evaluation

## Зачем нужен отдельный protocol

Real train diagnostics восстанавливают permutation из пары dirty/clean и
неизбежно содержат label noise. Новый evaluator строит точную label сам:
берёт только clean canvas из manifest `train`, независимо искажает каждый из
576 фрагментов, перемешивает их и сохраняет inverse shuffle. Все local top-k и
глобальные layouts замораживаются в отдельный label-free artifact до сравнения
с inverse shuffle.

Runner:
[`evaluate_socket_matcher_synthetic_exact.py`](../../scripts/evaluate_socket_matcher_synthetic_exact.py).
Переиспользуемые проверки lineage, source selection и exact R@K:
[`synthetic_socket_evaluation.py`](../../src/aiijc_puzzle/synthetic_socket_evaluation.py).

Все Socket OT solver arms в этом evaluator сохраняют real-vs-dustbin массу:
используется исходный real-real log block с его общей OT-нормировкой, без
повторного row normalization. Поэтому эти global числа сопоставимы с поздними
[v2 checkpoint reruns](socket-matcher-v2.md), но не с global arms старого
training report, который был создан до исправления conversion-а.

Protocol fail-closed:

- checkpoint ancestry рекурсивно читается через `continued_from`; filename
  digests проверяются, missing ancestor и malformed lineage завершают run;
- evaluation sources детерминированно выбираются только из manifest `train` и
  исключают весь checkpoint lineage;
- hashes clean targets сверяются с manifest; train inputs, calibration,
  holdout и competition test files не открываются;
- corruption — reverse-engineered official-like independent per-tile pipeline:
  brightness `[-30,30]`, contrast `[.70,1.30]`, Gaussian noise sigma
  `[40,55]`, separable 3x3 blur, JPEG quality `35..50`;
- frozen NPZ не содержит clean pixels или exact reference permutations.

## Первый exact run: v2, 8 source-disjoint boards

Authoritative report:
[`report.json`](../../outputs/socket-matcher/exact-synthetic-v2-source8-draw1/report.json).
Checkpoint — `v2-border-train512-s300-r100-dev24`; восемь источников находятся
вне его 512-source lineage. Это маленькая train-development panel и одна
corruption draw на source, поэтому она доказывает наличие signal, но ещё не
оценивает его variance.

Local exact outgoing-neighbour retrieval:

| Scores | R@1 | R@5 | R@16 | R@32 |
|---|---:|---:|---:|---:|
| Bilateral | 8.3673% | 21.3881% | 37.8510% | 52.0267% |
| Socket raw | 11.4357% | 29.3139% | 47.4298% | 61.6168% |
| Socket partial OT | 13.0435% | 31.3972% | 50.3284% | 64.3116% |
| 80% Socket OT + 20% bilateral | **13.7228%** | **32.2124%** | **50.7586%** | **64.7758%** |

Exact global geometry:

| Layout arm | Direct | Translation-aligned | Adjacency |
|---|---:|---:|---:|
| Bilateral buddies96 | 0.1302% | 1.1502% | 4.0534% |
| Socket OT buddies96 | **0.2387%** | 1.2153% | 5.9556% |
| Fused OT buddies96 | 0.1519% | 1.1936% | 6.4878% |
| Fused relaxation + analytic border | 0.1953% | 1.2370% | 7.5294% |
| Socket hard-partial component/QAP decoder144 | 0.2170% | **1.4974%** | **8.4013%** |

Итог согласуется с recovered-label evaluation, но теперь без двусмысленности
labels: contextual SocketMatcher действительно улучшает candidate supply и
adjacency conversion. Одновременно absolute direct placement остаётся около
chance `1/576 = 0.1736%`. Значит, текущий bottleneck — не отсутствие local
signal, а привязка хороших компонент к абсолютной сетке и их согласованная
упаковка. Decoder144 — лучший adjacency arm этой exact panel, но не готовый
submission solver.

Воспроизведение:

```bash
.venv/bin/python scripts/evaluate_socket_matcher_synthetic_exact.py \
  --checkpoint outputs/socket-matcher/v2-border-train512-s300-r100-dev24/socket_matcher.pt \
  --output-dir outputs/socket-matcher/exact-synthetic-v2-source8-draw1 \
  --source-limit 8 --draws-per-source 1 --device cpu
```
