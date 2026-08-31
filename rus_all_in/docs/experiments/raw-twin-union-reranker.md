# Raw d64 / fullres-twin union reranker v2

Дата: **2026-08-30**. Статус: **оба локальных D1 gate и frozen
fresh64 submission gate пройдены**.

## Зачем понадобился новый arm

Full-resolution ordered twin matcher проиграл frozen d64 как самостоятельный
ranker, но его top-32 добавлял `+7.416 pp` exact-neighbour coverage. На той же
уже открытой eval24 одноразовый target-aware diagnostic показал в среднем
`81.875` истинных twin-only edges/board и выделил incoming rank, margin и низкую
sequence variance как наиболее полезные target-free признаки. Этот diagnostic
мотивировал и затем заморозил семейство признаков; вся старая eval24 исключена
из нового fit/eval и не использовалась для настройки v2.

До capacity была исправлена важная ошибка первоначального контракта. Union
`raw32 ∪ twin32` не обязан воспроизводить full raw hard matching: после
row/column conflicts исходная OT может выбрать edge вне row-top32. Поэтому
финальный immutable roster равен
`raw32 ∪ twin32 ∪ every frozen-raw hard-projection edge`. Последняя часть даёт
готовое feasible matching из 552 edges/axis. Untouched full raw d64 OT остаётся
comparator; restricted-OT эквивалентность ему не заявляется. В permanent tests
zero-residual projection совпадает с full raw на adversarial 4×4, а каждый
learned projected edge обязан входить в union.

## Архитектура

Каждый из примерно 52.7k directed candidate edges получает:

- raw/twin score, outgoing/incoming ranks, margins и membership flags;
- принадлежность frozen raw hard projection и reciprocal flags;
- шесть statistics упорядоченных 20-position twin sequences;
- frozen d64 source/target/absolute-difference/product tokens (`4×64`);
- frozen Socket outgoing/incoming border evidence.

64-D edge encoder агрегируется permutation-equivariant DeepSets summaries по
outgoing row, incoming column и axis-board. Zero-initialized bounded residual
точно сохраняет raw score/order внутри union. Обучение использует равные
outgoing-row и incoming-column listwise CE, hard-pair auxiliary и малый
residual L2. На inference outside-union logits равны `−10000`; float32
exponential underflows to zero, и runtime отдельно отвергает любой hard edge
вне union. Затем идут unchanged frozen border logits, partial OT и exact
one-to-one projection.

Это не повтор fullres-relation-fusion: там единицей была component relation и
restored-d64 supply; здесь одновременно оптимизируются incoming/outgoing
tile-edge lists, supply даёт ordered twin field, а one-to-one OT projection
входит в сам оцениваемый treatment.

## Protocol и preflight

Preregistration SHA-256:
`6741e92e832a630f1b83bde6edc8a341a348f52daa82313c40a8f32c7c1173d4`.
Selection commitment был записан до target access; SHA-256
`71ae4f5095489613857fcd25c541fe496da0d6861f6ff604850147dd04b91cd2`.
Fit256/eval24 order digests: `33654879…` / `f2ad0e01…`. Помимо полной ancestry
и panel registry принудительно исключены fresh64 direct-hard-edge roster
`6056fcc5…` и все 256 fit + 32 eval component-placer sources из commitment
`2b2b0c90…`.

Procedural 4×4 capacity прошла: fresh R@1 `100%`, loss `2.763→0.253`, runtime
`3.07 s`. Идентичный full576 frozen-feature + backward update занял `2.612 s`
на CPU и `0.3763 s` на MPS, поэтому pilot был закреплён за MPS.

Первая operational попытка столкнулась с MPS OOM около step190 при
одновременном чужом MPS process. Никакой checkpoint или metric не выбирался.
Авторизованный recovery сохранил config, commitment, seed, model, features,
preprocessing и 400 steps; изменились только periodic cache release, reuse
того же commitment и перенос eval preparation после полного fit. Инцидент и
old/new runner hashes находятся в `mps-oom-recovery.json`. Recovery fit400
занял `179.697 s`.

## Frozen eval24 result

Comparator — untouched full frozen raw d64 assignment/projection.

| exact local / projected metric | raw d64 | learned union | delta |
|---|---:|---:|---:|
| pooled partial-OT R@1 | `17.9725%` | `18.4481%` | `+0.4755 pp` |
| pooled partial-OT R@5 | `35.9073%` | `36.1866%` | `+0.2793 pp` |
| all projected correct edges/board | `183.833` | `194.875` | `+11.042` |
| fixed top144/axis correct edges/board | `134.500` | `142.958` | `+8.458` |
| fixed top144/axis precision | `46.7014%` | `49.6383%` | `+2.9369 pp` |

Ranking arm требовал R@1 `+0.25 pp` при nonnegative R@5; hard arm требовал
`+2` correct top144 edges/board без precision loss. **Оба arm прошли**.

Только после этого автоматически открылся predeclared decoder144 + unchanged
cyclic-border5, descriptive на той же панели:

| strict layout metric | raw d64 | learned union | delta |
|---|---:|---:|---:|
| exact tiles/board | `0.792` | `1.208` | `+0.417` |
| direct exact fraction | `0.1374%` | `0.2098%` | `+0.0723 pp` |
| adjacency | `12.8472%` | `13.7530%` | `+0.9058 pp` |

Все 48 decoded layouts (две arms ×24) — строгие перестановки исходных upright
tile identities. Pixel prediction/replacement отсутствует; holdout и competition
test не открывались. Exact остаётся primary, и этот small-panel descriptive
рост не является production promotion или разрешением на same-panel sweep.

## Artifacts

- report SHA-256: `0a5f0bb990654a0e191430bbf05796332c2f6fce181d51b05ee8b11ba1477bc4`;
- checkpoint SHA-256: `a5f882ab3c827e4e3779be3372c62d2a8fb9cd95d3558fd30cc566a9c3137f79`;
- frozen dirty-only NPZ SHA-256:
  `6077f97a5f9bddf1306d4129583469b25e90d0fc3a3db5f33546fa282830dc8b`;
- frozen metadata SHA-256:
  `67cabfb2b6e4349c7290e116d8a5d3e74dd4995e18cd911842a97652bd82aa69`;
- capacity / benchmark SHA-256: `13a99959…` / `5056371b…`.

Implementation: `src/aiijc_puzzle/raw_twin_union_reranker.py`, runner
`scripts/run_raw_twin_union_reranker_v2.py`, tests
`tests/test_raw_twin_union_reranker.py` and
`tests/test_run_raw_twin_union_reranker_v2.py`. All artifacts are under
`outputs/raw-twin-union-reranker/v2-fit256-s400-eval24/`.

Decision по D1: сохранить checkpoint как сильный source-disjoint
relative-edge primitive и провести один заранее замороженный fresh exact
layout confirmation без tuning на eval24. Эта проверка завершена ниже.

## SHA-locked non-default production adapter

После явного выбора Union-v2 для submission добавлен
`src/aiijc_puzzle/raw_twin_union_production.py`. Он не меняет baseline/default:
без learned artifacts `predict_raw_twin_union_variants(...)` напрямую вызывает
существующий Socket `decoder144+cyclic5+identity` path. Если передан только один
из Twin/Union artifacts, SHA не совпал или lineage несовместим, вызов fail-closed.

Frozen production lineage:

- Socket d64: `0e9df49a503c65aac7f1468e9acd6a074a5e658ae8b61f8954be086272c49670`;
- fullres Twin: `c5b44901e8da459e3c48b6e7af7153c5d7eed26f1c1b52c8712c4fa0dc4ea8ae`;
- Union-v2 head: `a5f882ab3c827e4e3779be3372c62d2a8fb9cd95d3558fd30cc566a9c3137f79`;
- prereg config: `6741e92e832a630f1b83bde6edc8a341a348f52daa82313c40a8f32c7c1173d4`;
- selection commitment: `71ae4f5095489613857fcd25c541fe496da0d6861f6ff604850147dd04b91cd2`.

Public loading/inference contract:

```python
device = choose_deterministic_device("cpu")  # либо явно "mps"
socket = load_socket_checkpoint(socket_path, device=device)
twin = load_fullres_twin_checkpoint(twin_path, device=device)
union = load_raw_twin_union_checkpoint(
    union_path,
    config_path=config_path,
    selection_path=selection_path,
    device=device,
)
prediction = predict_raw_twin_union_variants(
    image_uint8_rgb_480x480,
    socket,
    device=device,
    twin=twin,
    union=union,
)
output = prediction.selected.output
layout = prediction.selected.layout
```

Inference ровно повторяет frozen treatment: original upright tiles → Socket
d64 context + Twin ordered sides → `raw32 ∪ twin32 ∪ raw hard projection` →
54,449-param reranker → outside-union `−10000` → restricted partial OT с
Socket border logits → exact 552-edge/axis hard projection audit →
decoder144 → cyclic-border5. API не принимает target, manifest, filename или
restored pixels. Full-board CPU smoke дважды дал одинаковый layout; baseline
fallback совпал bit-for-bit с существующим production callable, а обе arms
прошли exact 0..575 permutation и original-tile multiset audits.

## Frozen fresh64 confirmation

После выбора Union-v2 как submission candidate был однократно
заморожен source-disjoint panel `64×draw0`. Из него исключены все
предыдущие rosters, включая Union fit256/eval24 и component-placer
fit256/eval32. Конфиг был записан до target access:
`configs/raw_twin_union_reranker_fresh64_confirmation_v1.json`, SHA-256
`545890095e1ca928e4f56a837596be0d9801d3adc66ae44885a9aa10f262b721`;
source-order digest
`abc679c3dc96da372f74c8e46f2886c91c33779a6e62777de23b525f34cc444d`.
Использован ровно frozen checkpoint `a5f882ab…`: без retrain,
recalibration, смены hyperparameters или inference semantics. Target-free
scores и layouts были записаны до reference scoring.

| frozen fresh64 metric | raw d64 | Union-v2 | delta, clustered 95% CI |
|---|---:|---:|---:|
| exact tiles/board | `0.9375` | `1.2813` | `+0.3438` `[-0.1094, +0.7969]` |
| adjacency | `13.6676%` | `14.4192%` | `+0.7515 pp` `[+0.5364, +0.9794]` |
| projected correct edges/board | `188.109` | `197.703` | `+9.594` `[+7.391, +11.891]` |
| fixed top144 correct/board | `141.719` | `146.984` | `+5.266` `[+3.563, +6.984]` |
| fixed top144 precision | `49.2079%` | `51.0362%` | `+1.8283 pp` `[+1.2368, +2.4414]` |

Predeclared submission gate: exact mean delta `>=+0.25` tile/board, strictly
positive adjacency delta и все 128 layouts strict. **Gate пройден**:
exact `+0.3438`, adjacency positive, `128/128` строгих исходных upright
permutations. При этом exact CI пересекает ноль; это честно
фиксируется и не было дополнительным gate.

Final artifacts:

- report: `outputs/raw-twin-union-reranker/frozen-v2-fresh64-draw0/report.json`,
  SHA-256 `c4ae8cb6fff97cc5a2901f922273e0702db373e25eb03986dd8af089582d04f7`;
- frozen target-free NPZ:
  `outputs/raw-twin-union-reranker/frozen-v2-fresh64-draw0/frozen-target-free-predictions.npz`,
  SHA-256 `31a5d532aa5400960d7a548be6cf0f0b7e2547c5d8d8e921f00b4e39593dc2b8`;
- frozen metadata:
  `outputs/raw-twin-union-reranker/frozen-v2-fresh64-draw0/frozen-target-free-predictions.json`,
  SHA-256 `1b28edcc65c647fc68bfdf0b4322a99a69acfceb15d876b03c4713a99347ef9c`.

Frozen verdict: **`frozen-fresh64-submission-candidate-confirmed`**. Holdout/test
не открывались; pixel replacement/generation нет.

## Official combined-pipeline result

После frozen confirmation был собран legal `Union-v2 + historical h20` ZIP:
`outputs/union-v2-submission/submission-union-v2.zip`, SHA-256
`8866e060cae32d56277470f565779cd68826d9a766513e3e81eed2165f6d9725`.
Пользователь сообщил официальный leaderboard score
**`0.24201676406343967`**. Предыдущий `fixed-B standard + buddies96` submission
имел **`0.2762279116935955`**, поэтому новый combined arm проиграл
`-0.03421114763015583` (`-12.39%` относительно предыдущего результата) и не
должен быть отмечен лучшим.

Это сравнение **не является чистой оценкой Union-v2 solver-а**: одновременно
изменились layout (`buddies96 -> Union-v2`) и restoration
(`fixed-B DRUNet50/protected h28-h50 -> historical h20`). Поэтому официальный
результат отклоняет именно submitted `Union-v2+h20` pipeline, но не отменяет
source-disjoint fresh64 gains по adjacency/top144 и не доказывает, что Union
layout хуже buddies96 при одинаковом pixel tail. Следующий честный solver gate
должен сравнивать layouts по exact/adjacency, а end-to-end arms — только под
одним и тем же frozen restoration tail.

Production MPS replay также оказался не bit-exact. На `img_000004.png` Socket,
Twin, candidate features и Union edge encoder совпали побитно; первое
расхождение возникло в grouped mean через MPS `index_add_` (`~1e-7`), выросло
до `~1.5e-6` в residual и изменило decoder layout. Seeds, synchronize и
empty-cache не помогли, а `torch.use_deterministic_algorithms(True)` остановился
на недетерминированном `index_put_with_accumulate_mps`. Поэтому MPS layout
нельзя валидировать требованием повторного bit-exact inference; для текущего
run допустим только независимый audit зафиксированного strict layout, raw tile
multiset и pixel tail. Будущий полностью воспроизводимый Union run должен
считать grouped reductions на CPU либо выполнять весь inference на CPU.
