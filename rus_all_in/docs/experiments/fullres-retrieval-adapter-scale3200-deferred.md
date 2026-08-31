# Full-resolution retrieval adapter: fixed scale3200 deferred to server

Дата: 2026-08-31. Статус: **signed continuation; local run stopped before
its first checkpoint and before scoring**.

Scale1600 дал положительный matched scaling slope относительно checkpoint400:
pooled R@1 `+0.2321 pp`, R@5 `+0.3453 pp`, raw-union top32 coverage
`+0.7190 pp` и reciprocal precision `+0.6855 pp`. Поэтому до нового training
был подписан ровно один continuation без architecture/loss/augmentation sweep:

- from scratch, тот же seed и первые 1600 train specs;
- fixed checkpoints `1600/3200` внутри одной cosine-3200 trajectory;
- те же 32 fit, 16 already-opened local и закрытые terminal16 sources;
- terminal открывается только после полного local retrieval/reciprocal gate;
- adapter pixels остаются matcher-only.

Preregistration:
`configs/fullres_retrieval_adapter_scale3200_preregistered_v1.json`, SHA-256
`792dc3304bcd173fd954bf3f0484338c7c690afc28e3b597c9c1d6d362a91a82`.

Локальный MPS запуск был сознательно остановлен после update400, когда
one-board timing показал около 37 минут до step3200, чтобы отдать GPU более
информативному raw+adapter+DINO verifier-у. Первый signed checkpoint был только
на step1600, поэтому **ни одного checkpoint или candidate artifact не
сохранено**. Ни local target/reference, ни terminal, ни competition test не
открывались; Weco153/154 и decoder не запускались. Это resource-priority stop,
а не отрицательный model result.

Server-ready запуск:

```bash
uv run python scripts/run_fullres_retrieval_adapter_scale3200.py \
  --device mps --allow-nondeterministic-mps \
  --output-dir outputs/fullres-retrieval-adapter/scale3200-server-v1
```

На CUDA следует оставить все signed hyperparameters/roster/gates неизменными и
поменять только backend glue, если runner расширяется поддержкой CUDA. Partial
локальная директория содержит только
`ABORTED_BEFORE_CHECKPOINT_OR_SCORING.json` и не должна использоваться как
resume artifact.

Tests: `tests/test_fullres_retrieval_adapter_scale3200.py`; вместе с parent
adapter/scale1600 tests — `9 passed`, Ruff clean.
