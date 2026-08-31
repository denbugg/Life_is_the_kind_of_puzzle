# Joint reciprocal tri-emitter verifier

Status: **synthetic collision capacity passed. This is a plumbing result, not a
real-panel quality claim. No organizer source, opened local16 replay, terminal
panel, decoder, competition test, submission or Weco step was used.**

## Why this branch is materially new

The previous tri-emitter verifier learned only an outgoing row-listwise exact
neighbour objective. Incoming rank was a frozen scalar feature, so the model was
not trained against many-to-one column collisions. That model improved local16
R@1/R@5 but failed its matched reciprocal-precision gate.

The new module keeps the previous content scorer unchanged and attaches one
joint sparse-assignment objective:

- row cross-entropy over the frozen candidate row plus a learned `NONE` class;
- column cross-entropy over all sources nominating a target plus learned `NONE`;
- `0.25 × BCE` on exact edge truth using the differentiable minimum of row and
  column leave-one-out margins, with fixed `tau=0.25`;
- learned confidence bias and positive softplus temperature;
- the preregistered `1e-3 × mean(delta²)` raw-residual regularizer.

`NONE` is the label both for a true board border and for an exact neighbour
absent from the immutable raw+adapter+DINO union. Candidate identities are
never added, removed or rewritten by the model.

The inference policy is also fixed before any real evaluation: independently
for right and down, keep reciprocal row/column top-1 winners, order by the
two-sided confidence, and admit exactly `ceil(0.05 × tile_count)` when that many
reciprocal winners exist. There is no learned threshold and no 3/5/10/20%
coverage sweep.

## Signed capacity protocol

The synthetic board is exactly `4×4`, with eight candidates per source and two
direction axes. Matching ordered raw/DINO boundary content identifies the true
edge, while a deliberately higher raw baseline points at one shared target on
each axis. Each axis therefore contains at least twelve hard many-to-one
collision distractors. The gate simultaneously requires:

- row, column, exact-edge and `NONE` R@1 all equal to 100%;
- every positive two-sided confidence strictly above every hard collision;
- the fixed 5% head to contain its exact requested count at 100% precision;
- transpose and simultaneous-relabel confidence errors at most `1e-5`;
- final/initial loss ratio at most `0.25`.

The authoritative preregistration is
`configs/joint_reciprocal_tri_emitter_capacity_preregistered_v2.json`, SHA-256
`1a86fd0d5b6ae1d27e7a397103f88a8ce92111f7dc2d7997e874844c13b946db`.
It freezes one seed, 600-step endpoint, architecture, optimizer, objective and
gate before the v2 capacity run.

## Capacity result

The v2 run passed every gate:

| metric | right | down |
|---|---:|---:|
| row all / exact-edge / `NONE` R@1 | 100% / 100% / 100% | 100% / 100% / 100% |
| column all / exact-edge / `NONE` R@1 | 100% / 100% / 100% | 100% / 100% / 100% |
| minimum positive confidence | 6.4058 | 6.4181 |
| maximum hard-collision confidence | -8.7080 | -8.7101 |
| fixed 5% selected / precision | 1 / 100% | 1 / 100% |
| reciprocal winners available | 12 | 12 |
| transpose max error | 0 | 0 |
| relabel max error | `1.91e-6` | `1.91e-6` |

Loss fell from `4.264671` to `0.0299925`, ratio `0.007033`. The global minimum
positive-minus-maximum-collision confidence gap was `+15.1138`.

Authoritative capacity report SHA-256:
`94b769da114553ec98212abca4de758a587dc52661c54c66fd4eda03e8b8ed7c`.
The capacity-only checkpoint SHA-256 is
`6046113fb7320f0281469a55f7fd87e38f129386e1b1e1f8902bc7b20cecc6de`;
it must not seed a real fit.

## Audited numerical repair

The v1 signed run already reached 100% classification, 100% fixed-head
precision and a `+15.1136` confidence gap, but failed only the `1e-5` invariant:
direct float32 `log(exp(total)-exp(edge))` lost about `7e-4` when one edge
dominated. Its fail-stop report is preserved at SHA-256
`dd1fd17453fda056ea840fac76156fb2ef1d30b0d77ac7e823b16590a7bcd948`.

Before rerunning, v2 was signed with the identical model, synthetic case, seed,
endpoint, objective and gates. The only repair explicitly masks the selected
argmax and recomputes its leave-one-out `logsumexp`; non-winners keep the same
formula. A new high-dynamic-range unit test covers transpose and relabel
equivariance.

## Reproduction and next boundary

```bash
uv run ruff check \
  src/aiijc_puzzle/joint_reciprocal_tri_emitter_verifier.py \
  scripts/run_joint_reciprocal_tri_emitter_capacity.py \
  tests/test_joint_reciprocal_tri_emitter_verifier.py
uv run pytest -q tests/test_joint_reciprocal_tri_emitter_verifier.py
uv run python scripts/run_joint_reciprocal_tri_emitter_capacity.py
```

The next step, if separately authorised, is a newly signed small FIT and a new
source-disjoint discovery panel. This capacity result does not authorise
reopening local16, selecting a coverage threshold, opening terminal data or
feeding the head to a decoder.
