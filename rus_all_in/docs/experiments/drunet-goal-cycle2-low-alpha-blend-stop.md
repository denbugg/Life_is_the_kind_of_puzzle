# Cycle2 low-alpha B/C blend rescue: target-free stop

## Outcome

The bounded reused-train-only low-alpha rescue was stopped before any target
scoring because none of its three preregistered candidates passed the complete
target-free structure gate versus cycle2 B. The entire low-alpha blend line is
rejected. The prior rejection of cycle2 C and D is unchanged.

No new neural inference was performed. No train target PNG was opened by this
screen, and no calibration, holdout, competition-test, or production artifact
was accessed or changed.

## Scope and exposure

This was an adaptive follow-up designed after the cycle2 B/C/D train results
were known. It used the same previously exposed train offset 528, count 16:
the first eight were intended for selection and the last eight for unchanged
verification. It makes no freshness, independence, source-family, or
generalization claim.

The only candidates were exact half-up uint8 blends of already-frozen cycle2 B
and C PNGs:

- `E_B90_C10`: 90% B / 10% C;
- `F_B80_C20`: 80% B / 20% C;
- `G_B70_C30`: 70% B / 30% C.

The formula was
`floor((B_weight*uint16(B) + C_weight*uint16(C) + 5) / 10)`.

All 48 blended PNGs, their commitment, and the all-16 target-free safety
decision were frozen before the decision to stop. Because no arm was safe, the
first-eight selector and last-eight confirmation were never run.

## Target-free result versus B

| Candidate | Luma mean/min | Chroma mean/min | Laplacian mean/min | Grid mean/max | Exact-constant increase | Near-flat std<2 increase mean/max | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E 90/10 | 0.990216 / 0.987461 | 1.008803 / 0.999836 | 0.990601 / 0.985178 | **1.039159** / 1.050084 | 0 | 0.0 / 0 | FAIL: grid mean >1.02 |
| F 80/20 | **0.974514** / 0.969083 | 1.028090 / 0.994795 | 0.981957 / 0.960178 | **1.137521 / 1.213506** | 0 | 0.5 / 2 | FAIL |
| G 70/30 | **0.956918 / 0.948174** | 1.040392 / 0.989328 | **0.971244 / 0.919076** | **1.255527 / 1.425605** | 0 | **3.0** / 5 | FAIL |

The fixed bounds were luma/chroma/Laplacian mean/min at least `0.98/0.95`,
grid mean/max at most `1.02/1.10`, no exact-constant-tile increase, and
near-flat std<2 increase at most `+2` mean and `+6` on any board.

The nearest arm, E 90/10, preserved local gradients and flatness but increased
the mean grid ratio by about 3.92%, exceeding the 2% cap. The stop rule was not
weakened or reinterpreted.

## Immutable evidence

- Preregistration:
  `configs/drunet_goal_cycle2_low_alpha_blend_train_preregistered_v1.json`,
  SHA-256 `f19f743659bceaf4b7d237ce581ce4162a4c4d3b6a042cf2551f02f8af84a85e`.
- Frozen prediction commitment:
  `outputs/drunet-goal-cycle2-low-alpha-blend-v1/train-offset528-count16/prediction-commitment.json`,
  SHA-256 `290dd7bc7fe31c592ef1fade784c6007f1ad33d5888ac66b71be5d26b3b6815a`.
- Target-free safety decision:
  `outputs/drunet-goal-cycle2-low-alpha-blend-v1/train-offset528-count16/target-free-safety-decision.json`,
  SHA-256 `388cc765286c0ac0bb6fc6a36c2a29ab5daaee59b93fcf8775d6ea01885beb50`.
- External receipt:
  `outputs/drunet-goal-cycle2-low-alpha-blend-v1/train-offset528-count16.commitment-receipt.json`,
  SHA-256 `d41f4cda9c11d0023b018a7ae72fbe6adab95e34c8e23af24b89fcd7bd8bb109`.
- Runner:
  `scripts/run_drunet_goal_cycle2_low_alpha_blend_train.py`, SHA-256
  `9885d2bc787468d74775351339b64378b6963c12ce58dad8c986653611c01ba4`.

No `selection-decision.json` or `report.json` exists, because target scoring
was not authorized.
