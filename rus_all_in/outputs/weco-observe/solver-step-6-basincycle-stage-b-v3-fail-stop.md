# Solver adjusted-pairs step 6 — BasinCycle Stage-B v3

Status: completed preregistered 6x6 EVAL32 x two draws; **fail-stop / no
promotion**.

- final-only FIT endpoint: 2,000 updates, no checkpoint selection or resume;
- target-free proposal-oracle coverage: `54 / 58 = 0.931034` opportunity
  states;
- fixed conservative selector chose a non-KEEP action in `0 / 64` cases;
- pair delta versus the matched same-panel control: `0.0 / 60` pairs, source
  bootstrap CI95 `[0.0, 0.0]`;
- exact delta: `0.0` tiles; radius-2 delta: `0.0` tiles;
- every control, proposal and selected output is a strict permutation of the
  36 original upright input tiles;
- no threshold/checkpoint sweep, DEV/holdout/terminal/competition test, or
  official submission access occurred.

Interpretation: proposal supply passed strongly, but the learned decision head
collapsed to the safe KEEP action. This is evidence against the Stage-B v1
selector/objective as tested, not evidence against short-cycle proposals.

Immutable evidence:

- scientific config SHA-256:
  `133587c2e0257c206b8d8109e7ba2addfb6bd48a167527c0e9771334df05b91`;
- execution binding SHA-256:
  `fc01838457a9bc788c83dc5703c29434007b1a94ac963b6b3635b1f3c4ee2a9d`;
- endpoint SHA-256:
  `766238d956798544fb4a3dbb968028b5bc2bb7539f19173e83dc1c8f5c3ef71d`;
- target-free bundle SHA-256:
  `b0ff71b112c4370243cea9fad9fe7893cbbc9e086bc1a30f5ba8a5d6ddb79b5d`;
- score report SHA-256:
  `7879f0b38136a6f44962a9151bc751543e44fe2362b31930a05ae47552972e62`.
