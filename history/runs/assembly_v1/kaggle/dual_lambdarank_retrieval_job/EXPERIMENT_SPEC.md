# Dual-sided seven-origin LambdaRank retrieval gate

## Hypothesis

The current full-union HGB verifier optimizes pooled binary classification and
collapses the `softcycle`, `qap_w4`, and `qap_w1` candidate origins into a
single popcount.  A query-grouped ranker may recover useful successor ordering
without another expensive pixel model.

## Frozen protocol

- fit: first 24 whole sources from `edge_development`, both corruption panels;
- calibration: `edge_development[368:376]`, both panels, disjoint from fit and
  not present in existing assembly result artifacts at protocol freeze;
- candidate union: the same seven-origin recipe as V4 — C1/HBT outgoing
  top-32, incoming top-8, plus deterministic soft-cycle, qap-w4, and qap-w1
  layout edges — evaluated on new corruption/QAP replicas;
- features: legacy 25 rank/origin features with popcount corrected to `/7`,
  plus explicit bits for the three layout origins;
- models: separate outgoing and incoming LightGBM LambdaRank models;
- inference: mean of within-query outgoing and incoming percentile ranks;
- V4 and external assembly labels remain unopened during this gate.

## Retrieval continuation gate

On each corruption panel, the candidate must satisfy all of:

- Recall@1 improves by at least `0.01` over the strongest of HBT, w4, and the
  frozen legacy HGB;
- Recall@5 does not regress;
- MRR improves;
- candidate recall is byte-contract identical;
- top-1 destination collisions do not exceed w4.

Only a passing frozen model may open the external assembly/QAP gate.  Failure
closes this route and leaves ContinuationNet-0 as the next scorer hypothesis.
