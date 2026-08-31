# Solver step 56 — fresh exact-oriented override confirms positive means

Parent in both Weco runs: step 55.

The preregistered held pair promotion gate had failed, but held exact delta was
positive with CI95 entirely above zero. That independent signal triggered one
fixed no-tuning fresh32 replay as an explicit exact-oriented override. No
feature, model, selector, tail budget, or threshold changed after held scoring.

On fresh32, standalone stacker reached **344.46875 pairs**, recall
**0.312018795**, and **1.375 exact**. Five-arm-tail96 reached **347.15625
pairs**, recall **0.314453125**, and **1.34375 exact**, versus retained
four-arm-tail96 **346.0625 / 0.313462409 / 1.15625**.

Five-minus-four was **+1.09375 pairs**, source-cluster CI95
`[-0.09375,+3.0]`, case W/T/L `3/27/2`; exact was **+0.1875**, CI95
`[-0.03125,+0.4375]`, case W/T/L `4/27/1`. All target-free candidates were
SHA-frozen before reference reconstruction and every layout was a strict
original-tile permutation.

Verdict: retain as a promising optional fifth arm / exact-oriented portfolio.
Do not replace the pair-production default because held pair sign reversed and
fresh intervals still cross zero.
