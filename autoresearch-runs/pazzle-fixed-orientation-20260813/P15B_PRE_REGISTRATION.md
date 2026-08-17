# P15b Pre-Registration: Bounded MPRL-24 Seed Construction

> Status: **PRE-REGISTERED BEFORE P15b SOURCE MODIFICATION** on 2026-08-17.

## P15a record

P15a was stopped during synthetic G0a before any score cache, label cache, target PNG, CAL, DEV, held or test access. Its repeated component-packing starts each invoked `repair_passes=2`, duplicating the canonical exhaustive 96-by-576 swap repair. After exceeding the three-minute synthetic fast-futility checkpoint without a report, the process was stopped. This is an integration/runtime abort, not a metric outcome.

## P15b bounded correction

P15b changes exactly one implementation-level detail: the canonical seed remains the canonical rank96 decode with `repair_passes=2`, but the three auxiliary deterministic component-packing boards used **only to broaden sparse cell support** are created with `repair_passes=0`, `restarts=1`, and the same three locked seeds `20260816, 20260817, 20260818`. They are never candidate outputs and cannot inflate the final objective comparison. The MPRL support update, K=32, alpha=0.50, two phases, four iterations per phase, Hungarian projection, frozen inputs, P8 prohibition and all P15 gates remain unchanged.

## Additional runtime contract

P15b G0a must finish within 90 CPU seconds on the fixed synthetic field. Failure stops P15b before frozen-cache access. P15b G0b retains the original four-board under-ten-minute cap. This correction is pre-registered because runtime boundedness is part of the experimental mechanism after P14d was stopped for weak compute value.

## Integrity statement

P15b does not relax fixed orientation, strict 576-way bijection, candidate-order invariance, no-target-PNG control, or held/CAL/DEV/test closure. It continues to prohibit P8 artifacts.
