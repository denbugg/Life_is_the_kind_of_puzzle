# TASKA raw/calibrated layout portfolio gate

Status: **lower-total-seam-cost selector retained; harvested-edge selector
closed**.  This is a bounded diagnostic on the already-opened 32-case panel
and the historically model-selection-exposed held300 panel, not a fresh
promotion claim.

## Frozen portfolio

The two inputs are the unchanged legal full-harvest TASKA layouts:

- raw fused-cost component priority;
- the fixed train256 calibrated component priority from `calibrator.json`.

Both arms use the same original right/down cost matrices, component placer and
Hungarian tail.  Every candidate and selected layout is a strict permutation
of all 576 original upright fragments.

The target-derived pair oracle justified testing the two preregistered
selectors:

| Panel | Raw pairs / exact | Calibrated pairs / exact | Oracle pairs / exact | Oracle pair gain vs raw / calibrated |
|---|---:|---:|---:|---:|
| opened32 | 334.71875 / 4.46875 | 334.78125 / 4.28125 | 338.875 / 5.15625 | +4.15625 / +4.09375 |
| held300 | 329.625 / 2.90625 | 333.90625 / 2.71875 | 337.28125 / 3.34375 | +7.65625 / +3.375 |

The oracle was used only to open the bounded selector gate.  It does not enter
either selector.

## Retained selector: all-bond original seam cost

For layout `L`, original TASKA right cost `R`, and down cost `D`, the selector
computes exactly all 1,104 directed board bonds:

```text
C(L) = sum(r=0..23, c=0..22) R[L[r,c], L[r,c+1]]
     + sum(r=0..22, c=0..23) D[L[r,c], L[r+1,c]]
```

It selects the calibrated layout iff `C(calibrated) < C(raw)`; an exact tie
keeps raw.  This uses only matrices inferred from the current shuffled dirty
tile bag and is permutation-equivariant.

| Panel | Raw/calibrated selections | Selected pairs / exact | Pair delta vs raw, source-cluster CI95 | Exact delta vs raw, CI95 |
|---|---:|---:|---:|---:|
| opened32 | 18 / 14 | **336.8125 / 4.75** | **+2.09375 [0.25, 4.03125]**, source W/T/L 7/5/4 | +0.28125 [-0.15625, 0.875], 4/9/3 |
| held300 | 16 / 16 | **335.25 / 2.9375** | +5.625 [-0.3125, 14.96875], source W/T/L 6/5/5 | +0.03125 [-0.46875, 0.53125], 4/7/5 |

Against the calibrated arm, opened32 pair/exact deltas are +2.03125 and
+0.46875; held300 deltas are +1.34375 and +0.21875.  The direction transfers
for both metrics, though held intervals remain wide.

## Closed selector: harvested-edge realisation

The second fixed selector lexicographically maximised:

1. the count of frozen harvested edges realised in the requested direction;
2. `sum(-cost)` over those realised edges.

It selected raw/calibrated 25/7 times on opened32 and produced 335.90625 pairs
but only 3.875 exact: exact delta versus raw was -0.59375 with CI95
[-1.65625, 0.0625].  On held300 it selected 22/10 and produced 334.5625 pairs
and 3.0 exact.  The opened exact regression closes this selector; it is not
implemented as a production primitive.

## Legality and scope

- no target, exact permutation, filename, source coordinate, or competition
  test input enters selection;
- matcher outputs, raw solver, component placer, and tile pixels are unchanged;
- all 64 selected layouts across the two panels are strict original-tile
  permutations;
- targets are used only for the offline metrics and oracle analysis above.

