# Solver step 23: external train256 calibrator has a held pair signal

The fixed 15-feature TASKA edge calibrator was trained on 96,104 harvested
edges from a preregistered deterministic roster of 256 organizer-train sources.
The roster used only filename indices below 6700 and excluded every opened32
source; the complete historical held300 range remained outside training.
Inference uses only dirty-visible TASKA matrices and harvest evidence.

The trained model was replayed without adjustment on both frozen panels:

| Panel | Arm | Pairs / 1104 | Recall | Exact / 576 |
|---|---|---:|---:|---:|
| opened32 | raw parent | 334.71875 | 0.303187274 | 4.46875 |
| opened32 | train256 calibrator | 334.78125 | 0.303243886 | 4.28125 |
| held300 | raw parent | 329.625 | 0.298573370 | 2.90625 |
| held300 | train256 calibrator | **333.90625** | **0.302451313** | 2.71875 |

Held pair delta was +4.28125 with source-clustered 95% interval
`[-2.9070, +14.0008]` and source W/T/L `10/1/5`; exact delta was -0.1875.
All 64 layouts were strict original-tile permutations.  The transfer signal is
larger than the opened32 effect, but the interval still crosses zero and exact
does not improve.  This is a useful exploratory pair arm, not a confirmed
promotion over raw ordering.

Artifact SHA-256 values:

- training features: `2d1ef6267daab67d74971d625d2d446e7dfb8dc30a6165bd3459ab969e34f373`;
- calibrator JSON: `9a175b1cec1e6e6bd176e78ae80e16af78df8a00277836aaf67a8daf78782758`;
- locked report: `20311e8aede6b2165c1f86a224eab5251444498ddcafa08e22088324e21f7d26`.

