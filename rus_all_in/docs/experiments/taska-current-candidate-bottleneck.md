# TASKA current candidate/solver bottleneck diagnostic

Status: **target-assisted diagnostic on the already opened disjoint local32
panel**.  This is not a promotion experiment and does not authorize any
competition-test access.  Its purpose is to distinguish missing matcher supply
from losses inside the current four-arm + protected-tail96 solver.

## Frozen inputs

- target-free layouts, costs, harvested identities, and recovered focal scores:
  `outputs/taska-focal-feature-stacker/train96-v1/local32/`;
- exact synthetic references were recreated only for this offline diagnostic;
- evaluated output: the unchanged fixed four-arm all-bond selector followed by
  protected tail96;
- all 32 layouts are strict permutations of the 576 original upright tiles.

For every board, a harvested edge was marked true when its source and target
were exact neighbours in its declared right/down direction.  It was marked
realised when the scored layout placed the same identities in that direction.

## Decomposition

| Quantity, mean per board | Value |
|---|---:|
| Harvested candidate edges | 374.4375 |
| True harvested edges | 252.9375 |
| Candidate precision | 67.5513% |
| Candidate true-edge recall / 1104 | 22.9110% |
| Realised harvested edges | 340.0000 |
| True realised harvested edges | 245.3125 |
| Realised-candidate precision | 72.1508% |
| True harvested edges not realised | 7.6250 |
| Final satisfied pairs | 314.3750 |
| Final true pairs absent from the harvest | 69.0625 |
| False harvested edges realised | 94.6875 |

The current solver therefore realises about `97.0%` of the true relations that
already exist in the harvest (`245.31 / 252.94`).  More global search over the
same identities cannot create the missing relative signal.  The two useful
near-term levers are:

1. add genuinely complementary candidate supply while retaining raw evidence;
2. stop freezing low-confidence false realised candidates during the seam
   tail, rather than globally deleting the weak tier before assembly.

The final layout does find about 69 true non-harvest seams through the dense
raw costs and Hungarian/tail steps, so the harvest is not a hard oracle ceiling.
It is still the clearest current bottleneck because only 7.6 supplied true
relations are lost while 851 true relations are absent from the harvest.

## Focal protection diagnostic

An exploratory target-assisted cut audit inspected recovered focal thresholds
`-1, 0, 1, 2, 3` on the realised harvested edges.  This makes the panel touched
for threshold selection; no fresh claim may be made from it.  The natural
classifier boundary `logit >= 0` gave the most actionable separation:

| Realised-edge subset | Edges / board | True / board | Precision |
|---|---:|---:|---:|
| focal logit `>= 0` | 258.4375 | 228.65625 | 88.4764% |
| focal logit `< 0` | 81.5625 | 16.65625 | 20.4215% |

This does **not** justify a threshold sweep.  It preregisters exactly one
materially different continuation: retain the same pre-tail layout and
original TASKA seam objective, but protect only realised candidates with the
frozen focal logit at or above zero.  A second independent continuation tests
one fixed expansion of the dynamic vote target from 350 to 500.  Both must
freeze layouts before scoring and stop if pair gains do not transfer.

## No-repeat boundary

- Do not spend more search budget on rearranging only the existing harvested
  true identities: almost all are already realised.
- Do not tune focal thresholds on local32; the exploratory diagnostic already
  touched them.
- Do not confuse candidate recall with final adjacency: dense raw costs create
  useful noncandidate seams, so every new supply arm still needs a full legal
  solver gate.
- Restored/denoised pixels may be used only as matcher evidence.  The emitted
  layout must remain a one-to-one permutation of the original upright tiles.
