# Where this stands, and what to do next

Rewritten 2026-08-22, after a day that ended by locating the bottleneck instead
of nudging it. The shippable build is **honest_v4** (local SSIM 0.2690, expect
about 0.292 on the platform; no colour substitution anywhere).

## The one number that reorders everything

**The payoff from correct edges is a CLIFF, and every quantity this project has
optimised for months lives underneath it (M268).** Hand the unchanged pipeline
K edges drawn from the TRUE adjacency set -- the ceiling of every possible
selector at that volume -- and the board is worth:

| correct edges | largest clean component | fragment placement | SSIM |
|---:|---:|---:|---:|
| 159 (what we harvest) | 15.6 | 0.0258 | 0.2690 |
| 400 | 46.1 | 0.0123 | 0.2595 |
| **552** | **266.8** | **0.6613** | **0.5405** |
| 750 | 561.1 | 0.9986 | 0.6477 |

Going from 159 to 400 -- two and a half times the evidence, far beyond anything
any selector has achieved -- is worth **nothing**. Going from 400 to 552 wins the
competition outright against a leader at 0.40. This is M221's percolation
threshold, confirmed and priced.

**How clean they must be (M269, M270).** Without help, 552 edges at precision
0.95 collapse: twenty-nine false edges among 581 take clean coverage from 0.931
to 0.322, because in a dense graph a false edge fuses two LARGE components and
the damage scales with what it fuses. With the consistency filter the
requirement softens to **552 edges at 0.97 in the PREFIX**, after which garbage
is not merely tolerable but useful.

The filter (`scratchpad/consistency_filter.py`) is a union-find carrying an
offset. An edge is accepted only if it agrees with the relative position its
ends already have, puts no two fragments in one cell, **and does not push the
assembly past 24 cells** -- the photograph is 24x24, so anything wider is
provably wrong. That last check is not cosmetic: without it, five false edges
stretch the assembly out of the board, `search` silently drops what it cannot
place, and 0.8824 fragment placement reads as 0.1648.

Order is everything. The same edges shuffled give clean coverage 0.268 against
0.931. On the learned selector's real ordering the filter does nothing, because
that ordering already carries thirty-three false edges in its first two hundred.

## What we have against what is needed

| | now | needed |
|---|---:|---:|
| correct edges a board | 159 | 552 |
| precision at that volume | ~0.50 | ~0.97 in the prefix |
| matcher R@1 | 0.334 | ~0.52 |

## Why nothing on the assembly side can close it

**The placement objective is excellent and its own context destroys it (M263).**
Scored against the other components at their TRUE places, the true position of a
component ranks first **52.2%** of the time (74.2% for components of eight
fragments or more). Scored against the search's own arrangement it ranks first
**0.8%** of the time, 175th of 234. That is a coordination failure: from an
all-wrong arrangement no single correct move improves the score, so annealing,
exchanges, priors and restarts all push on a gradient that does not point
anywhere. It corrects the earlier reading that the objective was "flat".

Everything downstream follows from it and every remedy has now been measured:

- growth from a seed decays from 0.810 correct to 0.138 in nine steps even from
  an ORACLE seed, because errors compound (M265);
- a beam is worth nothing, because branches are selected by the same objective
  that cannot tell them apart (M266);
- tile-by-tile lattice growth from a 144-fragment correct seed recovers 5% of
  the rest, and global assignment is worse than greedy (M273);
- the frame prior is the only absolute cue that converts, worth +0.015 SSIM and
  a doubling of placement, and it speaks about the 21.7% of components that
  touch a border (M246, M247, M250).

## The task is SOLVABLE, and the whole gap is edge quality (M304)

Handed perfectly clean fragments, the CURRENT stack places **93.45%** of them:
874 mutual edges at precision 0.974, 852 correct, largest clean component 354.
M141 had bounded this programme for months with "the pipeline on clean tiles
measures edge precision 0.746 and place_acc 0.086, still not assembling" -- but
it fed clean tiles to the LEARNED matcher, which `distort.py` itself documents
as degrading there ("on CLEAN tiles R@1 0.388 where plain MGC scores 0.913"). A
plain analytic cost on the same fragments reaches 0.974.

So 24x24 assembly is not intrinsically out of reach. The entire deficit is one
number: **159 correct edges from the deployed harvest against 852 from clean
fragments**, with the cliff at 552.

### The matcher target, derived end to end (M305)

| corruption removed | single-seam R@1 | correct edges | fragment placement |
|---:|---:|---:|---:|
| 0% | 0.070 | 51 | 0.002 |
| 60% | 0.310 | 300 | 0.001 |
| **80%** | **0.505** | **515** | **0.125** |
| 90% | — | 658 | 0.294 |
| 100% | 0.803 | 858 | 0.947 |

**R@1 must reach about 0.52.** It is 0.334. That is the number the trainer's
docstring has called "the assembly threshold" all along, now derived from this
pipeline rather than inherited.

## Restoration is closed BY BOUNDS, not by plateaus

- *denoising*: the MMSE estimate under a prior of 184320 real clean fragments
  buys **0.31 dB** over the corrupted fragment (M301). M54's plateau was the
  limit, not a failure of capacity.
- *the affine*: predictable from a fragment at R^2 **0.093** for contrast and
  **0.199** for brightness (M302) -- about a tenth of the error. It is
  recoverable only from how a fragment's level agrees with its neighbours',
  which needs the assembly the affine is blocking.
- why the noise looks worse than it is: the generator's 3x3 blur averages nine
  pixels and cuts it threefold, so the fragment sits at 23.8 dB where sigma 47
  alone would give 14.7.

## Immediate work, in order

**Read before measuring.** This session lost most of a day to seven accidental
re-derivations (M286, M290, M291). Every one would have been avoided by a grep
of EXPERIMENTS.md, FINDINGS.md and the relevant trainer docstring. The file
holds 291 entries and documents its own negative results inside the code that
produced them.

**Closed, do not re-open without a new reason:**

- *matcher capacity and training length* -- M197, "CLOSES THE SCALING LEVER":
  20000 extra steps bought +0.002 edge precision, and R@20 never moved (0.660 to
  0.666), so the shortlist is saturated and only the ordering inside it improves;
- *restorer capacity* -- M54: 2.41M parameters against the shipped 0.40M plateaus
  at R@1 0.16-0.17, "capacity is not the binding constraint";
- *`--calibrate`* (M116, same collapse to 0.0009), *`--mix`* (M65), *`--ycc`*
  (M62), *`--target-mode noisefree`* (M55), *non-local denoising* (M22, M61);
- *the harvest* -- vote threshold, pool width and purity all swept, the shipping
  13-of-18 wins each time and maximises the largest clean component (M288, M289);
- *the placement stage* -- component search, seeded growth, beam, tile-lattice
  growth and global assignment, all measured (M263 to M266, M273).

**Also closed since:** `--restore-input` -- the matcher trained AND evaluated on
restored fragments plateaus at R@1 0.295 against 0.334 on raw (M292), which
upholds M66 and M91 for the right reason at last; the descriptor DIMENSION,
against a matched control (M306); global-averaging methods as a family --
spectral embedding, spectral prior and diffusion distance all buy coarse
geometry by spending the resolution the problem needs (M293, M294, M299); and
restoration entirely, by the two bounds above.

**Where 2026-08-24 left it.** The matcher side stayed closed all day and the
picture changed anyway, because the quantity everything was averaged over turned
out to be bimodal.

### The mean described no board (M342, M343)

The oracle-on-pool arm, taken from 12 boards to 24 and then run with the frame
prior ON, which every earlier entry had switched OFF for comparability:

| | prior off | prior on |
|---|---:|---:|
| boards placing more than half | 0.292 | **0.417** |
| boards at essentially zero | 0.542 | **0.250** |
| mean placement | 0.2455 | **0.3860** |
| **median placement** | **0.0069** | **0.4531** |

Nine boards of twenty-four are nearly solved and thirteen are at zero, with
almost nothing between -- so `0.4029` from M314 described no board at all, and
was besides measured on twelve, which this project's own rule says settles
nothing.

**Half the failures are ORIGIN failures, not assembly failures.** Boards
carrying 321, 215, 180, 145, 136 coherent fragments were scoring zero because
the block stood in the wrong place. The frame prior lifts six of the thirteen
above 0.2 -- 321 fragments go 0.0104 to 0.7413 -- and BREAKS two that worked
without it, 0.3941 to 0.0069 and 0.5243 to 0.0000.

**Read every earlier entry with this correction**: M314, M316, M321, M336 and
M342 all ran with the prior off and all understate the ceiling.

### What follows, in order of expected return

1. **A solvability diagnostic.** Tell a board that assembled from one that did
   not, without the answer. This is the first task where the project's
   subtractive signals are strong on their merits rather than weak as builders:
   M335 measured 5.6 sd separating a long correct assembly from a wrong one,
   M336's component health picks a correct core 0.836 of the time against 0.629
   by size, M334's agreement check runs 7.5 to 1.
2. **The frame prior as a per-board bet**, since it rescues six boards and
   breaks two. Same diagnostic, different consumer.
3. **A fallback arm for the hopeless boards.** Composition belongs here and only
   here -- not as the main strategy, which is what the user rightly refused, but
   as what to ship on the quarter to half of boards that do not assemble.

Arithmetic behind the priority: a solved board renders at SSIM ~0.49 against our
current 0.06 to 0.10, so converting 40% of boards is worth about +0.17 on the
mean, which is 0.28 -> 0.45 with no better matcher at all.

### Live, and unfinished when the day ended

- **The strip energy** (M340, M341). An energy trained against its OWN optimum
  is the first objective here ever to rank the truth above the arrangement the
  seam cost prefers -- 0.583 of boards where the seam cost manages 0.000 -- and
  it retracts M339's claim that no such objective can exist. It does not yet
  convert: contrastive rounds overfit at 240 boards, the loss reaching zero, and
  generate-and-rerank is a wash against the seam cost. `scratchpad/cd_scale.py`
  was re-running it at 2400 training boards and 40 evaluation boards and was
  killed at 1200 boards of pool building. That run is the open question.
- **A submission probe** across alphas 1.0 to 0.15 with `--place anneal
  --frame 1.0`, killed before its validation printed. The layout is the
  expensive part and re-rendering is nearly free, so one pass gives the whole
  front; the operating point on it is the user's call, not a measurement.

### The target, corrected from measurement (M314, M316)

M268's "552 correct edges" was derived by sampling the true adjacency set
UNIFORMLY, and our true edges are not uniform -- a matcher succeeds where the
photograph has texture, so they cluster. Clustering helps. Handed every true
edge our own roster already proposes, 432 a board, the placement stage reaches
**fragment placement 0.4029** with a largest coherent component of 194; the same
432 drawn uniformly reach only 0.2661 and a component of 105.

So the evidence ALREADY ON DISK is worth placement 0.40, and the shipping
harvest extracts **0.0012** from it. The target is not a better matcher and not
552 edges. It is **about 430 edges at precision 0.97**, which is where M316 puts
the collapse -- 0.97 is worth placement 0.21, 0.95 is worth 0.06.

### And why that target is out of reach anyway (M317 to M321)

| selection route | best point | placement |
|---|---:|---:|
| the shipping vote | 208 edges at 0.817 | 0.0022 |
| learned per-edge, 60 boards, segment features | 430 at 0.586 | 0.0014 |
| " at its clean end | 100 at 0.951 | 0.0019 |
| consistency filter as a search | 537 at 0.503 | 0.0029 |
| corroborated island merges | 470 at 0.514 | 0.0009 |
| **a perfect selector on the same pool** | **432 at 1.000** | **0.4029** |

Precision falls smoothly from 0.951 at a hundred edges to 0.586 at four hundred
and thirty, so no threshold hides a clean set. The independent evidence found in
M311 does rank above every vote count as a feature and is worth two hundredths.
Coherent coverage can be doubled, 0.385 to 0.494, by demanding a second contact
before two islands fuse -- and it does not convert.

**The payoff tracks the largest coherent block and nothing else.** Blocks of
19.6, 27.2 and 37.7 all pay about 0.002; a block of 194 pays 0.40. That is M263
from the other end: placing many medium components against each other is exactly
the coordination the objective cannot do, so the assembly is worth something only
once one component is large enough to anchor the board by itself.

### What is left

The composition front, which is monotone -- every unit of visible texture costs
score at a constant rate because our texture is in the wrong place (M145, M165).
The operating point on it is a judgement about what a submission should look
like, not a measurement, and the standing rule forbids buying score with
smoothness.

**Measurement rule, paid for twice today:** two runs of the SAME configuration
differ by 0.028 in R@1 at step 2500 (M306). A single-run A/B cannot see anything
smaller than about 0.03, and a control must be produced alongside its arm, never
borrowed from an earlier run.

**Genuinely new this session, and unexploited:** the border detector read off
Sinkhorn's slack column (M246, shipped, +0.015 SSIM and double the placement);
loop-closure merging at precision 0.938, which overturns M180 and M233 but fires
2.3 times a board (M248); and the consistency filter with the board bound, which
moves the requirement from precision everywhere to precision in the prefix
(M270) and does nothing on our real ordering. All three are mechanisms without
enough volume behind them yet.

## Rules this file exists to enforce

- **Rank by assembly, not SSIM.** They disagree: the frame weight that maximises
  SSIM (0.2) is two and a half times worse at placement than the one that
  maximises placement (1.0). Report component-absolute, component-relation and
  fragment placement; keep SSIM as the control. The organisers check placement
  by hand.
- **Twelve boards cannot settle anything.** Three effects looked real at twelve
  and vanished or reversed at twenty-four this week (M256, M258, the staged
  freeze).
- **Watch for arms that are too GOOD.** A degenerate tie in the Hungarian fill
  returned the identity permutation, which in this harness IS the answer, and
  produced a perfectly monotone "fewer components is better" curve that read
  exactly like a finding (M264). It was caught because an arm scored 1.0000.
- **`is_clean` is all-or-nothing** and stops meaning anything once components
  pass a few dozen fragments; use tile-level accuracy there (M270).
- **The restorers take [0,255]**, not [0,1]. Feeding them [0,1] returns noise
  and silently poisons whatever is built on it (M279).
