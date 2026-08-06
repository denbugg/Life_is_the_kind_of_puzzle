# FINDINGS.md — shared research board for pazzle-solution-20260806

## Champion

- exp_id: 10, corrected raw-ranker buddies with a solve-SSIM-calibrated edge budget of 96
- gate v1: mean solve-only SSIM 0.1089045302 (+0.0011076834 versus 512); final 0.2024201409 (+0.0051042196)
- untouched gate v2: mean solve-only SSIM 0.1202495180 (+0.0019082922); final 0.2381572666 (+0.0006808942)
- equal-size two-gate mean delta: +0.0015079878 solve-only SSIM and +0.0028925569 final SSIM
- immutable roots: v1 `ee3d74662f5326fbd1069763fd7b96dc3adb41bde0117cba1d78ff067c6bf23d`; v2 `7a5a5e68779a25fd8dc882062345a3e7b5e9e555da51dee97c5b5ca3e3558134`
- rotation: fixed orientation, permutation only
- uncertainty: each 24-scene bootstrap CI still includes zero; the keep decision follows the predeclared positive-mean rule and the repeated direction, not a claim of conventional statistical significance

## What works

| gen | exp_id | change | metric | delta | note |
|---|---|---|---:|---:|---|
| 0 | 0 | corrected shuffled-ID boundary bug | neighbour 0.1647 on historical gate | +0.0261 vs buggy path | historical, not immutable-gate verified |
| 0 | - | 18 exact source overrides | 18 exact test replacements | - | always preserve in final artifact |
| 0 | 0 | byte-frozen source-disjoint end-to-end gate | 24 scenes | - | actual target/tile/permutation bytes and checkpoint/code hashes verified |
| 0 | 9 | reduce buddies component-construction budget from 512 to 96 | solve SSIM 0.108905; final SSIM 0.202420 | +0.001108 solve; +0.005104 final | selected on 8 scenes, positive on separate 4 scenes, then positive on immutable 24-scene gate |
| 0 | 10 | repeat budget 96 unchanged on a new source/corruption gate | solve SSIM 0.120250; final SSIM 0.238157 | +0.001908 solve; +0.000681 final | 24 new source groups; zero overlap with v1; no sweep |

## What does not work

- sampled pair accuracy as the optimization target: 0.477 did not transfer to placement/SSIM
- generic TTA, structural auxiliary fine-tune, posterior weighting, simple flow/QAP/Sinkhorn, beam, GA, standalone GNN/Siamese/path/symbolic solvers
- direct z-score fusion is unverified end-to-end and was selected on its reporting scenes
- rotation search: empirically rejected; Type-1 placement only
- direct I21 spatial row-z fusion: edge R1 improved by 0.009133, but neighbour fell by 0.001359 and solve SSIM fell by 0.000746 (95% bootstrap CI [-0.003669, 0.002474])
- reciprocal spatial rank transplant: best calibration solve-SSIM delta was +0.003725, but selected physical-pair precision was only 0.265625 versus the predeclared 0.85 gate; confirmation was not opened
- atomic two-side growth: at precision >=0.95 mean tile coverage was at most 0.000868; at coverage >=0.15 precision was at most 0.110132; the hypothesis failed closed
- all nine RGB/Lab/MGC depth-0/1/2 rank donors: each reduced raw edge R@1 on calibration; no confirmation was opened
- I21-only cross-stage packing: packing edge R@1 rose by 0.013927, but sealed confirmation solve SSIM fell by 0.000129
- denoise-before-scoring: the ideal clean-tile oracle increased candidate recall and local edge/neighbour metrics, yet reduced solve-only SSIM by 0.007070 and final SSIM by 0.016292 versus the reproducible raw path; only 1/8 final wins and worst final delta -0.041554, so all predeclared E12 headroom checks failed
- label-free Rank96/Rank512 Lab selector: positive on the 48-scene untouched E11 gate, but final delta +0.000623 missed the strict +0.001 promotion threshold; solve delta was +0.000225 and the candidate was rejected without a resweep
- whole-board toroidal Lab cut: RR96 averaged +0.001688 solve and +0.002975 final, but only 2/8 final wins and worst -0.017817 failed every promotion check; CC96 became worse, so the global-frame defect is not one common cyclic row/column offset
- CC192 clean-coverage oracle: 192 selected claims per scene achieved mean precision 0.957031 and component coverage 0.472656, and neighbour accuracy rose by 0.141644 on all 8/8 scenes; solve SSIM still fell by 0.008794, final by 0.014433, placement by 0.001085, and worst final delta was -0.053652, proving that more high-purity islands alone do not repair the current global frame/packing path
- CC96/CC192 direct two-seam frame consensus: CC96 precision 0.983073 and coverage 0.268012 passed, but only 3 eligible same-pair/same-offset hypotheses existed across 8 scenes and mean relation-supported coverage was 0.003689 versus 0.15; all 3 were true, so the formulation failed from combinatorial scarcity before decoder/NLM rather than noisy evidence
- exact clean-tile rendering on the unchanged RR96 board: even perfect faithful pixel restoration reduced final SSIM by 0.015296 on average, won only 1/8 and lost as much as 0.035599; with globally wrong placement, sharper wrong-cell content is worse than NLM smoothing, so post-assembly diffusion is not funded yet
- quotienting global translation in the CC192 dense-top8 single-edge beam: despite exactly one root at `(0,0)` and no absolute frame, scene 10 still reached the fixed `500000` distinct proposal cap after 32 completed rounds; therefore the branching itself, not only origin copies, is excessive and the exact beam is closed without resweep
- top-8 triangle-supported signed-potential DSU: fixed-cost inference completed all 8 scenes, but selected-cluster exact pose coverage was only 0.036024, relative-pose precision 0.263673, accepted-relation precision 0.132524, seam precision 0.232264, and selected cycle ratio 0; correlated false single-edge paths corroborate false offsets, so longer hand-composed cycles over the same graph are not justified
- raw CC96 nontrivial-anchor top-8 relation pool: even an oracle verifier connected only 0.039497 mean / 0.019097 worst coverage (22.75 tiles mean) despite 616 true hypotheses and 494..557 pure tiles per scene; the dominant recall hole is that 420..437 singleton tiles never emit, so this exact pool is insufficient before any learned verifier is trained

## Open directions

- preserve the completed, independently verified Rank96 v1 ZIP as the fallback artifact
- preserve Rank96 v1 as production after E11 failed its predeclared promotion threshold
- E18/E19 show exhaustive absolute/relative branching is intractable, while E20 shows bounded two-hop support is fast but inaccurate; preserve E17's rigid islands and replace hand-composed top-8 relations with a learned multi-tile contextual relation scorer that must pass a precision/coverage prerequisite before any absolute embedding
- E21 shows the first verifier pool itself is too disconnected; the next ceiling must change candidate generation so singleton tiles emit fixed raw candidates, then reapply the same oracle-connectivity prerequisite before GPU work
- E15 still rules out requiring two direct seams between the same small-island pair, and E16 rules out faithful post-assembly diffusion spending until that global decoder materially improves placement

## Qualitative production audit

- visually inspected nine generated test scenes spanning ordinary Rank96 outputs and one exact verified override; no test target labels were used
- upright orientation is consistent; observed failures are not rotations
- Rank96 often recovers coherent local chains and coarse semantic/colour regions, but places those islands in the wrong global rows/columns
- low-texture backgrounds form large plausible components that can displace detailed foreground islands
- fixed NLM reduces corruption visibility but cannot repair global permutation errors; restoration is therefore secondary to board selection and multi-seam packing
- the exact verified override is visually coherent end to end, supporting preservation of the strict override path

## E14 visual decoder audit

- rendered all eight pinned E12 scenes as `target | RR96 | CC96 | CC192` panels after the identical NLM10 tail; the read-only manifest is `E:/pazzle_work/visual_audit_e14/manifest.json`
- CC192 visibly creates longer coherent strips and semantic micro-islands (faces, clothing, screens, table edges, balloons), matching its large neighbour gain, but those islands occupy unrelated rows/columns and almost never the correct absolute cells
- the worst scene 15 is especially diagnostic: CC192 reaches neighbour `0.240` with placement `0.000`, yet moves recognizable people/balloon fragments into the wrong frame regions and loses `0.05365` final SSIM versus RR96
- scene 12 likewise contains long monitor/table strips under CC192 while exact placement remains zero; scene 16 is the lone final-SSIM win, but still shows a misplaced wide horizontal foreground band rather than a recovered global composition
- repetitive dark/background regions still contain visibly plausible but wrong internal joins, so E15 deliberately keeps the purer CC96 geometry rigid and uses CC192 only as two-seam translation evidence

## E20 visual relation audit

- rendered all eight E20 sparse relative clusters as raw-corrupted and tile-wise NLM10 previews; no target/permutation labels affected placement, no absolute origin was chosen, missing cells stayed neutral, and every tile remained upright
- panel directory: `E:/pazzle_work/visual_audit_e20`; manifest SHA256 `213479011c3e837a82819d21fce474a1d334b8901377da5aa3ea191721cc5a96`
- scene 13 is the strongest visible island (`74` tiles, exact relative-pose precision `0.514`): several rails/strips align, but false inter-island offsets remain
- scene 15 is largest (`117` tiles) yet only `0.308` pose precision; scene 10 has `104` tiles at `0.125`, visibly demonstrating that size/neural support alone rewards long false chains
- NLM makes content easier to inspect but leaves every geometric error intact, reinforcing that restoration cannot substitute for a relation verifier

## Next levers

- seed-conditioned sparse message passing
- direct multi-neighbour context scorer
- smaller corruption-invariant 576-piece global transformer
- degradation-conditioned exact-SSIM restoration only after placement succeeds; never use independently restored tiles to drive current affinity/ranker scoring

## Stagnation / lever log

- generation 0: budget 96 repeated a positive paired mean on a second untouched gate and is the final production default; I21, reciprocal/classical transplant, cross-stage I21 packing, and atomic two-side growth were falsified
- generation 1: E12 killed denoising for matching, E11 missed promotion despite a small positive mean, E13 falsified a single cyclic-origin correction, E14 showed that even 95.7%-precision clean edges covering 47.3% of tiles do not transfer through the existing packer, E15 showed that two direct confirming seams are too sparse (3 total), and E16 showed perfect faithful output restoration loses to NLM on 7/8 with the current board; E17 passed the rigid-island prerequisite, E18/E19 killed absolute and translation-quotiented beams at `500000` proposals, E20 made inference fixed-cost but exposed very low relation precision and zero selected cycles, and E21 proved the first raw verifier pool has only 3.95% oracle connectivity because singleton emitters are absent; the remaining lever is higher-recall fixed candidate generation followed by learned multi-tile contextual relation evidence, not larger beams, longer cycles, or diffusion
