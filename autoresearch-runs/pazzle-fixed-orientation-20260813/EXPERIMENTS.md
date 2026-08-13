R0 | raw1 | PASS | 8 source-disjoint DEV images | overall R@1=0.098902 R@5=0.215014 R@20=0.352468 worst-image-R@20=0.233696 | 6.20s
R1 | multiband hand-crafted | DROP | same 8 DEV images | overall R@20=0.259964, delta=-0.092504 vs R0 | mechanism refuted: untrained multi-band cosine fusion amplifies distortion.
R2 | learned directional Siamese, 200 steps | PARTIAL | 8 source-disjoint DEV images | R@1=0.059047 R@5=0.184556 R@20=0.397758; R@20 delta=+0.045290 vs R0, but r1/r5 decline and gate fails (required r1>=0.25, neighbour>=0.18).
R3 | listwise hard-negative candidate ranker, 200 steps | KEEP for candidate union | held-out 8 images: coverage_all_true=0.688179, R@1=0.113281 on selected rows, all-true proxy R@1=0.077958/R@5=0.219088, reciprocal_true_mutual_candidate_coverage=0.898438, reciprocal_precision=0.406977. It does not pass final selection gates.
G1a | coarse 6x6 set prior, 200 steps | INCONCLUSIVE/DROP short run | held-out 8 images: macro_r1=0.0336, macro_r3=0.0940, macro_hungarian_acc=0.0295, top64_coverage=0.114. Near random macrocell classification; no fusion permitted from this checkpoint.
G1b | coarse 6x6 set prior, 1200 steps | DROP | held-out 8 images: macro_r1=0.038845, macro_r3=0.105686, macro_hungarian_acc=0.034288, top64_coverage=0.131510. Only marginally above chance (1/36=0.027778; 64/576=0.111111); insufficient for fusion.
F1 | direct-pose fusion, 200 steps | DROP as selector | held-out 8 images: mutual direct candidate coverage=0.912639 but reciprocal inverse precision=0.033840, coverage=0.171099, direct AP=0.069754. No assignment permitted.
F2/F2b | frozen PairwiseNet + F1 heuristic fusion | PARTIAL/DROP for assignment | n=1 held-out: top1 direct precision=0.4184 but all-direct recall=0.1091; top64 precision=0.0401 with recall=0.6689. No reported operating point meets planned precision>=0.25 and recall>=0.15.

| C1 | 2x2 cycle-support coverage gate (2 DEV, R3 union top64) | REJECTED pre-implementation | Exact true oriented C4 coverage is 1.51% at 128 and 2.93% at 512 motifs/anchor; too sparse to cause the required +5 pp global top-4 precision gain. |

| R2L | Directional Siamese scale, 800 steps (best step 600, 8 DEV) | PARTIAL / retain checkpoint | R@20=49.88% versus prior R2 39.78% (+10.10 pp); R@1=9.81%, b384-neighbour=7.39%; strict gate failed. |

| U1 | R3 top64 ∪ R2L step-600 directional top8/direction (8 DEV) | KEPT candidate source | Direct coverage 69.34%→73.95% (+4.61 pp), edges/tile 81.69→90.80 (+11.16%); coverage and density gate passed. |

| U2 | U1 union + frozen pair/pose fusion (1 DEV smoke) | REJECTED early | top-4 direct precision=18.27%, recall=19.07%; misses pre-registered 35%/20% gate, so no 4-board run. |

| D1 | Frozen matchden seam diagnostic (2 DEV) | REJECTED | Denoising improves tile L1 0.07790→0.07104 but border seam R1(all) 13.7%→13.5%, below raw and far below +5 pp criterion. |

| R3L | 800-step listwise candidate ranker scale | STOPPED (time-bounded, inconclusive) | After ~10 minutes, no first training step or metric; RAM 12.1 GB / committed 20.6 GB. Avoid this full-bag configuration. |

| P1S | Hard-negative micro-cache (n=4, K=16, 5-min budget) | REJECTED timing gate | No cache emitted in 5m29s; pairwise candidate mining overhead remains unsuitable even at micro scale. |

| P2 | Posterior seam marginalization reuse (1 DEV, 192 rows) | REJECTED early | Gate fail: raw R1=17.19%, best posterior/hybrid R1≤16.67%; raw R5=38.02%, best 42.19% but R1/brier checks fail. |

| E2 | Streaming generative-contrastive continuation predictor | REJECTED timing gate | First step ran but took 26.57 s/it at bs=1, rows=16+16; 100-step validation would take ~44 min, outside rapid-evidence budget. |

| OH1 | Online hard-negative PairwiseNet, 200 steps | CHECKPOINT RETAINED pending fusion | First step 8.77s then 0.20–0.48s/it; best held-out online-hard acc 54.69% at step 50, no board-wide cache. |

| OH2 | OH1 best PairwiseNet in U1 fusion smoke (1 DEV) | REJECTED downstream gate | top-1 precision improved 25.52%→28.82%, but top-4=18.36%, recall=19.16%; misses 30%/20% gate. |

| OH3 | U1-aligned online hard-negative PairwiseNet, 200 steps | CHECKPOINT RETAINED pending fusion | First step 11.4s then 0.79–1.29s/it; best U1-row online-hard acc 32.81% at step 150, covered-row fraction 62.77%. |

| OH4 | OH3 best scorer in U1 fusion smoke (1 DEV) | REJECTED downstream gate | top-1 precision=30.38%, but top-4=18.32% and recall=19.11%, below 30%/20% gate. |

| OH5 | U1-aligned full-row (M=64) online hard PairwiseNet, 200 steps | CHECKPOINT RETAINED pending fusion | First step 13.97s then 0.99–1.55s/it; best U1-row online-hard acc 32.81% at step 200. |

| OH6 | OH5 full-row scorer in U1 fusion smoke (1 DEV) | REJECTED downstream gate | top-1=29.17%, top-4=17.93%, recall=18.70%; worse than OH4 top-4. |

| Q1 | Scene-conditioned confidence calibration (4 fit / 2 cal / 2 heldout) | REJECTED gate | No calibrated threshold met required confidence/coverage; best label-free reciprocal+both-affinity rule: 33.33% precision but 14.06% row coverage, only 4.69% exact-edge coverage. |

| G2 | U1 sparse 2×2 growing-consensus diagnostic (2 DEV) | REJECTED pre-gate | Best prefix K=8: direct precision 3.19%→3.56% (1.12×, required 2×); recall 6.66%→1.20% (required >=10%). K=16/32 similarly fail. |

| PN1 | U1-aligned online hard PairwiseNet with per-tile photometric normalization, 200 steps | CHECKPOINT RETAINED pending normalized fusion | First step 11.26s then 0.78–1.27s/it; best normalized held-out U1-row hard accuracy 18.75% at step 150. |

| PN2 | PN1 normalized PairwiseNet in matched U1 fusion smoke (1 DEV) | REJECTED downstream gate | top-4 direct precision=15.15%, recall=15.81%; below OH4 and far below 23.32%/20% gate. |
