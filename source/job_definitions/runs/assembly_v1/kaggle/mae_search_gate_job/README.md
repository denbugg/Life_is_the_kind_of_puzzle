# MAE-guided population-search falsification gate

This isolated Kaggle job answers a narrow question: can the frozen
`facebook/vit-mae-base` energy rank **competitive, seam-preserving global
mutations** of the authoritative boundary-QAP solution well enough to improve
real16 denoised SSIM?

It is intentionally a cheap falsification gate, not an aggressive production
solver. The preceding MAE experiment had strong correlation only when clearly
weak component layouts were mixed into the pool. Among QAP-only or
near-baseline layouts, rank signal was approximately random. This job therefore
uses at most 192 layouts per source and must demonstrate signal inside a
competitive set before the direction can be promoted.

The script is self-contained in `run_mae_search_gate.py`; it does not import
supplementary local Python modules. The mounted code dataset supplies only the
authoritative JSON layout artifact, while the runtime dataset supplies the
frozen denoiser checkpoint.

## Exact starting point

Every source starts from `qap_softcycle_l1_k8` in the authoritative
`qap_l1w4_boundary_real16.json` report. The runner requires the exact input-only
layout-manifest SHA-256:

```text
2a7cc81a95ea03fe339f37032dcb29e5139e386d402e8d1522e7567b94ba4020
```

It also verifies the complete boundary-QAP configuration:

```text
score=l1w4, iterations=25, restarts=2, boundary_weight=0.05,
initial_weight=0.75, noisy_components=3, noise_scale=1.0,
refine_swaps=8, seed=softcycle_l1_k8
```

Raw and denoised variants must contain the same valid 576-entry permutation for
all 16 fixed sources. In Phase B the job reruns the promoted denoiser and must
reproduce the recorded mean denoised SSIM `0.1828199150` within `1e-5`; otherwise
the gate fails regardless of search results.

## Bounded input-only search

Default budget per source:

- 64-candidate initial population;
- Pareto beam of 8 candidates;
- one expansion to at most 192 unique candidates total;
- four deterministic MAE masks;
- candidate batch 8, producing MAE forward batches of 32;
- source-level data parallelism across one or two GPUs.

The baseline is always present and cannot be removed. Mutations preserve an
exact permutation and cover genuinely nonlocal moves:

- same-size distant rectangle swaps;
- explicit 4x4 and 8x8 block swaps;
- full row-band and column-band swaps;
- cyclic shifts inside blocks and across full row/column bands;
- cut/insert component translations with intervening-strip displacement;
- three-block translation cycles;
- three-block destroy-and-repair, choosing the best non-identity reassignment
  by input-only seam cost.

All RNG state is derived from the fixed global seed plus the source filename.
Candidate identity is the SHA-256 of its int32 permutation. Duplicate and
seam-rejected mutations are counted by operator.

## Objective and conservative selection

The promoted TileNAF checkpoint first denoises all 576 shuffled tiles without
changing their slots. Its SHA-256 is pinned to
`77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734`.

For each source, exact directional denoised RGB L1w4 matrices are computed from
the four-pixel tile borders. Search uses a two-level guard:

- generation rejects layouts whose mean L1w4 seam cost is more than 4% above
  the boundary-QAP baseline;
- final replacement requires seam loss no greater than 2%.

The frozen MAE checkpoint is `facebook/vit-mae-base` at revision
`25b184bea5538bf5c4c852c79d221195fdd2778d`. The script pins
`transformers==4.57.1`, supplies four identical fixed noise/mask tensors to
every candidate and GPU, computes exact per-sample masked-patch loss, and loads
the model with `low_cpu_mem_usage=False` to avoid the thread-unsafe meta-device
failure seen in the prior two-T4 run.

MAE may replace the baseline only when every input-only condition passes:

- mean MAE error improves by at least 0.1% relative;
- the candidate beats baseline on at least 3/4 fixed masks;
- denoised L1w4 seam loss is at most 2%.

Otherwise the authoritative baseline is retained for that source. The beam
contains the baseline plus MAE/seam Pareto candidates so search cannot collapse
onto a single low-MAE but seam-damaged lineage.

## Strict anti-leakage boundary

Phase A performs only:

1. authoritative layout discovery (real evaluation fields are discarded while
   decoding the report);
2. raw input loading and tile denoising;
3. L1w4 mutation filtering, MAE scoring, beam expansion and conservative
   selection;
4. writing every searched layout, every energy and every selection to
   `/kaggle/working/mae_search_frozen.json`;
5. hashing that frozen artifact and emitting
   `mae_search_layouts_and_energies_frozen`.

Only Phase B may open `train/targets` or reread the report's recorded SSIM. It
reconstructs denoised predictions from the **reloaded frozen permutations**.
The target oracle and the competitive-set definition are explicitly post-hoc
diagnostics and never feed candidate generation or selection.

## Falsification metrics and promotion rule

The report includes baseline, frozen input-only selection, and post-hoc oracle
SSIM for every source. It also reports MAE rank signal over:

- all searched seam-guarded candidates;
- the target-defined competitive set: candidates no worse than 0.005 denoised
  SSIM below that source's boundary-QAP baseline.

Promotion requires every condition:

```text
authoritative baseline reproduced
all 16 sources evaluable
mean selected-minus-baseline denoised SSIM >= +0.010
selected win rate >= 11/16 (0.6875)
mean selected seam loss <= 1%
maximum per-source selected seam loss <= 2%
mean per-source competitive Spearman >= 0.20
micro competitive pairwise accuracy >= 0.60
```

A failure means the MAE-guided search should not be expanded to a larger budget
with this mutation family. A pass justifies a separate confirmation run; it is
not automatic authorization to change the production solver.

## Risks and interpretation

- MAE reconstruction error is not an image likelihood. Smooth or semantically
  generic wrong arrangements may receive a low error.
- The 480x480 mosaic is resized to 224x224; a 20-pixel puzzle tile becomes about
  9.3 pixels, smaller than the model's 16-pixel patch. MAE is mainly a coarse
  global signal, not a seam detector.
- The previously high aggregate correlation was confounded by obviously weak
  candidates. Competitive Spearman/pairwise metrics are the decisive evidence
  here.
- A strict seam guard prevents MAE exploitation but can also reject a correct
  global relocation that temporarily damages pairwise seams.
- The bounded mutation family may not contain the required correction. The
  post-hoc oracle quantifies that candidate-recall ceiling but must never be
  mistaken for an achievable selector.
- Four fixed masks reduce variance but do not eliminate it. Per-mask errors and
  the 3/4 consensus decision are retained in the frozen artifact.
- Two-GPU execution uses one independent MAE replica per device. It is
  source-level data parallelism, not model parallelism. Actual Kaggle allocation
  and runtime remain quota/environment dependent.

References: [MAE paper](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html),
[official MAE repository](https://github.com/facebookresearch/mae), and
[Hugging Face ViT-MAE documentation](https://huggingface.co/docs/transformers/model_doc/vit_mae).

## Lightweight local validation

These commands do not download models or open puzzle images:

```bash
/Users/rusyalain/Documents/test/.conda/bin/python \
  run_mae_search_gate.py --validate-config-only

/Users/rusyalain/Documents/test/.conda/bin/python \
  run_mae_search_gate.py --synthetic-test
```

The synthetic test exercises every mutation operator, permutation validity,
exact 4x4/8x8 moves, bounded population/beam construction, conservative
selection, Spearman and pairwise metrics.

This task prepares the job only. It does not push or launch it. After refreshing
the code dataset with authoritative v2, the coordinator may explicitly request
T4 allocation:

```bash
conda run -p /Users/rusyalain/Documents/test/.conda \
  kaggle kernels push -p \
  /Users/rusyalain/Documents/test/runs/assembly_v1/kaggle/mae_search_gate_job \
  --accelerator NvidiaTeslaT4
```
