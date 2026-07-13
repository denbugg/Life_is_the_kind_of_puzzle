# Neural tile-solver upgrade worklog

Status date: 2026-07-11. This is a live research log, not a promoted release report.

## Frozen reference

- Best known leaderboard score reported by the user: `0.203`.
- Promoted local real16 solver SSIM: `0.18281991502795386`.
- Promoted path: HBT `l1` soft-cycle seed -> C1/HBT rank fusion weight 4 -> directional QAP, 25 iterations, 2 restarts, boundary weight 0.05.
- Promoted artifacts remain under `runs/assembly_v1/kaggle/final_qap_submission_output/v1/` and are not overwritten by these pilots.

## Classical and stochastic refinements

### QAP fusion matrix

The lighter `qap_w1_b0.05_i25` configuration is the best fixed local fallback found in the new matrix:

- real16 SSIM: `0.184866179`;
- delta versus promoted weight-4 QAP: `+0.002046264`;
- source wins/losses: `13/3`;
- bootstrap 95% interval: `[+0.000174818, +0.003975691]`.

This did not meet the strict `+0.005` promotion threshold. A 700-image submission was intentionally deferred while neural branches were being tested.

Artifacts:

- `runs/assembly_v1/kaggle/upgrade_matrix_203_output/v1/upgrade_matrix_203.json`
- report SHA256: `27581dd51bf54b029dc7ee6b021f9b107fb1dae06947aa5d7d6c25e0223ee8a5`
- frozen layouts SHA256: `289d1a2f5c5699dc87cdca865e914c3b85618ad90ee563e5731ffe8984cad547`

### Protected annealing

Closed after an exact gate:

- weight-4 QAP layouts changed on `0/8` sources across protection strengths;
- pure-HBT annealing changed `4/8`, but SSIM delta was `-0.000040564`;
- the older generic annealer was also below QAP (`0.170495` versus `0.182820`).

Artifact: `runs/assembly_v1/kaggle/anneal_upgrade_gate_output/v1/protected_anneal_exact_gate.json`, SHA256 `a7e9f879c081fd09a576e9adde9015518fe6c2e19d22c515d36f3fb018900b93`.

## ViT-Sinkhorn absolute assignment pilot

Decision: **closed for no useful signal**.

Kaggle kernel: `pasha883/vsos-vit-sinkhorn-pilot-t4x2`, successful model run in version 4 on two Tesla T4 GPUs. Earlier versions were infrastructure/AMP-preflight attempts and consumed negligible compute.

Configuration:

- 5,321,571 trainable parameters;
- 256 whole training sources, 3 epochs;
- raw plus restored tile inputs;
- no synthetic truth-derived QAP prior;
- independent selection and libjpeg holdout gates.

Results:

| Split | Model SSIM | Classical SSIM | Delta | Model adjacency | Position accuracy |
|---|---:|---:|---:|---:|---:|
| Selection | 0.1759015964 | 0.2014403448 | -0.0255387484 | 0.0067934783 | 0.0021701389 |
| Holdout | 0.1960045233 | 0.2226000581 | -0.0265955348 | 0.0069067029 | 0.0015190972 |

AMP was stable after the corrected loss-scale policy: initial/final scale 1024 and zero skipped updates. The failure is therefore scientific, not an infrastructure artifact. The real pilot took about 310 seconds on 2xT4.

Artifacts:

- report: `runs/assembly_v1/kaggle/vit_sinkhorn_pilot_output/v4_reports/vit_sinkhorn_pilot/vit_sinkhorn_report.json`
- report SHA256: `1541bace8c67ba0f12f1e0a0ab31420b294ef261030316dadf9e506fa1d59e1f`
- wrapper: `runs/assembly_v1/kaggle/vit_sinkhorn_pilot_output/v4_reports/vit_sinkhorn_pilot_wrapper.json`
- kernel log: `runs/assembly_v1/kaggle/vit_sinkhorn_pilot_output/v4_reports/vsos-vit-sinkhorn-pilot-t4x2.log`

## Pair Transformer

Current state: all core/staging blockers are closed, the final independent audit is APPROVE, and Kaggle pilot version 2 is running on two explicit Tesla T4 GPUs.

Architecture: approximately 26.5M parameters, full raw/restored tile CNN, explicit side-band tokens, eight joint Transformer layers, sparse HBT/current-layout candidate graph, iterative neural rescoring plus QAP.

Closed review blockers:

- evaluation sources must be disjoint from the upstream denoiser and HBT training sets;
- neural and non-neural controls must receive exactly the same QAP compute budget and must also beat the fixed promoted i25 envelope;
- DDP evaluation cannot leave one rank waiting through a long rank-0-only gate;
- training requires real resume state and runtime/memory telemetry;
- one query must share the same augmentation across every candidate in a ranking group;
- inference should cache the shared tile CNN instead of re-encoding roughly 58k sparse pairs per image.

No target-selected candidates or direct target leakage were found.

Final local core has 26,507,009 trainable parameters and passes 18 pair-transformer tests plus a finite default-model microstep. The staged runner pins the exact overlay core hashes, performs a real two-T4 fp16 forward/backward preflight, and requests a 512-source x 3-epoch pilot.

The independent staging audit found no ML/split/equal-budget defects, but correctly blocked the first launch attempt because adversarial garbage checkpoints and inconsistent report gates were accepted, command telemetry was lost on subprocess failure, wrapper writes were non-atomic, the core `latest` checkpoint was not crash-safe, and transitive base/runtime hashes were not pinned. All were closed. A second semantic audit then exposed structurally tagged but unusable model/resume states; strict model/config/optimizer/scaler/scheduler/training-state validation and semantic fallback were added. The final adversarial matrix passed 23/23.

Final core SHA256 values are model `99e14f3741528cf277a5b10fb0a01fac761debc5370fd55fd81f6235a0ae303b`, trainer `37f4585ff42eeeaf31a8142f2068d86debe1bc169358cc2f21277fd7aa6f3473`, and tests `2a66d5b588b43f8d56b12b4faffeec262ddef746eabf8dab0ece0049e8343b7d`; overlay ZIP SHA256 is `f9572a559fd3d536f6a01a51dd46333e55a9d1cc5c1ea53c49ffcb7152dbc6f4`.

Kaggle kernel `pasha883/vsos-pair-transformer-pilot-t4x2` version 1 failed before any GPU work because Kaggle's live dataset mount convention is `/kaggle/input/datasets/pasha883/<slug>` rather than the old short path. The wrapper remained fail-closed. The mount-only patch passed 22 tests and an independent review; runner SHA256 is `252ee71f62afb448efc0ba5698c9e4568b6d3c6b3cb26d3ab8295ddbf0eca9b7`.

Version 2 passed two-T4 sm75 fp16 forward/backward, 22 tests and the actual-asset smoke. Two complete 512-source epochs processed 1,966,080 ranking pairs each. Quick exact recall was consistently worse than HBT: epoch 1 `0.188406` versus `0.199275` (delta `-0.010870`), epoch 2 `0.187047` (delta `-0.012228`). Training loss moved only from 3.77838 to 3.72544. During epoch 3 the dynamic scaler reached the bounded maximum of four skipped updates and the fifth non-finite update triggered the intended fail-closed abort after source 192/256. The route therefore has neither stable full training nor a positive quick signal; it must not be restarted from scratch.

Crash-safe artifacts survived. Best epoch-1 checkpoint SHA256 is `3c76213fc9ccb960cb7d3171584232af53edd5c00b287e3bef08b03a6a280050` (106,169,147 bytes). Latest completed epoch-2 resume checkpoint SHA256 is `f94704a185645e76cad3b2e5eeb63ef90cfafb116d1fcab822aaf7cabccc4070` (318,381,091 bytes), with 7,678 successful of 7,680 attempted updates, two skips, scaler 2048 and exact epoch-boundary cursor. Failure wrapper SHA256 is `62e3311a02a1702b82dc92e73d44aee61a4448b6bcaef532409a7ea4c27022a6` on Kaggle; the downloaded copy is retained under `runs/assembly_v1/kaggle/pair_transformer_pilot_output/v2_failure/`.

An evaluation-only job on the frozen best checkpoint was specified to recover the full QAP/SSIM gate that the training abort prevented. Audit showed that the unchanged gate requires up to 3.74 million pair scores plus 128 QAP invocations and should reserve 30–150 minutes (up to about 2.5 hours) on one T4. Because both completed epochs already have negative selection deltas and the third epoch is numerically unstable, spending that quota would violate the project's no-signal pivot rule. The local eval-only bundle is retained for reproducibility, but it is not authorized for remote launch. Pair Transformer is closed without promotion.

## Raw-only whole-layout energy Transformer

This branch implements the user's proposed plausibility critic: score a proposed dirty layout, localize implausible positions, and use the score as an energy for target-free iterative repair.

Confirmed-correct properties:

- positive and negative layouts index the exact same independently corrupted raw tile multiset;
- no denoiser or clean target is required at inference;
- global context reaches all windows through learned global tokens;
- move-vector direction and position-to-slot conventions are correct;
- train/selection/holdout are whole-source disjoint.

The first implementation had 8,498,620 parameters and passed 20 tests, but independent review stopped the Kaggle launch because the protocol was not yet task-faithful:

- the family called `qap_sparse` was only visually similar sparse swaps, not a solver output;
- two swap steps could improve at most `4/576 = 0.006944`, making the old absolute `0.02` repair gate impossible;
- the energy loss separated perfect from imperfect layouts but did not order two imperfect layouts;
- no equal-budget classical-seam repair control was present;
- AMP recovery, DDP evaluation, and resume required hardening.

The replacement protocol uses a genuine input-only raw RGB L1-w2 plus soft-cycle first pass, followed by oracle-labelled one-move repair chains for graded monotonic supervision. A local timing check measured about 0.031 seconds for compatibility and 0.711 seconds for soft-cycle assembly per 576-tile source. The refiner has six iterations, 32 hot positions and up to 64 scored proposals per step; the theoretical repair ceiling is 33.3%, so the 25% relative-repair gate is reachable but demanding.

After hardening, 25 local tests, full resume microsteps, packaging simulation, exact core hash pins, strict artifact validation and two independent staging reviews passed. Kaggle dataset `pasha883/vsos-layout-energy-pilot-code` was created from code ZIP SHA256 `392db8809bd1f922d0b1220dc87f6504ecc67a675bd9e99dc6944945d750a600`. Kernel `pasha883/vsos-layout-energy-pilot-t4x2` version 1 ran on two explicit Tesla T4 GPUs.

Decision: **the complete raw-layout method failed the holdout gate and is unsafe for submission, but the critic/heatmap contains a narrow diagnostic signal.**

The 512-source, four-epoch pilot completed successfully in 2,537 seconds of model work (2,628 seconds including tests/wrapper). Hardware preflight and all 25 tests passed. AMP had one synchronized skipped update in epoch 1 and then remained stable at scale 512.

| Split | Learned ranking | Classical seam ranking | Graded ordering | Pooled local AUC | Relative repair | Adjacency delta | Learned control wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| Selection | 0.933594 | 0.972656 | 0.831597 | 0.929870 | 0.000000 | -0.001642 | 0.000 |
| Independent libjpeg holdout | 0.917969 | 0.996094 | 0.824653 | 0.960503 | 0.000054 | -0.002859 | 0.000 |

The critic reliably separates perfect from broad corruptions (holdout energy margin 11.706) and its local error head has high synthetic AUC. However, the simple raw-seam first pass is almost entirely wrong, neural energy ranks complete candidates worse than the raw seam baseline, and every accepted energy descent sequence fails to improve actual placement; adjacency is harmed. The current model therefore cannot be promoted or used to alter a submission.

The only scientifically defensible follow-up is a bounded, inference-only hybrid diagnostic: apply the frozen error heatmap to much stronger HBT/QAP layouts, let a classical localized assignment/search propose moves, and use the critic only as a secondary reranker with an equal-budget seam control. Do not spend another full training run unless that diagnostic shows real actionability; otherwise close this branch and prioritize pairwise/positional models.

Artifacts:

- report: `runs/assembly_v1/kaggle/layout_energy_pilot_output/v1/layout_energy_pilot/layout_energy_report.json`, SHA256 `21ba1c9686191b4e104e1a97afb67e5bdd8f1a5d37c1152c20624f6fbd594d1a`;
- model checkpoint: `runs/assembly_v1/kaggle/layout_energy_pilot_output/v1/layout_energy_pilot/layout_energy_checkpoint.pt`, SHA256 `039cd7638731006665a62064f658211fd288d8cdcae6df79347a2f038f5cb717`;
- epoch resume checkpoint SHA256 `0211907315f2aa8e3516e127a4e9eb432498d914f7e9e7dd4062ebde84b321e3`;
- wrapper SHA256 `8b326c8d54e2c77a6fafed5f647f809c6ed471f6ff6486f506daf2753727192e`.

## Positional Diffusion

Decision: **closed for a large, reproducible scientific failure on both corruption panels.**

Closed defects:

- seed is applied before model construction;
- DDIM warm start samples the correct seeded `q(x_T | baseline)` with Gaussian epsilon;
- filename-based QAP seed matches the production convention;
- train and evaluation share the same warm-start family;
- epoch-boundary optimizer/scaler/per-rank RNG resume is implemented;
- transitive code and model-asset provenance is recorded.

Closed review requirements:

- bounded synchronized GradScaler recovery instead of aborting on the first overflow;
- clean development sources outside upstream denoiser/HBT training;
- include `qap_w1_b0.05_i25` in the frozen comparator envelope;
- distribute evaluation across ranks;
- atomic durable latest checkpoints and strict standalone-evaluation contract validation;
- a larger/repeated development gate before promotion.

Closed final resume fixes:

- after fallback from a corrupt `latest` checkpoint, the next save must not rotate that corrupt file over the only valid `.previous` checkpoint;
- exact-resume validation must include `qap_iterations`, `qap_restarts`, `qap_boundary_weight`, and `qap_refine_swaps`, because they change the w4-QAP training ablation.

Both defects now have regressions. The full positional suite is `23/23`; a default 16,030,530-parameter, 576-tile CPU forward/backward has finite loss and gradients. Final core SHA256 values are model `25a4ace2f3aaa8e1371ca54a7e65efaddd8db9aafd37a78deb299290a914fae3`, trainer `d52ca2665740fff02cbc415c86871fa698f923c1207ca49ab0e9929a874d315d`, and tests `4e11218c782a125d55c137b0d7d6d64c51bac1e6e6830fee0fea99ff0d6c5648`.

The rebuilt overlay ZIP SHA256 is `e4987e110ff518b9d7e7b910158709890c181b6b176834c0ec28991d642ea201`. Local base+overlay merge, exact hashes, pycompile, 23 core tests, CPU dry-run and metadata checks pass. Independent adversarial review additionally forced true RNG-state restoration checks and exact recomputation of source-level bootstrap confidence intervals; 8 staging tests plus 23 core tests pass.

Kaggle kernel `pasha883/vsos-positional-diffusion-pilot-t4x2` version 1 failed before GPU work because exact mount enumeration inspected `/kaggle/input` instead of the owner directory under Kaggle's live dataset layout. Version 2 fixed only that enumeration, passed 31 tests and independent review, and has runner SHA256 `8e019ba44ae1a5f9b8678420b931b2d1a50e7473a7e96469c99d2562a9e2b44c`.

Version 2 completed 384 sources x four epochs on two T4 GPUs. All 192 optimizer attempts succeeded with zero AMP skips. Training loss decreased from 0.18735 to 0.12206 before ending at 0.12406, so the failure is not an optimizer crash. The frozen two-panel gate was decisively negative:

| Panel | Candidate SSIM | Candidate adjacency | SSIM delta vs envelope | Adjacency delta vs envelope | SSIM bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Primary kornia | 0.155543 | 0.001019 | -0.059778 | -0.118886 | [-0.072369, -0.049297] |
| Independent libjpeg | 0.153526 | 0.001981 | -0.063857 | -0.115885 | [-0.077049, -0.052028] |
| Macro | - | - | -0.061817 | -0.117386 | - |

The model effectively destroys adjacency and must not be scaled, blended, or used in a submission. Report: `runs/assembly_v1/kaggle/positional_diffusion_pilot_output/v2/positional_diffusion_pilot/positional_diffusion_report.json`, SHA256 `9b5bf0626bbb9a1b259f18842162caf43491644ad495673c7e0f0e69cd6d6294`. Final/latest/previous checkpoints have SHA256 `9bae8adbcf2aa427857c086eff093606baf12dc42463d0ad00fd80e013a809af`, `f85fb0d523c67bb8ceb275cfff96aadee644463e2577e388e7a5f72b96647eaa`, and `3f8d78d8fe014fd7c19201f6bf54aa58a06142c00eed9e5e3ef6eaedeffaac76`.

## Frozen critic heatmap on strong HBT/QAP layouts

Decision: **no actionable signal; the raw layout-energy branch is fully closed.**

Because the original critic had high synthetic local AUC but its raw soft-cycle first pass was 99.8% wrong, one bounded inference-only salvage diagnostic was run on 16 authoritative frozen real layouts from the existing HBT and QAP pipelines. For each base and K in {8,16,32}, the critic heatmap selected suspect positions, exactly 96 raw-seam swap proposals were scored, predictions were atomically frozen before targets were opened, and learned choices were compared with an equal-budget seam-only control and no-op.

The best learned-looking cell was QAP + critic heatmap + seam selector at K=16: mean raw-render SSIM delta `+0.000117464`, source wins `0.6875`, and paired CI vs the unchanged base `[+0.000011753,+0.000216753]`. However, the gain was far below the `+0.001` actionability threshold and did not beat the equal-budget seam-only control: learned-minus-control mean `+0.000078272`, CI `[-0.000075315,+0.000213174]`. Every one of the 12 learned gates failed. The strict version-3 report therefore says `no_actionable_signal`.

Artifacts:

- report: `runs/assembly_v1/kaggle/layout_energy_hybrid_diagnostic_output/v3/layout_energy_hybrid_diagnostic/layout_energy_hybrid_report.json`, SHA256 `36fd481cc8362174db121b5eee5b1a2201c32d1979128dcb910aa7269cfb1b37`;
- frozen predictions SHA256 `8a6fbb3f9c696725fa37fc1b89e2eb9d34b3ca68d74878e8d1f7ce08d8f7ec5c`;
- wrapper SHA256 `5d32a9e919a17ca9928a1fbe722ffc2cb1b3e4aa58f87bdbc15341e3c4e1887c`.

## HBT weights-only continuation on new scenes

Decision: **the simple "more scenes and more corruption draws" continuation is
closed without promotion.**

The frozen best denoised RGB+Sobel HBT already performs dense all-pairs
retrieval over every other tile. A bounded weights-only fine-tune tested the
user-proposed incremental route before attempting a larger pair model. It used
the next 2048 disjoint `edge_train` sources, two fresh primary-Kornia replica
seeds, two requested epochs and learning rate `1e-4`. The AdamW optimizer was
reset; this was not an exact optimizer resume.

Kaggle version 1 was accidentally capped at 300 seconds by the CLI runtime
flag and produced no scientific result. Version 2 verified two Tesla T4 GPUs
and completed epoch 1 in 2328 seconds. Relative to the frozen checkpoint on the
same historical selection panel, R1 improved `0.223845 -> 0.229676`, MRR
`0.321852 -> 0.328905`, and R32 `0.703889 -> 0.710711`. The precommitted screen
required R1 `>=0.233845`, MRR `>=0.331852`, and R32 `>=0.698889`, so only R32
passed. Epoch 2 reached source 149/2048 and then received external `SIGKILL`
(`-9`); it has no valid metric.

The valid epoch-1 checkpoint was recovered and audited, but the untouched
`edge_development[96:128]` primary/libjpeg comparator and QAP gate were not
opened because the cheap selection prerequisite failed. Checkpoint SHA256 is
`18d79abe4c571afdbd1f02db1b7c2ee2579a408de0e052aa87eff8316cf22f80`;
full evidence is in
`runs/assembly_v1/kaggle/hbt_continuation_output/v2_failure/RESULT_SUMMARY.md`.

This modest positive movement supports improving pairwise compatibility, but
not another identical continuation. The next model must be a more expressive
dense pair scorer or residual expert fusion, not more epochs of the current
429k-parameter cosine HBT.

## Fixed QAP-weight confirmation (precommitted)

The only unclosed deployable candidate after the neural failures is the fixed
`qap_w1_b0.05_i25` configuration from the earlier upgrade matrix. It was
selected by exact synthetic metrics before the old real16 targets were opened
and subsequently produced SSIM `0.184866179` versus production w4
`0.182819915`: mean delta `+0.002046264`, `13/16` wins, and source-bootstrap
95% CI `[+0.000174818,+0.003975691]`. It did **not** pass the old precommitted
major-promotion threshold of `+0.005` and is not retrospectively promoted.

Before opening any new assembly target, a single confirmation protocol was
frozen in `configs/qap_weight_confirmation_v1.json`, SHA256
`30732463fb200bdff8f909ef06be6cb6c4e7859692e01c9d33c5d55175ffe262`.
It compares only fixed w1 and production w4 on the assembler-fresh
`assembly_incremental_gate[128:192]` slice (64 sources, names SHA256
`e5fb7fc6b3d24e9c080b4f33224b863c181e72452de4e54e602a80a321c13251`).
The denoiser previously opened these sources in its own frozen upstream gate,
so the panel is not globally unseen, but no assembly solver has scored its
targets.

The confirmation retains the original `+0.005` mean-delta contract, requires
a positive paired-bootstrap lower bound, at least `40/64` wins, no more than
`6/64` regressions worse than `-0.01`, and valid permutations for every source.
All layouts and denoised render bytes must be frozen and rehashed before a
durable target-access marker is written. Passing makes w1 eligible only for a
sealed audit, not directly safe for submission. A positive CI below `+0.005`
is reported as a confirmed small gain without promotion or weight retuning.

The fixed confirmation completed on Kaggle kernel
`pasha883/vsos-fixed-qap-weight-confirmation-t4x2` version 3. The downloaded
prediction archive was independently rehashed and every reported SSIM was
recomputed from its exact PNG bytes and the 64 local targets.

| Metric | Production w4 | Candidate w1 | Delta |
|---|---:|---:|---:|
| Mean RGB SSIM | 0.1942174611 | 0.1955921961 | +0.0013747350 |

The paired bootstrap interval was
`[+0.0002195541,+0.0025803167]`, with `35/0/29` wins/ties/losses and zero
regressions below `-0.01`. This confirms that w1 is a small average
improvement, but it failed the precommitted `+0.005` mean-delta and `40/64`
win gates. Status is therefore `confirmed_small_gain_no_promotion`:
production remains w4, the sealed audit stays closed, and the weight must not
be retuned or routed per source.

Verified report SHA256 is
`229f3751b85f26f9066c7fae0ed055f5a308354db1a408f0bcee88e8fa5189e7`;
the frozen Phase-A archive SHA256 is
`5f37369fbe3d943158fdef6dc04fb2f124cd305f6ae32191eb2ee47d6e512751`.
Full readback evidence is in
`runs/assembly_v1/kaggle/qap_weight_confirmation_output/v3_verified/RESULT_SUMMARY.md`.

## Dense all-pairs residual scorer

Decision: **closed at the first retrieval gate; no promotion, QAP/SSIM gate, or
submission.**

This branch implemented a 2.59M-parameter full 576x576 residual pair scorer
over frozen denoised C1 + HBT-w4 costs. Each sampled outgoing row and incoming
column scored all 575 valid non-self alternatives. Epoch 0 was an exact
zero-residual reproduction of the frozen promoted pair scorer, so the pilot
could always fall back without silently degrading the existing solver.

The Kaggle pilot completed normally on two Tesla T4 GPUs. All 32 remote tests
passed, the full-model two-GPU forward/backward smoke passed, there were zero
AMP skips, and peak CUDA reservation was about 3.756 GB per rank. The bounded
pilot step took 960.83 seconds.

On the fixed `edge_development[96:128]` selection set (32 whole sources, both
primary-Kornia and independent-libjpeg panels), epoch 1 was decisively worse:

| Metric | Frozen base | Epoch 1 | Delta |
|---|---:|---:|---:|
| Recall@1 | 0.182476 | 0.113423 | -0.069053 |
| MRR | 0.273801 | 0.204989 | -0.068811 |
| Recall@5 | 0.356191 | 0.282595 | -0.073596 |
| Recall@32 | 0.654637 | 0.629501 | -0.025136 |

Recall@1 fell on both panels (`-0.070992` primary-Kornia and `-0.067114`
independent-libjpeg), with the pooled bootstrap interval entirely negative
`[-0.075478, -0.062825]`. The precommitted early-stop rule therefore fired
after epoch 1. Epoch 0 remained the best checkpoint, and the final candidate
was exactly the frozen base (`delta = 0`), which correctly failed the required
positive retrieval gate.

The synthetic transfer holdout, QAP/SSIM metrics, original-real-input gate,
confirmation set, and final audit were not opened. `safe_for_submission` is
false. A strict local verifier independently rehashed both checkpoint
manifests, reconstructed all authoritative source slices, checked the gate
state machine, and exited successfully.

Artifacts:

- verified report:
  `runs/assembly_v1/kaggle/dense_pair_residual_pilot_output/verified_v1/dense_pair_residual_pilot/dense_pair_residual_report.json`,
  SHA256 `b435b213e51a3e1f5baca82190a53aad1e8c28b1060984dbb0da1bab92efe119`;
- epoch-0 best checkpoint SHA256
  `2f1050faccf34979fbc6e5d96e424b4ea4436b4f9bb78c36a91082bd24c0756a`;
- epoch-1 terminal checkpoint SHA256
  `9d7456e0c88c7710c8c04af2acbbfdef3a1c59ff1a4dd21ab118d39ffd9a978c`;
- downloaded wrapper SHA256
  `27a4a7df921c9bac4b5cbaf227c1fb2b4af8478cf04a919e122295c6d3d4aef6`.

This specific training recipe must not be extended to epoch 2. The negative
result is strong enough to require a genuinely different objective or model
rather than more compute on the same residual formulation.

## Compute audit

Kaggle API quota checked after the positional, hybrid and aborted pair runs:

- GPU used: `6.62h`;
- GPU remaining: `23.38h` of `30h`;
- refresh: `2026-07-18T00:00:00`.

Heavy local training is not used. Local work is restricted to tests, static audits, tiny CPU microsteps, packaging, and single-source timing checks.

## Promotion rule

No neural branch can modify a submission unless it passes, in order:

1. source-disjoint synthetic selection and independent corruption holdout;
2. equal-compute baselines, including the best fixed QAP envelope;
3. target-free improvement from a real input-only first-pass layout;
4. frozen real-layout evaluation with target opened only after predictions are fixed;
5. an independent seed or replica check;
6. full provenance, hashes, and `safe_for_submission=false` until every prior gate passes.
