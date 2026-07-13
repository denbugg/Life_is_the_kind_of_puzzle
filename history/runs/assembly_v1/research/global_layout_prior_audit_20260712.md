# Global absolute-layout prior audit after rigid projection v1

Date: 2026-07-12  
Scope: research-only audit; no production/submission mutation; no
`assembly_incremental`, `assembly_final`, `test`, or oracle labels opened.

## Decision

**Close the proposed absolute-position / generic global-layout-prior family for
now. Do not launch another learned position head, colour/border heuristic, or
generic no-reference QAP-vs-rigid selector.**

The component-preserving projection exposes a real local/global trade-off, but
the audited input-only signals either have already failed leakage-safe gates or
do not distinguish the two layouts on the four already-exposed Stage-A
records. A new binary chooser would be a narrower fit of the already-tested
MAE/layout-energy reranking family, not a genuinely new source of evidence.

The only global route in this neighborhood not scientifically evaluated is
LaMa large-mask conditional consistency. It is **infrastructure-inconclusive**
after three bounded attempts, not a positive lead. Reopening it would require
a safe modern checkpoint/runtime rather than more compatibility shims.

## What was already tested

| Signal family | Leakage-safe evidence | Outcome |
|---|---|---|
| Handcrafted tile absolute position | ExtraTrees on 85 colour, quantile, texture, 4x4 pooled RGB and pooled-gradient features; 512 train and 64 validation whole sources | Exact position `0.001953` versus center baseline `0.001736`; row MAE `5.56` cells; column MAE `5.99`; border recall `0.0` |
| Embedding absolute position (`L2b`) | Frozen denoised L1 embeddings, MLP row/column head | Four-source validation exact position `0.003038`; row `0.05556`, column `0.03906` |
| Set-conditioned tile position (`T0`) | 1,024 train and 32 validation whole sources | Exact position `0.002658`; row `0.06630`, column `0.04612`; closed as weak prior |
| Learned outside/border unary | HBT/L1 outside logits plus component placement | Small real4 improvements did not transfer: real64 compact base `0.184521`, compact+outside `0.182321` |
| DINOv2 coherent 4x4 superblocks | 512 train, development64, then exact8; real target remained unopened after kill gate | Development cell accuracy `0.044705` (chance `0.027778`, gate `0.10`), Manhattan reduction `0.0780` (gate `0.25`); exact8 wrong positions worsened |
| Context reorganization | Learned correction around frozen QAP fragments | Exact8 wrong positions `4597 -> 4597`; real16 SSIM unchanged at `0.182819915` |
| ViT-Sinkhorn direct global assignment | 256 source-disjoint train, 8 selection, 8 independent holdout | Selection adjacency `0.00679` vs classical `0.06782`, SSIM delta `-0.02554`; holdout adjacency `0.00691`, SSIM delta `-0.02660` |
| Positional diffusion | 192 successful optimizer steps; fixed exact-panel comparator envelope | Adjacency delta `-0.117386`, SSIM delta `-0.061817`; stop/pivot |
| Learned full-layout energy | Source-disjoint selection and independent holdout | It ranks synthetic damage, but actual repair adjacency changed `-0.00164` selection and `-0.00286` holdout; learned control wins `0.0` |
| Layout-energy hybrid on frozen real16 | QAP/HBT layouts, equal-budget seam control | Best QAP raw-SSIM delta only `+0.000117`; below `+0.001` gate and not positive versus equal-budget seam control |
| Frozen MAE naturalness | First mixed pool, then 192 seam-guarded competitive candidates/source | Mixed pool selector only `+0.000730`; competitive run selected `-0.000813` vs QAP, `4/16` wins, CI `[-0.001512,-0.000222]`, Spearman `0.0574`, pairwise `0.5202` |
| LaMa masked consistency | Three bounded infrastructure attempts | No energy/correlation metric and no target access; infrastructure-inconclusive |

The direct semantic follow-ups (fragment diffusion and GANzzle-style latent
retrieval) were originally dependency-gated on the DINO block probe. The DINO
signal did not pass its preregistered transfer gate, so treating those as
untried near-term candidates would ignore the failed prerequisite.

## Specific audit of QAP versus rigid projection

Stage A already contains four records: two whole `edge_development` sources
under `primary_kornia` and `independent_libjpeg`. The rigid layout preserves
all accepted component edges and raises mean adjacency from `0.070426` to
`0.129303`, but lowers mean RGB SSIM from `0.254126` to `0.247029`.

An oracle that chose the better of QAP and rigid on each already-exposed record
would gain only about `+0.003380` mean SSIM over always retaining QAP. That is a
small ceiling and is not available at inference.

| Source / panel | Rigid minus QAP SSIM | Denoised seam-L1: QAP -> rigid | Low-frequency TV (sigma 20): QAP -> rigid | Denoised Haar faces: QAP -> rigid |
|---|---:|---:|---:|---:|
| `img_005666`, primary | `-0.032659` | `0.06929 -> 0.11453` | `0.002946 -> 0.002737` | `0 -> 1` |
| `img_003853`, primary | `+0.009980` | `0.09588 -> 0.14015` | `0.002719 -> 0.002579` | `2 -> 1` |
| `img_005666`, independent | `+0.003542` | `0.06339 -> 0.11893` | `0.002474 -> 0.002372` | `1 -> 1` |
| `img_003853`, independent | `-0.009251` | `0.09765 -> 0.14263` | `0.002551 -> 0.002490` | `0 -> 0` |

This descriptive check reused only the four already-exposed Stage-A sources
and regenerated their frozen exact panels from recorded seeds. It was not
precommitted, so it is diagnostic only and cannot support promotion.

The result is still useful:

- seam-L1 always prefers QAP, including both small rigid wins;
- low-frequency smoothness always prefers rigid, including both rigid losses;
- face count is sparse and can prefer the catastrophic rigid loss;
- the same underlying source flips QAP/rigid winner between corruption
  engines, so scene semantics alone cannot explain the choice.

A face-center or low-frequency selector would therefore be an unsupported
post-hoc rule. A learned binary chooser would need many new QAP/rigid pairs and
would still duplicate the already-negative competitive MAE/layout-energy
selector family for an oracle ceiling that is only `+0.00338` in Stage A.

## Closure rule and safe next pivot

Do not spend GPU time on:

1. another isolated-tile row/column classifier;
2. hand-tuned sky/top, floor/bottom, face/center, flat-colour/border rules;
3. generic CLIP/MAE/NR-IQA reranking of QAP and rigid candidates;
4. scaling ViT-Sinkhorn, positional diffusion, or DINO position heads without
   a new transferable fragment-level signal.

The next solver effort should target candidate-edge correctness or a genuinely
task-conditional global reconstruction signal. If a safe modern large-mask
conditional model becomes available, LaMa-style eroded-mask consistency can be
revisited under its original source-disjoint frozen-energy gate; otherwise
retain QAP as the conservative global layout.

## Evidence hashes

- `configs/component_preserving_qap_projection_v1.json`:
  `45233e617619b3d06cb8fddd189c7cd4126ddf3ad279466399ae39b75a143f22`
- `runs/assembly_v1/development/component_preserving_qap_projection_v1/RESULT.json`:
  `f7b4337a1901ae36316b581ccfbe5b968ad2fdda048470e0b0402ccc6bba1301`
- `runs/assembly_v1/spatial_prior/spatial_prior_512.json`:
  `8467ec7fea1605514e04af235d382f35a554882d432f41eadfe26866f61f32be`
- `runs/assembly_v1/l2b/local_64x2.json`:
  `cc70559fc426e329e517d25b4a091042b0f9bb64a352a5ebe9d6e9e319db5231`
- `runs/assembly_v1/kaggle/t0_gpu_full/t0_gpu_full.json`:
  `9766ac58045b4718288d84653642165ef30048bc8b9917982c6da3dc033ce469`
- `runs/assembly_v1/real_cal/real_cal_64_l1full_x0full_t0full.json`:
  `5881e8bea82f062d678060909fd0bd69696d153e4a85a48badba0b24e9d87309`
- `runs/assembly_v1/kaggle/context_reorg_gate_output/v1/context_reorg_gate_report.json`:
  `14811e890c54431b82865af7923eaaf487dc076b9352d4a0d2cd26aca00e4053`
- `runs/assembly_v1/kaggle/dino_superblock_probe_output/v1/dino_superblock_probe_report.json`:
  `0d1e95b7ff5635642907936c26b1f4055decebc645c7c9d8c4aad816b0969555`
- `runs/assembly_v1/kaggle/positional_diffusion_pilot_output/v2/positional_diffusion_pilot/positional_diffusion_report.json`:
  `9b5bf0626bbb9a1b259f18842162caf43491644ad495673c7e0f0e69cd6d6294`
- `runs/assembly_v1/kaggle/vit_sinkhorn_pilot_output/v4/vit_sinkhorn_pilot/vit_sinkhorn_report.json`:
  `1541bace8c67ba0f12f1e0a0ab31420b294ef261030316dadf9e506fa1d59e1f`
- `runs/assembly_v1/kaggle/layout_energy_pilot_output/v1/layout_energy_pilot/layout_energy_report.json`:
  `21ba1c9686191b4e104e1a97afb67e5bdd8f1a5d37c1152c20624f6fbd594d1a`
- `runs/assembly_v1/kaggle/layout_energy_hybrid_diagnostic_output/v3/layout_energy_hybrid_diagnostic/layout_energy_hybrid_report.json`:
  `36fd481cc8362174db121b5eee5b1a2201c32d1979128dcb910aa7269cfb1b37`
- `runs/assembly_v1/kaggle/mae_search_gate_output/v3/mae_search_gate_report.json`:
  `0d92e55d6b38383b8cabc950c9b6f5ca71d5065ad83a441e068910fbfcb29a7f`

