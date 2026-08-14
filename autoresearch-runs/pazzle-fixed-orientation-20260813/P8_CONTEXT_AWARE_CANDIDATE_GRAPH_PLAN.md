# P8 — Context-Aware Virtual-Halo Candidate Graph Scorer

**Series:** ORBIT-24  
**Status:** pre-registered; no P8 code, training, target-bearing evaluation, or submission has run.  
**Research basis:** `ORBIT24_P8_CONTEXT_GRAPH_RESEARCH.md`; P1–P7 experiment ledger.

## 1. Hypothesis

P7 establishes a strong corruption-invariant whole-tile identity embedding, while P3 establishes that a **locally isolated** two-pixel boundary scorer does not discriminate hard rank96 candidates sufficiently. A shuffled input cannot provide true physical neighbourhood pixels at inference. Its only legal context is the set of plausible candidate tiles from the frozen rank96 candidate graph.

> **H-P8.** For a fixed `(anchor, direction)`, a Transformer that jointly sees the anchor and its ordered rank96 candidate-neighbour set can form a *virtual context halo*. Combining frozen P7 whole-tile embeddings with directional boundary-band embeddings lets it suppress visually similar but spatially incoherent alternatives, raising source-disjoint true-neighbour ranking above both frozen rank96 and a context-free P7 local scorer.

P8 does not use absolute source position, true target neighbourhood, input index, rotation, CAL/DEV/test imagery, restorer, or postprocessing.

## 2. Model and data contracts

A FIT synthetic bag has the existing challenge-matched independent per-tile corruption and known permutation only for source-supervised labels. Frozen rank96 candidate generation supplies an ordered 32-candidate directional list per anchor. The true neighbour is injected only if absent, with its original rank recorded; this makes the listwise target defined without ever changing a target-derived score.

Each candidate token contains (a) a frozen P7 `128-D` whole-tile embedding of candidate and anchor, (b) a learned encoding of their directional 2-pixel boundary band, (c) candidate-rank embedding, and (d) direction embedding. A 4-block, 192-wide, 8-head attention encoder makes candidate tokens contextual; a shared MLP outputs one logit per candidate. The loss is softmax cross-entropy over the 32-token set. A matched local-only ablation receives exactly the same token ingredients but applies no cross-candidate attention.

At test/capacity inference the P7 encoder is frozen at `p7_g1_encoder.pt`; the P7 reconstruction decoder is omitted. P8 never claims that the rejected P7 photometric decoder is usable.

## 3. Fixed gates

| Gate | Budget and access | Pass condition | Failure consequence |
|---|---|---|---|
| **G0: candidate-context contract** | 4 FIT synthetic bags; 32 lists per direction sampled; no CAL/DEV/test | every list has exactly 32 unique legal tile IDs, correct-direction target is present once, candidate token ordering permutes equivariantly with an independently permuted candidate order, and finite logits/loss/gradients | implementation correction only; no CAL |
| **G1: FIT context capacity** | 128 FIT train sources + 32 source-disjoint FIT held-out sources; cache frozen candidate lists; 4,000 steps context model and 4,000 matched local-only ablation | held-out contextual directional top-1 is at least **+5 pp** over frozen rank96 list score and at least **+3 pp** over the matched local-only ablation; contextual top-20 does not fall | reject P8 before any solver / CAL |
| **G2: score–decoder alignment** | only after G1 pass; 16 extra FIT source-disjoint bags; no CAL target | injecting P8 score into directed matrices improves mean raw-layout proxy objective over rank96 on at least 12/16 source-known FIT bags without violating bijection | reject P8 before CAL |
| **G3: CAL raw-layout** | only after G2 pass; one pre-registered CAL image; direct P8 score injection only | raw-layout SSIM strictly exceeds 0.2621234038 | reject before DEV |
| **G4: DEV paired confirmation** | only after G3 pass; eight pinned DEV boards | mean paired raw SSIM delta >0 and lower bootstrap-95% >0 | reject before test |

The CAL rule is intentionally deferred. P2 demonstrates that a locally trained score can be anti-aligned with the decoder, so P8 must first show a G2 *FIT score–decoder alignment* result after its ranking result.

## 4. Isolation and resource controls

| Control | Requirement |
|---|---|
| Puzzle | fixed upright `24×24`; a tile is only permuted, never rotated |
| Pretraining feature | frozen P7 encoder checkpoint from FIT-only G1; no P7 decoder feature |
| Data | pinned FIT split only until G3 explicitly passes G2 |
| GPU | one local RTX 2070, FP32, no AMP, no parallel GPU job |
| Artifacts | `E:\pazzle_work\pazzle_fixed_orientation_20260813\P8_context_candidate_graph\` |
| Forbidden G0–G2 | CAL/DEV/test target or input, output assembly, restorer, NLM, submission |

## 5. Falsification and escalation

If G1 fails, candidate context cannot exploit the P7 tile identity feature under these graph lists; the next lever must increase the information entering graph tokens — e.g., multi-scale interior-plus-boundary feature maps or a learned discrete border-token language model — rather than tune attention depth. If G1 passes and G2 fails, the contextual score is retrieval-relevant but still solver-anti-aligned; a globally trained differentiable assignment objective becomes the next lever.

## References

[1] Doersch, Gupta & Efros, “Unsupervised Visual Representation Learning by Context Prediction,” ICCV 2015. <http://graphics.cs.cmu.edu/projects/deepContext/>

[2] Heck, Lermé & Le Hégarat-Mascle, “Solving jigsaw puzzles with vision transformers,” 2025. <https://link.springer.com/article/10.1007/s10044-025-01484-z>

[3] Ofir et al., “Seq2Seq Models Reconstruct Visual Jigsaw Puzzles without Seeing Them,” 2025. <https://arxiv.org/html/2511.06315v1>
