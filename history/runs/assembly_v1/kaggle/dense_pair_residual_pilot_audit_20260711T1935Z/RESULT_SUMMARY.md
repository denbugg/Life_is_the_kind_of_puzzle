# Dense-pair residual pilot: independently verified result

## Verdict

**NO PROMOTION.** The learned epoch-1 residual materially degraded adjacency retrieval on the fixed 32-source, two-panel cheap-selection set. The precommitted early-stop rule selected epoch 0, whose exact-zero residual reproduces the frozen C1+HBTw4 baseline. Consequently the final cheap-selection deltas are exactly zero, the first retrieval gate failed, and the pipeline stopped before synthetic-transfer QAP, original-real-input QAP/SSIM, final audit, or confirmation.

- Kernel: `pasha883/vsos-dense-pair-residual-pilot-t4x2`
- Kernel/wrapper status: `COMPLETE` / `complete`
- Scientific status: `stop_cheap_selection_retrieval`
- Best epoch: `0`
- Early-stop reason: `epoch1_recall_at_1_and_mrr_nonpositive`
- `safe_for_submission`: `false`
- New QAP/SSIM or submission result: **none**

## Fixed-selection retrieval

Selection is the exact authoritative `edge_development[96:128]` slice: 32 whole source images, SHA-256 `a20a4f638af4c28b807d0d194a6be69cee6b4d8bee7847eefee70fb1817c02de`, evaluated once under each of `primary_kornia` and `independent_libjpeg` (64 source-panel records, 1,104 directed adjacency queries per record).

Epoch-1 absolute values below are derived as the frozen-base absolute aggregate plus the epoch-1 quick-selection delta. Both quantities use the same fixed source names, panel set, and replica contract.

| Panel | Metric | Frozen base / epoch 0 | Learned epoch 1 | Delta |
|---|---:|---:|---:|---:|
| Both panels | MRR | 0.273800577434 | 0.204989433644 | -0.068811143790 |
| Both panels | Recall@1 | 0.182475656703 | 0.113422780797 | -0.069052875906 |
| Both panels | Recall@5 | 0.356190557065 | 0.282594542572 | -0.073596014493 |
| Both panels | Recall@32 | 0.654636548913 | 0.629500679348 | -0.025135869565 |
| `primary_kornia` | MRR | 0.273781611300 | 0.203255684917 | -0.070525926383 |
| `primary_kornia` | Recall@1 | 0.181838768116 | 0.110846920290 | -0.070991847826 |
| `primary_kornia` | Recall@5 | 0.356884057971 | 0.281448143116 | -0.075435914855 |
| `primary_kornia` | Recall@32 | 0.656504755435 | 0.629245923913 | -0.027258831522 |
| `independent_libjpeg` | MRR | 0.273819543568 | 0.206723182371 | -0.067096361197 |
| `independent_libjpeg` | Recall@1 | 0.183112545290 | 0.115998641304 | -0.067113903986 |
| `independent_libjpeg` | Recall@5 | 0.355497056159 | 0.283740942029 | -0.071756114130 |
| `independent_libjpeg` | Recall@32 | 0.652768342391 | 0.629755434783 | -0.023012907609 |

The whole-source bootstrap intervals for epoch-1 delta were also strictly negative:

- Recall@1: `[-0.075478374094, -0.062825167006]`
- MRR: `[-0.075681693942, -0.062393634150]`
- Recall@5: `[-0.081791001472, -0.065712041440]`
- Recall@32: `[-0.031419836957, -0.019304446898]`

Epoch-1 training diagnostics: loss `6.853461077437`, sampled outgoing Recall@1 `0.119303389067`, sampled incoming Recall@1 `0.131917321778`, residual L2 `0.005861520162`, AMP skips `0`.

## Final cheap-selection gate on selected checkpoint

Because epoch 0 was selected, candidate and base are bitwise-equivalent off diagonal and all final retrieval deltas are exactly `0.0` on both panels.

| Precommitted check | Result |
|---|---:|
| Mean Recall@1 delta >= 0.01 | FAIL |
| Mean MRR delta >= 0.01 | FAIL |
| Bootstrap Recall@1 lower bound > 0 | FAIL (`[0.0, 0.0]`) |
| Every panel Recall@1 delta > 0 | FAIL |
| Mean Recall@32 delta >= -0.005 | PASS |
| Overall first gate | **FAIL** |

Protocol state after failure:

- selection synthetic target files opened: `true`
- selection QAP metrics computed: `false`
- synthetic-transfer holdout: not opened/evaluated (`null`)
- original-real-input gate: not opened/evaluated (`null`)
- final audit and confirmation: sealed
- `gate_opened.synthetic_transfer`: `false`
- `gate_opened.original_real_input`: `false`
- `gate_opened.true_final_audit`: `false`
- `gate_opened.true_confirmation`: `false`

## Hardware and runtime

- 2 x Tesla T4, compute capability 7.5, 15,636,037,632 bytes each
- PyTorch `2.10.0+cu128`, CUDA runtime `12.8`
- Full-model two-rank smoke: return code 0, 34.574 s
- Bounded pilot: return code 0, 960.832 s
- Total wrapper wall time: 1,095.656 s (18 min 15.656 s)
- Pilot peak per rank: 3,261,651,968 bytes allocated (20.86%); 3,755,999,232 bytes reserved (24.02%)
- Model parameters: 2,590,178
- Full candidate set: 575 non-self candidates per query
- Dense pairs per source step: 55,296
- Remote unit tests: 32 passed, 45 warnings, 15.46 s

## Independent integrity verification

The repository verifier completed with `verified: true` and exit code 0. It reconstructed every named protocol slice from the authoritative local configs, rehashed all downloaded artifacts, validated the sequential gate state machine, and confirmed `audit_unopened: true`. Both `SHA256SUMS.txt` files also passed `sha256sum -c` independently.

Artifact hashes:

- wrapper: `27a4a7df921c9bac4b5cbaf227c1fb2b4af8478cf04a919e122295c6d3d4aef6`
- pilot report: `b435b213e51a3e1f5baca82190a53aad1e8c28b1060984dbb0da1bab92efe119`
- pilot checksum manifest: `15d72b70ab670c3333b5445ded7041e51204288df7d38fe49ca66ee5553333cf`
- pilot best checkpoint (epoch 0): `2f1050faccf34979fbc6e5d96e424b4ea4436b4f9bb78c36a91082bd24c0756a`
- pilot latest checkpoint (epoch 1): `9d7456e0c88c7710c8c04af2acbbfdef3a1c59ff1a4dd21ab118d39ffd9a978c`
- pilot log: `17b7829e5ab584d3b98e2b73bb40be401caef6f74194b715d104d4acfcbbcc4d`
- smoke report: `1e4d70dc2a74baf2a5d78b2b41e69df0ce8e36f061784d9dddd3bfcc4239988e`
- smoke best checkpoint: `5a02f8420baa9370e06d19eaa9db96ce262ef6726616c37c1d6dc0abdfbf738c`
- smoke latest checkpoint: `6bc8483605f69adbbf5b8e944d20e31816e21e59a5f252244b33a9eca1b01e7a`
- remote test log: `9244152bd424f3f2b929a1b268abc9cee295dac66d011dad5b24cc4b867044fc`

This summary reports retrieval results only. It must not be interpreted as a new SSIM score or a promoted assembly solver.
