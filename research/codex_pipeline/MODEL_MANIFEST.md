# Model and artifact manifest

Weights and submission ZIP files are intentionally excluded from Git. Kaggle
kernel outputs are the canonical distribution mechanism.

## Kaggle kernels

| Component | Kernel |
|---|---|
| DDPM restoration | `phoenix0501/pazzle-ddpm-denoise-fragments-continue` |
| Supervised restorer | `phoenix0501/pazzle-fragment-restorer` |
| Edge and position models | `phoenix0501/pazzle-puzzle-assembly-models` |
| RL actor-critic | `phoenix0501/pazzle-rl-puzzle-assembler` |
| End-to-end solver | `phoenix0501/pazzle-full-puzzle-solver` |

## Selected local artifacts

| File | Size | SHA256 | Status |
|---|---:|---|---|
| `ddpm_frag_epoch14.pt` | about 24 MB | `85797dc34df968460c1285867f7d87594db8e594f89fe06790b8c37b3382687e` | historical |
| `fragment_restorer_epoch8.pt` | about 6.4 MB | `5db11a55da07d7db9bb51ac9b0f94efe410f330a272566310cc60f88e18b32fe` | selected restorer |
| `rl_swap_actor_critic_epoch1.pt` | about 643 KB | `ec56aa1115cf02b3db8dcab094857e9fb13e57b42883bdcff28495550e58b036` | selected RL policy |
| `submission_pazzle_solver_v3.zip` | about 185 MB | `c666aa70c0385b7212544bf64f1e6988fc4bef913f345cd4b13107038f98dee3` | historical |
| `submission_pazzle_solver_rl_v5.zip` | about 185 MB | `83b4e2c3a26507f05e51408659a680054cd28f0c0d4266a9432fe4e54c80efae` | pre-audit RL |
| `submission_pazzle_solver_audit_fixed.zip` | about 185 MB | `617b39ec3983fb74db0761932f5961db5770f50c4d00cd5b6a588c5313fdc29e` | verified 700-file fallback |
| `submission_pazzle_solver_restorer_rl.zip` | about 199 MB | `f21de3ef38996e9fa7e4f6c914593a2c40b68e799485169d48ed83535776f778` | verified restorer + guarded RL candidate |

Every promoted submission must pass all of these checks:

- ZIP CRC integrity;
- exactly 700 entries;
- all entries are unique root-level `.png` files;
- the inference log reports `archived=700`;
- the existing audit-fixed archive is retained until the replacement is verified.
