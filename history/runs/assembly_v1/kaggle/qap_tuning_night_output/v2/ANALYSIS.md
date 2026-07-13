# QAP real16 tuning analysis

All layouts were produced from input pixels and frozen before target images were
opened by `evaluate_real_assembly.py`.  The fixed validation set contains 16
whole source images from `assembly_cal`.

## Fixed-setting results (denoised render SSIM)

| Setting | Mean SSIM | Delta vs soft-cycle | Wins | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| soft-cycle L1 k8 baseline | 0.165431 | - | - | - |
| L1w4 QAP, 25 iterations, 2 restarts | 0.182330 | +0.016898 | 16/16 | [+0.012229, +0.021855] |
| cross-L1w4 QAP, 25 iterations, 2 restarts | 0.182305 | +0.016874 | 16/16 | [+0.012446, +0.022116] |
| L1w4 QAP, 40 iterations, 4 restarts | 0.181305 | +0.015874 | 16/16 | [+0.012027, +0.019990] |
| L1w4 QAP + boundary weight 0.05 | **0.182820** | **+0.017389** | **16/16** | **[+0.012173, +0.023101]** |

The boundary result is only +0.000490 above the ordinary 25-iteration result;
the paired bootstrap interval for that difference is [-0.001814, +0.002633].
It is therefore a minor validation tie, not evidence for a robust extra gain.

Alternative initial layouts did not beat the soft-cycle seed:

| Initialization | Mean SSIM | Delta vs soft-cycle baseline |
|---|---:|---:|
| component L1 fusion q50 | 0.180439 | +0.015008 |
| denoised C1 component fusion | 0.177494 | +0.012063 |
| cross-L1w4 component q50 | 0.175051 | +0.009620 |

The best possible target-oracle selection among the four fixed QAP settings is
only 0.185735.  Input-only compatibility selectors did not exceed the best
fixed setting (best selector result: 0.182724).  The QAP search is consistently
useful, but changing search effort or initialization cannot close the gap to
0.3 while the same pairwise energy is used.

## Artifact hashes

- `qap_cross_multiseed_real16.json`: `1aa5df6d6cca404dd91a5075495822eee6f9e6a92a5f7cb189657630de2d9fd7`
- `qap_l1w4_boundary_real16.json`: `cc1b694b1501ba9b02e5618ad838e155ae40af7990bbbf4542b281fc21adec60`
- `qap_l1w4_heavy_real16.json`: `2a2871922f171c99829bc0254b3c3ee20a27fe4739611c63658e55579f4e5520`
- `qap_l1w4_multiseed_real16.json`: `2c486b433cd93654cb9eee7a155b0605da094dc5d4d47ef5b65bb99fcde75117`
- `qap_tuning_night_wrapper.json`: `194cb065620a32b962a4277a2646055336b857e9fcdf9a687f3299b9dc850886`

