# Positional Diffusion Pilot v2

Decision: `development_gate_passed=false`; stop and pivot. The model is unsafe for submission.

The run completed normally on two Tesla T4 GPUs. All 192 optimizer attempts succeeded with zero AMP skips, so this is a scientific failure rather than an infrastructure failure.

## Configuration

- 16,030,530 trainable parameters
- 384 whole-source training images, four epochs
- raw plus restored tile encoder
- input-only HBT relative graph
- 300 diffusion steps, deterministic 30-step DDIM inference
- exact Hungarian projection
- two independent corruption panels, 8 sources x 2 replicas each
- envelope: equal-budget w1-QAP, w4-QAP and pure-HBT QAP, i25/r2/b0.05

## Results

| Panel | Candidate SSIM | Candidate adjacency | SSIM delta vs envelope | Adjacency delta vs envelope | SSIM bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Primary kornia | 0.155543 | 0.001019 | -0.059778 | -0.118886 | [-0.072369, -0.049297] |
| Independent libjpeg | 0.153526 | 0.001981 | -0.063857 | -0.115885 | [-0.077049, -0.052028] |
| Macro | - | - | -0.061817 | -0.117386 | - |

Training loss decreased from 0.18735 to 0.12206 before ending at 0.12406. Despite clean optimization, the sampled assignments have almost zero adjacency and strongly underperform every comparator. Do not scale, blend or submit this branch.

## Artifacts

- `positional_diffusion_pilot/positional_diffusion_report.json`: `9b5bf0626bbb9a1b259f18842162caf43491644ad495673c7e0f0e69cd6d6294`
- `positional_diffusion_pilot/positional_diffusion.pt`: `9bae8adbcf2aa427857c086eff093606baf12dc42463d0ad00fd80e013a809af`
- `positional_diffusion_pilot/positional_diffusion_latest.pt`: `f85fb0d523c67bb8ceb275cfff96aadee644463e2577e388e7a5f72b96647eaa`
- `positional_diffusion_pilot/positional_diffusion_latest.pt.previous`: `3f8d78d8fe014fd7c19201f6bf54aa58a06142c00eed9e5e3ef6eaedeffaac76`
- `positional_diffusion_pilot_wrapper.json`: `5934eb5d6351c870837730cb23952ddef5bca00f37bb1a4de1a34be185f4b2df`
