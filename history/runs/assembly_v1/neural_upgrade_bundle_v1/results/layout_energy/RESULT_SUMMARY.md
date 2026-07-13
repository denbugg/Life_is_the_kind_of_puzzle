# Raw Layout-Energy Transformer Pilot v1

Decision: `holdout_gate_failed`; `safe_for_submission=false`.

The run completed normally on two Tesla T4 GPUs. This is a scientific failure, not an infrastructure failure.

## Configuration

- 8,498,620 trainable parameters
- 512 whole-source training images, four epochs
- 16-source selection and 16-source independent libjpeg holdout
- two corruption replicas per evaluation source
- raw dirty tiles at inference; no denoiser
- genuine raw RGB border-L1 plus soft-cycle first pass
- six-step learned-energy beam repair

## Results

| Split | Learned ranking | Classical seam ranking | Graded ordering | Pooled local AUC | Relative repair | Adjacency delta | Learned control wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| Selection | 0.933594 | 0.972656 | 0.831597 | 0.929870 | 0.000000 | -0.001642 | 0.000 |
| Independent libjpeg holdout | 0.917969 | 0.996094 | 0.824653 | 0.960503 | 0.000054 | -0.002859 | 0.000 |

The energy margin and local heatmap are real signals, but they are not actionable in the current solver. Energy descent consistently lowers predicted energy without improving true placement, and it harms adjacency. The learned complete-layout ranking also remains below the input-only seam baseline.

## Runtime

- model pilot: 2,537 seconds
- wrapper total: 2,628 seconds
- one DDP-synchronized AMP skip in epoch 1
- final AMP scale: 512
- peak allocated CUDA memory: about 642 MB per rank

## Artifacts

- `layout_energy_pilot/layout_energy_report.json`: `21ba1c9686191b4e104e1a97afb67e5bdd8f1a5d37c1152c20624f6fbd594d1a`
- `layout_energy_pilot/layout_energy_checkpoint.pt`: `039cd7638731006665a62064f658211fd288d8cdcae6df79347a2f038f5cb717`
- `layout_energy_pilot/layout_energy_resume_epoch.pt`: `0211907315f2aa8e3516e127a4e9eb432498d914f7e9e7dd4062ebde84b321e3`
- `layout_energy_pilot_wrapper.json`: `8b326c8d54e2c77a6fafed5f647f809c6ed471f6ff6486f506daf2753727192e`

## Follow-up rule

Do not retrain this full method yet. At most run one bounded inference-only diagnostic on stronger HBT/QAP layouts: use the frozen heatmap to choose a small suspect set, classical compatibility/search to propose assignments, and the critic only as a secondary reranker against an equal-budget seam-only control. Close the branch if that diagnostic has no trustworthy gain.
