# LaMa large-mask consistency gate

Decision: **infrastructure-inconclusive; stop after three bounded attempts**.

No LaMa correlation metric was produced and no target image was opened. The
three runs failed before the Phase-A energy artifact could be frozen:

1. Kaggle exposed two code roots; the runner did not yet select the one that
   uniquely contained all four fixed-QAP reports. The contract-based selector
   was added and its local report-manifest test passed.
2. The pinned Hugging Face value `b2a4ef...` was the repository's Xet object
   hash, not the downloaded archive SHA-256. The exact revision page lists the
   archive SHA-256 as `f1b358...`; the runner was corrected to record both
   values separately.
3. The verified archive downloaded and all 16 inputs were denoised, but the
   legacy checkpoint's pickle references
   `pytorch_lightning.callbacks`. The intentionally minimal inference shim
   exposes only the helper needed by current LaMa source and is not a complete
   legacy Lightning package, so both workers failed during `torch.load` before
   inference.

Extending arbitrary pickle-class shims or installing an old Lightning stack
would materially increase compatibility and serialization risk for a last,
weakly motivated no-reference gate. Per the predeclared final-attempt rule,
the branch is closed rather than mutating the modern Kaggle PyTorch environment.

The experimental design remains on disk and passed static checks: four fixed
QAP candidates/source are the only promotion population; generated block/band
moves are diagnostic; four masks cover all 36 macroblocks; all layouts and
energies would be frozen before targets; thresholds are Spearman `>=0.25` and
micro pairwise accuracy `>=0.60` over the fixed-QAP set.

Verified pins at closure:

- official LaMa source commit: `786f5936b27fb3dacd2b1ad799e4de968ea697e7`
- official source archive SHA-256: `6759af2b68f942c32c52ecfed42d46b414cb1a8c1960a7b1167b88d40828deb7`
- Big-LaMa mirror revision: `05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9`
- downloaded Big-LaMa ZIP SHA-256: `f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6`
- Big-LaMa Xet hash: `b2a4ef7f88e28fb6c15f0be152d7265a770b54a719774df975847430fa92a283`
