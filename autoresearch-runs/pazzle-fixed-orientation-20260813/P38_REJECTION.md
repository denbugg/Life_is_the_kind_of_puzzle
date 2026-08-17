# P38 SRIT-24 G2 rejection

The scaled raw-image transformer completed the whole pre-registered 80-epoch / 7,680-update FP32 run on exactly 96 FIT-train raw input PNGs plus permitted cached adjacency labels. It used no score cache, DINO feature, filename, source ID, coordinate, target PNG or P8 artifact. Terminal loss was 12.708713 and directed Top-20 recall was 3.504302%, statistically at the random 20/575 baseline and below the 20% gate. Runtime 783.18 s, 0 invalid.

Thus more capacity and epochs alone did not create a learnable signal for direct 576-way raw-pixel adjacency CE. G3, CAL, DEV, held, test and target PNGs remain unopened. A successor must change the objective: image-only self-supervised masked/reconstruction or contrastive edge pretraining before label-limited relational fine-tuning. Evidence: P38_G2_REPORT.json.
