# P37 RIT-24 G2 rejection

The first raw-RGB, 8-layer width-384 relational transformer was trained entirely in FP32 on 96 FIT-train input images and their permitted cached adjacency labels. It saw no score caches, DINO data, target PNGs, source IDs, filenames, absolute positions or P8 artifacts. After the pre-registered four-epoch bounded smoke, its mean directed true-neighbor Top-20 recall was 3.488262%, far below the 20% G2 gate; the last training loss was 12.7089, effectively the random two-direction cross-entropy regime.

This rejects the exact P37 smoke configuration, not raw-image transformer research as a whole: it establishes that 384 full-set updates are insufficient for a 576-way raw-pixel adjacency objective. G3, CAL, DEV, held, test and target PNGs remain unopened. Evidence: P37_G2_REPORT.json.
