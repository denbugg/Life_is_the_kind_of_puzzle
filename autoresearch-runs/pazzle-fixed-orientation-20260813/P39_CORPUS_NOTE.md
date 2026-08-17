# P39 corpus definition

P39’s unlabeled RGB pretraining uses exactly the 5,360 names under `splits.fit` in `PGA1_set_slot/source_disjoint_split_v1.json`. The manifest reports 7,000 unique names allocated into disjoint it=5360, cal=670, dev=670, and eserve=300 groups. No image outside it may be read in G1; CAL, DEV, reserve and test are excluded even though G1 has no labels.
