R0 | raw1 | PASS | 8 source-disjoint DEV images | overall R@1=0.098902 R@5=0.215014 R@20=0.352468 worst-image-R@20=0.233696 | 6.20s
R1 | multiband hand-crafted | DROP | same 8 DEV images | overall R@20=0.259964, delta=-0.092504 vs R0 | mechanism refuted: untrained multi-band cosine fusion amplifies distortion.
R2 | learned directional Siamese, 200 steps | PARTIAL | 8 source-disjoint DEV images | R@1=0.059047 R@5=0.184556 R@20=0.397758; R@20 delta=+0.045290 vs R0, but r1/r5 decline and gate fails (required r1>=0.25, neighbour>=0.18).
