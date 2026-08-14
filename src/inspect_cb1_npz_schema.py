from __future__ import annotations

import numpy as np

CACHE = r"E:\pazzle_work\edge_confidence\full_graph_cache\image_0051_k64.npz"

with np.load(CACHE, allow_pickle=False) as archive:
    for key in archive.files:
        value = archive[key]
        print(f"{key}\tshape={value.shape}\tdtype={value.dtype}")
