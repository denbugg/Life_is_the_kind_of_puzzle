from pathlib import Path
import numpy as np
root = Path(r"E:\pazzle_work\edge_confidence\full_graph_cache")
for name in ["image_0014_k64.npz", "image_0020_k64.npz"]:
    with np.load(root / name, allow_pickle=False) as z:
        print(name)
        for key in z.files:
            a = z[key]
            print(key, a.shape, a.dtype, a.min() if a.size else None, a.max() if a.size else None)
