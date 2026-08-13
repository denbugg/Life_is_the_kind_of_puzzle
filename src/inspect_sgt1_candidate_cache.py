from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('path', type=Path)
a = p.parse_args()
with np.load(a.path, allow_pickle=False) as z:
    print('PATH', a.path)
    print('KEYS', list(z.keys()))
    for k in z.keys():
        x = z[k]
        print(k, 'shape=', x.shape, 'dtype=', x.dtype, 'min=', np.min(x) if x.size else None, 'max=', np.max(x) if x.size else None)
