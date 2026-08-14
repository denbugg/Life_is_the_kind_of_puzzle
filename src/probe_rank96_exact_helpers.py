from __future__ import annotations

import inspect

import infer_rank96 as rank96

for name in ("mine_affinity_candidates", "score_full_graph", "dense_rd"):
    value = getattr(rank96, name)
    print({"name": name, "module": getattr(value, "__module__", None), "signature": str(inspect.signature(value))})
    print(inspect.getsource(value))
