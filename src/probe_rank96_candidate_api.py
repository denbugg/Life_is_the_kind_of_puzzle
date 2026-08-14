from __future__ import annotations

import inspect

import infer_rank96 as rank96

for name, value in sorted(vars(rank96).items()):
    if callable(value) and any(token in name.lower() for token in ("candidate", "affinity", "dense", "score", "rank")):
        try:
            signature = str(inspect.signature(value))
        except Exception:
            signature = "<unavailable>"
        print(f"{name}{signature}")
