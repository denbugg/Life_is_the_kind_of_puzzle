from __future__ import annotations

import inspect

import infer_rank96 as rank96

cv = inspect.getclosurevars(rank96.infer_one)
for scope, mapping in (("globals", cv.globals), ("nonlocals", cv.nonlocals)):
    for name, value in sorted(mapping.items()):
        if any(token in name.lower() for token in ("candidate", "affinity", "dense", "score", "mine")):
            print({"scope": scope, "name": name, "type": type(value).__name__, "module": getattr(value, "__module__", None)})
            if callable(value):
                print("signature", inspect.signature(value))
                print(inspect.getsource(value))
