from __future__ import annotations

import inspect
import infer_rank96 as rank96

cv = inspect.getclosurevars(rank96.infer_one)
for name, value in sorted(cv.globals.items()):
    print(f"{name}\ttype={type(value).__name__}\tmodule={getattr(value, '__module__', '')}")
