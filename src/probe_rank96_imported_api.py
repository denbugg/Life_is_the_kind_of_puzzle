from __future__ import annotations

import inspect
import infer_rank96 as rank96

for name, value in sorted(vars(rank96).items()):
    if callable(value):
        module = getattr(value, "__module__", "")
        if module and module != "builtins":
            try:
                signature = str(inspect.signature(value))
            except Exception:
                signature = "<unavailable>"
            print(f"{name}\tmodule={module}\t{signature}")
