from __future__ import annotations

import inspect
import train_offset_pose as module

for name, value in sorted(vars(module).items()):
    if callable(value) and any(token in name.lower() for token in ("candidate", "affinity", "mine", "score", "dense")):
        try:
            signature = str(inspect.signature(value))
        except Exception:
            signature = "<unavailable>"
        print(f"{name}{signature}")
