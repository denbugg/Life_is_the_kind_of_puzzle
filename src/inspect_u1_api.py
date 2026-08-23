import inspect
import json
import eval_r2l_affinity_union as u

public = {}
for name, value in sorted(vars(u).items()):
    if callable(value) and (name.startswith("_") or name in {"main"}):
        try:
            public[name] = str(inspect.signature(value))
        except (TypeError, ValueError):
            pass
print(json.dumps(public, indent=2, sort_keys=True))
