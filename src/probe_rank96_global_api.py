import inspect
import infer_rank96
import solve_buddies

for module, prefix in [(infer_rank96, "infer_rank96"), (solve_buddies, "solve_buddies")]:
    print(prefix)
    for name in sorted(n for n in dir(module) if not n.startswith("_") and any(k in n.lower() for k in ("infer", "score", "solve", "tile", "dense", "assemble", "objective"))):
        value = getattr(module, name)
        if callable(value):
            try:
                print(name, inspect.signature(value))
            except (TypeError, ValueError):
                print(name, "<noninspectable>")
