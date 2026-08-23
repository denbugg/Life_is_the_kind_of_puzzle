import inspect
import infer_rank96
for name in sorted(n for n in dir(infer_rank96) if n.startswith('_') and any(k in n.lower() for k in ('infer','score','model','tile','load'))):
    value = getattr(infer_rank96, name)
    if callable(value):
        try:
            print(name, inspect.signature(value))
        except (TypeError, ValueError):
            pass
