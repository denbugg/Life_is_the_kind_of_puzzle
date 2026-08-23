import inspect
import json
import torch
from eval_r2l_affinity_union import _load_r2, DEFAULT_R2L
m = _load_r2(DEFAULT_R2L, torch.device("cpu"))
print(json.dumps({
    "class": type(m).__name__,
    "forward": str(inspect.signature(m.forward)),
    "methods": [x for x in dir(m) if not x.startswith("_") and x in {"forward", "encode", "score", "pair_scores", "directional_scores", "embed"}],
}, indent=2, sort_keys=True))
