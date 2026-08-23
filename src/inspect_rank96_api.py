from __future__ import annotations
import inspect
import infer_rank96 as m
for name in ('InferenceConfig', 'LoadedModels', '_default_checkpoints', 'load_models', 'infer_one', 'fixed_nlm', 'split_upright_tiles', 'assemble_upright_tiles'):
    value = getattr(m, name)
    try:
        sig = str(inspect.signature(value))
    except (TypeError, ValueError):
        sig = '<no signature>'
    print(name, sig)
    if name == 'InferenceConfig':
        print('fields', list(value.__dataclass_fields__.keys()))
    if name == 'LoadedModels':
        print('fields', list(value.__dataclass_fields__.keys()))
print('defaults', {k: str(v) for k, v in m._default_checkpoints().items()})
