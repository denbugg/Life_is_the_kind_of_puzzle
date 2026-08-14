from __future__ import annotations

import ast
import inspect

import infer_rank96 as rank96

tree = ast.parse(inspect.getsource(rank96.infer_one))
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        print({"module": node.module, "names": [alias.name for alias in node.names]})
    elif isinstance(node, ast.Import):
        print({"module": None, "names": [alias.name for alias in node.names]})
