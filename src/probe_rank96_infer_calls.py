from __future__ import annotations

import ast
import inspect

import infer_rank96 as rank96

source = inspect.getsource(rank96.infer_one)
tree = ast.parse(source)
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            target = node.func.id
        elif isinstance(node.func, ast.Attribute):
            target = ast.unparse(node.func)
        else:
            target = ast.dump(node.func)
        print(f"line={node.lineno}\t{target}")
