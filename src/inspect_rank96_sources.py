from __future__ import annotations

import inspect
from pathlib import Path

import infer_rank96 as rank96

out = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\rank96_native_source.txt")
chunks = []
for name in ("infer_one", "solve_dense_tiles"):
    chunks.append(f"\n===== {name} =====\n")
    chunks.append(inspect.getsource(getattr(rank96, name)))
out.write_text("\n".join(chunks), encoding="utf-8")
print(out)
