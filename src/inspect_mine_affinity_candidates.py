from __future__ import annotations

import inspect
from pathlib import Path

from train_offset_pose import mine_affinity_candidates

out = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\mine_affinity_candidates_source.txt")
out.write_text(inspect.getsource(mine_affinity_candidates), encoding="utf-8")
print(inspect.signature(mine_affinity_candidates))
print(out)
