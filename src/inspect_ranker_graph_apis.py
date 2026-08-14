from __future__ import annotations

import inspect
from pathlib import Path

from eval_candidate_rank import score_full_graph
from eval_seeded_qap import dense_rd

out = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P1_CB1_boundary_buddies\ranker_graph_api_source.txt")
out.write_text(
    "score_full_graph\n" + str(inspect.signature(score_full_graph)) + "\n" + inspect.getsource(score_full_graph)
    + "\n\ndense_rd\n" + str(inspect.signature(dense_rd)) + "\n" + inspect.getsource(dense_rd),
    encoding="utf-8",
)
print(out)
