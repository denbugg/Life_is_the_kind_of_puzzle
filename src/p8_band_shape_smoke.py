import importlib.util
from pathlib import Path
import torch
p = Path(__file__).with_name("p8_context_candidate_graph.py")
spec = importlib.util.spec_from_file_location("p8", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
tiles = torch.zeros((4, 3, 20, 20), dtype=torch.float32)
anchors = torch.tensor([0, 0, 0, 0])
dirs = torch.tensor([0, 1, 2, 3])
members = torch.tensor([[1], [1], [1], [1]])
out = m.bands(tiles, anchors, dirs, members)
assert tuple(out.shape) == (4, 3, 20, 4), tuple(out.shape)
print("P8 directional band smoke PASS", tuple(out.shape))
