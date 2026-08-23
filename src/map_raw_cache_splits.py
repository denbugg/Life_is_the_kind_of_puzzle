import json
from pathlib import Path

split = json.loads(Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\PGA1_set_slot\source_disjoint_split_v1.json").read_text(encoding="utf-8"))["splits"]
membership = {name: label for label, names in split.items() for name in names}
cache = Path(r"E:\pazzle_work\edge_confidence\full_graph_cache")
rows = []
for path in sorted(cache.glob("image_*_k64.npz")):
    idx = int(path.name.removeprefix("image_").removesuffix("_k64.npz"))
    name = f"img_{idx:06d}.png"
    rows.append((path.name, name, membership.get(name, "MISSING")))
for row in rows:
    print("\t".join(row))
summary = {}
for _, _, label in rows:
    summary[label] = summary.get(label, 0) + 1
print("SUMMARY", json.dumps(summary, sort_keys=True))
