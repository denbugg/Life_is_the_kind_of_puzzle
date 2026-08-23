import inspect
import json
from train_direct_pose import candidate_direct_labels
from train_offset_pose import mine_affinity_candidates, load_frozen_affinity
print(json.dumps({
    "candidate_direct_labels": str(inspect.signature(candidate_direct_labels)),
    "mine_affinity_candidates": str(inspect.signature(mine_affinity_candidates)),
    "load_frozen_affinity": str(inspect.signature(load_frozen_affinity)),
}, indent=2, sort_keys=True))
