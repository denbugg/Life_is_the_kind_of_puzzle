import inspect
import train_offset_pose
for name in ["load_frozen_affinity", "mine_affinity_candidates", "make_loader"]:
    value = getattr(train_offset_pose, name)
    print(name, inspect.signature(value))
