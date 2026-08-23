import inspect
import direct_pose
import train_direct_pose
for module, names in [
    (direct_pose, ["DirectPoseNet", "DirectionalSiamese"]),
    (train_direct_pose, ["evaluate", "score_candidate_graph", "build_affinity", "load_affinity"]),
]:
    for name in names:
        if hasattr(module, name):
            obj = getattr(module, name)
            print(name, inspect.signature(obj))
            if inspect.isclass(obj) and hasattr(obj, "forward"):
                print(name + ".forward", inspect.signature(obj.forward))
