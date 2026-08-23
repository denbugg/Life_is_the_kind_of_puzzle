import inspect
import direct_pose
import train_direct_pose
for module, names in [
    (direct_pose, ["DirectionalSiamese"]),
    (train_direct_pose, ["direct_pose_loss", "candidate_direct_labels", "main"]),
]:
    for name in names:
        if hasattr(module, name):
            print(f"--- {module.__name__}.{name} ---")
            print(inspect.getsource(getattr(module, name)))
