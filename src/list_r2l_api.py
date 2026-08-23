import inspect
import train_direct_pose
import direct_pose
for module in [train_direct_pose, direct_pose]:
    print(module.__name__)
    for name, value in sorted(vars(module).items()):
        if inspect.isfunction(value) or inspect.isclass(value):
            if getattr(value, "__module__", None) == module.__name__:
                print(name, str(inspect.signature(value)) if not inspect.isclass(value) else str(inspect.signature(value)))
