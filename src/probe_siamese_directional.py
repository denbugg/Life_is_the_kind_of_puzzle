import inspect
import siamese_directional
import train_siamese_directional
for module in [siamese_directional, train_siamese_directional]:
    print(module.__name__)
    for name, value in sorted(vars(module).items()):
        if (inspect.isfunction(value) or inspect.isclass(value)) and getattr(value, "__module__", None) == module.__name__:
            print(name, inspect.signature(value))
            if inspect.isclass(value) and hasattr(value, "forward"):
                print(name + ".forward", inspect.signature(value.forward))
