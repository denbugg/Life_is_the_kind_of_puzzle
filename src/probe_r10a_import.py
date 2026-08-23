from pathlib import Path
import eval_r10a_frozen_layout as module

print(module.pinned_dev_names(module.SPLIT, 8))
print("r10_import_ok")
