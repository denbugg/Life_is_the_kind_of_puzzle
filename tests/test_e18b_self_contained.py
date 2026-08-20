"""Ensure the Kaggle code_file is deterministic and truly self-contained."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "kaggle_solve_puzzles_e18b.py"


class E18bSelfContainedTest(unittest.TestCase):
    def test_bundle_is_current_and_has_no_sidecar_imports(self) -> None:
        subprocess.run(
            [sys.executable, str(REPO / "build_e18b_self_contained.py"), "--check"],
            cwd=REPO,
            check=True,
        )
        source = BUNDLE.read_text()
        self.assertNotIn("import kaggle_e14_solver", source)
        self.assertNotIn("import kaggle_e18b_postprocess", source)

    def test_isolated_import_without_sidecar_files(self) -> None:
        smoke = r'''
import importlib.util, pathlib, sys, types, numpy as np
pkg = types.ModuleType("tqdm")
auto = types.ModuleType("tqdm.auto")
auto.tqdm = lambda iterable, **kwargs: iterable
sys.modules["tqdm"] = pkg
sys.modules["tqdm.auto"] = auto
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("e18b_isolated", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.kaggle_e14_solver.__file__.startswith("<embedded:")
assert module.kaggle_e18b_postprocess.__file__.startswith("<embedded:")
raw = np.random.default_rng(18).integers(0, 256, (480, 480, 3), dtype=np.uint8)
out, used, reason, stats = module.kaggle_e18b_postprocess.polish_or_raw(raw)
assert used and reason is None and out.shape == raw.shape
assert stats["guarded_gray_count"] <= stats["raw_gray_count"]
'''
        with tempfile.TemporaryDirectory(prefix="e18b-isolated-") as directory:
            isolated = Path(directory) / BUNDLE.name
            shutil.copy2(BUNDLE, isolated)
            subprocess.run(
                [sys.executable, "-I", "-c", smoke, str(isolated)],
                cwd=directory,
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
