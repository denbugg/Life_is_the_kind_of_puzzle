from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_taska_incidence_gnn.py"


def _load_runner() -> ModuleType:
    sys.path.insert(0, str(ROOT / "scripts"))
    specification = importlib.util.spec_from_file_location(
        "run_taska_incidence_gnn_test", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


experiment = _load_runner()


def test_strict_layout_contract_accepts_only_full_permutations() -> None:
    layout = np.arange(experiment.COUNT, dtype=np.int32)
    np.testing.assert_array_equal(experiment._strict_layout(layout), layout)
    duplicate = layout.copy()
    duplicate[-1] = duplicate[-2]
    with pytest.raises(ValueError, match="strict 576-tile permutation"):
        experiment._strict_layout(duplicate)


def test_fixed_gates_are_pair_primary() -> None:
    assert experiment.PAIR_DENOMINATOR == 1104
    assert experiment.PANEL_ARMS == (
        "incidence_gnn",
        "four_arm_tail96",
        "five_arm_tail96",
    )
