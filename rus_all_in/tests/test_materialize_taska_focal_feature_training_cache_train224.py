from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_module() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "materialize_taska_focal_feature_training_cache_train224.py"
    spec = importlib.util.spec_from_file_location("train224_materializer_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_combine_offsets_keeps_board_boundaries_and_skips_local32() -> None:
    module = _load_module()
    first = np.arange(module.TRAIN96_COUNT + 1, dtype=np.int32) * 2
    extension = np.arange(
        module.TRAIN256_COUNT - module.EXTENSION_START + 1, dtype=np.int32
    ) * 3
    combined = module._combine_offsets(first, extension)
    assert combined.shape == (module.TRAIN224_COUNT + 1,)
    assert np.array_equal(combined[: module.TRAIN96_COUNT + 1], first)
    assert np.array_equal(
        combined[module.TRAIN96_COUNT + 1 :], extension[1:] + first[-1]
    )


def test_combine_offsets_rejects_empty_board() -> None:
    module = _load_module()
    first = np.arange(module.TRAIN96_COUNT + 1, dtype=np.int32)
    extension = np.arange(
        module.TRAIN256_COUNT - module.EXTENSION_START + 1, dtype=np.int32
    )
    extension[5] = extension[4]
    with pytest.raises(ValueError, match="must contain harvested edges"):
        module._combine_offsets(first, extension)


def test_edge_digest_is_order_sensitive() -> None:
    module = _load_module()
    source = np.asarray([0, 1], dtype=np.int32)
    target = np.asarray([1, 2], dtype=np.int32)
    axis = np.asarray([0, 1], dtype=np.uint8)
    assert module._edge_digest(source, target, axis) != module._edge_digest(
        source[::-1], target[::-1], axis[::-1]
    )
