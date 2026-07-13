#!/usr/bin/env python3
"""Versioned v4 binding of the frozen deterministic Kaggle bundle builder."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_BUILDER = REPO_ROOT / "scripts/build_candidate_graph_oracle_kaggle_bundles.py"
HISTORICAL_BUILDER_SHA256 = (
    "65ff9e1548ed36d4241275199f7b1bd4dfd3c21a547306895beb2ccb9603b3f5"
)


raw = HISTORICAL_BUILDER.read_bytes()
if hashlib.sha256(raw).hexdigest() != HISTORICAL_BUILDER_SHA256:
    raise RuntimeError("historical bundle-builder source drift")
spec = importlib.util.spec_from_file_location(
    "_candidate_graph_oracle_v4_bundle_builder_base_65ff", HISTORICAL_BUILDER
)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load historical bundle-builder base")
_base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _base
spec.loader.exec_module(_base)

_base.CONFIG_MEMBER = "configs/candidate_graph_oracle_ceiling_v4.json"
_base.DATASETS = (
    _base.DatasetSpec(
        key="code",
        slug="pasha883/vsos-candidate-graph-oracle-v4-code",
        title="VSOS Candidate Graph Oracle V4 Code",
        archive_name="candidate_graph_oracle_v4_code_v2.zip",
    ),
    _base.DatasetSpec(
        key="input",
        slug="pasha883/vsos-candidate-graph-oracle-v4-inputs",
        title="VSOS Candidate Graph Oracle V4 Inputs",
        archive_name="candidate_graph_oracle_v4_inputs_v2.zip",
    ),
    _base.DatasetSpec(
        key="runtime",
        slug="pasha883/vsos-candidate-graph-oracle-v4-runtime",
        title="VSOS Candidate Graph Oracle V4 Runtime",
        archive_name="candidate_graph_oracle_v4_runtime_v2.zip",
    ),
)
_base.DATASET_BY_KEY = {item.key: item for item in _base.DATASETS}


def __getattr__(name: str):
    return getattr(_base, name)


def main() -> None:
    _base.main()


if __name__ == "__main__":
    main()
