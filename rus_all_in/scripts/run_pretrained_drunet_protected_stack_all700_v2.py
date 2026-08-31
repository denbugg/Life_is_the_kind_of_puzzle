#!/usr/bin/env python3
"""Versioned all-700 runner with canonical JSON audit comparison.

V1 stopped target-blind after its first board because tuple-valued audit fields
round-trip through JSON as lists.  This wrapper preserves the immutable V1
source and changes only the audit object's ``as_dict`` representation to the
same canonical JSON-compatible form on both sides of the equality check.  All
inference, commitment, score, safety, and fail-closed holdout logic remains V1.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from aiijc_puzzle.protocol import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_RUNNER = PROJECT_ROOT / "scripts/run_pretrained_drunet_protected_stack_all700.py"


def _load_v1() -> Any:
    spec = importlib.util.spec_from_file_location("all700_measurement_v1", V1_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load immutable V1 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


class _CanonicalAudit:
    def __init__(self, audit: Any) -> None:
        self._audit = audit

    @property
    def passed(self) -> bool:
        return bool(self._audit.passed)

    def as_dict(self) -> dict[str, Any]:
        return _json_compatible(self._audit.as_dict())


def main() -> None:
    module = _load_v1()
    original_audit = module.audit_raw_permutation
    original_sources = module.source_hashes

    def canonical_audit(*args: Any, **kwargs: Any) -> _CanonicalAudit:
        return _CanonicalAudit(original_audit(*args, **kwargs))

    def v2_sources() -> dict[str, str]:
        hashes = original_sources()
        wrapper = Path(__file__).resolve()
        hashes[str(wrapper.relative_to(PROJECT_ROOT))] = sha256_file(wrapper)
        return hashes

    module.CONFIG = (
        PROJECT_ROOT / "configs/pretrained_drunet_protected_stack_all700_measurement_v2.json"
    )
    module.CONFIG_SIDECAR = Path(f"{module.CONFIG}.sha256")
    module.OUTPUT_ROOT = (
        PROJECT_ROOT / "outputs/pretrained-drunet-protected-stack/all700-measurement-v2"
    )
    module.STAGE_ROOTS = {
        "calibration": module.OUTPUT_ROOT / "calibration700",
        "holdout": module.OUTPUT_ROOT / "holdout700",
    }
    module.audit_raw_permutation = canonical_audit
    module.source_hashes = v2_sources
    module.main()


if __name__ == "__main__":
    main()
