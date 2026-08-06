"""Thin root-pinning launcher for the hashed E11 gate-v3 evaluation core."""
from __future__ import annotations

from collections.abc import Sequence

from rank96_lab_selector_v3_core import run_cli


# This launcher is intentionally excluded from gate-v3 code hashes. Freeze the
# gate first, then replace None with the exact create-once ROOT_SHA256.
EXPECTED_GATE_ROOT_SHA256: str | None = (
    "d95493de3a306a550fec92962c44c7494ef87143cba42f62f451979af0ccda1f"
)


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(expected_root=EXPECTED_GATE_ROOT_SHA256, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
