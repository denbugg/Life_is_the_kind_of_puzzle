from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/taska_protected_tail_fresh32_confirmation_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_fresh32_preregistration_is_signed_and_current_disjoint() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sidecar = Path(f"{CONFIG_PATH}.sha256").read_text(encoding="utf-8").split()[0]
    assert sidecar == _sha256(CONFIG_PATH)

    panel = config["panel"]
    roster = panel["source_filenames"]
    excluded = set(panel["excluded_current_held16"]) | set(panel["excluded_opened32"])
    assert len(roster) == 16
    assert len(set(roster)) == 16
    assert not set(roster) & excluded
    assert panel["case_count"] == 2 * len(roster)

    namespace = panel["selection_namespace"]
    seed = panel["selection_seed"]
    full_range = tuple(f"img_{index:06d}.png" for index in range(6_700, 7_000))
    eligible = tuple(name for name in full_range if name not in excluded)
    prefix = f"{namespace}\0{seed}\0".encode()
    expected = sorted(
        eligible,
        key=lambda name: (hashlib.sha256(prefix + name.encode()).digest(), name),
    )[:16]
    assert roster == expected


def test_fresh32_arms_and_runtime_sources_are_byte_pinned() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["protocol"]["budget_sweep"] is False
    assert config["polish"]["arms"] == [
        {"name": "taska_legal_raw_tail", "max_swaps": 0},
        {"name": "protected_tail_24", "max_swaps": 24},
        {"name": "protected_tail_96", "max_swaps": 96},
    ]
    for record in (*config["artifacts"].values(), *config["runtime_sources"].values()):
        path = PROJECT_ROOT / record["path"]
        assert _sha256(path) == record["sha256"]

    raw_solver = PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"
    assert _sha256(raw_solver) == (
        "97859e1f4ff6ceadf56ebccf429f2944ae7a5726631731d9f776224a667eb486"
    )

