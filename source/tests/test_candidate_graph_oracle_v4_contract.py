from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SNAPSHOT_SRC = (
    REPO / "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src"
)
for _name in tuple(sys.modules):
    if _name == "puzzle_assembly" or _name.startswith("puzzle_assembly."):
        del sys.modules[_name]
    elif _name == "puzzle_denoise_v2" or _name.startswith("puzzle_denoise_v2."):
        del sys.modules[_name]
sys.path.insert(0, str(SNAPSHOT_SRC))


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load(
    "candidate_graph_oracle_v4_evaluator_contract_test",
    "scripts/evaluate_candidate_graph_oracle_v4.py",
)
verifier = _load(
    "candidate_graph_oracle_v4_verifier_contract_test",
    "scripts/verify_candidate_graph_oracle_v4_result.py",
)


def _valid_diagnostics() -> dict[str, object]:
    qap = {
        "objective": 1.25,
        "relaxed_objective": 1.0,
        "restart": 0,
        "iterations": 25,
        "converged": False,
    }
    return {
        "hbt_outside_logits": {
            "dtype": "float32",
            "shape": [576, 4],
            "c_order_sha256": "a" * 64,
        },
        "softcycle": {
            "accepted_edges": 575,
            "component_sizes": [576],
        },
        "qap": {"qap_w1": copy.deepcopy(qap), "qap_w4": copy.deepcopy(qap)},
    }


def _assert_both_accept(payload: dict[str, object]) -> None:
    evaluator._validate_derivation_diagnostics(copy.deepcopy(payload))
    verifier._verify_phase_a_derivation_diagnostics(copy.deepcopy(payload))


def _assert_both_reject(payload: dict[str, object]) -> None:
    with pytest.raises((RuntimeError, verifier.VerificationError)):
        evaluator._validate_derivation_diagnostics(copy.deepcopy(payload))
    with pytest.raises(verifier.VerificationError):
        verifier._verify_phase_a_derivation_diagnostics(copy.deepcopy(payload))


def test_producer_and_verifier_accept_exact_nested_diagnostics() -> None:
    _assert_both_accept(_valid_diagnostics())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("hbt_outside_logits"),
        lambda value: value.__setitem__("extra", {}),
        lambda value: value["hbt_outside_logits"].__setitem__("dtype", "float64"),
        lambda value: value["hbt_outside_logits"].__setitem__("shape", [576, 5]),
        lambda value: value["hbt_outside_logits"].__setitem__("c_order_sha256", "x" * 64),
        lambda value: value["softcycle"].__setitem__("component_sizes", [575]),
        lambda value: value["softcycle"].update(
            {"accepted_edges": 573, "component_sizes": [575, 1]}
        ),
        lambda value: value["softcycle"].update(
            {"accepted_edges": 574, "component_sizes": [1, 575]}
        ),
        lambda value: value["qap"]["qap_w1"].__setitem__("extra", 1),
        lambda value: value["qap"]["qap_w1"].__setitem__("objective", float("nan")),
        lambda value: value["qap"]["qap_w1"].__setitem__("iterations", 24),
        lambda value: value["qap"]["qap_w1"].__setitem__("restart", 2),
        lambda value: value["qap"]["qap_w1"].__setitem__("converged", 0),
    ],
)
def test_producer_and_verifier_reject_nested_diagnostics_drift(mutation) -> None:
    payload = _valid_diagnostics()
    mutation(payload)
    _assert_both_reject(payload)


def test_all_saved_phase_a_diagnostics_are_in_v4_lockstep() -> None:
    manifest = json.loads(
        (
            REPO
            / "runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_readback/"
            "candidate_graph_oracle_v3_phase_a/finalized/"
            "FROZEN_CANDIDATE_GRAPH_MANIFEST.json"
        ).read_text(encoding="utf-8")
    )
    records = manifest["records"]
    assert len(records) == 64
    for record in records:
        _assert_both_accept(record["derivation_diagnostics"])
