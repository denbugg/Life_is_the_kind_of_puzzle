from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import diagnose_candidate_graph_oracle_v3_for_v4 as diagnostic


EXPECTED = "owner/fresh-oracle-v4-phase-a"


@pytest.mark.parametrize("raw", [EXPECTED, f"/code/{EXPECTED}"])
def test_v4_ref_contract_normalizes_only_two_exact_forms(raw: str) -> None:
    assert diagnostic.normalize_exact_kaggle_kernel_ref(raw, EXPECTED) == EXPECTED


@pytest.mark.parametrize(
    "raw",
    [
        f"/code/{EXPECTED}/",
        f"https://www.kaggle.com/code/{EXPECTED}",
        f"code/{EXPECTED}",
        f"/CODE/{EXPECTED}",
        f"/code//{EXPECTED}",
        "other/fresh-oracle-v4-phase-a",
        None,
    ],
)
def test_v4_ref_contract_rejects_near_aliases(raw: object) -> None:
    with pytest.raises(RuntimeError, match="exact /code alias|exact string"):
        diagnostic.normalize_exact_kaggle_kernel_ref(raw, EXPECTED)


def test_forbidden_namespace_is_rejected_lexically_before_resolution(
    tmp_path: Path,
) -> None:
    forbidden = tmp_path / "fixture_label" / "never_touch.json"
    with pytest.raises(RuntimeError, match="forbidden namespace"):
        diagnostic._guard_path(
            forbidden, label="synthetic_forbidden", must_exist=False
        )


def test_latest_diagnostic_report_can_never_accept_or_resume_v3() -> None:
    report = (
        diagnostic.REPO_ROOT
        / "runs/assembly_v1/kaggle/candidate_graph_oracle_v3_phase_a_readback/"
        "V4_PREFLIGHT_DIAGNOSTIC_ONLY_V3.json"
    )
    envelope = json.loads(report.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    assert payload["accepted_v3_result"] is False
    assert payload["phase_b_authorized"] is False
    assert payload["label_access_claimed"] is False
    assert payload["safe_for_submission"] is False
    assert payload["worker"]["records"] == 64
    assert payload["worker"]["graph_artifacts"] == 64
    assert payload["worker"]["renders"] == 192
