from __future__ import annotations

from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

import pytest

from scripts import push_candidate_graph_oracle_v4_phase_a as launch


TITLE = "VSOS Candidate Graph Oracle V4 Phase A T4x2"
RUNNER_BYTES = b"print('phase-a')\n"
RUNNER_SHA256 = hashlib.sha256(RUNNER_BYTES).hexdigest()
SYNTHETIC_RESERVED_KERNEL_ID = 987654321


@pytest.fixture(autouse=True)
def _bind_synthetic_reserved_kernel_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise launch semantics without changing the unresolved source template."""

    monkeypatch.setattr(launch, "EXPECTED_KERNEL_ID", SYNTHETIC_RESERVED_KERNEL_ID)
    monkeypatch.setattr(launch, "RESERVATION_RECEIPT_SHA256", "d" * 64)


def _metadata() -> dict:
    sources = [f"{slug}/2" for slug in launch.EXPECTED_DATASETS.values()]
    return {
        "id": launch.EXPECTED_KERNEL_SLUG,
        "id_no": launch.EXPECTED_KERNEL_ID,
        "reservation_receipt_sha256": launch.RESERVATION_RECEIPT_SHA256,
        "title": TITLE,
        "code_file": "run_phase_a.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": sources,
        "oracle_launch_expectation": {
            "kernel_id": launch.EXPECTED_KERNEL_ID,
            "kernel_slug": launch.EXPECTED_KERNEL_SLUG,
            "kernel_version": launch.EXPECTED_KERNEL_VERSION,
            "reservation_receipt_sha256": launch.RESERVATION_RECEIPT_SHA256,
            "dataset_versions": {
                label: {"slug": slug, "version": 2}
                for label, slug in launch.EXPECTED_DATASETS.items()
            },
        },
    }


class FakeApi:
    def __init__(
        self,
        *,
        dataset_version: int = 2,
        current_version: int = 1,
        tamper_current_source: bool = False,
        versioned_dataset_sources: bool = False,
        push_raises_after_remote_commit: bool = False,
        push_response_version: int = 2,
        push_response_kernel_id: int | None = None,
        push_response_ref: str = launch.EXPECTED_KERNEL_SLUG,
    ) -> None:
        self.dataset_version = dataset_version
        self.current_version = current_version
        self.tamper_current_source = tamper_current_source
        self.versioned_dataset_sources = versioned_dataset_sources
        self.push_raises_after_remote_commit = push_raises_after_remote_commit
        self.push_response_version = push_response_version
        self.push_response_kernel_id = (
            launch.EXPECTED_KERNEL_ID
            if push_response_kernel_id is None
            else push_response_kernel_id
        )
        self.push_response_ref = push_response_ref
        self.push_calls = 0

    def dataset_status(self, dataset: str, format: str) -> str:
        assert dataset in launch.EXPECTED_DATASETS.values()
        assert format == "json(status,current_version_number)"
        return json.dumps(
            {"status": "ready", "current_version_number": self.dataset_version}
        )

    def candidate_graph_oracle_current_readback(self):
        if self.current_version == 1:
            source_sha256 = launch.RESERVATION_RUNNER_SHA256
            dataset_sources: list[str] = []
        else:
            source_sha256 = RUNNER_SHA256
            dataset_sources = list(launch.EXPECTED_DATASETS.values())
            if self.versioned_dataset_sources:
                dataset_sources = [f"{value}/2" for value in dataset_sources]
        if self.tamper_current_source and self.current_version >= 2:
            source_sha256 = "f" * 64
        return (
            {
                "id": launch.EXPECTED_KERNEL_ID,
                "ref": launch.EXPECTED_KERNEL_SLUG,
                "title": TITLE,
                "slug": launch.EXPECTED_KERNEL_SLUG.split("/", 1)[1],
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu_observation": False,
                "enable_internet": False,
                "enable_tpu_observation": False,
                "dataset_sources": dataset_sources,
                "kernel_sources": [],
                "competition_sources": [],
                "model_sources": [],
                "current_version_number": self.current_version,
                "docker_image": "gcr.io/kaggle-private-byod/python@sha256:test",
                "machine_shape_observation": None,
            },
            source_sha256,
        )

    def kernels_push(self, folder: str, timeout, acc):
        assert Path(folder).is_dir()
        self.push_calls += 1
        self.current_version = 2
        if self.push_raises_after_remote_commit:
            raise RuntimeError("connection dropped after remote commit")
        return SimpleNamespace(
            error="",
            error_message="",
            kernel_session_id=0,
            machine_shape=None,
            invalid_dataset_sources=[],
            invalid_competition_sources=[],
            invalid_kernel_sources=[],
            invalid_model_sources=[],
            kernel_id=self.push_response_kernel_id,
            version_number=self.push_response_version,
            ref=self.push_response_ref,
            url="https://www.kaggle.com/code/pasha883/oracle",
        )


def _job(tmp_path: Path) -> Path:
    root = tmp_path / "job"
    root.mkdir()
    (root / "kernel-metadata.json").write_text(
        json.dumps(_metadata()), encoding="utf-8"
    )
    (root / "run_phase_a.py").write_bytes(RUNNER_BYTES)
    return root


def test_unresolved_v4_kernel_id_refuses_before_filesystem_or_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launch, "EXPECTED_KERNEL_ID", -1)
    api = FakeApi()
    with pytest.raises(RuntimeError, match="reservation id is unresolved"):
        launch.push_and_record(
            job_dir=tmp_path / "missing-job",
            receipt_path=tmp_path / "receipt.json",
            api=api,
        )
    assert api.push_calls == 0
    assert not (tmp_path / "receipt.json").exists()


def test_zero_v4_kernel_id_refuses_before_filesystem_or_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launch, "EXPECTED_KERNEL_ID", 0)
    api = FakeApi()
    with pytest.raises(RuntimeError, match="non-positive"):
        launch.push_and_record(
            job_dir=tmp_path / "missing-job",
            receipt_path=tmp_path / "receipt.json",
            api=api,
        )
    assert api.push_calls == 0
    assert not (tmp_path / "receipt.json").exists()


def test_push_writes_exact_self_hashed_launch_receipt(tmp_path: Path) -> None:
    api = FakeApi()
    receipt = tmp_path / "launch.json"
    result = launch.push_and_record(
        job_dir=_job(tmp_path), receipt_path=receipt, api=api
    )
    assert api.push_calls == 1
    assert result["kernel_version"] == 2
    envelope = json.loads(receipt.read_text())
    assert set(envelope) == {"payload", "payload_sha256"}
    payload = envelope["payload"]
    assert payload["schema_version"] == 2
    assert payload["kernel"]["kernel_id"] == launch.EXPECTED_KERNEL_ID
    assert payload["push_response"]["version_number"] == 2
    assert payload["push_response"]["raw_response_file"] == launch.RAW_RESPONSE_NAME
    assert payload["dataset_versions_before_push"]["input"]["version"] == 2
    assert payload["dataset_versions_after_push"] == payload[
        "dataset_versions_before_push"
    ]
    assert payload["server_readback"]["metadata"]["enable_gpu_observation"] is False
    assert payload["server_readback"]["metadata"]["machine_shape_observation"] is None
    assert payload["server_readback"]["source_sha256"] == RUNNER_SHA256
    assert envelope["payload_sha256"] == hashlib.sha256(
        launch._canonical_object_bytes(payload)
    ).hexdigest()
    assert envelope["payload_sha256"] != hashlib.sha256(
        launch._canonical_bytes(payload)
    ).hexdigest()
    state = Path(result["state_dir"])
    assert hashlib.sha256((state / launch.INTENT_NAME).read_bytes()).hexdigest() == result[
        "intent_sha256"
    ]
    assert hashlib.sha256(
        (state / launch.RAW_RESPONSE_NAME).read_bytes()
    ).hexdigest() == result["raw_push_response_sha256"]
    assert hashlib.sha256((state / launch.RESPONSE_NAME).read_bytes()).hexdigest() == result[
        "push_response_sha256"
    ]
    raw = json.loads((state / launch.RAW_RESPONSE_NAME).read_text())
    assert raw["object_state"]["error_message"] == ""
    assert raw["object_state"]["kernel_session_id"] == 0
    assert raw["object_state"]["machine_shape"] is None
    assert payload["raw_push_response"] == raw


def test_exact_code_ref_alias_writes_normal_validated_receipt(tmp_path: Path) -> None:
    api = FakeApi(push_response_ref=f"/code/{launch.EXPECTED_KERNEL_SLUG}")
    receipt = tmp_path / "launch_alias.json"
    result = launch.push_and_record(
        job_dir=_job(tmp_path), receipt_path=receipt, api=api
    )
    assert api.push_calls == 1
    envelope = json.loads(receipt.read_text())
    payload = envelope["payload"]
    assert payload["schema_version"] == 2
    assert payload["kind"] == "candidate_graph_oracle_kaggle_launch_receipt"
    assert payload["raw_push_response"]["public_fields"]["ref"] == (
        f"/code/{launch.EXPECTED_KERNEL_SLUG}"
    )
    assert payload["push_response"]["ref"] == launch.EXPECTED_KERNEL_SLUG
    assert payload["kernel"]["slug"] == launch.EXPECTED_KERNEL_SLUG
    assert payload["push_response_recovered_from_raw_journal"] is False
    assert envelope["payload_sha256"] == hashlib.sha256(
        launch._canonical_object_bytes(payload)
    ).hexdigest()
    assert result["kernel_version"] == 2


@pytest.mark.parametrize(
    "bad_ref",
    [
        lambda slug: f"code/{slug}",
        lambda slug: f"/code//{slug}",
        lambda slug: f"/code/{slug}/",
        lambda slug: f"/CODE/{slug}",
    ],
)
def test_near_code_ref_aliases_fail_closed(
    tmp_path: Path, bad_ref
) -> None:
    api = FakeApi(push_response_ref=bad_ref(launch.EXPECTED_KERNEL_SLUG))
    state = tmp_path / "state"
    with pytest.raises(RuntimeError, match="frozen launch expectation"):
        launch.push_and_record(
            job_dir=_job(tmp_path),
            receipt_path=tmp_path / "launch.json",
            state_dir=state,
            api=api,
        )
    assert api.push_calls == 1
    raw = json.loads((state / launch.RAW_RESPONSE_NAME).read_text())
    assert raw["public_fields"]["ref"] == bad_ref(launch.EXPECTED_KERNEL_SLUG)
    assert not (state / launch.RESPONSE_NAME).exists()


def test_push_refuses_stale_dataset_before_kernel_write(tmp_path: Path) -> None:
    api = FakeApi(dataset_version=1)
    with pytest.raises(RuntimeError, match="not frozen at ready version 2"):
        launch.push_and_record(
            job_dir=_job(tmp_path),
            receipt_path=tmp_path / "launch.json",
            api=api,
        )
    assert api.push_calls == 0


def test_retry_uses_durable_push_response_without_creating_v3(tmp_path: Path) -> None:
    api = FakeApi()
    job = _job(tmp_path)
    state = tmp_path / "state"
    first = launch.push_and_record(
        job_dir=job,
        receipt_path=tmp_path / "launch_first.json",
        state_dir=state,
        api=api,
    )
    second = launch.push_and_record(
        job_dir=job,
        receipt_path=tmp_path / "launch_retry.json",
        state_dir=state,
        api=api,
    )
    assert first["kernel_version"] == second["kernel_version"] == 2
    assert api.push_calls == 1
    retry_payload = json.loads((tmp_path / "launch_retry.json").read_text())[
        "payload"
    ]
    assert retry_payload["push_performed_in_this_process"] is False


def test_crash_gap_after_remote_commit_is_unrecoverable_and_never_pushes_v3(
    tmp_path: Path,
) -> None:
    api = FakeApi(push_raises_after_remote_commit=True)
    job = _job(tmp_path)
    state = tmp_path / "state"
    with pytest.raises(RuntimeError, match="connection dropped"):
        launch.push_and_record(
            job_dir=job,
            receipt_path=tmp_path / "launch.json",
            state_dir=state,
            api=api,
        )
    assert api.push_calls == 1
    assert (state / launch.INTENT_NAME).is_file()
    assert not (state / launch.RAW_RESPONSE_NAME).exists()
    assert not (state / launch.RESPONSE_NAME).exists()
    with pytest.raises(
        RuntimeError, match="refusing any retry that could create another kernel version"
    ):
        launch.push_and_record(
            job_dir=job,
            receipt_path=tmp_path / "launch_retry.json",
            state_dir=state,
            api=api,
        )
    assert api.push_calls == 1


@pytest.mark.parametrize(
    ("response_version", "response_kernel_id"),
    [(3, launch.EXPECTED_KERNEL_ID), (2, launch.EXPECTED_KERNEL_ID + 1)],
)
def test_push_rejects_wrong_exact_response_version_or_kernel_id(
    tmp_path: Path, response_version: int, response_kernel_id: int
) -> None:
    api = FakeApi(
        push_response_version=response_version,
        push_response_kernel_id=response_kernel_id,
    )
    state = tmp_path / "state"
    with pytest.raises(RuntimeError, match="push response violates"):
        launch.push_and_record(
            job_dir=_job(tmp_path),
            receipt_path=tmp_path / "launch.json",
            state_dir=state,
            api=api,
        )
    assert api.push_calls == 1
    assert (state / launch.RAW_RESPONSE_NAME).is_file()
    assert not (state / launch.RESPONSE_NAME).exists()

    # The rejected raw response is terminal for this parser and retry never
    # reaches the remote write endpoint again.
    with pytest.raises(RuntimeError, match="push response violates"):
        launch.push_and_record(
            job_dir=tmp_path / "job",
            receipt_path=tmp_path / "launch_retry.json",
            state_dir=state,
            api=api,
        )
    assert api.push_calls == 1


def test_parser_failure_after_raw_commit_recovers_without_second_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeApi()
    job = _job(tmp_path)
    state = tmp_path / "state"
    original_parser = launch._push_response_payload

    def reject_after_raw_commit(*args, **kwargs):
        raise RuntimeError("synthetic parser rejection")

    monkeypatch.setattr(launch, "_push_response_payload", reject_after_raw_commit)
    with pytest.raises(RuntimeError, match="synthetic parser rejection"):
        launch.push_and_record(
            job_dir=job,
            receipt_path=tmp_path / "launch_failed.json",
            state_dir=state,
            api=api,
        )
    assert api.push_calls == 1
    assert (state / launch.RAW_RESPONSE_NAME).is_file()
    assert not (state / launch.RESPONSE_NAME).exists()

    monkeypatch.setattr(launch, "_push_response_payload", original_parser)
    result = launch.push_and_record(
        job_dir=job,
        receipt_path=tmp_path / "launch_recovered.json",
        state_dir=state,
        api=api,
    )
    assert api.push_calls == 1
    assert result["kernel_version"] == 2
    recovered = json.loads((tmp_path / "launch_recovered.json").read_text())[
        "payload"
    ]
    assert recovered["push_performed_in_this_process"] is False
    assert recovered["push_response_recovered_from_raw_journal"] is True


def test_alias_raw_journal_recovers_to_normal_receipt_without_second_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeApi(push_response_ref=f"/code/{launch.EXPECTED_KERNEL_SLUG}")
    job = _job(tmp_path)
    state = tmp_path / "state"
    original_parser = launch._push_response_payload

    def reject_after_raw_commit(*args, **kwargs):
        raise RuntimeError("synthetic alias parser rejection")

    monkeypatch.setattr(launch, "_push_response_payload", reject_after_raw_commit)
    with pytest.raises(RuntimeError, match="synthetic alias parser rejection"):
        launch.push_and_record(
            job_dir=job,
            receipt_path=tmp_path / "launch_failed.json",
            state_dir=state,
            api=api,
        )
    assert api.push_calls == 1
    assert (state / launch.RAW_RESPONSE_NAME).is_file()
    assert not (state / launch.RESPONSE_NAME).exists()

    monkeypatch.setattr(launch, "_push_response_payload", original_parser)
    launch.push_and_record(
        job_dir=job,
        receipt_path=tmp_path / "launch_recovered.json",
        state_dir=state,
        api=api,
    )
    assert api.push_calls == 1
    payload = json.loads((tmp_path / "launch_recovered.json").read_text())["payload"]
    assert payload["raw_push_response"]["public_fields"]["ref"] == (
        f"/code/{launch.EXPECTED_KERNEL_SLUG}"
    )
    assert payload["push_response"]["ref"] == launch.EXPECTED_KERNEL_SLUG
    assert payload["kernel"]["slug"] == launch.EXPECTED_KERNEL_SLUG
    assert payload["push_response_recovered_from_raw_journal"] is True


def test_existing_v2_without_response_journal_never_creates_v3(tmp_path: Path) -> None:
    api = FakeApi(current_version=2)
    with pytest.raises(RuntimeError, match="current kernel differs"):
        launch.push_and_record(
            job_dir=_job(tmp_path),
            receipt_path=tmp_path / "launch.json",
            state_dir=tmp_path / "fresh_state",
            api=api,
        )
    assert api.push_calls == 0


def test_post_push_readback_rejects_tampered_source_and_versioned_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launch.time, "sleep", lambda _: None)
    for index, kwargs in enumerate(
        ({"tamper_current_source": True}, {"versioned_dataset_sources": True})
    ):
        root = tmp_path / str(index)
        root.mkdir()
        api = FakeApi(**kwargs)
        with pytest.raises(RuntimeError, match="did not become verifiable"):
            launch.push_and_record(
                job_dir=_job(root),
                receipt_path=root / "launch.json",
                state_dir=root / "state",
                api=api,
            )
        assert api.push_calls == 1
