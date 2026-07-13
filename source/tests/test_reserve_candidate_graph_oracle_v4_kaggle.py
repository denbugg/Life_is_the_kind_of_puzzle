from __future__ import annotations

import base64
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from types import SimpleNamespace

import pytest
from requests.exceptions import HTTPError


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/reserve_candidate_graph_oracle_v4_kaggle.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "reserve_candidate_graph_oracle_v4_kaggle_for_tests", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
reservation = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = reservation
_SPEC.loader.exec_module(reservation)

_TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_reservations"
)
_NOW = lambda: "2026-07-12T00:00:00Z"


def _copy_templates(tmp_path: Path) -> Path:
    root = tmp_path / "candidate_graph_oracle_v4_reservations"
    shutil.copytree(_TEMPLATE_ROOT, root)
    for evidence in (root / "journal").glob("*.json"):
        evidence.unlink()
    receipt = root / "RESERVATION_RECEIPT.json"
    if receipt.exists():
        receipt.unlink()
    return root


class FakeReservationAPI:
    def __init__(
        self,
        root: Path,
        *,
        existing_datasets: bool = False,
        existing_kernel: bool = False,
        fail_kernel_create: bool = False,
        commit_then_fail_kernel_create: bool = False,
    ) -> None:
        self.root = root
        self.dataset_snapshots: dict[str, dict[str, object]] = {}
        self.kernel_snapshot: dict[str, object] | None = None
        self.fail_kernel_create = fail_kernel_create
        self.commit_then_fail_kernel_create = commit_then_fail_kernel_create
        self.dataset_snapshot_sequences: dict[
            str, list[dict[str, object] | None]
        ] = {}
        self.kernel_snapshot_sequence: list[dict[str, object] | None] = []
        self.create_dataset_calls: list[str] = []
        self.create_kernel_calls = 0
        self.read_dataset_calls: list[str] = []
        self.read_kernel_calls = 0
        if existing_datasets:
            for index, spec in enumerate(reservation.DATASET_SPECS, start=1):
                self.dataset_snapshots[spec.slug] = self._dataset_snapshot(spec, 1000 + index)
        if existing_kernel:
            self.kernel_snapshot = self._kernel_snapshot(424242)

    def _dataset_snapshot(self, spec, dataset_id: int) -> dict[str, object]:
        marker = reservation._marker_bytes(spec.role)
        return {
            "dataset": {
                "id": dataset_id,
                "ref": spec.slug,
                "title": spec.title,
                "is_private": True,
                "current_version_number": 1,
                "total_bytes": len(marker),
            },
            "status": {"status": "ready", "current_version_number": 1},
            "file_list": {
                "files": [
                    {"name": reservation.MARKER_NAME, "total_bytes": len(marker)}
                ],
                "next_page_token": None,
                "error_message": None,
            },
            "marker": {
                "name": reservation.MARKER_NAME,
                "base64": base64.b64encode(marker).decode("ascii"),
            },
            "sdk_objects": {"fake": True},
        }

    def _kernel_snapshot(self, kernel_id: int) -> dict[str, object]:
        source = (self.root / "kernel/reservation_runner.py").read_bytes()
        return {
            "metadata": {
                "id": kernel_id,
                "ref": reservation.KERNEL_SLUG,
                "slug": reservation.KERNEL_SLUG.split("/", 1)[1],
                "title": reservation.KERNEL_TITLE,
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_tpu": False,
                "enable_internet": False,
                "dataset_sources": [],
                "kernel_sources": [],
                "competition_sources": [],
                "model_sources": [],
                "current_version_number": 1,
            },
            "source": {"base64": base64.b64encode(source).decode("ascii")},
            "status": {"status": "complete", "failure_message": ""},
            "sdk_objects": {"fake": True},
        }

    def get_dataset_snapshot(self, slug: str, marker_name: str):
        assert marker_name == reservation.MARKER_NAME
        self.read_dataset_calls.append(slug)
        sequence = self.dataset_snapshot_sequences.get(slug)
        value = sequence.pop(0) if sequence else self.dataset_snapshots.get(slug)
        return copy.deepcopy(value) if value is not None else None

    def create_dataset(self, directory: Path):
        role = directory.name
        spec = next(value for value in reservation.DATASET_SPECS if value.role == role)
        self.create_dataset_calls.append(spec.slug)
        dataset_id = 2000 + len(self.create_dataset_calls)
        self.dataset_snapshots[spec.slug] = self._dataset_snapshot(spec, dataset_id)
        return SimpleNamespace(
            ref=spec.slug,
            url=f"https://www.kaggle.com/datasets/{spec.slug}",
            status="ok",
            error=None,
            invalid_tags=[],
        )

    def get_kernel_snapshot(self, slug: str):
        assert slug == reservation.KERNEL_SLUG
        self.read_kernel_calls += 1
        value = (
            self.kernel_snapshot_sequence.pop(0)
            if self.kernel_snapshot_sequence
            else self.kernel_snapshot
        )
        return copy.deepcopy(value)

    def create_kernel(self, directory: Path):
        assert directory == self.root / "kernel"
        self.create_kernel_calls += 1
        if self.fail_kernel_create:
            raise ConnectionError("simulated connection loss after possible dispatch")
        self.kernel_snapshot = self._kernel_snapshot(424242)
        if self.commit_then_fail_kernel_create:
            raise ConnectionError("simulated connection loss after committed kernel")
        return SimpleNamespace(
            ref=f"/code/{reservation.KERNEL_SLUG}",
            url=f"https://www.kaggle.com/code/{reservation.KERNEL_SLUG}",
            version_number=1,
            error=None,
            invalid_tags=[],
            invalid_dataset_sources=[],
            invalid_competition_sources=[],
            invalid_kernel_sources=[],
            invalid_model_sources=[],
            kernel_id=424242,
        )


def _reserve(root: Path, api: FakeReservationAPI, *, attempts: int = 1):
    return reservation.reserve_and_record(
        reservation_root=root,
        receipt_path=root / "RESERVATION_RECEIPT.json",
        api=api,
        attempts=attempts,
        sleep=lambda _: None,
        now=_NOW,
    )


def _load_receipt(root: Path) -> dict[str, object]:
    return json.loads((root / "RESERVATION_RECEIPT.json").read_text("utf-8"))


def _pending_dataset_snapshot(api: FakeReservationAPI, spec) -> dict[str, object]:
    snapshot = api._dataset_snapshot(spec, 7777)
    snapshot["status"]["status"] = "pending"
    return snapshot


def _pending_kernel_snapshot(api: FakeReservationAPI) -> dict[str, object]:
    snapshot = api._kernel_snapshot(424242)
    snapshot["status"]["status"] = "pending"
    return snapshot


class _ForbiddenGetDatasetClient:
    def __init__(self) -> None:
        self.datasets = SimpleNamespace(
            dataset_api_client=SimpleNamespace(get_dataset=self._get_dataset)
        )

    @staticmethod
    def _get_dataset(_request):
        response = SimpleNamespace(status_code=403)
        raise HTTPError("forbidden dataset read", response=response)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _ForbiddenGetDatasetApi:
    def __init__(self, pages: dict[int, list[object]]) -> None:
        self.pages = pages
        self.list_calls: list[tuple[bool, int]] = []

    @staticmethod
    def build_kaggle_client():
        return _ForbiddenGetDatasetClient()

    def dataset_list(self, *, mine: bool, page: int):
        self.list_calls.append((mine, page))
        return self.pages.get(page, [])


class _ForbiddenGetKernelClient:
    def __init__(self) -> None:
        self.kernels = SimpleNamespace(
            kernels_api_client=SimpleNamespace(get_kernel=self._get_kernel)
        )

    @staticmethod
    def _get_kernel(_request):
        response = SimpleNamespace(status_code=403)
        raise HTTPError("forbidden kernel read", response=response)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _ForbiddenGetKernelApi:
    def __init__(self, pages: dict[int, list[object]]) -> None:
        self.pages = pages
        self.list_calls: list[tuple[bool, int, int]] = []

    @staticmethod
    def build_kaggle_client():
        return _ForbiddenGetKernelClient()

    def kernels_list(self, *, mine: bool, page: int, page_size: int):
        self.list_calls.append((mine, page, page_size))
        return self.pages.get(page, [])


def test_repository_v4_reservation_templates_are_exact_and_pixel_free() -> None:
    result = reservation.validate_local_templates(_TEMPLATE_ROOT)
    assert result["protocol_instance_id"] == reservation.PROTOCOL_INSTANCE_ID
    assert result["contains_fixture_pixels"] is False
    assert result["gpu_requested"] is False
    assert set(result["datasets"]) == {"code", "input", "runtime"}
    for spec in reservation.DATASET_SPECS:
        value = result["datasets"][spec.role]
        assert value["slug"] == spec.slug
        assert value["reserved_version"] == 1
        assert value["expected_private"] is True
        assert {item["path"] for item in value["manifest"]["files"]} == {
            "dataset-metadata.json",
            reservation.MARKER_NAME,
        }
    assert result["kernel"]["reservation_runner_sha256"] == (
        reservation.RESERVATION_RUNNER_SHA256
    )
    assert result["kernel"]["expected_enable_gpu"] is False
    assert result["kernel"]["expected_enable_tpu"] is False
    assert result["kernel"]["expected_enable_internet"] is False
    assert {item["path"] for item in result["kernel"]["manifest"]["files"]} == {
        "kernel-metadata.json",
        "reservation_runner.py",
    }


def test_dataset_create_response_accepts_current_sdk_prefixed_ref() -> None:
    spec = reservation.DATASET_SPECS[0]
    raw = reservation._raw_response_payload(
        SimpleNamespace(
            ref=f"/datasets/{spec.slug}",
            url=f"https://www.kaggle.com/datasets/{spec.slug}",
            status="Ok",
            error=None,
            invalid_tags=[],
        ),
        kind="candidate_graph_oracle_v4_dataset_raw_create_response",
        fields=reservation.DATASET_RAW_FIELDS,
        now=_NOW,
    )
    payload = reservation._dataset_response_payload(
        raw,
        spec=spec,
        raw_file="journal/raw.json",
        raw_sha256="a" * 64,
        now=_NOW,
    )
    assert payload["ref"] == spec.slug


def test_forbidden_unused_private_dataset_slug_requires_exhausted_mine_listing() -> None:
    adapter = object.__new__(reservation.KaggleSdkReservationAPI)
    adapter._api = _ForbiddenGetDatasetApi(
        {
            1: [SimpleNamespace(ref="pasha883/unrelated-private-dataset")],
            2: [],
        }
    )

    assert (
        adapter.get_dataset_snapshot(
            reservation.DATASET_SPECS[0].slug, reservation.MARKER_NAME
        )
        is None
    )
    assert adapter._api.list_calls == [(True, 1), (True, 2)]


def test_forbidden_existing_private_dataset_slug_is_pending_not_absent() -> None:
    spec = reservation.DATASET_SPECS[0]
    adapter = object.__new__(reservation.KaggleSdkReservationAPI)
    adapter._api = _ForbiddenGetDatasetApi(
        {1: [SimpleNamespace(ref=spec.slug)]}
    )

    snapshot = adapter.get_dataset_snapshot(spec.slug, reservation.MARKER_NAME)
    assert snapshot["status"]["status"] == "pending"
    assert adapter._api.list_calls == [(True, 1), (True, 2)]


def test_forbidden_dataset_absence_requires_intended_owner_identity() -> None:
    spec = reservation.DATASET_SPECS[0]
    adapter = object.__new__(reservation.KaggleSdkReservationAPI)
    adapter._api = _ForbiddenGetDatasetApi(
        {1: [SimpleNamespace(ref="someone-else/unrelated")], 2: []}
    )

    with pytest.raises(RuntimeError, match="unable to establish exact dataset absence"):
        adapter.get_dataset_snapshot(spec.slug, reservation.MARKER_NAME)


def test_forbidden_dataset_malformed_empty_slug_fails_closed() -> None:
    spec = reservation.DATASET_SPECS[0]
    adapter = object.__new__(reservation.KaggleSdkReservationAPI)
    adapter._api = _ForbiddenGetDatasetApi(
        {1: [SimpleNamespace(ref="pasha883/")], 2: []}
    )

    with pytest.raises(RuntimeError, match="unable to establish exact dataset absence"):
        adapter.get_dataset_snapshot(spec.slug, reservation.MARKER_NAME)


def test_forbidden_unused_private_kernel_slug_requires_profile_listing() -> None:
    adapter = object.__new__(reservation.KaggleSdkReservationAPI)
    adapter._api = _ForbiddenGetKernelApi(
        {
            1: [SimpleNamespace(ref="pasha883/unrelated-private-kernel")],
            2: [],
        }
    )

    assert adapter.get_kernel_snapshot(reservation.KERNEL_SLUG) is None
    assert adapter._api.list_calls == [(True, 1, 100), (True, 2, 100)]


def test_forbidden_kernel_listing_ignores_only_exact_redacted_sentinel() -> None:
    adapter = object.__new__(reservation.KaggleSdkReservationAPI)
    adapter._api = _ForbiddenGetKernelApi(
        {
            1: [
                SimpleNamespace(
                    ref="", slug="", author="", title="[Private Notebook]"
                ),
                SimpleNamespace(ref="pasha883/unrelated-private-kernel"),
            ],
            2: [],
        }
    )

    assert adapter.get_kernel_snapshot(reservation.KERNEL_SLUG) is None


def test_forbidden_existing_private_kernel_slug_is_pending_not_absent() -> None:
    adapter = object.__new__(reservation.KaggleSdkReservationAPI)
    adapter._api = _ForbiddenGetKernelApi(
        {1: [SimpleNamespace(ref=reservation.KERNEL_SLUG)]}
    )

    snapshot = adapter.get_kernel_snapshot(reservation.KERNEL_SLUG)
    assert snapshot["status"]["status"] == "pending"
    assert adapter._api.list_calls == [(True, 1, 100), (True, 2, 100)]


def test_forbidden_kernel_malformed_empty_slug_fails_closed() -> None:
    adapter = object.__new__(reservation.KaggleSdkReservationAPI)
    adapter._api = _ForbiddenGetKernelApi(
        {1: [SimpleNamespace(ref="pasha883/")], 2: []}
    )

    with pytest.raises(RuntimeError, match="unable to establish exact kernel absence"):
        adapter.get_kernel_snapshot(reservation.KERNEL_SLUG)


def test_validate_only_never_constructs_api(tmp_path: Path, capsys) -> None:
    root = _copy_templates(tmp_path)
    calls = 0

    def forbidden_factory():
        nonlocal calls
        calls += 1
        raise AssertionError("validate-only attempted to construct a remote API")

    assert (
        reservation.main(
            ["--validate-only", "--reservation-root", str(root)],
            api_factory=forbidden_factory,
        )
        == 0
    )
    assert calls == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "candidate_graph_oracle_v4_local_reservation_validation"
    assert not (root / "RESERVATION_RECEIPT.json").exists()
    assert list((root / "journal").iterdir()) == [root / "journal/.gitkeep"]


def test_exact_tree_rejects_any_extra_upload_file(tmp_path: Path) -> None:
    root = _copy_templates(tmp_path)
    (root / "code/fixture.png").write_bytes(b"not allowed")
    with pytest.raises(RuntimeError, match="exact-tree drift"):
        reservation.validate_local_templates(root)


def test_happy_path_journals_every_write_and_self_hashes_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    root = _copy_templates(tmp_path)
    api = FakeReservationAPI(root)
    fsync_calls: list[int] = []
    real_fsync = reservation.os.fsync

    def recording_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(reservation.os, "fsync", recording_fsync)
    result = _reserve(root, api)

    assert result["status"] == "private_version_1_reservations_attested"
    assert result["kernel_id"] == 424242
    assert result["remote_write_calls_performed_in_this_process"] is True
    assert api.create_dataset_calls == [spec.slug for spec in reservation.DATASET_SPECS]
    assert api.create_kernel_calls == 1
    assert fsync_calls

    envelope = _load_receipt(root)
    assert set(envelope) == {"payload", "payload_sha256"}
    assert envelope["payload_sha256"] == reservation._canonical_sha256(
        envelope["payload"]
    )
    payload = envelope["payload"]
    assert payload["contains_fixture_pixels"] is False
    assert payload["gpu_requested"] is False
    assert payload["dataset_v2_uploaded"] is False
    assert payload["phase_a_push_performed"] is False
    assert payload["safe_for_submission"] is False
    assert payload["kernel"]["mode"] == "created"
    assert payload["kernel"]["reserved_version"] == 1
    assert payload["kernel"]["enable_gpu"] is False
    assert payload["kernel"]["journal"]["dispatch"] is not None
    assert payload["kernel"]["journal"]["raw_create_response"] is not None
    for role in ("code", "input", "runtime"):
        assert payload["datasets"][role]["mode"] == "created"
        journal = payload["datasets"][role]["journal"]
        assert all(journal[name] is not None for name in journal)

    for path in (root / "journal").glob("*.json"):
        raw = path.read_bytes()
        assert raw.endswith(b"\n")
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert json.loads(raw.decode("utf-8")) is not None
    assert stat.S_IMODE((root / "RESERVATION_RECEIPT.json").stat().st_mode) == 0o600

    existing = root / "journal/kernel.00_intent.json"
    with pytest.raises(FileExistsError):
        reservation._write_exclusive(existing, {"would": "overwrite"})


def test_exact_existing_private_v1_is_adopted_without_remote_writes(
    tmp_path: Path,
) -> None:
    root = _copy_templates(tmp_path)
    api = FakeReservationAPI(root, existing_datasets=True, existing_kernel=True)
    result = _reserve(root, api)
    payload = _load_receipt(root)["payload"]

    assert result["remote_write_calls_performed_in_this_process"] is False
    assert api.create_dataset_calls == []
    assert api.create_kernel_calls == 0
    assert payload["kernel"]["mode"] == "adopted_existing"
    assert payload["kernel"]["journal"]["dispatch"] is None
    assert payload["kernel"]["journal"]["raw_create_response"] is None
    for role in ("code", "input", "runtime"):
        assert payload["datasets"][role]["mode"] == "adopted_existing"
        assert payload["datasets"][role]["journal"]["dispatch"] is None
        assert payload["datasets"][role]["journal"]["raw_create_response"] is None


def test_pending_existing_dataset_is_polled_then_adopted_without_create(
    tmp_path: Path,
) -> None:
    root = _copy_templates(tmp_path)
    api = FakeReservationAPI(root, existing_datasets=True, existing_kernel=True)
    spec = reservation.DATASET_SPECS[0]
    ready = copy.deepcopy(api.dataset_snapshots[spec.slug])
    api.dataset_snapshot_sequences[spec.slug] = [
        _pending_dataset_snapshot(api, spec),
        ready,
    ]

    _reserve(root, api, attempts=2)
    payload = _load_receipt(root)["payload"]

    assert api.create_dataset_calls == []
    assert payload["datasets"][spec.role]["mode"] == "adopted_existing"
    assert not (root / f"journal/dataset_{spec.role}.01_dispatch.json").exists()


def test_pending_existing_dataset_timeout_fails_closed_without_create(
    tmp_path: Path,
) -> None:
    root = _copy_templates(tmp_path)
    api = FakeReservationAPI(root, existing_datasets=True, existing_kernel=True)
    spec = reservation.DATASET_SPECS[0]
    api.dataset_snapshots[spec.slug] = _pending_dataset_snapshot(api, spec)

    with pytest.raises(RuntimeError, match="refusing create because it could create version 2"):
        _reserve(root, api, attempts=2)

    assert api.create_dataset_calls == []
    assert not (root / f"journal/dataset_{spec.role}.01_dispatch.json").exists()
    assert not (root / "RESERVATION_RECEIPT.json").exists()


def test_pending_existing_kernel_is_polled_then_adopted_without_push(
    tmp_path: Path,
) -> None:
    root = _copy_templates(tmp_path)
    api = FakeReservationAPI(root, existing_datasets=True, existing_kernel=True)
    ready = copy.deepcopy(api.kernel_snapshot)
    api.kernel_snapshot_sequence = [_pending_kernel_snapshot(api), ready]

    _reserve(root, api, attempts=2)
    payload = _load_receipt(root)["payload"]

    assert api.create_kernel_calls == 0
    assert payload["kernel"]["mode"] == "adopted_existing"
    assert not (root / "journal/kernel.01_dispatch.json").exists()


def test_pending_existing_kernel_timeout_fails_closed_without_push(
    tmp_path: Path,
) -> None:
    root = _copy_templates(tmp_path)
    api = FakeReservationAPI(root, existing_datasets=True, existing_kernel=True)
    api.kernel_snapshot = _pending_kernel_snapshot(api)

    with pytest.raises(RuntimeError, match="refusing push because it could create version 2"):
        _reserve(root, api, attempts=2)

    assert api.create_kernel_calls == 0
    assert not (root / "journal/kernel.01_dispatch.json").exists()
    assert not (root / "RESERVATION_RECEIPT.json").exists()


def test_unknown_kernel_commit_is_never_retried_on_same_slug(tmp_path: Path) -> None:
    root = _copy_templates(tmp_path)
    api = FakeReservationAPI(
        root, existing_datasets=True, fail_kernel_create=True
    )

    with pytest.raises(RuntimeError, match="unknown commit state"):
        _reserve(root, api)
    assert api.create_kernel_calls == 1
    assert (root / "journal/kernel.01_dispatch.json").is_file()
    assert not (root / "journal/kernel.02_raw_response.json").exists()
    assert not (root / "RESERVATION_RECEIPT.json").exists()

    with pytest.raises(RuntimeError, match="could create version 2"):
        _reserve(root, api)
    assert api.create_kernel_calls == 1
    assert not (root / "journal/kernel.02_raw_response.json").exists()


def test_raw_kernel_response_recovers_locally_without_second_push(
    tmp_path: Path, monkeypatch
) -> None:
    root = _copy_templates(tmp_path)
    api = FakeReservationAPI(root, existing_datasets=True)
    original = reservation._write_exclusive
    failed_once = False

    def crash_after_raw(path: Path, payload):
        nonlocal failed_once
        if path.name == "kernel.03_response.json" and not failed_once:
            failed_once = True
            raise RuntimeError("simulated parser crash after durable raw response")
        return original(path, payload)

    monkeypatch.setattr(reservation, "_write_exclusive", crash_after_raw)
    with pytest.raises(RuntimeError, match="simulated parser crash"):
        _reserve(root, api)
    assert api.create_kernel_calls == 1
    assert (root / "journal/kernel.02_raw_response.json").is_file()
    assert not (root / "journal/kernel.03_response.json").exists()

    monkeypatch.setattr(reservation, "_write_exclusive", original)
    result = _reserve(root, api)
    assert result["status"] == "private_version_1_reservations_attested"
    assert api.create_kernel_calls == 1
    payload = _load_receipt(root)["payload"]
    assert payload["kernel"]["mode"] == "recovered_raw_response"


def test_kernel_raw_readback_crash_replays_projection_and_persisted_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    root = _copy_templates(tmp_path)
    api = FakeReservationAPI(root, existing_datasets=True, existing_kernel=True)
    original_write = reservation._write_exclusive
    original_receipt = reservation._write_receipt
    failed_once = False

    def crash_before_projected_readback(path: Path, payload):
        nonlocal failed_once
        if path.name == "kernel.05_readback.json" and not failed_once:
            failed_once = True
            raise RuntimeError("simulated crash after durable raw kernel readback")
        return original_write(path, payload)

    monkeypatch.setattr(reservation, "_write_exclusive", crash_before_projected_readback)
    with pytest.raises(RuntimeError, match="after durable raw kernel readback"):
        _reserve(root, api)
    raw_path = root / "journal/kernel.04_raw_readback.json"
    projected_path = root / "journal/kernel.05_readback.json"
    assert raw_path.is_file()
    assert not projected_path.exists()
    assert api.create_kernel_calls == 0

    monkeypatch.setattr(reservation, "_write_exclusive", original_write)

    def crash_before_receipt(path: Path, payload):
        raise RuntimeError("simulated crash before reservation receipt")

    monkeypatch.setattr(reservation, "_write_receipt", crash_before_receipt)
    with pytest.raises(RuntimeError, match="before reservation receipt"):
        _reserve(root, api)
    assert projected_path.is_file()
    assert not (root / "RESERVATION_RECEIPT.json").exists()

    monkeypatch.setattr(reservation, "_write_receipt", original_receipt)
    result = _reserve(root, api)
    assert result["status"] == "private_version_1_reservations_attested"
    readback = json.loads(projected_path.read_text("utf-8"))
    assert readback["raw_readback_sha256"] == reservation._sha256_file(raw_path)
    assert api.create_kernel_calls == 0

    replay = _reserve(root, api)
    assert replay["status"] == "reservation_receipt_already_exists"


def test_invalid_dataset_readback_is_raw_journaled_before_rejection(
    tmp_path: Path,
) -> None:
    root = _copy_templates(tmp_path)
    api = FakeReservationAPI(root, existing_datasets=True, existing_kernel=True)
    spec = reservation.DATASET_SPECS[0]
    api.dataset_snapshots[spec.slug]["dataset"]["current_version_number"] = 2

    with pytest.raises(RuntimeError, match="violates exact private v1 reservation"):
        _reserve(root, api)

    assert (root / f"journal/dataset_{spec.role}.04_raw_readback.json").is_file()
    assert not (root / f"journal/dataset_{spec.role}.05_readback.json").exists()
    assert api.create_dataset_calls == []


def test_invalid_kernel_readback_is_raw_journaled_before_rejection(
    tmp_path: Path,
) -> None:
    root = _copy_templates(tmp_path)
    api = FakeReservationAPI(root, existing_datasets=True, existing_kernel=True)
    assert api.kernel_snapshot is not None
    api.kernel_snapshot["metadata"]["enable_gpu"] = True

    with pytest.raises(RuntimeError, match="violates exact private CPU-only v1 reservation"):
        _reserve(root, api)

    assert (root / "journal/kernel.04_raw_readback.json").is_file()
    assert not (root / "journal/kernel.05_readback.json").exists()
    assert api.create_kernel_calls == 0


def test_commit_then_exception_recovery_reports_current_process_remote_write(
    tmp_path: Path,
) -> None:
    root = _copy_templates(tmp_path)
    api = FakeReservationAPI(
        root,
        existing_datasets=True,
        commit_then_fail_kernel_create=True,
    )

    result = _reserve(root, api)
    payload = _load_receipt(root)["payload"]

    assert api.create_kernel_calls == 1
    assert payload["kernel"]["mode"] == "recovered_unknown_commit"
    assert result["remote_write_calls_performed_in_this_process"] is True


def test_existing_receipt_is_read_only_and_self_hash_checked(tmp_path: Path) -> None:
    root = _copy_templates(tmp_path)
    api = FakeReservationAPI(root, existing_datasets=True, existing_kernel=True)
    _reserve(root, api)
    before_calls = (
        list(api.read_dataset_calls),
        api.read_kernel_calls,
        list(api.create_dataset_calls),
        api.create_kernel_calls,
    )
    result = _reserve(root, api)
    assert result["status"] == "reservation_receipt_already_exists"
    assert (
        list(api.read_dataset_calls),
        api.read_kernel_calls,
        list(api.create_dataset_calls),
        api.create_kernel_calls,
    ) == before_calls

    path = root / "RESERVATION_RECEIPT.json"
    envelope = json.loads(path.read_text("utf-8"))
    envelope["payload"]["kernel"]["kernel_id"] += 1
    path.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="self-hash mismatch"):
        _reserve(root, api)


def test_self_hashed_receipt_cannot_rebind_a_tampered_readback(tmp_path: Path) -> None:
    root = _copy_templates(tmp_path)
    api = FakeReservationAPI(root, existing_datasets=True, existing_kernel=True)
    _reserve(root, api)

    readback_path = root / "journal/dataset_code.05_readback.json"
    readback = json.loads(readback_path.read_text("utf-8"))
    readback["dataset_id"] += 7
    readback_path.write_bytes(reservation._canonical_file_bytes(readback))

    receipt_path = root / "RESERVATION_RECEIPT.json"
    envelope = json.loads(receipt_path.read_text("utf-8"))
    envelope["payload"]["datasets"]["code"]["dataset_id"] += 7
    envelope["payload"]["datasets"]["code"]["journal"]["readback"][
        "sha256"
    ] = reservation._sha256_file(readback_path)
    envelope["payload_sha256"] = reservation._canonical_sha256(
        envelope["payload"]
    )
    receipt_path.write_bytes(reservation._canonical_file_bytes(envelope))

    with pytest.raises(RuntimeError, match="receipt/readback crosslink drift"):
        reservation._validate_existing_receipt(receipt_path)
