from __future__ import annotations

import json
import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest

from puzzle_assembly.protocol import source_names_for_split


REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE = REPO_ROOT / "runs/assembly_v1/kaggle"
JOBS = {
    "stage1": KAGGLE / "masked_gap_stage1_job",
    "phase_a": KAGGLE / "masked_gap_phasea_job",
    "phase_b": KAGGLE / "masked_gap_phaseb_job",
    "holdout_prepare": KAGGLE / "masked_gap_holdout_prepare_job",
}
SCRIPTS = {
    "stage1": "run_stage1_train_prepare.py",
    "phase_a": "run_phasea_authorize.py",
    "phase_b": "run_phaseb_isolated.py",
    "holdout_prepare": "run_holdout_prepare.py",
}
REQUIRED_LOCAL_FILES = {
    "scripts/train_evaluate_masked_gap.py",
    "src/puzzle_assembly/__init__.py",
    "src/puzzle_assembly/compatibility.py",
    "src/puzzle_assembly/components.py",
    "src/puzzle_assembly/geometry.py",
    "src/puzzle_assembly/learned.py",
    "src/puzzle_assembly/masked_gap.py",
    "src/puzzle_assembly/metrics.py",
    "src/puzzle_assembly/panels.py",
    "src/puzzle_assembly/protocol.py",
    "src/puzzle_assembly/solvers.py",
    "src/puzzle_denoise_v2/__init__.py",
    "src/puzzle_denoise_v2/degradation.py",
    "src/puzzle_denoise_v2/inference.py",
    "src/puzzle_denoise_v2/losses.py",
    "src/puzzle_denoise_v2/metrics.py",
    "src/puzzle_denoise_v2/model.py",
    "src/puzzle_denoise_v2/tiles.py",
    "src/puzzle_denoise_v2/training.py",
    "configs/denoise_splits_seed20260710.json",
    "configs/denoise_validation_quarantine_v1.json",
}


def _metadata(name: str) -> dict:
    return json.loads((JOBS[name] / "kernel-metadata.json").read_text(encoding="utf-8"))


def _source(name: str) -> str:
    return (JOBS[name] / SCRIPTS[name]).read_text(encoding="utf-8")


def _load_job(name: str):
    path = JOBS[name] / SCRIPTS[name]
    spec = importlib.util.spec_from_file_location(f"masked_gap_{name}_job", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_metadata_enforces_physical_dataset_isolation() -> None:
    stage1 = _metadata("stage1")
    phase_a = _metadata("phase_a")
    phase_b = _metadata("phase_b")
    holdout = _metadata("holdout_prepare")
    for metadata in (stage1, phase_a, phase_b, holdout):
        assert metadata["enable_gpu"] is True
        assert metadata["machine_shape"] == "NvidiaTeslaT4"
        assert metadata["enable_internet"] is False
    assert any("pazzle" in source for source in stage1["dataset_sources"])
    assert any("pazzle" in source for source in holdout["dataset_sources"])
    assert any("CALIBRATION_B" in source for source in holdout["dataset_sources"])
    assert phase_a["dataset_sources"] == [
        "pasha883/vsos-masked-gap-gate-code",
        "pasha883/vsos-masked-gap-stage1-checkpoint-v1",
        "pasha883/vsos-masked-gap-calb-input-v1",
    ]
    assert not any(
        token in " ".join(phase_a["dataset_sources"]).lower()
        for token in ("puzzle", "pazzle", "label", "target")
    )
    assert not any(
        token in " ".join(phase_b["dataset_sources"]).lower()
        for token in ("puzzle", "pazzle", "train-target", "train_target")
    )
    assert any("calb-labels" in source for source in phase_b["dataset_sources"])
    assert any("calb-phasea" in source for source in phase_b["dataset_sources"])


def test_phase_a_accepts_current_kaggle_nested_dataset_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_job("phase_a")
    nested = tmp_path / "datasets" / "pasha883"
    for source in module.EXPECTED_DATASET_SOURCES:
        (nested / source.rsplit("/", 1)[-1]).mkdir(parents=True)
    monkeypatch.setattr(module, "INPUT", tmp_path)
    isolation = module.verify_dataset_isolation()
    assert isolation["mount_layout"] == "kaggle_datasets_hierarchy"
    assert isolation["mounted_names"] == sorted(module.EXPECTED_DATASET_SOURCES)

    (nested / "unexpected-dataset").mkdir()
    with pytest.raises(RuntimeError, match="differs from allowlist"):
        module.verify_dataset_isolation()


def test_every_gpu_stage_uses_exact_two_process_torchrun_and_no_dataparallel() -> None:
    for name in JOBS:
        source = _source(name)
        assert '"torch.distributed.run"' in source
        assert '"--nproc_per_node=2"' in source
        assert "DataParallel" not in source
        assert "--data-parallel" not in source
        assert "chmod" not in source
        prepared = json.loads(
            (JOBS[name] / "prepared-not-pushed.json").read_text(encoding="utf-8")
        )
        assert prepared["status"] == "prepared_not_pushed"


def test_every_wrapper_pins_complete_recursive_local_import_closure() -> None:
    for name in JOBS:
        source = _source(name)
        for relative in REQUIRED_LOCAL_FILES:
            assert f'"{relative}"' in source
        assert '"masked_gap_recursive_code_manifest_v1"' in source
        assert "code hash mismatch" in source or "recursive code hash mismatch" in source


def test_staged_code_manifest_and_every_member_are_hash_pinned() -> None:
    root = KAGGLE / "masked_gap_gate_code_dataset"
    manifest_path = root / "masked_gap_code_manifest_v1.json"
    manifest_hash = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["file_sha256"]) == REQUIRED_LOCAL_FILES
    for relative, expected in manifest["file_sha256"].items():
        assert _sha256(root / relative) == expected
    for name in JOBS:
        assert f'EXPECTED_CODE_MANIFEST_SHA256 = "{manifest_hash}"' in _source(name)
    stage1 = _source("stage1")
    assert "__CAPACITY_REPORT_SHA256__" not in stage1
    assert "__CAPACITY_WRAPPER_SHA256__" not in stage1

    expected_archive_files = {"masked_gap_code_manifest_v1.json", *REQUIRED_LOCAL_FILES}
    archive_path = root / "upload_v1/masked_gap_gate_code.zip"
    with zipfile.ZipFile(archive_path) as archive:
        archive_files = {info.filename for info in archive.infolist() if not info.is_dir()}
        assert archive_files == expected_archive_files
        assert hashlib.sha256(
            archive.read("masked_gap_code_manifest_v1.json")
        ).hexdigest() == manifest_hash
        for relative, expected in manifest["file_sha256"].items():
            assert hashlib.sha256(archive.read(relative)).hexdigest() == expected

    roundtrip = root / "roundtrip_v1"
    assert _sha256(roundtrip / "masked_gap_code_manifest_v1.json") == manifest_hash
    for relative, expected in manifest["file_sha256"].items():
        assert _sha256(roundtrip / relative) == expected


def test_phase_b_wrapper_hashes_target_blind_inputs_before_label_manifest() -> None:
    source = _source("phase_b")
    auth = source.index('authorization = find_hash_pinned("phase_b_authorization.json"')
    label = source.index('label_manifest = find_hash_pinned("label_manifest.json"')
    label_verify = source.index("verify_manifest_records(", label)
    assert auth < label < label_verify
    evaluator = (REPO_ROOT / "scripts/train_evaluate_masked_gap.py").read_text(encoding="utf-8")
    authorization_check = evaluator.index('raise RuntimeError("missing global Phase B authorization")')
    first_label_access = evaluator.index("# This is the first label-manifest access in the function.")
    assert authorization_check < first_label_access


def test_holdout_prepare_checks_passing_calibration_before_puzzle_access() -> None:
    main = _source("holdout_prepare").split("def main() -> None:", 1)[1]
    calibration_check = main.index("holdout remains sealed because calibration B did not pass")
    puzzle_access = main.index("data_root()")
    assert calibration_check < puzzle_access


def test_secret_seed_material_stays_out_of_stage_reports_and_phase_a() -> None:
    for name in ("stage1", "holdout_prepare"):
        report_block = _source(name).rsplit("report = {", 1)[1]
        assert "secret_seed_mapping" not in report_block
        assert "label_only_archive_sha256" not in report_block
    phase_b_report_block = _source("phase_b").rsplit("report = {", 1)[1]
    assert "label_only_archive_sha256" not in phase_b_report_block
    phase_a = _source("phase_a")
    assert "secret_panel_seeds" not in phase_a
    assert "secret-seed-mapping" not in phase_a
    assert not any("label" in value.lower() for value in _metadata("phase_a")["dataset_sources"])


def test_label_archive_preserves_secret_seeds_reproducibly(tmp_path: Path) -> None:
    runner = _load_job("stage1")
    labels = tmp_path / "labels_only"
    labels.mkdir()
    seeds = [2**63 + index for index in range(8)]
    payload = {
        "kind": "masked_gap_label_manifest_v1",
        "split": "calibration_b",
        "records": [{"seed": seed} for seed in seeds],
    }
    (labels / "label_manifest.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    assert runner.deterministic_zip(labels, first) == runner.deterministic_zip(labels, second)
    with zipfile.ZipFile(first) as archive:
        restored = json.loads(archive.read("label_manifest.json"))
    assert [record["seed"] for record in restored["records"]] == seeds


def test_secret_seed_wrappers_use_exact_frozen_names_and_complete_unique_uint64_maps(
    tmp_path: Path,
) -> None:
    development = source_names_for_split(
        "edge_development",
        manifest_path=REPO_ROOT / "configs/denoise_splits_seed20260710.json",
        quarantine_path=REPO_ROOT / "configs/denoise_validation_quarantine_v1.json",
    )
    cases = (
        ("stage1", "calibration_b", development[388:392]),
        ("holdout_prepare", "holdout", development[392:400]),
    )
    for job_name, split, expected_names in cases:
        runner = _load_job(job_name)
        names = runner.frozen_gate_names(REPO_ROOT, split)
        assert names == expected_names
        path = tmp_path / job_name / "labels_only" / "secret_panel_seeds.json"
        runner.write_secret_seed_mapping(path, split=split, names=names)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) == {"kind", "split", "records"}
        assert payload["kind"] == "masked_gap_secret_panel_seed_mapping_v1"
        assert payload["split"] == split
        expected_identities = [
            (name, panel)
            for name in names
            for panel in ("primary_kornia", "independent_libjpeg")
        ]
        actual_identities = [
            (record["name"], record["panel"]) for record in payload["records"]
        ]
        seeds = [record["seed"] for record in payload["records"]]
        assert actual_identities == expected_identities
        assert len(seeds) == len(set(seeds)) == len(expected_identities)
        assert all(
            isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**64
            for seed in seeds
        )
