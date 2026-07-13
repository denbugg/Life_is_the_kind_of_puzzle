from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import scripts.apply_denoise_v2 as apply_cli
from scripts.apply_denoise_v2 import _plan_jobs, _validate_write_plan


def _touch(path: Path, payload: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_single_png_can_target_existing_or_new_output_directory(tmp_path) -> None:
    source = _touch(tmp_path / "source.png")
    existing = tmp_path / "existing"
    existing.mkdir()

    assert _plan_jobs(source, existing) == [(source, existing / source.name)]
    assert _plan_jobs(source, tmp_path / "new-directory") == [
        (source, tmp_path / "new-directory" / source.name)
    ]
    assert _plan_jobs(source, tmp_path / "renamed.png") == [
        (source, tmp_path / "renamed.png")
    ]


def test_write_plan_rejects_canonical_input_checkpoint_and_report_collisions(tmp_path) -> None:
    source = _touch(tmp_path / "source.png")
    checkpoint = _touch(tmp_path / "checkpoint.pt")

    with pytest.raises(SystemExit, match="collides with an input"):
        _validate_write_plan(
            [(source, source)],
            checkpoint,
            None,
            overwrite=True,
        )
    with pytest.raises(SystemExit, match="collides with an input"):
        _validate_write_plan(
            [(source, checkpoint)],
            checkpoint,
            None,
            overwrite=True,
        )
    output = tmp_path / "output.png"
    with pytest.raises(SystemExit, match="collide after canonical"):
        _validate_write_plan(
            [(source, output)],
            checkpoint,
            output,
            overwrite=True,
        )


def test_write_plan_requires_explicit_overwrite_and_rejects_symlinks(tmp_path) -> None:
    source = _touch(tmp_path / "source.png")
    checkpoint = _touch(tmp_path / "checkpoint.pt")
    output = _touch(tmp_path / "output.png")

    with pytest.raises(SystemExit, match="--overwrite"):
        _validate_write_plan(
            [(source, output)],
            checkpoint,
            None,
            overwrite=False,
        )
    _validate_write_plan(
        [(source, output)],
        checkpoint,
        None,
        overwrite=True,
    )

    hard_link = tmp_path / "hard-link.png"
    hard_link.hardlink_to(source)
    with pytest.raises(SystemExit, match="hard-link collision"):
        _validate_write_plan(
            [(source, hard_link)],
            checkpoint,
            None,
            overwrite=True,
        )

    symlink = tmp_path / "symlink.png"
    symlink.symlink_to(output)
    with pytest.raises(SystemExit, match="symlink"):
        _validate_write_plan(
            [(source, symlink)],
            checkpoint,
            None,
            overwrite=True,
        )


def test_cli_records_and_passes_allow_unpromoted(tmp_path, monkeypatch) -> None:
    source = _touch(tmp_path / "source.png")
    checkpoint = _touch(tmp_path / "checkpoint.pt")
    output = tmp_path / "output.png"
    report = tmp_path / "report.json"
    args = argparse.Namespace(
        checkpoint=str(checkpoint),
        input=str(source),
        output=str(output),
        state="ema",
        device="cpu",
        batch_size=1,
        report=str(report),
        overwrite=False,
        allow_unpromoted=True,
    )
    captured = {}

    def fake_load(path, *, device, state, allow_unpromoted):
        captured["allow_unpromoted"] = allow_unpromoted
        return object(), "cpu", {"checkpoint": str(path)}

    def fake_restore(_model, _source, destination, _device, _batch_size, *, overwrite):
        assert overwrite is False
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"png")

    monkeypatch.setattr(apply_cli, "parse_args", lambda: args)
    monkeypatch.setattr(apply_cli, "load_restorer", fake_load)
    monkeypatch.setattr(apply_cli, "restore_png", fake_restore)

    apply_cli.main()

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert captured["allow_unpromoted"] is True
    assert payload["allow_unpromoted"] is True
