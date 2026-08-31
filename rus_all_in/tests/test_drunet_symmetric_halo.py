from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

import aiijc_puzzle.drunet_symmetric_halo as halo
from aiijc_puzzle.drunet_goal_cycle2 import DIRECT_SIGMA, MODEL_BATCH_SIZE


class CaptureIdentityRgb(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.inputs.append(value.detach().cpu().clone())
        return value[:, :3]


def tile_roster() -> np.ndarray:
    base = np.arange(20 * 20 * 3, dtype=np.uint16).reshape(20, 20, 3)
    return np.stack([((base + index) % 256).astype(np.uint8) for index in range(576)])


def test_symmetric_halo_is_four_sided_32_and_center_crop_is_exact() -> None:
    tiles = tile_roster()
    model = CaptureIdentityRgb()
    restored, diagnostics = halo.render_drunet_tiles_symmetric_halo(
        model,
        tiles,
        sigma_255=50,
        device=torch.device("cpu"),
        batch_size=72,
    )

    np.testing.assert_array_equal(restored, tiles)
    assert len(model.inputs) == 8
    assert all(value.shape == (72, 4, 32, 32) for value in model.inputs)
    expected = functional.pad(
        torch.from_numpy(tiles[:72]).permute(0, 3, 1, 2).float() / 255.0,
        (6, 6, 6, 6),
        mode="reflect",
    )
    torch.testing.assert_close(model.inputs[0][:, :3], expected)
    torch.testing.assert_close(
        model.inputs[0][:, 3],
        torch.full((72, 32, 32), 50.0 / 255.0),
    )
    assert diagnostics.halo_top == diagnostics.halo_bottom == 6
    assert diagnostics.halo_left == diagnostics.halo_right == 6
    assert diagnostics.padded_tile_size == 32
    assert (diagnostics.crop_start, diagnostics.crop_stop) == (6, 26)
    assert diagnostics.batch_size == 72


@pytest.mark.parametrize(
    ("tiles", "sigma", "batch_size", "message"),
    [
        (np.zeros((575, 20, 20, 3), dtype=np.uint8), 50.0, 72, "576"),
        (np.zeros((576, 20, 20, 3), dtype=np.uint8), 50.1, 72, "sigma"),
        (np.zeros((576, 20, 20, 3), dtype=np.uint8), 50.0, 0, "batch_size"),
    ],
)
def test_symmetric_halo_rejects_geometry_sigma_and_batch_drift(
    tiles: np.ndarray,
    sigma: float,
    batch_size: int,
    message: str,
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        halo.render_drunet_tiles_symmetric_halo(
            torch.nn.Identity(),
            tiles,
            sigma_255=sigma,
            device=torch.device("cpu"),
            batch_size=batch_size,
        )


@dataclass
class DummyDiagnostics:
    label: str

    def as_dict(self) -> dict[str, str]:
        return {"label": self.label}


def test_b_only_ablation_keeps_tail_and_changes_only_tile_geometry(monkeypatch) -> None:
    tiles = np.full((576, 20, 20, 3), 40, dtype=np.uint8)
    calls: list[tuple[str, float, int]] = []
    tail_inputs: list[np.ndarray] = []

    def fake_baseline(model, value, *, sigma_255, device, batch_size):
        del model, device
        calls.append(("baseline", float(sigma_255), int(batch_size)))
        return np.full_like(value, 70), DummyDiagnostics("baseline24")

    def fake_halo(model, value, *, sigma_255, device, batch_size):
        del model, device
        calls.append(("halo", float(sigma_255), int(batch_size)))
        return np.full_like(value, 90), DummyDiagnostics("halo32")

    def fake_tail(value: np.ndarray):
        tail_inputs.append(value.copy())
        level = int(value[0, 0, 0, 0])
        return np.full((480, 480, 3), level, dtype=np.uint8), {"level": level}

    monkeypatch.setattr(halo, "render_drunet_tiles", fake_baseline)
    monkeypatch.setattr(halo, "render_drunet_tiles_symmetric_halo", fake_halo)
    monkeypatch.setattr(halo, "_protected_tail", fake_tail)

    predictions, diagnostics = halo.render_symmetric_halo_arms(
        torch.nn.Identity(),
        tiles,
        device=torch.device("cpu"),
    )

    assert tuple(predictions) == halo.ARM_NAMES
    assert calls == [
        ("baseline", DIRECT_SIGMA, MODEL_BATCH_SIZE),
        ("halo", DIRECT_SIGMA, halo.HALO_BATCH_SIZE),
    ]
    assert [int(value[0, 0, 0, 0]) for value in tail_inputs] == [70, 90]
    assert np.all(predictions[halo.BASELINE_B] == 70)
    assert np.all(predictions[halo.SYMMETRIC_HALO_B] == 90)
    assert diagnostics["baseline_neural"] == {"label": "baseline24"}
    assert diagnostics["symmetric_halo_neural"] == {"label": "halo32"}
    assert diagnostics["baseline_mask"] == {"level": 70}
    assert diagnostics["symmetric_halo_mask"] == {"level": 90}


def test_protected_tail_keeps_frozen_independent_h20_h28_h50_order(monkeypatch) -> None:
    tiles = np.full((576, 20, 20, 3), 80, dtype=np.uint8)
    calls: list[int] = []

    def fake_nlm(image: np.ndarray, strength: int) -> np.ndarray:
        calls.append(strength)
        return np.full_like(image, strength)

    def fake_blend(h20: np.ndarray, h28: np.ndarray, h50: np.ndarray):
        assert int(h20[0, 0, 0]) == 20
        assert int(h28[0, 0, 0]) == 28
        assert int(h50[0, 0, 0]) == 50
        return h28, {"frozen": True}

    monkeypatch.setattr(halo, "colored_nlm", fake_nlm)
    monkeypatch.setattr(halo, "blend_h28_safe_h50_flat_t60", fake_blend)
    output, diagnostics = halo._protected_tail(tiles)

    assert calls == [20, 28, 50]
    assert np.all(output == 28)
    assert diagnostics == {"frozen": True}


def test_frozen_geometry_constants_are_exact() -> None:
    assert halo.SYMMETRIC_HALO == 6
    assert halo.PADDED_TILE_SIZE == 32
    assert halo.HALO_BATCH_SIZE == 72
    assert halo.ARM_NAMES == (
        "B_drunet50_protected_h28_h50_t60",
        "E_drunet50_symmetric_halo32_protected_h28_h50_t60",
    )


def load_runner_module():
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "scripts/run_drunet_symmetric_halo_train512.py"
    name = "_test_drunet_symmetric_halo_train512_runner"
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_preregistration_cycle2_binding_and_train512_roster_are_fail_closed() -> None:
    runner = load_runner_module()
    config, records = runner.load_context()

    assert runner.CONFIG_SHA256 == (
        "662270187b1a93d85a7423ad7be52959a0df289bf2f9fafa277cfc693654dc09"
    )
    assert runner.CONFIG.stat().st_mode & 0o222 == 0
    assert Path(f"{runner.CONFIG}.sha256").stat().st_mode & 0o222 == 0
    assert config["train_panel"]["offset"] == 512
    assert config["train_panel"]["count"] == 16
    assert len(records) == 16
    assert records[0]["filename"] == "img_005961.png"
    assert records[-1]["filename"] == "img_001637.png"
    assert config["cycle2_binding"]["fixed_safe_candidate_for_this_ablation"] == halo.BASELINE_B
    assert config["target_access_contract"]["calibration_access"] is False
    assert config["target_access_contract"]["holdout_access"] is False
    assert config["target_access_contract"]["competition_test_access"] is False


def test_fail_closed_json_writer_rejects_nan_before_creating_artifact(tmp_path: Path) -> None:
    runner = load_runner_module()
    destination = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="Out of range float"):
        runner.write_json_exclusive_readonly(destination, {"invalid": float("nan")})
    assert not destination.exists()
