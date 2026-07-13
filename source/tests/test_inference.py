from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

import puzzle_denoise_v2.inference as inference
from puzzle_denoise_v2.inference import (
    load_restorer,
    restore_png,
    restore_shuffled_image,
    restore_tiles_uint8,
)


class IdentityModel(torch.nn.Module):
    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        return tiles


def test_identity_inference_preserves_pixels_and_every_tile_slot() -> None:
    image = np.random.default_rng(37).integers(0, 256, size=(480, 480, 3), dtype=np.uint8)
    restored = restore_shuffled_image(IdentityModel(), image, torch.device("cpu"), batch_size=37)
    assert np.array_equal(restored, image)


def test_tile_inference_validates_dtype_shape_and_batch_size() -> None:
    tiles = np.zeros((2, 20, 20, 3), dtype=np.uint8)
    assert restore_tiles_uint8(IdentityModel(), tiles, torch.device("cpu"), 1).shape == tiles.shape
    with pytest.raises(TypeError, match="uint8"):
        restore_tiles_uint8(IdentityModel(), tiles.astype(np.float32), torch.device("cpu"))
    with pytest.raises(ValueError, match="batch_size"):
        restore_tiles_uint8(IdentityModel(), tiles, torch.device("cpu"), 0)


def test_restore_png_is_fail_closed_and_allows_explicit_overwrite(tmp_path) -> None:
    image = np.random.default_rng(38).integers(0, 256, size=(480, 480, 3), dtype=np.uint8)
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    Image.fromarray(image).save(source)
    Image.fromarray(np.zeros_like(image)).save(output)

    with pytest.raises(FileExistsError, match="already exists"):
        restore_png(IdentityModel(), source, output, torch.device("cpu"))
    restore_png(IdentityModel(), source, output, torch.device("cpu"), overwrite=True)
    with Image.open(output) as restored:
        assert np.array_equal(np.asarray(restored), image)

    with pytest.raises(ValueError, match="must be different"):
        restore_png(
            IdentityModel(),
            source,
            source,
            torch.device("cpu"),
            overwrite=True,
        )

    hard_link = tmp_path / "source-hard-link.png"
    hard_link.hardlink_to(source)
    with pytest.raises(ValueError, match="same file"):
        restore_png(
            IdentityModel(),
            source,
            hard_link,
            torch.device("cpu"),
            overwrite=True,
        )


def test_load_restorer_reports_safe_checkpoint_provenance(tmp_path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "tiny_latest.pt"
    checkpoint = {
        "schema_version": 1,
        "kind": "conservative_real_pair_fine_tune",
        "model_name": "tile-naf",
        "model_state": {},
        "ema_state": {},
        "step": 500,
        "best_step": 500,
        "manifest_sha256": "manifest",
        "source_code_sha256": "source",
        "fine_tune_code_sha256": "fine-tune",
        "init_checkpoint_sha256": "initial",
        "legacy_checkpoint_sha256": "legacy",
        "validation_quarantine_sha256": "quarantine",
        "maps_1024_sha256": "maps",
        "train_pairs_sha256": "train-pairs",
        "val_pairs_sha256": "val-pairs",
        "best_real_ssim": 0.75,
        "gate_validation": {
            "panel": "frozen_gate",
            "selected_step": 500,
            "assessment": {"eligible": True},
        },
        "promotion_status": "promoted",
        "safe_for_inference": True,
        "runtime_versions": {"torch": "test"},
        "best_validation": {"step": 500, "metric": 0.75},
        "unsafe_tensor": torch.ones(1),
    }
    torch.save(checkpoint, checkpoint_path)
    monkeypatch.setattr(inference, "build_model", lambda _name: IdentityModel())

    with pytest.raises(ValueError, match=r"\*_latest\.pt"):
        load_restorer(checkpoint_path, device="cpu")
    _, device, metadata = load_restorer(
        checkpoint_path,
        device="cpu",
        allow_unpromoted=True,
    )

    assert device == torch.device("cpu")
    assert metadata["checkpoint_resolved"] == str(checkpoint_path.resolve())
    assert metadata["checkpoint_is_latest"] is True
    assert metadata["allow_unpromoted"] is True
    assert metadata["promotion_issues"] == ["checkpoint filename ends with *_latest.pt"]
    assert metadata["kind"] == checkpoint["kind"]
    assert metadata["manifest_sha256"] == "manifest"
    assert metadata["legacy_checkpoint_sha256"] == "legacy"
    assert metadata["validation_quarantine_sha256"] == "quarantine"
    assert metadata["maps_1024_sha256"] == "maps"
    assert metadata["best_real_ssim"] == 0.75
    assert metadata["promotion_status"] == "promoted"
    assert metadata["safe_for_inference"] is True
    assert metadata["gate_validation"]["panel"] == "frozen_gate"
    assert metadata["runtime_versions"] == {"torch": "test"}
    assert metadata["best_validation"] == {"step": 500, "metric": 0.75}
    assert "unsafe_tensor" not in metadata


@pytest.mark.parametrize(
    (
        "step",
        "best_step",
        "best_validation",
        "gate_validation",
        "promotion_status",
        "safe_for_inference",
        "message",
    ),
    [
        (500, 400, {"step": 400}, {"panel": "frozen_gate"}, "promoted", True, "does not match best_step"),
        (500, 500, None, {"panel": "frozen_gate"}, "promoted", True, "no best_validation"),
        (500, 500, {"step": 500}, None, "promoted", True, "no frozen gate_validation"),
        (500, 500, {"step": 500}, {"panel": "frozen_gate"}, "calibration_candidate", True, "promotion_status"),
        (500, 500, {"step": 500}, {"panel": "frozen_gate"}, "promoted", False, "safe_for_inference"),
        (
            500,
            500,
            {"step": 500},
            {"panel": "calibration", "selected_step": 500, "assessment": {"eligible": True}},
            "promoted",
            True,
            "panel",
        ),
        (
            500,
            500,
            {"step": 500},
            {"panel": "frozen_gate", "selected_step": 400, "assessment": {"eligible": True}},
            "promoted",
            True,
            "selected_step",
        ),
        (
            500,
            500,
            {"step": 500},
            {"panel": "frozen_gate", "selected_step": 500, "assessment": {"eligible": False}},
            "promoted",
            True,
            "eligible",
        ),
    ],
)
def test_load_restorer_rejects_unpromoted_real_fine_tune(
    tmp_path,
    monkeypatch,
    step,
    best_step,
    best_validation,
    gate_validation,
    promotion_status,
    safe_for_inference,
    message,
) -> None:
    checkpoint_path = tmp_path / f"unpromoted-{step}-{best_step}.pt"
    checkpoint = {
        "kind": "conservative_real_pair_fine_tune",
        "model_name": "tile-naf",
        "model_state": {},
        "ema_state": {},
        "step": step,
        "best_step": best_step,
        "promotion_status": promotion_status,
        "safe_for_inference": safe_for_inference,
    }
    if best_validation is not None:
        checkpoint["best_validation"] = best_validation
    if gate_validation is not None:
        checkpoint["gate_validation"] = gate_validation
    torch.save(checkpoint, checkpoint_path)
    monkeypatch.setattr(inference, "build_model", lambda _name: IdentityModel())

    with pytest.raises(ValueError, match=message):
        load_restorer(checkpoint_path, device="cpu")
    _, _, metadata = load_restorer(
        checkpoint_path,
        device="cpu",
        allow_unpromoted=True,
    )
    assert metadata["allow_unpromoted"] is True
    assert metadata["promotion_issues"]


def test_load_restorer_allows_explicit_rollback_artifact(tmp_path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "rollback.pt"
    torch.save(
        {
            "kind": "conservative_real_pair_fine_tune_rollback",
            "model_name": "tile-naf",
            "model_state": {},
            "ema_state": {},
            "rolled_back": True,
            "promotion_status": "rollback_safe",
            "safe_for_inference": True,
            "step": 0,
        },
        checkpoint_path,
    )
    monkeypatch.setattr(inference, "build_model", lambda _name: IdentityModel())

    _, _, metadata = load_restorer(checkpoint_path, device="cpu")

    assert metadata["rolled_back"] is True
    assert metadata["allow_unpromoted"] is False
    assert metadata["promotion_issues"] == []


def test_load_restorer_rejects_unsafe_rollback_artifact(tmp_path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "unsafe-rollback.pt"
    torch.save(
        {
            "kind": "conservative_real_pair_fine_tune_rollback",
            "model_name": "tile-naf",
            "model_state": {},
            "ema_state": {},
            "rolled_back": True,
            "promotion_status": "diagnostic_placeholder",
            "safe_for_inference": False,
            "step": 500,
        },
        checkpoint_path,
    )
    monkeypatch.setattr(inference, "build_model", lambda _name: IdentityModel())

    with pytest.raises(ValueError, match="rollback checkpoint"):
        load_restorer(checkpoint_path, device="cpu")
