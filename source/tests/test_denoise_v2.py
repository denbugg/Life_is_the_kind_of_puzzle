from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
from PIL import Image
import pytest
import torch
from skimage.metrics import structural_similarity

from puzzle_denoise_v2.degradation import SyntheticTileDegrader, pillow_libjpeg_degrade
from puzzle_denoise_v2.losses import RestorationLoss, skimage_like_ssim
from puzzle_denoise_v2.metrics import ordered_image_ssim, tile_metrics
from puzzle_denoise_v2.model import FullResolutionTileNAF, TileNAFNet
from puzzle_denoise_v2.tiles import (
    merge_tiles_numpy,
    merge_tiles_torch,
    split_tiles_numpy,
    split_tiles_torch,
)
from puzzle_denoise_v2.training import (
    TrainConfig,
    make_fixed_validation,
    make_fixed_validation_plan,
    render_fixed_validation,
    validate_resume_compatibility,
    validate_train_config,
)


def test_tile_roundtrip_numpy_and_torch() -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(480, 480, 3), dtype=np.uint8)
    assert np.array_equal(merge_tiles_numpy(split_tiles_numpy(image)), image)

    tensor = torch.from_numpy(image).permute(2, 0, 1)[None]
    restored = merge_tiles_torch(split_tiles_torch(tensor))
    assert torch.equal(restored, tensor)


def test_degrader_is_seeded_and_nontrivial() -> None:
    clean = torch.linspace(0.0, 1.0, 3 * 20 * 20).reshape(1, 3, 20, 20).repeat(4, 1, 1, 1)
    degrader = SyntheticTileDegrader()
    first, first_params = degrader(clean, generator=torch.Generator().manual_seed(123))
    second, second_params = degrader(clean, generator=torch.Generator().manual_seed(123))
    assert torch.equal(first, second)
    assert torch.equal(first_params.normalized(), second_params.normalized())
    assert first.shape == clean.shape
    assert float((first - clean).abs().mean()) > 0.02
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0


def test_degrader_is_chunk_invariant_with_fixed_parameters_and_noise() -> None:
    clean = torch.rand(7, 3, 20, 20, generator=torch.Generator().manual_seed(17))
    degrader = SyntheticTileDegrader()
    params = degrader.sample_parameters(7, torch.device("cpu"), torch.Generator().manual_seed(18))
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(19))
    complete, _ = degrader(clean, params=params, noise=noise)
    pieces = []
    for start, stop in ((0, 2), (2, 6), (6, 7)):
        piece, _ = degrader(clean[start:stop], params=params.index(slice(start, stop)), noise=noise[start:stop])
        pieces.append(piece)
    assert torch.equal(complete, torch.cat(pieces))


def test_pillow_libjpeg_branch_is_deterministic_and_chunk_invariant() -> None:
    rng = np.random.default_rng(23)
    clean = rng.integers(0, 256, size=(7, 20, 20, 3), dtype=np.uint8)
    noise = rng.standard_normal(clean.shape).astype(np.float32)
    degrader = SyntheticTileDegrader()
    params = degrader.sample_parameters(7, torch.device("cpu"), torch.Generator().manual_seed(24))

    complete = pillow_libjpeg_degrade(clean, params, noise)
    pieces = []
    for start, stop in ((0, 2), (2, 6), (6, 7)):
        pieces.append(
            pillow_libjpeg_degrade(clean[start:stop], params.index(slice(start, stop)), noise[start:stop])
        )
    assert complete.dtype == np.uint8
    assert np.array_equal(complete, np.concatenate(pieces))
    assert float(np.abs(complete.astype(np.float32) - clean).mean()) > 1.0


def test_pillow_libjpeg_branch_honors_operation_order_variant() -> None:
    rng = np.random.default_rng(27)
    clean = rng.integers(0, 256, size=(4, 20, 20, 3), dtype=np.uint8)
    noise = rng.standard_normal(clean.shape).astype(np.float32)
    degrader = SyntheticTileDegrader()
    params = degrader.sample_parameters(4, torch.device("cpu"), torch.Generator().manual_seed(28))
    primary = replace(params, variant=torch.zeros(4, dtype=torch.long))
    blur_first = replace(params, variant=torch.ones(4, dtype=torch.long))
    equivalent_brightness_first = replace(params, variant=torch.full((4,), 2, dtype=torch.long))

    primary_output = pillow_libjpeg_degrade(clean, primary, noise)
    blur_first_output = pillow_libjpeg_degrade(clean, blur_first, noise)
    assert not np.array_equal(primary_output, blur_first_output)
    assert np.array_equal(
        primary_output,
        pillow_libjpeg_degrade(clean, equivalent_brightness_first, noise),
    )

    invalid = replace(params, variant=torch.full((4,), 3, dtype=torch.long))
    with pytest.raises(ValueError, match="variant"):
        pillow_libjpeg_degrade(clean, invalid, noise)
    tensor = torch.from_numpy(clean.transpose(0, 3, 1, 2)).float().div_(255.0)
    with pytest.raises(ValueError, match="variant"):
        degrader(tensor, params=invalid)


def test_paired_codec_renderers_share_one_fixed_plan(tmp_path) -> None:
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    image = np.random.default_rng(29).integers(0, 256, size=(480, 480, 3), dtype=np.uint8)
    Image.fromarray(image).save(target_dir / "sample.png")
    degrader = SyntheticTileDegrader()
    plan = make_fixed_validation_plan(target_dir, ["sample.png"], 9, 30, degrader)

    kornia = render_fixed_validation(plan, degrader, batch_size=4, codec="kornia")
    pillow = render_fixed_validation(plan, degrader, batch_size=5, codec="pillow")
    assert kornia.shape == pillow.shape == plan.clean.shape
    assert not np.array_equal(kornia, pillow)
    assert np.array_equal(kornia, render_fixed_validation(plan, degrader, 9, "kornia"))
    assert np.array_equal(pillow, render_fixed_validation(plan, degrader, 9, "pillow"))


@pytest.mark.parametrize("codec", ["kornia", "pillow"])
def test_fixed_validation_is_batch_invariant(tmp_path, codec: str) -> None:
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    rng = np.random.default_rng(25)
    image = rng.integers(0, 256, size=(480, 480, 3), dtype=np.uint8)
    Image.fromarray(image).save(target_dir / "sample.png")
    degrader = SyntheticTileDegrader()

    first = make_fixed_validation(
        target_dir, ["sample.png"], 7, 26, degrader, batch_size=3, codec=codec
    )
    second = make_fixed_validation(
        target_dir, ["sample.png"], 7, 26, degrader, batch_size=7, codec=codec
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_models_start_as_identity_and_return_auxiliary_parameters() -> None:
    tiles = torch.rand(2, 3, 20, 20)
    for model in (FullResolutionTileNAF(width=16, blocks=2), TileNAFNet(width=16, middle_blocks=2)):
        prediction, auxiliary = model(tiles, return_aux=True)
        assert prediction.shape == tiles.shape
        assert auxiliary.shape == (2, 5)
        assert torch.allclose(prediction, tiles)


def test_soft_ssim_matches_skimage_for_float_tiles() -> None:
    generator = torch.Generator().manual_seed(99)
    target = torch.rand(3, 3, 20, 20, generator=generator)
    prediction = (target + torch.randn(target.shape, generator=generator) * 0.05).clamp(0.0, 1.0)
    actual = float(skimage_like_ssim(prediction, target))
    expected = np.mean(
        [
            structural_similarity(
                true.permute(1, 2, 0).numpy(),
                pred.permute(1, 2, 0).numpy(),
                channel_axis=2,
                data_range=1.0,
                win_size=7,
            )
            for pred, true in zip(prediction, target, strict=True)
        ]
    )
    assert abs(actual - expected) < 2e-5


def test_loss_and_uint8_metrics_have_identity_ceiling() -> None:
    target = torch.rand(2, 3, 20, 20)
    loss, components = RestorationLoss()(target, target)
    assert float(loss) < 0.002
    assert abs(float(components["ssim"]) - 1.0) < 1e-5

    uint8 = np.rint(target.numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    metrics = tile_metrics(uint8, uint8)
    assert metrics["tile_ssim"] == 1.0
    assert metrics["psnr"] == float("inf")
    assert metrics["mae"] == 0.0

    complete = np.tile(uint8[:1], (576, 1, 1, 1))
    assert ordered_image_ssim(complete, complete) == 1.0


def test_resume_rejects_trajectory_or_output_family_changes() -> None:
    config = TrainConfig(data_root="puzzle", manifest="split.json", output="first.pt", steps=10)
    checkpoint = {
        "schema_version": 2,
        "model_name": config.model,
        "manifest_sha256": "abc",
        "config": asdict(config),
        "step": 4,
        "model_state": {},
        "ema_state": {},
        "optimizer_state": {},
        "scheduler_state": {},
        "rng_state": {},
    }
    validate_resume_compatibility(checkpoint, config, "abc")

    with pytest.raises(ValueError, match="steps"):
        validate_resume_compatibility(checkpoint, replace(config, steps=11), "abc")
    with pytest.raises(ValueError, match="output"):
        validate_resume_compatibility(checkpoint, replace(config, output="second.pt"), "abc")
    with pytest.raises(ValueError, match="manifest hash"):
        validate_resume_compatibility(checkpoint, config, "different")


def test_invalid_training_and_variant_configuration_is_rejected() -> None:
    config = TrainConfig(data_root="puzzle", manifest="split.json", output="model.pt")
    with pytest.raises(ValueError, match="steps"):
        validate_train_config(replace(config, steps=0))
    with pytest.raises(ValueError, match="val_tiles_per_image"):
        validate_train_config(replace(config, val_tiles_per_image=577))
    with pytest.raises(ValueError, match="non-negative"):
        validate_train_config(replace(config, variant_weights=(2.0, -1.0, 0.0)))
