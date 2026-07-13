from pathlib import Path

import torch

from puzzle_assembly.binary_edge_verifier import (
    BinaryEdgeVerifierNet,
    load_binary_edge_verifier,
    save_binary_edge_verifier,
)


def test_binary_edge_verifier_forward_and_checkpoint(tmp_path: Path) -> None:
    model = BinaryEdgeVerifierNet(
        tabular_dim=25, channels=16, side_band=6, dropout=0.0
    )
    raw = torch.rand(5, 3, 20, 12)
    denoised = torch.rand(5, 3, 20, 12)
    tabular = torch.rand(5, 25)
    logits = model(raw, denoised, tabular)
    assert logits.shape == (5,)
    assert torch.isfinite(logits).all()

    checkpoint = tmp_path / "model.pt"
    save_binary_edge_verifier(
        checkpoint,
        model,
        feature_names=[f"f{index}" for index in range(25)],
        metadata={"seed": 7},
    )
    loaded, names, metadata = load_binary_edge_verifier(checkpoint)
    assert names == [f"f{index}" for index in range(25)]
    assert metadata == {"seed": 7}
    loaded.eval()
    model.eval()
    with torch.inference_mode():
        assert torch.equal(model(raw, denoised, tabular), loaded(raw, denoised, tabular))


def test_binary_edge_verifier_rejects_bad_shapes() -> None:
    model = BinaryEdgeVerifierNet(
        tabular_dim=4, channels=16, side_band=6, dropout=0.0
    )
    raw = torch.rand(2, 3, 20, 12)
    denoised = torch.rand(2, 3, 20, 12)
    try:
        model(raw[:, :, :, :-1], denoised[:, :, :, :-1], torch.rand(2, 4))
    except ValueError as error:
        assert "raw_patches" in str(error)
    else:
        raise AssertionError("bad patch shape was accepted")
