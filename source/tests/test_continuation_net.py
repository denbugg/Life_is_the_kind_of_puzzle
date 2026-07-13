from __future__ import annotations

from pathlib import Path

import pytest
import torch

from puzzle_assembly.continuation_net import (
    CHECKPOINT_KIND,
    SCHEMA_VERSION,
    ContinuationNet0,
    load_continuation_net0_checkpoint,
    load_continuation_net0_checkpoint_payload,
    save_continuation_net0_checkpoint,
)


def test_forward_shapes_ranges_and_finite_values() -> None:
    torch.manual_seed(3)
    model = ContinuationNet0().eval()
    values = torch.rand(2, 6, 20, 20)
    with torch.inference_mode():
        outputs = model(values)

    assert set(outputs) == {"continuation", "reconstruction"}
    assert outputs["continuation"].shape == (2, 3, 20, 4)
    assert outputs["reconstruction"].shape == (2, 3, 20, 20)
    for output in outputs.values():
        assert torch.isfinite(output).all()
        assert bool((output >= 0.0).all())
        assert bool((output <= 1.0).all())


def test_both_heads_and_input_receive_gradients() -> None:
    torch.manual_seed(5)
    model = ContinuationNet0().train()
    values = torch.rand(1, 6, 20, 20, requires_grad=True)
    outputs = model(values)
    loss = outputs["continuation"].mean() + outputs["reconstruction"].mean()
    loss.backward()

    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert torch.count_nonzero(values.grad) > 0
    for parameter in (
        model.continuation_projection.weight,
        model.reconstruction_projection.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_checkpoint_roundtrip_preserves_config_outputs_and_metadata(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    model = ContinuationNet0().eval()
    values = torch.rand(1, 6, 20, 20)
    path = tmp_path / "continuation_net0.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    save_continuation_net0_checkpoint(
        path,
        model,
        metadata={"experiment": "unit"},
        optimizer_state=optimizer.state_dict(),
        training_state={"epoch": 2},
    )

    payload = load_continuation_net0_checkpoint_payload(path)
    loaded, metadata = load_continuation_net0_checkpoint(path)
    loaded.eval()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["kind"] == CHECKPOINT_KIND
    assert payload["model_config"] == model.config()
    assert payload["training_state"] == {"epoch": 2}
    assert metadata == {"experiment": "unit", "safe_for_submission": False}
    with torch.inference_mode():
        expected = model(values)
        actual = loaded(values)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0.0, rtol=0.0)


def test_checkpoint_schema_and_inputs_fail_closed(tmp_path: Path) -> None:
    model = ContinuationNet0()
    with pytest.raises(ValueError, match="shape"):
        model(torch.rand(1, 6, 19, 20))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.zeros(1, 6, 20, 20, dtype=torch.uint8))
    invalid = torch.rand(1, 6, 20, 20)
    invalid[..., 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(invalid)

    path = tmp_path / "valid.pt"
    save_continuation_net0_checkpoint(path, model)
    payload = load_continuation_net0_checkpoint_payload(path)
    payload["model_config"] = dict(payload["model_config"])
    payload["model_config"]["width"] = 47
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(RuntimeError, match="config mismatch"):
        load_continuation_net0_checkpoint_payload(tampered)

