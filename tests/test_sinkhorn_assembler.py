import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sinkhorn_assembler import SinkhornAssembler, decode, log_sinkhorn


def test_sinkhorn_is_doubly_stochastic():
    z = log_sinkhorn(torch.randn(2, 16, 16), 20).exp()
    assert torch.allclose(z.sum(1), torch.ones(2, 16), atol=1e-4)
    assert torch.allclose(z.sum(2), torch.ones(2, 16), atol=1e-4)


def test_model_is_equivariant_to_tile_permutation():
    torch.manual_seed(1)
    m = SinkhornAssembler(d=32, rounds=2, blocks=1).eval()
    x = torch.rand(1, 16, 8, 8, 3) * 255
    p = torch.randperm(16)
    with torch.no_grad():
        a, _, _ = m(x, 4)
        b, _, _ = m(x[:, p], 4)
    assert torch.allclose(b, a[:, :, p], atol=2e-5, rtol=2e-5)


def test_decode_is_bijective():
    out = decode(torch.randn(3, 25, 25))
    for row in out:
        assert np.array_equal(np.sort(row), np.arange(25))
