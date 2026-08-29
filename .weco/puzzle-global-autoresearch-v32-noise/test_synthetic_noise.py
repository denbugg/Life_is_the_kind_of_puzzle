from __future__ import annotations

import numpy as np

import synthetic_noise as n


def synthetic_image() -> np.ndarray:
    y, x = np.mgrid[:n.IMAGE, :n.IMAGE]
    return np.stack(((x + y) % 256, (2 * x - y) % 256, (x // 2 + 3 * y) % 256), -1).astype(np.uint8)


def test_contract_and_determinism():
    first = n.make_sample(synthetic_image(), 17, "synthetic.png")
    second = n.make_sample(synthetic_image(), 17, "synthetic.png")
    assert np.array_equal(first["noisy_tiles"], second["noisy_tiles"])
    assert np.array_equal(first["permutation"], second["permutation"])
    assert first["filenames"] == second["filenames"]
    assert first["observed_tiles"].shape == (576, 20, 20, 3)
    assert np.array_equal(np.sort(first["permutation"]), np.arange(576))
    assert len(set(first["filenames"])) == 576


def test_manifest_ranges_and_mapping():
    sample = n.make_sample(synthetic_image(), 23, "synthetic.png")
    for observed, row in enumerate(sample["manifest"]["tiles"]):
        assert row["observed_index"] == observed
        assert row["canonical_index"] == int(sample["permutation"][observed])
        assert -.0001 + .70 <= row["contrast"] <= 1.30 + .0001
        assert -30.0001 <= row["brightness"] <= 30.0001
        assert 39.9999 <= row["noise_sigma"] <= 55.0001
        assert 35 <= row["jpeg_quality"] <= 50


def test_clean_round_trip():
    image = synthetic_image()
    assert np.array_equal(n.tiles_to_image(n.image_to_tiles(image)), image)
