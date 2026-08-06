from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from build_source_groups import (
    Fingerprint,
    build_manifest,
    build_groups,
    fingerprint,
    parse_excluded_val_indices,
    select_confirmation,
)


class SourceGroupTests(unittest.TestCase):
    def test_exclusion_parser_accepts_ids_and_start_count_ranges(self) -> None:
        self.assertEqual(
            parse_excluded_val_indices("7, 10:3, 2", val_count=20),
            [2, 7, 10, 11, 12],
        )
        self.assertEqual(parse_excluded_val_indices("", val_count=20), [])

    def test_exclusion_parser_rejects_duplicates_malformed_and_bounds(self) -> None:
        invalid = ("2,2", "2:2,3", "1:0", "1:-2", "-1", "10", "1::2", "1,")
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_excluded_val_indices(spec, val_count=10)

    def test_manifest_preserves_v1_when_empty_and_records_v2_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for index in range(6):
                (root / f"img_{index:06d}.png").touch()

            def fake_fingerprint(path: Path) -> Fingerprint:
                index = int(path.stem.rsplit("_", 1)[1])
                return Fingerprint(
                    name=path.name,
                    sha256=f"{index:064x}",
                    phash=index,
                    dhash=index,
                    mean_rgb=(float(index),) * 3,
                )

            base_args = dict(
                targets=str(root),
                train_count=3,
                val_count=3,
                tune_val_max=0,
                candidate_val_min=1,
                select_count=1,
                seed="fixed",
                phash_threshold=-1,
                dhash_threshold=6,
                mean_rgb_threshold=36.0,
            )
            with patch("build_source_groups.fingerprint", side_effect=fake_fingerprint):
                legacy = build_manifest(Namespace(**base_args, exclude_val_indices=""))
                excluded = build_manifest(Namespace(**base_args, exclude_val_indices="1"))

            self.assertEqual(legacy["schema_version"], 1)
            self.assertNotIn("excluded_val_ids", legacy["split"])
            self.assertNotIn("selection_contract", legacy["split"])
            self.assertEqual(excluded["schema_version"], 2)
            self.assertEqual(excluded["split"]["excluded_val_ids"], [1])
            self.assertIn("selection_contract", excluded["split"])
            self.assertEqual(excluded["split"]["selected_confirmation"], ["img_000005.png"])

    def test_exact_and_brightness_near_duplicate_group(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = np.zeros((480, 480, 3), np.uint8)
            base[:, :240] = (20, 90, 180)
            base[:, 240:] = (220, 140, 30)
            same = base.copy()
            brighter = np.clip(base.astype(np.int16) + 12, 0, 255).astype(np.uint8)
            different = np.roll(base, 150, axis=1)
            for name, image in (("a.png", base), ("b.png", same), ("c.png", brighter), ("d.png", different)):
                Image.fromarray(image).save(root / name)
            items = [fingerprint(root / name) for name in ("a.png", "b.png", "c.png", "d.png")]
            mapping, groups, stats = build_groups(items)
            self.assertEqual(mapping["a.png"], mapping["b.png"])
            self.assertEqual(mapping["a.png"], mapping["c.png"])
            self.assertNotEqual(mapping["a.png"], mapping["d.png"])
            self.assertGreaterEqual(stats["non_singleton_groups"], 1)

    def test_selection_rejects_group_present_in_training(self) -> None:
        names = [f"img_{index:06d}.png" for index in range(8)]
        mapping = {name: f"g{index}" for index, name in enumerate(names)}
        mapping[names[6]] = mapping[names[0]]
        groups = {f"g{index}": [name] for index, name in enumerate(names)}
        groups[mapping[names[0]]] = [names[0], names[6]]
        groups.pop("g6", None)
        eligible, selected = select_confirmation(
            names,
            mapping,
            groups,
            train_count=4,
            val_count=4,
            tune_val_max=0,
            candidate_val_min=1,
            count=1,
            seed="fixed",
        )
        self.assertNotIn(names[6], eligible)
        self.assertEqual(len(selected), 1)

    def test_selection_excludes_validation_ids_before_ranking(self) -> None:
        names = [f"img_{index:06d}.png" for index in range(9)]
        mapping = {name: f"g{index}" for index, name in enumerate(names)}
        groups = {mapping[name]: [name] for name in names}
        baseline_eligible, _ = select_confirmation(
            names,
            mapping,
            groups,
            train_count=3,
            val_count=6,
            tune_val_max=0,
            candidate_val_min=1,
            count=2,
            seed="fixed",
        )
        eligible, selected = select_confirmation(
            names,
            mapping,
            groups,
            train_count=3,
            val_count=6,
            tune_val_max=0,
            candidate_val_min=1,
            count=2,
            seed="fixed",
            excluded_val_ids=[2, 4],
        )
        self.assertEqual(baseline_eligible, names[4:])
        self.assertEqual(eligible, [names[4], names[6], names[8]])
        self.assertNotIn(names[5], selected)
        self.assertNotIn(names[7], selected)

    def test_selection_rejects_invalid_explicit_exclusions(self) -> None:
        names = [f"img_{index:06d}.png" for index in range(6)]
        mapping = {name: f"g{index}" for index, name in enumerate(names)}
        groups = {mapping[name]: [name] for name in names}
        for exclusions in ([1, 1], [-1], [3], [True]):
            with self.subTest(exclusions=exclusions), self.assertRaises(ValueError):
                select_confirmation(
                    names,
                    mapping,
                    groups,
                    train_count=3,
                    val_count=3,
                    tune_val_max=0,
                    candidate_val_min=1,
                    count=1,
                    seed="fixed",
                    excluded_val_ids=exclusions,
                )


if __name__ == "__main__":
    unittest.main()
