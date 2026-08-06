from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import build_source_groups as legacy
import build_source_groups_v3 as v3


def _names() -> list[str]:
    return [f"img_{index:06d}.png" for index in range(v3.TRAIN_COUNT + v3.VALIDATION_COUNT)]


def _singleton_mapping(names: list[str]) -> dict[str, str]:
    return {name: f"group_{index:06d}" for index, name in enumerate(names)}


def _manifest(names: list[str], mapping: dict[str, str]) -> dict[str, object]:
    selection = v3.select_confirmation_v3(names, mapping)
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(mapping[name], []).append(name)
    return {
        "schema_version": 2,
        "algorithms": copy.deepcopy(v3.ALGORITHMS_CONTRACT),
        "builder_contract": copy.deepcopy(v3.BUILDER_CONTRACT),
        "stats": {"files": len(names)},
        "groups": groups,
        "files": {
            name: {"sha256": "0" * 64, "source_group": mapping[name]}
            for name in names
        },
        "split": {
            "train_count": v3.TRAIN_COUNT,
            "val_count": v3.VALIDATION_COUNT,
            "known_tune_val_ids": [0, v3.KNOWN_TUNE_VAL_MAX],
            "candidate_val_min": v3.CANDIDATE_VAL_MIN,
            "selection_seed": v3.SELECTION_SEED,
            "excluded_val_ids": list(v3.PRIOR_GATE_VALIDATION_IDS),
            "selection_contract": copy.deepcopy(v3.BASE_SELECTION_CONTRACT),
            "v3_selection_contract": copy.deepcopy(v3.V3_SELECTION_CONTRACT),
            "prior_scene_names": list(selection.prior_scene_names),
            "prior_source_groups_v3": list(selection.prior_source_groups_v3),
            "eligible_confirmation": list(selection.eligible),
            "selected_confirmation": list(selection.selected),
        },
    }


class FiveChunkIndexTests(unittest.TestCase):
    def test_adversarial_four_bit_pair_missed_by_old_chunks_is_retrieved(self) -> None:
        left = 0
        right = sum(1 << bit for bit in (0, 16, 32, 48))
        self.assertEqual((left ^ right).bit_count(), 4)
        self.assertNotIn((0, 1), set(legacy._candidate_pairs([left, right])))
        self.assertIn((0, 1), set(v3.candidate_pairs_five_chunks([left, right])))

    def test_all_sampled_pairs_with_hamming_at_most_four_are_retrieved(self) -> None:
        values = [
            0,
            1,
            (1 << 13) | (1 << 27) | (1 << 41) | (1 << 63),
            (1 << 64) - 1,
            ((1 << 64) - 1) ^ (1 << 5) ^ (1 << 19) ^ (1 << 38) ^ (1 << 60),
        ]
        retrieved = set(v3.candidate_pairs_five_chunks(values))
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                if (values[left] ^ values[right]).bit_count() <= 4:
                    self.assertIn((left, right), retrieved)

    def test_legacy_build_logic_is_reused_and_candidate_hook_is_restored(self) -> None:
        left_hash = 0
        right_hash = sum(1 << bit for bit in (0, 16, 32, 48))
        items = [
            legacy.Fingerprint("a.png", "a" * 64, left_hash, 0, (10.0, 20.0, 30.0)),
            legacy.Fingerprint("b.png", "b" * 64, right_hash, 0, (10.0, 20.0, 30.0)),
        ]
        original = legacy._candidate_pairs
        mapping, groups, stats = v3.build_groups_v3(items)
        self.assertIs(legacy._candidate_pairs, original)
        self.assertEqual(mapping["a.png"], mapping["b.png"])
        self.assertEqual(list(groups.values()), [["a.png", "b.png"]])
        self.assertEqual(stats["perceptual_unions"], 1)

        with mock.patch.object(legacy, "build_groups", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                v3.build_groups_v3(items)
        self.assertIs(legacy._candidate_pairs, original)


class FixedManifestContractTests(unittest.TestCase):
    def test_algorithms_and_all_controls_are_exact_literals(self) -> None:
        self.assertEqual(v3.PHASH_CHUNK_SIZES, (13, 13, 13, 13, 12))
        self.assertEqual(v3.PHASH_CHUNK_OFFSETS, (0, 13, 26, 39, 52))
        self.assertEqual(v3.PHASH_THRESHOLD, 4)
        self.assertEqual(v3.DHASH_THRESHOLD, 6)
        self.assertEqual(v3.MEAN_RGB_THRESHOLD, 36.0)
        self.assertEqual(
            v3.ALGORITHMS_CONTRACT["candidate_index"]["chunk_sizes"],
            [13, 13, 13, 13, 12],
        )
        self.assertEqual(v3.TRAIN_COUNT, 6700)
        self.assertEqual(v3.VALIDATION_COUNT, 300)
        self.assertEqual(v3.CANDIDATE_VAL_MIN, 100)
        self.assertEqual(v3.SELECTION_SEED, "20260808")
        self.assertEqual(v3.SELECTION_COUNT, 48)
        self.assertEqual(len(v3.PRIOR_GATE_VALIDATION_IDS), 48)

    def test_current_v3_group_of_prior_name_is_forbidden_even_if_id_changed(self) -> None:
        names = _names()
        mapping = _singleton_mapping(names)
        validation = names[v3.TRAIN_COUNT :]
        prior_name = validation[v3.PRIOR_GATE_VALIDATION_IDS[0]]
        candidate_name = validation[101]
        self.assertNotIn(101, v3.PRIOR_GATE_VALIDATION_IDS)
        mapping[candidate_name] = mapping[prior_name]

        selection = v3.select_confirmation_v3(names, mapping)
        self.assertNotIn(candidate_name, selection.eligible)
        self.assertIn(mapping[prior_name], selection.prior_source_groups_v3)
        self.assertEqual(len(selection.selected), 48)

    def test_exact_manifest_validates_and_algorithm_drift_fails_closed(self) -> None:
        names = _names()
        mapping = _singleton_mapping(names)
        payload = _manifest(names, mapping)
        selection = v3.validate_manifest_v3(payload, names, mapping)
        self.assertEqual(len(selection.selected), 48)

        changed = copy.deepcopy(payload)
        changed["algorithms"]["phash_threshold"] = 5  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "algorithms"):
            v3.validate_manifest_v3(changed, names, mapping)

    def test_cli_has_only_target_path_and_output_is_canonical_create_once(self) -> None:
        parser = v3._build_parser()
        options = {option for action in parser._actions for option in action.option_strings}
        self.assertEqual(options - {"-h", "--help"}, {"--targets-dir"})
        self.assertEqual(
            str(v3.CANONICAL_OUTPUT).replace("\\", "/"),
            "E:/pazzle_work/rank96_e11_v4/source_groups_v4.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source_groups_v4.json"
            with mock.patch.object(v3, "CANONICAL_OUTPUT", output):
                first_digest, first_status = v3.write_canonical_manifest({"fixed": True})
                second_digest, second_status = v3.write_canonical_manifest({"fixed": True})
                self.assertEqual((first_status, second_status), ("created", "already_identical"))
                self.assertEqual(first_digest, second_digest)
                with self.assertRaises(FileExistsError):
                    v3.write_canonical_manifest({"fixed": False})


if __name__ == "__main__":
    unittest.main()
