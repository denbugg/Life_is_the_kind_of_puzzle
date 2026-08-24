from __future__ import annotations

import copy
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


E24_TEST_ROOT = Path("E:/pazzle_work/posegraph_e24_selector/test_tmp")
E24_TEST_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(E24_TEST_ROOT)
os.environ["TMP"] = str(E24_TEST_ROOT)
os.environ["TMPDIR"] = str(E24_TEST_ROOT)
if sys.pycache_prefix is None or Path(sys.pycache_prefix).drive.upper() != "E:":
    sys.pycache_prefix = str(
        Path("E:/pazzle_work/posegraph_e24_selector/test_pycache")
    )

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import e24_context_relation_selector as selector  # noqa: E402


def _feature_table(
    query_sizes: tuple[int, ...] = (3, 2),
    *,
    scene_query_counts: tuple[int, ...] | None = None,
) -> selector.RelationFeatureTable:
    if scene_query_counts is None:
        scene_query_counts = (len(query_sizes),)
    if sum(scene_query_counts) != len(query_sizes):
        raise AssertionError("scene/query accounting drifted in test fixture")
    rows = sum(query_sizes)
    features = np.zeros((rows, len(selector.FEATURE_NAMES)), dtype=np.float32)
    hypothesis_ids = np.full(rows, selector.NONE_HYPOTHESIS_ID, dtype=np.int64)
    relation_ids = np.full(rows, selector.NONE_RELATION_ID, dtype=np.int64)
    relations = np.zeros((rows, 4), dtype=np.int64)
    row_kind = np.full(rows, selector.ROW_NONE, dtype=np.uint8)
    support = np.zeros(rows, dtype=np.int64)
    query_offsets = [0]
    scene_offsets = [0]
    cursor = 0
    identity = 0
    scene_stop_queries = set(np.cumsum(scene_query_counts).tolist())
    for query, size in enumerate(query_sizes):
        u, v = 2 * query, 2 * query + 1
        for local in range(size - 1):
            hypothesis_ids[cursor] = identity
            relation_ids[cursor] = identity
            relations[cursor] = (u, v, local, 0)
            row_kind[cursor] = selector.ROW_OFFSET
            support[cursor] = 99 if local else 1
            identity += 1
            cursor += 1
        relations[cursor] = (u, v, 0, 0)
        features[cursor, selector.FEATURE_INDEX["is_none"]] = 1.0
        cursor += 1
        query_offsets.append(cursor)
        if query + 1 in scene_stop_queries:
            scene_offsets.append(cursor)
    return selector.RelationFeatureTable(
        features=np.ascontiguousarray(features),
        hypothesis_ids=np.ascontiguousarray(hypothesis_ids),
        relation_ids=np.ascontiguousarray(relation_ids),
        relations=np.ascontiguousarray(relations),
        row_kind=np.ascontiguousarray(row_kind),
        support=np.ascontiguousarray(support),
        query_offsets=np.asarray(query_offsets, dtype=np.int64),
        scene_offsets=np.asarray(scene_offsets, dtype=np.int64),
    )


def _two_category_labels(table: selector.RelationFeatureTable) -> np.ndarray:
    labels = np.zeros(table.rows, dtype=np.int8)
    for query_index, (start, stop) in enumerate(
        zip(table.query_offsets[:-1], table.query_offsets[1:])
    ):
        labels[int(start) if query_index % 2 == 0 else int(stop) - 1] = 1
    return labels


def _minimal_extractor_fixture() -> tuple[
    object, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    candidate_ids = np.full((576, 128), -1, dtype=np.int64)
    candidate_ids[:, 0] = np.arange(576, dtype=np.int64) ^ 1
    raw_logits = np.full((4, 576, 128), -np.inf, dtype=np.float32)
    raw_logits[:, :, 0] = np.arange(4, dtype=np.float32)[:, None]
    base_pairs = tuple(
        selector.e22.AffinityPair(pair_id, first, first + 1, 0, 0)
        for pair_id, first in enumerate(range(0, 576, 2))
    )
    components = tuple(
        selector.e23.RigidComponent(tile, ((tile, 0, 0),)) for tile in range(576)
    )
    owner = np.arange(576, dtype=np.int64)
    local_rows = np.zeros(576, dtype=np.int64)
    local_cols = np.zeros(576, dtype=np.int64)
    selected = np.empty((4, 576, 64), dtype=np.int64)
    for direction in range(4):
        for source in range(576):
            selected[direction, source] = (
                source + 2 + np.arange(64, dtype=np.int64)
            ) % 576
    forward = selector.e22.LogitObservation(
        0, 1, selector.e23.RIGHT, 0, float(raw_logits[selector.e23.RIGHT, 0, 0])
    )
    reverse = selector.e22.LogitObservation(
        1, 0, selector.e23.LEFT, 0, float(raw_logits[selector.e23.LEFT, 1, 0])
    )
    claim = selector.e23.RCCE4Claim(
        0, 0, 0, 1, 0, 1, 0, 1, forward, reverse
    )
    relation = selector.e23.RelationCandidate(0, 0, 1, 0, 1, (0,))
    hypothesis = selector.e23.PoseHypothesis(0, 0, 0, 1, 0, 1, (0,))
    result = selector.e23.CandidatePoolResult(
        components=components,
        owner=owner,
        local_rows=local_rows,
        local_cols=local_cols,
        nontrivial_component_ids=frozenset(),
        affinity_pairs=base_pairs,
        base_affinity_pairs=base_pairs,
        spatial_selected_ids=np.ascontiguousarray(selected),
        spatial_pairs=(),
        claims=(claim,),
        relation_candidates=(relation,),
        hypotheses=(hypothesis,),
        rejections=(),
        diagnostics=None,
    )
    spatial_logits = np.zeros((4, 576, 576), dtype=np.float32)
    tiles = np.zeros((576, 20, 20, 3), dtype=np.uint8)
    return result, candidate_ids, raw_logits, spatial_logits, tiles, selected


class FeatureContractTests(unittest.TestCase):
    def test_permutation_is_absent_from_extractor_signature_and_feature_columns(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(selector.extract_relation_features).parameters),
            (
                "result",
                "candidate_ids",
                "raw_logits",
                "spatial_logits",
                "tiles_uint8",
            ),
        )
        lowered = tuple(name.lower() for name in selector.FEATURE_NAMES)
        self.assertFalse(any("permutation" in name for name in lowered))
        self.assertFalse(any("tile_id" in name or "component_id" in name for name in lowered))

    def test_candidate_specific_margin_and_canonical_percentile_ties(self) -> None:
        robust, percentile, margin = selector._robust_statistics(
            np.asarray((3.0, 3.0, 1.0), dtype=np.float32),
            np.asarray((9, 2, 5), dtype=np.int64),
        )
        self.assertEqual(margin.tolist(), [0.0, 0.0, -2.0])
        self.assertEqual(percentile.tolist(), [0.5, 1.0, 0.0])
        self.assertTrue(np.isfinite(robust).all())
        self.assertEqual(robust.tolist(), [0.0, 0.0, -8.0])
        _z, _percentile, unique_margin = selector._robust_statistics(
            np.asarray((4.0, 3.0, 1.0), dtype=np.float32),
            np.asarray((0, 1, 2), dtype=np.int64),
        )
        self.assertEqual(unique_margin.tolist(), [1.0, -1.0, -3.0])
        for required in (
            "raw_valid_row_size_mean",
            "spatial_wrong_percentile_mean",
            "spatial_correct_minus_wrong_robust_z_mean",
            "incidental_evidence_missing",
            "incidental_spatial_e0_mean",
            "incidental_seam_w1_rgb_mse_mean",
            "incidental_seam_w1_lab_mse_mean",
            "incidental_seam_w1_ncc_mean",
        ):
            self.assertIn(required, selector.FEATURE_INDEX)

    def test_label_free_extractor_smoke_materializes_exact_offset_plus_none(self) -> None:
        result, candidate_ids, raw_logits, spatial_logits, tiles, selected = (
            _minimal_extractor_fixture()
        )
        with mock.patch.object(
            selector.e23,
            "_select_spatial_residuals",
            return_value=(selected, {}),
        ):
            table = selector.extract_relation_features(
                result, candidate_ids, raw_logits, spatial_logits, tiles
            )
        self.assertEqual((table.rows, table.queries), (2, 1))
        self.assertEqual(table.row_kind.tolist(), [selector.ROW_OFFSET, selector.ROW_NONE])
        self.assertTrue(np.isfinite(table.features).all())
        offset = table.features[0]
        none = table.features[1]
        self.assertEqual(offset[selector.FEATURE_INDEX["has_offset"]], 1.0)
        self.assertEqual(none[selector.FEATURE_INDEX["is_none"]], 1.0)
        self.assertAlmostEqual(
            float(offset[selector.FEATURE_INDEX["raw_valid_row_size_mean"]]),
            1.0 / 128.0,
        )
        self.assertGreater(
            float(offset[selector.FEATURE_INDEX["projected_contact_count_log1p"]]),
            0.0,
        )
        self.assertEqual(
            float(offset[selector.FEATURE_INDEX["incidental_evidence_missing"]]),
            1.0,
        )
        self.assertEqual(
            float(offset[selector.FEATURE_INDEX["twohop_path_count_log1p"]]),
            0.0,
        )
        self.assertEqual(
            float(offset[selector.FEATURE_INDEX["twohop_min_l1_scaled"]]),
            0.0,
        )

    def test_component_endpoint_features_are_symmetric_under_swap(self) -> None:
        first = selector._ComponentGeometry(1, 0, 0, 0, 0, 1, 1, 1, 1.0)
        second = selector._ComponentGeometry(7, -1, 2, 3, 7, 4, 5, 20, 0.35)
        left = np.zeros(len(selector.FEATURE_NAMES), dtype=np.float32)
        right = np.zeros_like(left)
        selector._base_component_features(left, first, second)
        selector._base_component_features(right, second, first)
        np.testing.assert_array_equal(left, right)

    def test_physical_seam_is_exactly_pixel_19_against_pixel_0(self) -> None:
        rgb = np.zeros((2, 20, 20, 3), dtype=np.float32)
        lab = np.zeros_like(rgb)
        rgb[0, :, 19, :] = 4.0
        rgb[1, :, 0, :] = 4.0
        rgb[0, :, 18, :] = 1.0
        rgb[1, :, 1, :] = 2.0
        lab[0, :, 19, :] = 7.0
        lab[1, :, 0, :] = 7.0
        claim = SimpleNamespace(first=0, second=1, dx=1, dy=0)
        rgb_mse, _normal, ncc, lab_mse = selector._seam_values(rgb, lab, claim, 1)
        self.assertEqual(rgb_mse, 0.0)
        self.assertEqual(lab_mse, 0.0)
        self.assertEqual(ncc, 0.0)
        full = selector._seam_values_full(rgb, lab, claim, 1)
        self.assertEqual(len(full), 5)
        self.assertTrue(np.isfinite(full).all())

    def test_incidental_contact_evidence_is_present_and_cached_by_physical_seam(self) -> None:
        components = (
            selector.e23.RigidComponent(0, ((0, 0, 0), (2, 1, 0))),
            selector.e23.RigidComponent(1, ((1, 0, 0), (3, 1, 0))),
        )
        claim = selector.e23.RCCE4Claim(0, 0, 0, 1, 0, 1, 0, 1, None, None)
        hypothesis = selector.e23.PoseHypothesis(0, 0, 0, 1, 0, 1, (0,))
        result = SimpleNamespace(components=components, claims=(claim,))
        percentile = np.zeros((4, 4, 4), dtype=np.float32)
        percentile[selector.e23.RIGHT, 2, 3] = 0.8
        percentile[selector.e23.LEFT, 3, 2] = 0.6
        stats = selector._DirectionalStatistics(
            np.zeros_like(percentile), percentile, np.zeros_like(percentile)
        )
        rgb = np.zeros((4, 20, 20, 3), dtype=np.float32)
        lab = np.zeros_like(rgb)
        cache: dict[tuple[int, int, int, int], tuple[float, float, float, float]] = {}
        row = np.zeros(len(selector.FEATURE_NAMES), dtype=np.float32)
        selector._projected_contact_features(
            row, result, hypothesis, rgb, lab, stats, cache
        )
        self.assertEqual(row[selector.FEATURE_INDEX["incidental_evidence_missing"]], 0.0)
        self.assertAlmostEqual(
            float(row[selector.FEATURE_INDEX["incidental_spatial_e0_mean"]]),
            0.7,
            places=6,
        )
        self.assertEqual(tuple(cache), ((2, 3, 0, 1),))
        second = np.zeros_like(row)
        selector._projected_contact_features(
            second, result, hypothesis, rgb, lab, stats, cache
        )
        self.assertEqual(cache, cache.copy())
        np.testing.assert_array_equal(row, second)

    def test_context_shortlists_are_exactly_top4_and_top32_with_canonical_ties(self) -> None:
        relations: list[tuple[int, int, int, int]] = []
        values: list[float] = []
        groups: list[tuple[int, int]] = []
        for neighbour in range(1, 41):
            start = len(relations)
            for offset in range(6):
                relations.append((0, neighbour, offset, 0))
                values.append(1.0 if offset == 0 else 1.0 - offset / 10.0)
            groups.append((start, len(relations)))
        relation_array = np.asarray(relations, dtype=np.int64)
        e0 = np.asarray(values, dtype=np.float32)
        _best, top, incident, counts = selector._bounded_context_shortlists(
            relation_array, e0, groups, 41
        )
        self.assertEqual({len(indices) for indices in top.values()}, {4})
        self.assertEqual(counts[0], 40)
        self.assertEqual(len(incident[0]), 32)
        self.assertEqual(tuple(incident[0]), tuple(range(1, 33)))
        self.assertEqual(relation_array.shape[0], 40 * 6)
        self.assertFalse(selector.PROTOCOL["two_hop_context"]["scored_rows_truncated"])


class LearnerAndSelectionTests(unittest.TestCase):
    def test_weights_are_category_balanced_inside_each_scene(self) -> None:
        table = _feature_table(
            (2, 3, 4, 2), scene_query_counts=(2, 2)
        )
        labels = _two_category_labels(table)
        weights = selector.balanced_query_row_weights(table, labels)
        for scene_start, scene_stop in zip(
            table.scene_offsets[:-1], table.scene_offsets[1:]
        ):
            offset_mass = 0.0
            none_mass = 0.0
            for start, stop in zip(table.query_offsets[:-1], table.query_offsets[1:]):
                if int(start) < int(scene_start) or int(stop) > int(scene_stop):
                    continue
                mass = float(weights[int(start) : int(stop)].sum())
                if labels[int(stop) - 1] == 1:
                    none_mass += mass
                else:
                    offset_mass += mass
            self.assertAlmostEqual(offset_mass, none_mass, places=6)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_canonical_pair_ties_global_order_strict_margin_and_cap(self) -> None:
        table = _feature_table((3, 2, 2))
        scores = np.asarray((1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        selected, attempted, cap = selector.select_pair_winners(
            table, scores, component_count=2
        )
        self.assertEqual(selected[0].relation, (0, 1, 0, 0))
        self.assertEqual(selected[1].relation, (2, 3, 0, 0))
        self.assertEqual(tuple(item.relation for item in attempted), tuple(item.relation for item in selected))
        self.assertEqual(cap, 2)
        self.assertEqual(len(selected), 2)  # third query has exactly zero margin
        _selected, _attempted, sparse_cap = selector.select_pair_winners(
            table, scores, component_count=10
        )
        self.assertEqual(sparse_cap, 2)  # min(selected count, 2 * (C - 1))

    def test_exact_lightgbm_config_and_all_fold_seeds(self) -> None:
        expected = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [1],
            "label_gain": [0, 1],
            "n_estimators": 256,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 200,
            "max_bin": 255,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "lambda_l2": 1.0,
            "lambda_l1": 0.0,
            "lambdarank_truncation_level": 30,
            "lambdarank_norm": True,
            "n_jobs": 8,
            "deterministic": True,
            "force_col_wise": True,
        }
        for key, value in expected.items():
            self.assertEqual(selector.LIGHTGBM_CONFIG[key], value)
        table = _feature_table((2, 2))
        labels = _two_category_labels(table)
        for fold in range(4):
            captured: dict[str, object] = {}

            class FakeRanker:
                def __init__(self, **config: object) -> None:
                    captured["config"] = config

                def fit(self, *args: object, **kwargs: object) -> "FakeRanker":
                    captured["fit_kwargs"] = kwargs
                    return self

            with mock.patch.dict(
                sys.modules, {"lightgbm": SimpleNamespace(LGBMRanker=FakeRanker)}
            ):
                selector.fit_lambdarank(table, labels, fold=fold)
            config = captured["config"]
            for seed_name in (
                "random_state",
                "data_random_seed",
                "feature_fraction_seed",
            ):
                self.assertEqual(config[seed_name], 1234 + fold)
            self.assertEqual(set(captured["fit_kwargs"]), {"group", "sample_weight"})


class DecoderAndStorageTests(unittest.TestCase):
    def test_rejected_dsu_read_is_a_literal_full_state_noop(self) -> None:
        dsu = object.__new__(selector._PotentialDSU)
        dsu.result = None
        dsu.components = ()
        dsu.parent = {0: 0, 1: 0, 2: 1}
        dsu.size = {0: 3, 1: 1, 2: 1}
        dsu.delta = {0: (0, 0), 1: (1, 0), 2: (1, 0)}
        dsu.members = {0: {0, 1, 2}}
        dsu.entries = {0: {(0, 0): 0, (1, 0): 1, (2, 0): 2}}
        dsu.translations = {0: {0: (0, 0), 1: (1, 0), 2: (2, 0)}}
        before = copy.deepcopy(dsu.__dict__)
        selection = selector.SelectedRelation(
            0, 0, 2, 0, 99, 99, 1.0, 0.0, 1.0, 1
        )
        self.assertEqual(
            dsu.try_accept(selection), (False, "conflict", False, False)
        )
        self.assertEqual(dsu.__dict__, before)

    def test_feature_cache_roundtrip_is_restricted_to_frozen_e_root(self) -> None:
        table = _feature_table((2, 2))
        with tempfile.TemporaryDirectory(dir=E24_TEST_ROOT) as temporary:
            path = Path(temporary) / "features.npz"
            selector.save_feature_table_npz(path, table)
            self.assertEqual(path.read_bytes(), selector.feature_table_npz_bytes(table))
            self.assertEqual(
                selector.feature_table_npz_bytes(table),
                selector.feature_table_npz_bytes(table),
            )
            loaded = selector.load_feature_table_npz(path)
            np.testing.assert_array_equal(loaded.features, table.features)
            np.testing.assert_array_equal(loaded.query_offsets, table.query_offsets)
            self.assertFalse(loaded.features.flags.writeable)
            tampered = Path(temporary) / "dtype_tampered.npz"
            np.savez(
                tampered,
                schema_version=np.asarray([selector.SCHEMA_VERSION], dtype=np.int64),
                feature_names=np.asarray(selector.FEATURE_NAMES),
                features=table.features.astype(np.float64),
                hypothesis_ids=table.hypothesis_ids,
                relation_ids=table.relation_ids,
                relations=table.relations,
                row_kind=table.row_kind,
                support=table.support,
                query_offsets=table.query_offsets,
                scene_offsets=table.scene_offsets,
            )
            with self.assertRaisesRegex(
                selector.ContextRelationSelectorError, "features dtype drifted"
            ):
                selector.load_feature_table_npz(tampered)
            noncanonical = Path(temporary) / "timestamped.npz"
            np.savez(
                noncanonical,
                schema_version=np.asarray([selector.SCHEMA_VERSION], dtype=np.int64),
                feature_names=np.asarray(
                    selector.FEATURE_NAMES,
                    dtype=f"<U{max(map(len, selector.FEATURE_NAMES))}",
                ),
                features=table.features,
                hypothesis_ids=table.hypothesis_ids,
                relation_ids=table.relation_ids,
                relations=table.relations,
                row_kind=table.row_kind,
                support=table.support,
                query_offsets=table.query_offsets,
                scene_offsets=table.scene_offsets,
            )
            with self.assertRaisesRegex(
                selector.ContextRelationSelectorError, "not canonical"
            ):
                selector.load_feature_table_npz(noncanonical)
        with self.assertRaisesRegex(selector.ContextRelationSelectorError, "E24 feature caches"):
            selector.save_feature_table_npz(Path("C:/forbidden/features.npz"), table)


if __name__ == "__main__":
    unittest.main()
