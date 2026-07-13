from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from puzzle_assembly.compatibility import CompatibilityMatrices
from puzzle_assembly.geometry import GRID, TILE_COUNT, inverse_permutation


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_candidate_graph_oracle.py"
_SPEC = importlib.util.spec_from_file_location("candidate_graph_oracle", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
oracle = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = oracle
_SPEC.loader.exec_module(oracle)


def _empty_graph() -> oracle.CandidateGraph:
    return oracle.CandidateGraph(
        direction=np.empty(0, dtype=np.uint8),
        source=np.empty(0, dtype=np.uint16),
        destination=np.empty(0, dtype=np.uint16),
        origin_mask=np.empty(0, dtype=np.uint8),
        c1_cost=np.empty(0, dtype=np.float32),
        hbt_cost=np.empty(0, dtype=np.float32),
        w1_cost=np.empty(0, dtype=np.float32),
        w4_cost=np.empty(0, dtype=np.float32),
    )


def _graph_from_edges(
    edges: list[tuple[int, int, int]], *, masks: list[int] | None = None
) -> oracle.CandidateGraph:
    ordered = sorted(edges)
    mask_by_edge = {
        edge: (masks[index] if masks is not None else 1)
        for index, edge in enumerate(edges)
    }
    count = len(ordered)
    costs = np.arange(count, dtype=np.float32) / max(count, 1)
    return oracle.CandidateGraph(
        direction=np.asarray([value[0] for value in ordered], dtype=np.uint8),
        source=np.asarray([value[1] for value in ordered], dtype=np.uint16),
        destination=np.asarray([value[2] for value in ordered], dtype=np.uint16),
        origin_mask=np.asarray([mask_by_edge[value] for value in ordered], dtype=np.uint8),
        c1_cost=costs.copy(),
        hbt_cost=costs.copy(),
        w1_cost=costs.copy(),
        w4_cost=costs.copy(),
    )


def _true_edges_identity() -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for row in range(GRID):
        for column in range(GRID):
            tile = row * GRID + column
            if column + 1 < GRID:
                result.append((0, tile, tile + 1))
            if row + 1 < GRID:
                result.append((1, tile, tile + GRID))
    return result


def _flat_score() -> CompatibilityMatrices:
    matrix = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    np.fill_diagonal(matrix, np.inf)
    return CompatibilityMatrices("flat_w4", matrix.copy(), matrix.copy())


def _canonical_components(
    components: list[dict[int, tuple[int, int]]],
) -> list[tuple[tuple[int, int, int], ...]]:
    result = []
    for component in components:
        min_x = min(value[0] for value in component.values())
        min_y = min(value[1] for value in component.values())
        result.append(
            tuple(
                (tile, coordinate[0] - min_x, coordinate[1] - min_y)
                for tile, coordinate in sorted(component.items())
            )
        )
    return sorted(result, key=lambda value: (-len(value), value))


def test_exact_empty_and_split_candidate_graph_ceilings() -> None:
    truth = np.arange(TILE_COUNT, dtype=np.int32)
    exact = _graph_from_edges(_true_edges_identity())
    exact_recall = oracle.candidate_recall_metrics(exact, truth)
    exact_components, exact_diagnostics = oracle.truth_filtered_components(exact, truth)
    assert exact_recall["unique_true_edge_recall"] == 1.0
    assert exact_recall["four_side_recall"] == 1.0
    assert exact_diagnostics["largest_connected_component"] == TILE_COUNT
    assert len(exact_components) == 1

    empty = _empty_graph()
    empty_recall = oracle.candidate_recall_metrics(empty, truth)
    empty_components, empty_diagnostics = oracle.truth_filtered_components(empty, truth)
    assert empty_recall["unique_true_edge_recall"] == 0.0
    assert empty_diagnostics["largest_connected_component"] == 1
    assert len(empty_components) == TILE_COUNT

    horizontal = _graph_from_edges(
        [edge for edge in _true_edges_identity() if edge[0] == 0]
    )
    _, split_diagnostics = oracle.truth_filtered_components(horizontal, truth)
    assert split_diagnostics["largest_connected_component"] == GRID
    assert split_diagnostics["component_sizes"] == [GRID] * GRID


def test_false_candidate_edge_is_filtered_before_component_growth() -> None:
    truth = np.arange(TILE_COUNT, dtype=np.int32)
    graph = _graph_from_edges([(0, 0, 2), (0, 0, 1)])
    graph.w4_cost[:] = np.asarray([0.0, 1.0], dtype=np.float32)
    components, diagnostics = oracle.truth_filtered_components(graph, truth)
    assert diagnostics["truth_filtered_candidate_edges"] == 1
    assert diagnostics["largest_connected_component"] == 2
    assert set(components[0]) == {0, 1}
    assert 2 not in components[0]


def test_stable_ties_and_origin_bits_are_or_deduplicated() -> None:
    matrix = np.zeros((TILE_COUNT, TILE_COUNT), dtype=np.float32)
    np.fill_diagonal(matrix, np.inf)
    identity = np.arange(TILE_COUNT, dtype=np.int32)
    base = CompatibilityMatrices("base", matrix.copy(), matrix.copy())
    fused = oracle.fuse_ranked_scores(
        {"c1": base, "hbt": base},
        names=["c1", "hbt"],
        weights={"hbt": 1.0},
        name="fused",
    )
    fused_w4 = oracle.fuse_ranked_scores(
        {"c1": base, "hbt": base},
        names=["c1", "hbt"],
        weights={"hbt": 4.0},
        name="fused_w4",
    )
    arrays = {
        "c1_right": matrix.copy(),
        "c1_down": matrix.copy(),
        "hbt_right": matrix.copy(),
        "hbt_down": matrix.copy(),
        "w1_right": fused.right.copy(),
        "w1_down": fused.down.copy(),
        "w4_right": fused_w4.right.copy(),
        "w4_down": fused_w4.down.copy(),
        "softcycle_layout": identity.copy(),
        "qap_w4_layout": identity.copy(),
        "qap_w1_layout": identity.copy(),
        "denoised_tiles": np.zeros((TILE_COUNT, 20, 20, 3), dtype=np.uint8),
    }
    graph = oracle.build_candidate_graph(arrays)
    lookup = {
        (int(direction), int(first), int(second)): int(mask)
        for direction, first, second, mask in zip(
            graph.direction,
            graph.source,
            graph.destination,
            graph.origin_mask,
            strict=True,
        )
    }
    assert lookup[(0, 0, 1)] == 127
    assert lookup[(1, 0, GRID)] & (
        oracle.ORIGIN_BITS["softcycle"]
        | oracle.ORIGIN_BITS["qap_w4"]
        | oracle.ORIGIN_BITS["qap_w1"]
    )
    outgoing, incoming = oracle._stable_candidates(matrix, outgoing=32, incoming=8)
    assert len(outgoing) == TILE_COUNT * 32
    assert len(incoming) == TILE_COUNT * 8
    assert outgoing[:32, 1].tolist() == list(range(1, 33))


def test_component_partition_and_offsets_are_order_invariant() -> None:
    edges = _true_edges_identity()[:200]
    proposals = [
        oracle.ProposedEdge(
            first=first,
            second=second,
            dx=1 if direction == 0 else 0,
            dy=0 if direction == 0 else 1,
            cost=float(index),
            margin=0.0,
            reciprocal=False,
            in_loop=False,
        )
        for index, (direction, first, second) in enumerate(edges)
    ]
    reference, _ = oracle.grow_components(proposals)
    shuffled = proposals.copy()
    random.Random(20260712).shuffle(shuffled)
    candidate, _ = oracle.grow_components(shuffled)
    assert _canonical_components(candidate) == _canonical_components(reference)


def test_singleton_translation_diagnostic_cannot_truth_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = [{tile: (0, 0)} for tile in range(TILE_COUNT)]
    truth_layout = np.roll(np.arange(TILE_COUNT, dtype=np.int32), 1)
    slot_to_target = inverse_permutation(truth_layout)

    def completion(
        grid: np.ndarray, _compatibility: CompatibilityMatrices, **kwargs: object
    ) -> tuple[np.ndarray, int]:
        assert kwargs == {"boundary_weight": 0.05, "placement_costs": None}
        assert np.all(grid < 0)
        grid[:] = np.arange(TILE_COUNT, dtype=np.int32).reshape(GRID, GRID)
        return grid.ravel().copy(), TILE_COUNT

    monkeypatch.setattr(oracle, "_complete_with_hungarian", completion)
    layout, diagnostics = oracle.target_assisted_translation_ceiling(
        components, _flat_score(), slot_to_target
    )
    np.testing.assert_array_equal(layout, np.arange(TILE_COUNT, dtype=np.int32))
    assert not np.array_equal(layout, truth_layout)
    assert diagnostics["singleton_truth_placements"] == 0
    assert diagnostics["non_singleton_assisted_tiles"] == 0
    assert diagnostics["eligible_for_gate"] is False


def test_exact_beam_hungarian_qap_call_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[tuple[str, dict[str, object]]] = []
    components = [{0: (0, 0), 1: (1, 0)}, *({tile: (0, 0)} for tile in range(2, TILE_COUNT))]
    grid = np.full((GRID, GRID), -1, dtype=np.int32)
    grid[0, 0:2] = [0, 1]

    def place(
        values: list[dict[int, tuple[int, int]]],
        _compatibility: CompatibilityMatrices,
        **kwargs: object,
    ) -> tuple[np.ndarray, int]:
        assert values == components
        trace.append(("beam", kwargs))
        return grid.copy(), 2

    def complete(
        value: np.ndarray,
        _compatibility: CompatibilityMatrices,
        **kwargs: object,
    ) -> tuple[np.ndarray, int]:
        trace.append(("hungarian", kwargs))
        assert value is not grid
        value.ravel()[2:] = np.arange(2, TILE_COUNT, dtype=np.int32)
        return value.ravel().copy(), TILE_COUNT - 2

    def qap(
        _compatibility: CompatibilityMatrices, **kwargs: object
    ) -> object:
        trace.append(("qap", kwargs))
        return SimpleNamespace(
            position_to_slot=np.arange(TILE_COUNT, dtype=np.int32),
            objective=1.0,
            relaxed_objective=2.0,
            restart=0,
        )

    monkeypatch.setattr(oracle, "_place_components_beam", place)
    monkeypatch.setattr(oracle, "_complete_with_hungarian", complete)
    monkeypatch.setattr(oracle, "directional_qap", qap)
    layout, diagnostics = oracle.oracle_filter_beam_hungarian_qap(
        components, _flat_score(), qap_seed=12345
    )
    np.testing.assert_array_equal(layout, np.arange(TILE_COUNT, dtype=np.int32))
    assert [value[0] for value in trace] == ["beam", "hungarian", "qap"]
    assert trace[0][1] == {
        "boundary_weight": 0.05,
        "beam_width": 8,
        "beam_components": 8,
        "translations_per_state": 8,
        "placement_costs": None,
    }
    assert trace[1][1] == {"boundary_weight": 0.05, "placement_costs": None}
    qap_call = trace[2][1]
    assert qap_call["iterations"] == 25
    assert qap_call["restarts"] == 2
    assert qap_call["seed"] == 12345
    assert qap_call["boundary_weight"] == 0.05
    assert qap_call["initial_weight"] == 0.75
    assert qap_call["noisy_components"] == 3
    assert qap_call["noise_scale"] == 1.0
    assert qap_call["refine_swaps"] == 8
    assert qap_call["refine_weak_cells"] == 32
    assert diagnostics["qap_iterations"] == 25


def _passing_panels() -> dict[str, dict[str, float]]:
    return {
        panel: {
            "mean_union_true_edge_recall": 0.65,
            "median_largest_connected_component": 128.0,
            "mean_beam_qap_adjacency_delta": 0.10,
            "mean_beam_qap_ssim_delta": 0.0,
        }
        for panel in oracle.PANELS
    }


def test_gate_thresholds_are_inclusive_and_major_gain_is_explicit_or() -> None:
    adjacency = oracle.evaluate_continuation_gate(_passing_panels())
    assert adjacency["major_gain_adjacency_ge_0.10"] is True
    assert adjacency["major_gain_ssim_ge_0.02"] is False
    assert adjacency["major_gain_or_passed"] is True
    assert adjacency["continue_to_cycle_factor_synchronizer"] is True

    panels = _passing_panels()
    for panel in oracle.PANELS:
        panels[panel]["mean_beam_qap_adjacency_delta"] = 0.0
        panels[panel]["mean_beam_qap_ssim_delta"] = 0.02
    ssim = oracle.evaluate_continuation_gate(panels)
    assert ssim["major_gain_adjacency_ge_0.10"] is False
    assert ssim["major_gain_ssim_ge_0.02"] is True
    assert ssim["continue_to_cycle_factor_synchronizer"] is True


def test_gate_rejects_nan_missing_and_negative_panel() -> None:
    panels = _passing_panels()
    panels[oracle.PANELS[0]]["mean_union_true_edge_recall"] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        oracle.evaluate_continuation_gate(panels)
    with pytest.raises(RuntimeError, match="both exact panels"):
        oracle.evaluate_continuation_gate({oracle.PANELS[0]: _passing_panels()[oracle.PANELS[0]]})
    panels = _passing_panels()
    panels[oracle.PANELS[0]]["mean_beam_qap_ssim_delta"] = -1e-12
    result = oracle.evaluate_continuation_gate(panels)
    assert result["panel_guards"][oracle.PANELS[0]][
        "beam_qap_ssim_delta_nonnegative"
    ] is False
    assert result["continue_to_cycle_factor_synchronizer"] is False


def _opaque_manifest() -> dict[str, object]:
    records = []
    for index in range(64):
        opaque_id = f"{index:032x}"
        records.append(
            {
                "opaque_id": opaque_id,
                "artifact": {
                    "path": f"records/{opaque_id}.npz",
                    "bytes": 1,
                    "sha256": "0" * 64,
                },
                "arrays": {
                    "slot_tiles": {
                        "semantic": "opaque corrupted input slot tiles",
                        "dtype": "uint8",
                        "shape": [576, 20, 20, 3],
                        "c_order_sha256": "1" * 64,
                    },
                    "qap_seed": {
                        "semantic": "opaque nuisance seed",
                        "dtype": "uint64",
                        "shape": [],
                        "c_order_sha256": "2" * 64,
                    },
                },
            }
        )
    ids = [record["opaque_id"] for record in records]
    payload = {
        "schema_version": 1,
        "created_utc": "2026-07-12T00:00:00Z",
        "kind": "candidate_graph_oracle_fixture_inputs",
        "protocol_instance_id": oracle.EXPECTED_PROTOCOL_INSTANCE_ID,
        "frozen_contract_sha256": oracle.EXPECTED_FROZEN_CONTRACT_SHA256,
        "record_count": 64,
        "opaque_ids_sha256": oracle.hashlib.sha256(
            "\n".join(ids).encode("ascii")
        ).hexdigest(),
        "canonical_record_order": "ascending opaque_id",
        "allowed_record_metadata": ["opaque_id", "artifact", "arrays"],
        "records": records,
    }
    for key in oracle.COMMON_MANIFEST_SHA_FIELDS:
        if key not in payload:
            payload[key] = "4" * 64
    return payload


def test_phase_a_manifest_schema_is_opaque_and_exact() -> None:
    manifest = _opaque_manifest()
    records = oracle._opaque_input_records(manifest)
    assert len(records) == 64
    leaked = json.loads(json.dumps(manifest))
    leaked["records"][0]["source_name"] = "img_000001.png"
    with pytest.raises(RuntimeError, match="schema drift"):
        oracle._opaque_input_records(leaked)
    extra = json.loads(json.dumps(manifest))
    extra["panel"] = "primary_kornia"
    with pytest.raises(RuntimeError, match="schema-drift"):
        oracle._opaque_input_records(extra)


def test_phase_a_manifest_rejects_coverage_order_and_envelope_tamper(
    tmp_path: Path,
) -> None:
    manifest = _opaque_manifest()
    missing = json.loads(json.dumps(manifest))
    missing["records"].pop()
    with pytest.raises(RuntimeError, match="exactly 64"):
        oracle._opaque_input_records(missing)
    shuffled = json.loads(json.dumps(manifest))
    shuffled["records"][0], shuffled["records"][1] = (
        shuffled["records"][1],
        shuffled["records"][0],
    )
    with pytest.raises(RuntimeError, match="canonical lexicographic"):
        oracle._opaque_input_records(shuffled)

    path = tmp_path / "manifest.json"
    oracle._write_envelope(path, manifest)
    anchor = oracle._sha256(path)
    assert oracle._load_envelope(path, anchor)["payload"] == manifest
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        oracle._load_envelope(path, anchor)


def test_phase_b_report_contract_is_exact_canonical_envelope(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_ceiling_report",
        "safe_for_submission": False,
    }
    path = tmp_path / oracle.REPORT_NAME
    envelope = oracle._write_envelope(path, payload)
    assert set(envelope) == {"payload", "payload_sha256"}
    assert envelope["payload"] == payload
    assert path.read_bytes() == oracle._canonical_bytes(envelope) + b"\n"
    assert oracle._load_envelope(path, oracle._sha256(path)) == envelope


def _fake_shards() -> list[tuple[dict[str, object], list[tuple[dict[str, object], bytes, dict[str, bytes]]]]]:
    bindings = {
        "config_sha256": "a" * 64,
        "protocol_instance_id": oracle.EXPECTED_PROTOCOL_INSTANCE_ID,
        "frozen_contract_sha256": oracle.EXPECTED_FROZEN_CONTRACT_SHA256,
        "phase_a_lifecycle_sha256": "b" * 64,
        "script_sha256": "c" * 64,
        "fixture_manifest_sha256": "d" * 64,
        "fixture_manifest_name": "fixture_input_manifest.json",
        "runtime_asset_sha256": {"denoiser": "f" * 64, "hbt": "0" * 64},
        "runtime_pin_sha256": {"evaluator_sha256": "1" * 64},
    }
    shards = []
    for rank in (0, 1):
        payload = {**bindings, "rank": rank}
        records = []
        for index in range(rank, 64, 2):
            opaque_id = f"{index:032x}"
            record = {
                "opaque_id": opaque_id,
                "qap_seed": oracle._opaque_qap_seed(opaque_id),
            }
            records.append((record, b"graph", {}))
        shards.append((payload, records))
    return shards


def test_phase_a_two_shards_are_disjoint_and_cover_all_64() -> None:
    shards = _fake_shards()
    oracle._validate_shard_common_bindings(shards)
    records = oracle._validate_shard_coverage(shards)
    assert len(records) == 64
    assert [record[0]["opaque_id"] for record in records] == [
        f"{index:032x}" for index in range(64)
    ]


def test_phase_a_shard_finalize_rejects_missing_duplicate_and_mixed_binding() -> None:
    missing = _fake_shards()
    missing[1][1].pop()
    with pytest.raises(RuntimeError, match="missing Phase-A shard coverage"):
        oracle._validate_shard_coverage(missing)

    duplicate = _fake_shards()
    duplicate[1][1][0] = duplicate[0][1][0]
    with pytest.raises(RuntimeError, match="duplicate opaque id"):
        oracle._validate_shard_coverage(duplicate)

    mixed = _fake_shards()
    mixed[1][0]["config_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="mixed Phase-A shard binding"):
        oracle._validate_shard_common_bindings(mixed)


def test_exact_external_lifecycle_chain_is_verified_and_tamper_fails(
    tmp_path: Path,
) -> None:
    instance = oracle.EXPECTED_PROTOCOL_INSTANCE_ID
    final_config = "a" * 64
    previous: str | None = None
    for state, config_hash in (
        ("PREP", "b" * 64),
        ("SEALED", final_config),
        ("PHASE_A", final_config),
    ):
        payload = {
            "schema_version": 1,
            "kind": "candidate_graph_oracle_lifecycle",
            "protocol_instance_id": instance,
            "state": state,
            "frozen_contract_sha256": oracle.EXPECTED_FROZEN_CONTRACT_SHA256,
            "config_sha256_or_null": config_hash,
            "predecessor_sha256": previous,
        }
        encoded = oracle._ledger_canonical_bytes(payload)
        (tmp_path / f"{state}.json").write_bytes(encoded)
        previous = oracle._bytes_sha256(encoded)
    _, hashes = oracle._verify_lifecycle_chain(
        str(tmp_path),
        protocol_instance_id=instance,
        config_sha256=final_config,
        required_last_state="PHASE_A",
    )
    assert list(hashes) == ["PREP", "SEALED", "PHASE_A"]

    protocol = {
        "protocol_instance_id": instance,
        "runtime_pins": {
            "code_path": "scripts/code.py",
            "code_sha256": "1" * 64,
            "fixture_path": "fixture/input.json",
            "fixture_sha256": "2" * 64,
        },
        "runtime_pin_mutation_policy": {
            "code_pin_fields": [
                {"path_field": "code_path", "sha256_field": "code_sha256"}
            ],
            "fixture_pin_fields": [
                {
                    "path_field": "fixture_path",
                    "sha256_field": "fixture_sha256",
                }
            ],
        },
    }
    transition = tmp_path / "runtime_pin_transitions"
    transition.mkdir()

    def write_receipts(
        stage: str,
        index: int,
        prefix: str,
        previous_config: str,
        final_config_hash: str,
        pin_values: dict[str, str],
    ) -> None:
        common = {
            "schema_version": 1,
            "stage": stage,
            "stage_index": index,
            "protocol_instance_id": instance,
            "frozen_contract_sha256": oracle.EXPECTED_FROZEN_CONTRACT_SHA256,
            "config_relative_path": (
                "configs/candidate_graph_oracle_ceiling_v3.json"
            ),
        }
        intent = {
            **common,
            "kind": "candidate_graph_oracle_runtime_pin_transition_intent",
            "previous_config_sha256": previous_config,
            "intended_config_sha256": final_config_hash,
            "pin_sha256_values": pin_values,
            "created_utc": "2026-07-12T00:00:00Z",
        }
        intent_bytes = oracle._ledger_canonical_bytes(intent)
        (transition / f"{prefix}.intent.json").write_bytes(intent_bytes)
        completion = {
            **common,
            "kind": "candidate_graph_oracle_runtime_pin_transition_completion",
            "previous_config_sha256": previous_config,
            "final_config_sha256": final_config_hash,
            "pin_sha256_values": pin_values,
            "intent_sha256": oracle._bytes_sha256(intent_bytes),
            "completed_utc": "2026-07-12T00:00:01Z",
        }
        (transition / f"{prefix}.complete.json").write_bytes(
            oracle._ledger_canonical_bytes(completion)
        )

    write_receipts(
        "code", 0, "00_code_pins", "0" * 64, "b" * 64, {"code_sha256": "1" * 64}
    )
    write_receipts(
        "fixtures",
        1,
        "01_fixtures_pins",
        "b" * 64,
        final_config,
        {"fixture_sha256": "2" * 64},
    )
    _, receipt_hashes = oracle._verify_lifecycle_chain(
        str(tmp_path),
        protocol_instance_id=instance,
        config_sha256=final_config,
        required_last_state="PHASE_A",
        protocol=protocol,
    )
    assert receipt_hashes == hashes

    path = tmp_path / "SEALED.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="canonical bytes drift"):
        oracle._verify_lifecycle_chain(
            str(tmp_path),
            protocol_instance_id=instance,
            config_sha256=final_config,
            required_last_state="PHASE_A",
            protocol=protocol,
        )


def test_secure_open_rejects_symlink_hardlink_and_unlisted_entry(
    tmp_path: Path,
) -> None:
    records = tmp_path / "records"
    records.mkdir()
    original = records / "a.bin"
    original.write_bytes(b"safe")
    payload, _, _ = oracle._secure_relative_bytes(
        tmp_path, "records/a.bin", expected_parent="records"
    )
    assert payload == b"safe"

    symlink = records / "symlink.bin"
    symlink.symlink_to(original)
    with pytest.raises(OSError):
        oracle._secure_relative_bytes(
            tmp_path, "records/symlink.bin", expected_parent="records"
        )
    symlink.unlink()

    hardlink = records / "hardlink.bin"
    hardlink.hardlink_to(original)
    with pytest.raises(RuntimeError, match="nlink==1"):
        oracle._secure_relative_bytes(
            tmp_path, "records/a.bin", expected_parent="records"
        )
    hardlink.unlink()
    extra = tmp_path / "extra.txt"
    extra.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unlisted directory entry"):
        oracle._assert_exact_directory_entries(tmp_path, {"records"})


def test_label_only_secret_recomposition_is_byte_exact_and_input_tamper_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = bytes(range(32))
    source_name = "img_000123.png"
    panel = "primary_kornia"

    def material(prefix: str) -> bytes:
        return oracle.hmac.new(
            secret,
            f"{prefix}:{source_name}:{panel}".encode("utf-8"),
            oracle.hashlib.sha256,
        ).digest()

    opaque_id = material("id")[:16].hex()
    shuffle_seed = int.from_bytes(material("shuffle")[:8], "big")
    permutation = (
        np.random.Generator(np.random.PCG64(shuffle_seed))
        .permutation(TILE_COUNT)
        .astype(np.int32)
    )
    exact_tiles = np.arange(TILE_COUNT, dtype=np.uint16)[:, None, None, None]
    exact_tiles = np.broadcast_to(
        (exact_tiles % 251).astype(np.uint8), (TILE_COUNT, 20, 20, 3)
    ).copy()
    exact_truth = np.arange(TILE_COUNT, dtype=np.int32)
    clean = np.zeros((480, 480, 3), dtype=np.uint8)
    labels = {
        "opaque_slot_permutation": permutation,
        "composed_slot_to_target": exact_truth[permutation],
        "clean_target_rgb": clean,
    }
    input_arrays = {
        "slot_tiles": np.ascontiguousarray(exact_tiles[permutation]),
        "qap_seed": np.asarray(oracle._opaque_qap_seed(opaque_id), dtype=np.uint64),
    }
    panel_seed = oracle.per_source_seed(
        20260711, f"candidate-graph-oracle-{panel}", source_name, 0
    )
    label_record = {
        "source_name": source_name,
        "panel": panel,
        "panel_seed": panel_seed,
        "arrays": {
            key: oracle._array_descriptor(value, key) for key, value in labels.items()
        },
    }
    phase_record = {
        "input_slot_tiles_c_sha256": oracle._array_c_sha256(
            input_arrays["slot_tiles"]
        ),
        "qap_seed": oracle._opaque_qap_seed(opaque_id),
    }
    monkeypatch.setattr(
        oracle,
        "make_exact_panel",
        lambda *_args, **_kwargs: SimpleNamespace(
            slot_tiles=exact_tiles, slot_to_target=exact_truth
        ),
    )
    result = oracle.recompute_fixture_binding_after_marker(
        secret=secret,
        opaque_id=opaque_id,
        label_record=label_record,
        labels=labels,
        input_arrays=input_arrays,
        phase_record=phase_record,
        master_seed=20260711,
    )
    assert all(result.values())

    tampered = {**input_arrays, "slot_tiles": input_arrays["slot_tiles"].copy()}
    tampered["slot_tiles"][0, 0, 0, 0] ^= 1
    with pytest.raises(RuntimeError, match="decoded input tile recomputation mismatch"):
        oracle.recompute_fixture_binding_after_marker(
            secret=secret,
            opaque_id=opaque_id,
            label_record=label_record,
            labels=labels,
            input_arrays=tampered,
            phase_record=phase_record,
            master_seed=20260711,
        )


def test_phase_b_cli_exposes_only_opaque_bundle_for_label_side() -> None:
    args = oracle.parse_args(
        [
            "--action",
            "phase-b",
            "--fixture-bundle-root",
            "opaque-bundle-string",
        ]
    )
    assert args.fixture_bundle_root == "opaque-bundle-string"
    assert not hasattr(args, "labels_manifest")
    assert not hasattr(args, "labels_manifest_sha256")
    assert not hasattr(args, "labels_root")
    assert not hasattr(args, "label_secret")
    with pytest.raises(SystemExit):
        oracle.parse_args(
            [
                "--action",
                "phase-b",
                "--labels-root",
                "/forbidden",
            ]
        )
