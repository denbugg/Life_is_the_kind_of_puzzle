from __future__ import annotations

import copy
from io import BytesIO
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import push_candidate_graph_oracle_v4_phase_a as launcher

_REPO = Path(__file__).resolve().parents[1]
_SNAPSHOT_SRC = (
    _REPO
    / "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_source_snapshot/src"
)
for _name in tuple(sys.modules):
    if _name == "puzzle_assembly" or _name.startswith("puzzle_assembly."):
        del sys.modules[_name]
    elif _name == "puzzle_denoise_v2" or _name.startswith("puzzle_denoise_v2."):
        del sys.modules[_name]
sys.path.insert(0, str(_SNAPSHOT_SRC))


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/verify_candidate_graph_oracle_v4_result.py"
)
_SPEC = importlib.util.spec_from_file_location("candidate_graph_oracle_verifier", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
verifier = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verifier
_SPEC.loader.exec_module(verifier)


def _attestation_context() -> verifier.ProtocolContext:
    repository = Path(__file__).resolve().parents[1]
    config_path = repository / "configs/candidate_graph_oracle_ceiling_v4.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pins = config["runtime_pins"]
    for pair in config["runtime_pin_mutation_policy"]["code_pin_fields"]:
        relative = pins[pair["path_field"]]
        pins[pair["sha256_field"]] = hashlib.sha256(
            (repository / relative).read_bytes()
        ).hexdigest()
    pins["fixture_input_manifest_sha256"] = "1" * 64
    pins["fixture_label_manifest_sha256"] = "2" * 64
    pins["fixture_lock_sha256"] = "3" * 64
    return verifier.ProtocolContext(config, config_path, "a" * 64, repository)


def _npz(**arrays: np.ndarray) -> bytes:
    output = BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def _identity_derived() -> dict[str, np.ndarray]:
    flat = np.zeros((verifier.TILE_COUNT, verifier.TILE_COUNT), dtype=np.float32)
    np.fill_diagonal(flat, np.inf)
    identity = np.arange(verifier.TILE_COUNT, dtype=np.int32)
    arrays = {
        "c1_right": flat.copy(),
        "c1_down": flat.copy(),
        "hbt_right": flat.copy(),
        "hbt_down": flat.copy(),
        "softcycle_layout": identity.copy(),
        "qap_w4_layout": identity.copy(),
        "qap_w1_layout": identity.copy(),
        "denoised_tiles": np.zeros(
            (verifier.TILE_COUNT, verifier.TILE, verifier.TILE, 3), dtype=np.uint8
        ),
    }
    for suffix in ("right", "down"):
        arrays[f"w1_{suffix}"] = verifier._rank_fusion(
            arrays[f"c1_{suffix}"], arrays[f"hbt_{suffix}"], hbt_weight=1.0
        )
        arrays[f"w4_{suffix}"] = verifier._rank_fusion(
            arrays[f"c1_{suffix}"], arrays[f"hbt_{suffix}"], hbt_weight=4.0
        )
    return arrays


def _candidate_arrays(edges: list[tuple[int, int, int]]) -> dict[str, np.ndarray]:
    ordered = sorted(edges)
    return {
        "candidate_direction": np.asarray([edge[0] for edge in ordered], dtype=np.uint8),
        "candidate_source": np.asarray([edge[1] for edge in ordered], dtype=np.uint16),
        "candidate_destination": np.asarray([edge[2] for edge in ordered], dtype=np.uint16),
        "candidate_origin_mask": np.ones(len(ordered), dtype=np.uint8),
    }


def _truth_edges() -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for row in range(verifier.GRID):
        for column in range(verifier.GRID):
            tile = row * verifier.GRID + column
            if column + 1 < verifier.GRID:
                result.append((0, tile, tile + 1))
            if row + 1 < verifier.GRID:
                result.append((1, tile, tile + verifier.GRID))
    return result


def test_canonical_envelope_and_self_hash_fail_closed() -> None:
    base = {"kind": "x", "records": []}
    payload = {**base, "self_sha256": verifier._sha256_bytes(verifier._canonical_object_bytes(base))}
    envelope = {
        "payload": payload,
        "payload_sha256": verifier._sha256_bytes(verifier._canonical_object_bytes(payload)),
    }
    raw = verifier._canonical_file_bytes(envelope)
    loaded = verifier._load_envelope_bytes(
        raw, expected_file_sha256=hashlib.sha256(raw).hexdigest(), label="synthetic"
    )
    verifier._verify_self_sha256(loaded, label="synthetic")

    tampered = json.loads(raw)
    tampered["payload"]["records"] = [1]
    with pytest.raises(verifier.VerificationError, match="payload hash"):
        verifier._load_envelope_bytes(
            verifier._canonical_file_bytes(tampered),
            expected_file_sha256=None,
            label="tampered",
        )
    with pytest.raises(verifier.VerificationError, match="not canonical"):
        verifier._load_envelope_bytes(
            json.dumps(envelope).encode() + b"\n",
            expected_file_sha256=None,
            label="noncanonical",
        )
    with pytest.raises(verifier.VerificationError, match="non-finite"):
        verifier._parse_json(b'{"x":NaN}\n', label="nan", canonical_file=False)


def test_anchored_reader_rejects_symlink_hardlink_and_extra(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    (root_path / "good").write_bytes(b"ok")
    with verifier.AnchoredRoot.open(root_path) as root:
        assert root.read_file("good")[0] == b"ok"
        root.assert_exact_tree(top_files={"good"}, directories={})
        (root_path / "extra").write_bytes(b"x")
        with pytest.raises(verifier.VerificationError, match="root tree drift"):
            root.assert_exact_tree(top_files={"good"}, directories={})
        (root_path / "extra").unlink()
        (root_path / "link").symlink_to(root_path / "good")
        with pytest.raises(OSError):
            root.read_file("link")
        (root_path / "link").unlink()
        os.link(root_path / "good", root_path / "hard")
        with pytest.raises(verifier.VerificationError, match="one-link"):
            root.read_file("hard")


def test_strict_npz_rejects_extra_missing_and_wrong_dtype() -> None:
    specs = {"a": (np.dtype("int32"), (2,))}
    valid = verifier._strict_npz(
        _npz(a=np.asarray([1, 2], dtype=np.int32)), specs, label="valid"
    )
    assert valid["a"].tolist() == [1, 2]
    with pytest.raises(verifier.VerificationError, match="coverage"):
        verifier._strict_npz(
            _npz(a=np.asarray([1, 2], dtype=np.int32), extra=np.zeros(1)),
            specs,
            label="extra",
        )
    with pytest.raises(verifier.VerificationError, match="dtype/shape"):
        verifier._strict_npz(
            _npz(a=np.asarray([1, 2], dtype=np.int64)), specs, label="dtype"
        )


def test_candidate_union_stable_ties_origins_costs_and_tamper() -> None:
    arrays = _identity_derived()
    graph = verifier.rebuild_candidate_union(arrays)
    keys = list(
        zip(
            graph["candidate_direction"].tolist(),
            graph["candidate_source"].tolist(),
            graph["candidate_destination"].tolist(),
            strict=True,
        )
    )
    assert keys == sorted(keys)
    lookup = {
        key: int(mask)
        for key, mask in zip(keys, graph["candidate_origin_mask"], strict=True)
    }
    # Stable all-tie ordering is smaller slot index, excluding self.
    for destination in range(1, 33):
        assert lookup[(0, 0, destination)] & verifier.ORIGIN_BITS["c1_out32"]
        assert lookup[(0, 0, destination)] & verifier.ORIGIN_BITS["hbt_out32"]
    for source in range(1, 9):
        assert lookup[(0, source, 0)] & verifier.ORIGIN_BITS["c1_in8"]
        assert lookup[(0, source, 0)] & verifier.ORIGIN_BITS["hbt_in8"]
    assert lookup[(0, 0, 1)] & verifier.ORIGIN_BITS["softcycle"]
    assert lookup[(0, 0, 1)] & verifier.ORIGIN_BITS["qap_w4"]
    assert lookup[(0, 0, 1)] & verifier.ORIGIN_BITS["qap_w1"]
    assert all(
        np.all(np.isfinite(graph[key]))
        for key in (
            "candidate_c1_cost",
            "candidate_hbt_cost",
            "candidate_w1_cost",
            "candidate_w4_cost",
        )
    )
    tampered = graph["candidate_origin_mask"].copy()
    tampered[0] ^= np.uint8(1)
    assert not np.array_equal(tampered, graph["candidate_origin_mask"])


def test_candidate_union_rejects_frozen_rank_fusion_drift() -> None:
    arrays = _identity_derived()
    arrays["w4_right"] = arrays["w4_right"].copy()
    arrays["w4_right"][0, 1] += np.float32(0.25)
    with pytest.raises(verifier.VerificationError, match="rank fusion mismatch"):
        verifier.rebuild_candidate_union(arrays)


def test_phase_a_plain_self_manifest_requires_exact_ordered_two_shard_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = [f"{index:032x}" for index in range(64)]
    root_path = tmp_path / "phase_a"
    (root_path / "artifacts").mkdir(parents=True)
    (root_path / "renders").mkdir()
    for opaque_id in ids:
        (root_path / "artifacts" / f"{opaque_id}.graph.npz").write_bytes(b"g")
        (root_path / "renders" / f"{opaque_id}.png").write_bytes(b"r")
    pins = {
        "evaluator_sha256": "e" * 64,
        "fixture_input_manifest_sha256": "f" * 64,
    }
    context = SimpleNamespace(
        config_sha256="3" * 64,
        config={
            "runtime_pins": pins,
            "frozen_contract": {
                "assets": {
                    "denoiser": {"sha256": "d" * 64},
                    "hbt": {"sha256": "c" * 64},
                }
            },
        },
    )
    anchors = ("1" * 64, "2" * 64)
    payload = {
        "schema_version": 1,
        "kind": "frozen_candidate_graph_input_only",
        "config_sha256": context.config_sha256,
        "protocol_instance_id": verifier.EXPECTED_PROTOCOL_INSTANCE_ID,
        "frozen_contract_sha256": verifier.EXPECTED_FROZEN_CONTRACT_SHA256,
        "phase_a_lifecycle_sha256": "a" * 64,
        "script_sha256": pins["evaluator_sha256"],
        "fixture_manifest_sha256": "f" * 64,
        "fixture_manifest_name": verifier.INPUT_MANIFEST,
        "runtime_asset_sha256": {"denoiser": "d" * 64, "hbt": "c" * 64},
        "runtime_pin_sha256": {
            key: value for key, value in pins.items() if key.endswith("_sha256")
        },
        "shard_envelope_sha256s": list(anchors),
        "record_count": 64,
        "records": [{"opaque_id": opaque_id} for opaque_id in ids],
        "target_paths_constructed": False,
        "target_files_opened": False,
        "safe_for_submission": False,
    }
    payload["self_sha256"] = verifier._sha256_bytes(
        verifier._canonical_object_bytes(payload)
    )
    manifest_raw = verifier._canonical_file_bytes(payload)
    (root_path / verifier.PHASE_A_MANIFEST).write_bytes(manifest_raw)
    input_evidence = verifier.InputEvidence(
        tmp_path / "inputs",
        {},
        "f" * 64,
        {
            opaque_id: verifier.InputRecord({}, {})
            for opaque_id in ids
        },
    )

    def fake_record(
        record_value: dict[str, str],
        *,
        index: int,
        root: verifier.AnchoredRoot,
        input_record: verifier.InputRecord,
    ) -> tuple[verifier.PhaseARecord, str, set[str]]:
        del index, root, input_record
        opaque_id = record_value["opaque_id"]
        return (
            verifier.PhaseARecord(record_value),
            f"{opaque_id}.graph.npz",
            {f"{opaque_id}.png"},
        )

    monkeypatch.setattr(verifier, "_verify_phase_a_record", fake_record)
    expected_file_hash = hashlib.sha256(manifest_raw).hexdigest()
    result = verifier.verify_phase_a(
        context,
        phase_a_root=root_path,
        expected_envelope_sha256=expected_file_hash,
        shard_anchors=anchors,
        input_evidence=input_evidence,
    )
    assert result.shard_anchors == anchors
    with pytest.raises(verifier.VerificationError, match="two-shard anchor"):
        verifier.verify_phase_a(
            context,
            phase_a_root=root_path,
            expected_envelope_sha256=expected_file_hash,
            shard_anchors=anchors[::-1],
            input_evidence=input_evidence,
        )


def test_exact_empty_and_split_truth_graph_metrics() -> None:
    truth = np.arange(verifier.TILE_COUNT, dtype=np.int32)
    exact_recall, exact_components, _ = verifier._candidate_truth_metrics(
        _candidate_arrays(_truth_edges()), truth
    )
    assert exact_recall["unique_true_edge_recall"] == 1.0
    assert exact_recall["four_side_recall"] == 1.0
    assert exact_components["largest_connected_component"] == 576
    assert exact_components["component_sizes"] == [576]

    empty_recall, empty_components, _ = verifier._candidate_truth_metrics(
        _candidate_arrays([]), truth
    )
    assert empty_recall["unique_true_edge_recall"] == 0.0
    assert empty_components["largest_connected_component"] == 1
    assert empty_components["component_sizes"] == [1] * 576

    split = [
        edge
        for edge in _truth_edges()
        if not (edge[0] == 1 and edge[1] // verifier.GRID == 11)
    ]
    _, split_components, _ = verifier._candidate_truth_metrics(
        _candidate_arrays(split), truth
    )
    assert split_components["component_sizes"] == [288, 288]
    assert split_components["largest_connected_component"] == 288


def test_verifier_lifecycle_and_runtime_pin_receipt_chain_fail_closed(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    ledger_root = repository / "ledger"
    transitions = ledger_root / "runtime_pin_transitions"
    transitions.mkdir(parents=True)
    code_pins = {"evaluator_sha256": "e" * 64}
    fixture_pins = {"fixture_input_manifest_sha256": "f" * 64}
    config_sha = "b" * 64
    code_config_sha = "a" * 64
    context = SimpleNamespace(
        repository=repository,
        config_path=repository / "configs/protocol.json",
        config_sha256=config_sha,
        config={
            "runtime_pins": {**code_pins, **fixture_pins},
            "runtime_pin_mutation_policy": {
                "transition_ledger_root": "ledger",
                "code_pin_fields": [
                    {"path_field": "evaluator_path", "sha256_field": "evaluator_sha256"}
                ],
                "fixture_pin_fields": [
                    {
                        "path_field": "fixture_input_manifest_relative_path",
                        "sha256_field": "fixture_input_manifest_sha256",
                    }
                ],
            },
        },
    )
    context.config_path.parent.mkdir(parents=True)
    context.config_path.write_text("{}")

    previous = "0" * 64
    for stage, stage_index, prefix, pins, intended in (
        ("code", 0, "00_code_pins", code_pins, code_config_sha),
        ("fixtures", 1, "01_fixtures_pins", fixture_pins, config_sha),
    ):
        intent = {
            "schema_version": 1,
            "kind": "candidate_graph_oracle_runtime_pin_transition_intent",
            "stage": stage,
            "stage_index": stage_index,
            "protocol_instance_id": verifier.EXPECTED_PROTOCOL_INSTANCE_ID,
            "frozen_contract_sha256": verifier.EXPECTED_FROZEN_CONTRACT_SHA256,
            "config_relative_path": "configs/protocol.json",
            "previous_config_sha256": previous,
            "intended_config_sha256": intended,
            "pin_sha256_values": pins,
            "created_utc": "2026-07-12T00:00:00Z",
        }
        intent_raw = verifier._canonical_file_bytes(intent)
        (transitions / f"{prefix}.intent.json").write_bytes(intent_raw)
        completion = {
            "schema_version": 1,
            "kind": "candidate_graph_oracle_runtime_pin_transition_completion",
            "stage": stage,
            "stage_index": stage_index,
            "protocol_instance_id": verifier.EXPECTED_PROTOCOL_INSTANCE_ID,
            "frozen_contract_sha256": verifier.EXPECTED_FROZEN_CONTRACT_SHA256,
            "config_relative_path": "configs/protocol.json",
            "previous_config_sha256": previous,
            "final_config_sha256": intended,
            "pin_sha256_values": pins,
            "intent_sha256": hashlib.sha256(intent_raw).hexdigest(),
            "completed_utc": "2026-07-12T00:00:01Z",
        }
        (transitions / f"{prefix}.complete.json").write_bytes(
            verifier._canonical_file_bytes(completion)
        )
        previous = intended

    lifecycle_hashes: dict[str, str] = {}
    predecessor: str | None = None
    for state in verifier.LIFECYCLE_STATES:
        payload = {
            "schema_version": 1,
            "kind": "candidate_graph_oracle_lifecycle",
            "protocol_instance_id": verifier.EXPECTED_PROTOCOL_INSTANCE_ID,
            "state": state,
            "frozen_contract_sha256": verifier.EXPECTED_FROZEN_CONTRACT_SHA256,
            "config_sha256_or_null": (
                code_config_sha if state == "PREP" else config_sha
            ),
            "predecessor_sha256": predecessor,
        }
        raw = verifier._canonical_file_bytes(payload)
        (ledger_root / f"{state}.json").write_bytes(raw)
        predecessor = hashlib.sha256(raw).hexdigest()
        lifecycle_hashes[state] = predecessor
    phase_a = SimpleNamespace(
        payload={"phase_a_lifecycle_sha256": lifecycle_hashes["PHASE_A"]}
    )
    evidence = verifier.verify_lifecycle(
        context, lifecycle_ledger=ledger_root, phase_a=phase_a
    )
    assert evidence.hashes == lifecycle_hashes
    assert set(evidence.transition_hashes) == {
        "runtime_pin_transitions/00_code_pins.intent.json",
        "runtime_pin_transitions/00_code_pins.complete.json",
        "runtime_pin_transitions/01_fixtures_pins.intent.json",
        "runtime_pin_transitions/01_fixtures_pins.complete.json",
    }
    label_path = ledger_root / "LABEL_ACCESS.json"
    label_raw = label_path.read_bytes()
    label_path.unlink()
    phase_a_only = verifier.verify_phase_a_lifecycle(
        context, lifecycle_ledger=ledger_root, phase_a=phase_a
    )
    assert set(phase_a_only.hashes) == {"PREP", "SEALED", "PHASE_A"}
    assert phase_a_only.transition_hashes == evidence.transition_hashes
    hidden = transitions / "LABEL_ACCESS"
    hidden.mkdir()
    with pytest.raises(verifier.VerificationError, match="transition receipt coverage"):
        verifier.verify_phase_a_lifecycle(
            context, lifecycle_ledger=ledger_root, phase_a=phase_a
        )
    hidden.rmdir()
    label_path.write_bytes(label_raw)
    tampered = json.loads(label_path.read_text())
    tampered["predecessor_sha256"] = "9" * 64
    label_path.write_bytes(verifier._canonical_file_bytes(tampered))
    with pytest.raises(verifier.VerificationError, match="lifecycle chain mismatch"):
        verifier.verify_lifecycle(
            context, lifecycle_ledger=ledger_root, phase_a=phase_a
        )


@pytest.mark.parametrize(
    "component",
    ["fixture_label", "labels", "target", "FIXTURE_MASTER_SECRET.bin"],
)
def test_phase_a_read_path_guard_rejects_forbidden_components(
    tmp_path: Path, component: str
) -> None:
    forbidden = tmp_path / component / "evidence.json"
    forbidden.parent.mkdir()
    forbidden.write_text("{}")
    with pytest.raises(verifier.VerificationError, match="forbidden"):
        verifier._guard_phase_a_read_path(forbidden, label="synthetic")


def test_false_high_score_edge_never_changes_truth_components() -> None:
    truth = np.arange(verifier.TILE_COUNT, dtype=np.int32)
    only_false = _candidate_arrays([(0, 0, 2)])
    recall, components, _ = verifier._candidate_truth_metrics(only_false, truth)
    assert recall["unique_true_edge_recall"] == 0.0
    assert components["truth_filtered_candidate_edges"] == 0
    assert components["largest_connected_component"] == 1


def _summary(
    *, recall: float = 0.65, lcc: float = 128.0, adjacency: float = 0.10, ssim: float = 0.0
) -> dict[str, float]:
    return {
        "record_count": 32.0,
        "mean_union_true_edge_recall": recall,
        "median_largest_connected_component": lcc,
        "mean_beam_qap_adjacency_delta": adjacency,
        "mean_beam_qap_ssim_delta": ssim,
    }


def test_gate_exact_thresholds_negative_panel_and_translation_noncontribution() -> None:
    passing = {panel: _summary() for panel in verifier.PANELS}
    assert verifier._independent_gate(passing)[
        "continue_to_cycle_factor_synchronizer"
    ]
    negative = {panel: _summary() for panel in verifier.PANELS}
    negative[verifier.PANELS[1]]["mean_beam_qap_ssim_delta"] = -1e-12
    assert not verifier._independent_gate(negative)[
        "continue_to_cycle_factor_synchronizer"
    ]
    no_headroom = {panel: _summary(adjacency=0.099, ssim=0.019) for panel in verifier.PANELS}
    for value in no_headroom.values():
        value["target_assisted_translation_score"] = 10_000.0
    assert not verifier._independent_gate(no_headroom)[
        "continue_to_cycle_factor_synchronizer"
    ]
    nan = {panel: _summary() for panel in verifier.PANELS}
    nan[verifier.PANELS[0]]["mean_union_true_edge_recall"] = float("nan")
    with pytest.raises(verifier.VerificationError, match="finite"):
        verifier._independent_gate(nan)


def test_numeric_report_comparison_rejects_extra_missing_and_nan() -> None:
    expected = {"a": 1.0, "b": {"x": True}}
    verifier._compare_json_numeric(expected, expected, label="valid")
    with pytest.raises(verifier.VerificationError, match="schema drift"):
        verifier._compare_json_numeric({"a": 1.0}, expected, label="missing")
    with pytest.raises(verifier.VerificationError, match="schema drift"):
        verifier._compare_json_numeric(
            {"a": 1.0, "b": {"x": True}, "extra": 0}, expected, label="extra"
        )
    with pytest.raises(verifier.VerificationError, match="finite"):
        verifier._compare_json_numeric(
            {"a": float("nan"), "b": {"x": True}}, expected, label="nan"
        )


def test_npy_layout_rejects_trailing_and_invalid_permutation() -> None:
    layout = np.arange(verifier.TILE_COUNT, dtype=np.int32)
    output = BytesIO()
    np.save(output, layout, allow_pickle=False)
    payload = output.getvalue()
    assert np.array_equal(
        verifier._load_npy_int32_layout(payload, label="valid"), layout
    )
    with pytest.raises(verifier.VerificationError, match="trailing"):
        verifier._load_npy_int32_layout(payload + b"x", label="trailing")
    invalid = layout.copy()
    invalid[-1] = 0
    output = BytesIO()
    np.save(output, invalid, allow_pickle=False)
    with pytest.raises(verifier.VerificationError, match="invalid layout"):
        verifier._load_npy_int32_layout(output.getvalue(), label="invalid")


def test_synthetic_hmac_shuffle_panel_truth_and_input_recomposition() -> None:
    secret = bytes(range(32))
    source = "img_000001.png"
    panel = "primary_kornia"
    master_seed = 20260711
    clean = np.zeros((480, 480, 3), dtype=np.uint8)
    panel_seed = verifier._per_source_seed(
        master_seed, f"candidate-graph-oracle-{panel}", source
    )
    exact = verifier.make_exact_panel(clean, panel=panel, seed=panel_seed)
    opaque_id = verifier.hmac.new(
        secret, f"id:{source}:{panel}".encode(), verifier.hashlib.sha256
    ).digest()[:16].hex()
    shuffle_seed = int.from_bytes(
        verifier.hmac.new(
            secret, f"shuffle:{source}:{panel}".encode(), verifier.hashlib.sha256
        ).digest()[:8],
        "big",
    )
    permutation = (
        np.random.Generator(np.random.PCG64(shuffle_seed))
        .permutation(verifier.TILE_COUNT)
        .astype(np.int32)
    )
    opaque_tiles = np.ascontiguousarray(np.asarray(exact.slot_tiles)[permutation])
    composed = np.asarray(exact.slot_to_target)[permutation].astype(np.int32)
    input_record = verifier.InputRecord(
        {"artifact": {"sha256": "0" * 64}},
        {
            "slot_tiles": opaque_tiles,
            "qap_seed": np.asarray(verifier._opaque_qap_seed(opaque_id), dtype=np.uint64),
        },
    )
    label_record = verifier.LabelRecord(
        {
            "source_name": source,
            "panel": panel,
            "panel_seed": panel_seed,
        },
        {
            "opaque_slot_permutation": permutation,
            "composed_slot_to_target": composed,
            "clean_target_rgb": clean,
        },
    )
    context = SimpleNamespace(
        config={
            "frozen_contract": {"synthetic_corruption": {"master_seed": master_seed}}
        }
    )
    result = verifier.recompose_fixture(
        context,
        opaque_id=opaque_id,
        input_record=input_record,
        label_record=label_record,
        secret=secret,
        source_names=[source],
    )
    assert np.array_equal(result.truth, composed)
    tampered = opaque_tiles.copy()
    tampered[0, 0, 0, 0] ^= np.uint8(1)
    bad_input = verifier.InputRecord(
        input_record.manifest,
        {**input_record.arrays, "slot_tiles": tampered},
    )
    with pytest.raises(verifier.VerificationError, match="input bytes mismatch"):
        verifier.recompose_fixture(
            context,
            opaque_id=opaque_id,
            input_record=bad_input,
            label_record=label_record,
            secret=secret,
            source_names=[source],
        )


def test_phase_b_uses_single_opaque_bundle_and_does_not_touch_label_pin_pre_marker(
    tmp_path: Path,
) -> None:
    class GuardPins(dict[str, str]):
        def __getitem__(self, key: str) -> str:
            if key.startswith("fixture_label") or key.startswith("fixture_lock"):
                raise AssertionError("label/control pin touched before marker")
            return super().__getitem__(key)

    pre_marker_context = SimpleNamespace(
        config={
            "runtime_pins": GuardPins(
                {
                    "fixture_input_manifest_relative_path": (
                        "fixture_input/fixture_input_manifest.json"
                    )
                }
            )
        }
    )
    bundle = tmp_path / "opaque_bundle"
    assert verifier._bundle_input_root_before_marker(
        pre_marker_context, str(bundle)
    ) == bundle / "fixture_input"

    full_context = SimpleNamespace(
        config={
            "runtime_pins": {
                "fixture_input_manifest_relative_path": (
                    "fixture_input/fixture_input_manifest.json"
                ),
                "fixture_label_manifest_relative_path": (
                    "fixture_label/fixture_label_manifest.json"
                ),
                "fixture_lock_relative_path": "fixture_control/fixture_lock.json",
                "fixture_prep_marker_relative_path": (
                    "fixture_control/FIXTURE_PIXEL_ACCESS_STARTED.json"
                ),
            }
        }
    )
    resolved_bundle, label_root, control_root = verifier._bundle_paths_after_marker(
        full_context, str(bundle)
    )
    assert resolved_bundle == bundle
    assert label_root == bundle / "fixture_label"
    assert control_root == bundle / "fixture_control"

    with pytest.raises(SystemExit):
        verifier.parse_args(
            [
                "--action",
                "phase-b",
                "--config-sha256",
                "0" * 64,
                "--phase-a-dir",
                "phase-a",
                "--phase-a-envelope-sha256",
                "1" * 64,
                "--phase-a-shard-envelope-sha256",
                "2" * 64,
                "--phase-a-shard-envelope-sha256",
                "3" * 64,
                "--fixture-bundle-root",
                "bundle",
                "--labels-root",
                "forbidden",
            ]
        )


def _write_canonical(path: Path, value: object) -> str:
    payload = verifier._canonical_file_bytes(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _phase_a_attestation_files(
    tmp_path: Path,
    context: verifier.ProtocolContext,
    *,
    raw_ref: str | None = None,
) -> tuple[SimpleNamespace, verifier.InputEvidence, Path, str, Path, str]:
    metadata = json.loads(
        (
            context.repository
            / context.config["runtime_pins"]["phase_a_kernel_metadata_path"]
        ).read_text(encoding="utf-8")
    )
    launch = metadata["oracle_launch_expectation"]
    pins = context.config["runtime_pins"]
    phase_a = SimpleNamespace(
        envelope_sha256="e" * 64,
        shard_anchors=("b" * 64, "c" * 64),
        root_path=tmp_path / "finalized",
        kaggle_attestation=None,
    )
    input_evidence = verifier.InputEvidence(
        tmp_path / "bundle/fixture_input",
        {},
        "1" * 64,
        {},
    )
    environment_lock = json.loads(
        (
            context.repository
            / pins["environment_lock_path"]
        ).read_text(encoding="utf-8")
    )["kaggle_phase_a"]
    packages = environment_lock["packages"]
    hardware = {
        "python": environment_lock["python"],
        "torch": packages["torch"],
        "cuda_runtime": environment_lock["cuda_runtime"],
        "numpy": packages["numpy"],
        "scipy": packages["scipy"],
        "scikit_image": packages["scikit_image"],
        "pillow": packages["pillow"],
        "opencv": packages["opencv"],
        "kornia": packages["kornia"],
        "devices": [
            {**device, "tensor_probe": float(index + 1)}
            for index, device in enumerate(environment_lock["devices"])
        ],
    }
    wrapper = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_phase_a_kaggle_wrapper",
        "status": "phase_a_complete_pending_local_verification",
        "safe_for_submission": False,
        "kernel_slug": launch["kernel_slug"],
        "config_sha256": context.config_sha256,
        "runner_sha256": pins["phase_a_runner_sha256"],
        "kernel_metadata_sha256": pins["phase_a_kernel_metadata_sha256"],
        "launch_expectation": launch,
        "evaluator_sha256": pins["evaluator_sha256"],
        "tests_sha256": pins["tests_sha256"],
        "fixture_builder_tests_sha256": pins["fixture_builder_tests_sha256"],
        "environment_lock_sha256": pins["environment_lock_sha256"],
        "input_manifest_sha256": input_evidence.manifest_sha256,
        "runtime_assets": {
            "denoiser_sha256": context.config["frozen_contract"]["assets"]["denoiser"]["sha256"],
            "hbt_sha256": context.config["frozen_contract"]["assets"]["hbt"]["sha256"],
        },
        "dataset_mounts": {
            label: {
                **descriptor,
                "path": f"/kaggle/input/{descriptor['slug'].split('/')[-1]}",
            }
            for label, descriptor in launch["dataset_versions"].items()
        },
        "exact_code_mount_sha256": verifier._expected_phase_a_code_mount(context),
        "hardware": hardware,
        "shards": [
            {"rank": 0, "manifest_sha256": phase_a.shard_anchors[0]},
            {"rank": 1, "manifest_sha256": phase_a.shard_anchors[1]},
        ],
        "finalized_phase_a_manifest": f"finalized/{verifier.PHASE_A_MANIFEST}",
        "finalized_phase_a_manifest_sha256": phase_a.envelope_sha256,
        "seconds": 12.5,
    }
    wrapper_path = tmp_path / "phase_a_wrapper.json"
    wrapper_sha = _write_canonical(wrapper_path, wrapper)
    ready_datasets = {
        label: {**descriptor, "status": "ready"}
        for label, descriptor in launch["dataset_versions"].items()
    }

    def readback(
        *, version: int, source_sha256: str, dataset_sources: list[str]
    ) -> dict:
        normalized = {
            "id": metadata["id_no"],
            "ref": metadata["id"],
            "title": metadata["title"],
            "slug": metadata["id"].split("/", 1)[1],
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu_observation": False,
            "enable_internet": False,
            "enable_tpu_observation": False,
            "dataset_sources": dataset_sources,
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
            "current_version_number": version,
            "docker_image": "gcr.io/kaggle-private-byod/python@sha256:test",
            "machine_shape_observation": "None",
        }
        return {
            "access_mode": "unversioned_private_get_kernel",
            "version_qualified_pull_used": False,
            "metadata": normalized,
            "metadata_sha256": hashlib.sha256(
                verifier._canonical_object_bytes(normalized)
            ).hexdigest(),
            "source_sha256": source_sha256,
        }

    intent = {
        "schema_version": 2,
        "kind": "candidate_graph_oracle_kaggle_launch_intent",
        "created_utc": "2026-07-12T16:59:58Z",
        "protocol_instance_id": verifier.EXPECTED_PROTOCOL_INSTANCE_ID,
        "kernel": {
            "slug": launch["kernel_slug"],
            "kernel_id": launch["kernel_id"],
            "reserved_version": 1,
            "intended_version": 2,
        },
        "dataset_versions": ready_datasets,
        "local_kernel_metadata_sha256": pins["phase_a_kernel_metadata_sha256"],
        "local_runner_sha256": pins["phase_a_runner_sha256"],
        "local_launcher_sha256": pins["phase_a_launcher_sha256"],
        "reservation_receipt_sha256": launch["reservation_receipt_sha256"],
        "reservation_readback": readback(
            version=1,
            source_sha256=verifier.EXPECTED_RESERVATION_RUNNER_SHA256,
            dataset_sources=[],
        ),
        "safe_for_submission": False,
    }
    raw_push_response = {
        "schema_version": 1,
        "kind": "candidate_graph_oracle_kaggle_raw_push_response",
        "recorded_utc": "2026-07-12T16:59:59Z",
        "response_type": {
            "module": "kagglesdk.kernels.types.kernels_api_service",
            "qualname": "ApiSaveKernelResponse",
        },
        "public_fields": {
            "ref": raw_ref or launch["kernel_slug"],
            "url": "https://www.kaggle.com/code/pasha883/oracle",
            "version_number": launch["kernel_version"],
            "error": None,
            "invalid_tags": [],
            "invalid_dataset_sources": [],
            "invalid_competition_sources": [],
            "invalid_kernel_sources": [],
            "invalid_model_sources": [],
            "kernel_id": launch["kernel_id"],
        },
        "object_state": {
            "_ref": raw_ref or launch["kernel_slug"],
            "_url": "https://www.kaggle.com/code/pasha883/oracle",
            "_version_number": launch["kernel_version"],
            "_error": None,
            "_invalid_tags": [],
            "_invalid_dataset_sources": [],
            "_invalid_competition_sources": [],
            "_invalid_kernel_sources": [],
            "_invalid_model_sources": [],
            "_kernel_id": launch["kernel_id"],
        },
    }
    raw_push_response_sha256 = hashlib.sha256(
        verifier._canonical_file_bytes(raw_push_response)
    ).hexdigest()
    push_response = launcher._push_response_payload(
        raw_push_response,
        raw_response_sha256=raw_push_response_sha256,
    )
    receipt_payload = {
        "schema_version": 2,
        "kind": "candidate_graph_oracle_kaggle_launch_receipt",
        "created_utc": "2026-07-12T17:00:00Z",
        "protocol_instance_id": verifier.EXPECTED_PROTOCOL_INSTANCE_ID,
        "kernel": {
            "slug": launch["kernel_slug"],
            "kernel_id": launch["kernel_id"],
            "version": launch["kernel_version"],
            "url": "https://www.kaggle.com/code/pasha883/oracle",
        },
        "dataset_versions_before_push": ready_datasets,
        "dataset_versions_after_push": ready_datasets,
        "local_kernel_metadata_sha256": pins["phase_a_kernel_metadata_sha256"],
        "local_runner_sha256": pins["phase_a_runner_sha256"],
        "local_launcher_sha256": pins["phase_a_launcher_sha256"],
        "launch_journal": {
            "intent_file": "00_launch.intent.json",
            "intent_sha256": hashlib.sha256(
                verifier._canonical_file_bytes(intent)
            ).hexdigest(),
            "raw_push_response_file": "01_push.raw_response.json",
            "raw_push_response_sha256": raw_push_response_sha256,
            "push_response_file": "02_push.response.json",
            "push_response_sha256": hashlib.sha256(
                verifier._canonical_file_bytes(push_response)
            ).hexdigest(),
        },
        "launch_intent": intent,
        "raw_push_response": raw_push_response,
        "push_response": push_response,
        "server_readback": readback(
            version=2,
            source_sha256=pins["phase_a_runner_sha256"],
            dataset_sources=[
                launch["dataset_versions"][label]["slug"]
                for label in ("code", "input", "runtime")
            ],
        ),
        "gpu_and_machine_metadata_authority": (
            "executed_phase_a_wrapper_hardware_not_normalized_get_kernel_metadata"
        ),
        "push_performed_in_this_process": True,
        "push_response_recovered_from_raw_journal": False,
        "safe_for_submission": False,
    }
    receipt_envelope = {
        "payload": receipt_payload,
        "payload_sha256": hashlib.sha256(
            verifier._canonical_object_bytes(receipt_payload)
        ).hexdigest(),
    }
    receipt_path = tmp_path / "launch_receipt.json"
    receipt_sha = _write_canonical(receipt_path, receipt_envelope)
    return (
        phase_a,
        input_evidence,
        wrapper_path,
        wrapper_sha,
        receipt_path,
        receipt_sha,
    )


def test_phase_a_kaggle_wrapper_and_launch_receipt_are_exactly_crosslinked(
    tmp_path: Path,
) -> None:
    context = _attestation_context()
    (
        phase_a,
        input_evidence,
        wrapper_path,
        wrapper_sha,
        receipt_path,
        receipt_sha,
    ) = _phase_a_attestation_files(tmp_path, context)
    result = verifier.verify_phase_a_kaggle_attestation(
        context,
        phase_a=phase_a,
        input_evidence=input_evidence,
        wrapper_path=wrapper_path,
        expected_wrapper_sha256=wrapper_sha,
        launch_receipt_path=receipt_path,
        expected_launch_receipt_sha256=receipt_sha,
    )
    assert result.wrapper_sha256 == wrapper_sha
    assert result.launch_receipt_sha256 == receipt_sha

    envelope = json.loads(receipt_path.read_text(encoding="utf-8"))
    alias = copy.deepcopy(envelope)
    alias_raw = alias["payload"]["raw_push_response"]
    alias_raw["public_fields"]["ref"] = (
        f"/code/{alias['payload']['push_response']['ref']}"
    )
    alias_raw["object_state"]["_ref"] = alias_raw["public_fields"]["ref"]
    alias_raw_sha = hashlib.sha256(
        verifier._canonical_file_bytes(alias_raw)
    ).hexdigest()
    alias["payload"]["launch_journal"]["raw_push_response_sha256"] = alias_raw_sha
    alias["payload"]["push_response"]["raw_response_sha256"] = alias_raw_sha
    alias["payload"]["launch_journal"]["push_response_sha256"] = hashlib.sha256(
        verifier._canonical_file_bytes(alias["payload"]["push_response"])
    ).hexdigest()
    alias["payload_sha256"] = hashlib.sha256(
        verifier._canonical_object_bytes(alias["payload"])
    ).hexdigest()
    alias_sha = _write_canonical(receipt_path, alias)
    alias_result = verifier.verify_phase_a_kaggle_attestation(
        context,
        phase_a=phase_a,
        input_evidence=input_evidence,
        wrapper_path=wrapper_path,
        expected_wrapper_sha256=wrapper_sha,
        launch_receipt_path=receipt_path,
        expected_launch_receipt_sha256=alias_sha,
    )
    assert alias_result.launch_receipt["push_response"]["ref"] == (
        alias_result.launch_receipt["kernel"]["slug"]
    )
    assert alias_result.launch_receipt["raw_push_response"]["public_fields"][
        "ref"
    ].startswith("/code/")

    near_alias = copy.deepcopy(alias)
    near_alias["payload"]["raw_push_response"]["public_fields"]["ref"] += "/"
    near_alias_raw_sha = hashlib.sha256(
        verifier._canonical_file_bytes(near_alias["payload"]["raw_push_response"])
    ).hexdigest()
    near_alias["payload"]["launch_journal"][
        "raw_push_response_sha256"
    ] = near_alias_raw_sha
    near_alias["payload"]["push_response"]["raw_response_sha256"] = near_alias_raw_sha
    near_alias["payload"]["launch_journal"]["push_response_sha256"] = hashlib.sha256(
        verifier._canonical_file_bytes(near_alias["payload"]["push_response"])
    ).hexdigest()
    near_alias["payload_sha256"] = hashlib.sha256(
        verifier._canonical_object_bytes(near_alias["payload"])
    ).hexdigest()
    near_alias_sha = _write_canonical(receipt_path, near_alias)
    with pytest.raises(verifier.VerificationError, match="drift: ref"):
        verifier.verify_phase_a_kaggle_attestation(
            context,
            phase_a=phase_a,
            input_evidence=input_evidence,
            wrapper_path=wrapper_path,
            expected_wrapper_sha256=wrapper_sha,
            launch_receipt_path=receipt_path,
            expected_launch_receipt_sha256=near_alias_sha,
        )

    bad = copy.deepcopy(envelope)
    bad["payload"]["dataset_versions_after_push"]["input"]["version"] = 3
    bad["payload_sha256"] = hashlib.sha256(
        verifier._canonical_object_bytes(bad["payload"])
    ).hexdigest()
    bad_sha = _write_canonical(receipt_path, bad)
    with pytest.raises(verifier.VerificationError, match="dataset readback drift"):
        verifier.verify_phase_a_kaggle_attestation(
            context,
            phase_a=phase_a,
            input_evidence=input_evidence,
            wrapper_path=wrapper_path,
            expected_wrapper_sha256=wrapper_sha,
            launch_receipt_path=receipt_path,
            expected_launch_receipt_sha256=bad_sha,
        )

    bad_version = copy.deepcopy(envelope)
    current = bad_version["payload"]["server_readback"]
    current["metadata"]["current_version_number"] = 3
    current["metadata_sha256"] = hashlib.sha256(
        verifier._canonical_object_bytes(current["metadata"])
    ).hexdigest()
    bad_version["payload_sha256"] = hashlib.sha256(
        verifier._canonical_object_bytes(bad_version["payload"])
    ).hexdigest()
    bad_version_sha = _write_canonical(receipt_path, bad_version)
    with pytest.raises(verifier.VerificationError, match="normalized metadata drift"):
        verifier.verify_phase_a_kaggle_attestation(
            context,
            phase_a=phase_a,
            input_evidence=input_evidence,
            wrapper_path=wrapper_path,
            expected_wrapper_sha256=wrapper_sha,
            launch_receipt_path=receipt_path,
            expected_launch_receipt_sha256=bad_version_sha,
        )

    bad_sources = copy.deepcopy(envelope)
    current = bad_sources["payload"]["server_readback"]
    current["metadata"]["dataset_sources"] = metadata_sources = [
        f"{item}/2" for item in current["metadata"]["dataset_sources"]
    ]
    assert all(value.endswith("/2") for value in metadata_sources)
    current["metadata_sha256"] = hashlib.sha256(
        verifier._canonical_object_bytes(current["metadata"])
    ).hexdigest()
    bad_sources["payload_sha256"] = hashlib.sha256(
        verifier._canonical_object_bytes(bad_sources["payload"])
    ).hexdigest()
    bad_sources_sha = _write_canonical(receipt_path, bad_sources)
    with pytest.raises(verifier.VerificationError, match="normalized metadata drift"):
        verifier.verify_phase_a_kaggle_attestation(
            context,
            phase_a=phase_a,
            input_evidence=input_evidence,
            wrapper_path=wrapper_path,
            expected_wrapper_sha256=wrapper_sha,
            launch_receipt_path=receipt_path,
            expected_launch_receipt_sha256=bad_sources_sha,
        )

    bad_response = copy.deepcopy(envelope)
    bad_response["payload"]["push_response"]["kernel_id"] += 1
    bad_response["payload"]["launch_journal"]["push_response_sha256"] = (
        hashlib.sha256(
            verifier._canonical_file_bytes(
                bad_response["payload"]["push_response"]
            )
        ).hexdigest()
    )
    bad_response["payload_sha256"] = hashlib.sha256(
        verifier._canonical_object_bytes(bad_response["payload"])
    ).hexdigest()
    bad_response_sha = _write_canonical(receipt_path, bad_response)
    with pytest.raises(verifier.VerificationError, match="push response binding drift"):
        verifier.verify_phase_a_kaggle_attestation(
            context,
            phase_a=phase_a,
            input_evidence=input_evidence,
            wrapper_path=wrapper_path,
            expected_wrapper_sha256=wrapper_sha,
            launch_receipt_path=receipt_path,
            expected_launch_receipt_sha256=bad_response_sha,
        )

    bad_raw = copy.deepcopy(envelope)
    bad_raw["payload"]["raw_push_response"]["public_fields"]["kernel_id"] += 1
    raw_sha = hashlib.sha256(
        verifier._canonical_file_bytes(bad_raw["payload"]["raw_push_response"])
    ).hexdigest()
    bad_raw["payload"]["launch_journal"]["raw_push_response_sha256"] = raw_sha
    bad_raw["payload"]["push_response"]["raw_response_sha256"] = raw_sha
    bad_raw["payload"]["launch_journal"]["push_response_sha256"] = hashlib.sha256(
        verifier._canonical_file_bytes(bad_raw["payload"]["push_response"])
    ).hexdigest()
    bad_raw["payload_sha256"] = hashlib.sha256(
        verifier._canonical_object_bytes(bad_raw["payload"])
    ).hexdigest()
    bad_raw_sha = _write_canonical(receipt_path, bad_raw)
    with pytest.raises(verifier.VerificationError, match="raw/validated"):
        verifier.verify_phase_a_kaggle_attestation(
            context,
            phase_a=phase_a,
            input_evidence=input_evidence,
            wrapper_path=wrapper_path,
            expected_wrapper_sha256=wrapper_sha,
            launch_receipt_path=receipt_path,
            expected_launch_receipt_sha256=bad_raw_sha,
        )


def test_launcher_exact_alias_projection_is_accepted_by_verifier(
    tmp_path: Path,
) -> None:
    context = _attestation_context()
    launch = json.loads(
        (
            context.repository
            / context.config["runtime_pins"]["phase_a_kernel_metadata_path"]
        ).read_text(encoding="utf-8")
    )["oracle_launch_expectation"]
    (
        phase_a,
        input_evidence,
        wrapper_path,
        wrapper_sha,
        receipt_path,
        receipt_sha,
    ) = _phase_a_attestation_files(
        tmp_path,
        context,
        raw_ref=f"/code/{launch['kernel_slug']}",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))["payload"]
    assert receipt["raw_push_response"]["public_fields"]["ref"] == (
        f"/code/{launch['kernel_slug']}"
    )
    assert receipt["push_response"]["ref"] == launch["kernel_slug"]
    result = verifier.verify_phase_a_kaggle_attestation(
        context,
        phase_a=phase_a,
        input_evidence=input_evidence,
        wrapper_path=wrapper_path,
        expected_wrapper_sha256=wrapper_sha,
        launch_receipt_path=receipt_path,
        expected_launch_receipt_sha256=receipt_sha,
    )
    assert result.launch_receipt["kind"] == (
        "candidate_graph_oracle_kaggle_launch_receipt"
    )
    assert result.launch_receipt["push_response"]["ref"] == launch["kernel_slug"]


@pytest.mark.skipif(
    launcher.EXPECTED_KERNEL_ID < 0,
    reason="requires the externally reserved positive v4 Kaggle kernel id",
)
def test_launcher_push_and_record_exact_alias_receipt_feeds_verifier(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    job = repository / "runs/assembly_v1/kaggle/candidate_graph_oracle_v4_phase_a_job"
    metadata = json.loads((job / "kernel-metadata.json").read_text(encoding="utf-8"))
    runner_sha = hashlib.sha256((job / metadata["code_file"]).read_bytes()).hexdigest()

    class AliasApi:
        def __init__(self) -> None:
            self.version = 1
            self.push_calls = 0

        def dataset_status(self, dataset: str, format: str) -> str:
            assert dataset in launcher.EXPECTED_DATASETS.values()
            assert format == "json(status,current_version_number)"
            return json.dumps({"status": "ready", "current_version_number": 2})

        def candidate_graph_oracle_current_readback(self):
            if self.version == 1:
                source_sha = launcher.RESERVATION_RUNNER_SHA256
                dataset_sources: list[str] = []
            else:
                source_sha = runner_sha
                dataset_sources = list(launcher.EXPECTED_DATASETS.values())
            return (
                {
                    "id": launcher.EXPECTED_KERNEL_ID,
                    "ref": launcher.EXPECTED_KERNEL_SLUG,
                    "title": metadata["title"],
                    "slug": launcher.EXPECTED_KERNEL_SLUG.split("/", 1)[1],
                    "language": "python",
                    "kernel_type": "script",
                    "is_private": True,
                    "enable_gpu_observation": False,
                    "enable_internet": False,
                    "enable_tpu_observation": False,
                    "dataset_sources": dataset_sources,
                    "kernel_sources": [],
                    "competition_sources": [],
                    "model_sources": [],
                    "current_version_number": self.version,
                    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:test",
                    "machine_shape_observation": None,
                },
                source_sha,
            )

        def kernels_push(self, folder: str, timeout, acc):
            assert Path(folder) == job
            self.push_calls += 1
            self.version = 2
            return SimpleNamespace(
                error="",
                invalid_tags=[],
                invalid_dataset_sources=[],
                invalid_competition_sources=[],
                invalid_kernel_sources=[],
                invalid_model_sources=[],
                kernel_id=launcher.EXPECTED_KERNEL_ID,
                version_number=2,
                ref=f"/code/{launcher.EXPECTED_KERNEL_SLUG}",
                url="https://www.kaggle.com/code/pasha883/oracle",
            )

    api = AliasApi()
    receipt_path = tmp_path / "actual_launcher_receipt.json"
    launcher.push_and_record(
        job_dir=job,
        receipt_path=receipt_path,
        state_dir=tmp_path / "actual_launcher_state",
        api=api,
    )
    assert api.push_calls == 1
    context = _attestation_context()
    pins = context.config["runtime_pins"]
    pins["phase_a_launcher_path"] = "scripts/push_candidate_graph_oracle_v4_phase_a.py"
    pins["phase_a_launcher_sha256"] = hashlib.sha256(
        (repository / pins["phase_a_launcher_path"]).read_bytes()
    ).hexdigest()
    fixture_root = tmp_path / "attestation_fixture"
    fixture_root.mkdir()
    (
        phase_a,
        input_evidence,
        wrapper_path,
        wrapper_sha,
        _,
        _,
    ) = _phase_a_attestation_files(fixture_root, context)
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    result = verifier.verify_phase_a_kaggle_attestation(
        context,
        phase_a=phase_a,
        input_evidence=input_evidence,
        wrapper_path=wrapper_path,
        expected_wrapper_sha256=wrapper_sha,
        launch_receipt_path=receipt_path,
        expected_launch_receipt_sha256=receipt_sha,
    )
    assert result.launch_receipt["raw_push_response"]["public_fields"]["ref"] == (
        f"/code/{launcher.EXPECTED_KERNEL_SLUG}"
    )
    assert result.launch_receipt["push_response"]["ref"] == (
        launcher.EXPECTED_KERNEL_SLUG
    )
    assert result.launch_receipt["kind"] == (
        "candidate_graph_oracle_kaggle_launch_receipt"
    )


def test_phase_b_stdout_attestation_binds_report_sandbox_and_roots(
    tmp_path: Path,
) -> None:
    context = _attestation_context()
    phase_a = SimpleNamespace(
        envelope_sha256="e" * 64,
        root_path=(tmp_path / "phase_a").absolute(),
    )
    phase_a.root_path.mkdir()
    fixture_bundle = (tmp_path / "bundle").absolute()
    input_root = fixture_bundle / "fixture_input"
    input_root.mkdir(parents=True)
    input_evidence = verifier.InputEvidence(input_root, {}, "1" * 64, {})
    lifecycle_root = (tmp_path / "lifecycle").absolute()
    lifecycle_root.mkdir()
    lifecycle = SimpleNamespace(root_path=lifecycle_root)
    output_root = (tmp_path / "phase_b").absolute()
    output_root.mkdir()
    report_payload = {"kind": "candidate_graph_oracle_ceiling_report", "status": "stop_or_pivot"}
    report_envelope = {
        "payload": report_payload,
        "payload_sha256": hashlib.sha256(
            verifier._canonical_object_bytes(report_payload)
        ).hexdigest(),
    }
    report_path = output_root / verifier.REPORT_NAME
    report_sha = _write_canonical(report_path, report_envelope)
    verification = verifier.PhaseBVerification(report_sha, "stop_or_pivot", False)
    pins = context.config["runtime_pins"]
    local_lock = json.loads(
        (context.repository / pins["environment_lock_path"]).read_text(encoding="utf-8")
    )["fixture_preparation_and_phase_b"]
    payload = {
        "schema_version": 2,
        "kind": "candidate_graph_oracle_phase_b_runner_attestation",
        "status": verification.status,
        "safe_for_submission": False,
        "process_id": 123,
        "config_sha256": context.config_sha256,
        "runner_sha256": pins["phase_b_runner_sha256"],
        "evaluator_sha256": pins["evaluator_sha256"],
        "tests_sha256": pins["tests_sha256"],
        "environment": {
            "lock_sha256": pins["environment_lock_sha256"],
            "platform": local_lock["platform"],
            "python": local_lock["python"],
            "packages": local_lock["packages"],
        },
        "sandbox": {
            "backend": "/usr/bin/sandbox-exec",
            "profile_sha256": "f" * 64,
            "default_deny": True,
            "network_policy": "deny network*",
            "config_readable_and_sha256_verified": True,
            "fresh_output_write_probe": True,
            "denial_probes": [
                {"label": label, "denied": True, "errno": 1, "errno_name": "EPERM"}
                for label in (
                    "repo_puzzle_train_read",
                    "repo_puzzle_train_targets_read",
                    "phase_a_write",
                )
            ]
            + [
                {
                    "label": "network_outbound",
                    "denied": True,
                    "errno": 1,
                    "errno_name": "EPERM",
                    "denied_at": "connect",
                }
            ],
        },
        "phase_a_envelope_sha256": phase_a.envelope_sha256,
        "fixture_input_manifest_sha256": input_evidence.manifest_sha256,
        "filesystem_bindings": {
            "phase_a_root": str(phase_a.root_path),
            "phase_a_artifact_envelope_sha256": phase_a.envelope_sha256,
            "fixture_bundle_root": str(fixture_bundle),
            "fixture_input_root": str(input_root),
            "fixture_input_manifest": str(input_root / verifier.INPUT_MANIFEST),
            "fixture_input_manifest_sha256": input_evidence.manifest_sha256,
            "lifecycle_ledger_root": str(lifecycle_root),
            "output_root": str(output_root),
        },
        "report_path": verifier.REPORT_NAME,
        "report_sha256": report_sha,
        "report_payload_sha256": report_envelope["payload_sha256"],
        "preflight_output_sha256": "4" * 64,
        "preflight_output_bytes": 100,
        "evaluator_output_sha256": "5" * 64,
        "evaluator_output_bytes": 200,
        "evaluator_output_tree_mutated_by_runner": False,
    }
    envelope = {
        "payload": payload,
        "payload_sha256": hashlib.sha256(
            verifier._canonical_object_bytes(payload)
        ).hexdigest(),
    }
    attestation_path = tmp_path / "phase_b_attestation.json"
    attestation_sha = _write_canonical(attestation_path, envelope)
    result = verifier.verify_phase_b_runner_attestation(
        context,
        attestation_path=attestation_path,
        expected_attestation_sha256=attestation_sha,
        verification=verification,
        phase_a=phase_a,
        input_evidence=input_evidence,
        lifecycle=lifecycle,
        phase_b_root=output_root,
        fixture_bundle_root=fixture_bundle,
    )
    assert result.sha256 == attestation_sha

    bad = copy.deepcopy(envelope)
    bad["payload"]["sandbox"]["default_deny"] = False
    bad["payload_sha256"] = hashlib.sha256(
        verifier._canonical_object_bytes(bad["payload"])
    ).hexdigest()
    bad_sha = _write_canonical(attestation_path, bad)
    with pytest.raises(verifier.VerificationError, match="sandbox policy"):
        verifier.verify_phase_b_runner_attestation(
            context,
            attestation_path=attestation_path,
            expected_attestation_sha256=bad_sha,
            verification=verification,
            phase_a=phase_a,
            input_evidence=input_evidence,
            lifecycle=lifecycle,
            phase_b_root=output_root,
            fixture_bundle_root=fixture_bundle,
        )
