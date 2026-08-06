"""V3-protocol core for clean E11 v4 artifacts and the CPU selector.

This file contains every E11 experiment, isolation, evaluation, and report
contract that must be hashed into gate v3.  It deliberately contains no pinned
gate root: the tiny launcher passes that trust anchor after the gate has been
created, avoiding a self-hash cycle.  Preflight verifies the complete immutable
gate/cache contract and source-disjointness from gate v1/v2 without solving or
restoring.  Evaluation compares exactly two outputs:

* the current Rank96 baseline;
* E11, which selects Rank96 or Rank512 by fixed upright CIE-Lab depth-1 seam
  continuity and then applies the same fixed OpenCV NLM(10) restoration.

There are no selector thresholds, solver knobs, device controls, sweeps, or
orientation controls.  The immutable keep rule is:

``mean(final E11 - final Rank96) > 0.001`` and
``mean(solve E11 - solve Rank96) >= 0``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import eval_frozen_end_to_end_gate as frozen
from build_source_groups_v3 import (
    ALGORITHMS_CONTRACT as SOURCE_ALGORITHMS_CONTRACT,
    BUILDER_CONTRACT as SOURCE_BUILDER_CONTRACT,
    V3_SELECTION_CONTRACT,
    validate_manifest_v3,
)
from freeze_rank96_lab_selector_v3 import (
    REQUIRED_CODE_ROLES,
    e11_code_registry,
)
from rank96_lab_selector import (
    DEPTH,
    MIN_MARGIN,
    RANK512_MAX_EDGES,
    RANK96_MAX_EDGES,
    REPAIR_PASSES,
    RANK96_ARM,
    solve_and_select_lab_depth1,
)


EXPECTED_GATE_SEED = 20_260_808
EXPECTED_SCENE_COUNT = 48
EXPECTED_PRIOR_SCENE_COUNT = 48
EXPECTED_VALIDATION_COUNT = 300
EXPECTED_CANDIDATE_VAL_MIN = 100
EXPECTED_KNOWN_TUNE_VAL_IDS = [0, 99]
FIXED_ORIENTATION = "fixed_type1_no_rotation"
REPORT_SCHEMA = "pazzle-rank96-lab-selector-gate-v3-report-v1"
CACHE_PREPARATION_STARTED_SCHEMA = "pazzle-rank96-e11-score-cache-prepare-started-v1"
CACHE_PREPARATION_COMPLETE_SCHEMA = "pazzle-rank96-e11-score-cache-prepare-complete-v1"
EVALUATION_STARTED_SCHEMA = "pazzle-rank96-e11-evaluation-started-v1"
FIXED_SCORE_CACHE_DEVICE = "cuda:0"
ARM_ORDER = ("rank96", "selector")
PAIR_BOOTSTRAP_SEED = 96_512_20_260_808
KEEP_FINAL_DELTA_STRICTLY_GREATER_THAN = 0.001
KEEP_SOLVE_DELTA_GREATER_THAN_OR_EQUAL_TO = 0.0
EXPECTED_TARGET_CORPUS_SHA256 = "64b171c929c73e2f812e9d1c33d63935c3400e1470be51fc1ee9f47ab1ffd9c9"
PRIOR_REPORT_CONTRACTS: dict[str, dict[str, Any]] = {
    "gate_v1": {
        "path_key": "prior_report_v1",
        "filename": "report_budget96_vs512_v1.json",
        "schema": "pazzle-frozen-budget96-vs512-report-v1",
        "sha256": "2ea813849d45562d2e5af77ac73fdb1258a2b900dbc6290b645abf12b3810db6",
        "gate_root_sha256": "ee3d74662f5326fbd1069763fd7b96dc3adb41bde0117cba1d78ff067c6bf23d",
        "gate_sha256sums_sha256": "ee3d74662f5326fbd1069763fd7b96dc3adb41bde0117cba1d78ff067c6bf23d",
    },
    "gate_v2": {
        "path_key": "prior_report_v2",
        "filename": "report_budget96_vs512_v2.json",
        "schema": "pazzle-frozen-budget96-vs512-report-v2",
        "sha256": "6c0d8ecf07b505d85bf7a831d5b31a22e3ccbdf3637ca42fd41ee359a8fc92dc",
        "gate_root_sha256": "7a5a5e68779a25fd8dc882062345a3e7b5e9e555da51dee97c5b5ca3e3558134",
        "gate_sha256sums_sha256": "7a5a5e68779a25fd8dc882062345a3e7b5e9e555da51dee97c5b5ca3e3558134",
    },
}
E11_ARTIFACT_ROOT = Path("E:/pazzle_work/rank96_e11_v4")
REQUIRED_E11_CODE_ROLES = frozenset(REQUIRED_CODE_ROLES)

FIXED_RESTORATION = {
    "method": "opencv_fast_nlm_colored",
    "h": 10,
    "h_color": 10,
    "template_window": 7,
    "search_window": 21,
}
FIXED_ARMS: dict[str, dict[str, Any]] = {
    "rank96": {
        "candidate_source": "frozen_candidate_ranker_raw_logits",
        "dense_conversion": "eval_seeded_qap.dense_rd_cpu_float32",
        "solver": "solve_buddies.solve_buddies_from_scores",
        "max_edges": RANK96_MAX_EDGES,
        "min_margin": MIN_MARGIN,
        "repair_passes": REPAIR_PASSES,
        "orientation": FIXED_ORIENTATION,
        "restoration": FIXED_RESTORATION,
    },
    "selector": {
        "candidate_source": "same_frozen_candidate_ranker_raw_logits",
        "dense_conversion": "same_dense_right_down_as_rank96",
        "candidate_solvers": [
            {
                "name": "rank96",
                "max_edges": RANK96_MAX_EDGES,
                "min_margin": MIN_MARGIN,
                "repair_passes": REPAIR_PASSES,
            },
            {
                "name": "rank512",
                "max_edges": RANK512_MAX_EDGES,
                "min_margin": MIN_MARGIN,
                "repair_passes": REPAIR_PASSES,
            },
        ],
        "selection": {
            "feature": "mean_negative_squared_scaled_CIE_Lab_seam_distance",
            "depth": DEPTH,
            "scope": "all_1104_horizontal_and_vertical_upright_board_seams",
            "choose": "larger_score",
            "tie": "rank96",
            "label_free": True,
        },
        "orientation": FIXED_ORIENTATION,
        "restoration": FIXED_RESTORATION,
    },
}


class GateV3NotPinnedError(frozen.IntegrityError):
    """The verifier was invoked before the immutable gate root was pinned."""


def _default_paths() -> dict[str, Path]:
    workspace = Path(__file__).resolve().parent.parent
    prior_root = workspace / "artifacts" / "frozen_gate"
    return {
        "artifact_root": E11_ARTIFACT_ROOT,
        "source_manifest": E11_ARTIFACT_ROOT / "source_groups_v4.json",
        "gate": E11_ARTIFACT_ROOT / "gate_v4",
        "score_cache": E11_ARTIFACT_ROOT / "score_cache_v4",
        "report": E11_ARTIFACT_ROOT / "report_rank96_lab_selector_v4.json",
        "evaluation_started": E11_ARTIFACT_ROOT / "EVALUATION_STARTED.json",
        "prior_report_v1": prior_root / "report_budget96_vs512_v1.json",
        "prior_report_v2": prior_root / "report_budget96_vs512_v2.json",
    }


def _require_pinned_root_configuration(expected_root: str | None) -> str:
    digest = expected_root
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise GateV3NotPinnedError(
            "gate_v4 root is intentionally unpinned; freeze gate_v4, then pin its exact "
            "ROOT_SHA256 in EXPECTED_GATE_ROOT_SHA256 before preflight"
        )
    return digest


def _require_expected_root(root_digest: str, *, expected_root: str | None) -> None:
    expected = _require_pinned_root_configuration(expected_root)
    if root_digest != expected:
        raise frozen.IntegrityError(
            f"this verifier accepts only gate_v4 root {expected}; received {root_digest}"
        )


def _require_exact_paths(gate_dir: Path, score_cache_dir: Path) -> None:
    defaults = _default_paths()
    if gate_dir.resolve() != defaults["gate"].resolve():
        raise frozen.IntegrityError(
            f"this verifier reads only {defaults['gate'].resolve()}; received {gate_dir.resolve()}"
        )
    if score_cache_dir.resolve() != defaults["score_cache"].resolve():
        raise frozen.CacheContractError(
            "this verifier reads only "
            f"{defaults['score_cache'].resolve()}; received {score_cache_dir.resolve()}"
        )


def _source_manifest(gate_dir: Path) -> dict[str, Any]:
    path = gate_dir.resolve() / "source_groups.input.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise frozen.IntegrityError(f"could not read archived v3 source manifest: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise frozen.IntegrityError("gate_v3 requires an archived build-source-groups-v2 manifest")
    return value


def _load_prior_identities() -> tuple[set[str], set[str]]:
    """Load only report-proven prior identities; never touch legacy gate directories."""

    paths = _default_paths()
    names: set[str] = set()
    groups: set[str] = set()
    expected_per_gate = EXPECTED_PRIOR_SCENE_COUNT // len(PRIOR_REPORT_CONTRACTS)
    for gate_name, contract in PRIOR_REPORT_CONTRACTS.items():
        report_path = paths[str(contract["path_key"])].resolve()
        sidecar_path = report_path.with_suffix(report_path.suffix + ".sha256")
        if not report_path.is_file() or not sidecar_path.is_file():
            raise frozen.IntegrityError(f"{gate_name} prior report pair is missing")
        expected_report_sha256 = str(contract["sha256"])
        if frozen.sha256_file(report_path) != expected_report_sha256:
            raise frozen.IntegrityError(f"{gate_name} prior report digest changed")
        expected_sidecar = (
            f"{expected_report_sha256}  {report_path.name}\n".encode("ascii")
        )
        if sidecar_path.read_bytes() != expected_sidecar:
            raise frozen.IntegrityError(f"{gate_name} prior report sidecar differs")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise frozen.IntegrityError(f"{gate_name} prior report is invalid JSON") from exc
        hashes = report.get("hashes") if isinstance(report, dict) else None
        scenes = report.get("per_scene") if isinstance(report, dict) else None
        if (
            not isinstance(report, dict)
            or report.get("schema") != contract["schema"]
            or report.get("gate_root_sha256") != contract["gate_root_sha256"]
            or type(report.get("scene_count")) is not int
            or report.get("scene_count") != expected_per_gate
            or not isinstance(hashes, dict)
            or hashes.get("gate_sha256sums_sha256")
            != contract["gate_sha256sums_sha256"]
            or not isinstance(scenes, list)
            or len(scenes) != expected_per_gate
        ):
            raise frozen.IntegrityError(f"{gate_name} prior report contract differs")
        gate_names: set[str] = set()
        gate_groups: set[str] = set()
        for scene in scenes:
            if not isinstance(scene, dict):
                raise frozen.IntegrityError(f"{gate_name} prior scene row is malformed")
            name = scene.get("name")
            group = scene.get("source_group")
            if (
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or not isinstance(group, str)
                or not group
            ):
                raise frozen.IntegrityError(f"{gate_name} prior identity is malformed")
            gate_names.add(name)
            gate_groups.add(group)
        if len(gate_names) != expected_per_gate or len(gate_groups) != expected_per_gate:
            raise frozen.IntegrityError(
                f"{gate_name} prior report identities are not unique"
            )
        if names.intersection(gate_names) or groups.intersection(gate_groups):
            raise frozen.IntegrityError("gate v1/v2 identities are not source-disjoint")
        names.update(gate_names)
        groups.update(gate_groups)
    if len(names) != EXPECTED_PRIOR_SCENE_COUNT or len(groups) != EXPECTED_PRIOR_SCENE_COUNT:
        raise frozen.IntegrityError(
            f"gate v1/v2 must contribute exactly {EXPECTED_PRIOR_SCENE_COUNT} exclusions"
        )
    return names, groups


def _validate_v3_code_contract(manifest: Mapping[str, Any]) -> None:
    workspace = Path(__file__).resolve().parent.parent
    try:
        expected_paths = e11_code_registry(workspace)
    except frozen.FrozenGateError as exc:
        raise frozen.IntegrityError(f"the E11 code registry is invalid: {exc}") from exc
    if set(expected_paths) != REQUIRED_E11_CODE_ROLES:
        raise frozen.IntegrityError("the freezer E11 code roles differ from the exact registry")
    launcher = Path(__file__).resolve().with_name("eval_rank96_lab_selector_v3.py")
    if any(path.resolve() == launcher for path in expected_paths.values()):
        raise frozen.IntegrityError("the root-pinning launcher must not enter the gate hash cycle")
    records = manifest.get("code")
    if not isinstance(records, dict) or set(records) != set(expected_paths):
        raise frozen.IntegrityError("gate_v3 code roles differ from the exact hashed E11 contract")
    for role, expected_path in expected_paths.items():
        record = records.get(role)
        if not isinstance(record, dict):
            raise frozen.IntegrityError(f"gate_v3 code role {role!r} is malformed")
        try:
            recorded_path = Path(str(record.get("path", ""))).resolve()
        except Exception as exc:
            raise frozen.IntegrityError(f"gate_v3 code role {role!r} has an invalid path") from exc
        digest = record.get("sha256")
        if recorded_path != expected_path.resolve():
            raise frozen.IntegrityError(f"gate_v3 code role {role!r} points to the wrong file")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise frozen.IntegrityError(f"gate_v3 code role {role!r} has an invalid digest")


def _validate_environment_contract(manifest: Mapping[str, Any]) -> dict[str, str | None]:
    """Require bit-for-bit equality with the runtime recorded at freeze time."""

    current = frozen._package_versions()
    if manifest.get("environment") != current:
        raise frozen.IntegrityError(
            "gate_v3 runtime environment differs from the exact frozen environment"
        )
    return current


def _archived_source_groups(archived: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    files = archived.get("files")
    groups = archived.get("groups")
    if not isinstance(files, dict) or not isinstance(groups, dict) or not files:
        raise frozen.IntegrityError("archived source files/groups are malformed")
    ordered_names = sorted(files)
    known_names = set(ordered_names)
    group_for_name: dict[str, str] = {}
    for raw_group, raw_members in groups.items():
        group = str(raw_group)
        if not group or not isinstance(raw_members, list) or not raw_members:
            raise frozen.IntegrityError("archived source-group membership is malformed")
        if not all(isinstance(name, str) for name in raw_members) or len(raw_members) != len(
            set(raw_members)
        ):
            raise frozen.IntegrityError("archived source-group members must be unique filenames")
        for name in raw_members:
            if name not in known_names or name in group_for_name:
                raise frozen.IntegrityError("archived source groups do not partition the file set")
            group_for_name[name] = group
    if set(group_for_name) != known_names:
        raise frozen.IntegrityError("archived source groups do not cover every file")
    for name, raw_record in files.items():
        if not isinstance(raw_record, dict) or raw_record.get("source_group") != group_for_name[name]:
            raise frozen.IntegrityError("archived file/source-group records disagree")
    return ordered_names, group_for_name


def _validate_v3_isolation(manifest: Mapping[str, Any], gate_dir: Path) -> None:
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != EXPECTED_SCENE_COUNT:
        raise frozen.IntegrityError(
            f"gate_v3 must contain exactly {EXPECTED_SCENE_COUNT} scenes"
        )
    if manifest.get("scene_count") != EXPECTED_SCENE_COUNT:
        raise frozen.IntegrityError("gate_v3 manifest scene_count differs from the frozen count")
    if manifest.get("gate_seed") != EXPECTED_GATE_SEED:
        raise frozen.IntegrityError(f"gate_v3 must use seed {EXPECTED_GATE_SEED}")
    if manifest.get("geometry", {}).get("orientation") != FIXED_ORIENTATION:
        raise frozen.IntegrityError("gate_v3 violates the fixed upright orientation contract")
    source_contract = manifest.get("source_groups", {})
    if source_contract.get("target_corpus_sha256") != EXPECTED_TARGET_CORPUS_SHA256:
        raise frozen.IntegrityError("gate_v3 target corpus differs from the precommitted corpus")
    selection_contract = manifest.get("selection", {})
    if selection_contract.get("validation_count") != EXPECTED_VALIDATION_COUNT:
        raise frozen.IntegrityError("gate_v3 validation count differs from the frozen contract")
    _validate_v3_code_contract(manifest)

    scene_names = [str(scene.get("name", "")) for scene in scenes]
    scene_groups = [str(scene.get("source_group", "")) for scene in scenes]
    if len(set(scene_names)) != EXPECTED_SCENE_COUNT or len(set(scene_groups)) != EXPECTED_SCENE_COUNT:
        raise frozen.IntegrityError("gate_v3 scene names and source groups must both be unique")

    prior_names, _archived_prior_groups = _load_prior_identities()
    if prior_names.intersection(scene_names):
        raise frozen.IntegrityError("gate_v3 overlaps a prior gate filename")

    archived = _source_manifest(gate_dir)
    if archived.get("algorithms") != SOURCE_ALGORITHMS_CONTRACT:
        raise frozen.IntegrityError("gate_v3 archived algorithms differ from exact v3 contract")
    if archived.get("builder_contract") != SOURCE_BUILDER_CONTRACT:
        raise frozen.IntegrityError("gate_v3 archived builder contract differs from exact v3 contract")
    split = archived.get("split", {})
    if not isinstance(split, Mapping):
        raise frozen.IntegrityError("archived source split is malformed")
    if split.get("v3_selection_contract") != V3_SELECTION_CONTRACT:
        raise frozen.IntegrityError("gate_v3 archived v3 selection contract differs")

    ordered_names, group_for_name = _archived_source_groups(archived)
    try:
        selection = validate_manifest_v3(archived, ordered_names, group_for_name)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise frozen.IntegrityError(
            f"archived source manifest violates the exact E11 v3 contract: {exc}"
        ) from exc

    train_count = int(split["train_count"])
    val_count = int(split["val_count"])
    validation_names = ordered_names[train_count : train_count + val_count]
    validation_id = {name: index for index, name in enumerate(validation_names)}
    if any(name not in validation_id for name in prior_names.union(scene_names)):
        raise frozen.IntegrityError("a prior/v3 gate scene is absent from the archived validation split")
    if set(selection.prior_scene_names) != prior_names:
        raise frozen.IntegrityError(
            "fixed prior validation IDs no longer identify the archived gate-v1/v2 scene names"
        )

    # Group IDs are derived from complete membership.  Corrected v3 grouping
    # may rename groups or merge two identities that were separate in v1/v2;
    # only the prior filenames are stable trust anchors.  Map those names
    # through the current v3 partition before applying source isolation.
    mapped_prior_groups = {group_for_name[name] for name in prior_names}
    if list(split.get("prior_source_groups_v3", [])) != sorted(mapped_prior_groups):
        raise frozen.IntegrityError("archived mapped prior source groups are not exact")
    if mapped_prior_groups.intersection(scene_groups):
        raise frozen.IntegrityError("gate_v3 overlaps a v3-mapped prior source group")

    required_exclusions = sorted(validation_id[name] for name in prior_names)
    excluded = split.get("excluded_val_ids")
    if excluded != required_exclusions or len(required_exclusions) != EXPECTED_PRIOR_SCENE_COUNT:
        raise frozen.IntegrityError(
            "gate_v3 excluded_val_ids must be exactly the 48 gate-v1/v2 validation IDs"
        )
    train_names = ordered_names[:train_count]
    selected = list(selection.selected)
    if scene_names != selected:
        raise frozen.IntegrityError(
            "gate_v3 selected order differs from the independently recomputed fixed-seed order"
        )
    expected_scene_groups = [group_for_name[name] for name in selected]
    if scene_groups != expected_scene_groups:
        raise frozen.IntegrityError("gate_v3 scene source groups differ from the archived mapping")

    recorded_training_groups = set(
        manifest.get("splits", {}).get("training", {}).get("source_groups", [])
    )
    expected_training_groups = {group_for_name[name] for name in train_names}
    if recorded_training_groups != expected_training_groups:
        raise frozen.IntegrityError("gate_v3 recorded training groups differ from the archived mapping")
    if recorded_training_groups.intersection(scene_groups):
        raise frozen.IntegrityError("gate_v3 source group leaks into its training split")


def _load_verified_inputs(
    gate_dir: Path, score_cache_dir: Path, *, expected_root: str | None
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    str,
    dict[str, Any],
    dict[str, Any],
]:
    """Load only a complete cache bound to the future pinned gate_v3."""

    _require_pinned_root_configuration(expected_root)
    _require_exact_paths(gate_dir, score_cache_dir)
    gate_dir = gate_dir.resolve()
    score_cache_dir = score_cache_dir.resolve()
    if not score_cache_dir.is_dir():
        raise frozen.CacheContractError(f"score cache directory is missing: {score_cache_dir}")

    manifest, scene_arrays, root_digest = frozen.load_and_verify_gate(gate_dir)
    _require_expected_root(root_digest, expected_root=expected_root)
    _validate_v3_isolation(manifest, gate_dir)
    _validate_environment_contract(manifest)
    _require_upright_orientations(scene_arrays)

    frozen._verify_external_files(manifest)
    verification = frozen.verify_score_cache_directory(
        manifest, root_digest, score_cache_dir, require_complete=True
    )
    expected_verification = {"verified": EXPECTED_SCENE_COUNT, "missing": []}
    if verification != expected_verification:
        raise frozen.CacheContractError("score_cache_v4 verification was not exact and complete")
    preparation = _verify_cache_preparation_receipts(
        manifest, root_digest, score_cache_dir
    )

    caches: dict[str, dict[str, np.ndarray]] = {}
    for scene in manifest["scenes"]:
        cache_path, _ = frozen._score_cache_paths(score_cache_dir, scene["name"])
        caches[scene["name"]] = frozen.load_score_cache(
            cache_path,
            frozen._score_cache_contract(manifest, scene, root_digest),
        )
    return manifest, scene_arrays, caches, root_digest, verification, preparation


def _dense_matrices(candidates: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from eval_seeded_qap import dense_rd

    candidate_array = np.asarray(candidates)
    score_array = np.asarray(scores)
    if candidate_array.ndim != 2 or candidate_array.shape[0] != frozen.NFRAG:
        raise frozen.CacheContractError("candidate matrix has an invalid shape")
    expected_scores = (frozen.NFRAG, frozen.NUM_DIRECTIONS, candidate_array.shape[1])
    if score_array.shape != expected_scores:
        raise frozen.CacheContractError(
            f"raw score matrix must have shape {expected_scores}, got {score_array.shape}"
        )
    candidate_tensor = torch.from_numpy(candidate_array.astype(np.int64, copy=False)).long()
    score_tensor = torch.from_numpy(np.ascontiguousarray(score_array)).permute(1, 0, 2).contiguous()
    right_t, down_t = dense_rd(candidate_tensor, score_tensor)
    right = np.ascontiguousarray(right_t.numpy(), dtype=np.float32)
    down = np.ascontiguousarray(down_t.numpy(), dtype=np.float32)
    expected_dense = (frozen.NFRAG, frozen.NFRAG)
    if right.shape != expected_dense or down.shape != expected_dense:
        raise frozen.IntegrityError("dense conversion returned an invalid matrix shape")
    if not np.isfinite(right).all() or not np.isfinite(down).all():
        raise frozen.IntegrityError("dense conversion returned non-finite values")
    if np.any(right < 0.0) or np.any(down < 0.0):
        raise frozen.IntegrityError("dense conversion returned negative probabilities")
    return right, down


def keep_rule(*, mean_final_delta: float, mean_solve_delta: float) -> bool:
    final_delta = float(mean_final_delta)
    solve_delta = float(mean_solve_delta)
    if not np.isfinite(final_delta) or not np.isfinite(solve_delta):
        raise ValueError("keep-rule deltas must be finite")
    return (
        final_delta > KEEP_FINAL_DELTA_STRICTLY_GREATER_THAN
        and solve_delta >= KEEP_SOLVE_DELTA_GREATER_THAN_OR_EQUAL_TO
    )


def _hash_records(records: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {role: str(record["sha256"]) for role, record in sorted(records.items())}


def _canonical_report_bytes(report: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise frozen.IntegrityError(f"report is not strict canonical JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _fsync_parent_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_exclusive_exact(path: Path, content: bytes) -> bool:
    """Publish complete bytes without replacement; accept only an exact winner."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise frozen.IntegrityError(f"immutable artifact differs; refusing overwrite: {path}")
        return False

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.stage-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Windows rename is same-directory atomic and refuses an existing
            # destination; POSIX rename replaces, so use an atomic hard link
            # there. Neither branch overwrites the winning immutable bytes.
            if os.name == "nt":
                os.rename(temporary, path)
            else:
                os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != content:
                raise frozen.IntegrityError(
                    f"concurrent immutable artifact differs; refusing overwrite: {path}"
                )
            return False
        _fsync_parent_directory(path.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _cache_receipt_paths(cache_dir: Path) -> tuple[Path, Path]:
    root = cache_dir.resolve()
    return (
        root / "PREPARE_CACHE_STARTED.json",
        root / "PREPARE_CACHE_COMPLETE.json",
    )


def _fixed_cuda_runtime() -> dict[str, Any]:
    """Return the one permitted neural-scoring runtime or fail before writes."""

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise frozen.CacheContractError(
            "E11 score-cache preparation requires the fixed CUDA device cuda:0"
        )
    capability = torch.cuda.get_device_capability(0)
    cuda_version = torch.version.cuda
    if not isinstance(cuda_version, str) or not cuda_version:
        raise frozen.CacheContractError("PyTorch reports CUDA available without a CUDA version")
    return {
        "device": FIXED_SCORE_CACHE_DEVICE,
        "device_index": 0,
        "device_name": str(torch.cuda.get_device_name(0)),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "torch_cuda_version": cuda_version,
        "cudnn_version": (
            None if torch.backends.cudnn.version() is None else int(torch.backends.cudnn.version())
        ),
    }


def _cache_index_contract_sha256(manifest: Mapping[str, Any], root_digest: str) -> str:
    return frozen._sha256_bytes(
        frozen._canonical_json_bytes(frozen._cache_index_contract(manifest, root_digest))
    )


def _cache_start_contract(
    manifest: Mapping[str, Any], root_digest: str, cuda_runtime: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": CACHE_PREPARATION_STARTED_SCHEMA,
        "gate_root_sha256": root_digest,
        "scoring_device": FIXED_SCORE_CACHE_DEVICE,
        "cuda_runtime": dict(cuda_runtime),
        "environment": dict(manifest["environment"]),
        "cache_index_contract_sha256": _cache_index_contract_sha256(manifest, root_digest),
        "checkpoints": _hash_records(manifest["checkpoints"]),
        "code": _hash_records(manifest["code"]),
    }


def _expected_cache_entry_names(manifest: Mapping[str, Any]) -> set[str]:
    result = {
        "CACHE_INDEX.json",
        "PREPARE_CACHE_STARTED.json",
        "PREPARE_CACHE_COMPLETE.json",
    }
    for scene in manifest["scenes"]:
        cache_name = f"{scene['name']}.scores.npz"
        result.add(cache_name)
        result.add(f"{cache_name}.sha256")
    return result


def _reject_unexpected_cache_entries(cache_dir: Path, manifest: Mapping[str, Any]) -> None:
    if not cache_dir.exists():
        return
    if not cache_dir.is_dir():
        raise frozen.CacheContractError(f"score cache path is not a directory: {cache_dir}")
    unexpected = sorted(
        path.name
        for path in cache_dir.iterdir()
        if path.name not in _expected_cache_entry_names(manifest)
    )
    if unexpected:
        raise frozen.CacheContractError(
            f"score_cache_v4 contains unexpected entries: {unexpected[:3]}"
        )


def _begin_cache_preparation(
    cache_dir: Path,
    manifest: Mapping[str, Any],
    root_digest: str,
    cuda_runtime: Mapping[str, Any],
) -> tuple[str, str]:
    """Create/resume the immutable cache-start contract on the same CUDA runtime."""

    _reject_unexpected_cache_entries(cache_dir, manifest)
    started_path, _ = _cache_receipt_paths(cache_dir)
    if cache_dir.exists() and not started_path.exists():
        existing = list(cache_dir.iterdir())
        if existing:
            raise frozen.CacheContractError(
                "refusing to adopt a pre-existing score cache without E11 start provenance"
            )
    content = _canonical_report_bytes(
        _cache_start_contract(manifest, root_digest, cuda_runtime)
    )
    created = _publish_exclusive_exact(started_path, content)
    return frozen._sha256_bytes(content), "created" if created else "resumed"


def _cache_file_hash_records(
    manifest: Mapping[str, Any], cache_dir: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ordinal, scene in enumerate(manifest["scenes"]):
        cache, sidecar = frozen._score_cache_paths(cache_dir, scene["name"])
        if not cache.is_file() or not sidecar.is_file():
            raise frozen.CacheContractError(
                f"score cache pair is incomplete at ordinal {ordinal}"
            )
        records.append(
            {
                "ordinal": ordinal,
                "score_cache_sha256": frozen.sha256_file(cache),
                "score_cache_sidecar_sha256": frozen.sha256_file(sidecar),
            }
        )
    return records


def _cache_complete_contract(
    manifest: Mapping[str, Any],
    root_digest: str,
    cache_dir: Path,
    started_sha256: str,
) -> dict[str, Any]:
    index_path = cache_dir / "CACHE_INDEX.json"
    if not index_path.is_file():
        raise frozen.CacheContractError("score cache index is missing at completion")
    return {
        "schema": CACHE_PREPARATION_COMPLETE_SCHEMA,
        "gate_root_sha256": root_digest,
        "scoring_device": FIXED_SCORE_CACHE_DEVICE,
        "started_receipt_sha256": started_sha256,
        "cache_index_sha256": frozen.sha256_file(index_path),
        "cache_files": _cache_file_hash_records(manifest, cache_dir),
    }


def _load_strict_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise frozen.CacheContractError(f"could not read {label}: {exc}") from exc
    if not isinstance(payload, dict) or path.read_bytes() != _canonical_report_bytes(payload):
        raise frozen.CacheContractError(f"{label} is not strict canonical JSON")
    return payload


def _verify_cache_preparation_receipts(
    manifest: Mapping[str, Any], root_digest: str, cache_dir: Path
) -> dict[str, Any]:
    """Bind complete generic caches to the fixed E11 CUDA preparation action."""

    _reject_unexpected_cache_entries(cache_dir, manifest)
    started_path, complete_path = _cache_receipt_paths(cache_dir)
    if not started_path.is_file() or not complete_path.is_file():
        raise frozen.CacheContractError(
            "score_cache_v4 requires both immutable E11 preparation receipts"
        )
    started = _load_strict_json(started_path, label="cache preparation start receipt")
    runtime = started.get("cuda_runtime")
    if (
        set(started)
        != {
            "schema",
            "gate_root_sha256",
            "scoring_device",
            "cuda_runtime",
            "environment",
            "cache_index_contract_sha256",
            "checkpoints",
            "code",
        }
        or started.get("schema") != CACHE_PREPARATION_STARTED_SCHEMA
        or started.get("gate_root_sha256") != root_digest
        or started.get("scoring_device") != FIXED_SCORE_CACHE_DEVICE
        or started.get("environment") != manifest.get("environment")
        or started.get("cache_index_contract_sha256")
        != _cache_index_contract_sha256(manifest, root_digest)
        or started.get("checkpoints") != _hash_records(manifest["checkpoints"])
        or started.get("code") != _hash_records(manifest["code"])
        or not isinstance(runtime, dict)
        or set(runtime)
        != {
            "device",
            "device_index",
            "device_name",
            "compute_capability",
            "torch_cuda_version",
            "cudnn_version",
        }
        or runtime.get("device") != FIXED_SCORE_CACHE_DEVICE
        or runtime.get("device_index") != 0
        or not isinstance(runtime.get("device_name"), str)
        or not runtime.get("device_name")
        or not isinstance(runtime.get("compute_capability"), list)
        or len(runtime["compute_capability"]) != 2
        or any(type(value) is not int for value in runtime["compute_capability"])
        or not isinstance(runtime.get("torch_cuda_version"), str)
        or (
            runtime.get("cudnn_version") is not None
            and type(runtime.get("cudnn_version")) is not int
        )
    ):
        raise frozen.CacheContractError("cache preparation start receipt differs from contract")
    started_sha256 = frozen.sha256_file(started_path)
    expected_complete = _cache_complete_contract(
        manifest, root_digest, cache_dir, started_sha256
    )
    complete = _load_strict_json(complete_path, label="cache preparation completion receipt")
    if complete != expected_complete:
        raise frozen.CacheContractError(
            "cache preparation completion receipt differs from current cache bytes"
        )
    return {
        "scoring_device": FIXED_SCORE_CACHE_DEVICE,
        "cuda_runtime": runtime,
        "started_receipt_sha256": started_sha256,
        "complete_receipt_sha256": frozen.sha256_file(complete_path),
    }


def _require_upright_orientations(
    scene_arrays: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    for name, arrays in scene_arrays.items():
        orientations = np.asarray(arrays["orientations_quarter_turns"])
        if orientations.shape != (frozen.NFRAG,) or np.any(orientations != 0):
            raise frozen.IntegrityError(f"scene {name} contains a rotated tile")


def prepare_score_cache_v3(*, expected_root: str | None) -> dict[str, Any]:
    """Create neural score rows without solver or label-derived operations."""

    _require_pinned_root_configuration(expected_root)
    paths = _default_paths()
    gate_dir = paths["gate"]
    score_cache_dir = paths["score_cache"]
    _require_exact_paths(gate_dir, score_cache_dir)
    gate_dir = gate_dir.resolve()
    score_cache_dir = score_cache_dir.resolve()

    manifest, scene_arrays, root_digest = frozen.load_and_verify_gate(gate_dir)
    _require_expected_root(root_digest, expected_root=expected_root)
    _validate_v3_isolation(manifest, gate_dir)
    _validate_environment_contract(manifest)
    _require_upright_orientations(scene_arrays)
    frozen._verify_external_files(manifest)

    # Device selection is part of the hashed experiment.  There is no CLI or
    # environment-driven fallback to CPU or another CUDA ordinal.
    cuda_runtime = _fixed_cuda_runtime()
    started_sha256, start_status = _begin_cache_preparation(
        score_cache_dir, manifest, root_digest, cuda_runtime
    )
    frozen._ensure_cache_index(
        score_cache_dir, frozen._cache_index_contract(manifest, root_digest)
    )

    _, complete_path = _cache_receipt_paths(score_cache_dir)
    missing: list[Mapping[str, Any]] = []
    reused = 0
    for ordinal, scene in enumerate(manifest["scenes"]):
        cache_path, sidecar = frozen._score_cache_paths(score_cache_dir, scene["name"])
        if cache_path.exists() != sidecar.exists():
            raise frozen.CacheContractError(
                f"orphan score-cache file at ordinal {ordinal}; refusing automatic repair"
            )
        if cache_path.exists():
            frozen.load_score_cache(
                cache_path,
                frozen._score_cache_contract(manifest, scene, root_digest),
            )
            reused += 1
        else:
            missing.append(scene)
    if complete_path.exists() and missing:
        raise frozen.CacheContractError(
            "completed score-cache receipt exists but scene caches are missing"
        )

    created = 0
    if missing:
        models = frozen._ScoringModels(manifest, FIXED_SCORE_CACHE_DEVICE)
        for scene in missing:
            # Deliberately pass only corrupted upright tiles.  Targets,
            # permutations, solver functions, restoration, and metrics are not
            # referenced anywhere in this action.
            computed = models.score(scene_arrays[scene["name"]]["tiles"])
            cache_path, _ = frozen._score_cache_paths(score_cache_dir, scene["name"])
            contract = frozen._score_cache_contract(manifest, scene, root_digest)
            frozen.write_score_cache(cache_path, contract=contract, **computed)
            frozen.load_score_cache(cache_path, contract)
            created += 1
            print(
                json.dumps(
                    {
                        "prepare_cache": {
                            "completed": reused + created,
                            "total": len(manifest["scenes"]),
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    verification = frozen.verify_score_cache_directory(
        manifest, root_digest, score_cache_dir, require_complete=True
    )
    expected_verification = {"verified": len(manifest["scenes"]), "missing": []}
    if verification != expected_verification:
        raise frozen.CacheContractError("prepared score_cache_v4 is not exact and complete")

    complete_content = _canonical_report_bytes(
        _cache_complete_contract(
            manifest, root_digest, score_cache_dir, started_sha256
        )
    )
    complete_created = _publish_exclusive_exact(complete_path, complete_content)
    preparation = _verify_cache_preparation_receipts(
        manifest, root_digest, score_cache_dir
    )
    return {
        "status": "score_cache_ready",
        "gate_root_sha256": root_digest,
        "scene_count": len(manifest["scenes"]),
        "created": created,
        "reused": reused,
        "cache_verification": verification,
        "scoring_device": FIXED_SCORE_CACHE_DEVICE,
        "cuda_runtime": cuda_runtime,
        "start_status": start_status,
        "complete_status": "created" if complete_created else "already_identical",
        "preparation": preparation,
        "label_derived_metrics_computed": False,
    }


def _claim_evaluation_once(
    *,
    path: Path,
    report_path: Path,
    root_digest: str,
    preparation: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Atomically spend the gate before any metric is calculated or printed."""

    path = path.resolve()
    report_path = report_path.resolve()
    report_sidecar = report_path.with_suffix(report_path.suffix + ".sha256")
    if report_path.exists() or report_sidecar.exists():
        raise frozen.IntegrityError(
            "an E11 report artifact already exists; refusing to recompute or reveal metrics"
        )
    if path.exists():
        raise frozen.IntegrityError(
            "EVALUATION_STARTED already exists; the one-shot E11 gate is spent"
        )
    claim = {
        "schema": EVALUATION_STARTED_SCHEMA,
        "gate_root_sha256": root_digest,
        "report_path": str(report_path),
        "score_cache_complete_receipt_sha256": preparation[
            "complete_receipt_sha256"
        ],
    }
    content = _canonical_report_bytes(claim)
    created = _publish_exclusive_exact(path, content)
    if not created:
        # A concurrent identical claimant won the atomic create.  Only that
        # process may cross the metrics boundary.
        raise frozen.IntegrityError(
            "another process already claimed the one-shot E11 evaluation"
        )
    return claim, frozen._sha256_bytes(content)


def _write_immutable_report(path: Path, report: Mapping[str, Any]) -> tuple[str, str]:
    path = path.resolve()
    content = _canonical_report_bytes(report)
    digest = frozen._sha256_bytes(content)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_content = f"{digest}  {path.name}\n".encode("ascii")
    report_existed = path.exists()
    sidecar_existed = sidecar.exists()
    report_created = _publish_exclusive_exact(path, content)
    _publish_exclusive_exact(sidecar, sidecar_content)
    if path.read_bytes() != content or sidecar.read_bytes() != sidecar_content:
        raise frozen.IntegrityError("immutable report pair failed final byte verification")
    if report_created:
        status = "created"
    elif report_existed and sidecar_existed:
        status = "already_identical"
    else:
        status = "recovered"
    return digest, status


def _canonical_report_path(requested: Path | None = None) -> Path:
    canonical = _default_paths()["report"].resolve()
    candidate = canonical if requested is None else requested.resolve()
    if candidate != canonical:
        raise frozen.IntegrityError(
            f"gate_v4 report path must be exactly {canonical}; received {candidate}"
        )
    return canonical


def preflight_rank96_lab_selector_v3(*, expected_root: str | None) -> dict[str, Any]:
    """Verify future frozen inputs without solving, restoring, or reading metrics."""

    paths = _default_paths()
    manifest, _, caches, root_digest, cache_verification, cache_preparation = _load_verified_inputs(
        paths["gate"], paths["score_cache"], expected_root=expected_root
    )
    return {
        "status": "preflight_ok",
        "gate_root_sha256": root_digest,
        "gate_seed": EXPECTED_GATE_SEED,
        "scene_count": len(manifest["scenes"]),
        "cache_count": len(caches),
        "cache_verification": cache_verification,
        "score_cache_preparation": cache_preparation,
        "execution_device": "cpu",
        "orientation": FIXED_ORIENTATION,
        "arms": list(ARM_ORDER),
        "keep_rule": {
            "mean_final_delta_strictly_greater_than": KEEP_FINAL_DELTA_STRICTLY_GREATER_THAN,
            "mean_solve_delta_greater_than_or_equal_to": (
                KEEP_SOLVE_DELTA_GREATER_THAN_OR_EQUAL_TO
            ),
        },
        "checkpoints": _hash_records(manifest["checkpoints"]),
        "code": _hash_records(manifest["code"]),
        "environment": dict(manifest["environment"]),
        "source_group_algorithms": SOURCE_ALGORITHMS_CONTRACT,
        "source_group_builder_contract": SOURCE_BUILDER_CONTRACT,
    }


def evaluate_rank96_lab_selector_v3(
    *, expected_root: str | None, report_path: Path | None = None
) -> dict[str, Any]:
    _require_pinned_root_configuration(expected_root)
    report_path = _canonical_report_path(report_path)
    paths = _default_paths()
    gate_dir = paths["gate"]
    score_cache_dir = paths["score_cache"]
    manifest, scenes, caches, root_digest, cache_verification, cache_preparation = _load_verified_inputs(
        gate_dir, score_cache_dir, expected_root=expected_root
    )
    evaluation_claim, evaluation_claim_sha256 = _claim_evaluation_once(
        path=paths["evaluation_started"],
        report_path=report_path,
        root_digest=root_digest,
        preparation=cache_preparation,
    )
    cache_dir = score_cache_dir.resolve()
    cache_index = cache_dir / "CACHE_INDEX.json"
    per_scene: list[dict[str, Any]] = []

    for scene in manifest["scenes"]:
        name = scene["name"]
        frozen_scene = scenes[name]
        cache = caches[name]
        candidates_stored = cache["candidate_ids"]
        candidates = candidates_stored.astype(np.int64)
        valid = cache["candidate_valid"]
        raw = cache["raw_scores"]
        expanded_valid = np.broadcast_to(valid[:, None, :], raw.shape)
        raw_masked = np.where(expanded_valid, raw, -np.inf).astype(np.float32, copy=False)
        right, down = _dense_matrices(candidates, raw_masked)
        selection = solve_and_select_lab_depth1(
            frozen_scene["tiles"], right, down
        )
        permutation = frozen_scene["permutation"]
        raw_edge_r1 = frozen.edge_r1(candidates, valid, raw_masked, permutation)

        rank96_metrics = {
            "edge_r1": raw_edge_r1,
            **frozen._board_metrics(
                tiles=frozen_scene["tiles"],
                target=frozen_scene["target"],
                permutation=permutation,
                board=selection.rank96_board,
                restorer=frozen._fixed_nlm,
            ),
        }
        if selection.selected_arm == RANK96_ARM:
            selector_metrics = dict(rank96_metrics)
        else:
            selector_metrics = {
                "edge_r1": raw_edge_r1,
                **frozen._board_metrics(
                    tiles=frozen_scene["tiles"],
                    target=frozen_scene["target"],
                    permutation=permutation,
                    board=selection.selected_board,
                    restorer=frozen._fixed_nlm,
                ),
            }
        arms = {"rank96": rank96_metrics, "selector": selector_metrics}

        cache_path, sidecar = frozen._score_cache_paths(cache_dir, name)
        per_scene.append(
            {
                "name": name,
                "source_group": scene["source_group"],
                "hashes": {
                    "scene_file_sha256": scene["file_sha256"],
                    "tiles_sha256": scene["arrays_sha256"]["tiles"],
                    "target_sha256": scene["arrays_sha256"]["target"],
                    "permutation_sha256": scene["arrays_sha256"]["permutation"],
                    "orientations_quarter_turns_sha256": scene["arrays_sha256"][
                        "orientations_quarter_turns"
                    ],
                    "score_cache_sha256": frozen.sha256_file(cache_path),
                    "score_cache_sidecar_sha256": frozen.sha256_file(sidecar),
                    "candidate_ids_sha256": frozen.sha256_array(candidates_stored),
                    "candidate_valid_sha256": frozen.sha256_array(valid),
                    "raw_scores_sha256": frozen.sha256_array(raw),
                    "masked_raw_scores_sha256": frozen.sha256_array(raw_masked),
                    "dense_right_sha256": frozen.sha256_array(right),
                    "dense_down_sha256": frozen.sha256_array(down),
                    "rank96_board_sha256": frozen.sha256_array(selection.rank96_board),
                    "rank512_board_sha256": frozen.sha256_array(selection.rank512_board),
                    "selected_board_sha256": frozen.sha256_array(selection.selected_board),
                },
                "selection": {
                    "selected_arm": selection.selected_arm,
                    "rank96_lab_score": selection.rank96_lab_score,
                    "rank512_lab_score": selection.rank512_lab_score,
                    "lab_margin_rank96_minus_rank512": (
                        selection.lab_margin_rank96_minus_rank512
                    ),
                    "rank96_solver_objective": selection.rank96_objective,
                    "rank512_solver_objective": selection.rank512_objective,
                },
                "arms": arms,
            }
        )
        print(json.dumps({"scene": name, "selection": per_scene[-1]["selection"], "arms": arms}, sort_keys=True), flush=True)

    aggregate = {
        arm: frozen._summarize_arm([row["arms"][arm] for row in per_scene])
        for arm in ARM_ORDER
    }
    paired = frozen._paired_summary(
        per_scene, left="selector", right="rank96", seed=PAIR_BOOTSTRAP_SEED
    )
    mean_final_delta = float(paired["final_ssim"]["mean_delta"])
    mean_solve_delta = float(paired["solve_ssim"]["mean_delta"])
    keep = keep_rule(
        mean_final_delta=mean_final_delta,
        mean_solve_delta=mean_solve_delta,
    )
    expected_index_contract = frozen._cache_index_contract(manifest, root_digest)
    selector_path = Path(__file__).resolve().with_name("rank96_lab_selector.py")
    core_path = Path(__file__).resolve()
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "gate_root_sha256": root_digest,
        "gate_seed": EXPECTED_GATE_SEED,
        "scene_count": len(per_scene),
        "execution_device": "cpu",
        "orientation": FIXED_ORIENTATION,
        "selection_or_sweep": "one_fixed_label_free_selector_no_sweep",
        "arms": FIXED_ARMS,
        "aggregate": aggregate,
        "paired_selector_minus_rank96": paired,
        "keep_rule": {
            "mean_final_delta_strictly_greater_than": KEEP_FINAL_DELTA_STRICTLY_GREATER_THAN,
            "mean_solve_delta_greater_than_or_equal_to": (
                KEEP_SOLVE_DELTA_GREATER_THAN_OR_EQUAL_TO
            ),
        },
        "decision": {
            "status": "keep" if keep else "reject",
            "keep": keep,
            "mean_final_delta": mean_final_delta,
            "mean_solve_delta": mean_solve_delta,
        },
        "contracts": {
            "score_cache_name": "score_cache_v4",
            "score_cache_schema": frozen.SCORE_CACHE_SCHEMA,
            "candidate_k_per_encoder": frozen.FIXED_ARMS["i11"]["candidate_k_per_encoder"],
            "cache_verification": cache_verification,
            "score_cache_preparation": cache_preparation,
            "cache_index_contract_sha256": frozen._sha256_bytes(
                frozen._canonical_json_bytes(expected_index_contract)
            ),
            "checkpoints": _hash_records(manifest["checkpoints"]),
            "code": _hash_records(manifest["code"]),
            "environment": dict(manifest["environment"]),
            "source_group_algorithms": SOURCE_ALGORITHMS_CONTRACT,
            "source_group_builder_contract": SOURCE_BUILDER_CONTRACT,
        },
        "hashes": {
            "gate_manifest_sha256": frozen.sha256_file(gate_dir.resolve() / "manifest.json"),
            "gate_sha256sums_sha256": frozen.sha256_file(gate_dir.resolve() / "SHA256SUMS"),
            "score_cache_index_sha256": frozen.sha256_file(cache_index),
            "selector_code_sha256": frozen.sha256_file(selector_path),
            "e11_core_code_sha256": frozen.sha256_file(core_path),
            "evaluation_started_sha256": evaluation_claim_sha256,
        },
        "evaluation_claim": evaluation_claim,
        "paired_bootstrap_seed": PAIR_BOOTSTRAP_SEED,
        "per_scene": per_scene,
    }
    report_digest, write_status = _write_immutable_report(report_path, report)
    report["_write_status"] = {
        "created": write_status == "created",
        "status": write_status,
        "report_sha256": report_digest,
        "path": str(report_path.resolve()),
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "prepare-cache",
        help="create/verify fixed cuda:0 model-score caches without metrics",
    )
    subparsers.add_parser(
        "preflight",
        help="verify pinned frozen contracts only; do not solve or run NLM",
    )
    subparsers.add_parser(
        "evaluate",
        help="atomically claim and run the single permitted selector evaluation",
    )
    return parser


def run_cli(*, expected_root: str | None, argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare-cache":
        print(
            json.dumps(
                prepare_score_cache_v3(expected_root=expected_root), sort_keys=True
            ),
            flush=True,
        )
        return 0
    if args.command == "preflight":
        print(
            json.dumps(
                preflight_rank96_lab_selector_v3(expected_root=expected_root), sort_keys=True
            ),
            flush=True,
        )
        return 0
    if args.command != "evaluate":
        raise AssertionError(f"unexpected E11 command {args.command!r}")
    result = evaluate_rank96_lab_selector_v3(expected_root=expected_root)
    print(
        json.dumps(
            {"decision": result["decision"], "write": result["_write_status"]},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "rank96_lab_selector_v3_core is not a launcher; use eval_rank96_lab_selector_v3.py"
    )
