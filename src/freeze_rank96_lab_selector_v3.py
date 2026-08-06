"""Freeze the v3 protocol into the clean E11 artifact attempt v4.

This is only a narrow wrapper around :func:`eval_frozen_end_to_end_gate.freeze_gate`.
All experimental controls are constants: 48 scenes, seed 20260808, a 300-image
validation split, and the exact validation IDs previously touched by tuning,
gate v1, or gate v2.  The CLI exposes paths only.

Only while the generic freezer call is active, a versioned root-independent
code registry binds E11, this freezer, the v3/legacy source-group builders, and
their contract tests.  The generic harness stays byte-for-byte unchanged and
the root-pinning launcher is deliberately excluded so pinning the newly
created root cannot invalidate the gate's own code hashes.

Running ``--help`` is safe.  The explicit ``freeze`` subcommand creates bytes;
tests mock that call and this module performs no implicit freeze at import.
"""
from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import build_source_groups_v3 as source_groups_v3
import eval_frozen_end_to_end_gate as frozen


GATE_V3_SCENES = 48
GATE_V3_SEED = 20_260_808
VALIDATION_COUNT = 300
CANDIDATE_VALIDATION_MIN = 100

# Validation-local IDs derived from the already-frozen scene names.  They are
# versioned literals in the v3 builder, rather than discovered at freeze time,
# so future gate bytes cannot change the exclusion contract.
GATE_V1_VALIDATION_IDS = source_groups_v3.GATE_V1_VALIDATION_IDS
GATE_V2_VALIDATION_IDS = source_groups_v3.GATE_V2_VALIDATION_IDS
PRIOR_GATE_VALIDATION_IDS = source_groups_v3.PRIOR_GATE_VALIDATION_IDS
PREDECLARED_TUNING_IDS = tuple(range(CANDIDATE_VALIDATION_MIN)) + PRIOR_GATE_VALIDATION_IDS
PREDECLARED_TUNING_RANGES = "0:100," + ",".join(
    f"{validation_id}:1" for validation_id in PRIOR_GATE_VALIDATION_IDS
)

EXPECTED_GENERIC_HARNESS_SHA256 = (
    "8a0a5c05485813121db0dba375491157d8085a41c5f0afe271a9731840610ae8"
)
GENERIC_CODE_ROLES = (
    "frozen_gate_harness",
    "distort",
    "solve_buddies",
    "eval_candidate_rank",
    "eval_seeded_qap",
    "eval_symbolic_ranker_blend",
    "positional_ddpm",
    "pipeline",
    "placement_metrics",
)
E11_VERSIONED_CODE_ROLES = (
    "build_source_groups_legacy",
    "build_source_groups_v3",
    "test_build_source_groups_v3",
    "rank96_lab_selector",
    "rank96_lab_selector_v3_core",
    "rank96_lab_selector_v3_freezer",
    "test_rank96_lab_selector",
    "test_eval_rank96_lab_selector_v3",
    "test_freeze_rank96_lab_selector_v3",
)
REQUIRED_CODE_ROLES = GENERIC_CODE_ROLES + E11_VERSIONED_CODE_ROLES
E11_ARTIFACT_ROOT = Path("E:/pazzle_work/rank96_e11_v4")

_GENERIC_CODE_PATHS = frozen._default_code_paths
_FREEZE_INJECTION_LOCK = threading.RLock()


def _default_paths() -> dict[str, Path]:
    frozen_defaults = frozen._default_paths()
    return {
        "targets": frozen_defaults["targets"],
        "artifact_root": E11_ARTIFACT_ROOT,
        "source_groups": E11_ARTIFACT_ROOT / "source_groups_v4.json",
        "gate": E11_ARTIFACT_ROOT / "gate_v4",
        "score_cache": E11_ARTIFACT_ROOT / "score_cache_v4",
        "report": E11_ARTIFACT_ROOT / "report_rank96_lab_selector_v4.json",
        "ranker": frozen_defaults["ranker"],
        "affinity_primary": frozen_defaults["affinity_primary"],
        "affinity_secondary": frozen_defaults["affinity_secondary"],
        "spatial": frozen_defaults["spatial"],
    }


def _versioned_code_paths(workspace: Path) -> dict[str, Path]:
    src = workspace / "src"
    return {
        "build_source_groups_legacy": src / "build_source_groups.py",
        "build_source_groups_v3": src / "build_source_groups_v3.py",
        "test_build_source_groups_v3": src / "test_build_source_groups_v3.py",
        "rank96_lab_selector": src / "rank96_lab_selector.py",
        "rank96_lab_selector_v3_core": src / "rank96_lab_selector_v3_core.py",
        "rank96_lab_selector_v3_freezer": src / "freeze_rank96_lab_selector_v3.py",
        "test_rank96_lab_selector": workspace / "tests" / "test_rank96_lab_selector.py",
        "test_eval_rank96_lab_selector_v3": src / "test_eval_rank96_lab_selector_v3.py",
        "test_freeze_rank96_lab_selector_v3": src / "test_freeze_rank96_lab_selector_v3.py",
    }


def e11_code_registry(
    workspace: Path,
    *,
    base_factory: Callable[[Path], Mapping[str, Path]] | None = None,
) -> dict[str, Path]:
    """Return the exact versioned E11 registry without mutating the harness."""

    factory = _GENERIC_CODE_PATHS if base_factory is None else base_factory
    base = dict(factory(workspace))
    if tuple(base) != GENERIC_CODE_ROLES or set(base) != set(GENERIC_CODE_ROLES):
        raise frozen.FrozenGateError("generic frozen code registry differs from its audited contract")
    additions = _versioned_code_paths(workspace)
    if set(additions) != set(E11_VERSIONED_CODE_ROLES) or set(base).intersection(additions):
        raise frozen.FrozenGateError("versioned E11 code registry is malformed")
    return {**base, **additions}


def _require_hashed_code_registry() -> dict[str, Path]:
    workspace = Path(__file__).resolve().parent.parent
    paths = e11_code_registry(workspace)
    if tuple(paths) != REQUIRED_CODE_ROLES or set(paths) != set(REQUIRED_CODE_ROLES):
        raise frozen.FrozenGateError("frozen E11 code registry differs from the exact contract")
    launcher = Path(__file__).resolve().with_name("eval_rank96_lab_selector_v3.py")
    if any(path.resolve() == launcher for path in paths.values()):
        raise frozen.FrozenGateError("root-pinning launcher must not be hashed into gate_v3")
    for role, path in paths.items():
        if not path.is_file():
            raise frozen.FrozenGateError(f"required E11 code role is missing: {role}")
    harness_digest = frozen.sha256_file(paths["frozen_gate_harness"])
    if harness_digest != EXPECTED_GENERIC_HARNESS_SHA256:
        raise frozen.FrozenGateError(
            "generic frozen harness changed; E11 requires the audited v2 harness bytes"
        )
    return paths


def _validate_e11_v3_selection(
    payload: Mapping[str, Any], target_names: Sequence[str], groups: Mapping[str, str]
) -> None:
    try:
        source_groups_v3.validate_manifest_v3(payload, target_names, groups)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise frozen.SourceGroupManifestError(
            f"E11 v3 source manifest violates its fixed contract: {exc}"
        ) from exc


@contextmanager
def _temporary_e11_freeze_contract() -> Iterator[None]:
    """Inject E11-only hooks for one freeze call and restore exact identities."""

    with _FREEZE_INJECTION_LOCK:
        original_registry = frozen._default_code_paths
        original_validator = frozen._validate_builder_v2_selection
        frozen._default_code_paths = e11_code_registry
        frozen._validate_builder_v2_selection = _validate_e11_v3_selection
        try:
            yield
        finally:
            frozen._validate_builder_v2_selection = original_validator
            frozen._default_code_paths = original_registry


def freeze_rank96_lab_selector_v3(
    *,
    targets_dir: Path,
    source_groups_path: Path,
    gate_dir: Path,
    ranker_checkpoint: Path,
    affinity_primary_checkpoint: Path,
    affinity_secondary_checkpoint: Path,
    spatial_checkpoint: Path,
) -> dict[str, Any]:
    """Call the frozen byte builder with no caller-controlled experiment knobs."""

    checkpoints: Mapping[str, Path] = {
        "ranker": ranker_checkpoint,
        "affinity_primary": affinity_primary_checkpoint,
        "affinity_secondary": affinity_secondary_checkpoint,
        "spatial": spatial_checkpoint,
    }
    _require_hashed_code_registry()
    with _temporary_e11_freeze_contract():
        return frozen.freeze_gate(
            targets_dir=targets_dir,
            source_groups_path=source_groups_path,
            gate_dir=gate_dir,
            checkpoints=checkpoints,
            number=GATE_V3_SCENES,
            gate_seed=GATE_V3_SEED,
            validation_count=VALIDATION_COUNT,
            tuning_ranges=PREDECLARED_TUNING_RANGES,
            minimum_scenes=GATE_V3_SCENES,
        )


def _build_parser() -> argparse.ArgumentParser:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze", help="create the predeclared gate_v3 bytes")
    freeze_parser.add_argument("--targets-dir", type=Path, default=defaults["targets"])
    freeze_parser.add_argument("--source-groups", type=Path, default=defaults["source_groups"])
    freeze_parser.add_argument("--gate-dir", type=Path, default=defaults["gate"])
    freeze_parser.add_argument("--ranker", type=Path, default=defaults["ranker"])
    freeze_parser.add_argument(
        "--affinity-primary", type=Path, default=defaults["affinity_primary"]
    )
    freeze_parser.add_argument(
        "--affinity-secondary", type=Path, default=defaults["affinity_secondary"]
    )
    freeze_parser.add_argument("--spatial", type=Path, default=defaults["spatial"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "freeze":
        raise AssertionError(f"unexpected command {args.command!r}")
    result = freeze_rank96_lab_selector_v3(
        targets_dir=args.targets_dir,
        source_groups_path=args.source_groups,
        gate_dir=args.gate_dir,
        ranker_checkpoint=args.ranker,
        affinity_primary_checkpoint=args.affinity_primary,
        affinity_secondary_checkpoint=args.affinity_secondary,
        spatial_checkpoint=args.spatial,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
