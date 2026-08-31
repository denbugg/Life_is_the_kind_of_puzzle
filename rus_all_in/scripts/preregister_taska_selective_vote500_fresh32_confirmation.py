#!/usr/bin/env python3
"""Materialize the one-shot selective-target500 confirmation preregistration.

This helper is intentionally write-once.  It snapshots every explicit
organizer image filename already present in TASKA configs and TASKA output
metadata, then selects a new 16-source organizer-train roster by deterministic
SHA-256 ranking.  It never opens target pixels.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from aiijc_puzzle.protocol import compute_protocol_digest, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/taska_selective_vote500_fresh32_confirmation_v1.json"
SNAPSHOT = PROJECT_ROOT / "configs/taska_selective_vote500_fresh32_confirmation_v1.exclusions.json"
MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
TAIL192_CONFIG = PROJECT_ROOT / "configs/taska_focal_gated_tail192_fresh16_capacity_v1.json"
FULLRES_CONFIRM_CONFIG = (
    PROJECT_ROOT / "configs/taska_fullres_focal_gated_tail_fresh32_confirmation_v1.json"
)

SOURCE_MINIMUM = 6_700
SOURCE_MAXIMUM = 6_999
SOURCE_COUNT = 16
DRAWS = (0, 1)
SELECTION_NAMESPACE = "aiijc-taska-selective-vote500-fresh32-formal-confirmation-v1-source16xdraw2"
SELECTION_SEED = 2_026_083_198
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 500_160_198
IMAGE_PATTERN = re.compile(r"img_\d{6}\.png")


def _project_path(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _record(path: Path) -> dict[str, str]:
    return {"path": _project_path(path), "sha256": sha256_file(path)}


def _digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _cases_digest(names: list[str]) -> str:
    value = "\n".join(f"{name}\0{draw}" for name in names for draw in DRAWS)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def _write_sidecar(path: Path) -> None:
    sidecar = Path(f"{path}.sha256")
    sidecar.write_text(
        f"{sha256_file(path)}  {path.name}\n",
        encoding="utf-8",
    )


def _validate_existing_sidecar(path: Path) -> None:
    sidecar = Path(f"{path}.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"signed dependency is absent: {path}")
    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    if not tokens or tokens[0] != sha256_file(path):
        raise RuntimeError(f"signed dependency changed: {path}")


def _taska_json_inventory() -> list[Path]:
    paths = set(PROJECT_ROOT.glob("configs/taska*.json"))
    paths.update(PROJECT_ROOT.glob("outputs/taska-*/**/*.json"))
    paths.discard(CONFIG)
    paths.discard(SNAPSHOT)
    return sorted(path.resolve() for path in paths)


def _manifest_train_universe() -> list[str]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if compute_protocol_digest(payload) != payload.get("protocol_digest"):
        raise RuntimeError("organizer-train manifest digest is invalid")
    train = payload.get("splits", {}).get("train")
    if not isinstance(train, list):
        raise RuntimeError("organizer-train split is absent")
    names = sorted(
        str(row["filename"])
        for row in train
        if SOURCE_MINIMUM <= int(str(row["filename"])[4:10]) <= SOURCE_MAXIMUM
    )
    if len(names) != len(set(names)) or len(names) < SOURCE_COUNT:
        raise RuntimeError("confirmation universe is malformed")
    return names


def main() -> None:
    for path in (CONFIG, SNAPSHOT, Path(f"{CONFIG}.sha256"), Path(f"{SNAPSHOT}.sha256")):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite preregistration: {path}")
    _validate_existing_sidecar(TAIL192_CONFIG)
    _validate_existing_sidecar(FULLRES_CONFIRM_CONFIG)

    inventory = _taska_json_inventory()
    if TAIL192_CONFIG.resolve() not in inventory:
        raise RuntimeError("tail192 reservation is absent from TASKA inventory")
    if FULLRES_CONFIRM_CONFIG.resolve() not in inventory:
        raise RuntimeError("fullres confirmation is absent from TASKA inventory")
    names: set[str] = set()
    artifact_records: list[dict[str, Any]] = []
    for path in inventory:
        text = path.read_text(encoding="utf-8")
        explicit = sorted(set(IMAGE_PATTERN.findall(text)))
        names.update(explicit)
        artifact_records.append(
            {
                **_record(path),
                "explicit_source_count": len(explicit),
                "explicit_source_digest": _digest(explicit),
            }
        )
    exclusions = sorted(names)
    inventory_digest = hashlib.sha256(
        "\n".join(f"{row['path']}\0{row['sha256']}" for row in artifact_records).encode("utf-8")
    ).hexdigest()
    snapshot = {
        "schema": "aiijc-taska-selective-vote500-confirmation-exclusions-v1",
        "created_before_confirmation_inference_or_scoring": True,
        "collection": {
            "config_glob": "configs/taska*.json",
            "output_glob": "outputs/taska-*/**/*.json",
            "policy": (
                "Conservatively collect every explicit img_XXXXXX.png string "
                "from all prior TASKA config and JSON output metadata."
            ),
            "artifact_count": len(artifact_records),
            "artifact_inventory_digest": inventory_digest,
        },
        "required_signed_reservations": {
            "tail192": _record(TAIL192_CONFIG),
            "fullres_combo_confirmation": _record(FULLRES_CONFIRM_CONFIG),
        },
        "artifacts": artifact_records,
        "explicit_source_union": {
            "count": len(exclusions),
            "digest": _digest(exclusions),
            "source_filenames": exclusions,
        },
        "freshness_claim": (
            "Current TASKA-lineage source-disjoint only; not universal or model fresh."
        ),
    }
    _write_exclusive(SNAPSHOT, snapshot)
    _write_sidecar(SNAPSHOT)

    universe = _manifest_train_universe()
    eligible = [name for name in universe if name not in names]
    prefix = f"{SELECTION_NAMESPACE}\0{SELECTION_SEED}\0".encode()
    roster = sorted(
        eligible,
        key=lambda name: (hashlib.sha256(prefix + name.encode("utf-8")).digest(), name),
    )[:SOURCE_COUNT]
    if len(roster) != SOURCE_COUNT or set(roster) & names:
        raise RuntimeError("could not construct a disjoint confirmation roster")

    config = {
        "schema": "aiijc-taska-selective-vote500-fresh32-confirmation-config-v1",
        "experiment": "taska-selective-vote500-fresh32-confirmation-v1",
        "purpose": (
            "One formal source16 x draw2 confirmation of the unchanged selective "
            "target500 layout solver after pair-oriented promotion evidence."
        ),
        "protocol": {
            "config_and_sha_sidecar_created_before_inference_or_scoring": True,
            "exclusion_snapshot_created_before_roster_selection": True,
            "target_free_layouts_edges_logits_and_provenance_frozen_before_references": True,
            "single_fixed_panel_only": True,
            "threshold_arm_budget_or_roster_sweep": False,
            "current_taska_lineage_disjoint_only": True,
            "universal_or_model_freshness_claimed": False,
        },
        "panel": {
            "selection_namespace": SELECTION_NAMESPACE,
            "selection_seed": SELECTION_SEED,
            "selection_algorithm": (
                "From organizer-train filenames img_006700..img_006999, exclude "
                "the complete frozen explicit-source union, sort by "
                "sha256(namespace\\0seed\\0filename) with filename tie-break, "
                "and take the first 16."
            ),
            "universe_minimum": "img_006700.png",
            "universe_maximum": "img_006999.png",
            "organizer_train_universe_count": len(universe),
            "organizer_train_universe_digest": _digest(universe),
            "exclusion_union_count": len(exclusions),
            "exclusion_union_digest": _digest(exclusions),
            "excluded_in_universe_count": len(set(universe) & names),
            "eligible_count": len(eligible),
            "eligible_digest": _digest(eligible),
            "source_filenames": roster,
            "source_count": SOURCE_COUNT,
            "draws": list(DRAWS),
            "case_count": SOURCE_COUNT * len(DRAWS),
            "source_order_digest": _digest(roster),
            "cases_digest": _cases_digest(roster),
        },
        "candidate": {
            "entrypoint": "aiijc_puzzle.taska_selective_vote500.solve_selective_vote500",
            "matcher_passes_per_case": 1,
            "matcher_vote_target": 500,
            "same_pass_current_vote_target": 350,
            "new_edges": "target500 minus same-pass current350",
            "new_edge_acceptance": "recovered train_exact_top5 focal logit >= 0.0",
            "portfolio_arms": [
                "raw",
                "logistic",
                "focal_top5",
                "nonlinear",
                "selective_vote500_focal",
            ],
            "selector": "minimum original TASKA all-1104-bond seam cost",
            "control": "same-pass current350 four-arm winner plus focal-gated tail96",
            "candidate_layout": "five-arm winner plus winner-aligned focal-gated tail96",
            "tail_max_swaps": 96,
            "focal_logit_threshold": 0.0,
            "threshold_arm_or_budget_sweep": False,
        },
        "evaluation": {
            "primary_metric": "candidate_minus_control_satisfied_adjacent_pairs_per_board",
            "pair_denominator": 1104,
            "secondary_metrics": ["adjacency_recall", "exact_tiles_per_board"],
            "bootstrap_unit": "source_with_two_draws",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confirmation_gate": {
                "pair_delta_mean_at_least": 2.0,
                "pair_delta_ci95_lower_at_least": 0.0,
            },
        },
        "artifacts": {
            "manifest": _record(MANIFEST),
            "exclusion_snapshot": _record(SNAPSHOT),
            "exclusion_snapshot_sidecar": _record(Path(f"{SNAPSHOT}.sha256")),
            "tail192_reservation": _record(TAIL192_CONFIG),
            "fullres_combo_confirmation": _record(FULLRES_CONFIRM_CONFIG),
            "selective_solver": _record(
                PROJECT_ROOT / "src/aiijc_puzzle/taska_selective_vote500.py"
            ),
            "raw_solver": _record(PROJECT_ROOT / "src/aiijc_puzzle/raw_tail_global_solver.py"),
        },
        "legality": {
            "organizer_train_sources_only": True,
            "dirty_tiles_only_for_candidate_inference": True,
            "targets_or_exact_references_in_candidate_inference": False,
            "output_uses_each_original_upright_20x20_tile_exactly_once": True,
            "rotated_warped_replaced_or_constant_tiles": False,
            "competition_test_forbidden": True,
            "postprocessing_used": False,
        },
    }
    _write_exclusive(CONFIG, config)
    _write_sidecar(CONFIG)
    print(json.dumps({"config": _record(CONFIG), "roster": roster}, indent=2))


if __name__ == "__main__":
    main()
