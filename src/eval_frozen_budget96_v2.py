"""Immutable CPU confirmation of raw buddies budget 96 versus budget 512 on gate_v2.

This verifier is deliberately closed: it reads only the workspace's frozen
``gate_v2`` and ``score_cache_v2``, requires the exact precommitted gate root,
and evaluates exactly two hard-coded solver arms.  Tiles stay upright; neither
the scoring path nor the solver has a rotation degree of freedom.  Both arms
reuse byte-identical cached raw scores and the same dense right/down matrices,
then receive the same fixed OpenCV NLM(10) restoration.

There are no device, budget, margin, repair, blend, or sweep controls.  The
optional preflight mode verifies every gate/cache/checkpoint/code contract but
does not run either solver or NLM restoration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import eval_frozen_end_to_end_gate as frozen


EXPECTED_GATE_ROOT_SHA256 = (
    "7a5a5e68779a25fd8dc882062345a3e7b5e9e555da51dee97c5b5ca3e3558134"
)
REPORT_SCHEMA = "pazzle-frozen-budget96-vs512-report-v2"
ARM_ORDER = ("budget_96", "budget_512")
PAIR_BOOTSTRAP_SEED = 96_512_2026
FIXED_ORIENTATION = "fixed_type1_no_rotation"

# These values define the experiment.  They are intentionally not arguments.
# Duplicating all fields in both arms makes max_edges the only permitted delta.
FIXED_ARMS: dict[str, dict[str, Any]] = {
    "budget_96": {
        "candidate_source": "frozen_i11_candidate_ranker_raw_logits",
        "dense_conversion": "eval_seeded_qap.dense_rd",
        "solver": "solve_buddies.solve_buddies_from_scores",
        "max_edges": 96,
        "min_margin": 0.0,
        "repair_passes": 0,
        "orientation": FIXED_ORIENTATION,
    },
    "budget_512": {
        "candidate_source": "frozen_i11_candidate_ranker_raw_logits",
        "dense_conversion": "eval_seeded_qap.dense_rd",
        "solver": "solve_buddies.solve_buddies_from_scores",
        "max_edges": 512,
        "min_margin": 0.0,
        "repair_passes": 0,
        "orientation": FIXED_ORIENTATION,
    },
}


def _default_paths() -> dict[str, Path]:
    workspace = Path(__file__).resolve().parent.parent
    root = workspace / "artifacts" / "frozen_gate"
    return {
        "gate": root / "gate_v2",
        "score_cache": root / "score_cache_v2",
        "report": root / "report_budget96_vs512_v2.json",
    }


def _require_expected_root(root_digest: str) -> None:
    if root_digest != EXPECTED_GATE_ROOT_SHA256:
        raise frozen.IntegrityError(
            "this verifier accepts only gate_v2 root "
            f"{EXPECTED_GATE_ROOT_SHA256}; received {root_digest}"
        )


def _require_exact_paths(gate_dir: Path, score_cache_dir: Path) -> None:
    """Reject renamed, copied, or alternate gates/caches before loading them."""

    defaults = _default_paths()
    expected_gate = defaults["gate"].resolve()
    expected_cache = defaults["score_cache"].resolve()
    actual_gate = gate_dir.resolve()
    actual_cache = score_cache_dir.resolve()
    if actual_gate != expected_gate:
        raise frozen.IntegrityError(
            f"this verifier reads only {expected_gate}; received {actual_gate}"
        )
    if actual_cache != expected_cache:
        raise frozen.CacheContractError(
            f"this verifier reads only {expected_cache}; received {actual_cache}"
        )


def _load_verified_inputs(
    gate_dir: Path, score_cache_dir: Path
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    str,
    dict[str, Any],
]:
    """Load a complete score_cache_v2 bound to the exact immutable gate_v2."""

    _require_exact_paths(gate_dir, score_cache_dir)
    gate_dir = gate_dir.resolve()
    score_cache_dir = score_cache_dir.resolve()
    if not score_cache_dir.is_dir():
        raise frozen.CacheContractError(f"score cache directory is missing: {score_cache_dir}")

    manifest, scene_arrays, root_digest = frozen.load_and_verify_gate(gate_dir)
    _require_expected_root(root_digest)
    if manifest.get("geometry", {}).get("orientation") != FIXED_ORIENTATION:
        raise frozen.IntegrityError("gate_v2 violates the fixed upright orientation contract")
    for name, arrays in scene_arrays.items():
        orientations = np.asarray(arrays["orientations_quarter_turns"])
        if orientations.shape != (frozen.NFRAG,) or np.any(orientations != 0):
            raise frozen.IntegrityError(f"scene {name} contains a rotated tile")

    # Verify current bytes for every checkpoint and code dependency recorded by
    # the gate before consuming cached neural scores.
    frozen._verify_external_files(manifest)
    verification = frozen.verify_score_cache_directory(
        manifest, root_digest, score_cache_dir, require_complete=True
    )
    if verification != {"verified": len(manifest["scenes"]), "missing": []}:
        raise frozen.CacheContractError("score_cache_v2 verification was not exact and complete")

    caches: dict[str, dict[str, np.ndarray]] = {}
    for scene in manifest["scenes"]:
        cache_path, _ = frozen._score_cache_paths(score_cache_dir, scene["name"])
        caches[scene["name"]] = frozen.load_score_cache(
            cache_path,
            frozen._score_cache_contract(manifest, scene, root_digest),
        )
    return manifest, scene_arrays, caches, root_digest, verification


def _dense_matrices(candidates: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert the frozen raw rows once; both arms receive these same arrays."""

    import torch
    from eval_seeded_qap import dense_rd

    candidates = np.asarray(candidates)
    scores = np.asarray(scores)
    if candidates.ndim != 2 or candidates.shape[0] != frozen.NFRAG:
        raise frozen.CacheContractError("candidate matrix has an invalid shape")
    expected_scores = (frozen.NFRAG, frozen.NUM_DIRECTIONS, candidates.shape[1])
    if scores.shape != expected_scores:
        raise frozen.CacheContractError(
            f"raw score matrix must have shape {expected_scores}, got {scores.shape}"
        )
    candidate_tensor = torch.from_numpy(candidates.astype(np.int64, copy=False)).long()
    score_tensor = torch.from_numpy(np.ascontiguousarray(scores)).permute(1, 0, 2).contiguous()
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


def _solve_dense(
    right: np.ndarray, down: np.ndarray, *, arm: str
) -> tuple[np.ndarray, float]:
    """Run one of the two declared arms; arbitrary solver controls are absent."""

    from solve_buddies import solve_buddies_from_scores

    if arm not in FIXED_ARMS:
        raise ValueError(f"arm must be one of {ARM_ORDER}, got {arm!r}")
    expected = (frozen.NFRAG, frozen.NFRAG)
    if right.shape != expected or down.shape != expected:
        raise ValueError(f"dense score matrices must both have shape {expected}")
    config = FIXED_ARMS[arm]
    board, objective = solve_buddies_from_scores(
        right,
        down,
        max_edges=int(config["max_edges"]),
        min_margin=float(config["min_margin"]),
        repair_passes=int(config["repair_passes"]),
    )
    board = np.asarray(board, dtype=np.int64)
    frozen._assert_permutation(board, label=f"{arm} board")
    objective = float(objective)
    if not np.isfinite(objective):
        raise frozen.IntegrityError(f"{arm} solver returned a non-finite objective")
    return board, objective


def _hash_records(records: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {role: str(record["sha256"]) for role, record in sorted(records.items())}


def _write_immutable_report(path: Path, report: Mapping[str, Any]) -> tuple[str, bool]:
    """Create the report once, or accept only a byte-identical deterministic rerun."""

    path = path.resolve()
    content = frozen._canonical_json_bytes(report)
    digest = frozen._sha256_bytes(content)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_content = f"{digest}  {path.name}\n".encode("ascii")
    if path.exists():
        if path.read_bytes() != content:
            raise frozen.IntegrityError(f"existing report differs; refusing overwrite: {path}")
        if not sidecar.is_file() or sidecar.read_bytes() != sidecar_content:
            raise frozen.IntegrityError(
                f"existing report digest sidecar is missing or differs: {sidecar}"
            )
        return digest, False
    if sidecar.exists():
        raise frozen.IntegrityError(f"orphan report digest exists; refusing overwrite: {sidecar}")

    frozen._atomic_write_bytes(path, content, require_absent=True)
    try:
        frozen._atomic_write_bytes(sidecar, sidecar_content, require_absent=True)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return digest, True


def preflight_frozen_budget96_v2() -> dict[str, Any]:
    """Verify all frozen inputs without running a solver or restoration."""

    paths = _default_paths()
    manifest, _, caches, root_digest, cache_verification = _load_verified_inputs(
        paths["gate"], paths["score_cache"]
    )
    return {
        "status": "preflight_ok",
        "gate_root_sha256": root_digest,
        "scene_count": len(manifest["scenes"]),
        "cache_count": len(caches),
        "score_cache": "score_cache_v2",
        "cache_verification": cache_verification,
        "execution_device": "cpu",
        "orientation": FIXED_ORIENTATION,
        "arms": list(ARM_ORDER),
        "checkpoints": _hash_records(manifest["checkpoints"]),
        "code": _hash_records(manifest["code"]),
    }


def evaluate_frozen_budget96_v2(*, report_path: Path) -> dict[str, Any]:
    paths = _default_paths()
    gate_dir = paths["gate"]
    score_cache_dir = paths["score_cache"]
    manifest, scenes, caches, root_digest, cache_verification = _load_verified_inputs(
        gate_dir, score_cache_dir
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
        permutation = frozen_scene["permutation"]
        raw_edge_r1 = frozen.edge_r1(candidates, valid, raw_masked, permutation)

        arms: dict[str, dict[str, Any]] = {}
        for arm in ARM_ORDER:
            board, objective = _solve_dense(right, down, arm=arm)
            arms[arm] = {
                "edge_r1": raw_edge_r1,
                **frozen._board_metrics(
                    tiles=frozen_scene["tiles"],
                    target=frozen_scene["target"],
                    permutation=permutation,
                    board=board,
                    restorer=frozen._fixed_nlm,
                ),
                "solver_objective": objective,
                "board_sha256": frozen.sha256_array(board),
            }

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
                    "spatial_scores_sha256": frozen.sha256_array(cache["spatial_scores"]),
                    "masked_raw_scores_sha256": frozen.sha256_array(raw_masked),
                    "dense_right_sha256": frozen.sha256_array(right),
                    "dense_down_sha256": frozen.sha256_array(down),
                },
                "arms": arms,
            }
        )
        print(json.dumps({"scene": name, "arms": arms}, sort_keys=True), flush=True)

    aggregate = {
        arm: frozen._summarize_arm([row["arms"][arm] for row in per_scene])
        for arm in ARM_ORDER
    }
    paired = frozen._paired_summary(
        per_scene, left="budget_96", right="budget_512", seed=PAIR_BOOTSTRAP_SEED
    )
    expected_index_contract = frozen._cache_index_contract(manifest, root_digest)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "gate_root_sha256": root_digest,
        "scene_count": len(per_scene),
        "execution_device": "cpu",
        "orientation": FIXED_ORIENTATION,
        "selection_or_sweep": False,
        "arms": FIXED_ARMS,
        "restoration": frozen.FIXED_ARMS["raw_input"]["restoration"],
        "aggregate": aggregate,
        "paired_budget96_minus_budget512": paired,
        "primary": {
            "metric": "paired_mean_solve_ssim_budget96_minus_budget512",
            "value": paired["solve_ssim"]["mean_delta"],
        },
        "fixed_nlm_final": {
            "metric": "paired_mean_final_ssim_budget96_minus_budget512",
            "value": paired["final_ssim"]["mean_delta"],
        },
        "contracts": {
            "score_cache_name": "score_cache_v2",
            "score_cache_schema": frozen.SCORE_CACHE_SCHEMA,
            "candidate_k_per_encoder": frozen.FIXED_ARMS["i11"]["candidate_k_per_encoder"],
            "cache_verification": cache_verification,
            "cache_index_contract_sha256": frozen._sha256_bytes(
                frozen._canonical_json_bytes(expected_index_contract)
            ),
            "checkpoints": _hash_records(manifest["checkpoints"]),
            "code": _hash_records(manifest["code"]),
        },
        "hashes": {
            "gate_manifest_sha256": frozen.sha256_file(gate_dir.resolve() / "manifest.json"),
            "gate_sha256sums_sha256": frozen.sha256_file(gate_dir.resolve() / "SHA256SUMS"),
            "score_cache_index_sha256": frozen.sha256_file(cache_index),
            "verifier_code_sha256": frozen.sha256_file(Path(__file__).resolve()),
        },
        "paired_bootstrap_seed": PAIR_BOOTSTRAP_SEED,
        "per_scene": per_scene,
    }
    report_digest, created = _write_immutable_report(report_path, report)
    report["_write_status"] = {
        "created": created,
        "report_sha256": report_digest,
        "path": str(report_path.resolve()),
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="verify frozen contracts only; do not solve or run NLM",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_default_paths()["report"],
        help="create-once report destination (does not alter experiment controls)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.preflight:
        print(json.dumps(preflight_frozen_budget96_v2(), sort_keys=True), flush=True)
        return 0
    result = evaluate_frozen_budget96_v2(report_path=args.report)
    print(
        json.dumps(
            {
                "primary": result["primary"],
                "fixed_nlm_final": result["fixed_nlm_final"],
                "write": result["_write_status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
