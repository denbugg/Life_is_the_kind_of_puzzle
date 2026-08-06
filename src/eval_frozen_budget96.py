"""Final CPU-only confirmation of raw buddies budget 96 versus budget 512.

This verifier is intentionally narrow.  It accepts only the already frozen
``gate_v1`` and its complete ``score_cache_v1`` contract, requires their exact
precommitted gate root, and evaluates two hard-coded solver arms.  There is no
budget argument, sweep, selection step, cache generation, or model execution.

The two arms reuse the same cached candidate logits and the same dense right /
down matrices.  Their only differing input is ``max_edges`` (96 versus 512);
both use ``min_margin=0`` and ``repair_passes=0``.  Both assembled boards also
receive the same frozen NLM restoration before final-SSIM measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import eval_frozen_end_to_end_gate as frozen


EXPECTED_GATE_ROOT_SHA256 = (
    "ee3d74662f5326fbd1069763fd7b96dc3adb41bde0117cba1d78ff067c6bf23d"
)
REPORT_SCHEMA = "pazzle-frozen-budget96-vs512-report-v1"
ARM_ORDER = ("budget_96", "budget_512")
PAIR_BOOTSTRAP_SEED_OFFSET = 96_512

# These values are the experiment, not CLI defaults.  Keeping the score and
# solver descriptions duplicated in each arm makes the one permitted delta
# explicit in the immutable report.
FIXED_ARMS: dict[str, dict[str, Any]] = {
    "budget_96": {
        "candidate_source": "frozen_i11_candidate_ranker_raw_logits",
        "dense_conversion": "eval_seeded_qap.dense_rd",
        "solver": "solve_buddies.solve_buddies_from_scores",
        "max_edges": 96,
        "min_margin": 0.0,
        "repair_passes": 0,
    },
    "budget_512": {
        "candidate_source": "frozen_i11_candidate_ranker_raw_logits",
        "dense_conversion": "eval_seeded_qap.dense_rd",
        "solver": "solve_buddies.solve_buddies_from_scores",
        "max_edges": 512,
        "min_margin": 0.0,
        "repair_passes": 0,
    },
}


def _default_paths() -> dict[str, Path]:
    workspace = Path(__file__).resolve().parent.parent
    root = workspace / "artifacts" / "frozen_gate"
    return {
        "gate": root / "gate_v1",
        "score_cache": root / "score_cache_v1",
        "report": root / "report_budget96_vs512_v1.json",
    }


def _require_expected_root(root_digest: str) -> None:
    if root_digest != EXPECTED_GATE_ROOT_SHA256:
        raise frozen.IntegrityError(
            "this verifier accepts only gate root "
            f"{EXPECTED_GATE_ROOT_SHA256}; received {root_digest}"
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
    """Load only a complete cache bound to the exact frozen gate contract."""

    score_cache_dir = score_cache_dir.resolve()
    if not score_cache_dir.is_dir():
        raise frozen.CacheContractError(f"score cache directory is missing: {score_cache_dir}")
    manifest, scene_arrays, root_digest = frozen.load_and_verify_gate(gate_dir)
    _require_expected_root(root_digest)

    # This verifies the current checkpoint and code bytes recorded by the gate,
    # even though cached evaluation itself does not load the neural models.
    frozen._verify_external_files(manifest)
    verification = frozen.verify_score_cache_directory(
        manifest, root_digest, score_cache_dir, require_complete=True
    )
    if verification != {"verified": len(manifest["scenes"]), "missing": []}:
        raise frozen.CacheContractError("score cache verification was not exact and complete")

    caches: dict[str, dict[str, np.ndarray]] = {}
    for scene in manifest["scenes"]:
        cache_path, _ = frozen._score_cache_paths(score_cache_dir, scene["name"])
        caches[scene["name"]] = frozen.load_score_cache(
            cache_path,
            frozen._score_cache_contract(manifest, scene, root_digest),
        )
    return manifest, scene_arrays, caches, root_digest, verification


def _dense_matrices(candidates: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert the frozen raw score rows once for both fixed solver arms."""

    import torch
    from eval_seeded_qap import dense_rd

    candidates = np.asarray(candidates)
    scores = np.asarray(scores)
    expected_scores = (frozen.NFRAG, frozen.NUM_DIRECTIONS, candidates.shape[1])
    if candidates.ndim != 2 or candidates.shape[0] != frozen.NFRAG:
        raise frozen.CacheContractError("candidate matrix has an invalid shape")
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
    """Run one of the two predeclared arms; arbitrary budgets are rejected."""

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
    """Create a report once, or accept only an exact byte-identical rerun."""

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
        # The report was created by this call, so removing it restores the
        # pre-call state rather than deleting a pre-existing user artifact.
        path.unlink(missing_ok=True)
        raise
    return digest, True


def evaluate_frozen_budget96(
    *, gate_dir: Path, score_cache_dir: Path, report_path: Path
) -> dict[str, Any]:
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
        row = {
            "name": name,
            "source_group": scene["source_group"],
            "hashes": {
                "scene_file_sha256": scene["file_sha256"],
                "tiles_sha256": scene["arrays_sha256"]["tiles"],
                "target_sha256": scene["arrays_sha256"]["target"],
                "permutation_sha256": scene["arrays_sha256"]["permutation"],
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
        per_scene.append(row)
        print(json.dumps({"scene": name, "arms": arms}, sort_keys=True), flush=True)

    aggregate = {
        arm: frozen._summarize_arm([row["arms"][arm] for row in per_scene])
        for arm in ARM_ORDER
    }
    bootstrap_seed = int(manifest["gate_seed"]) + PAIR_BOOTSTRAP_SEED_OFFSET
    paired = frozen._paired_summary(
        per_scene, left="budget_96", right="budget_512", seed=bootstrap_seed
    )
    expected_index_contract = frozen._cache_index_contract(manifest, root_digest)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "gate_root_sha256": root_digest,
        "scene_count": len(per_scene),
        "execution_device": "cpu",
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
        "paired_bootstrap_seed": bootstrap_seed,
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
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, default=defaults["gate"])
    parser.add_argument("--score-cache-dir", type=Path, default=defaults["score_cache"])
    parser.add_argument("--report", type=Path, default=defaults["report"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = evaluate_frozen_budget96(
        gate_dir=args.gate_dir,
        score_cache_dir=args.score_cache_dir,
        report_path=args.report,
    )
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
