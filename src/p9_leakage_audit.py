"""ORBIT-24 P9 G0a: static leakage audit for P8 listwise cache.

This audit deliberately uses no model training and never reads CAL, DEV, or test.
It answers whether a trivial candidate position or sentinel rule can recover the
held labels and verifies that the P8 cache sources are source-disjoint.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813")
P8 = ROOT / "P8_context_candidate_graph" / "g0_g1_capacity"
CACHE = P8 / "cache"
PREP = P8 / "p8_prepare_report.json"
OUT = ROOT / "P9_loop_decoder" / "g0_leakage_audit"
SENTINEL = -1.0e9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_report() -> dict:
    return json.loads(PREP.read_text(encoding="utf-8"))


def split_sources(report: dict) -> tuple[list[str], list[str]]:
    """Support both early and final cache-report schemas without guessing."""
    candidates = []
    for key in ("train_sources", "fit_train_sources", "train", "sources_train"):
        if isinstance(report.get(key), list):
            candidates.append((key, report[key]))
    held = []
    for key in ("held_sources", "heldout_sources", "fit_held_sources", "held", "sources_held"):
        if isinstance(report.get(key), list):
            held.append((key, report[key]))
    if candidates and held:
        return [str(x) for x in candidates[0][1]], [str(x) for x in held[0][1]]
    # The prepare report may contain a per-source row structure.
    rows = report.get("rows") or report.get("sources") or []
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        train = [str(r["source"]) for r in rows if str(r.get("split", "")).lower() == "train"]
        held_out = [str(r["source"]) for r in rows if str(r.get("split", "")).lower() in {"held", "heldout", "val"}]
        if train and held_out:
            return train, held_out
    raise RuntimeError(
        "Could not identify train/held source lists in p8_prepare_report.json; "
        f"available keys={sorted(report)}"
    )


def cache_path(source: str) -> Path:
    direct = CACHE / (Path(source).stem + ".npz")
    if direct.exists():
        return direct
    matches = list(CACHE.glob(Path(source).stem + "*.npz"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"No unique cache file for {source}: {matches}")


def collect(sources: list[str]) -> dict:
    out = {
        "n_queries": 0,
        "label_by_dir": defaultdict(Counter),
        "label_all": Counter(),
        "sentinel_label_count": 0,
        "sentinel_any_count": 0,
        "sentinel_unique_count": 0,
        "true_is_first_count": 0,
        "row_hashes": Counter(),
        "files": [],
        "field_schemas": Counter(),
    }
    for source in sources:
        path = cache_path(source)
        with np.load(path, allow_pickle=False) as z:
            keys = tuple(sorted(z.files))
            out["field_schemas"][keys] += 1
            labels = np.asarray(z["labels"], dtype=np.int64)
            dirs = np.asarray(z["directions"], dtype=np.int64)
            members = np.asarray(z["members"], dtype=np.int64)
            base = np.asarray(z["baseline"], dtype=np.float64)
            if labels.ndim != 1 or dirs.shape != labels.shape or members.shape[0] != labels.size:
                raise RuntimeError(f"Invalid listwise shapes in {path}: labels={labels.shape}, dirs={dirs.shape}, members={members.shape}")
            if base.shape != members.shape:
                raise RuntimeError(f"Base/member shape mismatch in {path}: {base.shape} vs {members.shape}")
            if np.any(labels < 0) or np.any(labels >= members.shape[1]):
                raise RuntimeError(f"Label outside candidate list in {path}")
            out["n_queries"] += int(labels.size)
            out["label_all"].update(labels.tolist())
            for d in range(4):
                out["label_by_dir"][d].update(labels[dirs == d].tolist())
            row_idx = np.arange(labels.size)
            true_base = base[row_idx, labels]
            sent = base <= (SENTINEL / 2.0)
            sent_counts = sent.sum(axis=1)
            out["sentinel_label_count"] += int(np.sum(true_base <= (SENTINEL / 2.0)))
            out["sentinel_any_count"] += int(np.sum(sent_counts > 0))
            out["sentinel_unique_count"] += int(np.sum(sent_counts == 1))
            out["true_is_first_count"] += int(np.sum(labels == 0))
            # Membership ordering repeat rate across sources: a near-fixed row is suspicious,
            # but repeating IDs alone is not a proof because permutations reuse tile IDs.
            for row in members:
                out["row_hashes"][hashlib.sha256(row.tobytes()).hexdigest()] += 1
            out["files"].append({"source": source, "cache": path.name, "sha256": sha256(path)})
    return out


def positional_accuracy(train: dict, held: dict) -> dict:
    global_mode = train["label_all"].most_common(1)[0][0]
    global_acc = held["label_all"][global_mode] / held["n_queries"]
    direction_modes = {
        d: train["label_by_dir"][d].most_common(1)[0][0]
        for d in range(4)
    }
    direction_hits = sum(held["label_by_dir"][d][direction_modes[d]] for d in range(4))
    return {
        "global_position_mode": int(global_mode),
        "global_position_accuracy_held": float(global_acc),
        "direction_position_modes": {str(k): int(v) for k, v in direction_modes.items()},
        "direction_position_accuracy_held": float(direction_hits / held["n_queries"]),
        "chance_accuracy": float(1.0 / (max(train["label_all"].keys(), default=-1) + 1)),
    }


def summary_stats(stats: dict) -> dict:
    n = stats["n_queries"]
    unique_rows = len(stats["row_hashes"])
    return {
        "n_queries": n,
        "label_histogram": {str(k): int(v) for k, v in sorted(stats["label_all"].items())},
        "true_label_at_position_zero_rate": float(stats["true_is_first_count"] / n),
        "true_label_is_sentinel_rate": float(stats["sentinel_label_count"] / n),
        "query_has_any_sentinel_rate": float(stats["sentinel_any_count"] / n),
        "query_has_exactly_one_sentinel_rate": float(stats["sentinel_unique_count"] / n),
        "unique_member_rows": unique_rows,
        "member_row_repeat_rate": float(1.0 - unique_rows / (n * 1.0)),
        "schemas": {"|".join(k): int(v) for k, v in stats["field_schemas"].items()},
        "files": stats["files"],
    }


def main() -> None:
    if not PREP.exists() or not CACHE.exists():
        raise FileNotFoundError(f"Missing P8 prepared cache/report under {P8}")
    report = load_report()
    train_sources, held_sources = split_sources(report)
    overlap = sorted(set(train_sources) & set(held_sources))
    if overlap:
        raise RuntimeError(f"Source leakage: train/held overlap ({len(overlap)}): {overlap[:5]}")
    all_sources = train_sources + held_sources
    if len(set(all_sources)) != len(all_sources):
        raise RuntimeError("Duplicate source in P8 train+held lists")
    train = collect(train_sources)
    held = collect(held_sources)
    pos = positional_accuracy(train, held)
    # Static failure conditions: any trivial position or sentinel discriminator above
    # 90% recovery is incompatible with using the learned score as evidence.
    static_failures = []
    if pos["global_position_accuracy_held"] >= 0.90:
        static_failures.append("global_candidate_position_predicts_held_labels")
    if pos["direction_position_accuracy_held"] >= 0.90:
        static_failures.append("direction_conditioned_candidate_position_predicts_held_labels")
    if summary_stats(held)["true_label_is_sentinel_rate"] >= 0.90:
        static_failures.append("true_label_is_sentinel_for_most_held_queries")
    decision = "PASS_static_audit" if not static_failures else "REJECT_static_leakage"
    result = {
        "experiment": "P9_loop_decoder",
        "gate": "G0a_static_leakage_audit",
        "cache_report_sha256": sha256(PREP),
        "cache_dir": str(CACHE),
        "train_source_count": len(train_sources),
        "held_source_count": len(held_sources),
        "train_held_overlap": overlap,
        "source_disjoint": True,
        "candidate_position_control": pos,
        "train": summary_stats(train),
        "held": summary_stats(held),
        "static_failures": static_failures,
        "decision": decision,
        "CAL_target_opened": False,
        "DEV_targets_opened": False,
        "test_accessed": False,
        "layouts_assembled": False,
        "restorer_used": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "p9_g0a_static_leakage_audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "train_sources": len(train_sources),
        "held_sources": len(held_sources),
        "position_control": pos,
        "static_failures": static_failures,
        "output": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
