"""Deterministic source partitions from the authoritative denoise manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _hash_rank(prefix: str, name: str) -> tuple[bytes, str]:
    return hashlib.sha256(f"{prefix}{name}".encode("utf-8")).digest(), name


def source_names_for_split(
    split: str,
    *,
    manifest_path: str | Path,
    quarantine_path: str | Path,
    audit_exclusion_path: str | Path = "configs/assembly_audit_exclusion_v1.json",
) -> list[str]:
    manifest = _load_json(manifest_path)
    quarantine = _load_json(quarantine_path)
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("manifest is missing splits")
    train = [str(name) for name in splits.get("train", [])]
    validation = [str(name) for name in splits.get("val", [])]
    audit = [str(name) for name in splits.get("audit", [])]
    if len(train) != 4900 or len(validation) != 700 or len(audit) != 700:
        raise ValueError("authoritative split counts are not 4900/700/700")

    train_ranked = sorted(train, key=lambda name: _hash_rank("assembly-v1:20260710:", name))
    quarantine_names = {str(name) for name in quarantine.get("quarantine_names", [])}
    eligible = sorted(
        (name for name in validation if name not in quarantine_names),
        key=lambda name: _hash_rank("20260710:", name),
    )
    if len(eligible) != 607:
        raise ValueError(f"expected 607 clean validation sources, got {len(eligible)}")

    choices = {
        "edge_train": train_ranked[:4500],
        "edge_development": train_ranked[4500:],
        "assembly_cal": eligible[:257],
        "assembly_incremental_gate": eligible[257:],
        "assembly_excluded": sorted(quarantine_names),
        "audit": audit,
        "train": train,
        "val": validation,
    }
    if split in {"assembly_audit_exposed", "assembly_final_audit"}:
        exclusion = _load_json(audit_exclusion_path)
        excluded = {str(name) for name in exclusion.get("excluded_names", [])}
        if len(excluded) != 32 or not excluded <= set(audit):
            raise ValueError("assembly audit exclusion ledger is inconsistent")
        choices["assembly_audit_exposed"] = sorted(excluded)
        choices["assembly_final_audit"] = [
            name for name in audit if name not in excluded
        ]
    if split not in choices:
        available = sorted(
            set(choices) | {"assembly_audit_exposed", "assembly_final_audit"}
        )
        raise ValueError(f"unknown split {split!r}; choose from {available}")
    return list(choices[split])


def per_source_seed(master: int, stage: str, source: str, replica: int = 0) -> int:
    digest = hashlib.sha256(f"{master}:{stage}:{source}:{replica}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
