"""Untouched paired retrieval gate for a provisional HBT continuation.

The runner refuses to start until the downloaded candidate/report hashes are
patched into this source. It compares baseline and candidate on identical
primary and independent panels, averages replicas inside each whole source,
and never treats edge count or panel count as the statistical sample size.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np


INPUT = Path("/kaggle/input/datasets/pasha883")
WORKING = Path("/kaggle/working")
RUNTIME = INPUT / "vsos-assembly-v1-runtime"
DATA = INPUT / "vsos-ai-initiative-pazzle"
CANDIDATE_DATA = INPUT / "vsos-hbt-continuation-v1"
OUTPUT = WORKING / "hbt_continuation_gate"

BASELINE_SHA256 = "c2589bff58573d592227d522954a7ed6c22f6fec7ecbbea2bf0c755d7a720787"
DENOISER_SHA256 = "77a2d8607c9bc6b80e7aa99ed03329c1fae6bf94ee1d9a654241ab93a05cc734"
CODE_ZIP_SHA256 = "726a512fca9df5003e37575181cd877abfa0f47eada478e90f9a7fc481887cf2"
CANDIDATE_SHA256 = "PATCH_BEFORE_LAUNCH"
CANDIDATE_REPORT_SHA256 = "PATCH_BEFORE_LAUNCH"

EXPECTED_CODE = {
    "scripts/train_side_embeddings.py": "20cde60a1f67e5f61d7c043f54ee72452c708551831bda88a09f6bd038565081",
    "src/puzzle_assembly/learned.py": "a415aae32b3f38aae1f4fe36d91343ead3099d448b5490c4f6eeecf6ea6337d7",
    "src/puzzle_assembly/panels.py": "783356628517e3a23b8703672bca604c3d879c875f5b5f35f87182425500280f",
    "src/puzzle_assembly/protocol.py": "b711ad6d28a2fe60329e3e8236e58adbfbceea8ca4c8bf85e9a057e7619e24f4",
    "configs/denoise_splits_seed20260710.json": "de4fd2e596efa0d157d2d4480eed5fb84812d358138a1db53c1706bfb580e345",
    "configs/denoise_validation_quarantine_v1.json": "38dfd12f60579d77999c0cdb4a648fb4ff0343a8fb5e4c421f0b29f8b7bd6215",
}

GATE_SEED = 20260712
GATE_OFFSET = 96
GATE_SOURCES = 32
REPLICAS = (0, 1)
PANELS = ("primary_kornia", "independent_libjpeg")
METRICS = ("recall_at_1", "recall_at_32", "mrr")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def exact_file(root: Path, name: str) -> Path:
    paths = sorted({path.resolve() for path in root.glob(f"**/{name}") if path.is_file()})
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one {name} below {root}, found {paths}")
    return paths[0]


def exact_directory(root: Path, suffix: str) -> Path:
    paths = sorted({path.resolve() for path in root.glob(f"**/{suffix}") if path.is_dir()})
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one {suffix} below {root}, found {paths}")
    return paths[0]


def safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        if not members:
            raise RuntimeError("empty code archive")
        for member in members:
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise RuntimeError(f"symlink archive member rejected: {member.filename}")
        handle.extractall(destination)
    return destination


def resolve_code() -> tuple[Path, dict[str, str]]:
    direct = sorted(RUNTIME.glob("**/src/puzzle_assembly/__init__.py"))
    archives = sorted(RUNTIME.glob("**/assembly_v1_code.zip"))
    if len(direct) > 1 or len(archives) > 1:
        raise RuntimeError(f"ambiguous code payload: direct={direct}, archives={archives}")
    if len(direct) == 1:
        root = direct[0].parents[2]
        provenance = {"mode": "direct_tree"}
        if archives:
            if sha256(archives[0]) != CODE_ZIP_SHA256:
                raise RuntimeError("archive alongside direct tree has wrong hash")
            provenance["archive_sha256"] = CODE_ZIP_SHA256
    elif len(archives) == 1:
        if sha256(archives[0]) != CODE_ZIP_SHA256:
            raise RuntimeError("code archive hash mismatch")
        root = safe_extract(archives[0], WORKING / "assembly_v1_code")
        provenance = {"mode": "verified_zip", "archive_sha256": CODE_ZIP_SHA256}
    else:
        raise RuntimeError("runtime code payload not found")
    for relative, expected in EXPECTED_CODE.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise RuntimeError(f"required code file failed verification: {relative}")
        provenance[relative] = expected
    return root, provenance


def require_patched_hash(value: str, label: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError(f"{label} is not patched with a trusted SHA-256")


def bootstrap_ci(values: np.ndarray, *, seed: int, resamples: int = 10_000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (GATE_SOURCES,) or not np.isfinite(values).all():
        raise ValueError(f"bootstrap requires {GATE_SOURCES} finite source values")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    chunk = 1000
    for start in range(0, resamples, chunk):
        count = min(chunk, resamples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def directional_metrics(outputs: dict[str, object], labels: object) -> dict[str, dict[str, float]]:
    import torch

    ranks: dict[str, torch.Tensor] = {}
    device = outputs["q_right"].device
    for name, query_key, candidate_key, query_values, target_values in (
        ("right", "q_right", "k_left", labels.right_queries, labels.right_targets),
        ("down", "q_down", "k_up", labels.down_queries, labels.down_targets),
    ):
        queries = torch.as_tensor(query_values, device=device, dtype=torch.long)
        targets = torch.as_tensor(target_values, device=device, dtype=torch.long)
        logits = outputs[query_key][queries] @ outputs[candidate_key].T
        logits[torch.arange(len(queries), device=device), queries] = -torch.inf
        order = logits.argsort(dim=1, descending=True)
        rank = (order == targets[:, None]).nonzero(as_tuple=False)[:, 1] + 1
        ranks[name] = rank.float()
    ranks["combined"] = torch.cat([ranks["right"], ranks["down"]])
    return {
        name: {
            "recall_at_1": float((rank <= 1).float().mean().cpu()),
            "recall_at_32": float((rank <= 32).float().mean().cpu()),
            "mrr": float((1.0 / rank).mean().cpu()),
        }
        for name, rank in ranks.items()
    }


def selection_screen(report: dict[str, object]) -> dict[str, object]:
    history = report.get("history")
    best_epoch = report.get("best_epoch")
    if not isinstance(history, list) or not isinstance(best_epoch, int):
        raise RuntimeError("candidate training report lacks history/best_epoch")
    matching = [row for row in history if row.get("epoch") == best_epoch]
    if len(matching) != 1:
        raise RuntimeError("candidate training report best epoch is ambiguous")
    metrics = matching[0].get("validation", {})
    gates = {
        "recall_at_1": {"value": metrics.get("recall_at_1"), "minimum": 0.23384511260315777},
        "mrr": {"value": metrics.get("mrr"), "minimum": 0.33185246888548135},
        "recall_at_32": {"value": metrics.get("recall_at_32"), "minimum": 0.6988892796263099},
    }
    for gate in gates.values():
        value = gate["value"]
        gate["passed"] = isinstance(value, (int, float)) and np.isfinite(value) and value >= gate["minimum"]
    return {"metrics": metrics, "gates": gates, "passed": all(gate["passed"] for gate in gates.values())}


def main() -> None:
    started = time.time()
    require_patched_hash(CANDIDATE_SHA256, "CANDIDATE_SHA256")
    require_patched_hash(CANDIDATE_REPORT_SHA256, "CANDIDATE_REPORT_SHA256")
    for root in (RUNTIME, DATA, CANDIDATE_DATA):
        if not root.is_dir():
            raise RuntimeError(f"required Kaggle dataset mount missing: {root}")
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to reuse output path: {OUTPUT}")
    OUTPUT.mkdir(parents=True)

    code_root, code_provenance = resolve_code()
    baseline_path = exact_file(RUNTIME, "hbt_d320_denoised_rgb_sobel.pt")
    denoiser_path = exact_file(RUNTIME, "selected_tilenaf_synth_50k.pt")
    candidate_path = exact_file(CANDIDATE_DATA, "hbt_d320_denoised_rgb_sobel_cont.pt")
    candidate_report_path = exact_file(CANDIDATE_DATA, "hbt_d320_denoised_rgb_sobel_cont.json")
    for path, expected, label in (
        (baseline_path, BASELINE_SHA256, "baseline"),
        (denoiser_path, DENOISER_SHA256, "denoiser"),
        (candidate_path, CANDIDATE_SHA256, "candidate"),
        (candidate_report_path, CANDIDATE_REPORT_SHA256, "candidate report"),
    ):
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{label} hash mismatch: {actual}")

    sys.path.insert(0, str(code_root / "src"))
    import torch
    from PIL import Image
    from puzzle_assembly.learned import direction_labels, load_embedding_checkpoint
    from puzzle_assembly.panels import make_exact_panel
    from puzzle_assembly.protocol import per_source_seed, source_names_for_split
    from puzzle_denoise_v2.inference import load_restorer, restore_tiles_uint8

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device_name = torch.cuda.get_device_name(0)
    if "T4" not in device_name.upper():
        raise RuntimeError(f"expected T4, got {device_name}")
    probe_value = float((torch.randn(128, 128, device="cuda") @ torch.randn(128, 128, device="cuda")).mean().cpu())
    probe = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": device_name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "matmul_mean": probe_value,
    }

    manifest = code_root / "configs/denoise_splits_seed20260710.json"
    quarantine = code_root / "configs/denoise_validation_quarantine_v1.json"
    all_names = source_names_for_split(
        "edge_development", manifest_path=manifest, quarantine_path=quarantine
    )
    edge_train = source_names_for_split(
        "edge_train", manifest_path=manifest, quarantine_path=quarantine
    )
    names = all_names[GATE_OFFSET : GATE_OFFSET + GATE_SOURCES]
    if len(names) != GATE_SOURCES:
        raise RuntimeError("fixed comparator slice is incomplete")
    candidate_training_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    if (
        candidate_training_report.get("schema_version") != 1
        or candidate_training_report.get("kind") != "puzzle_side_embedding_l1_training_report"
    ):
        raise RuntimeError("candidate training report schema/kind mismatch")
    if candidate_training_report.get("checkpoint_sha256") != CANDIDATE_SHA256:
        raise RuntimeError("candidate training report does not bind the candidate checkpoint")
    report_args = candidate_training_report.get("args", {})
    expected_report_args = {
        "train_offset": 2048,
        "train_sources": 2048,
        "val_offset": 0,
        "val_sources": 32,
        "epochs": 2,
        "replica_offset": 2,
        "learning_rate": 0.0001,
        "panel": "primary_kornia",
        "view": "denoised",
    }
    if any(report_args.get(key) != value for key, value in expected_report_args.items()):
        raise RuntimeError("candidate training report configuration mismatch")

    restorer, device, denoiser_metadata = load_restorer(
        denoiser_path, device="cuda", state="ema"
    )
    baseline, baseline_metadata = load_embedding_checkpoint(baseline_path, device=device)
    candidate, candidate_metadata = load_embedding_checkpoint(candidate_path, device=device)
    if baseline.config() != candidate.config():
        raise RuntimeError("candidate changed the frozen HBT architecture")
    if list(baseline_metadata.get("train_names", [])) != edge_train[:2048]:
        raise RuntimeError("baseline training-source provenance mismatch")
    if list(candidate_metadata.get("train_names", [])) != edge_train[2048:4096]:
        raise RuntimeError("candidate training-source provenance mismatch")
    if list(candidate_metadata.get("val_names", [])) != all_names[:32]:
        raise RuntimeError("candidate selection-source provenance mismatch")
    if candidate_metadata.get("init_checkpoint_sha256") != BASELINE_SHA256:
        raise RuntimeError("candidate is not initialized from the frozen baseline")
    if list(candidate_training_report.get("train_names", [])) != edge_train[2048:4096]:
        raise RuntimeError("candidate report training-source provenance mismatch")
    if list(candidate_training_report.get("val_names", [])) != all_names[:32]:
        raise RuntimeError("candidate report selection-source provenance mismatch")
    if candidate_training_report.get("best_epoch") != candidate_metadata.get("epoch"):
        raise RuntimeError("candidate report/checkpoint best-epoch mismatch")
    if candidate_training_report.get("model_config") != candidate.config():
        raise RuntimeError("candidate report/checkpoint model-config mismatch")
    screen = selection_screen(candidate_training_report)
    baseline.eval()
    candidate.eval()
    target_root = exact_directory(DATA, "train/targets")
    records = []
    for panel in PANELS:
        for source_index, name in enumerate(names):
            with Image.open(target_root / name) as image:
                clean = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if clean.shape != (480, 480, 3):
                raise RuntimeError(f"unexpected target shape for {name}: {clean.shape}")
            for replica in REPLICAS:
                seed = per_source_seed(GATE_SEED, f"hbt-continuation-gate-{panel}", name, replica)
                exact = make_exact_panel(clean, panel=panel, seed=seed)
                denoised = restore_tiles_uint8(
                    restorer, exact.slot_tiles, device, batch_size=512
                )
                tensor = torch.from_numpy(
                    np.ascontiguousarray(denoised.transpose(0, 3, 1, 2))
                ).to(device=device, dtype=torch.float32)
                labels = direction_labels(exact.slot_to_target)
                with torch.inference_mode():
                    baseline_metrics = directional_metrics(baseline(tensor), labels)
                    candidate_metrics = directional_metrics(candidate(tensor), labels)
                records.append(
                    {
                        "panel": panel,
                        "source": name,
                        "source_index": source_index,
                        "replica": replica,
                        "seed": seed,
                        "slot_to_target_sha256": hashlib.sha256(exact.slot_to_target.tobytes()).hexdigest(),
                        "corrupted_tiles_sha256": hashlib.sha256(exact.slot_tiles.tobytes()).hexdigest(),
                        "denoised_tiles_sha256": hashlib.sha256(denoised.tobytes()).hexdigest(),
                        "baseline": baseline_metrics,
                        "candidate": candidate_metrics,
                    }
                )
                print(json.dumps({"event": "hbt_gate_record", "panel": panel, "source_index": source_index + 1, "source_count": len(names), "source": name, "replica": replica, "baseline_r1": baseline_metrics["combined"]["recall_at_1"], "candidate_r1": candidate_metrics["combined"]["recall_at_1"]}, sort_keys=True), flush=True)

    expected_records = len(PANELS) * len(names) * len(REPLICAS)
    if len(records) != expected_records:
        raise RuntimeError(f"incomplete comparator records: {len(records)} != {expected_records}")
    per_source = []
    aggregates: dict[str, dict[str, object]] = {}
    for panel_index, panel in enumerate(PANELS):
        panel_rows = []
        for name in names:
            replicas = [row for row in records if row["panel"] == panel and row["source"] == name]
            if len(replicas) != len(REPLICAS):
                raise RuntimeError(f"replica completeness failure for {panel}/{name}")
            deltas = {}
            for direction in ("combined", "right", "down"):
                deltas[direction] = {
                    metric: float(np.mean([
                        row["candidate"][direction][metric] - row["baseline"][direction][metric]
                        for row in replicas
                    ]))
                    for metric in METRICS
                }
            row = {"panel": panel, "source": name, "deltas": deltas}
            per_source.append(row)
            panel_rows.append(row)
        panel_aggregate: dict[str, object] = {}
        for direction in ("combined", "right", "down"):
            panel_aggregate[direction] = {}
            for metric_index, metric in enumerate(METRICS):
                values = np.asarray([row["deltas"][direction][metric] for row in panel_rows])
                panel_aggregate[direction][metric] = bootstrap_ci(
                    values,
                    seed=GATE_SEED + panel_index * 100 + metric_index * 10 + (0 if direction == "combined" else 1 if direction == "right" else 2),
                )
        aggregates[panel] = panel_aggregate

    independent = aggregates["independent_libjpeg"]
    primary = aggregates["primary_kornia"]
    gate_values = {
        "independent_r1_mean": (independent["combined"]["recall_at_1"]["mean"], 0.010, ">="),
        "independent_r1_lcb": (independent["combined"]["recall_at_1"]["lower_95"], 0.0, ">"),
        "independent_mrr_mean": (independent["combined"]["mrr"]["mean"], 0.010, ">="),
        "primary_r1_mean": (primary["combined"]["recall_at_1"]["mean"], 0.0, ">="),
        "primary_r1_lcb": (primary["combined"]["recall_at_1"]["lower_95"], -0.005, ">"),
    }
    for panel in PANELS:
        for direction in ("right", "down"):
            gate_values[f"{panel}_{direction}_r1_mean"] = (
                aggregates[panel][direction]["recall_at_1"]["mean"], -0.005, ">="
            )
        gate_values[f"{panel}_r32_mean"] = (
            aggregates[panel]["combined"]["recall_at_32"]["mean"], -0.005, ">="
        )
    gates = {
        name: {
            "value": float(value),
            "threshold": float(threshold),
            "operator": operator,
            "passed": bool(value >= threshold if operator == ">=" else value > threshold),
        }
        for name, (value, threshold, operator) in gate_values.items()
    }
    gates["selection_screen"] = {"passed": bool(screen["passed"]), "details": screen}
    passed = all(gate["passed"] for gate in gates.values())

    report = {
        "schema_version": 1,
        "kind": "hbt_continuation_untouched_retrieval_gate",
        "status": "passed_retrieval_gate" if passed else "failed_retrieval_gate",
        "safe_for_submission": False,
        "continue_to_qap_gate": passed,
        "anti_leakage": {
            "models_accept_target": False,
            "same_corrupted_and_denoised_tiles_for_both_models": True,
            "replicas_averaged_inside_whole_source_before_bootstrap": True,
            "statistical_unit": "whole_source",
            "panels_not_counted_as_independent_samples": True,
        },
        "configuration": {
            "split": "edge_development",
            "offset": GATE_OFFSET,
            "sources": GATE_SOURCES,
            "source_names": names,
            "source_names_sha256": names_sha256(names),
            "panels": list(PANELS),
            "replicas": list(REPLICAS),
            "seed": GATE_SEED,
        },
        "artifacts": {
            "baseline_sha256": BASELINE_SHA256,
            "candidate_sha256": CANDIDATE_SHA256,
            "candidate_report_sha256": CANDIDATE_REPORT_SHA256,
            "denoiser_sha256": DENOISER_SHA256,
            "code": code_provenance,
        },
        "probe": probe,
        "denoiser_metadata": denoiser_metadata,
        "baseline_metadata": baseline_metadata,
        "candidate_metadata": candidate_metadata,
        "selection_screen": screen,
        "aggregates": aggregates,
        "gates": gates,
        "per_source": per_source,
        "records": records,
        "elapsed_seconds": time.time() - started,
    }
    report_path = OUTPUT / "hbt_continuation_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    hashes_path = OUTPUT / "SHA256SUMS.txt"
    hashes_path.write_text(f"{sha256(report_path)}  {report_path.name}\n", encoding="utf-8")
    print(json.dumps({"event": "hbt_continuation_gate_complete", "status": report["status"], "report": str(report_path), "report_sha256": sha256(report_path), "continue_to_qap_gate": passed}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
