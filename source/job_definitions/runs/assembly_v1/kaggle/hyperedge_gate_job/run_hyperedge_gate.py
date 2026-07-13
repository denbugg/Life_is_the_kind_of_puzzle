#!/usr/bin/env python3
"""Train H0 on T4x2, run an exact8/real16 gate, and hash every artifact."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any


INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")


# Single bounded experiment configuration.  Keep the full job below ~1 hour.
CONFIG: dict[str, Any] = {
    "seed": 20260711,
    "train_sources": 64,
    "val_sources": 8,
    "epochs": 2,
    "train_timeout_seconds": 2700,
    "eval_timeout_seconds": 1200,
    "top_k": 8,
    "max_per_anchor": 4,
    "positives_per_source": 96,
    "negative_ratio": 3,
    "target_precision": 0.90,
    "max_hyperedges": 64,
    "displacement_weight": 0.35,
    "qap_iterations": 25,
    "qap_restarts": 2,
    "qap_boundary_weight": 0.05,
    "authoritative_v2_real16_baseline_ssim": 0.18281991502795386,
    "authoritative_v2_report_sha256": "cc1b694b1501ba9b02e5618ad838e155ae40af7990bbbf4542b281fc21adec60",
    "authoritative_v2_metric_path": "macro.qap_softcycle_l1_k8__denoised_render.predicted_layout_ssim",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def single(paths: list[Path], label: str) -> Path:
    candidates = sorted(set(path.resolve() for path in paths))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one {label}, found {candidates}")
    return candidates[0]


def find_data_root() -> Path:
    return single(
        [
            path.parent.parent
            for path in INPUT.glob("**/train/inputs")
            if path.is_dir() and (path.parent / "targets").is_dir()
        ],
        "puzzle data root",
    )


def find_runtime_root() -> Path:
    return single(
        [
            path.parent
            for path in INPUT.glob("**/selected_tilenaf_synth_50k.pt")
            if (path.parent / "hbt_d320_denoised_rgb_sobel.pt").is_file()
        ],
        "runtime checkpoint root",
    )


def valid_code_root(path: Path) -> bool:
    return (
        (path / "src" / "puzzle_assembly" / "hyperedge.py").is_file()
        and (path / "scripts" / "train_hyperedge_verifier.py").is_file()
        and (path / "scripts" / "evaluate_hyperedge_solver.py").is_file()
    )


def find_code_root() -> Path:
    for preferred in (
        INPUT / "datasets" / "pasha883" / "vsos-solver-rework-night-code",
        INPUT / "vsos-solver-rework-night-code",
    ):
        if valid_code_root(preferred):
            return preferred.resolve()
    return single(
        [
            path.parent.parent.parent
            for path in INPUT.glob("**/src/puzzle_assembly/hyperedge.py")
            if valid_code_root(path.parent.parent.parent)
        ],
        "hyperedge code root",
    )


def hardware_probe() -> dict[str, Any]:
    subprocess.run(["nvidia-smi"], check=False)
    import torch

    result = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "capabilities": [
            list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())
        ],
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "tensor_probe_means": [],
    }
    if not result["cuda_available"] or result["device_count"] < 2:
        raise RuntimeError(f"hyperedge gate requires two CUDA GPUs: {result}")
    if any("T4" not in name.upper() for name in result["devices"][:2]):
        raise RuntimeError(f"hyperedge gate requires T4x2: {result['devices']}")
    for index in range(2):
        left = torch.randn(128, 128, device=f"cuda:{index}")
        right = torch.randn(128, 128, device=f"cuda:{index}")
        result["tensor_probe_means"].append(float((left @ right).mean().item()))
    return result


def run_and_tee(
    name: str,
    command: list[str],
    *,
    environment: dict[str, str],
    code_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    log = WORKING / f"{name}.log"
    started = time.perf_counter()
    print(json.dumps({"event": "start", "name": name, "command": command}), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=code_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        timed_out = threading.Event()

        def terminate_on_timeout() -> None:
            if process.poll() is None:
                timed_out.set()
                process.kill()

        timer = threading.Timer(timeout_seconds, terminate_on_timeout)
        timer.daemon = True
        timer.start()
        assert process.stdout is not None
        try:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(f"[{name}] {line}", end="", flush=True)
            returncode = process.wait()
        finally:
            timer.cancel()
        if timed_out.is_set():
            raise TimeoutError(f"{name} exceeded {timeout_seconds} seconds")
    record = {
        "name": name,
        "returncode": returncode,
        "seconds": time.perf_counter() - started,
        "log": str(log),
        "log_sha256": sha256(log),
    }
    print(json.dumps({"event": "complete", **record}, sort_keys=True), flush=True)
    if returncode != 0:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(f"{name} failed with {returncode}:\n{tail}")
    return record


def weighted_mean(shards: list[dict[str, Any]], path: tuple[str, ...], weight_key: str) -> float:
    values = []
    weights = []
    for shard in shards:
        node: Any = shard
        for key in path:
            node = node[key]
        values.append(float(node))
        weights.append(float(shard[weight_key]["source_count"]))
    return float(sum(value * weight for value, weight in zip(values, weights, strict=True)) / sum(weights))


def main() -> None:
    started = time.perf_counter()
    data_root = find_data_root()
    runtime_root = find_runtime_root()
    code_root = find_code_root()
    probe = hardware_probe()
    print(json.dumps({"event": "hardware", **probe}, sort_keys=True), flush=True)
    denoiser = runtime_root / "selected_tilenaf_synth_50k.pt"
    embedding = runtime_root / "hbt_d320_denoised_rgb_sobel.pt"
    manifest = code_root / "configs" / "denoise_splits_seed20260710.json"
    quarantine = code_root / "configs" / "denoise_validation_quarantine_v1.json"
    checkpoint = WORKING / "hyperedge_h0.pt"
    training_report = WORKING / "hyperedge_h0_training.json"

    base_environment = os.environ.copy()
    base_environment["PYTHONPATH"] = str(code_root / "src")
    base_environment["PYTHONHASHSEED"] = "0"
    base_environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    training_environment = base_environment.copy()
    training_environment["CUDA_VISIBLE_DEVICES"] = "0,1"
    training_command = [
        sys.executable,
        str(code_root / "scripts" / "train_hyperedge_verifier.py"),
        "--data-root", str(data_root),
        "--denoiser", str(denoiser),
        "--embedding-checkpoint", str(embedding),
        "--manifest", str(manifest),
        "--quarantine", str(quarantine),
        "--train-sources", str(CONFIG["train_sources"]),
        "--val-sources", str(CONFIG["val_sources"]),
        "--epochs", str(CONFIG["epochs"]),
        "--seed", str(CONFIG["seed"]),
        "--device", "cuda",
        "--data-parallel",
        "--top-k", str(CONFIG["top_k"]),
        "--max-per-anchor", str(CONFIG["max_per_anchor"]),
        "--positives-per-source", str(CONFIG["positives_per_source"]),
        "--negative-ratio", str(CONFIG["negative_ratio"]),
        "--target-precision", str(CONFIG["target_precision"]),
        "--max-hyperedges", str(CONFIG["max_hyperedges"]),
        "--output", str(checkpoint),
        "--report", str(training_report),
        "--overwrite",
    ]
    training_record = run_and_tee(
        "hyperedge_training",
        training_command,
        environment=training_environment,
        code_root=code_root,
        timeout_seconds=float(CONFIG["train_timeout_seconds"]),
    )
    if not checkpoint.is_file() or not training_report.is_file():
        raise RuntimeError("training did not produce checkpoint and report")

    specs = [
        {
            "name": "hyperedge_gate_primary",
            "gpu": 0,
            "panel": "primary_kornia",
            "exact_offset": 64,
            "real_offset": 0,
        },
        {
            "name": "hyperedge_gate_independent",
            "gpu": 1,
            "panel": "independent_libjpeg",
            "exact_offset": 68,
            "real_offset": 8,
        },
    ]

    def run_eval(spec: dict[str, Any]) -> dict[str, Any]:
        output = WORKING / f"{spec['name']}.json"
        environment = base_environment.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
        command = [
            sys.executable,
            str(code_root / "scripts" / "evaluate_hyperedge_solver.py"),
            "--data-root", str(data_root),
            "--denoiser", str(denoiser),
            "--embedding-checkpoint", str(embedding),
            "--hyperedge-checkpoint", str(checkpoint),
            "--manifest", str(manifest),
            "--quarantine", str(quarantine),
            "--exact-panel", str(spec["panel"]),
            "--exact-offset", str(spec["exact_offset"]),
            "--exact-sources", "4",
            "--real-offset", str(spec["real_offset"]),
            "--real-sources", "8",
            "--seed", str(CONFIG["seed"]),
            "--device", "cuda",
            "--candidate-top-k", str(CONFIG["top_k"]),
            "--max-per-anchor", str(CONFIG["max_per_anchor"]),
            "--max-hyperedges", str(CONFIG["max_hyperedges"]),
            "--displacement-weight", str(CONFIG["displacement_weight"]),
            "--qap-iterations", str(CONFIG["qap_iterations"]),
            "--qap-restarts", str(CONFIG["qap_restarts"]),
            "--qap-boundary-weight", str(CONFIG["qap_boundary_weight"]),
            "--output", str(output),
            "--overwrite",
        ]
        record = run_and_tee(
            str(spec["name"]),
            command,
            environment=environment,
            code_root=code_root,
            timeout_seconds=float(CONFIG["eval_timeout_seconds"]),
        )
        if not output.is_file():
            raise RuntimeError(f"missing evaluation output: {output}")
        record.update(
            {
                "gpu": spec["gpu"],
                "panel": spec["panel"],
                "output": str(output),
                "output_sha256": sha256(output),
                "payload": json.loads(output.read_text(encoding="utf-8")),
            }
        )
        return record

    with ThreadPoolExecutor(max_workers=2) as executor:
        evaluation_records = list(executor.map(run_eval, specs))
    shards = [record["payload"] for record in evaluation_records]
    exact_names = [
        name for shard in shards for name in shard["source_lists"]["exact"]
    ]
    real_names = [name for shard in shards for name in shard["source_lists"]["real"]]
    if len(exact_names) != 8 or len(set(exact_names)) != 8:
        raise RuntimeError("exact gate is not eight distinct whole sources")
    if len(real_names) != 16 or len(set(real_names)) != 16:
        raise RuntimeError("real gate is not sixteen distinct whole sources")
    training_payload = json.loads(training_report.read_text(encoding="utf-8"))
    fitting_names = set(training_payload["train_names"]) | set(
        training_payload["val_names"]
    )
    if fitting_names.intersection(exact_names):
        raise RuntimeError("exact gate overlaps hyperedge train/calibration sources")

    accepted = sum(int(shard["exact"]["hyperedge_accepted"]) for shard in shards)
    correct = sum(int(shard["exact"]["hyperedge_correct"]) for shard in shards)
    precision = correct / accepted if accepted else 1.0
    coverage = weighted_mean(shards, ("exact", "coverage"), "exact")
    exact_baseline = weighted_mean(
        shards, ("exact", "baseline_layout", "combined_adjacency"), "exact"
    )
    exact_candidate = weighted_mean(
        shards, ("exact", "candidate_layout", "combined_adjacency"), "exact"
    )
    real_baseline = weighted_mean(
        shards, ("real", "baseline_image", "predicted_layout_ssim"), "real"
    )
    real_candidate = weighted_mean(
        shards, ("real", "candidate_image", "predicted_layout_ssim"), "real"
    )
    gates = {
        "authoritative_v2_baseline_reproduced": abs(
            real_baseline - float(CONFIG["authoritative_v2_real16_baseline_ssim"])
        ) <= 1e-6,
        "precision_at_least_0_90": precision >= 0.90,
        "coverage_at_least_0_15": coverage >= 0.15,
        "exact8_adjacency_gain_at_least_0_03": exact_candidate - exact_baseline >= 0.03,
        "real16_ssim_gain_at_least_0_015": real_candidate - real_baseline >= 0.015,
    }
    wrapper = {
        "schema_version": 1,
        "kind": "puzzle_hyperedge_h0_t4x2_gate",
        "config": CONFIG,
        "probe": probe,
        "mounts": {
            "data_root": str(data_root),
            "runtime_root": str(runtime_root),
            "code_root": str(code_root),
        },
        "input_hashes": {
            "denoiser": sha256(denoiser),
            "embedding_checkpoint": sha256(embedding),
            "manifest": sha256(manifest),
            "quarantine": sha256(quarantine),
            "hyperedge_module": sha256(
                code_root / "src" / "puzzle_assembly" / "hyperedge.py"
            ),
            "trainer": sha256(
                code_root / "scripts" / "train_hyperedge_verifier.py"
            ),
            "evaluator": sha256(
                code_root / "scripts" / "evaluate_hyperedge_solver.py"
            ),
            "qap_module": sha256(code_root / "src" / "puzzle_assembly" / "qap.py"),
        },
        "training": {
            **training_record,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "report": str(training_report),
            "report_sha256": sha256(training_report),
            "gate_source_overlap": 0,
        },
        "evaluation_records": [
            {key: value for key, value in record.items() if key != "payload"}
            for record in evaluation_records
        ],
        "exact8": {
            "source_names": exact_names,
            "accepted": accepted,
            "correct": correct,
            "precision": precision,
            "coverage": coverage,
            "baseline_adjacency": exact_baseline,
            "candidate_adjacency": exact_candidate,
            "adjacency_gain": exact_candidate - exact_baseline,
        },
        "real16": {
            "source_names": real_names,
            "baseline_ssim": real_baseline,
            "candidate_ssim": real_candidate,
            "ssim_gain": real_candidate - real_baseline,
            "authoritative_v2_baseline_ssim": CONFIG[
                "authoritative_v2_real16_baseline_ssim"
            ],
            "authoritative_v2_report_sha256": CONFIG[
                "authoritative_v2_report_sha256"
            ],
            "authoritative_v2_metric_path": CONFIG[
                "authoritative_v2_metric_path"
            ],
            "authoritative_v2_report_local_path": "runs/assembly_v1/kaggle/qap_tuning_night_output/v2/qap_l1w4_boundary_real16.json",
            "baseline_reproduction_delta": real_baseline
            - float(CONFIG["authoritative_v2_real16_baseline_ssim"]),
            "target_pixels_used_for_layout_selection": False,
        },
        "gates": gates,
        "accepted": bool(all(gates.values())),
        "known_risks": [
            "synthetic exact degradation may not cover every real false-cycle texture",
            "the 90-percent precision threshold can miss the required 15-percent coverage",
            "greedy absolute placement of correct relative anchors can still disrupt QAP global structure",
            "GPU0 performs denoiser and candidate preprocessing, so DataParallel scaling is limited",
            "exact8 and real16 are intentionally small bounded gates and retain sampling variance",
        ],
        "seconds": time.perf_counter() - started,
    }
    report = WORKING / "hyperedge_gate_report.json"
    report.write_text(json.dumps(wrapper, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_paths = [
        checkpoint,
        training_report,
        report,
        *(Path(record["output"]) for record in evaluation_records),
        *(Path(record["log"]) for record in evaluation_records),
        Path(training_record["log"]),
    ]
    hashes = {
        "schema_version": 1,
        "kind": "puzzle_hyperedge_gate_artifact_hashes",
        "artifacts": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifact_paths
        ],
    }
    hash_path = WORKING / "hyperedge_gate_hashes.json"
    hash_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "hyperedge_gate_wrapper_complete",
                "accepted": wrapper["accepted"],
                "gates": gates,
                "report": str(report),
                "report_sha256": sha256(report),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
