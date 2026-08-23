"""Durable, E:-only autonomous stage runner for PAZZLE E26.

The runner is deliberately generic: the scientific E26 protocol freezes a JSON
specification, while this module supplies the operational guarantees needed to
keep a long experiment alive after the interactive Codex session disappears.

Key properties
--------------
* a canonical, content-addressed plan is frozen before execution;
* one exact stage command runs at a time under an exclusive PID lock;
* stdout/stderr, attempt heartbeats and resource usage live on E:;
* a stage is skipped only when its receipt and every output hash re-verify;
* dependencies, source files and explicit inputs are re-hashed before a stage;
* time/RAM/disk/aggregate CPU caps fail closed;
* every transition updates ``status.json`` and ``recovery_report.json``;
* scientific gate failure is terminal and never starts downstream stages.

Importing this module has no filesystem side effects.  Real execution refuses a
non-E work root.  Unit tests may call the Python API with ``allow_non_e=True``;
there is intentionally no corresponding production CLI switch.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import site
import socket
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SPEC_SCHEMA = "pazzle-e26-autonomous-spec-v1"
PLAN_SCHEMA = "pazzle-e26-autonomous-plan-v1"
ATTEMPT_SCHEMA = "pazzle-e26-autonomous-attempt-v1"
RECEIPT_SCHEMA = "pazzle-e26-autonomous-stage-receipt-v1"
STATUS_SCHEMA = "pazzle-e26-autonomous-status-v1"
REPORT_SCHEMA = "pazzle-e26-autonomous-recovery-report-v1"
FINAL_REPORT_SCHEMA = "pazzle-e26-autonomous-final-report-v1"
LOCK_SCHEMA = "pazzle-e26-autonomous-lock-v1"
EVENT_SCHEMA = "pazzle-e26-autonomous-event-v1"
LAUNCH_REQUEST_SCHEMA = "pazzle-e26-stage-launch-request-v1"
BOOTSTRAP_PATH = (Path(__file__).resolve().parent / "e26_stage_bootstrap.py").resolve()
LAUNCHER_PATH = (Path(__file__).resolve().parent.parent / "launch_e26_autonomous.ps1").resolve()

HEX64 = re.compile(r"^[0-9a-f]{64}$")
STAGE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
INVOCATION_ID = re.compile(r"^[0-9a-f]{32}$")
PROGRESS_DEFAULT = re.compile(r"(?:step|item|image)=(?P<done>\d+)/(?:total=)?(?P<total>\d+)")
REQUIRED_E_ENV = (
    "TEMP",
    "TMP",
    "TMPDIR",
    "PYTHONPYCACHEPREFIX",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "JOBLIB_TEMP_FOLDER",
    "MPLCONFIGDIR",
    "CUDA_CACHE_PATH",
    "HOME",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PIP_CACHE_DIR",
    "NUMBA_CACHE_DIR",
    "TRITON_CACHE_DIR",
)
FROZEN_ENV_KEYS = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)
SUPERVISOR_POLL_SECONDS = 5.0
MAX_SUPERVISOR_POLL_SECONDS = 5.0
PREFLIGHT_DIRECTORY = "preflight"


class ContractError(RuntimeError):
    """The frozen plan, provenance, receipt, or artifact contract is invalid."""


class ResourceLimitError(RuntimeError):
    """A predeclared operational resource cap was crossed."""


class StageFailure(RuntimeError):
    """A stage process or verifier failed."""


class ScientificGateFailure(RuntimeError):
    """A completed scientific stage reached its predeclared terminal FAIL."""


class ProcessGoneError(ContractError):
    """The exact recorded process identity no longer exists."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_canonical_json(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    try:
        encoded = canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"JSON contains a non-canonical value: {path}: {exc}") from exc
    if raw != encoded:
        raise ContractError(f"JSON is not byte-canonical: {path}")
    return value


def _atomic_write_canonical(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(dict(value))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    delay = 0.002
    for attempt in range(8):
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            return
        except PermissionError:
            # Windows antivirus/indexers can briefly hold a just-written JSON
            # file.  Retry the *same bytes* for <0.3s; this is operational and
            # cannot change a scientific stage or artifact identity.
            if attempt == 7:
                raise
            if temporary.exists():
                temporary.unlink()
            time.sleep(delay)
            delay *= 2
        finally:
            if temporary.exists():
                temporary.unlink()


def _create_once_canonical(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(dict(value))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard-link publication is atomic and, unlike os.replace, never
        # overwrites an existing create-once identity.  The temp file lives in
        # the same directory/volume, so no cross-device case exists.
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ContractError(f"create-once artifact already exists: {path}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolved(path: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    result = Path(path)
    if not result.is_absolute():
        if base is None:
            raise ContractError(f"path must be absolute: {path}")
        result = base / result
    return result.resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _require_e_path(path: Path, label: str, *, allow_non_e: bool) -> None:
    path = Path(path).resolve()
    if allow_non_e:
        return
    if path.drive.upper() != "E:":
        raise ContractError(f"{label} must be on E:, got {path}")


def _path_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise ContractError(f"required file is missing: {path}")
    size = path.stat().st_size
    return {"path": str(path), "bytes": int(size), "sha256": sha256_file(path)}


def _verify_path_record(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if set(record) != {"path", "bytes", "sha256"}:
        raise ContractError(f"{label} has invalid path-record keys")
    path = _resolved(str(record["path"]))
    actual = _path_record(path)
    if actual != dict(record):
        raise ContractError(f"{label} hash/size drift: {path}")
    return actual


def _runtime_record(python_executable: Path) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy", "torch", "lightgbm", "scipy", "scikit-image", "opencv-python", "psutil"
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python_executable": str(Path(python_executable).resolve()),
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _default_e_environment(work_root: Path) -> dict[str, str]:
    runtime = work_root / "runtime"
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "2601",
        "PYTHONPATH": "",
        # Read the already-installed, runtime-hashed user packages from C:;
        # PYTHONPYCACHEPREFIX and every writable cache/profile remain on E:.
        "PYTHONUSERBASE": str(Path(site.getuserbase()).resolve()),
        "PAZZLE_DATA": r"E:\pazzle_data",
        "PAZZLE_WORK": str(work_root),
        "TEMP": str(runtime / "tmp"),
        "TMP": str(runtime / "tmp"),
        "TMPDIR": str(runtime / "tmp"),
        "PYTHONPYCACHEPREFIX": str(runtime / "pycache"),
        "TORCH_HOME": str(runtime / "torch_home"),
        "TORCH_EXTENSIONS_DIR": str(runtime / "torch_extensions"),
        "XDG_CACHE_HOME": str(runtime / "xdg_cache"),
        "HF_HOME": str(runtime / "hf_home"),
        "JOBLIB_TEMP_FOLDER": str(runtime / "joblib"),
        "MPLCONFIGDIR": str(runtime / "mpl"),
        "CUDA_CACHE_PATH": str(runtime / "cuda_cache"),
        "HOME": str(runtime / "home"),
        "USERPROFILE": str(runtime / "home"),
        "APPDATA": str(runtime / "appdata"),
        "LOCALAPPDATA": str(runtime / "local_appdata"),
        "PIP_CACHE_DIR": str(runtime / "pip_cache"),
        "NUMBA_CACHE_DIR": str(runtime / "numba_cache"),
        "TRITON_CACHE_DIR": str(runtime / "triton_cache"),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    # Subprocesses receive this exact mapping rather than os.environ.copy().
    # Freeze only the small OS/toolchain allowlist needed to start Python/CUDA;
    # all other ambient variables are deliberately absent.
    for name in FROZEN_ENV_KEYS:
        value = os.environ.get(name)
        if value is not None:
            environment[name] = value
    return environment


def _canonicalize_environment(environment: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    casefolded: dict[str, str] = {}
    for key, value in environment.items():
        if (
            not isinstance(key, str) or not isinstance(value, str)
            or not key or "=" in key or "\x00" in key or "\x00" in value
        ):
            raise ContractError(f"environment contains an invalid name/value: {key!r}")
        folded = key.casefold()
        if folded in casefolded and casefolded[folded] != key:
            raise ContractError(
                f"environment has a Windows case-insensitive collision: "
                f"{casefolded[folded]!r} versus {key!r}"
            )
        casefolded[folded] = key
        canonical_key = key.upper() if os.name == "nt" else key
        normalized[canonical_key] = value
    return dict(sorted(normalized.items()))


def _validate_stage_spec(stage: Mapping[str, Any], *, work_root: Path, repo_root: Path,
                         allow_non_e: bool) -> dict[str, Any]:
    allowed = {
        "name", "argv", "resume_argv", "dependencies", "inputs", "outputs",
        "verifier_argv", "working_directory", "timeout_seconds", "max_rss_bytes",
        "min_free_bytes", "max_attempts", "progress_regex", "gate", "resource_class",
    }
    extra = set(stage) - allowed
    if extra:
        raise ContractError(f"unknown stage keys: {sorted(extra)}")
    name = stage.get("name")
    if not isinstance(name, str) or not STAGE_NAME.fullmatch(name):
        raise ContractError(f"invalid stage name: {name!r}")
    if name.endswith("_verifier"):
        raise ContractError("stage names ending _verifier are reserved by the runner")
    argv = stage.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
        raise ContractError(f"stage {name} argv must be a non-empty string list")
    resume_argv = stage.get("resume_argv")
    if resume_argv is not None and (
        not isinstance(resume_argv, list)
        or not resume_argv
        or any(not isinstance(x, str) or not x for x in resume_argv)
    ):
        raise ContractError(f"stage {name} resume_argv is invalid")
    dependencies = stage.get("dependencies", [])
    if not isinstance(dependencies, list) or any(not isinstance(x, str) for x in dependencies):
        raise ContractError(f"stage {name} dependencies must be a string list")
    working_directory = _resolved(stage.get("working_directory", str(repo_root)), base=repo_root)
    if not _is_within(working_directory, work_root):
        raise ContractError(f"stage {name} working directory must be inside E26 work root")
    inputs: list[dict[str, Any]] = []
    for raw in stage.get("inputs", []):
        if not isinstance(raw, Mapping) or set(raw) != {"path", "bytes", "sha256"}:
            raise ContractError(
                f"stage {name} input record must contain exact path/bytes/sha256"
            )
        path = _resolved(str(raw.get("path", "")), base=repo_root)
        record: dict[str, Any] = {
            "path": str(path),
            "bytes": int(raw["bytes"]),
            "sha256": str(raw["sha256"]),
        }
        if record["bytes"] < 0 or not HEX64.fullmatch(record["sha256"]):
            raise ContractError(f"stage {name} input size/SHA is invalid")
        inputs.append(record)
    outputs: list[dict[str, Any]] = []
    output_paths: set[str] = set()
    for raw in stage.get("outputs", []):
        if not isinstance(raw, Mapping) or set(raw) - {"path", "min_bytes", "max_bytes"}:
            raise ContractError(f"stage {name} output record is invalid")
        path = _resolved(str(raw.get("path", "")), base=work_root)
        _require_e_path(path, f"stage {name} output", allow_non_e=allow_non_e)
        if not _is_within(path, work_root):
            raise ContractError(f"stage {name} output escapes work root: {path}")
        for reserved in ("orchestrator", "runtime", "preflight"):
            if _is_within(path, work_root / reserved):
                raise ContractError(
                    f"stage {name} output overlaps reserved {reserved} state: {path}"
                )
        if str(path) in output_paths:
            raise ContractError(f"stage {name} repeats output: {path}")
        output_paths.add(str(path))
        item = {"path": str(path), "min_bytes": int(raw.get("min_bytes", 1))}
        if "max_bytes" in raw:
            item["max_bytes"] = int(raw["max_bytes"])
        if item["min_bytes"] < 0 or item.get("max_bytes", item["min_bytes"]) < item["min_bytes"]:
            raise ContractError(f"stage {name} output byte bounds are invalid")
        outputs.append(item)
    verifier = stage.get("verifier_argv", [])
    if not isinstance(verifier, list) or any(not isinstance(x, str) or not x for x in verifier):
        raise ContractError(f"stage {name} verifier_argv must be a string list")
    timeout = int(stage.get("timeout_seconds", 0))
    max_rss = int(stage.get("max_rss_bytes", 0))
    min_free = int(stage.get("min_free_bytes", 1 << 30))
    max_attempts = int(stage.get("max_attempts", 1))
    if timeout <= 0 or max_rss <= 0 or min_free < 0 or max_attempts <= 0:
        raise ContractError(f"stage {name} resource/retry bounds must be positive")
    resource_class = str(stage.get("resource_class", "cpu"))
    if resource_class not in {"cpu", "gpu"}:
        raise ContractError(f"stage {name} resource_class must be cpu or gpu")
    progress_regex = str(stage.get("progress_regex", PROGRESS_DEFAULT.pattern))
    try:
        progress_compiled = re.compile(progress_regex)
    except re.error as exc:
        raise ContractError(f"stage {name} progress regex is invalid: {exc}") from exc
    if not {"done", "total"}.issubset(progress_compiled.groupindex):
        raise ContractError(f"stage {name} progress regex needs named done/total groups")
    gate = stage.get("gate")
    if gate is not None:
        if not isinstance(gate, Mapping) or set(gate) != {"path", "pointer", "pass_value"}:
            raise ContractError(f"stage {name} gate must contain path/pointer/pass_value")
        gate_path = _resolved(str(gate["path"]), base=work_root)
        if str(gate_path) not in output_paths:
            raise ContractError(f"stage {name} gate path must be a declared output")
        pointer = gate["pointer"]
        if not isinstance(pointer, list) or not pointer or any(not isinstance(x, str) for x in pointer):
            raise ContractError(f"stage {name} gate pointer must be a non-empty string list")
        gate = {"path": str(gate_path), "pointer": list(pointer), "pass_value": gate["pass_value"]}
    return {
        "name": name,
        "argv": list(argv),
        "resume_argv": list(resume_argv) if resume_argv is not None else None,
        "dependencies": list(dependencies),
        "inputs": inputs,
        "outputs": outputs,
        "verifier_argv": list(verifier),
        "working_directory": str(working_directory),
        "timeout_seconds": timeout,
        "max_rss_bytes": max_rss,
        "min_free_bytes": min_free,
        "max_attempts": max_attempts,
        "resource_class": resource_class,
        "progress_regex": progress_regex,
        "gate": gate,
    }


def _validate_python_commands(stages: Sequence[Mapping[str, Any]], python_executable: Path) -> None:
    expected = Path(python_executable).resolve()
    for stage in stages:
        for field in ("argv", "resume_argv", "verifier_argv"):
            command = stage.get(field)
            if not command:
                continue
            if len(command) < 3:
                raise ContractError(f"stage {stage['name']} {field} is too short")
            try:
                executable = Path(command[0]).resolve()
            except (OSError, TypeError) as exc:
                raise ContractError(f"stage {stage['name']} {field} executable is invalid") from exc
            if executable != expected or command[1] != "-B":
                raise ContractError(
                    f"stage {stage['name']} {field} must begin with exact frozen Python and -B"
                )


def _validate_command_sources(
    stages: Sequence[Mapping[str, Any]], source_records: Sequence[Mapping[str, Any]],
    repo_root: Path,
) -> None:
    frozen_paths = {str(Path(record["path"]).resolve()) for record in source_records}
    for stage in stages:
        for field in ("argv", "resume_argv", "verifier_argv"):
            command = stage.get(field)
            if not command:
                continue
            entry = command[2]
            if entry == "-m":
                raise ContractError(
                    f"stage {stage['name']} {field} cannot use -m; exact entry script must be frozen"
                )
            script = _resolved(entry)
            if script.suffix.lower() != ".py" or str(script) not in frozen_paths:
                raise ContractError(
                    f"stage {stage['name']} {field} entry script is not a frozen source: {script}"
                )
            pending = [script]
            visited: set[str] = set()
            while pending:
                current = pending.pop()
                if str(current) in visited:
                    continue
                visited.add(str(current))
                try:
                    tree = ast.parse(current.read_text(encoding="utf-8"), filename=str(current))
                except (OSError, SyntaxError, UnicodeError) as exc:
                    raise ContractError(f"cannot statically inspect source closure: {current}") from exc
                modules: list[tuple[str, int]] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules.extend((alias.name, 0) for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            modules.append((node.module, int(node.level)))
                        elif node.level:
                            modules.extend((alias.name, int(node.level)) for alias in node.names)
                for module, level in modules:
                    pieces = module.split(".")
                    candidates: list[Path] = []
                    if level:
                        base = current.parent
                        for _ in range(max(0, level - 1)):
                            base = base.parent
                        candidates.extend((
                            base.joinpath(*pieces).with_suffix(".py"),
                            base.joinpath(*pieces, "__init__.py"),
                        ))
                    else:
                        for base in (repo_root, repo_root / "src"):
                            candidates.extend((
                                base.joinpath(*pieces).with_suffix(".py"),
                                base.joinpath(*pieces, "__init__.py"),
                            ))
                    local = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
                    if local is None:
                        continue
                    if str(local) not in frozen_paths:
                        raise ContractError(
                            f"local import closure is not frozen: {current} -> {local}"
                        )
                    pending.append(local)


def freeze_plan(spec_path: Path, plan_path: Path, *, allow_non_e: bool = False) -> dict[str, Any]:
    """Freeze a canonical content-addressed execution plan from a JSON spec."""

    spec_path = Path(spec_path).resolve()
    raw = spec_path.read_bytes()
    try:
        spec = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid spec JSON: {exc}") from exc
    if not isinstance(spec, Mapping) or spec.get("schema") != SPEC_SCHEMA:
        raise ContractError(f"spec schema must be {SPEC_SCHEMA}")
    allowed = {
        "schema", "pipeline_id", "repo_root", "work_root", "python_executable",
        "source_files", "environment", "global_caps", "stages",
    }
    extra = set(spec) - allowed
    if extra:
        raise ContractError(f"unknown spec keys: {sorted(extra)}")
    pipeline_id = spec.get("pipeline_id")
    if not isinstance(pipeline_id, str) or not STAGE_NAME.fullmatch(pipeline_id):
        raise ContractError("pipeline_id must use lowercase stage-name syntax")
    repo_root = _resolved(str(spec.get("repo_root", spec_path.parent)))
    work_root = _resolved(str(spec.get("work_root", "")))
    plan_path = Path(plan_path).resolve()
    _require_e_path(work_root, "work_root", allow_non_e=allow_non_e)
    _require_e_path(plan_path, "plan_path", allow_non_e=allow_non_e)
    preflight_root = (work_root / PREFLIGHT_DIRECTORY).resolve()
    if not _is_within(plan_path, preflight_root):
        raise ContractError("plan_path must be inside the reserved work_root/preflight directory")
    python_executable = _resolved(str(spec.get("python_executable", sys.executable)))
    if not python_executable.is_file():
        raise ContractError(f"Python executable is missing: {python_executable}")
    if python_executable != Path(sys.executable).resolve():
        raise ContractError(
            "freeze must run under the same exact Python executable recorded in the plan"
        )
    sources_raw = spec.get("source_files")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ContractError("source_files must be a non-empty list")
    source_records: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for item in sources_raw:
        if not isinstance(item, str):
            raise ContractError("source_files entries must be strings")
        path = _resolved(item, base=repo_root)
        if not _is_within(path, repo_root):
            raise ContractError(f"source escapes repo_root: {path}")
        if str(path) in seen_sources:
            raise ContractError(f"duplicate source: {path}")
        seen_sources.add(str(path))
        source_records.append(_path_record(path))
    source_records.sort(key=lambda row: row["path"])
    if not allow_non_e:
        required_infrastructure = {
            str(Path(__file__).resolve()), str(BOOTSTRAP_PATH), str(LAUNCHER_PATH),
        }
        frozen_sources = {record["path"] for record in source_records}
        missing_infrastructure = required_infrastructure - frozen_sources
        if missing_infrastructure:
            raise ContractError(
                f"source_files omit autonomous infrastructure: {sorted(missing_infrastructure)}"
            )
    source_digest = canonical_digest(source_records)
    environment = _default_e_environment(work_root)
    custom_env = spec.get("environment", {})
    if not isinstance(custom_env, Mapping) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in custom_env.items()
    ):
        raise ContractError("environment must be a string mapping")
    environment.update(custom_env)
    environment = _canonicalize_environment(environment)
    for name in REQUIRED_E_ENV:
        value = _resolved(environment[name], base=work_root)
        _require_e_path(value, f"environment {name}", allow_non_e=allow_non_e)
        if not _is_within(value, work_root):
            raise ContractError(f"environment {name} escapes work_root")
        environment[name] = str(value)
    for name in ("PAZZLE_DATA", "PAZZLE_WORK"):
        value = _resolved(environment.get(name, ""), base=work_root)
        _require_e_path(value, f"environment {name}", allow_non_e=allow_non_e)
        if name == "PAZZLE_WORK" and value != work_root:
            raise ContractError("environment PAZZLE_WORK must equal the exact E26 work_root")
        environment[name] = str(value)
    if environment.get("PYTHONHASHSEED") != "2601":
        raise ContractError("environment PYTHONHASHSEED must be exactly 2601")
    if environment.get("PYTHONPATH") != "":
        raise ContractError("environment PYTHONPATH must be empty; local imports are source-frozen")
    python_user_base = _resolved(environment.get("PYTHONUSERBASE", ""), base=work_root)
    if not python_user_base.is_dir():
        raise ContractError("environment PYTHONUSERBASE must name the existing frozen package base")
    environment["PYTHONUSERBASE"] = str(python_user_base)
    if environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ContractError("environment CUBLAS_WORKSPACE_CONFIG must be exactly :4096:8")
    if not environment.get("PATH"):
        raise ContractError("frozen environment requires a non-empty PATH")
    global_caps_raw = spec.get("global_caps", {})
    if not isinstance(global_caps_raw, Mapping) or set(global_caps_raw) - {
        "max_cpu_seconds", "max_gpu_seconds", "max_wall_seconds", "max_artifact_bytes"
    }:
        raise ContractError("global_caps has invalid keys")
    global_caps = {
        "max_cpu_seconds": float(global_caps_raw.get("max_cpu_seconds", 48 * 3600)),
        "max_gpu_seconds": float(global_caps_raw.get("max_gpu_seconds", 72 * 3600)),
        "max_wall_seconds": float(global_caps_raw.get("max_wall_seconds", 7 * 24 * 3600)),
        "max_artifact_bytes": int(global_caps_raw.get("max_artifact_bytes", 48 << 30)),
    }
    float_cap_names = ("max_cpu_seconds", "max_gpu_seconds", "max_wall_seconds")
    if any(
        not math.isfinite(float(global_caps[name])) or float(global_caps[name]) <= 0
        for name in float_cap_names
    ) or int(global_caps["max_artifact_bytes"]) <= 0:
        raise ContractError("global caps must be positive")
    stages_raw = spec.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ContractError("stages must be a non-empty list")
    stages = [
        _validate_stage_spec(stage, work_root=work_root, repo_root=repo_root,
                             allow_non_e=allow_non_e)
        for stage in stages_raw
    ]
    _validate_python_commands(stages, python_executable)
    _validate_command_sources(stages, source_records, repo_root)
    names = [stage["name"] for stage in stages]
    if len(names) != len(set(names)):
        raise ContractError("stage names must be unique")
    earlier: set[str] = set()
    all_outputs: set[str] = set()
    immutable_paths = {
        str(spec_path),
        str(plan_path),
        *(record["path"] for record in source_records),
        *(item["path"] for stage in stages for item in stage["inputs"]),
    }
    for stage in stages:
        unknown = set(stage["dependencies"]) - earlier
        if unknown:
            raise ContractError(f"stage {stage['name']} has non-earlier dependencies: {sorted(unknown)}")
        earlier.add(stage["name"])
        for output in stage["outputs"]:
            if output["path"] in all_outputs:
                raise ContractError(f"output path is owned by multiple stages: {output['path']}")
            if output["path"] in immutable_paths:
                raise ContractError(f"stage output overlaps immutable input/source/plan: {output['path']}")
            if Path(output["path"]).exists():
                raise ContractError(f"stage output already exists at plan freeze: {output['path']}")
            all_outputs.add(output["path"])
    runtime = _runtime_record(python_executable)
    if runtime["packages"].get("psutil") is None:
        raise ContractError("psutil is required for durable process-tree resource accounting")
    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "pipeline_id": pipeline_id,
        "spec": {"path": str(spec_path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
        "repo_root": str(repo_root),
        "work_root": str(work_root),
        "runtime": runtime,
        "environment": environment,
        "environment_sha256": canonical_digest(environment),
        "global_caps": global_caps,
        "sources": source_records,
        "sources_sha256": source_digest,
        "stages": stages,
    }
    # The immutable external identity is the SHA-256 of the exact canonical
    # file bytes.  Avoid a self-referential embedded digest (which cannot equal
    # the digest of the document containing itself); callers pin this value in
    # the CLI, lock, attempts, receipts, and reports.
    plan_sha256 = canonical_digest(payload)
    _create_once_canonical(plan_path, payload)
    return {**payload, "plan_sha256": plan_sha256}


def load_and_verify_plan(plan_path: Path, expected_sha256: str, *,
                         allow_non_e: bool = False) -> dict[str, Any]:
    if not HEX64.fullmatch(expected_sha256):
        raise ContractError("expected plan SHA-256 is invalid")
    plan_path = Path(plan_path).resolve()
    plan = _read_canonical_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ContractError("plan schema mismatch")
    if "plan_sha256" in plan:
        raise ContractError("plan must not contain a self-referential embedded SHA")
    file_sha256 = sha256_file(plan_path)
    if file_sha256 != expected_sha256 or canonical_digest(plan) != expected_sha256:
        raise ContractError("plan file/recomputed SHA-256 mismatch")
    work_root = _resolved(str(plan.get("work_root", "")))
    _require_e_path(work_root, "work_root", allow_non_e=allow_non_e)
    if not _is_within(plan_path, (work_root / PREFLIGHT_DIRECTORY).resolve()):
        raise ContractError("plan is outside its reserved work_root/preflight directory")
    sources = plan.get("sources")
    if not isinstance(sources, list) or canonical_digest(sources) != plan.get("sources_sha256"):
        raise ContractError("plan source-list digest mismatch")
    for index, record in enumerate(sources):
        if not isinstance(record, Mapping):
            raise ContractError("plan source record is invalid")
        _verify_path_record(record, label=f"source[{index}]")
    environment = plan.get("environment")
    if not isinstance(environment, Mapping):
        raise ContractError("plan environment is invalid")
    normalized_environment = _canonicalize_environment(environment)
    if normalized_environment != environment:
        raise ContractError("plan environment does not round-trip canonically")
    if canonical_digest(normalized_environment) != plan.get("environment_sha256"):
        raise ContractError("plan environment digest mismatch")
    for name in REQUIRED_E_ENV:
        if name not in environment:
            raise ContractError(f"plan environment is missing {name}")
        path = _resolved(str(environment[name]))
        _require_e_path(path, f"environment {name}", allow_non_e=allow_non_e)
        if not _is_within(path, work_root):
            raise ContractError(f"environment {name} escapes work_root")
    pazzle_data = _resolved(str(environment.get("PAZZLE_DATA", "")), base=work_root)
    pazzle_work = _resolved(str(environment.get("PAZZLE_WORK", "")), base=work_root)
    _require_e_path(pazzle_data, "environment PAZZLE_DATA", allow_non_e=allow_non_e)
    _require_e_path(pazzle_work, "environment PAZZLE_WORK", allow_non_e=allow_non_e)
    if pazzle_work != work_root:
        raise ContractError("environment PAZZLE_WORK differs from work_root")
    if environment.get("PYTHONHASHSEED") != "2601" or environment.get("PYTHONPATH") != "":
        raise ContractError("plan Python hash/import environment is not frozen")
    python_user_base = _resolved(str(environment.get("PYTHONUSERBASE", "")))
    if not python_user_base.is_dir():
        raise ContractError("plan PYTHONUSERBASE is missing")
    if environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ContractError("plan CUBLAS workspace contract is invalid")
    caps = plan.get("global_caps")
    if not isinstance(caps, Mapping) or set(caps) != {
        "max_cpu_seconds", "max_gpu_seconds", "max_wall_seconds", "max_artifact_bytes"
    }:
        raise ContractError("plan global caps are invalid")
    for name in ("max_cpu_seconds", "max_gpu_seconds", "max_wall_seconds"):
        value = float(caps[name])
        if not math.isfinite(value) or value <= 0:
            raise ContractError(f"plan global cap {name} is invalid")
    if isinstance(caps["max_artifact_bytes"], bool) or int(caps["max_artifact_bytes"]) <= 0:
        raise ContractError("plan max_artifact_bytes is invalid")
    # Re-run structural stage validation on the already-normalized plan.
    repo_root = _resolved(str(plan.get("repo_root", "")))
    stages = plan.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ContractError("plan has no stages")
    normalized = [
        _validate_stage_spec(stage, work_root=work_root, repo_root=repo_root,
                             allow_non_e=allow_non_e)
        for stage in stages
    ]
    _validate_python_commands(normalized, Path(plan["runtime"]["python_executable"]))
    _validate_command_sources(normalized, sources, repo_root)
    if normalized != stages:
        raise ContractError("plan stages do not round-trip canonically")
    return {**plan, "plan_sha256": expected_sha256}


def _self_digest_payload(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    payload = dict(value)
    payload[key] = canonical_digest(payload)
    return payload


def _verify_self_digest(value: Mapping[str, Any], key: str, *, label: str) -> None:
    stored = value.get(key)
    body = dict(value)
    body.pop(key, None)
    if not isinstance(stored, str) or stored != canonical_digest(body):
        raise ContractError(f"{label} self-digest mismatch")


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return int(exit_code.value) == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _process_identity(pid: int) -> dict[str, Any]:
    try:
        import psutil  # type: ignore
    except ImportError as exc:
        raise ContractError("psutil is required for exact process identity") from exc
    try:
        process = psutil.Process(int(pid))
        identity = {
            "pid": int(pid),
            "create_time": float(process.create_time()),
            "executable": str(Path(process.exe()).resolve()),
        }
    except (psutil.NoSuchProcess, psutil.ZombieProcess) as exc:
        raise ProcessGoneError(f"process PID {pid} no longer exists") from exc
    except psutil.AccessDenied as exc:
        raise ContractError(f"access denied while authenticating process PID {pid}") from exc
    except OSError as exc:
        raise ContractError(f"cannot capture process identity for PID {pid}: {exc}") from exc
    if (
        identity["pid"] <= 0
        or not math.isfinite(identity["create_time"])
        or identity["create_time"] <= 0
        or not Path(identity["executable"]).is_absolute()
    ):
        raise ContractError(f"process identity for PID {pid} is invalid")
    return identity


def _process_identity_is_alive(identity: Mapping[str, Any]) -> bool:
    if set(identity) != {"pid", "create_time", "executable"}:
        raise ContractError("process identity has invalid keys")
    try:
        pid = int(identity["pid"])
        create_time = float(identity["create_time"])
        executable = str(identity["executable"])
    except (TypeError, ValueError) as exc:
        raise ContractError("process identity values are invalid") from exc
    if pid <= 0 or not math.isfinite(create_time) or create_time <= 0 or not executable:
        raise ContractError("process identity values are invalid")
    try:
        current = _process_identity(pid)
    except ProcessGoneError:
        return False
    return (
        abs(float(current["create_time"]) - create_time) <= 1.0e-6
        and os.path.normcase(current["executable"])
        == os.path.normcase(str(Path(executable).resolve()))
    )


class PipelineLock:
    def __init__(self, path: Path, *, plan_sha256: str, recover_stale: bool = False):
        self.path = Path(path)
        self.plan_sha256 = plan_sha256
        self.recover_stale = bool(recover_stale)
        self.nonce = uuid.uuid4().hex
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                prior = _read_canonical_json(self.path)
            except ContractError as exc:
                raise ContractError(f"unreadable existing runner lock: {self.path}") from exc
            expected_keys = {
                "schema", "pid", "process_identity", "host", "nonce",
                "plan_sha256", "started_utc", "lock_sha256",
            }
            if set(prior) != expected_keys:
                raise ContractError("existing runner lock has invalid keys")
            _verify_self_digest(prior, "lock_sha256", label="existing runner lock")
            if prior.get("schema") != LOCK_SCHEMA:
                raise ContractError("existing runner lock schema mismatch")
            if prior.get("plan_sha256") != self.plan_sha256:
                raise ContractError("existing runner lock belongs to another frozen plan")
            if prior.get("host") != socket.gethostname():
                raise ContractError("existing runner lock belongs to another host")
            identity = prior.get("process_identity")
            if not isinstance(identity, Mapping):
                raise ContractError("existing runner lock lacks exact process identity")
            pid = int(identity.get("pid", -1))
            if prior.get("pid") != pid:
                raise ContractError("existing runner lock PID/identity mismatch")
            if _process_identity_is_alive(identity):
                raise ContractError(f"another autonomous runner is alive (PID {pid})")
            if not self.recover_stale:
                raise ContractError(
                    "a stale runner lock exists; inspect attempts and rerun with "
                    "--recover-stale-lock"
                )
            quarantine = self.path.with_name(
                f"{self.path.name}.stale.{prior.get('nonce', 'unknown')}.{int(time.time())}"
            )
            if quarantine.exists():
                raise ContractError(f"stale-lock quarantine collision: {quarantine}")
            os.replace(self.path, quarantine)
        payload = _self_digest_payload(
            {
                "schema": LOCK_SCHEMA,
                "pid": os.getpid(),
                "process_identity": _process_identity(os.getpid()),
                "host": socket.gethostname(),
                "nonce": self.nonce,
                "plan_sha256": self.plan_sha256,
                "started_utc": utc_now(),
            },
            "lock_sha256",
        )
        _create_once_canonical(self.path, payload)
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            current = _read_canonical_json(self.path)
            _verify_self_digest(current, "lock_sha256", label="owned runner lock")
            if current.get("schema") != LOCK_SCHEMA or current.get("plan_sha256") != self.plan_sha256:
                raise ContractError("runner lock schema/plan changed")
            if current.get("nonce") != self.nonce or current.get("pid") != os.getpid():
                raise ContractError("runner lock ownership changed")
            identity = current.get("process_identity")
            if not isinstance(identity, Mapping) or not _process_identity_is_alive(identity):
                raise ContractError("runner lock process identity changed")
            self.path.unlink()
        finally:
            self.acquired = False

    def __enter__(self) -> "PipelineLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


def _tail(path: Path, *, max_bytes: int = 32_768, max_lines: int = 80) -> list[str]:
    if not Path(path).is_file():
        return []
    with Path(path).open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - max_bytes))
        raw = stream.read()
    return raw.decode("utf-8", errors="replace").splitlines()[-max_lines:]


def _process_usage(pid: int) -> tuple[float, int]:
    """Return cumulative CPU seconds and RSS for a process tree when psutil exists."""

    try:
        import psutil  # type: ignore

        try:
            root = psutil.Process(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return 0.0, 0
        try:
            processes = [root, *root.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return 0.0, 0
        cpu = 0.0
        rss = 0
        for process in processes:
            try:
                times = process.cpu_times()
                cpu += float(times.user + times.system)
                rss += int(process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return cpu, rss
    except (ImportError, OSError):
        return 0.0, 0


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    try:
        import psutil  # type: ignore

        try:
            root = psutil.Process(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return
        children = root.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        try:
            root.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        _, alive = psutil.wait_procs([*children, root], timeout=10)
        for item in alive:
            try:
                item.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (ImportError, OSError):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


class _KillOnCloseProcessJob:
    """Own a Windows process tree and kill it if the supervisor disappears.

    The bootstrap is assigned before it receives the GO token.  Every target
    descendant therefore inherits membership in this job.  Windows closes the
    supervisor's last job handle even on a hard crash, and
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE then terminates the whole tree.  On
    non-Windows test hosts the existing explicit process-tree termination is
    retained; E26 production is Windows-only.
    """

    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
    JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS = 1
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    HANDLE_FLAG_INHERIT = 0x00000001

    def __init__(self) -> None:
        self._handle: int | None = None
        self._kernel32: Any | None = None
        self._accounting_type: Any | None = None
        self._assigned = False
        if os.name != "nt":
            return
        from ctypes import wintypes

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class _BasicAccountingInformation(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_int64),
                ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64),
                ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.IsProcessInJob.argtypes = [
            wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL),
        ]
        kernel32.IsProcessInJob.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.SetHandleInformation.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
        ]
        kernel32.SetHandleInformation.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ContractError(
                f"CreateJobObjectW failed with WinError {ctypes.get_last_error()}"
            )
        if not kernel32.SetHandleInformation(
            handle, self.HANDLE_FLAG_INHERIT, 0
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise ContractError(f"SetHandleInformation(job) failed with WinError {error}")
        information = _ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle,
            self.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise ContractError(f"SetInformationJobObject failed with WinError {error}")
        self._kernel32 = kernel32
        self._handle = int(handle)
        self._accounting_type = _BasicAccountingInformation

    def assign(self, process: subprocess.Popen[Any]) -> None:
        if self._handle is None:
            return
        from ctypes import wintypes

        raw_process_handle = getattr(process, "_handle", None)
        if raw_process_handle is None or not self._kernel32.AssignProcessToJobObject(
            wintypes.HANDLE(self._handle), wintypes.HANDLE(int(raw_process_handle))
        ):
            raise ContractError(
                "AssignProcessToJobObject failed before bootstrap GO "
                f"(WinError {ctypes.get_last_error()})"
            )
        in_job = wintypes.BOOL(False)
        if not self._kernel32.IsProcessInJob(
            wintypes.HANDLE(int(raw_process_handle)),
            wintypes.HANDLE(self._handle),
            ctypes.byref(in_job),
        ) or not bool(in_job.value):
            raise ContractError(
                "Windows job assignment could not be independently verified before GO"
            )
        self._assigned = True

    def active_processes(self) -> int:
        if self._handle is None:
            return 0
        from ctypes import wintypes

        information = self._accounting_type()
        returned = wintypes.DWORD(0)
        if not self._kernel32.QueryInformationJobObject(
            wintypes.HANDLE(self._handle),
            self.JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS,
            ctypes.byref(information),
            ctypes.sizeof(information),
            ctypes.byref(returned),
        ):
            raise ContractError(
                f"QueryInformationJobObject failed with WinError {ctypes.get_last_error()}"
            )
        return int(information.ActiveProcesses)

    def terminate(self, process: subprocess.Popen[Any]) -> None:
        if self._handle is not None and self._assigned:
            if not self._kernel32.TerminateJobObject(self._handle, 1):
                error = ctypes.get_last_error()
                if process.poll() is None:
                    raise ContractError(f"TerminateJobObject failed with WinError {error}")
            deadline = time.monotonic() + 15.0
            while self.active_processes() != 0 and time.monotonic() < deadline:
                time.sleep(0.05)
            if self.active_processes() != 0:
                raise ContractError("Windows job did not drain within 15 seconds")
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            return
        _terminate_process_tree(process)

    def assert_empty_after_root_exit(self) -> None:
        active = self.active_processes()
        if active:
            raise StageFailure(
                f"stage bootstrap exited while {active} descendant process(es) remained alive"
            )

    def close(self) -> None:
        if self._handle is None:
            return
        handle = self._handle
        self._handle = None
        if not self._kernel32.CloseHandle(handle):
            raise ContractError(f"CloseHandle(job) failed with WinError {ctypes.get_last_error()}")


def _directory_bytes(root: Path) -> int:
    total = 0
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            try:
                total += (Path(directory) / filename).stat().st_size
            except FileNotFoundError:
                continue
    return total


def _verify_process_isolation(value: Any, *, label: str) -> None:
    expected = {
        "kind": "windows_job_object" if os.name == "nt" else "process_tree",
        "kill_on_supervisor_close": os.name == "nt",
        "assignment_verified": True,
    }
    if value != expected:
        raise ContractError(f"{label} process-isolation evidence mismatch")


def _write_startup_emergency(
    *, emergency_dir: Path, invocation_id: str, reserve_path: Path | None,
    plan_path: Path, expected_plan_sha256: str, failure: BaseException,
) -> dict[str, Any]:
    """Write a report even when the plan/runner object could not be trusted."""

    if not INVOCATION_ID.fullmatch(invocation_id):
        raise ContractError("emergency invocation ID is invalid")
    directory = Path(emergency_dir).resolve()
    production_root = Path(r"E:\pazzle_work\e26_contextual_edge").resolve()
    _require_e_path(directory, "emergency_dir", allow_non_e=False)
    if not _is_within(directory, production_root / "orchestrator" / "reports" / "emergency"):
        raise ContractError("emergency_dir is outside the fixed E26 emergency-report root")
    if reserve_path is not None:
        reserve = Path(reserve_path).resolve()
        if not _is_within(reserve, directory):
            raise ContractError("emergency reserve escapes emergency_dir")
        if reserve.exists():
            reserve.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "schema": "pazzle-e26-autonomous-startup-emergency-v1",
        "invocation_id": invocation_id,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "plan": str(Path(plan_path).resolve()),
        "expected_plan_sha256": expected_plan_sha256,
        "state": "startup_failed",
        "failure": {
            "type": type(failure).__name__,
            "message": str(failure),
            "traceback": traceback.format_exc().splitlines()[-80:],
        },
        "updated_utc": utc_now(),
    }
    payload = _self_digest_payload(body, "emergency_sha256")
    path = directory / f"emergency_{invocation_id}.json"
    _create_once_canonical(path, payload)
    return {**payload, "path": str(path)}


@dataclass
class StageOutcome:
    receipt: dict[str, Any]
    scientific_pass: bool


class AutonomousRunner:
    def __init__(self, plan_path: Path, expected_plan_sha256: str, *,
                 allow_non_e: bool = False, poll_seconds: float = SUPERVISOR_POLL_SECONDS,
                 recover_stale_lock: bool = False):
        self.allow_non_e = allow_non_e
        self.plan_path = Path(plan_path).resolve()
        self.plan = load_and_verify_plan(
            self.plan_path, expected_plan_sha256, allow_non_e=allow_non_e
        )
        self.plan_sha256 = str(self.plan["plan_sha256"])
        self.root = Path(self.plan["work_root"])
        self.poll_seconds = float(poll_seconds)
        if (
            not math.isfinite(self.poll_seconds)
            or self.poll_seconds <= 0
            or self.poll_seconds > MAX_SUPERVISOR_POLL_SECONDS
        ):
            raise ContractError(
                f"poll_seconds must be finite and in (0,{MAX_SUPERVISOR_POLL_SECONDS}]"
            )
        if not allow_non_e and self.poll_seconds != SUPERVISOR_POLL_SECONDS:
            raise ContractError("production supervisor poll interval is frozen at 5 seconds")
        self.recover_stale_lock = bool(recover_stale_lock)
        self.state_root = self.root / "orchestrator"
        self.receipt_root = self.state_root / "receipts"
        self.attempt_root = self.state_root / "attempts"
        self.log_root = self.state_root / "logs"
        self.event_root = self.state_root / "events"
        self.report_root = self.state_root / "reports"
        self.status_path = self.state_root / "status.json"
        self.recovery_path = self.report_root / "recovery_report.json"
        self.final_path = self.report_root / "final_report.json"
        self.lock_path = self.state_root / "runner.lock"
        self._run_started_monotonic = time.monotonic()
        self._run_started_utc = utc_now()
        self._current_process: subprocess.Popen[Any] | None = None
        self._lock_owned = False
        self._event_sequence = self._discover_event_sequence()

    def _discover_event_sequence(self) -> int:
        if not self.event_root.exists():
            return 0
        maximum = 0
        for path in self.event_root.glob("[0-9][0-9][0-9][0-9][0-9][0-9]_*.json"):
            try:
                maximum = max(maximum, int(path.name[:6]))
            except ValueError:
                continue
        return maximum

    def _ensure_runtime_dirs(self) -> None:
        for name in REQUIRED_E_ENV:
            Path(self.plan["environment"][name]).mkdir(parents=True, exist_ok=True)
        for path in (
            self.receipt_root, self.attempt_root, self.log_root,
            self.event_root, self.report_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _emit_event(self, event: str, **fields: Any) -> dict[str, Any]:
        self._event_sequence += 1
        payload = _self_digest_payload(
            {
                "schema": EVENT_SCHEMA,
                "sequence": self._event_sequence,
                "event": event,
                "time_utc": utc_now(),
                "plan_sha256": self.plan_sha256,
                **fields,
            },
            "event_sha256",
        )
        path = self.event_root / f"{self._event_sequence:06d}_{event}.json"
        _create_once_canonical(path, payload)
        return payload

    def _receipt_path(self, stage_name: str) -> Path:
        return self.receipt_root / f"{stage_name}.json"

    def _attempt_dirs(self, stage_name: str) -> list[Path]:
        parent = self.attempt_root / stage_name
        if not parent.exists():
            return []
        return sorted(path for path in parent.iterdir() if path.is_dir() and path.name.startswith("attempt_"))

    def _cumulative_cpu_seconds(self) -> float:
        return self._cumulative_attempt_usage()[0]

    def _cumulative_attempt_usage(self) -> tuple[float, float, float]:
        cpu_total = 0.0
        wall_total = 0.0
        gpu_total = 0.0
        for path in self.attempt_root.glob("*/attempt_*/attempt.json"):
            try:
                value = _read_canonical_json(path)
                if value.get("schema") != ATTEMPT_SCHEMA:
                    raise ContractError(f"attempt schema mismatch: {path}")
                if value.get("plan_sha256") != self.plan_sha256:
                    raise ContractError(f"attempt belongs to another plan: {path}")
                _verify_self_digest(value, "attempt_sha256", label=f"accounting attempt {path}")
                cpu_value = float(value.get("cpu_seconds", 0.0))
                elapsed_value = float(value.get("elapsed_seconds", 0.0))
                poll_value = float(value.get("accounting_poll_seconds", self.poll_seconds))
                resource_class = value.get("resource_class")
                if (
                    not math.isfinite(cpu_value) or cpu_value < 0
                    or not math.isfinite(elapsed_value) or elapsed_value < 0
                    or not math.isfinite(poll_value) or poll_value <= 0
                    or poll_value > MAX_SUPERVISOR_POLL_SECONDS
                    or resource_class not in {"cpu", "gpu"}
                ):
                    raise ContractError(f"attempt has invalid resource accounting: {path}")
                # A supervisor can die between two durable heartbeats.  For
                # every nonterminal attempt charge one full polling window
                # (and all logical CPUs) so crash/restart can never make
                # consumed budget disappear.
                terminal = value.get("state") in {
                    "process_complete", "process_failed", "failed",
                }
                surcharge_wall = 0.0 if terminal else poll_value
                surcharge_cpu = 0.0 if terminal else poll_value * max(1, os.cpu_count() or 1)
                cpu_total += cpu_value + surcharge_cpu
                elapsed = elapsed_value + surcharge_wall
                wall_total += elapsed
                if resource_class == "gpu":
                    gpu_total += elapsed
            except (ContractError, OSError, TypeError, ValueError) as exc:
                # An unreadable attempt ledger is provenance damage, not zero cost.
                raise ContractError(f"cannot account CPU usage from canonical attempt receipt {path}: {exc}")
        return cpu_total, wall_total, gpu_total

    def _verify_plan_provenance(self) -> None:
        disk_plan = _read_canonical_json(self.plan_path)
        if (
            sha256_file(self.plan_path) != self.plan_sha256
            or canonical_digest(disk_plan) != self.plan_sha256
            or disk_plan != {key: value for key, value in self.plan.items() if key != "plan_sha256"}
        ):
            raise ContractError("frozen plan bytes drifted after load")
        _verify_path_record(self.plan["spec"], label="frozen plan spec")
        sources = self.plan["sources"]
        for index, record in enumerate(sources):
            _verify_path_record(record, label=f"source[{index}]")
        if canonical_digest(sources) != self.plan["sources_sha256"]:
            raise ContractError("source-list digest changed in memory")
        current_runtime = _runtime_record(Path(self.plan["runtime"]["python_executable"]))
        if current_runtime != self.plan["runtime"]:
            raise ContractError("Python/dependency runtime drifted after plan freeze")

    def _enforce_global_caps(self, stage: Mapping[str, Any]) -> None:
        """Fail before spawning anything when a durable cap is already exhausted."""

        free = shutil.disk_usage(self.root).free
        if free < int(stage["min_free_bytes"]):
            raise ResourceLimitError(f"minimum free disk violated before stage: {free}")
        artifact_bytes = _directory_bytes(self.root)
        cpu_seconds, wall_seconds, gpu_seconds = self._cumulative_attempt_usage()
        caps = self.plan["global_caps"]
        if cpu_seconds >= float(caps["max_cpu_seconds"]):
            raise ResourceLimitError(
                f"aggregate CPU cap exhausted before stage: {cpu_seconds:.3f}s"
            )
        if gpu_seconds >= float(caps["max_gpu_seconds"]):
            raise ResourceLimitError(
                f"aggregate GPU-stage cap exhausted before stage: {gpu_seconds:.3f}s"
            )
        if wall_seconds >= float(caps["max_wall_seconds"]):
            raise ResourceLimitError(
                f"aggregate durable wall cap exhausted before stage: {wall_seconds:.3f}s"
            )
        if artifact_bytes >= int(caps["max_artifact_bytes"]):
            raise ResourceLimitError(
                f"aggregate artifact cap exhausted before stage: {artifact_bytes}"
            )

    def _completed_receipts(self) -> list[dict[str, Any]]:
        receipts: list[dict[str, Any]] = []
        for stage in self.plan["stages"]:
            path = self._receipt_path(stage["name"])
            if not path.exists():
                continue
            receipts.append(self._verify_receipt(stage, path))
        return receipts

    def _resume_command(self) -> list[str]:
        launcher = (Path(self.plan["repo_root"]) / "launch_e26_autonomous.ps1").resolve()
        if os.name == "nt":
            shell = Path(self.plan["environment"]["SYSTEMROOT"]) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
            prefix = [str(shell), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher)]
        else:
            prefix = [str(launcher)]
        return [
            *prefix,
            "-Plan", str(self.plan_path), "-PlanSha256", self.plan_sha256,
            "-RecoverStaleLock",
        ]

    def _report(self, *, state: str, current_stage: str | None,
                message: str, failure: Mapping[str, Any] | None = None,
                final: bool = False) -> dict[str, Any]:
        completed: list[dict[str, Any]] = []
        shallow_receipts = state == "running"
        for stage in self.plan["stages"]:
            receipt_path = self._receipt_path(stage["name"])
            if receipt_path.exists():
                try:
                    if shallow_receipts:
                        receipt = _read_canonical_json(receipt_path)
                        if (
                            receipt.get("schema") != RECEIPT_SCHEMA
                            or receipt.get("stage") != stage["name"]
                            or receipt.get("plan_sha256") != self.plan_sha256
                        ):
                            raise ContractError(f"receipt identity mismatch: {receipt_path}")
                        _verify_self_digest(
                            receipt, "receipt_sha256", label=f"receipt {stage['name']}"
                        )
                    else:
                        receipt = self._verify_receipt(stage, receipt_path)
                    completed.append({
                        "stage": stage["name"],
                        "receipt": str(receipt_path),
                        "receipt_sha256": receipt["receipt_sha256"],
                        "scientific_pass": receipt["scientific_pass"],
                    })
                except ContractError as exc:
                    completed.append({"stage": stage["name"], "receipt_invalid": str(exc)})
        completed_names = {item["stage"] for item in completed if "receipt_sha256" in item}
        next_stage = next(
            (stage["name"] for stage in self.plan["stages"] if stage["name"] not in completed_names),
            None,
        )
        resume_command = self._resume_command()
        latest_attempts: list[dict[str, Any]] = []
        for stage in self.plan["stages"]:
            directories = self._attempt_dirs(stage["name"])
            if not directories:
                continue
            attempt_path = directories[-1] / "attempt.json"
            try:
                attempt = _read_canonical_json(attempt_path)
                _verify_self_digest(
                    attempt, "attempt_sha256", label=f"latest attempt {stage['name']}"
                )
                expected_stdout = (directories[-1] / "stdout.log").resolve()
                expected_stderr = (directories[-1] / "stderr.log").resolve()
                if (
                    Path(str(attempt.get("stdout", ""))).resolve() != expected_stdout
                    or Path(str(attempt.get("stderr", ""))).resolve() != expected_stderr
                ):
                    raise ContractError("attempt log path escapes its immutable attempt directory")
                latest_attempts.append({
                    "stage": stage["name"],
                    "attempt": attempt.get("attempt"),
                    "state": attempt.get("state"),
                    "pid": attempt.get("pid"),
                    "progress": attempt.get("progress"),
                    "elapsed_seconds": attempt.get("elapsed_seconds"),
                    "cpu_seconds": attempt.get("cpu_seconds"),
                    "peak_rss_bytes": attempt.get("peak_rss_bytes"),
                    "failure": attempt.get("failure"),
                    "stdout": attempt.get("stdout"),
                    "stderr": attempt.get("stderr"),
                    "stdout_tail": _tail(Path(str(attempt.get("stdout", "")))),
                    "stderr_tail": _tail(Path(str(attempt.get("stderr", "")))),
                })
            except (ContractError, OSError) as exc:
                latest_attempts.append({
                    "stage": stage["name"],
                    "attempt_invalid": str(exc),
                    "attempt_path": str(attempt_path),
                })
        accounting_error: str | None = None
        try:
            cpu_seconds, durable_wall_seconds, gpu_seconds = self._cumulative_attempt_usage()
        except (ContractError, OSError, TypeError, ValueError) as exc:
            # Reporting must survive the exact ledger damage that blocks the
            # scientific run.  The run still fails closed; totals become null
            # and the damage is explicit rather than silently counted as zero.
            cpu_seconds = durable_wall_seconds = gpu_seconds = None
            accounting_error = str(exc)
        if final:
            invalid = [item for item in completed if "receipt_invalid" in item]
            if invalid or accounting_error is not None:
                raise ContractError("terminal report cannot authenticate receipts/resource ledger")
            completed_names_in_order = [item["stage"] for item in completed]
            planned_names = [stage["name"] for stage in self.plan["stages"]]
            if completed_names_in_order != planned_names[:len(completed_names_in_order)]:
                raise ContractError("terminal report receipts are not a contiguous DAG prefix")
            pass_values = [bool(item["scientific_pass"]) for item in completed]
            if state == "complete":
                if completed_names_in_order != planned_names or not all(pass_values):
                    raise ContractError("complete terminal report requires every exact PASS receipt")
            elif state == "scientific_fail":
                if (
                    not pass_values
                    or pass_values[-1]
                    or not all(pass_values[:-1])
                    or len(completed_names_in_order) >= len(planned_names)
                    and completed_names_in_order != planned_names
                ):
                    raise ContractError(
                        "scientific-fail terminal report requires a contiguous prefix ending FAIL"
                    )
            else:
                raise ContractError(f"invalid terminal report state: {state}")
            next_stage = None
        payload: dict[str, Any] = {
            "schema": FINAL_REPORT_SCHEMA if final else REPORT_SCHEMA,
            "pipeline_id": self.plan["pipeline_id"],
            "plan": str(self.plan_path),
            "plan_sha256": self.plan_sha256,
            "state": state,
            "current_stage": current_stage,
            "next_stage": next_stage,
            "message": message,
            "completed": completed,
            "latest_attempts": latest_attempts,
            "cumulative_cpu_seconds": cpu_seconds,
            "cumulative_attempt_wall_seconds": durable_wall_seconds,
            "cumulative_gpu_stage_seconds": gpu_seconds,
            "resource_accounting_error": accounting_error,
            "elapsed_this_invocation_seconds": max(0.0, time.monotonic() - self._run_started_monotonic),
            "updated_utc": utc_now(),
            "resume_command": resume_command,
            "failure": dict(failure) if failure is not None else None,
        }
        key = "final_report_sha256" if final else "report_sha256"
        payload = _self_digest_payload(payload, key)
        target_path = self.final_path if final else self.recovery_path
        if final:
            _create_once_canonical(target_path, payload)
        else:
            _atomic_write_canonical(target_path, payload)
        status = {
            "schema": STATUS_SCHEMA,
            "pipeline_id": self.plan["pipeline_id"],
            "plan_sha256": self.plan_sha256,
            "state": state,
            "current_stage": current_stage,
            "next_stage": next_stage,
            "message": message,
            "cumulative_cpu_seconds": payload["cumulative_cpu_seconds"],
            "updated_utc": payload["updated_utc"],
            "recovery_report": str(self.recovery_path),
            "final_report": str(self.final_path) if self.final_path.exists() else None,
        }
        status = _self_digest_payload(status, "status_sha256")
        _atomic_write_canonical(self.status_path, status)
        return payload

    def _write_invocation_diagnostic(
        self, *, state: str, message: str, failure: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Best-effort create-once report independent of attempts/receipts/lock."""

        self.report_root.mkdir(parents=True, exist_ok=True)
        body = {
            "schema": "pazzle-e26-autonomous-invocation-diagnostic-v1",
            "pipeline_id": self.plan["pipeline_id"],
            "plan": str(self.plan_path),
            "plan_sha256": self.plan_sha256,
            "state": state,
            "message": message,
            "failure": dict(failure),
            "resume_command": self._resume_command(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "updated_utc": utc_now(),
        }
        payload = _self_digest_payload(body, "diagnostic_sha256")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.report_root / f"invocation_{stamp}_{os.getpid()}_{uuid.uuid4().hex}.json"
        _create_once_canonical(path, payload)
        return {**payload, "path": str(path)}

    def _safe_report(self, **kwargs: Any) -> dict[str, Any]:
        try:
            return self._report(**kwargs)
        except BaseException as report_exc:
            original_failure = kwargs.get("failure")
            failure = {
                "type": type(report_exc).__name__,
                "message": str(report_exc),
                "while_reporting": dict(original_failure) if isinstance(original_failure, Mapping) else None,
            }
            try:
                diagnostic = self._write_invocation_diagnostic(
                    state="reporting_failed",
                    message="shared report could not be authenticated or written; use this immutable diagnostic",
                    failure=failure,
                )
                if self._lock_owned:
                    diagnostic_record = _path_record(Path(diagnostic["path"]))
                    fallback = _self_digest_payload({
                        "schema": REPORT_SCHEMA,
                        "pipeline_id": self.plan["pipeline_id"],
                        "plan": str(self.plan_path),
                        "plan_sha256": self.plan_sha256,
                        "state": "blocked",
                        "current_stage": kwargs.get("current_stage"),
                        "next_stage": None,
                        "message": "normal recovery report failed; immutable diagnostic attached",
                        "completed": [],
                        "latest_attempts": [],
                        "cumulative_cpu_seconds": None,
                        "cumulative_attempt_wall_seconds": None,
                        "cumulative_gpu_stage_seconds": None,
                        "resource_accounting_error": str(report_exc),
                        "elapsed_this_invocation_seconds": max(
                            0.0, time.monotonic() - self._run_started_monotonic
                        ),
                        "updated_utc": utc_now(),
                        "resume_command": self._resume_command(),
                        "failure": failure,
                        "emergency_diagnostic": diagnostic_record,
                    }, "report_sha256")
                    _atomic_write_canonical(self.recovery_path, fallback)
                    return fallback
                return diagnostic
            except BaseException as diagnostic_exc:
                print(
                    f"E26 reporting failure: {report_exc}; diagnostic failure: {diagnostic_exc}",
                    file=sys.stderr,
                    flush=True,
                )
                return {
                    "state": "reporting_failed",
                    "failure": failure,
                    "diagnostic_write_error": str(diagnostic_exc),
                }

    def _safe_emit_event(self, event: str, **fields: Any) -> None:
        try:
            self._emit_event(event, **fields)
        except BaseException as exc:
            try:
                self._write_invocation_diagnostic(
                    state="event_write_failed",
                    message=f"nonterminal event {event!r} could not be written",
                    failure={"type": type(exc).__name__, "message": str(exc)},
                )
            except BaseException as diagnostic_exc:
                print(
                    f"E26 event/report write failure: {exc}; {diagnostic_exc}",
                    file=sys.stderr,
                    flush=True,
                )

    def _verify_final_report(self) -> dict[str, Any]:
        value = _read_canonical_json(self.final_path)
        if (
            value.get("schema") != FINAL_REPORT_SCHEMA
            or value.get("plan_sha256") != self.plan_sha256
            or value.get("pipeline_id") != self.plan["pipeline_id"]
            or value.get("state") not in {"complete", "scientific_fail"}
        ):
            raise ContractError("terminal report schema/plan/state mismatch")
        _verify_self_digest(value, "final_report_sha256", label="terminal report")
        if value.get("resource_accounting_error") is not None:
            raise ContractError("terminal report contains a resource-accounting error")
        completed: list[dict[str, Any]] = []
        gap_seen = False
        for stage in self.plan["stages"]:
            receipt_path = self._receipt_path(stage["name"])
            if not receipt_path.exists():
                gap_seen = True
                continue
            if gap_seen:
                raise ContractError("terminal report has a receipt after a DAG gap")
            receipt = self._verify_receipt(stage, receipt_path)
            completed.append({
                "stage": stage["name"],
                "receipt": str(receipt_path),
                "receipt_sha256": receipt["receipt_sha256"],
                "scientific_pass": receipt["scientific_pass"],
            })
        if value.get("completed") != completed or value.get("next_stage") is not None:
            raise ContractError("terminal report receipt summary/next-stage mismatch")
        passes = [bool(item["scientific_pass"]) for item in completed]
        if value["state"] == "complete":
            if len(completed) != len(self.plan["stages"]) or not all(passes):
                raise ContractError("complete terminal report lacks all PASS receipts")
        else:
            if not passes or passes[-1] or not all(passes[:-1]):
                raise ContractError("scientific-fail report is not a PASS-prefix ending in FAIL")
        for name in (
            "cumulative_cpu_seconds", "cumulative_attempt_wall_seconds",
            "cumulative_gpu_stage_seconds", "elapsed_this_invocation_seconds",
        ):
            number = float(value.get(name))
            if not math.isfinite(number) or number < 0:
                raise ContractError(f"terminal report has invalid {name}")
        return value

    def verify_snapshot(self) -> dict[str, Any]:
        """Authenticate current state without creating or modifying any file."""

        self._verify_plan_provenance()
        receipts: list[dict[str, Any]] = []
        gap_seen = False
        for stage in self.plan["stages"]:
            path = self._receipt_path(stage["name"])
            if not path.exists():
                gap_seen = True
                continue
            if gap_seen:
                raise ContractError("receipt exists after a missing DAG predecessor")
            receipt = self._verify_receipt(stage, path)
            receipts.append({
                "stage": stage["name"],
                "receipt_sha256": receipt["receipt_sha256"],
                "scientific_pass": receipt["scientific_pass"],
            })
        final: dict[str, Any] | None = None
        if self.final_path.exists():
            verified_final = self._verify_final_report()
            final = {
                "state": verified_final["state"],
                "final_report_sha256": verified_final["final_report_sha256"],
            }
        for path, schema, key, label in (
            (self.recovery_path, REPORT_SCHEMA, "report_sha256", "recovery report"),
            (self.status_path, STATUS_SCHEMA, "status_sha256", "status"),
        ):
            if path.exists():
                payload = _read_canonical_json(path)
                if payload.get("schema") != schema or payload.get("plan_sha256") != self.plan_sha256:
                    raise ContractError(f"{label} schema/plan mismatch")
                _verify_self_digest(payload, key, label=label)
        cpu, wall, gpu = self._cumulative_attempt_usage()
        return {
            "schema": "pazzle-e26-autonomous-verification-snapshot-v1",
            "plan": str(self.plan_path),
            "plan_sha256": self.plan_sha256,
            "receipts": receipts,
            "final": final,
            "cumulative_cpu_seconds": cpu,
            "cumulative_attempt_wall_seconds": wall,
            "cumulative_gpu_stage_seconds": gpu,
        }

    def _verify_explicit_inputs(self, stage: Mapping[str, Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in stage["inputs"]:
            path = Path(item["path"])
            actual = _path_record(path)
            if "bytes" in item and int(item["bytes"]) != actual["bytes"]:
                raise ContractError(f"input byte-size mismatch for stage {stage['name']}: {path}")
            if "sha256" in item and item["sha256"] != actual["sha256"]:
                raise ContractError(f"input SHA mismatch for stage {stage['name']}: {path}")
            records.append(actual)
        return records

    def _dependency_records(self, stage: Mapping[str, Any]) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        by_name = {item["name"]: item for item in self.plan["stages"]}
        for name in stage["dependencies"]:
            receipt = self._verify_receipt(by_name[name], self._receipt_path(name))
            if not receipt["scientific_pass"]:
                raise ScientificGateFailure(f"dependency {name} is a terminal scientific FAIL")
            records.append({"stage": name, "receipt_sha256": receipt["receipt_sha256"]})
        return records

    def _execution_contract(self, stage: Mapping[str, Any], inputs: Sequence[Mapping[str, Any]],
                            dependencies: Sequence[Mapping[str, Any]], argv: Sequence[str]) -> dict[str, Any]:
        body = {
            "plan_sha256": self.plan_sha256,
            "sources_sha256": self.plan["sources_sha256"],
            "stage": dict(stage),
            "resolved_argv": list(argv),
            "inputs": list(inputs),
            "dependencies": list(dependencies),
            "environment": self.plan["environment"],
            "runtime": self.plan["runtime"],
        }
        body["execution_contract_sha256"] = canonical_digest(body)
        return body

    def _validate_outputs(self, stage: Mapping[str, Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for output in stage["outputs"]:
            path = Path(output["path"])
            actual = _path_record(path)
            if actual["bytes"] < int(output["min_bytes"]):
                raise ContractError(f"stage output is too small: {path}")
            if "max_bytes" in output and actual["bytes"] > int(output["max_bytes"]):
                raise ContractError(f"stage output is too large: {path}")
            records.append(actual)
        return records

    def _gate_value(self, stage: Mapping[str, Any]) -> tuple[bool, Any]:
        gate = stage["gate"]
        if gate is None:
            return True, None
        payload = _read_canonical_json(Path(gate["path"]))
        current: Any = payload
        for key in gate["pointer"]:
            if not isinstance(current, Mapping) or key not in current:
                raise ContractError(f"scientific gate pointer is missing: {gate['pointer']}")
            current = current[key]
        return current == gate["pass_value"], current

    def _verify_receipt(self, stage: Mapping[str, Any], path: Path) -> dict[str, Any]:
        self._verify_plan_provenance()
        value = _read_canonical_json(path)
        if value.get("schema") != RECEIPT_SCHEMA or value.get("stage") != stage["name"]:
            raise ContractError(f"receipt schema/stage mismatch: {path}")
        if value.get("plan_sha256") != self.plan_sha256:
            raise ContractError(f"receipt plan mismatch: {path}")
        _verify_self_digest(value, "receipt_sha256", label=f"receipt {stage['name']}")
        # Dependency receipts and explicit inputs are rechecked first, so every
        # subsequently recomputed execution contract uses current authenticated
        # identities rather than receipt-controlled values.
        dependencies = value.get("dependencies")
        if not isinstance(dependencies, list):
            raise ContractError("receipt dependencies are invalid")
        expected_dependencies = self._dependency_records_for_receipt(stage)
        if dependencies != expected_dependencies:
            raise ContractError(f"receipt dependency drift: {path}")
        inputs = value.get("inputs")
        if not isinstance(inputs, list):
            raise ContractError("receipt inputs are invalid")
        current_inputs = self._verify_explicit_inputs(stage)
        if inputs != current_inputs:
            raise ContractError(f"receipt input drift: {path}")
        outputs = value.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != len(stage["outputs"]):
            raise ContractError(f"receipt output list mismatch: {path}")
        for index, output in enumerate(outputs):
            if not isinstance(output, Mapping):
                raise ContractError(f"receipt output record is invalid: {path}")
            actual = _verify_path_record(output, label=f"receipt output[{index}]")
            if actual["path"] != stage["outputs"][index]["path"]:
                raise ContractError(f"receipt output path order mismatch: {path}")
        logs = value.get("logs")
        if not isinstance(logs, Mapping):
            raise ContractError(f"receipt logs are invalid: {path}")
        for stream_name in ("stdout", "stderr"):
            record = logs.get(stream_name)
            if not isinstance(record, Mapping):
                raise ContractError(f"receipt {stream_name} record is invalid: {path}")
            verified_log = _verify_path_record(
                record, label=f"receipt {stage['name']} {stream_name}"
            )
            # Exact expected location is checked after the attempt number/path
            # are parsed below.
        attempt_path = value.get("attempt_path")
        if not isinstance(attempt_path, str):
            raise ContractError(f"receipt attempt path is invalid: {path}")
        try:
            attempt_number = int(value.get("attempt"))
        except (TypeError, ValueError) as exc:
            raise ContractError(f"receipt attempt number is invalid: {path}") from exc
        expected_attempt_path = (
            self.attempt_root / stage["name"] / f"attempt_{attempt_number:04d}" / "attempt.json"
        ).resolve()
        if Path(attempt_path).resolve() != expected_attempt_path:
            raise ContractError(f"receipt attempt path escapes expected transaction: {path}")
        for stream_name in ("stdout", "stderr"):
            expected_log_path = expected_attempt_path.parent / f"{stream_name}.log"
            if Path(str(logs[stream_name]["path"])).resolve() != expected_log_path:
                raise ContractError(f"receipt {stream_name} path mismatch: {path}")
        attempt = _read_canonical_json(Path(attempt_path))
        if (
            attempt.get("schema") != ATTEMPT_SCHEMA
            or attempt.get("stage") != stage["name"]
            or attempt.get("plan_sha256") != self.plan_sha256
            or attempt.get("attempt") != attempt_number
            or attempt.get("resource_class") != stage["resource_class"]
            or Path(str(attempt.get("stdout", ""))).resolve()
            != expected_attempt_path.parent / "stdout.log"
            or Path(str(attempt.get("stderr", ""))).resolve()
            != expected_attempt_path.parent / "stderr.log"
            or attempt.get("state") != "process_complete"
            or attempt.get("returncode") != 0
        ):
            raise ContractError(f"receipt attempt identity/state mismatch: {path}")
        _verify_self_digest(attempt, "attempt_sha256", label=f"receipt attempt {stage['name']}")
        _verify_process_isolation(
            attempt.get("process_isolation"), label=f"receipt attempt {stage['name']}"
        )
        expected_argv = self._choose_argv(stage, attempt_number)
        if value.get("argv") != expected_argv or attempt.get("argv") != expected_argv:
            raise ContractError(f"receipt argv differs from frozen attempt command: {path}")
        expected_main_contract = self._execution_contract(
            stage,
            current_inputs,
            expected_dependencies,
            expected_argv,
        )
        if (
            value.get("execution_contract_sha256")
            != expected_main_contract["execution_contract_sha256"]
            or attempt.get("execution_contract_sha256")
            != expected_main_contract["execution_contract_sha256"]
        ):
            raise ContractError(f"receipt main execution contract drift: {path}")
        launch_request = attempt.get("launch_request")
        if not isinstance(launch_request, Mapping):
            raise ContractError(f"receipt launch request is missing: {path}")
        launch_record = _verify_path_record(
            launch_request, label=f"receipt launch request {stage['name']}"
        )
        expected_launch_path = expected_attempt_path.parent / "launch_request.json"
        if Path(launch_record["path"]).resolve() != expected_launch_path:
            raise ContractError(f"receipt launch request path mismatch: {path}")
        launch_payload = _read_canonical_json(expected_launch_path)
        expected_launch_payload = {
            "schema": LAUNCH_REQUEST_SCHEMA,
            "argv": expected_argv,
            "argv_sha256": canonical_digest(expected_argv),
            "environment_sha256": self.plan["environment_sha256"],
            "working_directory": stage["working_directory"],
            "execution_contract_sha256": expected_main_contract["execution_contract_sha256"],
        }
        if launch_payload != expected_launch_payload:
            raise ContractError(f"receipt launch request payload mismatch: {path}")
        seal_record = value.get("main_output_seal")
        if not isinstance(seal_record, Mapping):
            raise ContractError(f"receipt main-output seal record is missing: {path}")
        verified_seal_record = _verify_path_record(
            seal_record, label=f"receipt main-output seal {stage['name']}"
        )
        expected_seal_path = expected_attempt_path.parent / "main_outputs_seal.json"
        if Path(verified_seal_record["path"]).resolve() != expected_seal_path:
            raise ContractError(f"receipt main-output seal path mismatch: {path}")
        seal = _read_canonical_json(expected_seal_path)
        _verify_self_digest(seal, "seal_sha256", label=f"receipt seal {stage['name']}")
        if (
            seal.get("plan_sha256") != self.plan_sha256
            or seal.get("stage") != stage["name"]
            or seal.get("attempt") != value.get("attempt")
            or seal.get("execution_contract_sha256")
            != expected_main_contract["execution_contract_sha256"]
            or seal.get("outputs") != outputs
            or value.get("main_output_seal_sha256") != seal.get("seal_sha256")
        ):
            raise ContractError(f"receipt main-output seal identity mismatch: {path}")
        verifier = value.get("verifier")
        if stage["verifier_argv"]:
            if not isinstance(verifier, Mapping):
                raise ContractError(f"receipt verifier evidence is missing: {path}")
            verifier_attempt = verifier.get("attempt")
            verifier_logs = verifier.get("logs")
            if not isinstance(verifier_attempt, Mapping) or not isinstance(verifier_logs, Mapping):
                raise ContractError(f"receipt verifier evidence is invalid: {path}")
            _verify_self_digest(
                verifier_attempt,
                "attempt_sha256",
                label=f"receipt verifier attempt {stage['name']}",
            )
            if (
                verifier_attempt.get("schema") != ATTEMPT_SCHEMA
                or verifier_attempt.get("plan_sha256") != self.plan_sha256
                or verifier_attempt.get("stage") != stage["name"] + "_verifier"
                or verifier_attempt.get("resource_class") != stage["resource_class"]
                or verifier_attempt.get("state") != "process_complete"
                or verifier_attempt.get("returncode") != 0
            ):
                raise ContractError(f"receipt verifier state mismatch: {path}")
            _verify_process_isolation(
                verifier_attempt.get("process_isolation"),
                label=f"receipt verifier attempt {stage['name']}",
            )
            verifier_stage = {
                **stage,
                "name": stage["name"] + "_verifier",
                "argv": stage["verifier_argv"],
                "resume_argv": None,
                "outputs": [],
                "verifier_argv": [],
                "gate": None,
                "timeout_seconds": min(int(stage["timeout_seconds"]), 3600),
                "max_attempts": 1,
            }
            verifier_contract = self._execution_contract(
                verifier_stage,
                outputs,
                [],
                stage["verifier_argv"],
            )
            if (
                verifier.get("execution_contract_sha256")
                != verifier_contract["execution_contract_sha256"]
                or verifier_attempt.get("execution_contract_sha256")
                != verifier_contract["execution_contract_sha256"]
            ):
                raise ContractError(f"receipt verifier contract drift: {path}")
            verifier_launch = verifier_attempt.get("launch_request")
            if not isinstance(verifier_launch, Mapping):
                raise ContractError(f"receipt verifier launch request is missing: {path}")
            verifier_launch_record = _verify_path_record(
                verifier_launch,
                label=f"receipt verifier launch request {stage['name']}",
            )
            verifier_attempt_number = int(verifier_attempt.get("attempt", -1))
            if verifier_attempt_number <= 0:
                raise ContractError(f"receipt verifier attempt number is invalid: {path}")
            expected_verifier_attempt_path = (
                self.attempt_root / (stage["name"] + "_verifier")
                / f"attempt_{verifier_attempt_number:04d}" / "attempt.json"
            ).resolve()
            if _read_canonical_json(expected_verifier_attempt_path) != verifier_attempt:
                raise ContractError(f"receipt verifier attempt artifact mismatch: {path}")
            expected_verifier_launch_path = (
                self.attempt_root / (stage["name"] + "_verifier")
                / f"attempt_{verifier_attempt_number:04d}" / "launch_request.json"
            ).resolve()
            if Path(verifier_launch_record["path"]).resolve() != expected_verifier_launch_path:
                raise ContractError(f"receipt verifier launch path mismatch: {path}")
            expected_verifier_launch = {
                "schema": LAUNCH_REQUEST_SCHEMA,
                "argv": stage["verifier_argv"],
                "argv_sha256": canonical_digest(stage["verifier_argv"]),
                "environment_sha256": self.plan["environment_sha256"],
                "working_directory": stage["working_directory"],
                "execution_contract_sha256": verifier_contract["execution_contract_sha256"],
            }
            if _read_canonical_json(expected_verifier_launch_path) != expected_verifier_launch:
                raise ContractError(f"receipt verifier launch payload mismatch: {path}")
            for stream_name in ("stdout", "stderr"):
                record = verifier_logs.get(stream_name)
                if not isinstance(record, Mapping):
                    raise ContractError(f"receipt verifier {stream_name} is invalid: {path}")
                _verify_path_record(
                    record,
                    label=f"receipt verifier {stage['name']} {stream_name}",
                )
        elif verifier is not None:
            raise ContractError(f"receipt has unexpected verifier evidence: {path}")
        expected_pass, observed_gate = self._gate_value(stage)
        expected_gate = ({
            "contract": stage["gate"],
            "observed_value": observed_gate,
            "passed": expected_pass,
        } if stage["gate"] is not None else None)
        if value.get("gate") != expected_gate or value.get("scientific_pass") is not expected_pass:
            raise ContractError(f"receipt scientific gate evidence drift: {path}")
        return value

    def _dependency_records_for_receipt(self, stage: Mapping[str, Any]) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        by_name = {item["name"]: item for item in self.plan["stages"]}
        for name in stage["dependencies"]:
            path = self._receipt_path(name)
            value = _read_canonical_json(path)
            if value.get("schema") != RECEIPT_SCHEMA or value.get("stage") != name:
                raise ContractError(f"dependency receipt is invalid: {path}")
            if value.get("plan_sha256") != self.plan_sha256:
                raise ContractError(f"dependency receipt plan mismatch: {path}")
            _verify_self_digest(value, "receipt_sha256", label=f"dependency {name}")
            # Fully validate transitive output hashes without recursing through
            # dependencies indefinitely: the outer sequential scan validates
            # every earlier receipt before a downstream stage can run.
            for output in value.get("outputs", []):
                _verify_path_record(output, label=f"dependency {name} output")
            records.append({"stage": name, "receipt_sha256": value["receipt_sha256"]})
        return records

    def _next_attempt_number(self, stage_name: str) -> int:
        numbers: list[int] = []
        for path in self._attempt_dirs(stage_name):
            try:
                number = int(path.name.removeprefix("attempt_"))
            except ValueError:
                raise ContractError(f"unexpected attempt directory: {path}")
            attempt_path = path / "attempt.json"
            attempt = _read_canonical_json(attempt_path)
            if attempt.get("schema") != ATTEMPT_SCHEMA or attempt.get("stage") != stage_name:
                raise ContractError(f"attempt identity mismatch: {attempt_path}")
            _verify_self_digest(attempt, "attempt_sha256", label=f"attempt {stage_name}/{number}")
            identity = attempt.get("process_identity")
            if identity is not None and not isinstance(identity, Mapping):
                raise ContractError(f"attempt process identity is invalid: {attempt_path}")
            pid = attempt.get("pid")
            if isinstance(identity, Mapping) and _process_identity_is_alive(identity):
                raise ContractError(
                    f"prior stage process is still alive (stage {stage_name}, PID {pid})"
                )
            numbers.append(number)
        return max(numbers, default=0) + 1

    def _choose_argv(self, stage: Mapping[str, Any], attempt_number: int) -> list[str]:
        if attempt_number > 1:
            if stage["resume_argv"] is None:
                raise ContractError(
                    f"stage {stage['name']} has an earlier incomplete attempt but no resume_argv"
                )
            return list(stage["resume_argv"])
        return list(stage["argv"])

    def _recover_completed_process(
        self,
        *,
        stage: Mapping[str, Any],
        attempt_number: int,
        inputs: Sequence[Mapping[str, Any]],
        dependencies: Sequence[Mapping[str, Any]],
        argv: Sequence[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attempt_dir = self.attempt_root / stage["name"] / f"attempt_{attempt_number:04d}"
        attempt_path = attempt_dir / "attempt.json"
        attempt = _read_canonical_json(attempt_path)
        _verify_self_digest(
            attempt, "attempt_sha256", label=f"recover attempt {stage['name']}/{attempt_number}"
        )
        contract = self._execution_contract(stage, inputs, dependencies, argv)
        if (
            attempt.get("schema") != ATTEMPT_SCHEMA
            or attempt.get("plan_sha256") != self.plan_sha256
            or attempt.get("stage") != stage["name"]
            or attempt.get("attempt") != attempt_number
            or attempt.get("state") != "process_complete"
            or attempt.get("returncode") != 0
            or attempt.get("execution_contract_sha256")
            != contract["execution_contract_sha256"]
            or attempt.get("argv") != list(argv)
        ):
            raise ContractError(
                f"orphan outputs are not backed by an exact completed attempt: {attempt_path}"
            )
        _verify_process_isolation(
            attempt.get("process_isolation"), label=f"recovered attempt {stage['name']}"
        )
        launch_record = attempt.get("launch_request")
        if not isinstance(launch_record, Mapping):
            raise ContractError(f"completed attempt lacks launch request: {attempt_path}")
        launch_actual = _verify_path_record(
            launch_record, label=f"recover launch request {stage['name']}"
        )
        expected_launch_path = (attempt_dir / "launch_request.json").resolve()
        if Path(launch_actual["path"]).resolve() != expected_launch_path:
            raise ContractError(f"completed attempt launch path mismatch: {attempt_path}")
        expected_launch = {
            "schema": LAUNCH_REQUEST_SCHEMA,
            "argv": list(argv),
            "argv_sha256": canonical_digest(list(argv)),
            "environment_sha256": self.plan["environment_sha256"],
            "working_directory": stage["working_directory"],
            "execution_contract_sha256": contract["execution_contract_sha256"],
        }
        if _read_canonical_json(expected_launch_path) != expected_launch:
            raise ContractError(f"completed attempt launch payload mismatch: {attempt_path}")
        stdout_path = Path(str(attempt.get("stdout", "")))
        stderr_path = Path(str(attempt.get("stderr", "")))
        logs = {
            "stdout": _path_record(stdout_path),
            "stderr": _path_record(stderr_path),
            "stdout_tail": _tail(stdout_path),
            "stderr_tail": _tail(stderr_path),
        }
        return attempt, {
            "contract": contract,
            "logs": logs,
            "attempt_path": str(attempt_path),
        }

    def _write_attempt(self, path: Path, payload: Mapping[str, Any]) -> None:
        value = dict(payload)
        value.pop("attempt_sha256", None)
        value = _self_digest_payload(value, "attempt_sha256")
        _atomic_write_canonical(path, value)

    def _run_process(self, *, stage: Mapping[str, Any], argv: Sequence[str],
                     attempt_number: int, inputs: Sequence[Mapping[str, Any]],
                     dependencies: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        stage_name = stage["name"]
        attempt_dir = self.attempt_root / stage_name / f"attempt_{attempt_number:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        attempt_path = attempt_dir / "attempt.json"
        contract = self._execution_contract(stage, inputs, dependencies, argv)
        launch_request_path = attempt_dir / "launch_request.json"
        launch_request = {
            "schema": LAUNCH_REQUEST_SCHEMA,
            "argv": list(argv),
            "argv_sha256": canonical_digest(list(argv)),
            "environment_sha256": self.plan["environment_sha256"],
            "working_directory": stage["working_directory"],
            "execution_contract_sha256": contract["execution_contract_sha256"],
        }
        _create_once_canonical(launch_request_path, launch_request)
        launch_request_sha256 = sha256_file(launch_request_path)
        attempt: dict[str, Any] = {
            "schema": ATTEMPT_SCHEMA,
            "plan_sha256": self.plan_sha256,
            "stage": stage_name,
            "resource_class": stage["resource_class"],
            "accounting_poll_seconds": self.poll_seconds,
            "attempt": attempt_number,
            "execution_contract_sha256": contract["execution_contract_sha256"],
            "argv": list(argv),
            "launch_request": {
                "path": str(launch_request_path),
                "bytes": launch_request_path.stat().st_size,
                "sha256": launch_request_sha256,
            },
            "state": "starting",
            "pid": None,
            "process_identity": None,
            "started_utc": utc_now(),
            "updated_utc": utc_now(),
            "ended_utc": None,
            "elapsed_seconds": 0.0,
            "cpu_seconds": 0.0,
            "peak_rss_bytes": 0,
            "returncode": None,
            "failure": None,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "progress": None,
        }
        self._write_attempt(attempt_path, attempt)
        environment = dict(self.plan["environment"])
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        start = time.monotonic()
        process: subprocess.Popen[Any] | None = None
        process_job: _KillOnCloseProcessJob | None = None
        try:
            process_job = _KillOnCloseProcessJob()
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                bootstrap_argv = [
                    str(self.plan["runtime"]["python_executable"]),
                    "-B",
                    str(BOOTSTRAP_PATH),
                    "--request",
                    str(launch_request_path),
                    "--request-sha256",
                    launch_request_sha256,
                ]
                process = subprocess.Popen(
                    bootstrap_argv,
                    cwd=stage["working_directory"],
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=creationflags,
                    close_fds=True,
                )
                # Assign before recording RUNNING and before the child receives
                # GO.  If the supervisor disappears after this point, Windows
                # closes the job handle and kills bootstrap + target together.
                process_job.assign(process)
                self._current_process = process
                attempt["pid"] = process.pid
                attempt["process_identity"] = _process_identity(process.pid)
                attempt["process_isolation"] = {
                    "kind": "windows_job_object" if os.name == "nt" else "process_tree",
                    "kill_on_supervisor_close": os.name == "nt",
                    "assignment_verified": True,
                }
                attempt["state"] = "running"
                # Close the Popen -> durable PID window before any polling,
                # logging, or resource inspection can fail.
                attempt["updated_utc"] = utc_now()
                self._write_attempt(attempt_path, attempt)
                if process.stdin is None:
                    raise ContractError("stage bootstrap stdin pipe is unavailable")
                process.stdin.write(f"GO {launch_request_sha256}\n".encode("ascii"))
                process.stdin.flush()
                process.stdin.close()
                progress_re = re.compile(stage["progress_regex"])
                last_progress: dict[str, int | float] | None = None
                last_disk_check = -float("inf")
                while True:
                    returncode = process.poll()
                    cpu, rss = _process_usage(process.pid)
                    attempt["cpu_seconds"] = max(float(attempt["cpu_seconds"]), cpu)
                    attempt["peak_rss_bytes"] = max(int(attempt["peak_rss_bytes"]), rss)
                    attempt["elapsed_seconds"] = max(0.0, time.monotonic() - start)
                    attempt["updated_utc"] = utc_now()
                    for line in reversed(_tail(stdout_path, max_lines=30)):
                        match = progress_re.search(line)
                        if match:
                            done = int(match.group("done"))
                            total = int(match.group("total"))
                            if total > 0 and 0 <= done <= total:
                                last_progress = {
                                    "done": done,
                                    "total": total,
                                    "fraction": done / total,
                                }
                            break
                    attempt["progress"] = last_progress
                    self._write_attempt(attempt_path, attempt)
                    self._safe_report(
                        state="running", current_stage=stage_name,
                        message=f"stage {stage_name} attempt {attempt_number} is running",
                    )
                    elapsed = float(attempt["elapsed_seconds"])
                    violation: str | None = None
                    if elapsed > float(stage["timeout_seconds"]):
                        violation = f"stage timeout exceeded: {elapsed:.1f}s"
                    elif rss > int(stage["max_rss_bytes"]):
                        violation = f"stage RSS cap exceeded: {rss}"
                    # CPU/GPU/wall accounting is checked every supervisor poll
                    # (five seconds in production).  Filesystem scans are the
                    # only expensive checks and remain on a 30-second cadence.
                    cumulative_cpu, cumulative_wall, cumulative_gpu = (
                        self._cumulative_attempt_usage()
                    )
                    global_caps = self.plan["global_caps"]
                    if violation is None and cumulative_cpu > float(global_caps["max_cpu_seconds"]):
                        violation = f"aggregate CPU cap exceeded: {cumulative_cpu:.3f}s"
                    elif violation is None and cumulative_gpu > float(global_caps["max_gpu_seconds"]):
                        violation = f"aggregate GPU-stage cap exceeded: {cumulative_gpu:.3f}s"
                    elif violation is None and cumulative_wall > float(global_caps["max_wall_seconds"]):
                        violation = f"aggregate durable wall cap exceeded: {cumulative_wall:.3f}s"
                    if violation is None and elapsed - last_disk_check >= 30.0:
                        last_disk_check = elapsed
                        free = shutil.disk_usage(self.root).free
                        artifact_bytes = _directory_bytes(self.root)
                        if free < int(stage["min_free_bytes"]):
                            violation = f"minimum free disk violated: {free}"
                        elif artifact_bytes > int(global_caps["max_artifact_bytes"]):
                            violation = f"aggregate artifact cap exceeded: {artifact_bytes}"
                    if violation is not None:
                        if process_job is None:
                            raise ContractError("stage process job was not initialized")
                        process_job.terminate(process)
                        raise ResourceLimitError(violation)
                    if returncode is not None:
                        break
                    time.sleep(self.poll_seconds)
                if process_job is None:
                    raise ContractError("stage process job was not initialized")
                process_job.assert_empty_after_root_exit()
                attempt["returncode"] = int(returncode)
                attempt["state"] = "process_complete" if returncode == 0 else "process_failed"
                attempt["ended_utc"] = utc_now()
                attempt["updated_utc"] = attempt["ended_utc"]
                attempt["elapsed_seconds"] = max(0.0, time.monotonic() - start)
                attempt["cpu_seconds"] = float(attempt["cpu_seconds"]) + (
                    self.poll_seconds * max(1, os.cpu_count() or 1)
                )
                self._write_attempt(attempt_path, attempt)
                if returncode != 0:
                    raise StageFailure(f"stage {stage_name} exited with code {returncode}")
        except BaseException as exc:
            if process is not None:
                try:
                    if process_job is not None:
                        process_job.terminate(process)
                    else:
                        _terminate_process_tree(process)
                except BaseException as termination_exc:
                    exc = ContractError(
                        f"{type(exc).__name__}: {exc}; additionally failed to terminate "
                        f"the owned process tree: {termination_exc}"
                    )
            attempt["state"] = "failed"
            attempt["ended_utc"] = utc_now()
            attempt["updated_utc"] = attempt["ended_utc"]
            attempt["elapsed_seconds"] = max(0.0, time.monotonic() - start)
            attempt["cpu_seconds"] = float(attempt["cpu_seconds"]) + (
                self.poll_seconds * max(1, os.cpu_count() or 1)
            )
            attempt["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            self._write_attempt(attempt_path, attempt)
            raise exc
        finally:
            self._current_process = None
            if process_job is not None:
                active_exception = sys.exc_info()[0] is not None
                try:
                    process_job.close()
                except BaseException as close_exc:
                    if not active_exception:
                        raise
                    print(
                        f"E26 job-handle close failed while propagating another error: {close_exc}",
                        file=sys.stderr,
                        flush=True,
                    )
        logs = {
            "stdout": _path_record(stdout_path),
            "stderr": _path_record(stderr_path),
            "stdout_tail": _tail(stdout_path),
            "stderr_tail": _tail(stderr_path),
        }
        return attempt, {"contract": contract, "logs": logs, "attempt_path": str(attempt_path)}

    def _commit_or_verify_main_output_seal(
        self,
        stage: Mapping[str, Any],
        attempt_number: int,
        execution_contract_sha256: str,
        outputs: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attempt_dir = self.attempt_root / stage["name"] / f"attempt_{attempt_number:04d}"
        path = attempt_dir / "main_outputs_seal.json"
        body = {
            "schema": "pazzle-e26-autonomous-main-output-seal-v1",
            "plan_sha256": self.plan_sha256,
            "stage": stage["name"],
            "attempt": attempt_number,
            "execution_contract_sha256": execution_contract_sha256,
            "outputs": list(outputs),
        }
        payload = _self_digest_payload(body, "seal_sha256")
        if path.exists():
            existing = _read_canonical_json(path)
            if existing != payload:
                raise ContractError(
                    f"main-output seal conflicts for stage {stage['name']}: {path}"
                )
        else:
            _create_once_canonical(path, payload)
        verified = _read_canonical_json(path)
        _verify_self_digest(
            verified, "seal_sha256", label=f"main-output seal {stage['name']}"
        )
        for index, record in enumerate(verified["outputs"]):
            _verify_path_record(
                record, label=f"main-output seal {stage['name']} output[{index}]"
            )
        return verified, _path_record(path)

    def _run_verifier(
        self,
        stage: Mapping[str, Any],
        attempt_number: int,
        main_outputs: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        argv = stage["verifier_argv"]
        if not argv:
            return None
        verifier_stage = {
            **stage,
            "name": stage["name"] + "_verifier",
            "argv": argv,
            "resume_argv": None,
            "outputs": [],
            "verifier_argv": [],
            "gate": None,
            "timeout_seconds": min(int(stage["timeout_seconds"]), 3600),
            "max_attempts": 1,
        }
        # Verifier attempts live under a distinct stage-like directory and are
        # charged to the same durable aggregate CPU budget.  A complete verifier
        # can be authenticated after a supervisor interruption.  A dead,
        # incomplete read-only verifier may be rerun under a fresh attempt ID.
        verifier_attempt_number = self._next_attempt_number(verifier_stage["name"])
        if verifier_attempt_number > 1:
            prior_number = verifier_attempt_number - 1
            prior_path = (
                self.attempt_root / verifier_stage["name"]
                / f"attempt_{prior_number:04d}" / "attempt.json"
            )
            prior = _read_canonical_json(prior_path)
            if prior.get("state") == "process_complete" and prior.get("returncode") == 0:
                attempt, details = self._recover_completed_process(
                    stage=verifier_stage,
                    attempt_number=prior_number,
                    inputs=main_outputs,
                    dependencies=[],
                    argv=argv,
                )
            elif prior.get("state") == "failed":
                raise StageFailure(
                    f"verifier for {stage['name']} previously failed; inspect {prior_path}"
                )
            else:
                attempt, details = self._run_process(
                    stage=verifier_stage,
                    argv=argv,
                    attempt_number=verifier_attempt_number,
                    inputs=main_outputs,
                    dependencies=[],
                )
        else:
            attempt, details = self._run_process(
                stage=verifier_stage,
                argv=argv,
                attempt_number=verifier_attempt_number,
                inputs=main_outputs,
                dependencies=[],
            )
        return {
            "attempt": _read_canonical_json(Path(details["attempt_path"])),
            "logs": details["logs"],
            "execution_contract_sha256": details["contract"]["execution_contract_sha256"],
        }

    def _run_stage(self, stage: Mapping[str, Any]) -> StageOutcome:
        self._verify_plan_provenance()
        receipt_path = self._receipt_path(stage["name"])
        if receipt_path.exists():
            receipt = self._verify_receipt(stage, receipt_path)
            return StageOutcome(receipt=receipt, scientific_pass=bool(receipt["scientific_pass"]))
        inputs = self._verify_explicit_inputs(stage)
        dependencies = self._dependency_records(stage)
        attempt_number = self._next_attempt_number(stage["name"])
        orphan_outputs = [
            output["path"] for output in stage["outputs"]
            if Path(output["path"]).exists()
        ]
        if orphan_outputs:
            if attempt_number <= 1:
                raise ContractError(
                    f"stage {stage['name']} has outputs without any attempt: {orphan_outputs}"
                )
            completed_number = attempt_number - 1
            completed_argv = self._choose_argv(stage, completed_number)
            attempt, details = self._recover_completed_process(
                stage=stage,
                attempt_number=completed_number,
                inputs=inputs,
                dependencies=dependencies,
                argv=completed_argv,
            )
            attempt_number = completed_number
            argv = completed_argv
            self._safe_emit_event(
                "stage_main_recovered",
                stage=stage["name"],
                attempt=attempt_number,
            )
        else:
            while True:
                attempt_number = self._next_attempt_number(stage["name"])
                self._enforce_global_caps(stage)
                if attempt_number > int(stage["max_attempts"]):
                    raise StageFailure(
                        f"stage {stage['name']} exhausted {stage['max_attempts']} predeclared attempts"
                    )
                argv = self._choose_argv(stage, attempt_number)
                self._safe_emit_event(
                    "stage_started", stage=stage["name"], attempt=attempt_number, argv=argv
                )
                try:
                    attempt, details = self._run_process(
                        stage=stage,
                        argv=argv,
                        attempt_number=attempt_number,
                        inputs=inputs,
                        dependencies=dependencies,
                    )
                    break
                except StageFailure as exc:
                    if (
                        attempt_number >= int(stage["max_attempts"])
                        or stage["resume_argv"] is None
                    ):
                        raise
                    self._safe_emit_event(
                        "stage_retry_scheduled",
                        stage=stage["name"],
                        failed_attempt=attempt_number,
                        next_attempt=attempt_number + 1,
                        reason=str(exc),
                    )
                    self._safe_report(
                        state="retrying",
                        current_stage=stage["name"],
                        message=(
                            f"stage {stage['name']} failed attempt {attempt_number}; "
                            "starting its frozen resume command"
                        ),
                        failure={"type": type(exc).__name__, "message": str(exc)},
                    )
                    time.sleep(min(5.0, self.poll_seconds))
        outputs_before_verifier = self._validate_outputs(stage)
        main_output_seal, main_output_seal_record = self._commit_or_verify_main_output_seal(
            stage,
            attempt_number,
            details["contract"]["execution_contract_sha256"],
            outputs_before_verifier,
        )
        self._enforce_global_caps(stage)
        verifier = self._run_verifier(stage, attempt_number, outputs_before_verifier)
        # The verifier is an external process.  Re-authenticate every immutable
        # capability it could have touched before a receipt can bless anything.
        self._verify_plan_provenance()
        if self._verify_explicit_inputs(stage) != inputs:
            raise ContractError(f"verifier mutated explicit inputs for stage {stage['name']}")
        if self._dependency_records_for_receipt(stage) != dependencies:
            raise ContractError(f"verifier mutated dependency evidence for stage {stage['name']}")
        outputs = self._validate_outputs(stage)
        if outputs != outputs_before_verifier:
            raise ContractError(
                f"verifier mutated main outputs for stage {stage['name']}"
            )
        scientific_pass, observed_gate = self._gate_value(stage)
        receipt_body: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "pipeline_id": self.plan["pipeline_id"],
            "plan_sha256": self.plan_sha256,
            "sources_sha256": self.plan["sources_sha256"],
            "stage": stage["name"],
            "attempt": attempt_number,
            "execution_contract_sha256": details["contract"]["execution_contract_sha256"],
            "argv": argv,
            "dependencies": dependencies,
            "inputs": inputs,
            "outputs": outputs,
            "main_output_seal": main_output_seal_record,
            "main_output_seal_sha256": main_output_seal["seal_sha256"],
            "logs": details["logs"],
            "attempt_path": details["attempt_path"],
            "resource": {
                "elapsed_seconds": attempt["elapsed_seconds"],
                "cpu_seconds": attempt["cpu_seconds"],
                "peak_rss_bytes": attempt["peak_rss_bytes"],
            },
            "verifier": verifier,
            "gate": {
                "contract": stage["gate"],
                "observed_value": observed_gate,
                "passed": scientific_pass,
            } if stage["gate"] is not None else None,
            "scientific_pass": scientific_pass,
            "completed_utc": utc_now(),
        }
        receipt = _self_digest_payload(receipt_body, "receipt_sha256")
        _create_once_canonical(receipt_path, receipt)
        verified = self._verify_receipt(stage, receipt_path)
        self._safe_emit_event(
            "stage_committed", stage=stage["name"],
            receipt_sha256=verified["receipt_sha256"], scientific_pass=scientific_pass,
        )
        return StageOutcome(receipt=verified, scientific_pass=scientific_pass)

    def run(self) -> int:
        try:
            self._ensure_runtime_dirs()
        except BaseException as exc:
            failure = {
                "type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc().splitlines()[-80:],
            }
            try:
                self._write_invocation_diagnostic(
                    state="startup_failed",
                    message="runtime directories could not be prepared",
                    failure=failure,
                )
            except BaseException as diagnostic_exc:
                print(f"E26 startup/report failure: {exc}; {diagnostic_exc}", file=sys.stderr)
            return 1

        lock = PipelineLock(
            self.lock_path,
            plan_sha256=self.plan_sha256,
            recover_stale=self.recover_stale_lock,
        )
        try:
            lock.acquire()
            self._lock_owned = True
        except BaseException as exc:
            # Never touch shared status/recovery state without owning the lock;
            # a live runner may be updating it.  Write only a unique diagnostic.
            failure = {
                "type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc().splitlines()[-80:],
            }
            try:
                self._write_invocation_diagnostic(
                    state="lock_failed",
                    message="autonomous invocation did not acquire exclusive ownership",
                    failure=failure,
                )
            except BaseException as diagnostic_exc:
                print(f"E26 lock/report failure: {exc}; {diagnostic_exc}", file=sys.stderr)
            return 1

        result = 1
        try:
            if self.final_path.exists():
                prior_final = self._verify_final_report()
                # Idempotent terminal replay is read-only apart from acquiring
                # and releasing the transient ownership lock.
                terminal_code = 0 if prior_final["state"] == "complete" else 20
                try:
                    lock.release()
                finally:
                    self._lock_owned = False
                return terminal_code
            self._safe_emit_event("pipeline_invocation_started", pid=os.getpid())
            self._safe_report(
                state="starting", current_stage=None, message="plan verified; starting"
            )
            for stage in self.plan["stages"]:
                self._safe_report(
                    state="verifying", current_stage=stage["name"],
                    message=f"verifying or executing stage {stage['name']}",
                )
                outcome = self._run_stage(stage)
                if not outcome.scientific_pass:
                    raise ScientificGateFailure(
                        f"stage {stage['name']} reached its frozen scientific FAIL"
                    )
            report = self._safe_report(
                state="complete", current_stage=None,
                message="all frozen stages completed and passed", final=True,
            )
            if "final_report_sha256" not in report:
                raise ContractError("terminal PASS report could not be authenticated")
            self._verify_final_report()
            self._safe_emit_event(
                "pipeline_complete", final_report_sha256=report["final_report_sha256"]
            )
            result = 0
        except ScientificGateFailure as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
            report = self._safe_report(
                state="scientific_fail", current_stage=None,
                message="pipeline stopped at a predeclared scientific gate",
                failure=failure, final=True,
            )
            if "final_report_sha256" not in report:
                result = 1
            else:
                self._verify_final_report()
                self._safe_emit_event(
                    "pipeline_scientific_fail",
                    final_report_sha256=report["final_report_sha256"],
                    failure=failure,
                )
                result = 20
        except BaseException as exc:
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc().splitlines()[-80:],
            }
            self._safe_report(
                state="blocked", current_stage=None,
                message="pipeline stopped safely; see failure and resume command",
                failure=failure,
            )
            self._safe_emit_event("pipeline_blocked", failure=failure)
            result = 1
        finally:
            try:
                lock.release()
            except BaseException as exc:
                result = 1
                failure = {"type": type(exc).__name__, "message": str(exc)}
                try:
                    self._write_invocation_diagnostic(
                        state="lock_release_failed",
                        message="pipeline result exists but lock release failed",
                        failure=failure,
                    )
                except BaseException as diagnostic_exc:
                    print(f"E26 lock-release/report failure: {exc}; {diagnostic_exc}", file=sys.stderr)
            finally:
                self._lock_owned = False
        return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="freeze a canonical plan from an audited spec")
    freeze.add_argument("--spec", type=Path, required=True)
    freeze.add_argument("--plan", type=Path, required=True)
    run = sub.add_parser("run", help="run or resume a frozen plan")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    run.add_argument("--recover-stale-lock", action="store_true")
    run.add_argument("--emergency-dir", type=Path, required=True)
    run.add_argument("--emergency-reserve", type=Path, required=True)
    run.add_argument("--invocation-id", required=True)
    verify = sub.add_parser("verify", help="verify plan, receipts, and current report")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--plan-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "freeze":
        plan = freeze_plan(args.spec, args.plan)
        print(canonical_json({
            "plan": str(Path(args.plan).resolve()),
            "plan_sha256": plan["plan_sha256"],
        }).decode("utf-8"), flush=True)
        return 0
    if args.command == "run":
        try:
            runner = AutonomousRunner(
                args.plan,
                args.plan_sha256,
                recover_stale_lock=bool(args.recover_stale_lock),
            )
            emergency_directory = args.emergency_dir.resolve()
            reserve = args.emergency_reserve.resolve()
            fixed_emergency_root = (
                Path(r"E:\pazzle_work\e26_contextual_edge")
                / "orchestrator" / "reports" / "emergency"
            ).resolve()
            _require_e_path(emergency_directory, "emergency_dir", allow_non_e=False)
            if (
                not _is_within(emergency_directory, fixed_emergency_root)
                or not _is_within(reserve, emergency_directory)
            ):
                raise ContractError("startup emergency reserve location is invalid")
            if not reserve.is_file() or reserve.stat().st_size != 262_144:
                raise ContractError("startup emergency reserve is missing or has wrong size")
            reserve.unlink()
            return runner.run()
        except BaseException as exc:
            try:
                report = _write_startup_emergency(
                    emergency_dir=args.emergency_dir,
                    invocation_id=args.invocation_id,
                    reserve_path=args.emergency_reserve,
                    plan_path=args.plan,
                    expected_plan_sha256=args.plan_sha256,
                    failure=exc,
                )
                print(canonical_json(report).decode("utf-8"), file=sys.stderr, flush=True)
            except BaseException as report_exc:
                print(
                    f"E26 startup failure: {exc}; emergency report failure: {report_exc}",
                    file=sys.stderr,
                    flush=True,
                )
            return 1
    runner = AutonomousRunner(args.plan, args.plan_sha256)
    snapshot = runner.verify_snapshot()
    print(canonical_json(snapshot).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
