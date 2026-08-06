"""Build the self-contained, inference-only Kaggle notebook for rank96.

The builder intentionally does not reuse ``build_kaggle.py`` or the historical
training notebook.  It packages the production inference entry point and its
transitive local imports, embeds the three frozen checkpoints, and refuses to
build unless both the checkpoint hashes and the immutable rank96 gate proof
match the predeclared contract.

The resulting notebook needs only the Kaggle competition data mount.  It runs
on a T4, accepts a hash-checked partial resume snapshot, writes predictions
atomically through ``infer_rank96.py``, and verifies the final 700-image ZIP.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "pazzle-rank96-kaggle-bundle-v1"
NOTEBOOK_NAME = "pazzle_rank96_inference.ipynb"
ENTRYPOINT = "infer_rank96.py"
CHAMPION_REPORT = Path("artifacts/frozen_gate/report_budget96_vs512_v1.json")
CHAMPION_REPORT_SHA256 = "2ea813849d45562d2e5af77ac73fdb1258a2b900dbc6290b645abf12b3810db6"
EXPECTED_GATE_ROOT_SHA256 = "ee3d74662f5326fbd1069763fd7b96dc3adb41bde0117cba1d78ff067c6bf23d"
MAX_INFERENCE_SECONDS = 7_000  # leaves 200 s inside the authorised two GPU hours
GRACEFUL_RUNTIME_SECONDS = 6_900  # inference writes a resumable partial manifest first
EXPECTED_TEST_IMAGES = 700

EXPECTED_CHECKPOINTS: dict[str, dict[str, str]] = {
    "ranker": {
        "path": "artifacts/candidate_rank/rank_v2w64_best.pt",
        "filename": "rank_v2w64_best.pt",
        "sha256": "42685373b1a450a4cb3d7a9b22370dfcfaa2335e9e8ada609f21b7cc64abbfbc",
    },
    "affinity_primary": {
        "path": "artifacts/macro_affinity/affinity_r1_1200_best.pt",
        "filename": "affinity_r1_1200_best.pt",
        "sha256": "708565329c7661a965215d98e85f462a90930071f36a0f75b4813c0c5797ec4f",
    },
    "affinity_secondary": {
        "path": "artifacts/macro_affinity/affinity_r3_1000_best.pt",
        "filename": "affinity_r3_1000_best.pt",
        "sha256": "0fceafdb110bde59149fe1ad1e800a69d116041bc627af369aaecd60be53b6c8",
    },
}

# ``infer_rank96.py`` must expose this literal as RANK96_CONTRACT.  Parsing the
# literal avoids importing a GPU pipeline while building and makes accidental
# protocol drift fail before any notebook is emitted.
EXPECTED_INFERENCE_CONTRACT: dict[str, Any] = {
    "schema": "pazzle-rank96-inference-v1",
    "image_size": 480,
    "grid": 24,
    "tile_size": 20,
    "num_tiles": 576,
    "orientation": "fixed",
    "orientation_detail": "upright_tiles_no_orientation_search",
    "candidate_k_per_encoder": 64,
    "candidate_union": "ordered_deduplicated_primary_then_secondary",
    "candidate_score": "candidate_ranker_raw_logits",
    "dense_conversion": "eval_seeded_qap.dense_rd",
    "dense_device": "cpu_float32",
    "solver": "solve_buddies.solve_buddies_from_scores",
    "max_edges": 96,
    "min_margin": 0.0,
    "repair_passes": 0,
    "restoration": "opencv_fast_nlm_colored",
    "nlm_h": 10,
    "nlm_h_color": 10,
    "nlm_template_window": 7,
    "nlm_search_window": 21,
}


class BuildContractError(RuntimeError):
    """Raised when a supposedly frozen input differs from the build contract."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise BuildContractError(f"missing {label}: {resolved}")
    return resolved


def extract_rank96_contract(entrypoint: Path) -> dict[str, Any]:
    """Read ``RANK96_CONTRACT`` without importing the inference module."""

    source = _require_file(entrypoint, "rank96 inference entrypoint").read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(entrypoint))
    except SyntaxError as error:
        raise BuildContractError(f"cannot parse {entrypoint}: {error}") from error
    value_node: ast.AST | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RANK96_CONTRACT"
            for target in node.targets
        ):
            value_node = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "RANK96_CONTRACT"
        ):
            value_node = node.value
    if value_node is None:
        raise BuildContractError(
            f"{entrypoint} must expose a literal RANK96_CONTRACT for fail-closed packaging"
        )
    try:
        value = ast.literal_eval(value_node)
    except (TypeError, ValueError) as error:
        raise BuildContractError("RANK96_CONTRACT must be a literal mapping") from error
    if not isinstance(value, dict):
        raise BuildContractError("RANK96_CONTRACT must be a dictionary")
    return value


def validate_inference_contract(actual: Mapping[str, Any]) -> None:
    missing = sorted(set(EXPECTED_INFERENCE_CONTRACT) - set(actual))
    if missing:
        raise BuildContractError(f"RANK96_CONTRACT is missing fields: {missing}")
    changed = {
        key: {"expected": expected, "actual": actual[key]}
        for key, expected in EXPECTED_INFERENCE_CONTRACT.items()
        if actual[key] != expected
    }
    if changed:
        raise BuildContractError(
            "rank96 production inference differs from the frozen winner: "
            + json.dumps(changed, sort_keys=True)
        )


def _load_champion_proof(workspace: Path) -> dict[str, Any]:
    path = _require_file(workspace / CHAMPION_REPORT, "immutable rank96 gate report")
    digest = sha256_file(path)
    if digest != CHAMPION_REPORT_SHA256:
        raise BuildContractError(
            f"rank96 gate report hash mismatch: expected {CHAMPION_REPORT_SHA256}, got {digest}"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise BuildContractError(f"invalid rank96 gate report: {error}") from error
    required = {
        "schema": "pazzle-frozen-budget96-vs512-report-v1",
        "scene_count": 24,
        "selection_or_sweep": False,
        "gate_root_sha256": EXPECTED_GATE_ROOT_SHA256,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise BuildContractError(f"champion proof field {key!r} is not frozen value {expected!r}")
    arms = report.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"budget_96", "budget_512"}:
        raise BuildContractError("champion proof must compare exactly budget_96 and budget_512")
    for name, budget in (("budget_96", 96), ("budget_512", 512)):
        arm = arms[name]
        expected_arm = {
            "max_edges": budget,
            "min_margin": 0.0,
            "repair_passes": 0,
            "candidate_source": "frozen_i11_candidate_ranker_raw_logits",
            "dense_conversion": "eval_seeded_qap.dense_rd",
            "solver": "solve_buddies.solve_buddies_from_scores",
        }
        if arm != expected_arm:
            raise BuildContractError(f"champion proof arm {name} differs from the fixed protocol")
    restoration = report.get("restoration")
    if restoration != {
        "h": 10,
        "h_color": 10,
        "method": "opencv_fast_nlm_colored",
        "search_window": 21,
        "template_window": 7,
    }:
        raise BuildContractError("champion proof restoration contract differs from fixed NLM h=10")
    primary = report.get("primary", {})
    final = report.get("fixed_nlm_final", {})
    if primary.get("metric") != "paired_mean_solve_ssim_budget96_minus_budget512" or not (
        isinstance(primary.get("value"), (int, float)) and primary["value"] > 0.0
    ):
        raise BuildContractError("champion proof does not contain a positive solve-SSIM delta")
    if final.get("metric") != "paired_mean_final_ssim_budget96_minus_budget512" or not (
        isinstance(final.get("value"), (int, float)) and final["value"] > 0.0
    ):
        raise BuildContractError("champion proof does not contain a positive final-SSIM delta")
    proof_hashes = report.get("contracts", {}).get("checkpoints", {})
    for role, expected in EXPECTED_CHECKPOINTS.items():
        if proof_hashes.get(role) != expected["sha256"]:
            raise BuildContractError(f"champion proof checkpoint hash differs for {role}")
    return {
        "path": CHAMPION_REPORT.as_posix(),
        "sha256": digest,
        "gate_root_sha256": report["gate_root_sha256"],
        "scene_count": report["scene_count"],
        "solve_ssim_delta": float(primary["value"]),
        "final_ssim_delta": float(final["value"]),
    }


def _local_import_names(path: Path) -> set[str]:
    try:
        # ``utf-8-sig`` strips an optional BOM.  CPython accepts such source
        # files during normal import, so dependency discovery must mirror it.
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as error:
        raise BuildContractError(f"cannot parse local module {path}: {error}") from error
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            result.add(node.module.split(".", 1)[0])
    return result


def discover_source_closure(src_dir: Path, entrypoint: str = ENTRYPOINT) -> list[Path]:
    """Return the deterministic transitive closure of flat local imports."""

    src_dir = src_dir.resolve()
    entry = _require_file(src_dir / entrypoint, "rank96 inference entrypoint")
    pending = [entry]
    selected: dict[str, Path] = {}
    while pending:
        path = pending.pop()
        if path.name in selected:
            continue
        selected[path.name] = path
        for module in sorted(_local_import_names(path)):
            candidate = src_dir / f"{module}.py"
            if candidate.is_file() and candidate.name not in selected:
                pending.append(candidate)
    return [selected[name] for name in sorted(selected)]


def _validate_checkpoints(workspace: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    records: dict[str, dict[str, Any]] = {}
    contents: dict[str, bytes] = {}
    for role, expected in EXPECTED_CHECKPOINTS.items():
        path = _require_file(workspace / expected["path"], f"{role} checkpoint")
        content = path.read_bytes()
        digest = sha256_bytes(content)
        if digest != expected["sha256"]:
            raise BuildContractError(
                f"{role} checkpoint hash mismatch: expected {expected['sha256']}, got {digest}"
            )
        records[role] = {
            "filename": expected["filename"],
            "sha256": digest,
            "bytes": len(content),
        }
        contents[role] = content
    return records, contents


def _validate_overrides(overrides_dir: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    if overrides_dir is None:
        return {}, {}
    root = overrides_dir.resolve()
    if not root.is_dir():
        raise BuildContractError(f"override directory does not exist: {root}")
    try:
        from PIL import Image
    except ImportError as error:
        raise BuildContractError("Pillow is required to validate exact source overrides") from error
    nested = [path for path in root.rglob("*.png") if path.parent != root]
    if nested:
        raise BuildContractError("override PNGs must be direct children of the override directory")
    paths = sorted(root.glob("*.png"), key=lambda path: path.name)
    if not paths:
        raise BuildContractError(f"override directory has no PNGs: {root}")
    if len(paths) > EXPECTED_TEST_IMAGES:
        raise BuildContractError("override directory contains more than 700 PNGs")
    records: dict[str, dict[str, Any]] = {}
    contents: dict[str, bytes] = {}
    for path in paths:
        if Path(path.name).name != path.name:
            raise BuildContractError(f"unsafe override filename: {path.name!r}")
        with Image.open(path) as image:
            image.load()
            if image.size != (480, 480) or image.mode != "RGB":
                raise BuildContractError(
                    f"override {path.name} must be RGB 480x480, got {image.mode} {image.size}"
                )
            if image.format != "PNG":
                raise BuildContractError(f"override {path.name} is not actually a PNG")
        content = path.read_bytes()
        records[path.name] = {"sha256": sha256_bytes(content), "bytes": len(content)}
        contents[path.name] = content
    return records, contents


def _b64_chunks(content: bytes, raw_chunk_size: int = 768 * 1024) -> list[str]:
    return [
        base64.b64encode(content[start : start + raw_chunk_size]).decode("ascii")
        for start in range(0, len(content), raw_chunk_size)
    ]


def _code_cell(cell_id: str, source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _markdown_cell(cell_id: str, source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source}


def _blob_cell(role: str, filename: str, content: bytes, expected_hash: str, target: str) -> str:
    chunks = json.dumps(_b64_chunks(content), separators=(",", ":"))
    return f'''# Embedded immutable {role}\n_write_blob(\n    {filename!r},\n    {chunks},\n    {expected_hash!r},\n    {target},\n)\n'''


def validate_notebook_structure(notebook: Mapping[str, Any]) -> None:
    """Compile every generated code cell and reject accidental credentials."""

    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise BuildContractError("generated notebook has an invalid nbformat structure")
    identifiers: set[str] = set()
    for index, cell in enumerate(notebook["cells"]):
        identifier = cell.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise BuildContractError(f"generated notebook has an invalid/duplicate cell id at {index}")
        identifiers.add(identifier)
        if cell.get("cell_type") == "code":
            source = cell.get("source")
            if not isinstance(source, str):
                raise BuildContractError(f"generated code cell {identifier} does not contain text")
            try:
                ast.parse(source, filename=f"<rank96-notebook:{identifier}>")
            except SyntaxError as error:
                raise BuildContractError(
                    f"generated notebook code cell {identifier} does not compile: {error}"
                ) from error
    raw = canonical_json(notebook)
    for marker in (b"WANDB_API_KEY", b"wandb_v1_"):
        if marker in raw:
            raise BuildContractError("generated inference notebook unexpectedly contains a credential marker")


def _notebook(
    manifest: Mapping[str, Any],
    source_contents: Mapping[str, bytes],
    checkpoint_contents: Mapping[str, bytes],
    override_contents: Mapping[str, bytes],
) -> dict[str, Any]:
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    source_payload = {
        name: _b64_chunks(content) for name, content in sorted(source_contents.items())
    }
    source_json = json.dumps(source_payload, sort_keys=True, separators=(",", ":"))
    bootstrap = f'''from pathlib import Path
import base64, hashlib, importlib, json, os, shutil, subprocess, sys, time, zipfile

BUNDLE_MANIFEST = json.loads({manifest_json!r})
WORK = Path("/kaggle/working/rank96")
SRC = WORK / "src"
CKPT = WORK / "checkpoints"
OUTPUT = WORK / "output"
for path in (SRC, CKPT, OUTPUT):
    path.mkdir(parents=True, exist_ok=True)
os.environ["PAZZLE_DATA"] = str(WORK / "data_contract")
os.environ["PAZZLE_WORK"] = str(WORK / "runtime")

for package in ("cv2", "numba", "numpy", "PIL", "scipy", "skimage", "torch"):
    try:
        importlib.import_module(package)
    except ImportError as error:
        raise RuntimeError(f"Kaggle image is missing required package {{package}}") from error
import torch
if not torch.cuda.is_available():
    raise RuntimeError("rank96 notebook requires a CUDA accelerator")
GPU_NAMES = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if not any("T4" in name.upper() for name in GPU_NAMES):
    raise RuntimeError(f"this frozen runtime is authorised for Kaggle T4 only, got {{GPU_NAMES}}")
print("bundle", BUNDLE_MANIFEST["bundle_id"], "GPU", GPU_NAMES)

def _sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()

def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def _write_blob(filename, chunks, expected_sha256, directory):
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    if destination.exists():
        actual = _sha256_file(destination)
        if actual != expected_sha256:
            raise RuntimeError(f"existing artifact {{destination}} has hash {{actual}}, expected {{expected_sha256}}")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for chunk in chunks:
            handle.write(base64.b64decode(chunk, validate=True))
    actual = _sha256_file(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"decoded artifact {{filename}} has hash {{actual}}, expected {{expected_sha256}}")
    temporary.replace(destination)
    return destination
'''
    source_cell = f'''SOURCE_FILES_B64 = json.loads({source_json!r})
for name, chunks in SOURCE_FILES_B64.items():
    record = BUNDLE_MANIFEST["sources"][name]
    _write_blob(name, chunks, record["sha256"], SRC)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
print("verified source closure:", len(SOURCE_FILES_B64), "files")
'''
    input_cell = f'''from PIL import Image

EXPECTED_IMAGES = {EXPECTED_TEST_IMAGES}
MAX_INFERENCE_SECONDS = {MAX_INFERENCE_SECONDS}

def _direct_pngs(path):
    return sorted(path.glob("*.png"), key=lambda item: item.name) if path.is_dir() else []

def _materialize_test_zip(zip_path):
    zip_path = Path(zip_path).resolve()
    destination = WORK / "extracted_test"
    marker = destination / ".rank96_test_zip.json"
    zip_sha256 = _sha256_file(zip_path)
    if marker.is_file():
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("zip_sha256") != zip_sha256:
            raise RuntimeError("cached test.zip extraction belongs to different archive bytes")
        records = value.get("records")
        files = _direct_pngs(destination)
        if not isinstance(records, dict) or [path.name for path in files] != sorted(records):
            raise RuntimeError("cached test.zip extraction is incomplete")
        for path in files:
            if _sha256_file(path) != records[path.name]:
                raise RuntimeError(f"cached test.zip extraction hash mismatch: {{path.name}}")
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt test.zip member: {{bad}}")
        members = [item for item in archive.infolist() if not item.is_dir() and Path(item.filename).suffix.lower() == ".png"]
        names = [Path(item.filename).name for item in members]
        if len(names) != EXPECTED_IMAGES or len(set(names)) != EXPECTED_IMAGES:
            raise RuntimeError("test.zip must contain exactly 700 uniquely named PNGs")
        extras = sorted({{path.name for path in _direct_pngs(destination)}} - set(names))
        if extras:
            raise RuntimeError(f"test.zip extraction directory contains unrelated PNGs: {{extras[:8]}}")
        for item, name in sorted(zip(members, names), key=lambda pair: pair[1]):
            if not name or name in (".", ".."):
                raise RuntimeError(f"unsafe test.zip member: {{item.filename!r}}")
            temporary = destination / (name + ".tmp")
            with archive.open(item) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(destination / name)
    marker.write_text(
        json.dumps(
            {{
                "zip_sha256": zip_sha256,
                "records": {{name: _sha256_file(destination / name) for name in sorted(names)}},
            }},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return destination

def _discover_test_dir():
    configured = os.environ.get("PAZZLE_RANK96_TEST_DIR", "").strip()
    if configured:
        candidates = [Path(configured)]
    else:
        roots = [path for path in Path("/kaggle/input").rglob("*") if path.is_dir()]
        candidates = [
            path
            for path in roots
            if len(_direct_pngs(path)) == EXPECTED_IMAGES
            # A completed/near-complete mounted resume snapshot can itself
            # contain 700 PNGs.  Its sibling index proves it is output, not
            # the competition input directory.
            and not (path.parent / "resume_index.json").is_file()
        ]
    candidates = sorted({{path.resolve() for path in candidates}})
    valid = [path for path in candidates if len(_direct_pngs(path)) == EXPECTED_IMAGES]
    if len(valid) == 0:
        configured_zip = os.environ.get("PAZZLE_RANK96_TEST_ZIP", "").strip()
        archives = [Path(configured_zip)] if configured_zip else sorted(Path("/kaggle/input").rglob("test.zip"))
        archives = [path.resolve() for path in archives if path.is_file()]
        if len(archives) == 1:
            valid = [_materialize_test_zip(archives[0]).resolve()]
        elif archives:
            raise RuntimeError(
                "multiple test.zip archives found; set PAZZLE_RANK96_TEST_ZIP explicitly: "
                f"{{archives}}"
            )
    if len(valid) != 1:
        raise RuntimeError(
            "expected exactly one directory/test.zip with 700 PNGs; set "
            f"PAZZLE_RANK96_TEST_DIR or PAZZLE_RANK96_TEST_ZIP explicitly; candidates={{valid}}"
        )
    names = [path.name for path in _direct_pngs(valid[0])]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate test basenames")
    return valid[0], names

TEST_DIR, TEST_NAMES = _discover_test_dir()
print("test directory:", TEST_DIR, "images:", len(TEST_NAMES))

def _validate_png(path):
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (480, 480):
            raise RuntimeError(f"invalid RGB 480x480 PNG: {{path}}")

def _resume_index(path):
    production_manifest_path = WORK / "run_manifest.json"
    production_manifest = None
    completed = {{}}
    if production_manifest_path.is_file():
        production_manifest = json.loads(production_manifest_path.read_text(encoding="utf-8"))
        completed = production_manifest.get("completed")
        if not isinstance(completed, dict):
            raise RuntimeError("production resume manifest has no completed mapping")
    records = {{}}
    for name in TEST_NAMES:
        output = OUTPUT / name
        if not output.is_file():
            continue
        _validate_png(output)
        production_record = completed.get(name)
        if not isinstance(production_record, dict):
            # A hard stop can land after atomic PNG replace but before the
            # production manifest replace.  Exclude that orphan: inference
            # will recompute it, while every recorded item remains provable.
            print("excluding uncommitted resume output:", name)
            continue
        input_sha256 = _sha256_file(TEST_DIR / name)
        output_sha256 = _sha256_file(output)
        contract_digest = production_manifest.get("contract_digest")
        if (
            production_record.get("pipeline_contract_digest") != contract_digest
            or production_record.get("input_sha256") != input_sha256
            or production_record.get("output_sha256") != output_sha256
        ):
            raise RuntimeError(f"production resume record hash mismatch: {{name}}")
        records[name] = {{
            "input_sha256": input_sha256,
            "output_sha256": output_sha256,
        }}
    payload = {{
        "schema": "pazzle-rank96-resume-v1",
        "bundle_id": BUNDLE_MANIFEST["bundle_id"],
        "run_manifest_sha256": (
            _sha256_file(production_manifest_path) if production_manifest_path.is_file() else None
        ),
        "records": records,
    }}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)
    return payload

def _restore_resume(root):
    root = Path(root)
    index_path = root / "resume_index.json"
    if not index_path.is_file():
        raise RuntimeError(f"resume snapshot lacks {{index_path.name}}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "pazzle-rank96-resume-v1" or payload.get("bundle_id") != BUNDLE_MANIFEST["bundle_id"]:
        raise RuntimeError("resume snapshot belongs to another pipeline contract")
    records = payload.get("records")
    if not isinstance(records, dict) or not set(records).issubset(TEST_NAMES):
        raise RuntimeError("resume snapshot contains invalid filenames")
    source_manifest = root / "run_manifest.json"
    manifest_sha256 = payload.get("run_manifest_sha256")
    if records and (
        not source_manifest.is_file()
        or not isinstance(manifest_sha256, str)
        or _sha256_file(source_manifest) != manifest_sha256
    ):
        raise RuntimeError("resume snapshot production manifest is missing or has a stale hash")
    restored = 0
    for name, record in sorted(records.items()):
        source = root / "output" / name
        if not source.is_file() or _sha256_file(source) != record.get("output_sha256"):
            raise RuntimeError(f"resume output hash mismatch: {{name}}")
        if _sha256_file(TEST_DIR / name) != record.get("input_sha256"):
            raise RuntimeError(f"resume input hash mismatch: {{name}}")
        _validate_png(source)
        destination = OUTPUT / name
        if destination.exists() and _sha256_file(destination) != record["output_sha256"]:
            raise RuntimeError(f"working output conflicts with resume snapshot: {{name}}")
        if not destination.exists():
            shutil.copy2(source, destination)
        restored += 1
    if records:
        destination_manifest = WORK / "run_manifest.json"
        if destination_manifest.exists() and _sha256_file(destination_manifest) != manifest_sha256:
            raise RuntimeError("working production manifest conflicts with resume snapshot")
        if not destination_manifest.exists():
            shutil.copy2(source_manifest, destination_manifest)
    print("restored hash-bound predictions:", restored)

resume_root = os.environ.get("PAZZLE_RANK96_RESUME_DIR", "").strip()
if not resume_root:
    resume_indices = sorted(Path("/kaggle/input").rglob("resume_index.json"))
    if len(resume_indices) == 1:
        resume_root = str(resume_indices[0].parent)
        print("auto-detected rank96 resume snapshot:", resume_root)
    elif len(resume_indices) > 1:
        raise RuntimeError(
            "multiple rank96 resume snapshots found; set PAZZLE_RANK96_RESUME_DIR explicitly"
        )
if resume_root:
    _restore_resume(resume_root)
'''
    command_cell = f'''RANKER = CKPT / {EXPECTED_CHECKPOINTS["ranker"]["filename"]!r}
AFFINITY1 = CKPT / {EXPECTED_CHECKPOINTS["affinity_primary"]["filename"]!r}
AFFINITY2 = CKPT / {EXPECTED_CHECKPOINTS["affinity_secondary"]["filename"]!r}
ZIP_PATH = WORK / "submission.zip"
RUN_MANIFEST = WORK / "run_manifest.json"
RUN_REPORT = WORK / "run_report.json"

command = [
    sys.executable, "-u", str(SRC / {ENTRYPOINT!r}),
    "--input-dir", str(TEST_DIR),
    "--output-dir", str(OUTPUT),
    "--output-zip", str(ZIP_PATH),
    "--ranker-ckpt", str(RANKER),
    "--affinity-ckpt", str(AFFINITY1),
    "--affinity-ckpt2", str(AFFINITY2),
    "--device", "cuda",
    "--pair-batch", "4096",
    "--seed", "20260806",
    "--expected-count", str(EXPECTED_IMAGES),
    "--max-runtime-seconds", "{GRACEFUL_RUNTIME_SECONDS}",
    "--resume",
    "--manifest", str(RUN_MANIFEST),
    "--report", str(RUN_REPORT),
]
if BUNDLE_MANIFEST["overrides"]:
    command.extend(["--override-dir", str(WORK / "overrides")])

# Validate all model/data/contract paths before starting the GPU clock.
subprocess.run(command + ["--dry-run"], check=True, cwd=SRC)
started = time.monotonic()
run_return_code = None
try:
    result = subprocess.run(command, check=False, cwd=SRC, timeout=MAX_INFERENCE_SECONDS)
    run_return_code = result.returncode
except subprocess.TimeoutExpired:
    # Production normally exits 75 at 6900 s.  This is a second hard guard;
    # all committed PNG/manifest records remain resumable after termination.
    run_return_code = 75
    print("hard runtime guard reached; preserving the verified partial snapshot")
finally:
    snapshot = _resume_index(WORK / "resume_index.json")
    print("resume snapshot records:", len(snapshot["records"]))
if run_return_code not in (0, 75):
    raise subprocess.CalledProcessError(run_return_code, command)
RANK96_COMPLETE = run_return_code == 0
if not RANK96_COMPLETE:
    print("partial rank96 run saved; mount this notebook output and set PAZZLE_RANK96_RESUME_DIR on the next run")
print("inference seconds:", round(time.monotonic() - started, 1))
'''
    verify_cell = f'''def _verify_complete_submission():
    outputs = _direct_pngs(OUTPUT)
    if [path.name for path in outputs] != TEST_NAMES:
        missing = sorted(set(TEST_NAMES) - {{path.name for path in outputs}})
        extra = sorted({{path.name for path in outputs}} - set(TEST_NAMES))
        raise RuntimeError(f"submission files differ from test set: missing={{missing[:8]}} extra={{extra[:8]}}")
    for path in outputs:
        _validate_png(path)
    if not ZIP_PATH.is_file():
        raise RuntimeError("inference did not create submission.zip")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = sorted(archive.namelist())
        if names != TEST_NAMES or any("/" in name or "\\\\" in name for name in names):
            raise RuntimeError("submission.zip must contain exactly 700 root-level test PNGs")
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt ZIP member: {{bad}}")
    final = {{
        "bundle_id": BUNDLE_MANIFEST["bundle_id"],
        "images": len(outputs),
        "submission_zip": str(ZIP_PATH),
        "submission_zip_sha256": _sha256_file(ZIP_PATH),
        "elapsed_seconds": time.monotonic() - started,
    }}
    if final["elapsed_seconds"] > MAX_INFERENCE_SECONDS + 5:
        raise RuntimeError("inference exceeded the authorised runtime envelope")
    return final

if RANK96_COMPLETE:
    print(json.dumps(_verify_complete_submission(), indent=2, sort_keys=True))
else:
    print(json.dumps({{
        "status": "partial_resumable",
        "completed": len(snapshot["records"]),
        "total": len(TEST_NAMES),
        "resume_index": str(WORK / "resume_index.json"),
    }}, indent=2, sort_keys=True))
'''

    cells: list[dict[str, Any]] = [
        _markdown_cell(
            "rank96-title",
            "# PAZZLE rank96 — frozen inference\n\n"
            "Inference only: upright 20×20 tiles, fixed orientation, two affinity top-64 "
            "graphs, raw seam ranker, buddies `max_edges=96`, and fixed NLM `h=10`. "
            "The notebook embeds hash-locked code and checkpoints; it performs no training "
            "and contains no credentials.",
        ),
        _code_cell("bootstrap", bootstrap),
        _code_cell("sources", source_cell),
    ]
    for role, record in manifest["checkpoints"].items():
        cells.append(
            _code_cell(
                f"ckpt-{role}"[:64],
                _blob_cell(
                    role,
                    record["filename"],
                    checkpoint_contents[role],
                    record["sha256"],
                    "CKPT",
                ),
            )
        )
    if override_contents:
        cells.append(_code_cell("override-dir", 'OVERRIDES = WORK / "overrides"\nOVERRIDES.mkdir(parents=True, exist_ok=True)\n'))
        for index, (name, content) in enumerate(sorted(override_contents.items())):
            cells.append(
                _code_cell(
                    f"override-{index:03d}",
                    _blob_cell(
                        f"verified source override {name}",
                        name,
                        content,
                        manifest["overrides"][name]["sha256"],
                        "OVERRIDES",
                    ),
                )
            )
    cells.extend(
        [
            _code_cell("input-resume", input_cell),
            _code_cell("run-rank96", command_cell),
            _code_cell("verify-submission", verify_cell),
        ]
    )
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pyversion": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_rank96_notebook(
    *,
    workspace: Path,
    output: Path,
    overrides_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    src_dir = workspace / "src"
    entrypoint = src_dir / ENTRYPOINT
    actual_contract = extract_rank96_contract(entrypoint)
    validate_inference_contract(actual_contract)
    champion = _load_champion_proof(workspace)
    checkpoint_records, checkpoint_contents = _validate_checkpoints(workspace)
    override_records, override_contents = _validate_overrides(overrides_dir)
    source_paths = discover_source_closure(src_dir)
    source_contents = {path.name: path.read_bytes() for path in source_paths}
    source_records = {
        name: {"sha256": sha256_bytes(content), "bytes": len(content)}
        for name, content in sorted(source_contents.items())
    }
    builder_path = Path(__file__).resolve()
    core: dict[str, Any] = {
        "schema": SCHEMA,
        "entrypoint": ENTRYPOINT,
        "inference_contract": dict(actual_contract),
        "champion_proof": champion,
        "checkpoints": checkpoint_records,
        "sources": source_records,
        "overrides": override_records,
        "runtime": {
            "accelerator": "Kaggle T4",
            "max_inference_seconds": MAX_INFERENCE_SECONDS,
            "graceful_runtime_seconds": GRACEFUL_RUNTIME_SECONDS,
            "expected_test_images": EXPECTED_TEST_IMAGES,
            "resume": "pazzle-rank96-resume-v1 input/output hash binding",
            "spatial_checkpoint_required": False,
        },
        "builder": {
            "path": "src/build_rank96_kaggle_notebook.py",
            "sha256": sha256_file(builder_path),
        },
    }
    core["bundle_id"] = sha256_bytes(canonical_json(core))
    notebook = _notebook(core, source_contents, checkpoint_contents, override_contents)
    validate_notebook_structure(notebook)
    notebook_bytes = json.dumps(
        notebook, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=1
    ).encode("utf-8") + b"\n"
    manifest = dict(core)
    manifest["notebook"] = {
        "filename": output.name,
        "sha256": sha256_bytes(notebook_bytes),
        "bytes": len(notebook_bytes),
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"
    sidecar = output.with_suffix(".manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, content in ((output, notebook_bytes), (sidecar, manifest_bytes)):
        if path.exists():
            if path.read_bytes() == content:
                continue
            if not force:
                raise BuildContractError(
                    f"refusing to overwrite divergent immutable bundle file: {path}; pass --force"
                )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace / "kaggle_rank96" / NOTEBOOK_NAME,
    )
    parser.add_argument(
        "--overrides-dir",
        type=Path,
        default=None,
        help="optional flat directory of independently verified RGB 480x480 clean PNGs to embed",
    )
    parser.add_argument("--force", action="store_true", help="replace a divergent prior generated bundle")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = build_rank96_notebook(
            workspace=args.workspace,
            output=args.output.resolve(),
            overrides_dir=args.overrides_dir,
            force=args.force,
        )
    except (BuildContractError, OSError) as error:
        print(f"rank96 bundle build failed closed: {error}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "manifest": str(args.output.resolve().with_suffix(".manifest.json")),
                "bundle_id": manifest["bundle_id"],
                "sources": len(manifest["sources"]),
                "overrides": len(manifest["overrides"]),
                "notebook_sha256": manifest["notebook"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
