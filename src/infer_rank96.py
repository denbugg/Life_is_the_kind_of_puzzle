"""Production inference for the fixed-orientation PAZZLE rank-96 champion.

The pipeline in this module is deliberately frozen to the configuration that
transferred from calibration to confirmation and then won the immutable gate:

1. split a strict 480x480 RGB input into 576 *upright* 20x20 tiles;
2. mine the union of top-64 candidates from each of two frozen affinity nets;
3. score only those candidates with the raw CandidateSeamRanker logits;
4. convert the sparse U/D/L/R rows with ``dense_rd``;
5. solve with corrected best-buddies at exactly 96 edges and no repair;
6. assemble the unchanged upright tiles and apply fixed NLM with ``h=10``.

There is no orientation variable, orientation search, or output-tile rotation.
The direction canonicalisation internal to CandidateSeamRanker is merely its
frozen scoring representation; it never changes a tile's board orientation.

Outputs are written one PNG at a time with an atomic replace.  The run manifest
is also atomically updated after every completed image.  ``--resume`` skips an
existing PNG only when its input/output hashes and the complete run contract
match the manifest.  An unrecorded PNG is recomputed, never blindly trusted.

Examples::

    python src/infer_rank96.py --smoke
    python src/infer_rank96.py --input-dir E:/pazzle_data/test \
      --output-dir E:/pazzle_work/submissions/rank96 --dry-run
    python src/infer_rank96.py --input-dir E:/pazzle_data/test \
      --output-dir E:/pazzle_work/submissions/rank96 --resume \
      --override-dir E:/pazzle_work/source_forensics/overrides/verified_source_clean \
      --output-zip E:/pazzle_work/submissions/rank96.zip --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image


GRID = 24
TILE_SIZE = 20
IMAGE_SIZE = 480
NUM_TILES = 576
NUM_DIRECTIONS = 4
CANDIDATE_K_PER_ENCODER = 64
CANDIDATE_STORAGE_WIDTH = 128
MAX_EDGES = 96
MIN_MARGIN = 0.0
REPAIR_PASSES = 0
NLM_H = 10
NLM_H_COLOR = 10
NLM_TEMPLATE_WINDOW = 7
NLM_SEARCH_WINDOW = 21
DEFAULT_PAIR_BATCH = 4096
DEFAULT_SEED = 20_260_806
DEFAULT_EXPECTED_COUNT = 700

MANIFEST_SCHEMA = "pazzle-rank96-inference-manifest-v1"
REPORT_SCHEMA = "pazzle-rank96-inference-report-v1"
INCOMPLETE_EXIT_CODE = 75

# This literal is imported by the Kaggle packager.  Do not derive experiment
# choices from CLI arguments: only execution/runtime knobs are configurable.
RANK96_CONTRACT: dict[str, Any] = {
    "schema": "pazzle-rank96-inference-v1",
    "grid": 24,
    "tile_size": 20,
    "image_size": 480,
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

EXPECTED_CHECKPOINT_SHA256: dict[str, str] = {
    "ranker": "42685373b1a450a4cb3d7a9b22370dfcfaa2335e9e8ada609f21b7cc64abbfbc",
    "affinity_primary": "708565329c7661a965215d98e85f462a90930071f36a0f75b4813c0c5797ec4f",
    "affinity_secondary": "0fceafdb110bde59149fe1ad1e800a69d116041bc627af369aaecd60be53b6c8",
}


class Rank96Error(RuntimeError):
    """A production inference contract or integrity check failed."""


class IncompleteRun(Rank96Error):
    """The run stopped at a declared safe point and can be resumed."""


@dataclass(frozen=True)
class InferenceConfig:
    input_dir: Path
    output_dir: Path
    output_zip: Path | None
    ranker_checkpoint: Path
    affinity_primary_checkpoint: Path
    affinity_secondary_checkpoint: Path
    device: str = "auto"
    limit: int = 0
    resume: bool = False
    seed: int = DEFAULT_SEED
    pair_batch: int = DEFAULT_PAIR_BATCH
    override_dir: Path | None = None
    manifest_path: Path | None = None
    report_path: Path | None = None
    expected_count: int = DEFAULT_EXPECTED_COUNT
    max_runtime_seconds: float = 0.0
    dry_run: bool = False


@dataclass(frozen=True)
class LoadedModels:
    ranker: Any
    affinity_primary: Any
    affinity_secondary: Any
    device: Any


@dataclass(frozen=True)
class InferredImage:
    output: np.ndarray
    board: np.ndarray
    objective: float
    candidate_ids_sha256: str
    raw_scores_sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _canonical_json_bytes(value))


def _png_bytes(value: np.ndarray) -> bytes:
    image = _validate_rgb_array(value, label="output")
    stream = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(
        stream, format="PNG", optimize=False, compress_level=6
    )
    return stream.getvalue()


def _atomic_write_png(path: Path, value: np.ndarray) -> str:
    content = _png_bytes(value)
    _atomic_write_bytes(path, content)
    return hashlib.sha256(content).hexdigest()


def _safe_basename(name: str) -> str:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise Rank96Error(f"image name must be a basename, got {name!r}")
    return name


def _validate_rgb_array(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or array.dtype != np.uint8:
        raise Rank96Error(
            f"{label} must be uint8 RGB {(IMAGE_SIZE, IMAGE_SIZE, 3)}, "
            f"got dtype={array.dtype} shape={array.shape}"
        )
    return np.ascontiguousarray(array)


def load_rgb_strict(path: Path) -> np.ndarray:
    """Load one exact RGB PNG without silently converting mode or geometry."""

    if path.suffix.lower() != ".png":
        raise Rank96Error(f"input must use a .png filename: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                raise Rank96Error(f"{path} has .png suffix but format={image.format!r}")
            if image.mode != "RGB":
                raise Rank96Error(f"{path} must be stored as RGB, got mode={image.mode!r}")
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise Rank96Error(
                    f"{path} must be {(IMAGE_SIZE, IMAGE_SIZE)}, got {image.size}"
                )
            value = np.asarray(image, dtype=np.uint8)
    except Rank96Error:
        raise
    except Exception as exc:
        raise Rank96Error(f"could not decode strict RGB PNG {path}: {exc}") from exc
    return _validate_rgb_array(value, label=str(path))


def split_upright_tiles(image: np.ndarray) -> np.ndarray:
    """Split the fixed mosaic; this function has intentionally no rotation input."""

    value = _validate_rgb_array(image, label="input image")
    tiles = (
        value.reshape(GRID, TILE_SIZE, GRID, TILE_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(NUM_TILES, TILE_SIZE, TILE_SIZE, 3)
    )
    if tiles.shape != (NUM_TILES, TILE_SIZE, TILE_SIZE, 3):
        raise AssertionError("fixed splitter did not produce exactly 576 tiles")
    return np.ascontiguousarray(tiles)


def _assert_board(board: np.ndarray) -> np.ndarray:
    value = np.asarray(board)
    if value.shape != (NUM_TILES,) or not np.issubdtype(value.dtype, np.integer):
        raise Rank96Error(f"board must be an integer vector of length {NUM_TILES}")
    value = value.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(value), np.arange(NUM_TILES, dtype=np.int64)):
        raise Rank96Error("board is not a tile permutation over 0..575")
    return value


def assemble_upright_tiles(tiles: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Place unchanged tile ``board[p]`` at cell ``p``; never rotate a tile."""

    value = np.asarray(tiles)
    expected = (NUM_TILES, TILE_SIZE, TILE_SIZE, 3)
    if value.shape != expected or value.dtype != np.uint8:
        raise Rank96Error(f"tiles must be uint8 {expected}, got {value.dtype} {value.shape}")
    order = _assert_board(board)
    output = (
        value[order]
        .reshape(GRID, GRID, TILE_SIZE, TILE_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    return np.ascontiguousarray(output)


def fixed_nlm(image: np.ndarray) -> np.ndarray:
    """Apply the exact restoration arm used by the immutable gate."""

    import cv2

    value = _validate_rgb_array(image, label="assembled image")
    cv2.setNumThreads(1)
    restored = cv2.fastNlMeansDenoisingColored(
        value,
        None,
        NLM_H,
        NLM_H_COLOR,
        NLM_TEMPLATE_WINDOW,
        NLM_SEARCH_WINDOW,
    )
    return _validate_rgb_array(np.asarray(restored, dtype=np.uint8), label="NLM output")


def solve_dense_tiles(
    tiles: np.ndarray,
    right: np.ndarray,
    down: np.ndarray,
    *,
    solver: Callable[..., tuple[np.ndarray, float]] | None = None,
    restorer: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the immutable solver tail; injectable callables support focused tests."""

    if solver is None:
        from solve_buddies import solve_buddies_from_scores

        solver = solve_buddies_from_scores
    if restorer is None:
        restorer = fixed_nlm
    right_array = np.ascontiguousarray(right, dtype=np.float32)
    down_array = np.ascontiguousarray(down, dtype=np.float32)
    expected = (NUM_TILES, NUM_TILES)
    if right_array.shape != expected or down_array.shape != expected:
        raise Rank96Error(f"dense score matrices must both have shape {expected}")
    if not np.isfinite(right_array).all() or not np.isfinite(down_array).all():
        raise Rank96Error("dense score matrices must be finite")
    board, objective = solver(
        right_array,
        down_array,
        max_edges=MAX_EDGES,
        min_margin=MIN_MARGIN,
        repair_passes=REPAIR_PASSES,
    )
    board_array = _assert_board(board)
    assembled = assemble_upright_tiles(tiles, board_array)
    output = _validate_rgb_array(restorer(assembled), label="restored output")
    if not np.isfinite(float(objective)):
        raise Rank96Error("solver returned a non-finite objective")
    return output, board_array, float(objective)


def _set_deterministic_runtime(seed: int, device: Any) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(device_text: str) -> Any:
    import torch

    text = str(device_text).strip().lower()
    if text == "auto":
        text = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(text)
    except Exception as exc:
        raise Rank96Error(f"invalid --device {device_text!r}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise Rank96Error("rank96 supports only CPU or CUDA execution")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Rank96Error("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _ranker_graph_hashes(payload: Mapping[str, Any]) -> list[str]:
    graph = payload.get("candidate_graph")
    if not isinstance(graph, Mapping):
        raise Rank96Error("ranker checkpoint has no candidate_graph contract")
    if int(graph.get("per_encoder_top_k", -1)) != CANDIDATE_K_PER_ENCODER:
        raise Rank96Error("ranker candidate graph was not trained with per-encoder K=64")
    if graph.get("union") is not True or int(graph.get("max_candidates_per_row", -1)) != 128:
        raise Rank96Error("ranker candidate graph is not the required two-encoder K=64 union")
    encoders = graph.get("encoders")
    if not isinstance(encoders, Sequence) or isinstance(encoders, (str, bytes)) or len(encoders) != 2:
        raise Rank96Error("ranker candidate graph must record exactly two encoders")
    hashes: list[str] = []
    for item in encoders:
        if not isinstance(item, Mapping) or not isinstance(item.get("sha256"), str):
            raise Rank96Error("ranker candidate graph has incomplete encoder provenance")
        hashes.append(str(item["sha256"]).lower())
    return hashes


def load_models(config: InferenceConfig, resolved_device: Any) -> LoadedModels:
    """Load only exact, already-hash-gated champion checkpoints."""

    from eval_candidate_rank import load_ranker
    from train_offset_pose import load_frozen_affinity

    _set_deterministic_runtime(config.seed, resolved_device)
    ranker, ranker_payload = load_ranker(str(config.ranker_checkpoint), resolved_device)
    graph_hashes = _ranker_graph_hashes(ranker_payload)
    expected_graph = [
        EXPECTED_CHECKPOINT_SHA256["affinity_primary"],
        EXPECTED_CHECKPOINT_SHA256["affinity_secondary"],
    ]
    if graph_hashes != expected_graph:
        raise Rank96Error("ranker candidate_graph hashes do not match the champion affinity pair")
    model_kwargs = ranker_payload.get("model_kwargs")
    if not isinstance(model_kwargs, Mapping) or int(model_kwargs.get("tile_size", -1)) != TILE_SIZE:
        raise Rank96Error("ranker checkpoint tile geometry differs from the fixed 20px contract")
    affinity_primary, _, primary_kwargs = load_frozen_affinity(
        str(config.affinity_primary_checkpoint), resolved_device
    )
    affinity_secondary, _, secondary_kwargs = load_frozen_affinity(
        str(config.affinity_secondary_checkpoint), resolved_device
    )
    for label, kwargs in (
        ("primary", primary_kwargs),
        ("secondary", secondary_kwargs),
    ):
        if int(kwargs.get("tiles", -1)) != NUM_TILES or int(kwargs.get("tile_size", -1)) != TILE_SIZE:
            raise Rank96Error(f"{label} affinity checkpoint has incompatible puzzle geometry")
    return LoadedModels(
        ranker=ranker,
        affinity_primary=affinity_primary,
        affinity_secondary=affinity_secondary,
        device=resolved_device,
    )


def infer_one(image: np.ndarray, models: LoadedModels, *, pair_batch: int) -> InferredImage:
    """Infer one image through raw ranker scores and the rank-96 solver."""

    import torch
    from eval_candidate_rank import score_full_graph
    from eval_seeded_qap import dense_rd
    from train_offset_pose import mine_affinity_candidates

    if pair_batch < 1:
        raise Rank96Error("pair_batch must be positive")
    tiles_uint8 = split_upright_tiles(image)
    tiles = (
        torch.from_numpy(tiles_uint8)
        .permute(0, 3, 1, 2)
        .contiguous()
        .float()
        .to(models.device)
        / 255.0
    )
    candidates_batched, valid_batched = mine_affinity_candidates(
        models.affinity_primary,
        tiles.unsqueeze(0),
        candidate_k=CANDIDATE_K_PER_ENCODER,
        device=models.device,
        affinity_secondary=models.affinity_secondary,
    )
    candidates = candidates_batched[0]
    valid = valid_batched[0]
    if tuple(candidates.shape) != (NUM_TILES, CANDIDATE_STORAGE_WIDTH):
        raise Rank96Error(f"candidate union has unexpected shape {tuple(candidates.shape)}")
    if valid.shape != candidates.shape or valid.dtype != torch.bool:
        raise Rank96Error("candidate validity mask is not aligned boolean storage")
    if not bool(valid.any(dim=1).all()):
        raise Rank96Error("at least one tile has no valid candidate")
    raw_scores = score_full_graph(
        models.ranker,
        tiles,
        candidates,
        valid,
        pair_batch=pair_batch,
        device=models.device,
    )
    expected_scores = (NUM_DIRECTIONS, NUM_TILES, CANDIDATE_STORAGE_WIDTH)
    if tuple(raw_scores.shape) != expected_scores:
        raise Rank96Error(f"raw scorer returned {tuple(raw_scores.shape)}, expected {expected_scores}")
    expanded_valid = valid.unsqueeze(0).expand_as(raw_scores)
    if not bool(torch.isfinite(raw_scores[expanded_valid]).all()):
        raise Rank96Error("ranker emitted a non-finite score for a valid candidate")
    # The immutable gate converted cached float32 logits with CPU torch.  Keep
    # that execution boundary exact: GPU softmax/scatter rounding can otherwise
    # perturb ties in the discrete buddies solver.
    candidates_cpu_tensor = candidates.detach().cpu().long()
    raw_cpu_tensor = raw_scores.detach().float().cpu()
    right, down = dense_rd(candidates_cpu_tensor, raw_cpu_tensor)
    output, board, objective = solve_dense_tiles(
        tiles_uint8,
        right.detach().float().cpu().numpy(),
        down.detach().float().cpu().numpy(),
    )
    candidates_cpu = candidates_cpu_tensor.numpy().astype(np.int16, copy=False)
    raw_cpu = raw_cpu_tensor.numpy().astype(np.float32, copy=False)
    return InferredImage(
        output=output,
        board=board,
        objective=objective,
        candidate_ids_sha256=sha256_array(candidates_cpu),
        raw_scores_sha256=sha256_array(raw_cpu),
    )


def _workspace() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_checkpoints() -> dict[str, Path]:
    root = _workspace() / "artifacts"
    return {
        "ranker": root / "candidate_rank" / "rank_v2w64_best.pt",
        "affinity_primary": root / "macro_affinity" / "affinity_r1_1200_best.pt",
        "affinity_secondary": root / "macro_affinity" / "affinity_r3_1000_best.pt",
    }


def _checkpoint_paths(config: InferenceConfig) -> dict[str, Path]:
    return {
        "ranker": config.ranker_checkpoint.resolve(),
        "affinity_primary": config.affinity_primary_checkpoint.resolve(),
        "affinity_secondary": config.affinity_secondary_checkpoint.resolve(),
    }


def _checkpoint_provenance(config: InferenceConfig) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for role, path in _checkpoint_paths(config).items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {role} checkpoint: {path}")
        digest = sha256_file(path)
        if digest != EXPECTED_CHECKPOINT_SHA256[role]:
            raise Rank96Error(
                f"{role} checkpoint SHA256 mismatch: expected "
                f"{EXPECTED_CHECKPOINT_SHA256[role]}, got {digest}"
            )
        records[role] = {"sha256": digest, "size": int(path.stat().st_size)}
    return records


def _code_provenance() -> dict[str, str]:
    source = Path(__file__).resolve().parent
    paths = {
        "infer_rank96.py": Path(__file__).resolve(),
        "candidate_rank.py": source / "candidate_rank.py",
        "eval_candidate_rank.py": source / "eval_candidate_rank.py",
        "train_offset_pose.py": source / "train_offset_pose.py",
        "macro_affinity.py": source / "macro_affinity.py",
        "eval_seeded_qap.py": source / "eval_seeded_qap.py",
        "solve_buddies.py": source / "solve_buddies.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing rank96 code dependency: " + ", ".join(missing))
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _list_input_names(input_dir: Path) -> list[str]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    names = sorted(
        path.name for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )
    if not names:
        raise Rank96Error(f"no PNG inputs found in {input_dir}")
    if len(names) != len(set(names)):
        raise Rank96Error("input directory contains duplicate PNG basenames")
    return [_safe_basename(name) for name in names]


def _build_inventory(config: InferenceConfig) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    names = _list_input_names(config.input_dir)
    if config.expected_count > 0 and len(names) != config.expected_count:
        raise Rank96Error(
            f"expected exactly {config.expected_count} input PNGs, found {len(names)}"
        )
    inputs: list[dict[str, Any]] = []
    for name in names:
        path = config.input_dir / name
        load_rgb_strict(path)
        inputs.append({"name": name, "sha256": sha256_file(path)})
    overrides: dict[str, dict[str, Any]] = {}
    if config.override_dir is not None:
        if not config.override_dir.is_dir():
            raise FileNotFoundError(f"override directory does not exist: {config.override_dir}")
        input_names = set(names)
        override_names = sorted(
            path.name
            for path in config.override_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        )
        extras = sorted(set(override_names) - input_names)
        if extras:
            raise Rank96Error(f"override PNGs are not members of the input set: {extras[:8]}")
        for name in override_names:
            path = config.override_dir / name
            load_rgb_strict(path)
            overrides[name] = {"sha256": sha256_file(path)}
    return inputs, overrides


def _resolved_manifest_path(config: InferenceConfig) -> Path:
    return (config.manifest_path or config.output_dir / "rank96_manifest.json").resolve()


def _resolved_report_path(config: InferenceConfig) -> Path:
    return (config.report_path or config.output_dir / "rank96_report.json").resolve()


def _build_contract(
    config: InferenceConfig,
    *,
    resolved_device: str,
    inputs: list[dict[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]],
    checkpoints: Mapping[str, Mapping[str, Any]],
    code: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "pipeline": RANK96_CONTRACT,
        "execution": {
            "seed": int(config.seed),
            "pair_batch": int(config.pair_batch),
            "device": resolved_device,
        },
        "checkpoints": checkpoints,
        "code": code,
        "inputs": inputs,
        "overrides": [
            {"name": name, **dict(overrides[name])} for name in sorted(overrides)
        ],
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Rank96Error(f"could not read JSON contract {path}: {exc}") from exc


def _initial_manifest(contract: Mapping[str, Any], digest: str) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "contract_digest": digest,
        "contract": contract,
        "status": "in_progress",
        "completed": {},
    }


def _validate_resume_manifest(
    value: Any,
    *,
    expected_contract: Mapping[str, Any],
    expected_digest: str,
    output_dir: Path,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != MANIFEST_SCHEMA:
        raise Rank96Error("resume manifest has an unexpected schema")
    if value.get("contract_digest") != expected_digest or value.get("contract") != expected_contract:
        raise Rank96Error("resume manifest belongs to different inputs/checkpoints/code/config")
    completed = value.get("completed")
    if not isinstance(completed, dict):
        raise Rank96Error("resume manifest completed field must be an object")
    input_hashes = {row["name"]: row["sha256"] for row in expected_contract["inputs"]}
    for name, record in completed.items():
        _safe_basename(name)
        if name not in input_hashes or not isinstance(record, Mapping):
            raise Rank96Error(f"resume manifest contains an invalid completed record: {name!r}")
        if record.get("pipeline_contract_digest") != expected_digest:
            raise Rank96Error(f"completed record {name} has a stale pipeline contract")
        if record.get("input_sha256") != input_hashes[name]:
            raise Rank96Error(f"completed record {name} has a stale input hash")
        output = output_dir / name
        if not output.is_file():
            raise Rank96Error(f"completed output is missing: {output}")
        load_rgb_strict(output)
        actual = sha256_file(output)
        if record.get("output_sha256") != actual:
            raise Rank96Error(f"completed output hash mismatch: {output}")
    return value


def _existing_output_names(output_dir: Path) -> set[str]:
    if not output_dir.exists():
        return set()
    if not output_dir.is_dir():
        raise Rank96Error(f"output path exists but is not a directory: {output_dir}")
    return {
        path.name for path in output_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    }


def _write_report(
    path: Path,
    *,
    status: str,
    contract_digest: str,
    input_count: int,
    completed_count: int,
    skipped_count: int,
    new_count: int,
    generic_count: int,
    override_count: int,
    elapsed_seconds: float,
    image_seconds: Sequence[float],
    output_zip: Path | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "contract_digest": contract_digest,
        "input_count": int(input_count),
        "completed_count": int(completed_count),
        "skipped_count": int(skipped_count),
        "new_count": int(new_count),
        "generic_count": int(generic_count),
        "override_count": int(override_count),
        "elapsed_seconds": float(elapsed_seconds),
        "mean_new_image_seconds": float(np.mean(image_seconds)) if image_seconds else None,
        "output_zip": str(output_zip.resolve()) if output_zip is not None else None,
        "output_zip_sha256": sha256_file(output_zip) if output_zip is not None and output_zip.is_file() else None,
        "error": error,
    }
    _atomic_write_json(path, report)
    return report


def _deterministic_zip(output_dir: Path, names: Sequence[str], output_zip: Path) -> str:
    output_zip = output_zip.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_zip.name}.", dir=output_zip.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in names:
                path = output_dir / _safe_basename(name)
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        # Windows requires a writable descriptor for FlushFileBuffers/fsync.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output_zip)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(output_zip)


def _validate_complete_outputs(output_dir: Path, names: Sequence[str]) -> None:
    expected = set(names)
    actual = _existing_output_names(output_dir)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise Rank96Error(
            f"output PNG set differs from input set (missing={missing[:8]}, extra={extra[:8]})"
        )
    for name in names:
        load_rgb_strict(output_dir / name)


def _validate_config(config: InferenceConfig) -> None:
    if config.limit < 0:
        raise Rank96Error("limit must be non-negative")
    if config.pair_batch < 1:
        raise Rank96Error("pair_batch must be positive")
    if config.expected_count < 0:
        raise Rank96Error("expected_count must be non-negative")
    if config.max_runtime_seconds < 0:
        raise Rank96Error("max_runtime_seconds must be non-negative")
    if config.limit and config.output_zip is not None:
        raise Rank96Error("--output-zip is incompatible with a deliberately partial --limit run")


def run_inference(config: InferenceConfig) -> dict[str, Any]:
    """Run or resume the immutable production path.

    A safe runtime/limit stop writes a resumable manifest and raises
    :class:`IncompleteRun`.  The CLI maps this condition to exit code 75.
    """

    _validate_config(config)
    started = time.perf_counter()
    resolved_device = resolve_device(config.device)
    inputs, overrides = _build_inventory(config)
    checkpoints = _checkpoint_provenance(config)
    code = _code_provenance()
    contract = _build_contract(
        config,
        resolved_device=str(resolved_device),
        inputs=inputs,
        overrides=overrides,
        checkpoints=checkpoints,
        code=code,
    )
    contract_digest = _canonical_digest(contract)
    names = [row["name"] for row in inputs]
    dry_summary = {
        "status": "dry_run",
        "contract_digest": contract_digest,
        "input_count": len(names),
        "override_count": len(overrides),
        "device": str(resolved_device),
        "pipeline": RANK96_CONTRACT,
        "checkpoints": checkpoints,
    }
    if config.dry_run:
        return dry_summary

    output_dir = config.output_dir.resolve()
    manifest_path = _resolved_manifest_path(config)
    report_path = _resolved_report_path(config)
    existing = _existing_output_names(output_dir)
    extras = sorted(existing - set(names))
    if extras:
        raise Rank96Error(f"output directory contains PNGs outside this input set: {extras[:8]}")
    if manifest_path.exists():
        if not config.resume:
            raise Rank96Error(f"manifest already exists; pass --resume to continue: {manifest_path}")
        manifest = _validate_resume_manifest(
            _load_json(manifest_path),
            expected_contract=contract,
            expected_digest=contract_digest,
            output_dir=output_dir,
        )
    else:
        if existing:
            raise Rank96Error("existing PNG outputs have no matching manifest and cannot be resumed")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = _initial_manifest(contract, contract_digest)
        _atomic_write_json(manifest_path, manifest)

    completed: dict[str, Any] = manifest["completed"]
    skipped_count = len(completed)
    new_count = 0
    # Source counters describe the complete materialized output set, while
    # new_count/skipped_count describe this invocation.  This keeps the final
    # report truthful even when a long Kaggle run spans several resumes.
    generic_count = sum(
        1 for record in completed.values() if record.get("source") == "rank96"
    )
    override_count = sum(
        1
        for record in completed.values()
        if record.get("source") == "verified_source_override"
    )
    image_seconds: list[float] = []
    models: LoadedModels | None = None
    input_hashes = {row["name"]: row["sha256"] for row in inputs}
    stop_status: str | None = None
    try:
        for name in names:
            if name in completed:
                continue
            elapsed = time.perf_counter() - started
            if config.max_runtime_seconds and elapsed >= config.max_runtime_seconds:
                stop_status = "partial_runtime"
                break
            if config.limit and new_count >= config.limit:
                stop_status = "partial_limit"
                break
            image_started = time.perf_counter()
            input_path = config.input_dir / name
            image = load_rgb_strict(input_path)
            if name in overrides:
                override_path = config.override_dir / name  # type: ignore[operator]
                output = load_rgb_strict(override_path)
                output_sha256 = _atomic_write_png(output_dir / name, output)
                record: dict[str, Any] = {
                    "pipeline_contract_digest": contract_digest,
                    "input_sha256": input_hashes[name],
                    "output_sha256": output_sha256,
                    "source": "verified_source_override",
                    "override_sha256": overrides[name]["sha256"],
                }
                override_count += 1
            else:
                if models is None:
                    models = load_models(config, resolved_device)
                inferred = infer_one(image, models, pair_batch=config.pair_batch)
                output_sha256 = _atomic_write_png(output_dir / name, inferred.output)
                record = {
                    "pipeline_contract_digest": contract_digest,
                    "input_sha256": input_hashes[name],
                    "output_sha256": output_sha256,
                    "source": "rank96",
                    "board_sha256": sha256_array(inferred.board.astype(np.int16)),
                    "candidate_ids_sha256": inferred.candidate_ids_sha256,
                    "raw_scores_sha256": inferred.raw_scores_sha256,
                    "solver_objective": float(inferred.objective),
                }
                generic_count += 1
            # The PNG exists before its completed record.  A crash in this tiny
            # window causes a safe recomputation on resume, never a blind skip.
            completed[name] = record
            manifest["completed"] = completed
            manifest["status"] = "in_progress"
            _atomic_write_json(manifest_path, manifest)
            duration = time.perf_counter() - image_started
            image_seconds.append(duration)
            new_count += 1
            _write_report(
                report_path,
                status="in_progress",
                contract_digest=contract_digest,
                input_count=len(names),
                completed_count=len(completed),
                skipped_count=skipped_count,
                new_count=new_count,
                generic_count=generic_count,
                override_count=override_count,
                elapsed_seconds=time.perf_counter() - started,
                image_seconds=image_seconds,
            )
            print(
                json.dumps(
                    {
                        "completed": len(completed),
                        "total": len(names),
                        "name": name,
                        "source": record["source"],
                        "seconds": round(duration, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if len(completed) != len(names):
            status = stop_status or "partial"
            manifest["status"] = status
            _atomic_write_json(manifest_path, manifest)
            _write_report(
                report_path,
                status=status,
                contract_digest=contract_digest,
                input_count=len(names),
                completed_count=len(completed),
                skipped_count=skipped_count,
                new_count=new_count,
                generic_count=generic_count,
                override_count=override_count,
                elapsed_seconds=time.perf_counter() - started,
                image_seconds=image_seconds,
            )
            raise IncompleteRun(
                f"rank96 stopped safely with {len(completed)}/{len(names)} outputs ({status}); "
                "rerun the same command with --resume"
            )

        _validate_complete_outputs(output_dir, names)
        manifest["status"] = "completed"
        _atomic_write_json(manifest_path, manifest)
        output_zip = config.output_zip.resolve() if config.output_zip is not None else None
        if output_zip is not None:
            _deterministic_zip(output_dir, names, output_zip)
        return _write_report(
            report_path,
            status="completed",
            contract_digest=contract_digest,
            input_count=len(names),
            completed_count=len(completed),
            skipped_count=skipped_count,
            new_count=new_count,
            generic_count=generic_count,
            override_count=override_count,
            elapsed_seconds=time.perf_counter() - started,
            image_seconds=image_seconds,
            output_zip=output_zip,
        )
    except IncompleteRun:
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        _atomic_write_json(manifest_path, manifest)
        _write_report(
            report_path,
            status="failed",
            contract_digest=contract_digest,
            input_count=len(names),
            completed_count=len(completed),
            skipped_count=skipped_count,
            new_count=new_count,
            generic_count=generic_count,
            override_count=override_count,
            elapsed_seconds=time.perf_counter() - started,
            image_seconds=image_seconds,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def smoke_contract() -> dict[str, Any]:
    """Small data-free proof of fixed splitting/assembly and no orientation state."""

    row = np.arange(IMAGE_SIZE, dtype=np.uint16)[:, None]
    col = np.arange(IMAGE_SIZE, dtype=np.uint16)[None, :]
    image = np.empty((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[..., 0] = row % 251
    image[..., 1] = col % 253
    image[..., 2] = (row + col) % 255
    tiles = split_upright_tiles(image)
    identity = np.arange(NUM_TILES, dtype=np.int64)
    rebuilt = assemble_upright_tiles(tiles, identity)
    if not np.array_equal(rebuilt, image):
        raise AssertionError("fixed upright identity round-trip failed")
    if any(key in RANK96_CONTRACT for key in ("rotation", "rotations", "orientation_search")):
        raise AssertionError("rank96 contract unexpectedly exposes an orientation-search knob")
    return {
        "status": "smoke_pass",
        "contract_digest": _canonical_digest(RANK96_CONTRACT),
        "tiles_shape": list(tiles.shape),
        "identity_roundtrip": True,
        "orientation": RANK96_CONTRACT["orientation"],
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = _default_checkpoints()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="directory containing exactly 700 strict RGB PNG mosaics")
    parser.add_argument("--output-dir", type=Path, help="crash-safe per-image PNG output directory")
    parser.add_argument("--output-zip", type=Path, default=None, help="optional deterministic complete submission ZIP")
    parser.add_argument("--ranker-ckpt", type=Path, default=defaults["ranker"])
    parser.add_argument("--affinity-ckpt", type=Path, default=defaults["affinity_primary"])
    parser.add_argument("--affinity-ckpt2", type=Path, default=defaults["affinity_secondary"])
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--limit", type=int, default=0, help="process at most N new images; 0 means all")
    parser.add_argument("--resume", action="store_true", help="resume only outputs proven by the matching manifest")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pair-batch", type=int, default=DEFAULT_PAIR_BATCH, help="ranker pair microbatch; no algorithm change")
    parser.add_argument("--override-dir", type=Path, default=None, help="optional strict verified clean PNG overrides")
    parser.add_argument("--manifest", type=Path, default=None, help="default: OUTPUT_DIR/rank96_manifest.json")
    parser.add_argument("--report", type=Path, default=None, help="default: OUTPUT_DIR/rank96_report.json")
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT, help="exact input count; 0 disables only for controlled smoke runs")
    parser.add_argument("--max-runtime-seconds", type=float, default=0.0, help="safe-point budget; 0 disables, incomplete exit code is 75")
    parser.add_argument("--dry-run", action="store_true", help="validate all files/hashes/contracts without loading models or writing outputs")
    parser.add_argument("--smoke", action="store_true", help="run a small data-free fixed-orientation contract smoke")
    return parser


def _config_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> InferenceConfig:
    if args.input_dir is None or args.output_dir is None:
        parser.error("--input-dir and --output-dir are required unless --smoke is used")
    return InferenceConfig(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        output_zip=args.output_zip.resolve() if args.output_zip is not None else None,
        ranker_checkpoint=args.ranker_ckpt.resolve(),
        affinity_primary_checkpoint=args.affinity_ckpt.resolve(),
        affinity_secondary_checkpoint=args.affinity_ckpt2.resolve(),
        device=args.device,
        limit=args.limit,
        resume=args.resume,
        seed=args.seed,
        pair_batch=args.pair_batch,
        override_dir=args.override_dir.resolve() if args.override_dir is not None else None,
        manifest_path=args.manifest.resolve() if args.manifest is not None else None,
        report_path=args.report.resolve() if args.report is not None else None,
        expected_count=args.expected_count,
        max_runtime_seconds=args.max_runtime_seconds,
        dry_run=args.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        if args.input_dir is not None or args.output_dir is not None:
            parser.error("--smoke does not accept --input-dir or --output-dir")
        print(json.dumps(smoke_contract(), indent=2, sort_keys=True), flush=True)
        return 0
    config = _config_from_args(args, parser)
    try:
        result = run_inference(config)
    except IncompleteRun as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return INCOMPLETE_EXIT_CODE
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
