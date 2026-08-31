#!/usr/bin/env python3
"""Train/gate a d64 component-relation reranker without decoding a layout.

The frozen Socket d64 model and decoder144 components define target-blind
pair/translation candidates.  Only the small relation head is trained.  Its
first gate is local retrieval on source-disjoint exact synthetic boards; this
runner intentionally has no global decoder or competition-test path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiijc_puzzle.component_anchor_diagnostic import rebuild_decoder_components
from aiijc_puzzle.component_relation_reranker import (
    ComponentRelationReranker,
    aggregate_relation_observations,
    build_component_relation_candidates,
    component_relation_targets,
    extract_frozen_socket_context,
    relation_listwise_loss,
    relation_query_observations,
)
from aiijc_puzzle.component_shift_head import component_descriptors_from_decoder
from aiijc_puzzle.pretrained_tile_denoiser import (
    load_drunet_color,
    render_drunet_tiles,
)
from aiijc_puzzle.protocol import (
    IMAGE_SIZE,
    compute_protocol_digest,
    select_manifest_records,
    sha256_file,
    split_tiles,
)
from aiijc_puzzle.restored_border_ranker import restored_descriptor_scores
from aiijc_puzzle.socket_sorter_production import (
    LoadedSocketCheckpoint,
    choose_deterministic_device,
    load_socket_checkpoint,
)
from aiijc_puzzle.synthetic_socket_evaluation import make_exact_synthetic_case

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/interim/validation_manifest.json"
DEFAULT_TARGETS = PROJECT_ROOT / "data/raw/train/targets"
DEFAULT_SOCKET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/socket-matcher/v2-d64-train1024-s1600-r400-dev32/socket_matcher.pt"
)
DEFAULT_DRUNET_CHECKPOINT = (
    PROJECT_ROOT / "artifacts/pretrained-denoisers/kair-fc1732f/drunet_color.pth"
)
EXPECTED_DRUNET_SHA256 = "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4"
SELECTION_NAMESPACE = "aiijc-component-relation-reranker-v1"
GRID = 24
TILE_COUNT = GRID * GRID
COMPONENT_EDGE_BUDGET = 144
MAX_TOTAL_SOURCES = 2048
MAX_STEPS = 800
MAX_LOCAL_EVAL_SOURCES = 64
HEAD_HIDDEN_DIMENSION = 64
EXPECTED_HEAD_PARAMETERS = 131_665
DEFAULT_PROPOSAL_TOPK = 8
DEFAULT_CANDIDATE_CAP = 64
HIGH_CONFIDENCE_CAPS = (16, 32, 64, 144)
GATE_MIN_ORACLE_QUERIES = 256
GATE_MIN_CANDIDATE_COVERAGE = 0.15
GATE_MIN_R1_GAIN = 0.03
GATE_MIN_R5_GAIN = -0.005
GATE_MIN_TOP32_CORRECT_GAIN_PER_BOARD = 1.0
GATE_MIN_TOP32_PRECISION_GAIN = 0.03


@dataclass(frozen=True)
class PreparedCase:
    case_id: str
    source_filename: str
    dirty_tiles: np.ndarray
    input_tile_to_position: np.ndarray


class CleanTileCache:
    """Bounded clean-board cache; a full 2048-board preload is over 1 GiB."""

    def __init__(self, targets: Path, *, maximum_boards: int = 32) -> None:
        if maximum_boards <= 0:
            raise ValueError("maximum_boards must be positive")
        self.targets = targets
        self.maximum_boards = maximum_boards
        self.values: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, record: Mapping[str, Any]) -> np.ndarray:
        filename = str(record["filename"])
        if filename in self.values:
            value = self.values.pop(filename)
            self.values[filename] = value
            return value
        path = self.targets / filename
        expected = record.get("target_sha256")
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ValueError(f"manifest target hash mismatch: {filename}")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"expected RGB 480x480 target: {path}")
            value = split_tiles(np.asarray(image, dtype=np.uint8)).copy()
        self.values[filename] = value
        while len(self.values) > self.maximum_boards:
            self.values.popitem(last=False)
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket-checkpoint", type=Path, default=DEFAULT_SOCKET_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sources", type=int, default=512)
    parser.add_argument("--local-eval-sources", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--proposal-topk", type=int, default=DEFAULT_PROPOSAL_TOPK)
    parser.add_argument("--candidate-cap", type=int, default=DEFAULT_CANDIDATE_CAP)
    parser.add_argument("--hidden-dimension", type=int, default=HEAD_HIDDEN_DIMENSION)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--allow-nondeterministic-mps", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--supply",
        choices=("raw", "raw-restored-drunet"),
        default="raw",
        help="raw-only first gate or optional E20 restored-descriptor supply expansion",
    )
    parser.add_argument("--drunet-checkpoint", type=Path, default=DEFAULT_DRUNET_CHECKPOINT)
    parser.add_argument("--drunet-sigma", type=float, default=40.0)
    parser.add_argument("--drunet-batch", type=int, default=144)
    parser.add_argument(
        "--exclude-report",
        type=Path,
        action="append",
        default=[],
        help="artifact whose recursively nested *_filenames lists are excluded",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.train_sources < MAX_TOTAL_SOURCES:
        raise ValueError(f"train-sources must be in [1, {MAX_TOTAL_SOURCES - 1}]")
    if not 1 <= args.local_eval_sources <= MAX_LOCAL_EVAL_SOURCES:
        raise ValueError(
            f"local-eval-sources must be in [1, {MAX_LOCAL_EVAL_SOURCES}]"
        )
    if args.train_sources + args.local_eval_sources > MAX_TOTAL_SOURCES:
        raise ValueError(f"fit+local source count must not exceed {MAX_TOTAL_SOURCES}")
    if not 1 <= args.steps <= MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}]")
    if not 1 <= args.proposal_topk < TILE_COUNT:
        raise ValueError("proposal-topk must be in [1, 575]")
    if args.candidate_cap <= 0 or args.hidden_dimension <= 0 or args.log_every <= 0:
        raise ValueError("candidate-cap, hidden-dimension and log-every must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay must be finite and non-negative")
    if args.allow_nondeterministic_mps and args.device != "mps":
        raise ValueError("allow-nondeterministic-mps requires --device mps")
    if not 0 <= args.drunet_sigma <= 50 or args.drunet_batch <= 0:
        raise ValueError("DRUNet sigma/batch contract is invalid")


def collect_filename_lists(value: Any, *, parent_key: str = "") -> set[str]:
    """Collect every recursively nested list whose key ends ``_filenames``."""

    names: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.endswith("_filenames"):
                if not isinstance(child, (list, tuple)) or not all(
                    isinstance(item, str) and item for item in child
                ):
                    raise ValueError(f"{key} must be a list of non-empty filenames")
                if len(set(child)) != len(child):
                    raise ValueError(f"{key} contains duplicate filenames")
                names.update(Path(item).name for item in child)
            names.update(collect_filename_lists(child, parent_key=key))
    elif isinstance(value, (list, tuple)) and not parent_key.endswith("_filenames"):
        for child in value:
            names.update(collect_filename_lists(child, parent_key=parent_key))
    return names


def _filename_digest(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def select_source_disjoint_records(
    manifest: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    exclude_reports: Sequence[Path],
    *,
    fit_count: int,
    local_count: int,
    checkpoint_sha256: str,
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    set[str],
    list[dict[str, Any]],
]:
    if manifest.get("protocol_digest") != compute_protocol_digest(dict(manifest)):
        raise ValueError("validation manifest protocol digest is invalid")
    splits = manifest.get("splits")
    train = splits.get("train") if isinstance(splits, Mapping) else None
    if not isinstance(train, list):
        raise ValueError("validation manifest has no train split")
    forbidden = collect_filename_lists(checkpoint_payload)
    exclusion_records: list[dict[str, Any]] = [
        {
            "kind": "socket-checkpoint",
            "declared_filename_count": len(forbidden),
        }
    ]
    for path in exclude_reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = collect_filename_lists(payload)
        if not names:
            raise ValueError(f"exclude report contains no *_filenames list: {path}")
        forbidden.update(names)
        exclusion_records.append(
            {
                "kind": "report",
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "declared_filename_count": len(names),
            }
        )
    ranked = select_manifest_records(
        dict(manifest),
        "train",
        limit=len(train),
        namespace=f"{SELECTION_NAMESPACE}\0{checkpoint_sha256}",
    )
    selected = tuple(
        record for record in ranked if Path(str(record["filename"])).name not in forbidden
    )[: fit_count + local_count]
    if len(selected) != fit_count + local_count:
        raise ValueError("could not form the requested source-disjoint fit/local roster")
    fit = selected[:fit_count]
    local = selected[fit_count:]
    fit_names = {str(record["filename"]) for record in fit}
    local_names = {str(record["filename"]) for record in local}
    if fit_names & local_names or (fit_names | local_names) & forbidden:
        raise RuntimeError("source-disjoint selection invariant failed")
    return fit, local, forbidden, exclusion_records


def prepare_case(
    cache: CleanTileCache,
    record: Mapping[str, Any],
    *,
    draw_index: int,
    seed: int,
) -> PreparedCase:
    clean = cache.load(record)
    dirty, reference = make_exact_synthetic_case(
        clean,
        source_filename=str(record["filename"]),
        draw_index=draw_index,
        seed=seed,
    )
    input_tile_to_position = np.empty(TILE_COUNT, dtype=np.int32)
    input_tile_to_position[reference.tile_at_position] = np.arange(
        TILE_COUNT,
        dtype=np.int32,
    )
    return PreparedCase(
        case_id=dirty.case_id,
        source_filename=dirty.source_filename,
        dirty_tiles=dirty.tiles,
        input_tile_to_position=input_tile_to_position,
    )


def _tile_tensor(tiles: np.ndarray, *, device: torch.device) -> torch.Tensor:
    value = np.asarray(tiles)
    if value.shape != (TILE_COUNT, 20, 20, 3) or value.dtype != np.uint8:
        raise ValueError("dirty tiles violate the exact 576x20x20x3 uint8 contract")
    return (
        torch.from_numpy(np.ascontiguousarray(value))
        .permute(0, 3, 1, 2)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )


def _additional_supply(
    case: PreparedCase,
    *,
    supply: str,
    drunet: torch.nn.Module | None,
    device: torch.device,
    sigma: float,
    batch_size: int,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any] | None]:
    if supply == "raw":
        return None, None
    if drunet is None:
        raise RuntimeError("raw-restored-drunet supply requires a loaded DRUNet")
    restored, diagnostics = render_drunet_tiles(
        drunet,
        case.dirty_tiles,
        sigma_255=sigma,
        device=device,
        batch_size=batch_size,
    )
    return (
        {
            "right": restored_descriptor_scores(restored, direction=0),
            "down": restored_descriptor_scores(restored, direction=1),
        },
        diagnostics.as_dict(),
    )


def _case_forward(
    case: PreparedCase,
    *,
    socket: LoadedSocketCheckpoint,
    head: ComponentRelationReranker,
    device: torch.device,
    proposal_topk: int,
    candidate_cap: int,
    supply: str,
    drunet: torch.nn.Module | None,
    drunet_sigma: float,
    drunet_batch: int,
) -> tuple[
    torch.Tensor,
    tuple[Any, ...],
    tuple[Any, ...],
    frozenset[tuple[int, str, int, int, int]],
    tuple[Any, ...],
    dict[str, float],
    dict[str, Any] | None,
]:
    runtime: dict[str, float] = {}
    started = perf_counter()
    tiles = _tile_tensor(case.dirty_tiles, device=device)
    with torch.no_grad():
        tile_tokens, socket_output = extract_frozen_socket_context(
            socket.model,
            tiles,
            grid=GRID,
        )
    runtime["frozen_socket_d64"] = perf_counter() - started

    restored_started = perf_counter()
    additional, drunet_diagnostics = _additional_supply(
        case,
        supply=supply,
        drunet=drunet,
        device=device,
        sigma=drunet_sigma,
        batch_size=drunet_batch,
    )
    runtime["optional_restored_supply"] = perf_counter() - restored_started

    started = perf_counter()
    component_build = rebuild_decoder_components(
        socket_output.right_log_assignment,
        socket_output.down_log_assignment,
        grid=GRID,
        edge_budget_per_axis=COMPONENT_EDGE_BUDGET,
    )
    components = component_descriptors_from_decoder(component_build, grid=GRID)
    candidates = build_component_relation_candidates(
        components,
        socket_output,
        grid=GRID,
        proposal_topk=proposal_topk,
        max_candidates_per_query=candidate_cap,
        additional_proposal_scores=additional,
    )
    labels, oracle_relations, profiles = component_relation_targets(
        candidates,
        components,
        case.input_tile_to_position,
        grid=GRID,
    )
    runtime["decoder_components_candidate_freeze_and_labels"] = perf_counter() - started

    started = perf_counter()
    logits = head(tile_tokens[0].detach(), components, candidates)
    runtime["relation_head"] = perf_counter() - started
    return (
        logits,
        candidates,
        labels,
        oracle_relations,
        profiles,
        runtime,
        drunet_diagnostics,
    )


def evaluate_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Predeclared local-only gate; passing only authorizes a root review."""

    learned = metrics["learned"]
    raw = metrics["raw_socket_component_baseline"]
    coverage = float(metrics["candidate_supply_coverage"])
    oracle_queries = int(metrics["oracle_query_count"])
    learned_r1 = float(learned["r1"])
    raw_r1 = float(raw["r1"])
    learned_r5 = float(learned["r5"])
    raw_r5 = float(raw["r5"])
    learned_top32 = learned["high_confidence"]["top32"]
    raw_top32 = raw["high_confidence"]["top32"]
    correct_gain = float(learned_top32["correct_per_board"]) - float(
        raw_top32["correct_per_board"]
    )
    precision_gain = float(learned_top32["precision"]) - float(
        raw_top32["precision"]
    )
    checks = {
        "minimum_oracle_queries": {
            "observed": oracle_queries,
            "required": GATE_MIN_ORACLE_QUERIES,
            "pass": oracle_queries >= GATE_MIN_ORACLE_QUERIES,
        },
        "candidate_supply_coverage": {
            "observed": coverage,
            "required": GATE_MIN_CANDIDATE_COVERAGE,
            "pass": coverage >= GATE_MIN_CANDIDATE_COVERAGE,
        },
        "pair_translation_r1_gain": {
            "learned": learned_r1,
            "raw": raw_r1,
            "observed_gain": learned_r1 - raw_r1,
            "required_gain": GATE_MIN_R1_GAIN,
            "pass": learned_r1 - raw_r1 >= GATE_MIN_R1_GAIN,
        },
        "pair_translation_r5_non_regression": {
            "learned": learned_r5,
            "raw": raw_r5,
            "observed_gain": learned_r5 - raw_r5,
            "required_gain": GATE_MIN_R5_GAIN,
            "pass": learned_r5 - raw_r5 >= GATE_MIN_R5_GAIN,
        },
        "top32_correct_attachments_per_board_gain": {
            "learned": learned_top32["correct_per_board"],
            "raw": raw_top32["correct_per_board"],
            "observed_gain": correct_gain,
            "required_gain": GATE_MIN_TOP32_CORRECT_GAIN_PER_BOARD,
            "pass": correct_gain >= GATE_MIN_TOP32_CORRECT_GAIN_PER_BOARD,
        },
        "top32_precision_gain": {
            "learned": learned_top32["precision"],
            "raw": raw_top32["precision"],
            "observed_gain": precision_gain,
            "required_gain": GATE_MIN_TOP32_PRECISION_GAIN,
            "pass": precision_gain >= GATE_MIN_TOP32_PRECISION_GAIN,
        },
    }
    passed = all(bool(value["pass"]) for value in checks.values())
    return {
        "status": "pass-await-root-review" if passed else "stop",
        "pass": passed,
        "quality_panel_authorized": False,
        "checks": checks,
        "interpretation": (
            "source-disjoint local relation gate only; no decoder/layout panel was opened"
        ),
    }


def _mean_runtime(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(np.mean([float(row.get(key, 0.0)) for row in rows])) for key in keys
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "mps" and args.allow_nondeterministic_mps:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        device = torch.device("mps")
    else:
        device = choose_deterministic_device(args.device)

    socket_payload = torch.load(
        args.socket_checkpoint.resolve(),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(socket_payload, Mapping):
        raise ValueError("Socket checkpoint payload must be a mapping")
    socket = load_socket_checkpoint(args.socket_checkpoint, device=device)
    if int(socket.contract["dimension"]) != 64:
        raise ValueError("this experiment is frozen to the d64 Socket checkpoint family")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fit_records, local_records, forbidden, exclusion_records = (
        select_source_disjoint_records(
            manifest,
            socket_payload,
            args.exclude_report,
            fit_count=args.train_sources,
            local_count=args.local_eval_sources,
            checkpoint_sha256=socket.sha256,
        )
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "component_relation_reranker.pt"
    report_path = output_dir / "report.json"
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite an existing relation artifact")

    drunet: torch.nn.Module | None = None
    drunet_metadata: dict[str, Any] | None = None
    if args.supply == "raw-restored-drunet":
        observed_hash = sha256_file(args.drunet_checkpoint)
        if observed_hash != EXPECTED_DRUNET_SHA256:
            raise ValueError("DRUNet checkpoint hash differs from the audited E20 artifact")
        drunet = load_drunet_color(args.drunet_checkpoint, device)
        drunet_metadata = {
            "path": str(args.drunet_checkpoint.resolve()),
            "sha256": observed_hash,
            "sigma_255": args.drunet_sigma,
            "batch_size": args.drunet_batch,
            "use": "matcher candidate-supply only; never rendered",
        }

    head = ComponentRelationReranker(
        64,
        grid=GRID,
        hidden_dimension=args.hidden_dimension,
    ).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in head.parameters())
    if (
        args.hidden_dimension == HEAD_HIDDEN_DIMENSION
        and trainable_parameters != EXPECTED_HEAD_PARAMETERS
    ):
        raise RuntimeError("default relation head no longer matches its 131,665-param contract")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.steps,
        eta_min=args.learning_rate * 0.08,
    )
    cache = CleanTileCache(args.targets)
    generator = np.random.default_rng(args.seed + 1)
    train_history: list[dict[str, Any]] = []
    train_runtime: list[dict[str, float]] = []
    recent_loss: list[float] = []
    started = perf_counter()
    head.train()
    for step in range(args.steps):
        record = fit_records[int(generator.integers(len(fit_records)))]
        case = prepare_case(
            cache,
            record,
            draw_index=step,
            seed=args.seed,
        )
        (
            logits,
            candidates,
            labels,
            oracle_relations,
            profiles,
            runtime,
            _,
        ) = _case_forward(
            case,
            socket=socket,
            head=head,
            device=device,
            proposal_topk=args.proposal_topk,
            candidate_cap=args.candidate_cap,
            supply=args.supply,
            drunet=drunet,
            drunet_sigma=args.drunet_sigma,
            drunet_batch=args.drunet_batch,
        )
        loss, diagnostics = relation_listwise_loss(logits, candidates, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0))
        optimizer.step()
        scheduler.step()
        train_runtime.append(runtime)
        recent_loss.append(float(loss.detach()))
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            observations = relation_query_observations(
                logits,
                candidates,
                labels,
                oracle_relations,
                profiles,
                board_id=case.case_id,
            )
            metrics = aggregate_relation_observations(
                observations,
                high_confidence_caps=(32,),
            )
            row = {
                "step": step + 1,
                "mean_loss": float(np.mean(recent_loss)),
                "gradient_norm": gradient_norm,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "candidate_count": int(diagnostics["candidate_count"]),
                "supervised_queries": int(diagnostics["supervised_queries"]),
                "candidate_supply_coverage": metrics["candidate_supply_coverage"],
                "learned_r1": metrics["learned"]["r1"],
                "raw_r1": metrics["raw_socket_component_baseline"]["r1"],
                "elapsed_seconds": perf_counter() - started,
            }
            train_history.append(row)
            print(json.dumps({"event": "train", **row}), flush=True)
            recent_loss.clear()

    training_seconds = perf_counter() - started
    head.eval()
    local_observations: list[dict[str, Any]] = []
    local_runtime: list[dict[str, float]] = []
    local_case_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, record in enumerate(local_records):
            case = prepare_case(cache, record, draw_index=0, seed=args.seed + 10_000)
            (
                logits,
                candidates,
                labels,
                oracle_relations,
                profiles,
                runtime,
                drunet_diagnostics,
            ) = _case_forward(
                case,
                socket=socket,
                head=head,
                device=device,
                proposal_topk=args.proposal_topk,
                candidate_cap=args.candidate_cap,
                supply=args.supply,
                drunet=drunet,
                drunet_sigma=args.drunet_sigma,
                drunet_batch=args.drunet_batch,
            )
            observations = relation_query_observations(
                logits,
                candidates,
                labels,
                oracle_relations,
                profiles,
                board_id=case.case_id,
            )
            local_observations.extend(observations)
            local_runtime.append(runtime)
            local_case_rows.append(
                {
                    "case_id": case.case_id,
                    "source_filename": case.source_filename,
                    "component_count": len(profiles),
                    "candidate_count": len(candidates),
                    "oracle_relation_count": len(oracle_relations),
                    "runtime_seconds": runtime,
                    "drunet": drunet_diagnostics,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "local-eval",
                        "done": index + 1,
                        "total": len(local_records),
                        "case_id": case.case_id,
                    }
                ),
                flush=True,
            )
    local_metrics = aggregate_relation_observations(
        local_observations,
        high_confidence_caps=HIGH_CONFIDENCE_CAPS,
    )
    gate = evaluate_gate(local_metrics)

    fit_names = [str(record["filename"]) for record in fit_records]
    local_names = [str(record["filename"]) for record in local_records]
    checkpoint_declared = collect_filename_lists(socket_payload)
    lineage_train = sorted(checkpoint_declared | set(fit_names))
    lineage_exposed = sorted(forbidden | set(fit_names) | set(local_names))
    selection = {
        "namespace": SELECTION_NAMESPACE,
        "fit_filenames": fit_names,
        "fit_digest": _filename_digest(fit_names),
        "local_eval_filenames": local_names,
        "local_eval_digest": _filename_digest(local_names),
        "lineage_train_filenames": lineage_train,
        "lineage_train_digest": _filename_digest(lineage_train),
        "lineage_exposed_filenames": lineage_exposed,
        "lineage_exposed_digest": _filename_digest(lineage_exposed),
        "forbidden_count_before_current_run": len(forbidden),
        "exclusions": exclusion_records,
    }
    contract = {
        "architecture": "d64-component-relation-reranker-v1",
        "grid": GRID,
        "tile_dimension": 64,
        "hidden_dimension": args.hidden_dimension,
        "parameters": trainable_parameters,
        "component_source": "frozen d64 Socket decoder144 partition",
        "candidate": (
            "component-direction query -> target component + collision-free relative "
            "translation from per-exposed-member top-k Socket supply"
        ),
        "component_encoding": (
            "d64 member token + relative coordinate MLP; permutation-invariant mean/max; "
            "size/shape/density/confidence + board mean"
        ),
        "relation_evidence": (
            "all induced boundary contacts: raw/OT, row/column margins and reciprocal ranks, "
            "facing border logits; permutation-invariant mean/max"
        ),
        "baseline": (
            "max(raw board-z contact)+0.25*mean(raw board-z contact)+"
            "0.10*log1p(contact_count)"
        ),
        "loss": (
            "zero-initialized learned residual over the frozen raw baseline; "
            "multi-positive listwise NLL per source-component/direction query"
        ),
        "supply": args.supply,
        "original_tiles_only": True,
        "global_decoder_present": False,
        "input_index_position_embedding": False,
    }
    checkpoint_payload = {
        "state_dict": head.state_dict(),
        "contract": contract,
        "socket_checkpoint": {
            "path": str(socket.path),
            "sha256": socket.sha256,
        },
        "drunet_checkpoint": drunet_metadata,
        "selection": selection,
        "local_training_gate": gate,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    report = {
        "experiment": contract["architecture"],
        "status": (
            "local-gate-pass-await-root-review" if gate["pass"] else "local-gate-fail-stop"
        ),
        "quality_panel_opened": False,
        "competition_test_opened": False,
        "contract": contract,
        "configuration": {
            "train_sources": args.train_sources,
            "local_eval_sources": args.local_eval_sources,
            "steps": args.steps,
            "proposal_topk_per_view": args.proposal_topk,
            "candidate_cap_per_query": args.candidate_cap,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
            "component_edge_budget_per_axis": COMPONENT_EDGE_BUDGET,
        },
        "socket_checkpoint": {
            "path": str(socket.path),
            "sha256": socket.sha256,
            "contract": socket.contract,
        },
        "drunet_checkpoint": drunet_metadata,
        "selection": selection,
        "train_history": train_history,
        "local_metrics": local_metrics,
        "gate": gate,
        "runtime_seconds": {
            "training_total": training_seconds,
            "mean_training_board": _mean_runtime(train_runtime),
            "mean_local_board": _mean_runtime(local_runtime),
        },
        "local_cases": local_case_rows,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "status": report["status"],
                "gate_pass": gate["pass"],
                "checkpoint": str(checkpoint_path),
                "report": str(report_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
