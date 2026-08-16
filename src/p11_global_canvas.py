"""P11 GCA-24: frozen-cache global-canvas assignment refiner.

Pre-registered in autoresearch-runs/pazzle-fixed-orientation-20260813/
P11_PRE_REGISTRATION.md before this file was created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

GRID = 24
N_TILES = GRID * GRID
TILE_SIDE = 20
WIDTH = 128
HEADS = 4
SINKHORN_STEPS = 20
SEED = 20260816


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fourier_slots(slots: torch.Tensor) -> torch.Tensor:
    """24x24 canonical coordinates as a 16-d Fourier feature."""
    row = torch.div(slots.long(), GRID, rounding_mode="floor").float() / (GRID - 1)
    col = torch.remainder(slots.long(), GRID).float() / (GRID - 1)
    freqs = torch.tensor([1.0, 2.0, 4.0, 8.0], device=slots.device, dtype=torch.float32)
    phases = 2.0 * math.pi * torch.stack((row[..., None] * freqs, col[..., None] * freqs), dim=-2)
    return torch.cat((phases.sin().flatten(-2), phases.cos().flatten(-2)), dim=-1)


def log_sinkhorn(logits: torch.Tensor, steps: int = SINKHORN_STEPS) -> torch.Tensor:
    log_plan = logits - logits.amax(dim=(-2, -1), keepdim=True)
    for _ in range(steps):
        log_plan = log_plan - torch.logsumexp(log_plan, dim=-1, keepdim=True)
        log_plan = log_plan - torch.logsumexp(log_plan, dim=-2, keepdim=True)
    return log_plan.exp()


def entropy_adaptive_sinkhorn(logits: torch.Tensor, beta_base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Stop-gradient local inverse-temperature controller per P11 pre-registration."""
    with torch.no_grad():
        provisional = log_sinkhorn(float(beta_base) * logits.detach())
        eps = torch.finfo(provisional.dtype).eps
        row_entropy = -(provisional * provisional.clamp_min(eps).log()).sum(-1) / math.log(N_TILES)
        col_entropy = -(provisional * provisional.clamp_min(eps).log()).sum(-2) / math.log(N_TILES)
        beta_row = float(beta_base) / (1.0 + row_entropy)
        beta_col = float(beta_base) / (1.0 + col_entropy)
        beta = 0.5 * (beta_row[..., :, None] + beta_col[..., None, :])
    return log_sinkhorn(beta * logits), beta


def decode_linear_assignment(logits: torch.Tensor) -> np.ndarray:
    matrix = logits.detach().float().cpu().numpy()
    if matrix.ndim != 2 or matrix.shape != (N_TILES, N_TILES):
        raise ValueError(f"expected [576,576] logits, received {matrix.shape}")
    tile, slot = linear_sum_assignment(-matrix)
    result = np.empty(N_TILES, dtype=np.int32)
    result[tile] = slot
    if not np.array_equal(np.sort(result), np.arange(N_TILES, dtype=np.int32)):
        raise RuntimeError("linear assignment did not produce a 576-way bijection")
    return result


def placement_accuracy(pred_tile_to_slot: np.ndarray, target_tile_to_slot: np.ndarray) -> float:
    return float(np.mean(pred_tile_to_slot == target_tile_to_slot))


def load_npz(cache_dir: Path, source: str, device: torch.device) -> dict[str, torch.Tensor | str]:
    path = cache_dir / f"{Path(source).stem}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as raw:
        required = {"tiles_uint8", "target_tile_to_slot", "initial_tile_to_slot", "edge_stats", "source", "p9_cache_sha256"}
        missing = required.difference(raw.files)
        if missing:
            raise RuntimeError(f"{path.name} is missing keys: {sorted(missing)}")
        tiles = torch.from_numpy(raw["tiles_uint8"].copy()).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
        target = torch.from_numpy(raw["target_tile_to_slot"].copy()).long()
        initial = torch.from_numpy(raw["initial_tile_to_slot"].copy()).long()
        edge = torch.from_numpy(raw["edge_stats"].copy()).float()
        source_value = str(raw["source"].item())
        cache_sha = str(raw["p9_cache_sha256"].item())
    if tiles.shape != (N_TILES, 3, TILE_SIDE, TILE_SIDE):
        raise RuntimeError(f"unexpected tile shape in {path.name}: {tuple(tiles.shape)}")
    if edge.shape != (N_TILES, 12) or target.shape != (N_TILES,) or initial.shape != (N_TILES,):
        raise RuntimeError(f"unexpected metadata shape in {path.name}")
    if not torch.equal(torch.sort(target).values, torch.arange(N_TILES)):
        raise RuntimeError(f"target labels are not a permutation in {path.name}")
    if not torch.equal(torch.sort(initial).values, torch.arange(N_TILES)):
        raise RuntimeError(f"initial layout is not a permutation in {path.name}")
    order_slot_to_tile = torch.argsort(target)
    target_canvas = tiles[order_slot_to_tile]
    return {
        "tiles": tiles.to(device, non_blocking=True),
        "target": target.to(device, non_blocking=True),
        "initial": initial.to(device, non_blocking=True),
        "edge": edge.to(device, non_blocking=True),
        "target_canvas": target_canvas.to(device, non_blocking=True),
        "source": source_value,
        "cache_sha": cache_sha,
    }


class TileAppearanceEncoder(nn.Module):
    def __init__(self, width: int = WIDTH) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batch, count = patches.shape[:2]
        latent = self.net(patches.reshape(batch * count, 3, TILE_SIDE, TILE_SIDE)).flatten(1)
        return self.norm(latent.reshape(batch, count, -1))


class GlobalCanvasRefiner(nn.Module):
    """Permutation-invariant tile encoder followed by image-conditional slot tokens."""
    def __init__(self, width: int = WIDTH, heads: int = HEADS) -> None:
        super().__init__()
        self.appearance = TileAppearanceEncoder(width)
        self.edge_project = nn.Sequential(nn.Linear(12, width), nn.GELU(), nn.LayerNorm(width))
        self.initial_project = nn.Sequential(nn.Linear(16, width), nn.GELU(), nn.LayerNorm(width))
        self.tile_fuse = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.LayerNorm(width))
        tile_layer = nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.tile_set = nn.TransformerEncoder(tile_layer, num_layers=3)
        slot_ids = torch.arange(N_TILES)
        self.register_buffer("canonical_fourier", fourier_slots(slot_ids), persistent=False)
        self.slot_token = nn.Parameter(torch.empty(N_TILES, width))
        nn.init.normal_(self.slot_token, std=0.02)
        self.slot_project = nn.Sequential(nn.Linear(16, width), nn.GELU(), nn.LayerNorm(width))
        self.cross_one = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.cross_two = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        canvas_layer = nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.canvas_set = nn.TransformerEncoder(canvas_layer, num_layers=2)
        self.canvas_norm = nn.LayerNorm(width)
        self.patch_decoder = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, 3 * TILE_SIDE * TILE_SIDE), nn.Tanh())
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.slot_bias = nn.Parameter(torch.zeros(N_TILES))

    def forward(self, tiles: torch.Tensor, edge: torch.Tensor, initial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tiles.ndim != 5:
            raise ValueError("tiles must have shape [batch,576,3,20,20]")
        b = tiles.shape[0]
        app = self.appearance(tiles)
        init = self.initial_project(fourier_slots(initial))
        tile_tokens = self.tile_fuse(torch.cat((app, self.edge_project(edge), init), dim=-1))
        tile_tokens = self.tile_set(tile_tokens)
        slots = self.slot_token.unsqueeze(0).expand(b, -1, -1) + self.slot_project(self.canonical_fourier).unsqueeze(0)
        attended, _ = self.cross_one(slots, tile_tokens, tile_tokens, need_weights=False)
        canvas = self.canvas_norm(slots + attended)
        canvas = self.canvas_set(canvas)
        attended, _ = self.cross_two(canvas, tile_tokens, tile_tokens, need_weights=False)
        canvas = self.canvas_norm(canvas + attended)
        generated_patches = self.patch_decoder(canvas).reshape(b, N_TILES, 3, TILE_SIDE, TILE_SIDE)
        generated_embedding = self.appearance(generated_patches)
        tile_score = F.normalize(tile_tokens, dim=-1)
        canvas_score = F.normalize(generated_embedding, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * torch.matmul(tile_score, canvas_score.transpose(-1, -2)) + self.slot_bias.view(1, 1, -1)
        return logits, generated_patches


def batch_from_item(item: dict[str, torch.Tensor | str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        item["tiles"].unsqueeze(0),  # type: ignore[union-attr]
        item["edge"].unsqueeze(0),  # type: ignore[union-attr]
        item["initial"].unsqueeze(0),  # type: ignore[union-attr]
        item["target"].unsqueeze(0),  # type: ignore[union-attr]
        item["target_canvas"].unsqueeze(0),  # type: ignore[union-attr]
    )


def g0a(args: argparse.Namespace) -> None:
    seed_all()
    device = torch.device("cpu")
    model = GlobalCanvasRefiner().to(device).train()
    tiles = torch.randn(1, N_TILES, 3, TILE_SIDE, TILE_SIDE, device=device)
    edge = torch.randn(1, N_TILES, 12, device=device)
    initial = torch.randperm(N_TILES, device=device).unsqueeze(0)
    logits, patches = model(tiles, edge, initial)
    plan, beta = entropy_adaptive_sinkhorn(logits, beta_base=1.0)
    target = torch.randperm(N_TILES, device=device).unsqueeze(0)
    loss = -torch.log(plan[0, torch.arange(N_TILES), target[0]].clamp_min(1e-9)).mean() + 0.01 * patches.square().mean()
    loss.backward()
    gradient = float(sum(p.grad.detach().abs().sum().item() for p in model.parameters() if p.grad is not None))
    decode = decode_linear_assignment(logits[0])
    altered = tiles.clone()
    altered[:, 0] = altered[:, 0] + 0.25
    with torch.no_grad():
        altered_logits, altered_patches = model(altered, edge, initial)
    report = {
        "experiment": "P11_global_canvas",
        "gate": "G0a_structural_contracts",
        "logits_shape": list(logits.shape),
        "patches_shape": list(patches.shape),
        "plan_shape": list(plan.shape),
        "row_max_abs_error": float((plan.sum(-1) - 1.0).abs().max().item()),
        "column_max_abs_error": float((plan.sum(-2) - 1.0).abs().max().item()),
        "finite": bool(torch.isfinite(logits).all() and torch.isfinite(plan).all() and torch.isfinite(beta).all()),
        "gradient_l1": gradient,
        "decode_bijection": bool(np.array_equal(np.sort(decode), np.arange(N_TILES))),
        "conditional_logit_delta_l1": float((altered_logits - logits.detach()).abs().mean().item()),
        "conditional_canvas_delta_l1": float((altered_patches - patches.detach()).abs().mean().item()),
        "amp_used": False,
        "p8_labels_imported": False,
    }
    passed = (
        report["logits_shape"] == [1, N_TILES, N_TILES]
        and report["patches_shape"] == [1, N_TILES, 3, TILE_SIDE, TILE_SIDE]
        and report["plan_shape"] == [1, N_TILES, N_TILES]
        and report["finite"]
        and report["row_max_abs_error"] < 1e-3
        and report["column_max_abs_error"] < 1e-3
        and gradient > 0.0
        and report["decode_bijection"]
        and report["conditional_logit_delta_l1"] > 1e-7
        and report["conditional_canvas_delta_l1"] > 1e-7
    )
    report["passes_G0a"] = bool(passed)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p11_g0a_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise RuntimeError("P11 G0a contract failed")


def g0b(args: argparse.Namespace) -> None:
    seed_all()
    device = torch.device("cpu")
    source = args.source
    if source is None:
        sources = json.loads(args.prepare_report.read_text(encoding="utf-8"))["train_sources"]
        source = str(sources[0])
    item = load_npz(args.cache_dir, source, device)
    model = GlobalCanvasRefiner().to(device).eval()
    with torch.no_grad():
        tiles, edge, initial, target, canvas = batch_from_item(item)
        logits_one, patches_one = model(tiles, edge, initial)
        logits_two, patches_two = model(tiles, edge, initial)
        plan, beta = entropy_adaptive_sinkhorn(logits_one, beta_base=1.0)
        decoded = decode_linear_assignment(logits_one[0])
    report = {
        "experiment": "P11_global_canvas",
        "gate": "G0b_one_fit_canonical_layout",
        "source": source,
        "cache_sha": item["cache_sha"],
        "logits_shape": list(logits_one.shape),
        "patches_shape": list(patches_one.shape),
        "target_shape": list(target.shape),
        "canvas_shape": list(canvas.shape),
        "deterministic_logits": bool(torch.equal(logits_one, logits_two)),
        "deterministic_patches": bool(torch.equal(patches_one, patches_two)),
        "row_max_abs_error": float((plan.sum(-1) - 1.0).abs().max().item()),
        "column_max_abs_error": float((plan.sum(-2) - 1.0).abs().max().item()),
        "decode_bijection": bool(np.array_equal(np.sort(decoded), np.arange(N_TILES))),
        "adaptive_beta_finite": bool(torch.isfinite(beta).all()),
        "uses_cached_fit_only_labels": True,
        "cal_target_opened": False,
        "dev_targets_opened": False,
        "test_accessed": False,
        "p8_labels_imported": False,
        "amp_used": False,
    }
    passed = (
        report["logits_shape"] == [1, N_TILES, N_TILES]
        and report["patches_shape"] == [1, N_TILES, 3, TILE_SIDE, TILE_SIDE]
        and report["target_shape"] == [1, N_TILES]
        and report["canvas_shape"] == [1, N_TILES, 3, TILE_SIDE, TILE_SIDE]
        and report["deterministic_logits"]
        and report["deterministic_patches"]
        and report["row_max_abs_error"] < 1e-3
        and report["column_max_abs_error"] < 1e-3
        and report["decode_bijection"]
        and report["adaptive_beta_finite"]
    )
    report["passes_G0b"] = bool(passed)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "p11_g0b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise RuntimeError("P11 G0b contract failed")


def load_sources(prepare_report: Path) -> tuple[list[str], list[str]]:
    report = json.loads(prepare_report.read_text(encoding="utf-8"))
    train = [str(x) for x in report["train_sources"]]
    held = [str(x) for x in report["held_sources"]]
    if len(train) != 128 or len(held) != 32 or set(train).intersection(held):
        raise RuntimeError("P11 requires the exact P10 G1 source-disjoint 128/32 split")
    return train, held


def train_eval(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("P11 G1 must execute in interactive CUDA session; CUDA unavailable")
    seed_all()
    device = torch.device("cuda")
    train_sources, held_sources = load_sources(args.prepare_report)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    reuse = {
        "experiment": "P11_global_canvas",
        "cache_dir": str(args.cache_dir),
        "cache_file_count": len(list(args.cache_dir.glob("*.npz"))),
        "prepare_report_sha256": sha256(args.prepare_report),
        "train_sources": train_sources,
        "held_sources": held_sources,
        "targets_opened": "cached_FIT_only",
        "cal_target_opened": False,
        "dev_targets_opened": False,
        "test_accessed": False,
        "p8_labels_imported": False,
        "p10_final_checkpoint_imported": False,
        "rank96_mining_invoked": False,
        "rank96_ranker_invoked": False,
        "amp_used": False,
    }
    (args.work_dir / "p11_reuse_cache_manifest.json").write_text(json.dumps(reuse, indent=2), encoding="utf-8")
    model = GlobalCanvasRefiner().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, 17):
        model.train()
        beta_base = 1.0 + 5.0 * (epoch - 1) / 15.0
        total = assign_total = generated_total = soft_total = 0.0
        for source in train_sources:
            item = load_npz(args.cache_dir, source, device)
            tiles, edge, initial, target, target_canvas = batch_from_item(item)
            logits, generated = model(tiles, edge, initial)
            plan, _ = entropy_adaptive_sinkhorn(logits, beta_base)
            gather = plan.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            assignment_loss = -gather.clamp_min(1e-9).log().mean()
            generated_loss = F.l1_loss(generated, target_canvas)
            soft_canvas = torch.einsum("bts,btchw->bschw", plan, tiles)
            soft_loss = F.l1_loss(soft_canvas, target_canvas)
            loss = assignment_loss + 0.25 * generated_loss + 0.25 * soft_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total += float(loss.detach().item())
            assign_total += float(assignment_loss.detach().item())
            generated_total += float(generated_loss.detach().item())
            soft_total += float(soft_loss.detach().item())
        row = {
            "epoch": epoch,
            "beta_base": beta_base,
            "train_loss": total / len(train_sources),
            "assignment_loss": assign_total / len(train_sources),
            "generated_canvas_l1": generated_total / len(train_sources),
            "soft_reassembly_l1": soft_total / len(train_sources),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "epoch": 16,
        "seed": SEED,
        "architecture": "GCA-24 width128 heads4 tile-set3 canvas2 cross2",
        "p11_preregistered": True,
    }
    torch.save(checkpoint, args.work_dir / "p11_g1_final.pt")
    model.eval()
    accuracies: list[float] = []
    invalid = 0
    with torch.no_grad():
        for source in held_sources:
            item = load_npz(args.cache_dir, source, device)
            tiles, edge, initial, target, _ = batch_from_item(item)
            logits, _ = model(tiles, edge, initial)
            try:
                pred = decode_linear_assignment(logits[0])
                accuracies.append(placement_accuracy(pred, target[0].cpu().numpy()))
            except Exception:
                invalid += 1
                accuracies.append(0.0)
    baseline = 0.0018988715277777778
    refined = float(np.mean(accuracies))
    report = {
        "experiment": "P11_global_canvas",
        "gate": "G1_train128_held32",
        "selected_by": "fixed final epoch 16; held never inspected during training",
        "baseline_held_accuracy": baseline,
        "refined_held_accuracy": refined,
        "held_delta_pp_vs_rank96": 100.0 * (refined - baseline),
        "invalid_decodes": invalid,
        "passes_G1": bool(invalid == 0 and refined >= baseline + 0.05),
        "decision": "PASS_to_CAL" if invalid == 0 and refined >= baseline + 0.05 else "REJECT_before_CAL",
        "train_history": history,
        "targets_opened": "cached_FIT_only",
        "cal_target_opened": False,
        "dev_targets_opened": False,
        "test_accessed": False,
        "p8_labels_imported": False,
        "p10_final_checkpoint_imported": False,
        "rank96_mining_invoked": False,
        "rank96_ranker_invoked": False,
        "amp_used": False,
    }
    (args.work_dir / "p11_g1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="P11 Global-Canvas Assignment Refiner")
    parser.add_argument("phase", choices=("g0a", "g0b", "train_eval"))
    parser.add_argument("--cache-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\cache"))
    parser.add_argument("--prepare-report", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P10_sinkhorn_refiner\g1\p10_g1_prepare_report.json"))
    parser.add_argument("--work-dir", type=Path, default=Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\P11_global_canvas"))
    parser.add_argument("--source", type=str, default=None)
    args = parser.parse_args()
    if args.phase == "g0a":
        g0a(args)
    elif args.phase == "g0b":
        g0b(args)
    else:
        train_eval(args)


if __name__ == "__main__":
    main()
