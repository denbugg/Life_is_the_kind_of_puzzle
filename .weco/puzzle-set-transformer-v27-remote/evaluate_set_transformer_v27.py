"""V27: query-conditioned set transformer over the V22/V23 candidate union."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/home/kva/pazzle_set_transformer_v27")
V26_ROOT = Path("/home/kva/pazzle_union_reranker_v26")
sys.path[:0] = [str(V26_ROOT), str(ROOT)]
import evaluate_union_reranker_v26 as v26
import global_solver

OUT = ROOT / "outputs"
V26_CHECKPOINT = V26_ROOT / "outputs/reranker_v26_beta1.pt"
TRAIN_SCENES = range(6700, 6720)
VALID_SCENES = range(6720, 6728)
TEST_SCENES = range(6973, 6989)
ASSEMBLY_SCENE = 6973
SEED = 270826


class SetReranker(nn.Module):
    def __init__(self, features=18, width=192, layers=3, heads=6):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(features, width), nn.LayerNorm(width), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=512, dropout=.10, activation="gelu",
            batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 96), nn.GELU(), nn.Linear(96, 1))

    def forward(self, x, mask=None):
        hidden = self.input(x)
        hidden = self.context(hidden, src_key_padding_mask=None if mask is None else ~mask)
        return self.output(hidden).squeeze(-1)


def train_model(train_groups, valid_groups, device):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = SetReranker().to(device)
    loader = DataLoader(TensorDataset(*train_groups), batch_size=96, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=2e-5)
    vx, vm, vy = [x.to(device) for x in valid_groups]
    best, best_loss, history = None, float("inf"), []
    for epoch in range(1, 31):
        model.train(); losses = []
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask).masked_fill(~mask, -1e4)
            loss = nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.5); optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step(); model.eval()
        valid_losses = []
        with torch.inference_mode():
            for start in range(0, len(vx), 256):
                logits = model(vx[start:start+256], vm[start:start+256]).masked_fill(
                    ~vm[start:start+256], -1e4)
                valid_losses.append(float(nn.functional.cross_entropy(logits, vy[start:start+256])))
        valid_loss = float(np.mean(valid_losses))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_loss": valid_loss,
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row); print(json.dumps({"event": "epoch", **row}), flush=True)
        if valid_loss < best_loss:
            best_loss, best = valid_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model.eval(), history, best_loss


@torch.inference_mode()
def rerank_scene(model, scores, beta, device):
    output = []
    a2, b2 = scores
    for direction in range(2):
        a, b = a2[direction], b2[direction]
        ctx = v26.context(a, b); final = ctx[0].copy()
        sources, _ = v26.sources_targets(direction)
        groups, candidate_sets = [], []
        for source in sources:
            candidates = v26.candidate_union(a, b, source)
            groups.append(v26.features(a, b, direction, source, candidates, ctx))
            candidate_sets.append(candidates)
        start = 0
        while start < len(groups):
            chunk = groups[start:start+128]
            x = np.zeros((len(chunk), v26.MAX_CANDIDATES, 18), np.float32)
            mask = np.zeros((len(chunk), v26.MAX_CANDIDATES), bool)
            for i, group in enumerate(chunk):
                x[i, :len(group)] = group; mask[i, :len(group)] = True
            xt, mt = torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)
            pred = model(xt, mt).float().cpu().numpy()
            for i, source in enumerate(sources[start:start+len(chunk)]):
                candidates = candidate_sets[start+i]
                values = pred[i, :len(candidates)]
                values = (values - values.mean()) / (values.std() + 1e-6)
                final[source, candidates] += beta * values
            start += len(chunk)
        output.append(final)
    return output


def aggregate(rows):
    return {key: float(np.mean([x[key] for x in rows])) for key in rows[0]}


def evaluate_set(model, scene_scores, beta, device):
    return aggregate([v26.v25.metrics(rerank_scene(model, scores, beta, device)) for scores in scene_scores])


def evaluate_v26(model, scene_scores, device):
    return aggregate([v26.v25.metrics(v26.rerank_scene(model, scores, 1., device)) for scores in scene_scores])


def render_board(tiles, board):
    grid = tiles[np.asarray(board).reshape(24, 24)]
    return grid.transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


def labelled(image, text):
    canvas = np.full((525, 480, 3), 255, np.uint8); canvas[45:] = image
    cv2.putText(canvas, text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, .67, (15, 15, 15), 2, cv2.LINE_AA)
    return canvas


def solve_and_visualize(scene, scores, model26, model27, beta, device, maps):
    matrices26 = v26.rerank_scene(model26, scores, 1., device)
    matrices27 = rerank_scene(model27, scores, beta, device)
    solved, metrics = {}, {}
    truth = np.arange(24 * 24, dtype=np.int32)
    for name, matrices in (("v26", matrices26), ("v27", matrices27)):
        anchor, _ = v26.v25.v10.assemble_components(matrices[0], matrices[1], 24)
        result = global_solver.solve_complete(
            matrices[0], matrices[1], 24, anchor, seed=SEED + scene,
            beam_width=4, hungarian_rounds=5, swap_proposals=12000)
        solved[name] = result.board
        metrics[name] = {**global_solver.placement_metrics(result.board, truth, 24),
                         "objective": result.objective, "anchor_size": result.anchor_size}
    tiles = v26.v25.load_raw_target_order(scene, maps).permute(0, 2, 3, 1).mul(255).byte().numpy()
    input_image = v26.v25.v10.load_rgb(v26.v25.RAW_INPUTS / f"img_{scene:06d}.png")
    target_path = v26.v25.RAW_INPUTS.parent / "targets" / f"img_{scene:06d}.png"
    target = v26.v25.v10.load_rgb(target_path)
    montage = np.hstack((labelled(input_image, "Shuffled noisy input"),
                         labelled(render_board(tiles, solved["v26"]), "V26 + global solver"),
                         labelled(render_board(tiles, solved["v27"]), "V27 + global solver"),
                         labelled(target, "Clean target (reference)")))
    path = OUT / f"assembly_scene_{scene}.png"
    cv2.imwrite(str(path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    return metrics, path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32 = True
    loaded = v26.v25.load_models(device)
    models, state18, state22 = loaded[:4], loaded[4], loaded[5]
    winner = v26.v25.load_winner(device); maps = np.load(v26.v25.MAP_FILE)["inv"]
    all_scenes = list(TRAIN_SCENES) + list(VALID_SCENES) + list(TEST_SCENES)
    score_by_scene, started = {}, time.perf_counter()
    for index, scene in enumerate(all_scenes, 1):
        score_by_scene[scene] = v26.cache_scene(scene, models, winner, device, maps)
        print(json.dumps({"event": "cache", "scene": scene, "of": len(all_scenes),
                          "seconds": time.perf_counter() - started}), flush=True)
    train_scores = [score_by_scene[x] for x in TRAIN_SCENES]
    valid_scores = [score_by_scene[x] for x in VALID_SCENES]
    test_scores = [score_by_scene[x] for x in TEST_SCENES]
    train_groups, valid_groups = v26.build_groups(train_scores), v26.build_groups(valid_scores)
    model27, history, best_loss = train_model(train_groups, valid_groups, device)
    model26 = v26.Reranker(18).to(device)
    checkpoint26 = torch.load(V26_CHECKPOINT, map_location=device, weights_only=True)
    model26.load_state_dict(checkpoint26["model"]); model26.eval()
    trials = []
    for beta in (0., .10, .20, .35, .50, .70, 1., 1.35, 1.75):
        value = evaluate_set(model27, valid_scores, beta, device)
        row = {"beta": beta, **value, "objective": v26.v25.objective(value)}
        trials.append(row); print(json.dumps({"event": "beta", **row}), flush=True)
    selected = max(trials, key=lambda x: x["objective"])
    test25 = evaluate_set(model27, test_scores, 0., device)
    test26 = evaluate_v26(model26, test_scores, device)
    test27 = evaluate_set(model27, test_scores, selected["beta"], device)
    assembly, image_path = solve_and_visualize(
        ASSEMBLY_SCENE, score_by_scene[ASSEMBLY_SCENE], model26, model27,
        selected["beta"], device, maps)
    torch.save({"model": model27.state_dict(), "beta": selected["beta"], "seed": SEED,
                "features": 18, "width": 192, "layers": 3, "heads": 6}, OUT / "set_reranker_best.pt")
    report = {
        "schema": "puzzle-set-transformer-v27", "v18_step": state18["step"], "v22_step": state22["step"],
        "train_scenes": [min(TRAIN_SCENES), max(TRAIN_SCENES)],
        "validation_scenes": [min(VALID_SCENES), max(VALID_SCENES)],
        "test_scenes": [min(TEST_SCENES), max(TEST_SCENES)],
        "parameters": sum(p.numel() for p in model27.parameters()),
        "train_groups": len(train_groups[2]), "validation_groups": len(valid_groups[2]),
        "best_validation_loss": best_loss, "selected": selected,
        "test_v25": test25, "test_v26": test26, "test_v27": test27,
        "assembly_scene": ASSEMBLY_SCENE, "assembly": assembly, "assembly_image": str(image_path),
        "trials": trials, "history": history, "seconds": time.perf_counter() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "complete", "report": report}), flush=True)


if __name__ == "__main__":
    main()
