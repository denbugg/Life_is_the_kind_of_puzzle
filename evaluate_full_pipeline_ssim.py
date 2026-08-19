"""Evaluate predicted assembly plus real-noise restoration with exact RGB SSIM."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch

import evaluate_real_noisy_student_assemblies as assembly
from global_solver_candidate import solve_layout
from train_real_tile_restorer import FragmentRestorer

COUNT = int(os.getenv("VAL_COUNT", "20"))
RESTORER_CKPT = Path(os.getenv("RESTORER_CKPT", "outputs_real_restorer/real_fragment_restorer_best.pt"))
OUT = Path(os.getenv("OUT_JSON", "outputs_real_restorer/full_pipeline_ssim.json"))


@torch.inference_mode()
def restore(model, tiles, device):
    x = torch.from_numpy(np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))).float().div_(255)
    chunks = [model(x[i:i + 256].to(device)).cpu() for i in range(0, len(x), 256)]
    return torch.cat(chunks).permute(0, 2, 3, 1).mul(255).round().clamp(0, 255).byte().numpy()


def main():
    device = torch.device("cuda")
    directional = assembly.base.DirectionalTransformer().to(device).eval()
    directional.load_state_dict(torch.load(assembly.CKPT, map_location="cpu", weights_only=False)["model"])
    position = assembly.PositionPrior().to(device).eval()
    position.load_state_dict(torch.load(assembly.POS_CKPT, map_location="cpu", weights_only=False)["model"])
    checkpoint = torch.load(RESTORER_CKPT, map_location="cpu", weights_only=False)
    restorer = FragmentRestorer().to(device).eval(); restorer.load_state_dict(checkpoint["model"])
    z = np.load(assembly.MAP_FILE); stems, maps = z["stems"], z["maps"]
    order = np.arange(len(stems)); np.random.default_rng(assembly.SEED).shuffle(order)
    val = order[-max(100, len(order) // 10):]
    chosen = val[np.linspace(0, len(val) - 1, min(COUNT, len(val)), dtype=int)]
    rows = []
    for k, j in enumerate(chosen):
        stem = str(stems[j]); tiles = assembly.split(assembly.ROOT / "inputs" / f"{stem}.png")
        target = np.asarray(Image.open(assembly.ROOT / "targets" / f"{stem}.png").convert("RGB"), np.uint8)
        right, down, pos = assembly.matrices(directional, position, tiles, device)
        layout = solve_layout(right, down, pos, assembly.SEED + k * 100)
        restored = restore(restorer, tiles, device)
        raw_image = assembly.assemble(tiles, layout)
        restored_image = assembly.assemble(restored, layout)
        item = assembly.metrics(layout, maps[j])
        item.update(stem=stem,
                    raw_assembled_ssim=float(structural_similarity(target, raw_image, channel_axis=2, data_range=255)),
                    restored_assembled_ssim=float(structural_similarity(target, restored_image, channel_axis=2, data_range=255)))
        rows.append(item); print(json.dumps(item), flush=True)
    report = {
        "count": len(rows), "checkpoint_epoch": checkpoint.get("epoch"),
        "mean_raw_assembled_ssim": float(np.mean([x["raw_assembled_ssim"] for x in rows])),
        "mean_restored_assembled_ssim": float(np.mean([x["restored_assembled_ssim"] for x in rows])),
        "mean_delta_ssim": float(np.mean([x["restored_assembled_ssim"] - x["raw_assembled_ssim"] for x in rows])),
        "mean_adjacency": float(np.mean([x["adjacency"] for x in rows])), "images": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
