from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import train_rl_swap_actor_critic_v2 as rl


def assemble(tiles, board):
    x = tiles[board].reshape(rl.GRID, rl.GRID, rl.TILE, rl.TILE, 3)
    return x.transpose(0, 2, 1, 3, 4).reshape(480, 480, 3)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path("rl_outputs/rl_swap_actor_critic_v2_epoch14.pt")
    restored_path = Path("restored_rl_targets/img_000013_v00.png")
    clean_path = Path("clean_targets/img_000013.png")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = rl.SwapActorCritic().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    restored_tiles = rl.split_tiles(restored_path)
    env = rl.Env(restored_tiles, np.random.default_rng(32027), corruption=96)
    shuffled = env.board.copy()
    best = env.board.copy()
    best_correct = int((best == np.arange(rl.N)).sum())
    no_gain = 0
    with torch.inference_mode():
        for _ in range(300):
            pairs = env.proposals()
            features, stats = env.features(pairs)
            logits, _ = model(torch.from_numpy(features)[None].to(device))
            action = int(logits.argmax(1).item())
            env.apply(pairs[action], stats[action])
            correct = int((env.board == np.arange(rl.N)).sum())
            if correct > best_correct:
                best, best_correct, no_gain = env.board.copy(), correct, 0
            else:
                no_gain += 1
            env.stagnation = no_gain
            if best_correct == rl.N or no_gain >= 80:
                break

    clean = np.asarray(Image.open(clean_path).convert("RGB").resize((480, 480)), np.uint8)
    panels = [
        ("Shuffled diffusion tiles", assemble(restored_tiles, shuffled)),
        ("RL v3 assembled", assemble(restored_tiles, best)),
        ("Diffusion v2 output", assemble(restored_tiles, np.arange(rl.N))),
        ("Clean target", clean),
    ]
    margin, title_h = 16, 38
    canvas = Image.new("RGB", (4*480 + 5*margin, 480 + title_h + 2*margin), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, arr) in enumerate(panels):
        x = margin + i*(480+margin)
        draw.text((x, margin+8), title, fill="black")
        canvas.paste(Image.fromarray(arr), (x, margin+title_h))
    out = Path("pipeline_example.png")
    canvas.save(out)
    initial = int((shuffled == np.arange(rl.N)).sum())
    mse = np.mean((assemble(restored_tiles, np.arange(rl.N)).astype(np.float32)-clean.astype(np.float32))**2)
    print(f"steps={env.steps}")
    print(f"correct={initial}/{rl.N}->{best_correct}/{rl.N}")
    print(f"diffusion_psnr={-10*np.log10(max(mse/(255**2),1e-12)):.4f}")
    print(out)


if __name__ == "__main__":
    main()
