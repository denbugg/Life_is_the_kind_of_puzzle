from pathlib import Path
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

import train_rl_swap_actor_critic_v2 as rl


def assemble(tiles, board):
    x = tiles[board].reshape(rl.GRID, rl.GRID, rl.TILE, rl.TILE, 3)
    return x.transpose(0, 2, 1, 3, 4).reshape(rl.GRID * rl.TILE, rl.GRID * rl.TILE, 3)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path("outputs/rl_swap_actor_critic_v2_epoch12.pt")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = rl.SwapActorCritic().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    image_path = os.getenv("IMAGE_PATH")
    path = Path(image_path) if image_path else sorted(Path("pseudo_targets").glob("*.png"))[12]
    tiles = rl.split_tiles(path)
    env = rl.Env(tiles, np.random.default_rng(12026), corruption=96)
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

    panels = [
        ("Shuffled", assemble(tiles, shuffled)),
        ("RL v2", assemble(tiles, best)),
        ("Clean target", assemble(tiles, np.arange(rl.N))),
    ]
    margin, title_h = 18, 44
    canvas = Image.new("RGB", (3 * 480 + 4 * margin, 480 + title_h + 2 * margin), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, arr) in enumerate(panels):
        x = margin + i * (480 + margin)
        draw.text((x, margin + 10), title, fill="black")
        canvas.paste(Image.fromarray(arr), (x, margin + title_h))
    out = Path("outputs/rl_v2_assembly_example.png")
    canvas.save(out)
    initial_correct = int((shuffled == np.arange(rl.N)).sum())
    print(f"source={path.name}")
    print(f"steps={env.steps}")
    print(f"initial_correct={initial_correct}/{rl.N}")
    print(f"final_correct={best_correct}/{rl.N}")
    print(f"output={out}")


if __name__ == "__main__":
    main()
