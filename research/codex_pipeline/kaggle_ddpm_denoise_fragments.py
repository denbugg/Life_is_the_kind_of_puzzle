import math
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm


try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


IMG_SIZE = int(os.getenv("IMG_SIZE", "480"))
GRID = int(os.getenv("GRID", "24"))
TILE = int(os.getenv("TILE", "20"))
MAX_TRAIN_IMAGES = int(os.getenv("MAX_TRAIN_IMAGES", "7000"))
TILES_PER_IMAGE = int(os.getenv("TILES_PER_IMAGE", "256"))
EPOCHS = int(os.getenv("EPOCHS", "8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
LR = float(os.getenv("LR", "1e-4"))
TIMESTEPS = int(os.getenv("TIMESTEPS", "200"))
X0_LOSS_WEIGHT = float(os.getenv("X0_LOSS_WEIGHT", "5.0"))
COND_START_T = int(os.getenv("COND_START_T", "80"))
BASE_CHANNELS = int(os.getenv("BASE_CHANNELS", "64"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))
CACHE_IMAGES = int(os.getenv("CACHE_IMAGES", "8"))
SEED = int(os.getenv("SEED", "42"))
OUT_DIR = Path(os.getenv("OUT_DIR", "/kaggle/working"))
RESUME_CKPT = os.getenv("RESUME_CKPT", "auto")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_data_root() -> Path:
    roots = [Path("/kaggle/input"), Path(".")]
    for root in roots:
        if not root.exists():
            continue
        for inputs_dir in root.rglob("train/inputs"):
            targets_dir = inputs_dir.parent / "targets"
            if targets_dir.exists():
                return inputs_dir.parent.parent
    raise FileNotFoundError(
        "Could not find train/inputs and train/targets. "
        "Attach the Kaggle dataset as an input to this kernel."
    )


def list_pairs(data_root: Path):
    inputs_dir = data_root / "train" / "inputs"
    targets_dir = data_root / "train" / "targets"
    input_files = sorted(inputs_dir.glob("*.png"))
    target_by_stem = {p.stem: p for p in targets_dir.glob("*.png")}
    pairs = [(p, target_by_stem[p.stem]) for p in input_files if p.stem in target_by_stem]
    if not pairs:
        raise FileNotFoundError(f"No paired PNG files found under {inputs_dir} and {targets_dir}")
    return pairs


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def split_tiles(img: np.ndarray) -> np.ndarray:
    # (480, 480, 3) -> (576, 20, 20, 3)
    return (
        img.reshape(GRID, TILE, GRID, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(GRID * GRID, TILE, TILE, 3)
    )


def tile_features(tiles: np.ndarray) -> np.ndarray:
    arr = tiles.astype(np.float32) / 255.0
    # Fast low-resolution descriptor: 20x20 -> 5x5 by average pooling.
    low = arr.reshape(arr.shape[0], 5, 4, 5, 4, 3).mean(axis=(2, 4))
    gray = low.mean(axis=3)
    gray = (gray - gray.mean(axis=(1, 2), keepdims=True)) / (gray.std(axis=(1, 2), keepdims=True) + 1e-5)
    color = arr.reshape(arr.shape[0], -1, 3).mean(axis=1)
    edge = np.concatenate(
        [
            arr[:, 0].mean(axis=1),
            arr[:, -1].mean(axis=1),
            arr[:, :, 0].mean(axis=1),
            arr[:, :, -1].mean(axis=1),
        ],
        axis=1,
    )
    feats = np.concatenate([gray.reshape(arr.shape[0], -1), color * 0.25, edge * 0.15], axis=1).astype(np.float32)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6)
    return feats


def greedy_assignment(cost: np.ndarray) -> np.ndarray:
    order = np.argsort(cost.min(axis=1))
    used_cols = set()
    assignment = np.empty(cost.shape[0], dtype=np.int64)
    for row in order:
        cols = np.argsort(cost[row])
        for col in cols:
            if int(col) not in used_cols:
                assignment[row] = col
                used_cols.add(int(col))
                break
    return assignment


def match_input_to_target(input_tiles: np.ndarray, target_tiles: np.ndarray) -> np.ndarray:
    a = tile_features(input_tiles)
    b = tile_features(target_tiles)
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        assignment = np.empty(len(rows), dtype=np.int64)
        assignment[rows] = cols
        return assignment
    return greedy_assignment(cost)


def to_tensor(tile: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(tile.transpose(2, 0, 1)).float() / 127.5 - 1.0
    return x


class FragmentPairDataset(Dataset):
    def __init__(self, pairs, tiles_per_image: int, augment: bool):
        self.pairs = pairs[:MAX_TRAIN_IMAGES]
        self.tiles_per_image = min(tiles_per_image, GRID * GRID)
        self.augment = augment
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.pairs) * self.tiles_per_image

    def _load_matched_tiles(self, image_idx: int):
        if image_idx in self.cache:
            self.cache.move_to_end(image_idx)
            return self.cache[image_idx]

        inp_path, tgt_path = self.pairs[image_idx]
        input_tiles = split_tiles(load_rgb(inp_path))
        target_tiles = split_tiles(load_rgb(tgt_path))
        assignment = match_input_to_target(input_tiles, target_tiles)
        clean_tiles = target_tiles[assignment]

        input_t = torch.stack([to_tensor(x) for x in input_tiles])
        clean_t = torch.stack([to_tensor(x) for x in clean_tiles])
        self.cache[image_idx] = (input_t, clean_t)
        while len(self.cache) > CACHE_IMAGES:
            self.cache.popitem(last=False)
        return self.cache[image_idx]

    def __getitem__(self, idx: int):
        image_idx = idx // self.tiles_per_image
        local_idx = idx % self.tiles_per_image
        cond_tiles, clean_tiles = self._load_matched_tiles(image_idx)

        # Deterministic spread across the full 24x24 grid while using a subset.
        tile_idx = (local_idx * (GRID * GRID)) // self.tiles_per_image
        cond = cond_tiles[tile_idx].clone()
        clean = clean_tiles[tile_idx].clone()

        if self.augment:
            if random.random() < 0.5:
                cond = torch.flip(cond, dims=[2])
                clean = torch.flip(clean, dims=[2])
            if random.random() < 0.5:
                cond = torch.flip(cond, dims=[1])
                clean = torch.flip(clean, dims=[1])
            k = random.randint(0, 3)
            if k:
                cond = torch.rot90(cond, k, dims=[1, 2])
                clean = torch.rot90(clean, k, dims=[1, 2])
        return cond, clean


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h + self.time(temb)[:, :, None, None])
        h = self.conv2(h)
        h = self.norm2(h)
        return F.silu(h + self.skip(x))


class TinyCondUNet(nn.Module):
    def __init__(self, base: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.in_block = TimeBlock(6, base, time_dim)
        self.down1 = nn.Sequential(nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.SiLU())
        self.block1 = TimeBlock(base * 2, base * 2, time_dim)
        self.down2 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1), nn.SiLU())
        self.block2 = TimeBlock(base * 4, base * 4, time_dim)
        self.mid = TimeBlock(base * 4, base * 4, time_dim)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.up_block1 = TimeBlock(base * 4, base * 2, time_dim)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.up_block2 = TimeBlock(base * 2, base, time_dim)
        self.out = nn.Conv2d(base, 3, 3, padding=1)
        self.time_dim = time_dim

    def forward(self, x_t, cond, t):
        temb = self.time_mlp(timestep_embedding(t, self.time_dim))
        x0 = self.in_block(torch.cat([x_t, cond], dim=1), temb)
        x1 = self.block1(self.down1(x0), temb)
        x2 = self.block2(self.down2(x1), temb)
        h = self.mid(x2, temb)
        h = self.up1(h)
        h = self.up_block1(torch.cat([h, x1], dim=1), temb)
        h = self.up2(h)
        h = self.up_block2(torch.cat([h, x0], dim=1), temb)
        return self.out(h)


class Diffusion:
    def __init__(self, timesteps: int, device: torch.device):
        self.timesteps = timesteps
        self.device = device
        betas = torch.linspace(1e-4, 2e-2, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def q_sample(self, x0, t, noise):
        a = self.sqrt_alpha_bar[t][:, None, None, None]
        b = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return a * x0 + b * noise

    def predict_x0(self, x_t, t, pred_noise):
        a = self.sqrt_alpha_bar[t][:, None, None, None]
        b = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return (x_t - b * pred_noise) / a.clamp_min(1e-6)

    @torch.no_grad()
    def sample(self, model, cond, steps: int | None = None, start_from_cond: bool = True):
        model.eval()
        if start_from_cond:
            start_t = min(max(COND_START_T, 1), self.timesteps - 1)
            t0 = torch.full((cond.size(0),), start_t, device=cond.device, dtype=torch.long)
            x = self.q_sample(cond, t0, torch.randn_like(cond))
            use_steps = steps or start_t + 1
            schedule = torch.linspace(start_t, 0, use_steps, device=cond.device).long()
        else:
            x = torch.randn_like(cond)
            use_steps = steps or self.timesteps
            schedule = torch.linspace(self.timesteps - 1, 0, use_steps, device=cond.device).long()
        for t_int in tqdm(schedule, desc="sampling", leave=False):
            t = torch.full((cond.size(0),), int(t_int.item()), device=cond.device, dtype=torch.long)
            pred_noise = model(x, cond, t)
            beta = self.betas[t][:, None, None, None]
            alpha = self.alphas[t][:, None, None, None]
            alpha_bar = self.alpha_bar[t][:, None, None, None]
            mean = (x - beta * pred_noise / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
            if int(t_int.item()) > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1)


def save_preview(model, diffusion, loader, device, out_path: Path, n: int = 16):
    cond, clean = next(iter(loader))
    cond = cond[:n].to(device)
    clean = clean[:n].to(device)
    pred = diffusion.sample(model, cond, start_from_cond=True)
    grid = make_grid(torch.cat([cond, pred, clean], dim=0), nrow=n, normalize=True, value_range=(-1, 1))
    save_image(grid, out_path)


def pick_device() -> torch.device:
    if not torch.cuda.is_available():
        print("CUDA is not available; using CPU")
        return torch.device("cpu")
    try:
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        probe = torch.linspace(0, 1, 8, device="cuda")
        _ = (probe * probe).sum().item()
        print(f"Using CUDA device: {name}, capability={capability}")
        return torch.device("cuda")
    except Exception as exc:
        print(f"CUDA is present but unusable ({type(exc).__name__}: {exc}); using CPU")
        return torch.device("cpu")


def find_resume_checkpoint() -> Path | None:
    if RESUME_CKPT.lower() in {"", "none", "false", "0"}:
        return None
    if RESUME_CKPT.lower() != "auto":
        path = Path(RESUME_CKPT)
        return path if path.exists() else None

    roots = [Path("/kaggle/input"), Path(".")]
    candidates = []
    for root in roots:
        if root.exists():
            candidates.extend(root.rglob("ddpm_frag_epoch*.pt"))
    if not candidates:
        return None

    def epoch_num(path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.rsplit("epoch", 1)[1])
        except Exception:
            return -1

    return max(candidates, key=epoch_num)


def main():
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_root = find_data_root()
    pairs = list_pairs(data_root)
    random.shuffle(pairs)

    n_train = min(MAX_TRAIN_IMAGES, len(pairs))
    n_val = min(64, max(1, len(pairs) - n_train))
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val] if len(pairs) > n_train else pairs[:n_val]

    print(f"data_root={data_root}")
    print(f"train_images={len(train_pairs)} val_images={len(val_pairs)} tiles_per_image={TILES_PER_IMAGE}")
    print(f"effective_train_tiles={len(train_pairs) * min(TILES_PER_IMAGE, GRID * GRID)}")

    train_ds = FragmentPairDataset(train_pairs, TILES_PER_IMAGE, augment=True)
    val_ds = FragmentPairDataset(val_pairs, min(64, TILES_PER_IMAGE), augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    device = pick_device()
    model = TinyCondUNet(base=BASE_CHANNELS).to(device)
    diffusion = Diffusion(TIMESTEPS, device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    global_step = 0
    start_epoch = 1
    resume_path = find_resume_checkpoint()
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        global_step = int(ckpt.get("global_step", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"resumed_from={resume_path} start_epoch={start_epoch} global_step={global_step}")
    else:
        print("resumed_from=None")

    end_epoch = start_epoch + EPOCHS - 1
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{end_epoch}")
        avg_loss = 0.0
        for cond, clean in pbar:
            cond = cond.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            t = torch.randint(0, TIMESTEPS, (clean.size(0),), device=device)
            noise = torch.randn_like(clean)
            x_t = diffusion.q_sample(clean, t, noise)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred_noise = model(x_t, cond, t)
                pred_x0 = diffusion.predict_x0(x_t, t, pred_noise).clamp(-1, 1)
                noise_loss = F.mse_loss(pred_noise, noise)
                x0_loss = F.l1_loss(pred_x0, clean) + 0.25 * F.mse_loss(pred_x0, clean)
                loss = noise_loss + X0_LOSS_WEIGHT * x0_loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            global_step += 1
            avg_loss = 0.98 * avg_loss + 0.02 * float(loss.item()) if global_step > 1 else float(loss.item())
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        ckpt_path = OUT_DIR / f"ddpm_frag_epoch{epoch}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": {
                    "img_size": IMG_SIZE,
                    "grid": GRID,
                    "tile": TILE,
                    "timesteps": TIMESTEPS,
                    "base_channels": BASE_CHANNELS,
                    "x0_loss_weight": X0_LOSS_WEIGHT,
                    "cond_start_t": COND_START_T,
                },
            },
            ckpt_path,
        )
        print(f"saved {ckpt_path}")
        save_preview(model, diffusion, val_loader, device, OUT_DIR / f"preview_epoch{epoch}.png")

    print("done")
    print(f"outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
