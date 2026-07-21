"""Fine-tune the pair-relation classifier on fragment-restorer outputs."""

import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


IMG_SIZE, GRID, TILE = 480, 24, 20
VAL_SAMPLES = 50_000
BATCH_SIZE = 512
TRAIN_IMAGES = 1_200
PAIRS_PER_IMAGE = 96
EPOCHS = 3
LR = 5e-5
CLEAN_REPLAY_WEIGHT = 0.15
CACHE_IMAGES = 16
RELATION_SEED = 44
RESTORER_SEED = 2026
CLASS_NAMES = ("not_adjacent", "left", "right", "up", "down")
OUT_DIR = Path("/kaggle/working")


def find_data_root():
    for targets in Path("/kaggle/input").rglob("train/targets"):
        if (targets.parent / "inputs").is_dir():
            return targets.parent.parent
    raise FileNotFoundError("Puzzle dataset not found")


def find_checkpoint(pattern, required):
    files = list(Path("/kaggle/input").rglob(pattern))
    files = [p for p in files if required in str(p)]
    if not files:
        raise FileNotFoundError(f"Checkpoint {pattern} containing {required!r} not found")
    return sorted(files)[-1]


def load_rgb(path):
    image = Image.open(path).convert("RGB")
    if image.size != (IMG_SIZE, IMG_SIZE):
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def split_tiles(image):
    return (
        image.reshape(GRID, TILE, GRID, TILE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(GRID * GRID, TILE, TILE, 3)
    )


def tile_features(tiles):
    x = tiles.astype(np.float32) / 255.0
    low = x.reshape(len(x), 5, 4, 5, 4, 3).mean((2, 4))
    gray = low.mean(3)
    gray = (gray - gray.mean((1, 2), keepdims=True)) / (gray.std((1, 2), keepdims=True) + 1e-5)
    dx = np.diff(gray, axis=2, append=gray[:, :, -1:])
    dy = np.diff(gray, axis=1, append=gray[:, -1:, :])
    color = x.mean((1, 2))
    features = np.concatenate(
        [gray.reshape(len(x), -1), 0.35 * dx.reshape(len(x), -1),
         0.35 * dy.reshape(len(x), -1), 0.2 * color], axis=1
    )
    return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)


def position_to_noisy(noisy, clean):
    cost = 2.0 - 2.0 * np.clip(tile_features(noisy) @ tile_features(clean).T, -1.0, 1.0)
    noisy_idx, position_idx = linear_sum_assignment(cost)
    inverse = np.empty(len(position_idx), dtype=np.int64)
    inverse[position_idx] = noisy_idx
    return inverse, float(cost[noisy_idx, position_idx].mean())


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return F.silu(x + self.body(x))


class FragmentRestorer(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.stem = nn.Conv2d(3, base, 3, padding=1)
        self.enc1 = nn.Sequential(ResidualBlock(base), ResidualBlock(base))
        self.down = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Sequential(ResidualBlock(base * 2), ResidualBlock(base * 2), ResidualBlock(base * 2))
        self.up = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec = nn.Sequential(ResidualBlock(base * 2), nn.Conv2d(base * 2, base, 3, padding=1), nn.SiLU())
        self.residual = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x):
        skip = self.enc1(self.stem(x))
        h = self.up(self.mid(self.down(skip)))
        correction = 0.35 * torch.tanh(self.residual(self.dec(torch.cat([h, skip], dim=1))))
        return (x + correction).clamp(0.0, 1.0)


class SeamScorer(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1), nn.GroupNorm(8, base * 4), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base * 4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class PairRelationClassifier(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.horizontal = SeamScorer(base)
        self.vertical = SeamScorer(base)
        self.not_adjacent_logit = nn.Parameter(torch.zeros(()))

    def forward(self, a, b):
        h = torch.cat([torch.cat([b, a], 3), torch.cat([a, b], 3)], 0)
        left, right = self.horizontal(h).chunk(2, 0)
        v = torch.cat([torch.cat([b, a], 2), torch.cat([a, b], 2)], 0)
        up, down = self.vertical(v).chunk(2, 0)
        return torch.stack([self.not_adjacent_logit.expand_as(left), left, right, up, down], 1)


def strict_holdout_stems(data_root):
    targets = sorted((data_root / "train" / "targets").glob("*.png"))[:7000]
    relation = targets.copy()
    random.Random(RELATION_SEED).shuffle(relation)
    relation_holdout = {p.stem for p in relation[:512]}
    restorer = [(data_root / "train" / "inputs" / p.name, p) for p in targets]
    random.Random(RESTORER_SEED).shuffle(restorer)
    restorer_holdout = {target.stem for _, target in restorer[:350]}
    return sorted(relation_holdout & restorer_holdout)


@torch.inference_mode()
def build_tile_banks(stems, data_root, restorer, device):
    banks = {"clean": [], "raw": [], "restored": []}
    costs = []
    for stem in tqdm(stems, desc="restore strict holdout images"):
        clean = split_tiles(load_rgb(data_root / "train" / "targets" / f"{stem}.png"))
        noisy = split_tiles(load_rgb(data_root / "train" / "inputs" / f"{stem}.png"))
        inverse, cost = position_to_noisy(noisy, clean)
        costs.append(cost)
        ordered_raw = noisy[inverse]
        x = torch.from_numpy(np.ascontiguousarray(noisy.transpose(0, 3, 1, 2))).float().div_(255)
        outputs = []
        for batch in x.split(512):
            outputs.append(restorer(batch.to(device)).cpu())
        restored = torch.cat(outputs)[torch.from_numpy(inverse)]
        restored = (restored.mul(255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy())
        banks["clean"].append(clean)
        banks["raw"].append(ordered_raw)
        banks["restored"].append(restored)
    return {k: np.stack(v) for k, v in banks.items()}, costs


class RelationPairs(Dataset):
    def __init__(self, bank):
        self.bank = bank

    def __len__(self):
        return VAL_SAMPLES

    def __getitem__(self, idx):
        rng = random.Random(91_000_003 + idx)
        label = idx % 5
        image = (idx // 5) % len(self.bank)
        if label == 1:
            r, c = rng.randrange(GRID), rng.randrange(1, GRID)
            ia, ib = r * GRID + c, r * GRID + c - 1
        elif label == 2:
            r, c = rng.randrange(GRID), rng.randrange(GRID - 1)
            ia, ib = r * GRID + c, r * GRID + c + 1
        elif label == 3:
            r, c = rng.randrange(1, GRID), rng.randrange(GRID)
            ia, ib = r * GRID + c, (r - 1) * GRID + c
        elif label == 4:
            r, c = rng.randrange(GRID - 1), rng.randrange(GRID)
            ia, ib = r * GRID + c, (r + 1) * GRID + c
        else:
            ra, ca = rng.randrange(GRID), rng.randrange(GRID)
            while True:
                rb, cb = rng.randrange(GRID), rng.randrange(GRID)
                if abs(ra - rb) + abs(ca - cb) > 1:
                    break
            ia, ib = ra * GRID + ca, rb * GRID + cb
        a, b = self.bank[image, ia], self.bank[image, ib]
        a = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float().div_(127.5).sub_(1)
        b = torch.from_numpy(np.ascontiguousarray(b.transpose(2, 0, 1))).float().div_(127.5).sub_(1)
        return a, b, label


def as_unit_tensor(tile):
    return torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).float().div_(255)


class FineTunePairs(Dataset):
    """Image-grouped sampler so Hungarian matching is amortised over many pairs."""

    def __init__(self, pairs):
        self.pairs = pairs
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.pairs) * PAIRS_PER_IMAGE

    def load_ordered(self, image_idx):
        if image_idx in self.cache:
            self.cache.move_to_end(image_idx)
            return self.cache[image_idx]
        input_path, target_path = self.pairs[image_idx]
        raw = split_tiles(load_rgb(input_path))
        clean = split_tiles(load_rgb(target_path))
        inverse, _ = position_to_noisy(raw, clean)
        result = (raw[inverse], clean)
        self.cache[image_idx] = result
        while len(self.cache) > CACHE_IMAGES:
            self.cache.popitem(last=False)
        return result

    def __getitem__(self, idx):
        image_idx = idx // PAIRS_PER_IMAGE
        raw, clean = self.load_ordered(image_idx)
        label = idx % 5
        rng = random
        if label == 1:
            r, c = rng.randrange(GRID), rng.randrange(1, GRID)
            ia, ib = r * GRID + c, r * GRID + c - 1
        elif label == 2:
            r, c = rng.randrange(GRID), rng.randrange(GRID - 1)
            ia, ib = r * GRID + c, r * GRID + c + 1
        elif label == 3:
            r, c = rng.randrange(1, GRID), rng.randrange(GRID)
            ia, ib = r * GRID + c, (r - 1) * GRID + c
        elif label == 4:
            r, c = rng.randrange(GRID - 1), rng.randrange(GRID)
            ia, ib = r * GRID + c, (r + 1) * GRID + c
        else:
            ra, ca = rng.randrange(GRID), rng.randrange(GRID)
            while True:
                rb, cb = rng.randrange(GRID), rng.randrange(GRID)
                if abs(ra - rb) + abs(ca - cb) > 1:
                    break
            ia, ib = ra * GRID + ca, rb * GRID + cb
        return (as_unit_tensor(raw[ia]), as_unit_tensor(raw[ib]),
                as_unit_tensor(clean[ia]), as_unit_tensor(clean[ib]), label)


def metrics(confusion):
    cm = confusion.double()
    tp = cm.diag()
    precision = tp / cm.sum(0).clamp_min(1)
    recall = tp / cm.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "adjacency_accuracy": float((cm[0, 0] + cm[1:, 1:].sum()) / cm.sum()),
        "direction_accuracy_on_adjacent": float(cm[1:, 1:].diag().sum() / cm[1:, :].sum()),
        "per_class_f1": {name: float(f1[i]) for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": confusion.tolist(),
    }


@torch.inference_mode()
def evaluate(model, bank, device):
    loader = DataLoader(RelationPairs(bank), batch_size=BATCH_SIZE, num_workers=2,
                        pin_memory=True, persistent_workers=True)
    confusion = torch.zeros(5, 5, dtype=torch.long)
    for a, b, labels in tqdm(loader, desc="relation validation"):
        logits = model(a.to(device, non_blocking=True), b.to(device, non_blocking=True))
        pred = logits.argmax(1).cpu()
        confusion += torch.bincount(labels * 5 + pred, minlength=25).reshape(5, 5)
    return metrics(confusion)


def training_pairs(data_root, holdout_stems):
    inputs = sorted((data_root / "train" / "inputs").glob("*.png"))
    targets = {p.stem: p for p in (data_root / "train" / "targets").glob("*.png")}
    pairs = [(p, targets[p.stem]) for p in inputs
             if p.stem in targets and p.stem not in holdout_stems]
    random.Random(7301).shuffle(pairs)
    return pairs[:TRAIN_IMAGES]


def train_epoch(model, restorer, loader, optimizer, scaler, device, epoch):
    model.train()
    restorer.eval()
    ema = None
    for raw_a, raw_b, clean_a, clean_b, labels in tqdm(loader, desc=f"fine-tune {epoch}/{EPOCHS}"):
        raw_a = raw_a.to(device, non_blocking=True)
        raw_b = raw_b.to(device, non_blocking=True)
        clean_a = clean_a.to(device, non_blocking=True).mul_(2).sub_(1)
        clean_b = clean_b.to(device, non_blocking=True).mul_(2).sub_(1)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            restored_a = restorer(raw_a).mul_(2).sub_(1)
            restored_b = restorer(raw_b).mul_(2).sub_(1)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            restored_loss = F.cross_entropy(model(restored_a, restored_b), labels)
            clean_loss = F.cross_entropy(model(clean_a, clean_b), labels)
            loss = restored_loss + CLEAN_REPLAY_WEIGHT * clean_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        value = float(loss)
        ema = value if ema is None else 0.98 * ema + 0.02 * value
    return ema


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    print(f"gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    data_root = find_data_root()
    stems = strict_holdout_stems(data_root)
    if len(stems) < 10:
        raise RuntimeError(f"Strict holdout intersection is unexpectedly small: {len(stems)}")
    print(f"strict_holdout_images={len(stems)}")

    restorer_path = find_checkpoint("fragment_restorer_epoch8.pt", "pazzle-fragment-restorer")
    relation_path = find_checkpoint("pair_relation_best.pt", "pazzle-pair-relation-classifier")
    restorer_ckpt = torch.load(restorer_path, map_location="cpu")
    restorer = FragmentRestorer(restorer_ckpt.get("config", {}).get("base_channels", 64))
    restorer.load_state_dict(restorer_ckpt["model"])
    restorer = restorer.to(device).eval()
    relation_ckpt = torch.load(relation_path, map_location="cpu")
    relation = PairRelationClassifier()
    relation.load_state_dict(relation_ckpt["model"])
    relation = relation.to(device).eval()
    print(f"restorer={restorer_path} epoch={restorer_ckpt.get('epoch')}")
    print(f"relation={relation_path} epoch={relation_ckpt.get('epoch')}")

    banks, matching_costs = build_tile_banks(stems, data_root, restorer, device)
    baseline = {mode: evaluate(relation, banks[mode], device) for mode in ("clean", "restored")}
    print("baseline=" + json.dumps(baseline))

    pairs = training_pairs(data_root, set(stems))
    train_loader = DataLoader(
        FineTunePairs(pairs), batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True, drop_last=True,
    )
    optimizer = torch.optim.AdamW(relation.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    best_f1 = baseline["restored"]["macro_f1"]
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(relation, restorer, train_loader, optimizer, scaler, device, epoch)
        restored_metrics = evaluate(relation, banks["restored"], device)
        clean_metrics = evaluate(relation, banks["clean"], device)
        record = {"epoch": epoch, "train_loss_ema": train_loss,
                  "restored": restored_metrics, "clean": clean_metrics}
        history.append(record)
        print("epoch_result=" + json.dumps(record))
        if restored_metrics["macro_f1"] > best_f1:
            best_f1 = restored_metrics["macro_f1"]
            torch.save({
                "model": relation.state_dict(), "epoch": epoch, "metrics": record,
                "class_names": CLASS_NAMES, "source_checkpoint": str(relation_path),
                "restorer_checkpoint": str(restorer_path),
            }, OUT_DIR / "pair_relation_restorer_finetuned_best.pt")

    results = {"baseline": baseline, "history": history, "best_restored_macro_f1": best_f1}
    results["metadata"] = {
        "strict_holdout_images": len(stems),
        "stems": stems,
        "samples_per_mode": VAL_SAMPLES,
        "train_images": len(pairs),
        "pairs_per_train_image": PAIRS_PER_IMAGE,
        "clean_replay_weight": CLEAN_REPLAY_WEIGHT,
        "mean_hungarian_cost": float(np.mean(matching_costs)),
    }
    print(json.dumps(results, indent=2))
    (OUT_DIR / "pair_relation_restorer_finetune_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
