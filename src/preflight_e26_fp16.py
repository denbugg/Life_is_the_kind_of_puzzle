from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported result type: {type(value)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounded FP16-only E26 GPU preflight for Turing-class hardware. It is not production training."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cnn-width", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--local-dim", type=int, default=48)
    parser.add_argument("--match-dim", type=int, default=32)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--precision", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--seed", type=int, default=2601)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    import train_e26_contextual_edge as trainer
    from e26_contextual_edge_net import ContextualDirectionalEdgeNet

    if not torch.cuda.is_available():
        raise RuntimeError("FP16 preflight requires an available CUDA device")
    if not str(args.output_dir.resolve()).lower().startswith("e:\\"):
        raise RuntimeError("FP16 preflight output directory must be on E:")
    if not str(args.target_root.resolve()).lower().startswith("e:\\"):
        raise RuntimeError("FP16 preflight target root must be on E:")
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("samples and batch-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 preflight requested but this CUDA runtime reports no BF16 support")
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16

    sources = trainer.load_authenticated_training_sources(args.source_manifest)
    names = tuple(sources.names[: args.samples])
    if len(names) != args.samples:
        raise RuntimeError(f"Requested {args.samples} samples but manifest admits only {len(names)}")
    for name in names:
        if not (args.target_root / name).is_file():
            raise FileNotFoundError(args.target_root / name)

    config = trainer.ContextualEdgeConfig(
        cnn_width=args.cnn_width,
        d_model=args.d_model,
        local_dim=args.local_dim,
        match_dim=args.match_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        encoder_chunk_size=144,
    )
    config.validate()
    loss_config = trainer.LossConfig()
    loss_config.validate()
    dataset = trainer.DeterministicE26Dataset(
        names=names,
        clean_root=args.target_root,
        split_sha256=sources.manifest_sha256,
        stage="FIT",
        epoch_or_zero=0,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    batch = next(iter(loader))
    batch = {name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}

    model = ContextualDirectionalEdgeNet(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"))
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
        outputs = model(batch["tiles"])
        loss, terms = trainer.compute_loss(outputs, batch, config, loss_config)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Preflight produced non-finite loss: {loss.detach().cpu().item()}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    max_memory = int(torch.cuda.max_memory_allocated(device))

    report = {
        "schema": "pazzle-e26-fp16-preflight-v1",
        "mode": "FP16_PREFLIGHT_ONLY",
        "not_production_equivalent": True,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "precision": ("float16_autocast_with_gradscaler" if args.precision == "fp16" else "bfloat16_autocast"),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "target_root": str(args.target_root.resolve()),
        "sample_names": list(names),
        "config": {
            "cnn_width": args.cnn_width,
            "d_model": args.d_model,
            "local_dim": args.local_dim,
            "match_dim": args.match_dim,
            "transformer_layers": args.transformer_layers,
            "attention_heads": args.attention_heads,
            "dropout": args.dropout,
        },
        "batch_size": args.batch_size,
        "loss": float(loss.detach().cpu()),
        "loss_terms": {name: float(value.detach().cpu()) for name, value in terms.items()},
        "grad_norm": grad_norm,
        "elapsed_seconds": elapsed,
        "max_memory_allocated_bytes": max_memory,
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "status": "PASS",
    }
    out_path = args.output_dir / "fp16_preflight_report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=json_safe) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=json_safe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
