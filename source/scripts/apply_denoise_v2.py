#!/usr/bin/env python3
"""Apply a trained denoiser without changing any shuffled tile position."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from puzzle_denoise_v2.inference import load_restorer, restore_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="one PNG or a directory of PNG files")
    parser.add_argument("--output", required=True, help="one PNG or an output directory")
    parser.add_argument("--state", choices=["ema", "model"], default="ema")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--report", help="optional JSON report path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output PNG/report files; input and checkpoint collisions remain forbidden",
    )
    parser.add_argument(
        "--allow-unpromoted",
        action="store_true",
        help="expert debug only: allow *_latest.pt or an unpromoted fine-tune checkpoint",
    )
    return parser.parse_args()


def _canonical(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _plan_jobs(input_path: Path, output_path: Path) -> list[tuple[Path, Path]]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".png":
            raise SystemExit("input file must be a PNG")
        if output_path.exists() and output_path.is_dir():
            return [(input_path, output_path / input_path.name)]
        if output_path.suffix.lower() == ".png":
            return [(input_path, output_path)]
        if output_path.exists():
            raise SystemExit("single-file output must be a PNG file or a directory")
        return [(input_path, output_path / input_path.name)]
    if input_path.is_dir():
        if output_path.exists() and not output_path.is_dir():
            raise SystemExit("directory input requires an output directory")
        names = [path for path in sorted(input_path.glob("*.png")) if path.is_file()]
        if not names:
            raise SystemExit("input directory contains no PNG files")
        return [(source, output_path / source.name) for source in names]
    raise SystemExit(f"input path does not exist: {input_path}")


def _path_is_occupied(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _validate_existing_parent(path: Path) -> None:
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        raise SystemExit(f"write path has a non-directory ancestor: {path}")


def _validate_write_plan(
    jobs: list[tuple[Path, Path]],
    checkpoint_path: Path,
    report_path: Path | None,
    *,
    overwrite: bool,
) -> None:
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint does not exist or is not a file: {checkpoint_path}")

    source_paths = [_canonical(source) for source, _ in jobs]
    if len(set(source_paths)) != len(source_paths):
        raise SystemExit("multiple input names resolve to the same source PNG")
    protected_paths = [source for source, _ in jobs] + [checkpoint_path]
    protected = set(source_paths)
    protected.add(_canonical(checkpoint_path))

    writes: list[tuple[str, Path]] = [("output", destination) for _, destination in jobs]
    if report_path is not None:
        writes.append(("report", report_path))
    canonical_writes = [_canonical(path) for _, path in writes]
    if len(set(canonical_writes)) != len(canonical_writes):
        raise SystemExit("output/report paths collide after canonical resolution")

    for (kind, path), canonical in zip(writes, canonical_writes, strict=True):
        if canonical in protected:
            raise SystemExit(f"{kind} path collides with an input PNG or checkpoint: {path}")
        if path.exists() and any(path.samefile(protected_path) for protected_path in protected_paths):
            raise SystemExit(f"{kind} path is a hard-link collision with an input PNG or checkpoint: {path}")
        if path.is_symlink():
            raise SystemExit(f"refusing to write through {kind} symlink: {path}")
        if path.exists() and path.is_dir():
            raise SystemExit(f"{kind} file path is a directory: {path}")
        if _path_is_occupied(path) and not overwrite:
            raise SystemExit(f"{kind} already exists; pass --overwrite to replace it: {path}")
        _validate_existing_parent(path)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()
    report_path = Path(args.report).expanduser() if args.report else None
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    jobs = _plan_jobs(input_path, output_path)
    _validate_write_plan(
        jobs,
        checkpoint_path,
        report_path,
        overwrite=args.overwrite,
    )

    model, device, metadata = load_restorer(
        checkpoint_path,
        device=args.device,
        state=args.state,
        allow_unpromoted=args.allow_unpromoted,
    )
    started = time.perf_counter()
    for index, (source, destination) in enumerate(jobs, start=1):
        restore_png(
            model,
            source,
            destination,
            device,
            args.batch_size,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "event": "image_restored",
                    "index": index,
                    "count": len(jobs),
                    "input": str(source),
                    "output": str(destination),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    report = {
        "event": "apply_complete",
        **metadata,
        "input": str(input_path),
        "output": str(output_path),
        "images": len(jobs),
        "batch_size": args.batch_size,
        "overwrite": args.overwrite,
        "allow_unpromoted": args.allow_unpromoted,
        "seconds": time.perf_counter() - started,
        "tile_order_preserved": True,
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    if report_path is not None:
        if report_path.is_symlink():
            raise SystemExit(f"refusing to write through report symlink: {report_path}")
        if report_path.exists():
            if report_path.is_dir():
                raise SystemExit(f"report file path is a directory: {report_path}")
            if not args.overwrite:
                raise SystemExit(f"report appeared during inference: {report_path}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
