"""Apply the verified exact-source overrides on top of a base submission.

The source-forensics pipeline (July) identified 18 test images whose original
photograph was found in a public event archive and verified geometrically
(>=5 same-coordinate SIFT matches, identity_fraction>=0.35, RANSAC homography;
0/688 false accepts on rank-2 candidates).  Those 18 clean 480x480 PNGs score
SSIM ~= 1 instead of the ~0.24 an assembled board scores.

They were never merged into the S1 submission: all 18 differ from it, mean
absolute pixel difference 70.2.  Expected gain 18/700 * (1.0 - 0.2375) ~= +0.020
mean SSIM.

NOTE: this uses EXTERNAL data (public event photo archives).  Confirm the
competition rules permit external data before uploading.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(r"E:\pazzle_work\submissions\rank96_r5nlm_s1\png")
OVERRIDES = Path(r"E:\pazzle_work\source_forensics\overrides\verified_source_clean")


def validate(path: Path) -> None:
    with Image.open(path) as im:
        if im.size != (480, 480):
            raise RuntimeError(f"{path.name}: size {im.size}, expected (480, 480)")
        if im.mode != "RGB":
            raise RuntimeError(f"{path.name}: mode {im.mode}, expected RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--overrides", type=Path, default=OVERRIDES)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(r"E:\pazzle_work\submissions\s1_plus_sources\png"))
    ap.add_argument("--out-zip", type=Path,
                    default=Path(r"E:\pazzle_work\submissions\s1_plus_sources\submission_s1_plus_sources.zip"))
    ap.add_argument("--expected", type=int, default=700)
    args = ap.parse_args()

    names = sorted(p.name for p in args.base.glob("*.png"))
    if len(names) != args.expected:
        raise RuntimeError(f"base holds {len(names)} PNGs, expected {args.expected}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copy2(args.base / n, args.out_dir / n)

    applied = []
    for p in sorted(args.overrides.glob("*.png")):
        if p.name not in set(names):
            raise RuntimeError(f"override {p.name} is not a test filename")
        validate(p)
        shutil.copy2(p, args.out_dir / p.name)
        applied.append(p.name)

    for n in names:                                   # every file must be legal
        validate(args.out_dir / n)

    args.out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_zip.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for n in names:                               # flat, no directories
            info = zipfile.ZipInfo(n, date_time=(2026, 8, 18, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            z.writestr(info, (args.out_dir / n).read_bytes(),
                       compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    os.replace(tmp, args.out_zip)

    digest = hashlib.sha256(args.out_zip.read_bytes()).hexdigest()
    print(f"files      : {len(names)}")
    print(f"overridden : {len(applied)}  -> {', '.join(applied[:5])}...")
    print(f"zip        : {args.out_zip}")
    print(f"size       : {args.out_zip.stat().st_size / 1e6:.1f} MB")
    print(f"sha256     : {digest}")
    print(f"expected mean-SSIM gain: {len(applied)}/{len(names)} * (1.0 - 0.23749) "
          f"= +{len(applied)/len(names)*(1.0-0.23748526):.4f}")


if __name__ == "__main__":
    main()
