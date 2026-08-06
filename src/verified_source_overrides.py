"""Materialize high-confidence public-source clean overrides for PAZZLE test.

The source forensics retrievers save accepted originals/crops on E:.  This
small handoff collects those independently verified outcomes into one folder
whose filenames exactly match test filenames.  A later generic jigsaw
submission can copy these 480x480 PNGs over its corresponding predictions.
It never treats a ranked-but-rejected candidate as an override.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_ROOT = Path(r"E:/pazzle_work/source_forensics")


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _default_reports(root: Path) -> list[Path]:
    matches = root / "matches"
    return [
        matches / "tbank_test_verified_sources.json",
        matches / "tbank_test_verified_sources_rank2.json",
        matches / "static_bag_central_static_news_test_verified_sources.json",
        matches / "static_bag_central_static_news_test_verified_sources_rank2.json",
        matches / "static_bag_central_static_casecontest_test_verified_sources.json",
        matches / "static_bag_central_static_telegram_test_verified_sources.json",
        matches / "static_bag_central_static_telegram_test_verified_sources_rank2.json",
        matches / "static_bag_central_static_prod_telegram_test_verified_sources.json",
        matches / "static_bag_central_static_prod_telegram_test_verified_sources_rank2.json",
        matches / "static_bag_central_static_bachelor_telegram_test_verified_sources.json",
        matches / "static_bag_central_static_bachelor_telegram_test_verified_sources_rank2.json",
        matches / "static_bag_central_static_master_telegram_test_verified_sources_rank1.json",
    ]


def materialize(
    root: Path,
    *,
    reports: list[Path],
    output_dir: Path,
    manifest: Path,
    base_dir: Path | None = None,
    merged_dir: Path | None = None,
    zip_path: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    choices: dict[str, tuple[dict[str, Any], Path]] = {}
    for report_path in reports:
        if not report_path.exists():
            continue
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            accepted = row.get("accepted")
            if accepted is None:
                continue
            test = str(row["test"])
            clean_path = Path(str(accepted["saved_clean"]))
            if not clean_path.exists():
                raise FileNotFoundError(f"accepted clean source is missing: {clean_path}")
            # A duplicate proof can occur only when two catalogues hold the
            # same source image.  Prefer the stronger spatial evidence.
            previous = choices.get(test)
            if previous is None or int(accepted.get("sift_identity_matches", 0)) > int(previous[0].get("sift_identity_matches", 0)):
                choices[test] = ({**accepted, "report": str(report_path)}, clean_path)
    rows: list[dict[str, Any]] = []
    for test, (accepted, clean_path) in sorted(choices.items()):
        with Image.open(clean_path) as image:
            clean = image.convert("RGB")
            if clean.size != (480, 480):
                raise ValueError(f"{clean_path} has {clean.size}, expected 480x480")
            destination = output_dir / test
            clean.save(destination)
        rows.append(
            {
                "test": test,
                "override": str(destination),
                "source": accepted,
            }
        )
    merged_path: Path | None = None
    if base_dir is not None:
        if not base_dir.exists():
            raise FileNotFoundError(f"base prediction directory does not exist: {base_dir}")
        merged_path = merged_dir or root / "overrides" / "merged_submission"
        merged_path.mkdir(parents=True, exist_ok=True)
        base_images = sorted(base_dir.glob("*.png"))
        if not base_images:
            raise RuntimeError(f"no PNG predictions found in {base_dir}")
        for image in base_images:
            shutil.copy2(image, merged_path / image.name)
        for row in rows:
            shutil.copy2(Path(row["override"]), merged_path / row["test"])
    if zip_path is not None:
        if merged_path is None:
            raise ValueError("--zip requires --base-dir so the archive has all test images")
        images = sorted(merged_path.glob("*.png"))
        if len(images) != 700:
            raise RuntimeError(f"expected 700 merged PNGs before zipping, found {len(images)}")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for image in images:
                archive.write(image, image.name)
    _atomic_json(
        manifest,
        {
            "purpose": "copy these PNGs over the same names in any base test submission",
            "accepted_overrides": len(rows),
            "merged_submission_dir": str(merged_path) if merged_path is not None else None,
            "submission_zip": str(zip_path) if zip_path is not None else None,
            "rows": rows,
        },
    )
    print(f"materialized {len(rows)} verified clean overrides -> {output_dir}", flush=True)
    print(f"saved override manifest -> {manifest}", flush=True)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", action="append", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=None, help="optional directory of 700 base PNG predictions")
    parser.add_argument("--merged-dir", type=Path, default=None, help="where to write the base predictions plus exact overrides")
    parser.add_argument("--zip", type=Path, default=None, help="optional 700-image ZIP built from --base-dir plus overrides")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    output_dir = args.out_dir or root / "overrides" / "verified_source_clean"
    manifest = args.manifest or root / "matches" / "verified_test_source_overrides.json"
    materialize(
        root,
        reports=args.report or _default_reports(root),
        output_dir=output_dir,
        manifest=manifest,
        base_dir=args.base_dir,
        merged_dir=args.merged_dir,
        zip_path=args.zip,
    )


if __name__ == "__main__":
    main()
