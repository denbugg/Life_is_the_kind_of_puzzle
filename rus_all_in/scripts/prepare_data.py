"""Safely extract and validate the original AIIJC puzzle archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ARCHIVES_DIR = RAW_DIR / "archives"


@dataclass(frozen=True)
class ArchiveSpec:
    filename: str
    destination: Path | None
    expected_files: int


SPECS = (
    ArchiveSpec("train.zip", RAW_DIR / "train", 14_000),
    ArchiveSpec("test.zip", RAW_DIR / "test", 700),
    ArchiveSpec("submission.zip", None, 700),
)


def _safe_png_members(archive: Path) -> list[ZipInfo]:
    with ZipFile(archive) as zip_file:
        members = [member for member in zip_file.infolist() if not member.is_dir()]

    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate member names in {archive}")

    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive member in {archive}: {name}")
        if path.suffix.lower() != ".png":
            raise ValueError(f"Unexpected non-PNG member in {archive}: {name}")
    return members


def _inspect_archives() -> dict[str, set[str]]:
    manifests: dict[str, set[str]] = {}
    for spec in SPECS:
        archive = ARCHIVES_DIR / spec.filename
        if not archive.is_file():
            raise FileNotFoundError(f"Missing organizer archive: {archive}")
        members = _safe_png_members(archive)
        if len(members) != spec.expected_files:
            raise ValueError(
                f"{archive} contains {len(members)} files, expected {spec.expected_files}"
            )
        manifests[spec.filename] = {member.filename for member in members}

    train_names = manifests["train.zip"]
    inputs = {name.removeprefix("inputs/") for name in train_names if name.startswith("inputs/")}
    targets = {name.removeprefix("targets/") for name in train_names if name.startswith("targets/")}
    if len(inputs) != 7_000 or inputs != targets:
        raise ValueError("train.zip must contain 7,000 matching input/target filenames")

    test_names = manifests["test.zip"]
    submission_names = manifests["submission.zip"]
    if any("/" in name for name in test_names | submission_names):
        raise ValueError("Test and submission PNG files must be in the archive root")
    if test_names != submission_names:
        raise ValueError("test.zip and submission.zip must contain the same filenames")

    return manifests


def _validate_images(destination: Path, expected_names: set[str]) -> None:
    actual = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*.png")
        if path.is_file()
    }
    if actual != expected_names:
        missing = len(expected_names - actual)
        extra = len(actual - expected_names)
        raise ValueError(f"Invalid extraction at {destination}: missing={missing}, extra={extra}")

    for relative_name in sorted(actual):
        with Image.open(destination / relative_name) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (480, 480):
                raise ValueError(
                    f"Invalid image {destination / relative_name}: "
                    f"format={image.format}, mode={image.mode}, size={image.size}"
                )


def _extract(archive: Path, destination: Path, expected_names: set[str]) -> None:
    if destination.exists():
        _validate_images(destination, expected_names)
        print(f"verified  {destination.relative_to(ROOT)} ({len(expected_names):,} PNG)")
        return

    temporary = destination.with_name(f".{destination.name}.extracting")
    if temporary.exists():
        raise FileExistsError(
            f"Incomplete extraction exists at {temporary}; inspect and remove it before retrying"
        )
    temporary.mkdir(parents=True)

    with ZipFile(archive) as zip_file:
        zip_file.extractall(temporary)
    _validate_images(temporary, expected_names)
    temporary.replace(destination)
    print(f"extracted {destination.relative_to(ROOT)} ({len(expected_names):,} PNG)")


def main() -> None:
    manifests = _inspect_archives()
    for spec in SPECS:
        if spec.destination is not None:
            _extract(
                ARCHIVES_DIR / spec.filename,
                spec.destination,
                manifests[spec.filename],
            )
    print("ready: organizer archives and extracted datasets are valid")


if __name__ == "__main__":
    main()
