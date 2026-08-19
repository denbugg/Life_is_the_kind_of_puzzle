#!/usr/bin/env bash
set -euo pipefail

cd /home/kva/pazzle_directional_transformer
out_dir="outputs_border_pipeline_weco_full"
mkdir -p "$out_dir"

MODE=submission \
OUT_DIR="$out_dir" \
MAX_TEST=0 \
.venv/bin/python evaluate_submit_border_pipeline.py \
  2>&1 | tee "$out_dir/run.log"

.venv/bin/python - <<'PY' | tee "$out_dir/validation.json"
import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image

root = Path("outputs_border_pipeline_weco_full")
archive = root / "submission_border_pipeline.zip"
images = root / "submission_images"
pngs = sorted(images.glob("*.png"))
with zipfile.ZipFile(archive) as zipped:
    names = zipped.namelist()
    bad_member = zipped.testzip()
    root_level = all("/" not in name and name.endswith(".png") for name in names)
sizes = set()
modes = set()
for path in pngs:
    with Image.open(path) as image:
        sizes.add(image.size)
        modes.add(image.mode)
report = {
    "image_count": len(pngs),
    "zip_count": len(names),
    "zip_unique": len(set(names)) == len(names),
    "zip_root_level_pngs": root_level,
    "zip_bad_member": bad_member,
    "sizes": sorted(map(list, sizes)),
    "modes": sorted(modes),
    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
}
assert report["image_count"] == 700
assert report["zip_count"] == 700
assert report["zip_unique"] and report["zip_root_level_pngs"]
assert report["zip_bad_member"] is None
assert sizes == {(480, 480)} and modes == {"RGB"}
print(json.dumps(report, indent=2))
PY
