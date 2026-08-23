from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont

ROOT = Path(r"E:\pazzle_work\pazzle_fixed_orientation_20260813\visual_audit")
TARGETS = Path(r"E:\pazzle_data\train\targets")
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "rank96_outputs"
NAMES = ["img_000014.png", "img_000020.png"]
TILE = 480
HEADER = 38
MARGIN = 12
COLUMNS = ["RAW INPUT", "CLEAN TARGET", "CANONICAL RANK96", "|TARGET - RANK96|"]


def load(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")

font = ImageFont.load_default()
width = MARGIN + len(COLUMNS) * (TILE + MARGIN)
height = MARGIN + len(NAMES) * (HEADER + TILE + MARGIN)
sheet = Image.new("RGB", (width, height), (20, 24, 32))
draw = ImageDraw.Draw(sheet)
for row, name in enumerate(NAMES):
    raw = load(INPUTS / name)
    target = load(TARGETS / name)
    canonical = load(OUTPUTS / name)
    difference = ImageEnhance.Contrast(ImageChops.difference(target, canonical)).enhance(2.5)
    images = [raw, target, canonical, difference]
    y = MARGIN + row * (HEADER + TILE + MARGIN)
    for col, (label, image) in enumerate(zip(COLUMNS, images)):
        x = MARGIN + col * (TILE + MARGIN)
        draw.text((x, y + 2), f"{name} — {label}", fill=(235, 240, 248), font=font)
        sheet.paste(image, (x, y + HEADER))

out = ROOT / "rank96_dev_visual_audit.png"
sheet.save(out, quality=95)
print(out)
