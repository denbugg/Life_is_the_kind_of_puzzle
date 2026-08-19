import re
import struct
import sys
import zipfile
from pathlib import Path


archive = Path(sys.argv[1])
name_pattern = re.compile(r"img_\d{6}\.png")

with zipfile.ZipFile(archive) as zf:
    names = zf.namelist()
    assert len(names) == 700, f"expected 700 files, got {len(names)}"
    assert len(set(names)) == len(names), "duplicate filenames"
    assert all(name_pattern.fullmatch(name) for name in names), "unexpected path or filename"

    for name in names:
        data = zf.read(name)
        assert data[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {name}"
        assert data[12:16] == b"IHDR", f"missing IHDR: {name}"
        width, height, bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
        assert (width, height) == (480, 480), (name, width, height)
        assert bit_depth == 8, (name, bit_depth)
        assert color_type == 2, (name, color_type)

print(f"archive={archive}")
print(f"files={len(names)}")
print(f"first={min(names)} last={max(names)}")
print("png_format=ok size=480x480 mode=RGB paths=flat")
