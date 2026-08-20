"""Deterministically bundle the E14/E18b sidecars into one Kaggle code file."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPO = Path(__file__).resolve().parent
MAIN = REPO / "kaggle_solve_puzzles.py"
SIDECARS = (
    ("kaggle_e14_solver", REPO / "kaggle_e14_solver.py"),
    ("kaggle_e18b_postprocess", REPO / "kaggle_e18b_postprocess.py"),
)
OUTPUT = REPO / "kaggle_solve_puzzles_e18b.py"


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def render() -> str:
    main_source = MAIN.read_text()
    for name, _ in SIDECARS:
        needle = f"import {name}\n"
        if main_source.count(needle) != 1:
            raise RuntimeError(f"expected exactly one {needle.strip()!r} in {MAIN}")
        main_source = main_source.replace(needle, "")

    lines = [
        '"""Generated self-contained E18b Kaggle entrypoint; do not edit."""',
        "# Generated deterministically by build_e18b_self_contained.py.",
        "import sys as _embedded_sys",
        "import types as _embedded_types",
        "",
        "def _load_embedded_module(name, source):",
        "    module = _embedded_types.ModuleType(name)",
        "    module.__file__ = f\"<embedded:{name}>\"",
        "    _embedded_sys.modules[name] = module",
        "    exec(compile(source, module.__file__, \"exec\"), module.__dict__)",
        "    return module",
        "",
    ]
    for name, path in SIDECARS:
        source = path.read_text()
        lines.extend([
            f"# embedded {path.name} sha256={_digest(source)}",
            f"{name} = _load_embedded_module({name!r}, {source!r})",
            "",
        ])
    lines.extend(["del _load_embedded_module", "", main_source])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text() != expected:
            raise SystemExit(f"stale self-contained bundle: {OUTPUT}")
        print(f"bundle_ok={OUTPUT.name} sha256={_digest(expected)}")
        return
    OUTPUT.write_text(expected)
    print(f"wrote={OUTPUT} sha256={_digest(expected)}")


if __name__ == "__main__":
    main()
