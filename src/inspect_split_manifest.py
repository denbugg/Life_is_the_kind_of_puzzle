"""Read-only locator for 5360/670/670 sample splits in JSON manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def is_name_list(value: Any) -> bool:
    return isinstance(value, list) and value and all(isinstance(item, str) and item.endswith('.png') for item in value)


def walk(value: Any, path: str = '$') -> list[tuple[str, int, list[str]]]:
    found: list[tuple[str, int, list[str]]] = []
    if is_name_list(value):
        found.append((path, len(value), value[:3]))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(walk(item, f'{path}.{key}'))
    elif isinstance(value, list):
        for index, item in enumerate(value[:1000]):
            found.extend(walk(item, f'{path}[{index}]'))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    with args.manifest.open(encoding='utf-8') as handle:
        data = json.load(handle)
    candidates = walk(data)
    print('root_keys=' + (','.join(data.keys()) if isinstance(data, dict) else type(data).__name__))
    if isinstance(data, dict) and 'split' in data:
        print('SPLIT_SECTION=' + json.dumps(data['split'], ensure_ascii=False))
    for path, size, sample in candidates:
        if not path.startswith('$.groups.'):
            print(json.dumps({'path': path, 'count': size, 'sample': sample}, ensure_ascii=False))


if __name__ == '__main__':
    main()
