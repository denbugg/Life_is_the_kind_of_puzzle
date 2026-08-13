"""Build a deterministic source-group-disjoint 5360/670/670 PAZZLE split.

The input manifest's groups are authoritative.  Group, not file, allocation
ensures no known duplicate source group can cross FIT/CAL/DEV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', default='ORBIT-24-PGA1-source-split-v1')
    args = parser.parse_args()
    payload: dict[str, Any] = json.loads(args.groups.read_text(encoding='utf-8'))
    groups: dict[str, list[str]] = payload['groups']
    ranked = sorted(
        groups,
        key=lambda group: (hashlib.sha256((args.seed + '\0' + group).encode()).hexdigest(), group),
    )
    sizes = {'fit': 5360, 'cal': 670, 'dev': 670}
    allocation: dict[str, list[str]] = {}
    cursor = 0
    for name, desired_count in sizes.items():
        chosen: list[str] = []
        while cursor < len(ranked) and len(chosen) < desired_count:
            members = groups[ranked[cursor]]
            if len(chosen) + len(members) > desired_count:
                raise RuntimeError(f'Group {ranked[cursor]} size {len(members)} prevents exact {name} allocation')
            chosen.extend(sorted(members))
            cursor += 1
        if len(chosen) != desired_count:
            raise RuntimeError(f'{name} got {len(chosen)}, expected {desired_count}')
        allocation[name] = chosen
    reserve: list[str] = []
    while cursor < len(ranked):
        reserve.extend(sorted(groups[ranked[cursor]]))
        cursor += 1
    allocation['reserve'] = reserve
    all_names = allocation['fit'] + allocation['cal'] + allocation['dev'] + allocation['reserve']
    if len(all_names) != 7000 or len(set(all_names)) != 7000:
        raise RuntimeError('Split coverage/uniqueness invariant failed')
    source_sha = sha256_file(args.groups)
    out = {
        'schema': 'orbit24-pga1-source-disjoint-split-v1',
        'seed': args.seed,
        'source_groups_manifest': str(args.groups),
        'source_groups_manifest_sha256': source_sha,
        'source_group_contract': payload.get('builder_contract', {}).get('fixed_algorithms', payload.get('schema_version')),
        'counts': {name: len(value) for name, value in allocation.items()},
        'splits': allocation,
        'invariants': {
            'total_unique_names': len(set(all_names)),
            'total_groups': len(groups),
            'groups_are_disjoint_by_allocation': True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(args.out), 'counts': out['counts'], 'source_sha256': source_sha, 'groups': len(groups)}, indent=2))


if __name__ == '__main__':
    main()
