"""Build reproducible inventories for the historical puzzle-research repository."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

M_REF = "origin/autoresearch/pazzle-fixed-orientation-20260813"
M_PATH = "autoresearch-runs/pazzle-mgc-restoration-20260818/EXPERIMENTS.md"
BASE_REF = "origin/pasha883"


@dataclass(frozen=True)
class RemoteRef:
    name: str
    full_name: str
    sha: str
    date: str
    subject: str
    symbolic_target: str


def git(repo: Path, *args: str) -> str:
    """Run a read-only Git query and return decoded stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def read_remote_refs(repo: Path) -> list[RemoteRef]:
    separator = "\x1f"
    output = git(
        repo,
        "for-each-ref",
        f"--format=%(refname:short){separator}%(refname){separator}%(objectname)"
        f"{separator}%(committerdate:short){separator}%(subject){separator}%(symref:short)",
        "refs/remotes/origin",
    )
    refs = []
    for line in output.splitlines():
        name, full_name, sha, date, subject, symbolic_target = line.split(separator)
        refs.append(RemoteRef(name, full_name, sha, date, subject, symbolic_target))
    return refs


def ref_family(name: str) -> str:
    if name in {"origin", "origin/MAESTRO", "origin/pasha883"}:
        return "legacy / alias"
    if name in {"origin/Taska-govna", "origin/таска-говно"}:
        return "архив / legacy"
    if name.startswith("origin/agent/"):
        return "ранние agent-прототипы"
    if name == "origin/autoresearch/fast-score-gen1":
        return "E: общий отчёт"
    if re.match(r"origin/autoresearch/e\d", name):
        return "E: быстрые абляции"
    if name.endswith("pazzle-fixed-orientation-cb1"):
        return "ORBIT / R / P"
    if name.endswith("pazzle-fixed-orientation-20260813"):
        return "M"
    if name == "origin/TASKA-GOVNO-EBANOE":
        return "M421–M479"
    if name == "origin/codex/contour-normalization":
        return "V10–V30"
    if name in {
        "origin/codex/autoresearch-puzzle-v32-noise",
        "origin/codex/autoresearch-puzzle-v33-transformer",
    }:
        return "V31–V33"
    return "прочее"


def count(repo: Path, *args: str) -> int:
    return int(git(repo, *args).strip())


def build_branch_inventory(repo: Path, refs: list[RemoteRef]) -> str:
    named_refs = [ref for ref in refs if not ref.symbolic_target]
    all_named = [ref.name for ref in named_refs]
    remote_url = git(repo, "remote", "get-url", "origin").strip()
    rows = []
    for ref in refs:
        if ref.symbolic_target:
            rows.append(
                (
                    ref.name,
                    ref.sha[:10],
                    "symbolic",
                    "—",
                    "—",
                    "—",
                    f"alias → {ref.symbolic_target}",
                )
            )
            continue

        others = [name for name in all_named if name != ref.name]
        exclusive_args = ["rev-list", "--count", ref.name]
        if others:
            exclusive_args.extend(["--not", *others])
        exclusive = count(repo, *exclusive_args)
        total = count(repo, "rev-list", "--count", ref.name)
        files = len(git(repo, "ls-tree", "-r", "--name-only", ref.name).splitlines())

        merge_base = subprocess.run(
            ["git", "merge-base", BASE_REF, ref.name],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if merge_base.returncode == 0:
            after_base = count(repo, "rev-list", "--count", f"{BASE_REF}..{ref.name}")
            relation = f"+{after_base} от `{BASE_REF}`"
        else:
            relation = "независимая история"

        rows.append(
            (
                ref.name,
                ref.sha[:10],
                str(total),
                str(exclusive),
                str(files),
                ref.date,
                f"{ref_family(ref.name)}; {relation}",
            )
        )

    lines = [
        "# Машинный инвентарь remote-веток",
        "",
        "> Генерируется `scripts/build_prior_research_index.py`; ручные выводы находятся в "
        "[`branches.md`](../branches.md).",
        "",
        f"Источник: `{remote_url}`. Зафиксировано именованных веток: **{len(named_refs)}**; "
        f"remote-ссылок вместе с `origin/HEAD`: **{len(refs)}**.",
        "",
        "`Уник.` — число коммитов, достижимых только из этой remote-ссылки среди "
        "зафиксированного набора. Ноль может означать alias или историю, полностью вошедшую "
        "в другую ветку.",
        "",
        "| Remote ref | Tip | Коммитов | Уник. | Файлов | Дата tip | Семейство / линия |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_cell(cell) for cell in row) + " |")
    lines.extend(
        [
            "",
            "## Заголовки tip-коммитов",
            "",
        ]
    )
    for ref in refs:
        suffix = f" (alias → `{ref.symbolic_target}`)" if ref.symbolic_target else ""
        lines.append(f"- `{ref.name}` `{ref.sha[:10]}` — {ref.subject}{suffix}")
    return "\n".join(lines) + "\n"


def blame_line_commits(repo: Path, ref: str, path: str) -> dict[int, str]:
    output = git(repo, "blame", "--line-porcelain", ref, "--", path)
    mapping: dict[int, str] = {}
    current_sha = ""
    current_final = 0
    for line in output.splitlines():
        header = re.match(r"^([0-9a-f]{40}) \d+ (\d+)(?: \d+)?$", line)
        if header:
            current_sha = header.group(1)
            current_final = int(header.group(2))
        elif line.startswith("\t"):
            mapping[current_final] = current_sha
            current_final += 1
    return mapping


def build_m_ledger(repo: Path) -> str:
    tip = git(repo, "rev-parse", M_REF).strip()
    source = git(repo, "show", f"{M_REF}:{M_PATH}")
    blame = blame_line_commits(repo, M_REF, M_PATH)
    records = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        if not re.match(r"^M\d+(?:[- ][A-Z][A-Z0-9 -]*)?\s*\|", line):
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 3:
            continue
        label, title, verdict = cells[:3]
        number = int(re.match(r"M(\d+)", label).group(1))
        records.append((number, label, title, verdict, blame[line_number][:10], line_number))

    # Corrections are sometimes appended much later than the experiment they amend.
    # Group them beside the original ID while preserving their source line as provenance.
    records.sort(key=lambda record: (record[0], record[5]))

    numbers = {record[0] for record in records}
    missing = sorted(set(range(1, max(numbers) + 1)) - numbers)
    missing_text = ", ".join(f"M{number}" for number in missing) or "нет"
    lines = [
        "# Полный поисковый реестр M-серии",
        "",
        "> Генерируется `scripts/build_prior_research_index.py` из итогового журнала ветки. "
        "Интерпретация и поправки к выводам находятся в "
        "[`knowledge-base.md`](../knowledge-base.md).",
        "",
        f"Источник: `{M_REF}` (`{tip[:10]}`), `{M_PATH}`. В таблице **{len(records)}** "
        f"именованных записей; базовые номера M1–M{max(numbers)}, пропущено в источнике: "
        f"**{missing_text}**. Варианты `CORRECTION`, `FINAL`, `GATES` и повторные проверки "
        "сохранены отдельными строками.",
        "",
        "`Вердикт журнала` воспроизводит текст источника в момент записи и сам по себе "
        "не является последним словом: строки `IN PROGRESS`, `RUNNING` и ранние `ACCEPTED` "
        "нужно читать вместе с одноимёнными `RESULT`/`CORRECTION`/`FINAL` и ручным аудитом.",
        "",
        "Искать идею удобнее обычным полнотекстовым поиском, например "
        "`rg -ni 'sinkhorn|spectral|restor|chooser' docs/prior-research`.",
        "",
    ]

    current_bucket = None
    for number, label, title, verdict, commit, line_number in records:
        bucket = ((number - 1) // 50) * 50 + 1
        if bucket != current_bucket:
            if current_bucket is not None:
                lines.append("")
            bucket_end = min(bucket + 49, max(numbers))
            lines.extend(
                [
                    f"## M{bucket}–M{bucket_end}",
                    "",
                    "| ID | Проверка / идея | Вердикт журнала | Commit | Строка |",
                    "|---|---|---|---:|---:|",
                ]
            )
            current_bucket = bucket
        cells = [
            f"`{label}`",
            markdown_cell(title),
            markdown_cell(verdict),
            f"`{commit}`",
            str(line_number),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-repo", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/prior-research/generated"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.research_repo.expanduser().resolve()
    output_dir = args.output_dir.resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"Not a Git working tree: {repo}")

    refs = read_remote_refs(repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "branch-inventory.md").write_text(
        build_branch_inventory(repo, refs), encoding="utf-8"
    )
    (output_dir / "m-experiments.md").write_text(build_m_ledger(repo), encoding="utf-8")
    print(f"Wrote {output_dir / 'branch-inventory.md'}")
    print(f"Wrote {output_dir / 'm-experiments.md'}")


if __name__ == "__main__":
    main()
