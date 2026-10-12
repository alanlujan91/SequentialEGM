"""reproduce.sh runs exactly the commands of code/README.md's exhibit table, so the
table and the script cannot drift apart."""

from __future__ import annotations

import os
import re
import subprocess

from repo_paths import ROOT

COMMAND = re.compile(r"PYTHONPATH=code uv run python [^`\n]+")


def exhibit_commands() -> list[str]:
    """The command column of code/README.md's exhibit table, in table order."""
    lines = (ROOT / "code" / "README.md").read_text().splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("| Paper exhibit")
    )
    table = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        table.append(line.split("|")[2])
    return [m.group(0).strip() for cell in table for m in COMMAND.finditer(cell)]


def script_commands(name: str) -> list[str]:
    text = (ROOT / name).read_text()
    return [line for line in text.splitlines() if COMMAND.fullmatch(line)]


def test_reproduce_runs_every_exhibit_command_and_nothing_else() -> None:
    table = exhibit_commands()
    assert len(table) > 10  # the parser found the table
    assert script_commands("reproduce.sh") == table


def test_the_reproduce_scripts_are_executable() -> None:
    """Executable on disk and, where git tracks them, in the index the public cut
    reads (with ``core.filemode`` false, git records new files as 100644)."""
    for name in ("reproduce.sh", "reproduce_min.sh"):
        assert os.access(ROOT / name, os.X_OK), name
        staged = subprocess.run(
            ["git", "ls-files", "-s", "--", name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        assert not staged or staged.startswith("100755"), staged
