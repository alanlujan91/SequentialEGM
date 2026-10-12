"""Repository locations that scripts resolve their inputs and outputs against,
independent of the directory they are launched from, and the CSV writer and
reader for the artifacts they persist there.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "content" / "figures"


def write_rows(path, header, rows) -> None:
    """Write ``header`` then ``rows`` as CSV to ``path``, creating its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def read_rows(path):
    """The rows of the CSV at ``path`` as dicts keyed by its header."""
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))
