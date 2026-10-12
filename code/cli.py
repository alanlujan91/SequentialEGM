"""Console setup shared by the command-line scripts (stdlib only, so every
script can import it).
"""

import logging


def start_logging() -> None:
    """Message-only INFO logging for a script's ``main``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def format_table(header, rows) -> str:
    """A header and rows of cells as an aligned plain-text table, the header
    underlined with hyphens: the tables the example notebooks print.
    """
    cells = [list(header), *([str(c) for c in row] for row in rows)]
    widths = [max(len(row[j]) for row in cells) for j in range(len(header))]
    lines = [
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in cells
    ]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(line.rstrip() for line in lines)
