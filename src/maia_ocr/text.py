"""Layout helpers: put OCR lines in reading order and group them into rows."""

from __future__ import annotations

from .types import TextLine


def rows(lines: list[TextLine], overlap: float = 0.5) -> list[list[TextLine]]:
    """Group lines whose vertical extents overlap (same row), top to bottom, each left to right."""
    out: list[list[TextLine]] = []
    for ln in sorted(lines, key=lambda line: line.center[1]):
        y1, y2 = ln.box[1], ln.box[3]
        for row in out:
            r1, r2 = min(r.box[1] for r in row), max(r.box[3] for r in row)
            inter = min(y2, r2) - max(y1, r1)
            if inter > overlap * min(y2 - y1, r2 - r1):
                row.append(ln)
                break
        else:
            out.append([ln])
    return [sorted(row, key=lambda line: line.center[0]) for row in out]


def reading_order(lines: list[TextLine]) -> list[TextLine]:
    return [ln for row in rows(lines) for ln in row]


def full_text(lines: list[TextLine]) -> str:
    """One string, rows separated by newlines."""
    return "\n".join(" ".join(ln.text for ln in row) for row in rows(lines))
