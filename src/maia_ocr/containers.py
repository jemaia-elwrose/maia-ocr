"""Shipping container codes (ISO 6346), e.g. ``CSQU 305438 3``.

Owner code (3 letters) + category (U, J or Z) + 6-digit serial + check digit.
The check digit makes OCR results verifiable: a misread character almost
always breaks it, so ports and yards can trust codes that pass.
"""

from __future__ import annotations

import re

from .plates import TO_DIGIT, TO_LETTER, normalize

# Letter values skip multiples of 11 (11, 22, 33).
_VALUES = {}
_v = 10
for _ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    if _v % 11 == 0:
        _v += 1
    _VALUES[_ch] = _v
    _v += 1
_VALUES.update({str(d): d for d in range(10)})

_PATTERN = re.compile(r"^[A-Z]{3}[UJZ]\d{7}$")


def check_digit(code10: str) -> int:
    """Check digit for the first 10 characters (owner + category + serial)."""
    total = sum(_VALUES[ch] * (2**i) for i, ch in enumerate(code10.upper()))
    return total % 11 % 10


def is_valid(code: str) -> bool:
    code = normalize(code)
    return bool(_PATTERN.match(code)) and check_digit(code[:10]) == int(code[10])


def correct(text: str) -> str | None:
    """Fix letter/digit confusions by position (4 letters, then 7 digits) and return the
    code if the check digit then agrees; otherwise ``None``."""
    clean = normalize(text)
    if len(clean) != 11:
        return None
    fixed = "".join(TO_LETTER.get(c, c) for c in clean[:4]) + "".join(TO_DIGIT.get(c, c) for c in clean[4:])
    return fixed if is_valid(fixed) else None


def find(lines_or_text) -> list[str]:
    """Valid container codes in OCR output (codes are often split over 2-3 lines, so lines are joined)."""
    if isinstance(lines_or_text, str):
        text = lines_or_text
    else:
        from .text import reading_order

        text = " ".join(ln.text for ln in reading_order(lines_or_text))
    found = []
    flat = normalize(text)
    for i in range(len(flat) - 10):
        code = correct(flat[i : i + 11])
        if code and code not in found:
            found.append(code)
    return found
