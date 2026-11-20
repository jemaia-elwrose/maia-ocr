"""UAE number plates from OCR lines: emirate, code and number.

UAE plates combine a number (1-5 digits) with a code that is either letters
(Dubai: "D 35071") or a category number (Abu Dhabi's red box "17", Sharjah "3"),
plus small print: the emirate name, "U.A.E", "A.D", "COM", "EXPORT", a year,
and Arabic text. Camera OCR garbles the small print ("SHARIAR", "DU2HT",
"UAEAD") and invents tokens from the Arabic script, so the parser works on
tokens and their sizes rather than one flat string:

* the plate number is the tallest token with digits; on a 6-character token the
  first character is the code (UAE numbers have at most 5 digits);
* a code is a nearby uppercase 1-2 letter token, or a 1-50 number, of reasonable size;
* emirate names are fuzzy-matched; known small print and years are ignored;
* a code that reads as ``0`` can't be a category (they start at 1), so it is a
  letter the model couldn't tell apart: reported as ``?`` with candidates
  (D, O, Q), for the per-track vote to settle on frames that read it clearly.

The plate *key* (code + number, e.g. ``D35071``) is what to vote on and store.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

from .plates import TO_DIGIT, TO_LETTER, PlateReading
from .types import TextLine

EMIRATES = {
    "DUBAI": "Dubai",
    "SHARJAH": "Sharjah",
    "AJMAN": "Ajman",
    "ABUDHABI": "Abu Dhabi",
    "FUJAIRAH": "Fujairah",
    "RASALKHAIMAH": "Ras Al Khaimah",
    "RAK": "Ras Al Khaimah",
    "UMMALQUWAIN": "Umm Al Quwain",
    "UAQ": "Umm Al Quwain",
}
SMALL_PRINT = {"UAE", "UAEAD", "AD", "COM", "EXPORT", "TAXI", "PRIVATE", "PUBLIC", "TRANSPORT", "CLASSIC", "TRADE"}
AMBIGUOUS_ZERO = ["D", "O", "Q"]


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _tokens(lines: list[TextLine]):
    for ln in lines:
        for raw in re.split(r"[\s\-_/|:]+", _ascii(ln.text)):
            raw = raw.strip(".,;'\"()[]{}")
            if raw and re.search(r"[A-Za-z0-9]", raw):
                yield raw, ln


def emirate_of(token: str) -> str | None:
    t = re.sub(r"[^A-Z0-9]", "", token.upper())
    if not t:
        return None
    if t.endswith("AD") and t.startswith("UAE"):  # "U.A.E A.D" read as one word
        return "Abu Dhabi"
    if t == "AD" or (len(t) == 3 and t.endswith("AD")):  # "A.D", sometimes read with a stray letter
        return "Abu Dhabi"
    letters = "".join(TO_LETTER.get(c, c) for c in t)
    if len(letters) < 3:
        return None
    best = max(EMIRATES, key=lambda k: difflib.SequenceMatcher(None, letters, k).ratio())
    ratio = difflib.SequenceMatcher(None, letters, best).ratio()
    # short words: one or two garbled letters already cost a lot of ratio ("DU2HT" for "DUBAI"),
    # so also accept a match on the first two letters at about the right length
    same_start = letters[:2] == best[:2] and abs(len(letters) - len(best)) <= 1 and ratio >= 0.4
    return EMIRATES[best] if ratio >= 0.7 or same_start else None


def _is_year(token: str, line: TextLine, lines: list[TextLine]) -> bool:
    t = re.sub(r"[^0-9]", "", token)
    return len(t) == 4 and 1990 <= int(t) <= 2040 and line.height < 0.75 * max(x.height for x in lines)


def _is_small_print(token: str, line: TextLine, lines: list[TextLine]) -> bool:
    word = re.sub(r"[^A-Z0-9]", "", token.upper())
    return word in SMALL_PRINT or emirate_of(word) is not None or _is_year(token, line, lines)


def _same_row(a: TextLine, b: TextLine) -> bool:
    inter = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
    return inter > 0.5 * min(a.height, b.height)


def parse(lines: list[TextLine]) -> PlateReading:
    raw = " ".join(ln.text for ln in lines)
    tokens = list(_tokens(lines))
    emirate = next((e for e in (emirate_of(t) for t, _ in tokens) if e), None)

    # 1. plate number: tallest token that is mostly digits once confusions are undone
    numberish = []
    for t, ln in tokens:
        clean = re.sub(r"[^A-Z0-9]", "", t.upper())
        digits = "".join(TO_DIGIT.get(c, c) for c in clean)
        if 1 <= len(clean) <= 6 and sum(c.isdigit() for c in clean) >= max(1, len(clean) - 2) and digits.isdigit():
            if _is_year(digits, ln, lines):
                continue  # a year in the small print
            numberish.append((ln.height, clean, ln))
    if not numberish:
        return PlateReading("", raw, False, 0, {"emirate": emirate})
    height, number_raw, number_line = max(numberish, key=lambda x: x[0])

    code, candidates, corrections = None, [], 0
    if len(number_raw) == 6:  # code glued to the number
        head, number_raw = number_raw[0], number_raw[1:]
        if head.isalpha():
            code = head
        elif head == "0":
            code, candidates = "?", list(AMBIGUOUS_ZERO)
        else:
            code, corrections = TO_LETTER.get(head, head), 1
    number = "".join(TO_DIGIT.get(c, c) for c in number_raw)
    corrections += sum(a != b for a, b in zip(number, number_raw, strict=True))

    # 2. code in its own token: uppercase letters, or a 1-50 category, of reasonable size.
    #    Small-print rows (emirate, year, COM, U.A.E) also hold stray tokens read from the
    #    Arabic text ("40"); a number the same size as small print on its row is rejected,
    #    while a real code box (Abu Dhabi's red "14") is much taller than the print beside it.
    small_rows = [ln for t, ln in tokens if ln is not number_line and _is_small_print(t, ln, lines)]
    if code is None:
        best = None
        nx, ny = number_line.center
        for t, ln in tokens:
            if ln is number_line or ln.height < 0.4 * height:
                continue
            word = re.sub(r"[^A-Za-z0-9]", "", t)
            if word.upper() in SMALL_PRINT or emirate_of(word) or not word:
                continue
            if word.isdigit() and any(
                _same_row(ln, o) and max(ln.height, o.height) <= 1.5 * min(ln.height, o.height)
                for o in small_rows
                if o is not ln
            ):
                continue  # as small as the small print beside it: part of it, not a code
            if (word.isalpha() and word.isupper() and len(word) <= 2) or (word.isdigit() and 1 <= int(word) <= 50):
                cx, cy = ln.center
                distance = abs(cx - nx) + abs(cy - ny)
                if best is None or distance < best[0]:
                    best = (distance, word.upper())
        code = best[1] if best else None

    key = f"{code or ''}{number}"
    valid = 1 <= len(number) <= 5 and number.isdigit()
    fields = {"emirate": emirate, "code": code, "number": number, "code_candidates": candidates}
    text = f"{code} {number}" if code else number
    return PlateReading(text if valid else "", raw, valid, corrections, fields | {"key": key})
