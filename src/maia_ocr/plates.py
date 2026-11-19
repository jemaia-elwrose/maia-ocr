"""Number-plate text: clean up, fix OCR confusions using the plate format, vote over frames.

A plate format is a *template* of character classes:

    L = letter, D = digit, A = letter or digit, X{m,n} = between m and n of X

e.g. ``"LL D{1,2} L{1,3} DDDD"`` covers Indian plates such as ``MH12AB1234``.
OCR often reads ``0`` for ``O`` or ``8`` for ``B``. Knowing that a position must be
a digit, those slips can be undone: :func:`correct` tries every length the
template allows and keeps the reading that needed the fewest substitutions.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

#: Common plate formats. Real plate rules have more detail (series, emirate codes, colours);
#: treat these as starting points and pass your own template for production use.
FORMATS: dict[str, list[str]] = {
    "generic": ["A{2,10}"],
    "uae": ["L{0,2} D{1,5}"],  # optional code letter(s) + up to 5 digits, e.g. "AA 12345"
    "india": ["LL D{1,2} L{0,3} DDDD"],  # e.g. MH 12 AB 1234; Bharat series not included
    "uk": ["LL DD LLL"],  # current format, e.g. AB12 CDE
    "eu-de": ["L{1,3} L{1,2} D{1,4}"],  # Germany, e.g. M AB 1234
}

TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "U": "0",
    "I": "1",
    "L": "1",
    "J": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}

_TOKEN = re.compile(r"([LDA])(?:\{(\d+),(\d+)\})?")


def normalize(text: str) -> str:
    """Upper-case and keep only A-Z and 0-9 (drops spaces, dashes, dots, emirate names' punctuation)."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def expand(template: str) -> list[str]:
    """``"LL D{1,2}"`` -> ``["LLD", "LLDD"]`` (spaces are ignored)."""
    parts = []
    for cls, lo, hi in _TOKEN.findall(template.replace(" ", "")):
        lo, hi = (int(lo), int(hi)) if lo else (1, 1)
        parts.append([cls * n for n in range(lo, hi + 1)])
    return sorted({"".join(p) for p in itertools.product(*parts)}, key=len)


def _fit(text: str, pattern: str) -> tuple[str, int] | None:
    if len(text) != len(pattern):
        return None
    out, changes = [], 0
    for ch, cls in zip(text, pattern, strict=True):
        if cls == "L" and ch.isdigit():
            if ch not in TO_LETTER:
                return None
            ch, changes = TO_LETTER[ch], changes + 1
        elif cls == "D" and ch.isalpha():
            if ch not in TO_DIGIT:
                return None
            ch, changes = TO_DIGIT[ch], changes + 1
        out.append(ch)
    return "".join(out), changes


@dataclass
class PlateReading:
    text: str  # corrected plate (or the normalised raw text if it fits no template)
    raw: str  # what OCR returned
    valid: bool  # matches a template
    corrections: int  # characters changed to fit


def correct(text: str, formats: str | list[str] = "generic", max_corrections: int = 2) -> PlateReading:
    """Fit OCR output to a plate format, undoing letter/digit confusions."""
    templates = FORMATS[formats] if isinstance(formats, str) else formats
    clean = normalize(text)
    best: tuple[str, int] | None = None
    for template in templates:
        for pattern in expand(template):
            fit = _fit(clean, pattern)
            if fit and fit[1] <= max_corrections and (best is None or fit[1] < best[1]):
                best = fit
    if best is None:
        return PlateReading(clean, text, False, 0)
    return PlateReading(best[0], text, True, best[1])


def merge_rows(lines) -> str:
    """Join a two-row plate read as separate lines: top row first, each row left to right."""
    from .text import reading_order

    return "".join(normalize(ln.text) for ln in reading_order(lines))


class PlateVoter:
    """Per-track plate decision. Single frames misread a character or two; most frames don't.

    Readings of the most common length are combined position by position, weighted
    by OCR confidence, and a decision is returned once ``min_votes`` agree on length.
    """

    def __init__(self, window: int = 20, min_votes: int = 3) -> None:
        self.min_votes = min_votes
        self._reads: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id: int, text: str, score: float = 1.0) -> str | None:
        text = normalize(text)
        if text:
            self._reads[track_id].append((text, float(score)))
        return self.decision(track_id)

    def decision(self, track_id: int) -> str | None:
        reads = self._reads.get(track_id)
        if not reads:
            return None
        length, count = Counter(len(t) for t, _ in reads).most_common(1)[0]
        if count < self.min_votes:
            return None
        same = [(t, s) for t, s in reads if len(t) == length]
        chars = []
        for i in range(length):
            votes: Counter = Counter()
            for t, s in same:
                votes[t[i]] += s
            chars.append(votes.most_common(1)[0][0])
        return "".join(chars)

    def forget(self, track_id: int) -> None:
        self._reads.pop(track_id, None)
