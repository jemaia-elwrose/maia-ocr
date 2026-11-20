"""UAE plate parsing on real OCR output (RapidOCR PP-OCRv6 on real plate photos).
Plate numbers are changed to keep real registrations out of the repo; everything else
(garbled small print, Arabic-script noise, box geometry) is exactly what OCR returned."""

import json
from pathlib import Path

import numpy as np
import pytest

from maia_ocr import PlateVoter, uae
from maia_ocr.types import TextLine

FIXTURES = json.loads((Path(__file__).parent / "uae_ocr_fixtures.json").read_text())


def lines(key):
    return [TextLine(r["text"], 0.9, np.array(r["poly"], np.float32)) for r in FIXTURES[key]]


@pytest.mark.parametrize(
    "key,emirate,code,number",
    [
        ("dubai", "Dubai", "?", "35071"),  # tilted photo: code letter read as 0 -> ambiguous
        ("00", "Sharjah", None, "41623"),  # "SHARIAR"; the small "3" code was never read by OCR
        ("01", "Dubai", "E", "52904"),  # "DU2HT"
        ("02", "Abu Dhabi", "17", "63218"),  # red category box; "UAEAD"; Arabic read as "a"
        ("03", "Ajman", None, "70135"),  # export plate
        ("10", None, None, "24186"),  # "2024" year and Arabic read as "40": neither is a code
        ("11", "Ajman", None, "38452"),  # "AJVAN"
        ("12", "Abu Dhabi", None, "46931"),  # "FAD" = "A.D"; year ignored
        ("13", "Abu Dhabi", "14", "57319"),  # code box beside small print, much taller than it
    ],
)
def test_parse_real_ocr(key, emirate, code, number):
    r = uae.parse(lines(key))
    assert (r.fields["emirate"], r.fields["code"], r.fields["number"]) == (emirate, code, number)
    assert r.valid


def test_ambiguous_code_lists_candidates_and_votes_resolve_it():
    r = uae.parse(lines("dubai"))
    assert r.fields["key"] == "?35071" and r.fields["code_candidates"] == ["D", "O", "Q"]
    v = PlateVoter(min_votes=3)
    v.update(9, r.fields["key"])
    v.update(9, "?35071")
    assert v.update(9, "D35071") == "D35071"  # one clear frame settles the unreadable character


@pytest.mark.parametrize(
    "token,emirate",
    [
        ("DUBAI", "Dubai"),
        ("DU2HT", "Dubai"),
        ("SHARIAR", "Sharjah"),
        ("AJVAN", "Ajman"),
        ("UAEAD", "Abu Dhabi"),
        ("A.D", "Abu Dhabi"),
        ("FAD", "Abu Dhabi"),
        ("NISSAN", None),
        ("PEUGEOT", None),
        ("COM", None),
        ("2024", None),
    ],
)
def test_emirate_names(token, emirate):
    assert uae.emirate_of(token) == emirate


def test_no_number_is_invalid():
    r = uae.parse([TextLine("DUBAI", 0.9, np.array([[0, 0], [50, 0], [50, 10], [0, 10]], np.float32))])
    assert not r.valid and r.fields["emirate"] == "Dubai"


def _box(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)


def test_letter_emirate_ignores_a_lone_digit():
    # plate border read as "1" next to the real code "K" (seen on a rendered Dubai plate)
    r = uae.parse(
        [
            TextLine("35071", 0.9, _box(220, 60, 300, 70)),
            TextLine("DUBAI", 0.9, _box(40, 40, 120, 30)),
            TextLine("1", 0.9, _box(8, 80, 12, 60)),
            TextLine("K", 0.9, _box(60, 90, 40, 50)),
        ]
    )
    assert (r.fields["code"], r.fields["number"]) == ("K", "35071")


def test_zero_in_its_own_box_is_an_ambiguous_code():
    # paddle reads the Dubai "D" as a separate "0"
    r = uae.parse(
        [
            TextLine("35071", 0.9, _box(200, 50, 400, 120)),
            TextLine("0", 0.9, _box(100, 60, 70, 110)),
            TextLine("DUBAI", 0.9, _box(100, 200, 150, 40)),
        ]
    )
    assert r.fields["code"] == "?" and r.fields["code_candidates"] == ["D", "O", "Q"]


def test_zero_beside_the_emirate_name_is_still_the_code():
    # paddle: "0" box on the same row as "DUBAI" and about its size
    r = uae.parse(
        [
            TextLine("35071", 0.9, _box(300, 40, 500, 160)),
            TextLine("0", 0.9, _box(60, 90, 70, 95)),
            TextLine("DUBAI", 0.9, _box(150, 110, 140, 70)),
        ]
    )
    assert r.fields["code"] == "?" and r.fields["number"] == "35071"


def _centred(text, cx, cy, w, h):
    return TextLine(text, 0.9, _box(cx - w / 2, cy - h / 2, w, h))


def test_small_zero_code_on_a_tilted_dubai_plate():
    # paddle on a real tilted photo: code box a third of the number's height (heights and
    # centres as measured; widths estimated)
    r = uae.parse(
        [
            _centred("35071", 1329, 539, 1150, 808),
            _centred("0", 728, 572, 170, 268),
            _centred("DUBAI", 862, 864, 420, 273),
        ]
    )
    assert (r.fields["emirate"], r.fields["code"], r.fields["number"]) == ("Dubai", "?", "35071")
    assert r.fields["code_candidates"] == ["D", "O", "Q"]
