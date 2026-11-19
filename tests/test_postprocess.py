import numpy as np
import pytest

from maia_ocr import PlateVoter, TextLine, containers, full_text, plates, reading_order


@pytest.mark.parametrize(
    "raw,fmt,expected,fixes",
    [
        ("MH 12 AB 1234", "india", "MH12AB1234", 0),
        ("MH12AB1Z34", "india", "MH12AB1234", 1),  # Z read for 2 in a digit position
        ("0L 7 C 4321", "india", "OL7C4321", 1),  # 0 read for O in a letter position
        ("A 12345", "uae", "A12345", 0),
        ("12345", "uae", "12345", 0),
        ("8X12 CDE", "uk", "BX12CDE", 1),
        ("M-AB 1234", "eu-de", "MAB1234", 0),
    ],
)
def test_correct_known_formats(raw, fmt, expected, fixes):
    r = plates.correct(raw, fmt)
    assert r.valid and r.text == expected and r.corrections == fixes


def test_correct_rejects_what_cannot_fit():
    r = plates.correct("ABCDEFG", "uae")
    assert not r.valid and r.text == "ABCDEFG"
    assert not plates.correct("88888", "uk").valid  # wrong length for the format


def test_custom_template_and_correction_limit():
    assert plates.correct("A8C 123", ["LLL DDD"]).text == "ABC123"
    assert not plates.correct("8B8 ABC", ["LLL DDD"], max_corrections=2).valid  # would need 4 changes


def test_expand_templates():
    assert plates.expand("LL D{1,2}") == ["LLD", "LLDD"]
    assert plates.expand("L{0,1} DD") == ["DD", "LDD"]


def test_plate_voter_fixes_single_frame_misreads():
    v = PlateVoter(min_votes=3)
    assert v.update(1, "AB12CDE", 0.9) is None  # not enough evidence yet
    v.update(1, "A812CDE", 0.6)  # one bad frame
    v.update(1, "AB12C0E", 0.7)  # another bad frame, different character
    assert v.update(1, "AB12CDE", 0.95) == "AB12CDE"
    v.update(1, "AB12CDEX", 0.99)  # odd length: outvoted
    assert v.decision(1) == "AB12CDE"


def test_container_check_digit():
    assert containers.is_valid("CSQU3054383")
    assert not containers.is_valid("CSQU3054384")
    assert containers.check_digit("CSQU305438") == 3


def test_container_correct_and_find():
    assert containers.correct("C5QU 3O5438 3") == "CSQU3054383"  # S/5 and O/0 confusions undone
    assert containers.correct("CSQU3054384") is None  # check digit still wrong: refuse
    assert containers.find("gate 7: CSQU 305438 3 load") == ["CSQU3054383"]


def _line(text, x, y, w=60, h=20):
    return TextLine(text, 0.9, np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32))


def test_reading_order_and_rows():
    lines = [_line("world", 100, 12), _line("second", 0, 60), _line("hello", 0, 10)]
    assert [ln.text for ln in reading_order(lines)] == ["hello", "world", "second"]
    assert full_text(lines) == "hello world\nsecond"


def test_two_row_plate_merges_top_row_first():
    lines = [_line("1234", 0, 40), _line("DL", 0, 0, w=30), _line("3C", 40, 2, w=30)]
    assert plates.merge_rows(lines) == "DL3C1234"
