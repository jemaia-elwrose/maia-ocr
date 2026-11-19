"""Real OCR (RapidOCR, PP-OCRv6 small, bundled models) on rendered text."""

import json
import socket
import threading
import time

from conftest import render
from maia_infer import Broker
from maia_infer.protocol import Request

from maia_ocr import containers, full_text, plates
from maia_ocr.workers import OCRWorker


def test_read_finds_lines_in_reading_order(rapid):
    lines = rapid.read(render("MAIA OCR", "PP-OCRv6 2026"))
    assert full_text(lines).replace(" ", "").upper().startswith("MAIAOCR")
    assert "2026" in full_text(lines)


def test_recognize_batch_of_crops(rapid):
    out = rapid.recognize([render("AB12CDE"), render("CSQU3054383")])
    assert [t.replace(" ", "") for t, _ in out] == ["AB12CDE", "CSQU3054383"]
    assert all(s > 0.8 for _, s in out)


def test_two_row_plate_end_to_end(rapid):
    lines = rapid.read(render("DL 3C", "1234"))
    assert plates.correct(plates.merge_rows(lines), "india").text == "DL3C1234"


def test_container_code_split_over_lines(rapid):
    assert containers.find(rapid.read(render("CSQU 305438", "3 45G1"))) == ["CSQU3054383"]


def test_worker_mixes_read_and_recognize_in_one_batch():
    w = OCRWorker()
    w.setup()
    out = w.predict_batch(
        [
            Request(arrays=[render("HELLO", "WORLD")], meta={"mode": "read"}),
            Request(arrays=[render("AB12CDE")], meta={"mode": "recognize"}),
            Request(arrays=[render("XY99ZZZ")], meta={"mode": "recognize"}),
        ]
    )
    assert [len(o.meta["texts"]) for o in out] == [2, 1, 1]
    assert out[1].meta["texts"] == ["AB12CDE"] and out[2].meta["texts"] == ["XY99ZZZ"]
    assert out[1].arrays[0].shape == (1, 4, 2) and out[0].meta["backend"].startswith("rapid")


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_service_end_to_end_plate():
    from maia_ocr.client import OCRClient

    front, back = f"tcp://127.0.0.1:{_port()}", f"tcp://127.0.0.1:{_port()}"
    broker, worker = Broker(front, back), OCRWorker(broker=back, heartbeat_interval=0.2)
    threading.Thread(target=broker.run, daemon=True).start()
    threading.Thread(target=worker.run, daemon=True).start()
    try:
        end = time.time() + 20
        while time.time() < end and "ocr" not in broker.stats()["services"]:
            time.sleep(0.05)
        ocr = OCRClient(front)
        assert ocr.read_plate(render("MH12AB1234"), "india").text == "MH12AB1234"
        text, score = ocr.recognize(render("AB12CDE"))
        assert text == "AB12CDE" and score > 0.8
    finally:
        worker.stop()
        broker.stop()
        time.sleep(0.3)


def test_cli(tmp_path, capsys):
    import cv2

    from maia_ocr.cli import main

    paths = {n: str(tmp_path / f"{n}.png") for n in ("sign", "plate", "box")}
    cv2.imwrite(paths["sign"], render("EXIT 3", "GATE B"))
    cv2.imwrite(paths["plate"], render("A 12345"))
    cv2.imwrite(paths["box"], render("CSQU 305438 3"))

    main(["read", paths["sign"], "--save", str(tmp_path / "out.png")])
    assert json.loads(capsys.readouterr().out)["text"].replace(" ", "") == "EXIT3\nGATEB"
    assert (tmp_path / "out.png").exists()
    main(["plate", paths["plate"], "--format", "uae"])
    assert json.loads(capsys.readouterr().out)["text"] == "A12345"
    main(["container", paths["box"]])
    assert json.loads(capsys.readouterr().out)["containers"] == ["CSQU3054383"]
