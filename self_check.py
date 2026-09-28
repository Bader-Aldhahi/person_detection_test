"""Small deterministic check for the monitor's important local logic."""

import math
import os
import tempfile
from pathlib import Path
from queue import Queue
from threading import Event
from unittest.mock import patch

import app

import numpy as np

from app import (
    LatestFrameCapture,
    append_alert,
    load_alerts,
    load_counts,
    load_roi,
    overlap_ratio,
    save_counts,
    save_roi,
    update_entry_count,
)


class FakeCapture:
    def __init__(self) -> None:
        self.frames: Queue[np.ndarray | None] = Queue()
        self.read_calls = 0
        self.waiting_for_fourth = Event()
        self.released = Event()
        self.release_calls = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_calls += 1
        if self.read_calls == 4:
            self.waiting_for_fourth.set()
        frame = self.frames.get()
        return frame is not None, frame

    def release(self) -> None:
        self.release_calls += 1
        self.released.set()
        self.frames.put(None)


def main() -> None:
    with patch.dict(os.environ, {}, clear=True), patch("sys.argv", ["app.py"]), patch.object(app, "DEFAULT_RTSP_URL", ""):
        assert app.main() == 1

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, :] = 1
    assert math.isclose(overlap_ratio((0, 0, 10, 10), mask), 0.10)
    assert overlap_ratio((0, 0, 10, 10), mask) >= 0.10

    mask[0, 9] = 0
    assert overlap_ratio((0, 0, 10, 10), mask) < 0.10
    assert overlap_ratio((5, 5, 5, 8), mask) == 0.0

    fake = FakeCapture()
    for value in (1, 2, 3):
        fake.frames.put(np.full((1, 1), value, dtype=np.uint8))
    latest_capture = LatestFrameCapture(fake)  # type: ignore[arg-type]
    assert fake.waiting_for_fourth.wait(timeout=1.0)
    ok, newest = latest_capture.read()
    assert ok and newest is not None and newest.item() == 3
    fake.frames.put(None)
    assert fake.released.wait(timeout=1.0)
    assert latest_capture.read() == (False, None)
    latest_capture.release()
    latest_capture.release()
    assert fake.release_calls == 1

    tracked_inside: dict[int, float] = {}
    total, live = update_entry_count([{"track_id": 7}], tracked_inside, 0, 1.0)
    assert (total, live) == (1, 1)
    total, live = update_entry_count([{"track_id": 7}], tracked_inside, total, 2.0)
    assert (total, live) == (1, 1)
    total, live = update_entry_count([{"track_id": 7}, {"track_id": 8}], tracked_inside, total, 3.0)
    assert (total, live) == (2, 2)
    total, live = update_entry_count([], tracked_inside, total, 6.1)
    assert (total, live) == (2, 0)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        alert_path = root / "alerts.jsonl"
        append_alert(alert_path, {"id": "first"})
        append_alert(alert_path, {"id": "second"})
        assert [alert["id"] for alert in load_alerts(alert_path)] == ["first", "second"]
        with alert_path.open("a", encoding="utf-8") as handle:
            handle.write("{interrupted")
        append_alert(alert_path, {"id": "after-crash"})
        assert [alert["id"] for alert in load_alerts(alert_path)] == ["first", "second", "after-crash"]

        roi_path = root / "roi.txt"
        points = [(0, 0), (99, 0), (99, 49), (0, 49)]
        save_roi(roi_path, points, 100, 50)
        assert load_roi(roi_path, 200, 100) == [(0, 0), (199, 0), (199, 99), (0, 99)]
        roi_path.write_text("[]", encoding="utf-8")
        try:
            load_roi(roi_path, 200, 100)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed ROI data was accepted")

        counts_path = root / "counts.json"
        save_counts(counts_path, total, live)
        assert load_counts(counts_path) == (2, 0)

    print("Self-check passed")


if __name__ == "__main__":
    main()
