"""Headless RTSP person-in-ROI monitor with file and terminal alerts."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Condition, Thread
from typing import Any

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer")

import cv2
import numpy as np


CLEAR_AFTER_SECONDS = 2.0
DEFAULT_RTSP_URL = ""


class LatestFrameCapture:
    """Continuously drain a live stream and keep only its newest frame."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self._capture = capture
        self._condition = Condition()
        self._latest: np.ndarray | None = None
        self._failed = False
        self._stopped = False
        self._thread = Thread(target=self._read_forever, name="rtsp-reader", daemon=True)
        self._thread.start()

    def _read_forever(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._stopped:
                        return
                ok, frame = self._capture.read()
                with self._condition:
                    if self._stopped:
                        return
                    if not ok or frame is None:
                        self._failed = True
                        self._condition.notify_all()
                        return
                    self._latest = frame
                    self._condition.notify_all()
        finally:
            self._capture.release()

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._latest is not None or self._failed or self._stopped,
                timeout=5.5,
            )
            if self._failed or self._stopped or self._latest is None:
                return False, None
            frame, self._latest = self._latest, None
            return True, frame

    def release(self) -> None:
        with self._condition:
            if self._stopped:
                return
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=6.0)


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Shrink large frames while preserving their aspect ratio."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA)


def make_roi_mask(frame_shape: tuple[int, ...], points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 1)
    return mask


def overlap_ratio(box: Any, roi_mask: np.ndarray) -> float:
    """Return the fraction of a [x1, y1, x2, y2) box covered by the ROI."""
    x1, y1, x2, y2 = (float(value) for value in box)
    height, width = roi_mask.shape[:2]
    left = max(0, min(width, math.floor(x1)))
    top = max(0, min(height, math.floor(y1)))
    right = max(0, min(width, math.ceil(x2)))
    bottom = max(0, min(height, math.ceil(y2)))
    area = (right - left) * (bottom - top)
    if area <= 0:
        return 0.0
    return float(np.count_nonzero(roi_mask[top:bottom, left:right])) / area


def save_roi(path: Path, points: list[tuple[int, int]], width: int, height: int) -> None:
    """Atomically save normalized polygon points so resolution changes are safe."""
    if len(set(points)) < 3 or width < 2 or height < 2:
        raise ValueError("An ROI needs at least three unique points")
    payload = {
        "version": 1,
        "frame_size": [width, height],
        "points": [[x / (width - 1), y / (height - 1)] for x, y in points],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def load_roi(path: Path, width: int, height: int) -> list[tuple[int, int]] | None:
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ROI file must contain a JSON object")
    if payload.get("version") != 1:
        raise ValueError("unsupported ROI file version")
    saved_width, saved_height = payload["frame_size"]
    if saved_width < 1 or saved_height < 1:
        raise ValueError("invalid saved frame size")
    saved_aspect = saved_width / saved_height
    current_aspect = width / height
    if abs(saved_aspect - current_aspect) / saved_aspect > 0.02:
        raise ValueError("camera aspect ratio changed")

    normalized_points = payload["points"]
    if len(normalized_points) < 3:
        raise ValueError("ROI needs at least three points")

    points: list[tuple[int, int]] = []
    for normalized_x, normalized_y in normalized_points:
        if not all(math.isfinite(value) for value in (normalized_x, normalized_y)):
            raise ValueError("ROI contains a non-finite coordinate")
        if not (0.0 <= normalized_x <= 1.0 and 0.0 <= normalized_y <= 1.0):
            raise ValueError("ROI coordinate is outside the frame")
        points.append(
            (
                round(normalized_x * (width - 1)),
                round(normalized_y * (height - 1)),
            )
        )
    if len(set(points)) < 3:
        raise ValueError("ROI needs at least three unique points")
    return points


def append_alert(path: Path, alert: dict[str, Any]) -> None:
    """Save one JSON alert, then print the same event to the terminal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(alert, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    with path.open("a+b") as handle:
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    print(line.decode("utf-8"), end="", flush=True)


def load_alerts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    alerts: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            alert = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(alert, dict):
            alerts.append(alert)
    return alerts


def load_counts(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("counts file must contain a JSON object")
    values = payload.get("total_count"), payload.get("live_count")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("counts must be non-negative integers")
    return values


def save_counts(path: Path, total_count: int, live_count: int) -> None:
    payload = {
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_count": total_count,
        "live_count": live_count,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def update_entry_count(
    inside_detections: list[dict[str, Any]],
    tracked_inside: dict[int, float],
    total_count: int,
    now: float,
) -> tuple[int, int]:
    current_ids = {
        detection["track_id"]
        for detection in inside_detections
        if detection.get("track_id") is not None
    }
    total_count += len(current_ids - tracked_inside.keys())
    tracked_inside.update({track_id: now for track_id in current_ids})
    for track_id, last_seen in list(tracked_inside.items()):
        if track_id not in current_ids and now - last_seen >= CLEAR_AFTER_SECONDS:
            del tracked_inside[track_id]
    return total_count, len(inside_detections)


def open_capture(source: str | int) -> cv2.VideoCapture:
    parameters: list[int] = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        parameters += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5_000]
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        parameters += [cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000]

    if isinstance(source, int):
        capture = cv2.VideoCapture(source)
    else:
        try:
            capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, parameters)
        except (TypeError, cv2.error):
            capture = cv2.VideoCapture(source)
    return capture


def connect_and_read(source: str | int) -> tuple[LatestFrameCapture | cv2.VideoCapture, np.ndarray]:
    capture = open_capture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Could not open the video source")
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError("Could not read from the video source")
    is_live = isinstance(source, str) and source.lower().startswith("rtsp://")
    return (LatestFrameCapture(capture) if is_live else capture), frame


def reconnect(source: str | int) -> tuple[LatestFrameCapture | cv2.VideoCapture, np.ndarray]:
    print("Stream lost - reconnecting. Press Ctrl+C to stop.", flush=True)
    while True:
        time.sleep(1.0)
        try:
            return connect_and_read(source)
        except RuntimeError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="RTSP URL, video path, or camera number (or set RTSP_URL)")
    parser.add_argument("--model", default="yolo11m.pt", help="Ultralytics model path/name")
    parser.add_argument("--confidence", type=float, default=0.40, help="YOLO confidence threshold")
    parser.add_argument("--overlap", type=float, default=0.10, help="Minimum person-box area inside ROI")
    parser.add_argument("--width", type=int, default=960, help="Maximum processing width")
    parser.add_argument("--roi-file", type=Path, default=Path("roi.txt"), help="Required saved ROI polygon")
    parser.add_argument("--alerts-file", type=Path, default=Path("alerts.jsonl"))
    parser.add_argument("--counts-file", type=Path, default=Path("counts.json"))
    parser.add_argument("--device", help="Ultralytics device, for example 0 or cpu")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    if not 0.0 <= args.overlap <= 1.0:
        parser.error("--overlap must be between 0 and 1")
    if args.width < 320:
        parser.error("--width must be at least 320")
    return args


def main() -> int:
    args = parse_args()
    source_text = args.source or os.environ.get("RTSP_URL") or DEFAULT_RTSP_URL
    if not source_text:
        print("Error: provide --source, set RTSP_URL, or configure DEFAULT_RTSP_URL.")
        return 1
    source: str | int = int(source_text) if source_text.isdecimal() else source_text

    try:
        capture, raw_frame = connect_and_read(source)
    except RuntimeError as error:
        print(f"Error: {error}. Check the URL, camera, and network.")
        return 1
    try:
        frame = resize_frame(raw_frame, args.width)
        height, width = frame.shape[:2]
        try:
            roi_points = load_roi(args.roi_file, width, height)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"Error: invalid ROI file {args.roi_file}: {error}")
            return 1
        if roi_points is None or np.count_nonzero(make_roi_mask(frame.shape, roi_points)) < 10:
            print(f"Error: provide a valid saved ROI polygon with --roi-file (current: {args.roi_file}).")
            return 1
        roi_mask = make_roi_mask(frame.shape, roi_points)
        print(f"Loading {args.model}...", flush=True)

        try:
            from ultralytics import YOLO

            model = YOLO(args.model)
        except Exception as error:
            print(f"Error loading {args.model}: {error}")
            return 1

        try:
            total_people, _ = load_counts(args.counts_file)
            save_counts(args.counts_file, total_people, 0)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"Error: counts could not be loaded or saved at {args.counts_file}: {error}")
            return 1

        occupied = False
        last_inside_at = 0.0
        tracked_inside: dict[int, float] = {}
        last_saved_counts = (total_people, 0)
        print(f"Monitoring ROI. Alerts: {args.alerts_file}. Press Ctrl+C to stop.", flush=True)

        try:
            while True:
                frame = resize_frame(raw_frame, args.width)
                if frame.shape[:2] != roi_mask.shape:
                    try:
                        roi_points = load_roi(args.roi_file, frame.shape[1], frame.shape[0])
                    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                        print(f"ROI no longer matches the stream: {error}")
                        return 1
                    if roi_points is None:
                        print("ROI file disappeared; stopping rather than monitoring without an ROI.")
                        return 1
                    roi_mask = make_roi_mask(frame.shape, roi_points)

                predict_options: dict[str, Any] = {
                    "source": frame,
                    "classes": [0],
                    "conf": args.confidence,
                    "verbose": False,
                    "show": False,
                }
                if args.device:
                    predict_options["device"] = args.device
                result = model.track(
                    **predict_options,
                    persist=True,
                    tracker="bytetrack.yaml",
                )[0]

                detections: list[dict[str, Any]] = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    track_ids = (
                        result.boxes.id.int().cpu().tolist()
                        if result.boxes.id is not None
                        else [None] * len(boxes)
                    )
                    for box, confidence, track_id in zip(boxes, confidences, track_ids):
                        ratio = overlap_ratio(box, roi_mask)
                        detections.append(
                            {
                                "box": box,
                                "confidence": float(confidence),
                                "track_id": track_id,
                                "overlap": ratio,
                                "inside": ratio >= args.overlap,
                            }
                        )

                inside_detections = [detection for detection in detections if detection["inside"]]
                inside_now = bool(inside_detections)
                now = time.monotonic()
                total_people, live_people = update_entry_count(
                    inside_detections,
                    tracked_inside,
                    total_people,
                    now,
                )
                if (total_people, live_people) != last_saved_counts:
                    try:
                        save_counts(args.counts_file, total_people, live_people)
                    except OSError as error:
                        print(f"Fatal: counts could not be saved to {args.counts_file}: {error}")
                        return 1
                    last_saved_counts = total_people, live_people

                if inside_now:
                    last_inside_at = now
                    if not occupied:
                        strongest = max(inside_detections, key=lambda detection: detection["overlap"])
                        alert = {
                            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                            "event": "person_in_roi",
                            "confidence": round(strongest["confidence"], 4),
                            "overlap": round(strongest["overlap"], 4),
                            "bbox": [round(float(value), 1) for value in strongest["box"]],
                            "person_count": len(inside_detections),
                            "live_count": live_people,
                            "total_count": total_people,
                        }
                        try:
                            append_alert(args.alerts_file, alert)
                        except OSError as error:
                            print(f"Fatal: alert could not be saved to {args.alerts_file}: {error}")
                            return 1
                        occupied = True
                elif occupied and now - last_inside_at >= CLEAR_AFTER_SECONDS:
                    occupied = False

                ok, raw_frame = capture.read()
                if not ok or raw_frame is None:
                    capture.release()
                    capture, raw_frame = reconnect(source)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.", flush=True)
        finally:
            try:
                save_counts(args.counts_file, total_people, 0)
            except OSError as error:
                print(f"Warning: final live count could not be cleared: {error}")
    finally:
        capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
