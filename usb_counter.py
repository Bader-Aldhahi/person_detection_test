"""Count people from a USB camera, without internet, ROI, or windows."""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

from app import LatestFrameCapture, append_alert, cv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="USB color camera index")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("yolo11m.pt"))
    parser.add_argument("--log-file", type=Path, default=Path("camera_counts.jsonl"))
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("--camera must be zero or greater")
    if not args.model.is_file():
        parser.error("model file is missing; copy yolo11m.pt here before running offline")
    if args.model.resolve() == args.log_file.resolve():
        parser.error("model and log file must be different")

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not camera.isOpened():
        camera.release()
        print("Could not open camera. Check USB access and --camera index.", flush=True)
        return 1
    camera = LatestFrameCapture(camera)
    seen_ids = set()
    next_log = 0.0
    print(f"Camera {args.camera} started. Counts reset for this run. Ctrl+C stops.", flush=True)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera stopped returning frames. Check the USB connection.", flush=True)
                return 1
            boxes = model.track(frame, classes=[0], conf=0.4, persist=True,
                                tracker="bytetrack.yaml", verbose=False, show=False)[0].boxes
            live = len(boxes) if boxes is not None else 0
            # ponytail: IDs approximate people; re-identification is needed to deduplicate lost tracks.
            if boxes is not None and boxes.id is not None:
                seen_ids.update(boxes.id.int().cpu().tolist())
            now = time.monotonic()
            if now >= next_log:
                append_alert(args.log_file, {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "live_count": live,
                    "total_count": len(seen_ids),
                })
                next_log = now + 2
    except KeyboardInterrupt:
        print(f"\nStopped. Total tracked people: {len(seen_ids)}", flush=True)
    except (OSError, RuntimeError, cv2.error) as error:
        print(f"Error: {error}", flush=True)
        return 1
    finally:
        camera.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
