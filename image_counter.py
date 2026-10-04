"""Run person detection on images.jpg 1,000 times with two seconds between runs."""

import os
import time
from pathlib import Path

os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

import cv2


def main():
    folder = Path(__file__).resolve().parent
    model_path = folder / "yolo11m.pt"
    if not model_path.is_file():
        raise SystemExit(f"Missing model: {model_path}")
    image = cv2.imread(str(folder / "images.jpg"))
    if image is None:
        raise SystemExit(f"Could not read {folder / 'images.jpg'}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    for counter in range(1, 1001):
        result = model.predict(image, classes=[0], conf=0.4, verbose=False, show=False)[0]
        people = len(result.boxes)
        average_confidence = f"{result.boxes.conf.mean().item():.1%}" if people else "N/A"
        print(
            f"Run {counter}/1000 | Inference: {result.speed['inference']:.1f} ms | "
            f"Average confidence: {average_confidence} | People: {people}",
            flush=True,
        )
        if counter < 1000:
            time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
