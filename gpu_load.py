"""Multiple resident YOLO models on a simulated fixed-FPS FIFO image stream.

Examples: python gpu_load.py --copies 4
          python gpu_load.py --models yolo11n.pt yolo11m.pt
          python gpu_load.py --self-test
"""

import argparse
import math
import os
import time
from pathlib import Path


def queued_frames(processed, elapsed, fps, total):
    # ponytail: identical images need only an arrival count, not stored copies.
    # A live camera needs a real frame queue; this simulates fixed-rate arrivals only.
    arrived = min(total, int(max(0, elapsed) * fps) + 1)
    return max(0, arrived - processed)


def self_test():
    assert queued_frames(0, 0, 10, 10) == 1
    assert queued_frames(1, 0.05, 10, 10) == 0
    assert queued_frames(1, 0.35, 10, 10) == 3
    elapsed = 0
    backlogs = []
    for slot in range(10):
        elapsed = max(elapsed, slot / 10) + 0.25  # slower than arrivals
        backlogs.append(queued_frames(slot + 1, elapsed, 10, 10))
    assert backlogs[0] == 2
    assert max(backlogs) > 2
    assert backlogs[-1] == 0  # drains after the last arrival, without dropping
    assert queued_frames(10, 20, 10, 10) == 0
    print("PASS: FIFO backlog grows under load and drains after all frames finish")


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", type=Path, default=[folder / "yolo11m.pt"])
    parser.add_argument("--copies", type=int, help="Copies of each model (default: 2 for one model, otherwise 1)")
    parser.add_argument("--image", type=Path, default=folder / "images.jpg")
    parser.add_argument("--frames", type=int, default=1000, help="Total frames to process in order")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    copies = args.copies if args.copies is not None else (2 if len(args.models) == 1 else 1)
    if copies < 1 or args.frames < 1 or args.imgsz < 32 or args.device < 0 or not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("copies/frames/FPS must be positive, imgsz >= 32, device >= 0")
    for path in [*args.models, args.image]:
        if not path.is_file():
            parser.error(f"Missing local file: {path}")

    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import cv2
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise SystemExit("Requested CUDA GPU unavailable. Run python check_gpu.py")
    torch.cuda.set_device(args.device)
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Cannot decode image: {args.image}")
    options = dict(device=args.device, imgsz=args.imgsz, classes=[0], verbose=False, save=False, show=False)
    models = []
    processed = 0
    total_ms = 0.0
    print(f"GPU: {torch.cuda.get_device_name(args.device)} | Models: {len(args.models) * copies} | "
          f"Simulated FPS: {args.fps:g} | Arrivals: {args.frames}", flush=True)
    print("FIFO: every frame runs through all models; backlog grows when processing falls behind. Ctrl+C stops.", flush=True)
    try:
        for path in args.models:
            for copy in range(copies):
                model = YOLO(str(path)).to(f"cuda:{args.device}")
                for _ in range(3):
                    model.predict(image, **options)
                models.append((f"{path.name}#{copy + 1}", model))
        torch.cuda.synchronize(args.device)
        print("Warm-up complete; starting measurements.", flush=True)
        start = time.perf_counter()
        for slot in range(args.frames):
            arrival = start + slot / args.fps
            delay = arrival - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            torch.cuda.synchronize(args.device)
            begin = time.perf_counter()
            wait_ms = max(0, begin - arrival) * 1000
            timings = []
            for label, model in models:
                result = model.predict(image, **options)[0]
                timings.append(f"{label}: {result.speed['inference']:.1f} ms")
                del result
            torch.cuda.synchronize(args.device)
            end = time.perf_counter()
            duration = (end - begin) * 1000
            processed += 1
            total_ms += duration
            queued = queued_frames(processed, end - start, args.fps, args.frames)
            free, total = torch.cuda.mem_get_info(args.device)
            used = total - free
            reserved = torch.cuda.memory_reserved(args.device) / 2**20
            print(f"Frame {slot + 1}/{args.frames} | {' | '.join(timings)} | "
                  f"Processing: {duration:.1f} ms | Avg: {total_ms / processed:.1f} ms | "
                  f"VRAM device: {used / 2**20:.0f}/{total / 2**20:.0f} MiB ({used / total:.1%}) | "
                  f"PyTorch reserved: {reserved:.0f} MiB | Processed: {processed} | "
                  f"Queued: {queued} | Queue wait: {wait_ms:.1f} ms", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    except torch.cuda.OutOfMemoryError:
        raise SystemExit("GPU out of memory: reduce --copies or --imgsz. Test stopped with unfinished frames.")
    finally:
        average = f"{total_ms / processed:.1f} ms" if processed else "N/A"
        print(f"SUMMARY | Processed: {processed} | "
              f"Unfinished/not reached: {args.frames - processed} | Avg processing: {average}", flush=True)


if __name__ == "__main__":
    main()
