"""Multiple resident YOLO models on a simulated fixed-FPS image stream.

Examples: python gpu_load.py --copies 4
          python gpu_load.py --models yolo11n.pt yolo11m.pt
          python gpu_load.py --self-test
"""

import argparse
import math
import os
import time
from pathlib import Path


def latest_slot(next_slot, elapsed, fps, total):
    # ponytail: repeated-image arrivals, not real camera losses; use a capture producer for live tests.
    return min(total, max(next_slot, int(elapsed * fps)))


def self_test():
    assert latest_slot(0, 0, 10, 100) == 0
    assert latest_slot(1, 0.05, 10, 100) == 1
    assert latest_slot(1, 0.35, 10, 100) == 3  # slots 1 and 2 skipped
    assert latest_slot(4, 0.4, 10, 100) == 4
    assert latest_slot(4, 20, 10, 100) == 100
    next_slot = processed = skipped = 0
    for elapsed in (0, 0.35, 0.6, 2):
        slot = latest_slot(next_slot, elapsed, 10, 10)
        skipped += slot - next_slot
        next_slot = slot
        if slot < 10:
            processed += 1
            next_slot += 1
        assert processed + skipped == next_slot
    assert processed + skipped == 10
    print("PASS: frame scheduling, skipped counter, and final accounting")


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", type=Path, default=[folder / "yolo11m.pt"])
    parser.add_argument("--copies", type=int, help="Copies of each model (default: 2 for one model, otherwise 1)")
    parser.add_argument("--image", type=Path, default=folder / "images.jpg")
    parser.add_argument("--frames", type=int, default=1000, help="Total simulated arriving frames, including skips")
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
    processed = skipped = next_slot = 0
    total_ms = 0.0
    print(f"GPU: {torch.cuda.get_device_name(args.device)} | Models: {len(args.models) * copies} | "
          f"Simulated FPS: {args.fps:g} | Arrivals: {args.frames}", flush=True)
    print("Sequential models per image; skips include processing and reporting delays. Ctrl+C stops.", flush=True)
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
        while next_slot < args.frames:
            delay = start + next_slot / args.fps - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            slot = latest_slot(next_slot, time.perf_counter() - start, args.fps, args.frames)
            skipped += slot - next_slot
            next_slot = slot
            if slot == args.frames:
                break
            torch.cuda.synchronize(args.device)
            begin = time.perf_counter()
            timings = []
            for label, model in models:
                result = model.predict(image, **options)[0]
                timings.append(f"{label}: {result.speed['inference']:.1f} ms")
                del result
            torch.cuda.synchronize(args.device)
            duration = (time.perf_counter() - begin) * 1000
            processed += 1
            next_slot += 1
            total_ms += duration
            free, total = torch.cuda.mem_get_info(args.device)
            used = total - free
            reserved = torch.cuda.memory_reserved(args.device) / 2**20
            print(f"Frame {slot + 1}/{args.frames} | {' | '.join(timings)} | "
                  f"Processing: {duration:.1f} ms | Avg: {total_ms / processed:.1f} ms | "
                  f"VRAM device: {used / 2**20:.0f}/{total / 2**20:.0f} MiB ({used / total:.1%}) | "
                  f"PyTorch reserved: {reserved:.0f} MiB | Processed: {processed} | Skipped: {skipped}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    except torch.cuda.OutOfMemoryError:
        raise SystemExit("GPU out of memory: reduce --copies or --imgsz. Test stopped; OOM is not counted as a skipped arrival.")
    finally:
        average = f"{total_ms / processed:.1f} ms" if processed else "N/A"
        print(f"SUMMARY | Processed: {processed} | Skipped: {skipped} | "
              f"Unfinished/not reached: {args.frames - processed - skipped} | Avg processing: {average}", flush=True)


if __name__ == "__main__":
    main()
