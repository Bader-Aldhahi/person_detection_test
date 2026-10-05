"""Diagnose a Linux USB color camera. Run: python check_camera.py (default: camera 4)."""

import argparse
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def run_command(command, timeout=15):
    """Isolate native camera calls so a stuck capture cannot block later checks."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout)
        return result.returncode, result.stdout + result.stderr
    except FileNotFoundError:
        return 127, f"Tool not installed: {command[0]}"
    except subprocess.TimeoutExpired as error:
        output = "".join(part.decode(errors="replace") if isinstance(part, bytes) else part or ""
                         for part in (error.stdout, error.stderr))
        return 124, output + f"\nTimed out after {timeout} seconds; test process killed."
    except OSError as error:
        return 126, str(error)


def probe_camera(index, mode):
    import cv2

    print(f"OpenCV {cv2.__version__}; opening camera {index} using {mode}", flush=True)
    camera = cv2.VideoCapture(index, cv2.CAP_ANY if mode == "auto" else cv2.CAP_V4L2)
    try:
        if not camera.isOpened():
            print("OPEN_FAILED", flush=True)
            return 1
        if mode == "v4l2":
            for prop, value in ((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV")),
                                (cv2.CAP_PROP_FRAME_WIDTH, 640), (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                                (cv2.CAP_PROP_FPS, 15)):
                if not camera.set(prop, value):
                    print(f"Driver rejected setting {prop}={value}", flush=True)
        ok, frame = camera.read()
        if not ok or frame is None:
            print("READ_FAILED", flush=True)
            return 1
        print(f"FRAME_OK: shape={frame.shape}; backend={camera.getBackendName()}", flush=True)
        return 0
    finally:
        camera.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=4)
    parser.add_argument("--probe", choices=("auto", "v4l2"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("camera index must be zero or greater")
    if args.probe:
        return probe_camera(args.camera, args.probe)
    if sys.platform != "linux":
        print("Run this diagnostic inside the Linux server VM.")
        return 1

    device = f"/dev/video{args.camera}"
    with Path("check_camera.log").open("a", encoding="utf-8") as log:
        def report(message):
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        def check(title, command):
            report(f"\n--- {title} ---")
            code, output = run_command(command)
            log.write(f"Command: {command!r}\n{output}\nExit code: {code}\n")
            log.flush()
            print("\n".join(output.splitlines()[-12:]) or "(no output)", flush=True)
            print(f"Exit code: {code}", flush=True)
            return code, output

        report(f"\nCamera check: {datetime.now().astimezone().isoformat(timespec='seconds')}")
        report(f"Kernel: {platform.release()}; Python: {sys.executable}")
        report("Stop other camera scripts first. Capture tests may take up to 45 seconds.")
        report("Tests request 640x480 YUYV at 15 FPS; no drivers or ZEDEDA settings are changed.")
        check("USB devices", ["lsusb"])
        check("USB drivers and connection speed", ["lsusb", "-t"])
        report(f"uvcvideo loaded/built-in: {Path('/sys/module/uvcvideo').exists()}")
        report("Video devices: " + (", ".join(str(p) for p in sorted(Path('/dev').glob('video*'))) or "none"))
        if not Path(device).exists():
            report(f"RESULT: {device} is missing. Check USB assignment, driver, or device numbering.")
            return 1
        check("Device capabilities and format", ["v4l2-ctl", "-d", device, "--all"])
        check("Supported formats", ["v4l2-ctl", "-d", device, "--list-formats"])
        check("Processes using this camera (optional tool)", ["fuser", "-v", device])
        probe = [sys.executable, str(Path(__file__).resolve()), "--camera", str(args.camera), "--probe"]
        auto_code, _ = check("OpenCV automatic backend (current script)", probe + ["auto"])
        v4l_code, _ = check("OpenCV explicit V4L2 with color settings", probe + ["v4l2"])
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "frames.raw"
            stream_code, _ = check("Direct driver capture: five frames", [
                "v4l2-ctl", "-d", device,
                "--set-fmt-video=width=640,height=480,pixelformat=YUYV", "--set-parm=15",
                "--stream-mmap", "--stream-count=5", f"--stream-to={raw}",
            ])
            captured_bytes = raw.stat().st_size if raw.exists() else 0
            report(f"Direct capture bytes: {captured_bytes}")

        if auto_code == 0:
            report(f"RESULT: OpenCV read a frame. Try python usb_counter.py --camera {args.camera}")
        elif v4l_code == 0:
            report("RESULT: Explicit V4L2 capture works; automatic capture fails. Update the script's camera setup.")
        elif stream_code == 0 and captured_bytes > 0:
            report("RESULT: Direct driver capture works, but OpenCV fails. Investigate OpenCV capture compatibility.")
        else:
            report("RESULT: No capture test succeeded. The exact cause is not yet established; inspect the capture errors.")
        check("Recent kernel messages", ["dmesg", "--color=never"])
        report("Full report saved to check_camera.log. Share this file or the RESULT and capture errors.")
    return 0 if auto_code == 0 or v4l_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
