"""Run with python test_check_camera.py; no camera or Linux host required."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import check_camera


code, output = check_camera.run_command([sys.executable, "-c", "print('ok')"])
assert code == 0 and output.strip() == "ok"
code, output = check_camera.run_command([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"], timeout=1)
assert code == 124 and "started" in output and "Timed out" in output
with patch.object(subprocess, "run", side_effect=FileNotFoundError):
    assert check_camera.run_command(["missing-tool"])[0] == 127

camera = Mock()
camera.read.return_value = True, SimpleNamespace(shape=(480, 640, 3))
cv2 = Mock()
cv2.__version__ = "test"
cv2.VideoCapture.return_value = camera
with patch.dict("sys.modules", {"cv2": cv2}):
    assert check_camera.probe_camera(4, "v4l2") == 0
    camera.release.assert_called_once()
    camera.reset_mock()
    camera.isOpened.return_value = False
    assert check_camera.probe_camera(4, "auto") == 1
    camera.release.assert_called_once()
    camera.reset_mock()
    camera.isOpened.return_value = True
    camera.read.side_effect = RuntimeError("read error")
    try:
        check_camera.probe_camera(4, "auto")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Read error was hidden")
    camera.release.assert_called_once()

print("Camera diagnostic self-check passed")
