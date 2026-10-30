"""A real camera, live: its image with the ArUco markers it sees (IDs 1-4: the wall's corners, 5: the robot), and the
camera's pose from the corner markers (at least two): its focal is fitted to the four markers when it sees them, and
remembered in outputs/perception/camera_focal.json for the times it sees fewer.

    uv run python -m climber.perception.capture --list        # the cameras OpenCV opens, and what they give
    uv run python -m climber.perception.capture --index 1     # live; s: save the image, q: quit

The images go to outputs/perception/.  On macOS, the first run asks for the camera permission of the terminal it
runs in (System Settings > Privacy & Security > Camera).
"""
import argparse
import time

import cv2
import numpy as np
from climbing_sim import Camera
from climbing_sim.cameras import d435i_intrinsics

from ..common import output_dir
from .markers import DICTIONARY, camera_pose, focal_for

COLOURS = {1: (0, 200, 0), 2: (0, 200, 0), 3: (0, 200, 0), 4: (0, 200, 0), 5: (0, 140, 255)}


def open_camera(index: int, width: int | None = None, height: int | None = None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def list_cameras(n: int = 5):
    for i in range(n):
        cap = open_camera(i)
        if not cap.isOpened():
            print(f"{i}: not opened")
            continue
        frame = None
        for _ in range(10):                            # the first frames may be black
            ok, frame = cap.read()
            time.sleep(0.05)
        print(f"{i}: {'no image' if frame is None else f'{frame.shape[1]} x {frame.shape[0]}, mean {frame.mean():.0f}'}")
        if frame is not None:
            path = output_dir("perception") / f"camera{i}.png"
            cv2.imwrite(str(path), frame)
            print(f"   {path}")
        cap.release()


def live(index: int, width: int | None, height: int | None):
    cap = open_camera(index, width, height)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} does not open (permission? another index? --list)")
    detector = cv2.aruco.ArucoDetector(DICTIONARY)
    camera, fitted = None, False
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        if camera is None:                             # nominal intrinsics at this resolution, until calibrated
            h, w = frame.shape[:2]
            try:
                k = d435i_intrinsics(w, h)
            except ValueError:                         # not a D435i colour mode: its field of view anyway
                f = (w / 2) / np.tan(np.radians(69.4 / 2))
                k = dict(fx=f, fy=f, cx=w / 2, cy=h / 2)
            camera = Camera("real", w, h, fx=k["fx"], fy=k["fy"], cx=k["cx"], cy=k["cy"])
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        shown = frame.copy()
        seen = [] if ids is None else ids.ravel().tolist()
        for c, i in zip(corners if ids is not None else [], seen):
            cv2.polylines(shown, [c.astype(np.int32)], True, COLOURS.get(i, (0, 0, 255)), 2)
            cv2.putText(shown, str(i), tuple(c[0, 0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOURS.get(i, (0, 0, 255)), 2)
        text = f"markers {sorted(seen)}"
        if len({1, 2, 3, 4} & set(seen)) >= 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not fitted:                             # the focal: fitted to the four markers, or remembered
                focal, source = focal_for(rgb, output_dir("perception") / "camera_focal.json")
                camera.fx = camera.fy = focal
                fitted = source.startswith("fitted")
                print(f"focal {focal:.0f} px: {source}")
            pose = camera_pose(rgb, camera)
            if pose is not None:
                R = cv2.Rodrigues(np.asarray(pose[0]))[0]
                text += f" | camera at {(-R.T @ pose[1]).round(2)} m (wall frame, focal {camera.fx:.0f} px)"
        cv2.putText(shown, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(f"camera {index} (s: save, q: quit)", shown)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            path = output_dir("perception") / f"camera{index}_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(str(path), frame)
            print(path, text)
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="try the first cameras, save one image of each")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    list_cameras() if args.list else live(args.index, args.width, args.height)
