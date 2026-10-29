"""The camera's pose from the four ArUco corner markers (DICT_4X4_50, IDs 1-4), as the rulebook asks the teams.

The corners of each marker's black code are known in the wall frame (x right, y up, z out of the wall); OpenCV's
detector finds them in the image, ``cv2.solvePnP`` gives rvec, tvec (wall -> camera), the convention of
``climbing_sim.Camera`` and of the camera editor (``climbing-sim-cameras``).
"""
import cv2
import numpy as np
from climbing_sim import Camera, aruco
from climbing_sim.wall import WALL_HEIGHT, WALL_WIDTH

MARKER_Z = 0.0005                            # m, the marker plates' face in front of the wall's (as in the scene)
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def marker_corners_wall(marker_id: int) -> np.ndarray:
    """4 x 3 wall-frame corners of the marker's code, in OpenCV's order (top-left, top-right, bottom-right,
    bottom-left of the marker as seen)."""
    x0, y0 = aruco.corner_plate_origin(marker_id, WALL_WIDTH, WALL_HEIGHT) + aruco.BORDER
    c = aruco.CODE
    return np.array([[x0, y0 + c, MARKER_Z], [x0 + c, y0 + c, MARKER_Z], [x0 + c, y0, MARKER_Z], [x0, y0, MARKER_Z]])


def detect(image: np.ndarray) -> dict:
    """marker id -> 4 x 2 image corners, for the wall's markers (1-4) found in an RGB image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = cv2.aruco.ArucoDetector(DICTIONARY).detectMarkers(gray)
    found = {} if ids is None else {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}
    return {i: c for i, c in found.items() if i in aruco.CORNERS}


def camera_pose(image: np.ndarray, camera: Camera):
    """(rvec, tvec, ids found) of the camera (wall -> camera, OpenCV), from its intrinsics and an image; None
    when fewer than two markers are found."""
    found = detect(image)
    if len(found) < 2:
        return None
    obj = np.concatenate([marker_corners_wall(i) for i in found])
    img = np.concatenate(list(found.values()))
    dist = np.zeros(5)                       # the simulator renders without distortion; use your calibration's
    ok, rvec, tvec = cv2.solvePnP(obj, img, camera.K(), dist, flags=cv2.SOLVEPNP_ITERATIVE)
    return (rvec.ravel(), tvec.ravel(), sorted(found)) if ok else None
