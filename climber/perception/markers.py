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


def _detector(relaxed: bool) -> cv2.aruco.ArucoDetector:
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if relaxed:                              # small markers far away (~30 px on the real wall's top ones)
        p.adaptiveThreshWinSizeMax = 53
        p.minMarkerPerimeterRate = 0.01
        p.polygonalApproxAccuracyRate = 0.05
    return cv2.aruco.ArucoDetector(DICTIONARY, p)


DETECTORS = (_detector(False), _detector(True))


def detect(image: np.ndarray) -> dict:
    """marker id -> 4 x 2 image corners, for the wall's markers (1-4) found in an image (RGB or grey: the D435i's
    infrared).  If one is missing, again on the image enlarged twice with a looser detector."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    found = {}
    for scale, detector in ((1, DETECTORS[0]), (2, DETECTORS[1])):
        g = gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = detector.detectMarkers(g)
        if ids is not None:
            for i, c in zip(ids.ravel(), corners):
                if int(i) in aruco.CORNERS and int(i) not in found:
                    found[int(i)] = (c.reshape(4, 2) + 0.5) / scale - 0.5     # pixel centres back to the image
        if set(aruco.CORNERS) <= set(found):
            break
    return found


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


def estimate_focal(image: np.ndarray, focals=np.arange(300.0, 2000.0, 2.0)) -> tuple[float, float]:
    """(focal in px, reprojection error in px) that best fit the four corner markers, the principal point at the
    image's centre and no distortion: a first calibration from one image (needs the four markers)."""
    found = detect(image)
    if len(found) < 4:
        raise ValueError(f"the four corner markers are needed, found {sorted(found)}")
    h, w = image.shape[:2]
    obj = np.concatenate([marker_corners_wall(i) for i in sorted(found)])
    pts = np.concatenate([found[i] for i in sorted(found)])
    best = (np.inf, None)
    for f in focals:
        K = np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])
        ok, rvec, tvec = cv2.solvePnP(obj, pts, K, np.zeros(5))
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros(5))
        error = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pts) ** 2, axis=1))))
        best = min(best, (error, float(f)))
    return best[1], best[0]


def focal_for(image: np.ndarray, path) -> tuple[float, str]:
    """(focal in px, where it comes from) for this camera: fitted to the four corner markers when the image shows
    them (then remembered in the JSON file ``path``, per resolution), else the remembered one, else the D435i
    infrared's (86 deg across)."""
    import json
    import pathlib
    path = pathlib.Path(path)
    h, w = image.shape[:2]
    key = f"{w}x{h}"
    known = json.loads(path.read_text()) if path.exists() else {}
    try:
        focal, error = estimate_focal(image)
        known[key] = focal
        path.write_text(json.dumps(known, indent=1))
        return focal, f"fitted to the 4 markers (reprojection {error:.2f} px)"
    except ValueError:
        if key in known:
            return known[key], f"remembered ({path.name}): fewer than 4 markers"
        return (w / 2) / np.tan(np.radians(86.0 / 2)), "the D435i infrared's nominal: fewer than 4 markers"
