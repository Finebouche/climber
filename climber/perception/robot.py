"""The robot's pose, live, from an ArUco marker on its trunk (the rulebook's ID 5, DICT_4X4_50 like the wall's).

The marker's corners are known in the trunk frame (``RobotMarker``).  Seen by both cameras (their poses from the corner
markers), each corner is triangulated, and the trunk's pose is the rigid fit of the four corners: a marker of ~20 px
at 2.4 m gives its depth badly from one camera (its apparent size), well from two cameras 1.1 m apart.  Seen by one
camera only: ``cv2.solvePnP`` on that image (less precise in depth).

The simulator has no robot marker: ``marked_sim`` adds one to the scene (a thin plate on the trunk's back, the side
the cameras see while it climbs).  Its size and place are guesses until we have the real one's.
"""
import pathlib
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass

import climbing_sim.sim
import cv2
import mujoco
import numpy as np

from ..common import make_sim

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_PARAMS = cv2.aruco.DetectorParameters()
_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY, _PARAMS)
_R_WORLD_WALL = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])   # wall-frame axes in the world


@dataclass
class RobotMarker:
    """The marker on the trunk: its code's side and white border (m), the centre of its face in the trunk frame (x
    forward, towards the wall; y left; z up the spine).  It faces backwards (-x), image up along the spine, so the
    cameras see it while the robot climbs.  Defaults: a guess on the trunk's back (108 x 128 mm)."""
    id: int = 5
    code: float = 0.056
    border: float = 0.010
    centre: tuple = (-0.046, 0.0, 0.068)

    @property
    def plate(self) -> float:
        return self.code + 2 * self.border

    def corners(self) -> np.ndarray:
        """4 x 3 trunk-frame corners of the code, in OpenCV's order (top-left, top-right, bottom-right, bottom-left
        as seen from behind the robot: right is the robot's right, -y)."""
        c, h = np.asarray(self.centre, float), self.code / 2
        right, up = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        return np.array([c - h * right + h * up, c + h * right + h * up, c + h * right - h * up, c - h * right - h * up])

    def image(self, px_per_cell: int = 40) -> np.ndarray:
        """RGB image of the marker with its white border (for the simulator's texture, or to print)."""
        code = cv2.aruco.generateImageMarker(DICTIONARY, self.id, 6 * px_per_cell)
        b = int(round(6 * px_per_cell * self.border / self.code))
        return np.repeat(np.pad(code, b, constant_values=255)[..., None], 3, axis=2)


def add_marker(spec: mujoco.MjSpec, marker: RobotMarker):
    """Add the marker to a scene's trunk: a thin plate whose back face carries the texture."""
    path = pathlib.Path(tempfile.gettempdir()) / f"climber_robot_marker_{marker.id}.png"
    cv2.imwrite(str(path), marker.image())
    spec.add_texture(name="robot_marker", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(path))
    spec.add_material(name="robot_marker").textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "robot_marker"
    # box axes in the trunk frame: x -> -y (image right), y -> +z (image up), z -> -x (out of the robot's back)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).flatten())
    thickness = 0.001
    pos = np.asarray(marker.centre) + [thickness / 2, 0.0, 0.0]       # its outer face at the centre given
    spec.body("trunk").add_geom(name="robot_marker", type=mujoco.mjtGeom.mjGEOM_BOX, material="robot_marker",
                                size=[marker.plate / 2, marker.plate / 2, thickness / 2], pos=pos.tolist(),
                                quat=quat.tolist(), contype=0, conaffinity=0, group=1, mass=0)


@contextmanager
def _marker_in_scenes(marker: RobotMarker):
    build = climbing_sim.sim.build_spec

    def build_with_marker(**kwargs):
        spec = build(**kwargs)
        add_marker(spec, marker)
        return spec

    climbing_sim.sim.build_spec = build_with_marker
    try:
        yield
    finally:
        climbing_sim.sim.build_spec = build


def marked_sim(wall="N2", marker: RobotMarker | None = None, **kwargs):
    """``make_sim`` (with the two default cameras unless ``cameras`` is given), the marker on the robot's trunk."""
    kwargs.setdefault("cameras", None)
    with _marker_in_scenes(marker or RobotMarker()):
        return make_sim(wall, **kwargs)


def detect(image: np.ndarray, marker_id: int):
    """4 x 2 image corners of the marker (sub-pixel) in an RGB or grey image, or None."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(gray)
    if ids is None or marker_id not in ids.ravel():
        return None
    return corners[list(ids.ravel()).index(marker_id)].reshape(4, 2).astype(float)


def _rigid_fit(a: np.ndarray, b: np.ndarray):
    """(R, t) with b ~ R a + t (N x 3 points)."""
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, cb - R @ ca


def robot_pose(images: dict, cameras: dict, marker: RobotMarker):
    """The trunk's pose in the world frame, (position, quaternion w x y z, number of cameras used), or None if no
    camera sees the marker.  ``images``, ``cameras``: name -> RGB image, Camera (its pose: from the corner markers)."""
    seen = {n: c for n, c in ((n, detect(img, marker.id)) for n, img in images.items()) if c is not None}
    if not seen:
        return None
    if len(seen) >= 2:                       # triangulate the corners (wall frame), fit the trunk to them
        (n1, c1), (n2, c2) = list(seen.items())[:2]
        P = [cameras[n].K() @ np.hstack([cameras[n].R_wall_to_cam(), np.reshape(cameras[n].tvec, (3, 1))])
             for n in (n1, n2)]
        X = cv2.triangulatePoints(P[0], P[1], c1.T, c2.T)
        corners_wall = (X[:3] / X[3]).T
        R_wall_trunk, p_wall = _rigid_fit(marker.corners(), corners_wall)
    else:                                    # one camera: PnP
        (name, img_corners), = seen.items()
        cam = cameras[name]
        ok, rvec, tvec = cv2.solvePnP(marker.corners(), img_corners, cam.K(), np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        R_cw, t_cw = cam.R_wall_to_cam(), np.asarray(cam.tvec, float)     # P_cam = R_cw P_wall + t_cw
        R_wall_trunk = R_cw.T @ cv2.Rodrigues(rvec)[0]
        p_wall = R_cw.T @ (tvec.ravel() - t_cw)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (_R_WORLD_WALL @ R_wall_trunk).flatten())
    return _R_WORLD_WALL @ p_wall, quat, len(seen)
