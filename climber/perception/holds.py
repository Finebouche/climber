"""The map of the wall: which holds are bolted where, from the two cameras' images.

A hold sits on the 15 mm grid of holes (``Hold(col, row)``: its left hole), so the map is a choice among the grid's
positions rather than free positions.  For every position, the hold's 3D shape placed there is projected into each
camera (pose from the corner markers); its score is the share of that silhouette the image shows as "hold" (a mask:
the simulator's orange, ``hold_mask``; holds darker than a light wall, ``dark_mask``: the real wall in the D435i's
infrared; those and the coloured ones in a colour image, ``colour_mask``).  The best positions are kept, two holds never overlapping (the wall's
own rule: ``climbing_sim.wall.OVERLAP``).  A hold hidden by the robot in one camera is seen by the other: the score
is the best of the two.

The mask is the part to adapt to the real holds (their colour, their darkness, or the difference with a photo of the
empty wall); the rest only needs the cameras' poses and the hold's shape.
"""
import cv2
import mujoco
import numpy as np
from climbing_sim import Hold, Wall
from climbing_sim.wall import GRID_COLS, GRID_ROWS, OVERLAP, WALL_HEIGHT, WALL_WIDTH, world_to_wall

SCORE_MIN = 0.5                              # share of a hold's silhouette that must look like a hold
SIDE = 0.006                                 # m: the strips of wall checked on each side of a hold
SIDE_PENALTY = 0.2                           # the score loses this times the darkest side strip's share of "hold"
BLOB_MAX = 4.0                               # a patch of "hold" bigger than this many holds is something else
# the space the robot takes around its trunk (trunk frame, m: x forward, y left, z up the spine), hanging or climbing
ROBOT_BOX = np.array([[-0.06, 0.05], [-0.10, 0.10], [-0.17, 0.20]])
HSV_LOW, HSV_HIGH = (3, 120, 60), (16, 255, 255)   # the simulator's orange holds (OpenCV HSV: hue 0-180)


DARK = 0.7                                   # dark_mask: a hold is darker than this share of the wall around it


def hold_mask(image: np.ndarray, camera=None) -> np.ndarray:
    """uint8 mask (255: hold) of an RGB image: the simulator's orange."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, HSV_LOW, HSV_HIGH)


SATURATION = 90                              # colour_mask: OpenCV saturation (0-255) of a coloured hold


def _board(shape, camera) -> np.ndarray:
    """uint8 mask of the board as the camera sees it (its pose)."""
    board = np.zeros(shape[:2], np.uint8)
    corners = np.array([[0, 0, 0], [WALL_WIDTH, 0, 0], [WALL_WIDTH, WALL_HEIGHT, 0], [0, WALL_HEIGHT, 0]], float)
    cv2.fillConvexPoly(board, np.round(camera.project_wall(corners)).astype(np.int32), 255)
    return board


def colour_mask(image: np.ndarray, camera) -> np.ndarray:
    """uint8 mask (255: hold) for a light, grey wall in a colour image: the holds darker than the wall
    (``dark_mask``) and the coloured ones, however light (saturation > SATURATION; the wall and the holds' shadows
    are grey).  On a grey image (infrared), the same as ``dark_mask``."""
    dark = dark_mask(image, camera)
    if image.ndim == 2:
        return dark
    saturated = (cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[..., 1] > SATURATION) & (_board(image.shape, camera) > 0)
    saturated = cv2.morphologyEx(saturated.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return dark | saturated


def dark_mask(image: np.ndarray, camera) -> np.ndarray:
    """uint8 mask (255: hold) of the pixels of the board darker than DARK times the board around them (holds on a
    light wall, e.g. the real wall in the D435i's infrared).  The board is where the camera sees it (its pose); the
    grid's holes (a pixel or two) are opened away."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    board = _board(gray.shape, camera)
    light = cv2.dilate(gray, np.ones((15, 15), np.uint8))           # the wall around, without the holds
    light = cv2.blur(light, (15, 15)).astype(float)
    dark = ((gray < DARK * light) & (board > 0)).astype(np.uint8) * 255
    return cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def hold_shape(model: mujoco.MjModel, data: mujoco.MjData, directions: int = 400) -> np.ndarray:
    """Points of the hold's visual mesh on its convex hull (support points along many directions), in the wall frame,
    relative to the hold's socket (N x 3), as the scene places it (MuJoCo recentres meshes: the geom's pose)."""
    mesh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "hold_visual")
    geom = next(g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                and model.geom_dataid[g] == mesh)
    start = model.mesh_vertadr[mesh]
    v = model.mesh_vert[start:start + model.mesh_vertnum[mesh]].astype(float)
    v = v @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom] - data.xpos[model.geom_bodyid[geom]]
    u = np.random.default_rng(0).normal(size=(directions, 3))
    support = np.unique(np.argmax(v @ u.T, axis=0))
    return world_to_wall(v[support])


def candidates() -> list[Hold]:
    """Every position of a hold on the grid that keeps it off the corner markers."""
    out = []
    for col in range(GRID_COLS - 2):
        for row in range(GRID_ROWS):
            h = Hold(col, row)
            if not h.on_markers():
                out.append(h)
    return out


def hold_area(camera, shape: np.ndarray) -> float:
    """Area (px) of a hold's silhouette in the middle of the wall, as the camera sees it."""
    middle = world_to_wall(Hold(GRID_COLS // 2 - 1, GRID_ROWS // 2).socket)
    return float(cv2.contourArea(cv2.convexHull(cam_points(camera, middle + shape))))


def cam_points(camera, points_wall: np.ndarray) -> np.ndarray:
    return np.round(camera.project_wall(points_wall)).astype(np.int32)


def drop_large(mask: np.ndarray, area_max: float) -> np.ndarray:
    """The mask without its connected patches bigger than area_max px (a robot, a person: not holds)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    big = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] > area_max)
    big = big[big != 0]
    if big.size:
        mask = mask.copy()
        mask[np.isin(labels, big)] = False
    return mask


def robot_region(camera, trunk_pos, trunk_quat) -> np.ndarray:
    """Image polygon (convex hull) of the space the robot takes (ROBOT_BOX around its trunk), from its pose in the
    world frame (``climber.perception.robot_pose``)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(trunk_quat, float))
    corners = np.array([[x, y, z] for x in ROBOT_BOX[0] for y in ROBOT_BOX[1] for z in ROBOT_BOX[2]])
    world = corners @ R.reshape(3, 3).T + np.asarray(trunk_pos, float)
    return cv2.convexHull(cam_points(camera, world_to_wall(world)))


def scores(images: dict, cameras: dict, shape: np.ndarray, holds: list[Hold], mask=hold_mask,
           exclude: dict | None = None) -> np.ndarray:
    """For each candidate hold, the best over the cameras of the share of its projected silhouette that is in the
    camera's hold mask (``mask(image, camera)``), minus SIDE_PENALTY times the less "hold-like" of the two strips of
    wall just left and right of it: a real hold has wall on at least one side, a position one hole off inside two
    touching holds has hold on both.  ``images``, ``cameras``: name -> image, Camera.  Patches of "hold" bigger than
    BLOB_MAX holds (the robot, a person) are ignored, as are the polygons of ``exclude`` (name -> list of image
    polygons, e.g. ``robot_region``)."""
    best = np.zeros(len(holds))
    for name, image in images.items():
        cam = cameras[name]
        mask_ = drop_large(mask(image, cam) > 0, BLOB_MAX * hold_area(cam, shape))
        if exclude and exclude.get(name):
            hidden = np.zeros(mask_.shape, np.uint8)
            for polygon in exclude[name]:
                cv2.fillConvexPoly(hidden, polygon, 1)
            mask_ &= hidden == 0
        h_img, w_img = mask_.shape
        integral = cv2.integral(mask_.astype(np.uint8))
        sockets = np.array([world_to_wall(h.socket) for h in holds])
        centre = cam.project_wall(sockets)
        for k, h in enumerate(holds):
            cx, cy = centre[k]
            if not (0 <= cx < w_img and 0 <= cy < h_img):
                continue
            # quick reject: almost nothing of the hold's colour around its centre
            x0, y0, x1, y1 = int(cx) - 4, int(cy) - 4, int(cx) + 5, int(cy) + 5
            if x0 < 0 or y0 < 0 or x1 > w_img or y1 > h_img:
                continue
            if integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0] < 10:
                continue
            pts = cam.project_wall(sockets[k] + shape)
            hull = cv2.convexHull(np.round(pts).astype(np.int32))
            bx, by, bw, bh = cv2.boundingRect(hull)
            if bx < 0 or by < 0 or bx + bw > w_img or by + bh > h_img or bw * bh == 0:
                continue
            poly = np.zeros((bh, bw), np.uint8)
            cv2.fillConvexPoly(poly, hull - [bx, by], 1)
            inside = poly.astype(bool)
            score = mask_[by:by + bh, bx:bx + bw][inside].mean()
            sides = []
            for dx in (-SIDE, SIDE):                  # the strip the hold would cover if moved by dx, minus itself
                moved = cv2.convexHull(np.round(cam.project_wall(sockets[k] + shape + [dx, 0.0, 0.0])).astype(np.int32))
                sx, sy, sw, sh = cv2.boundingRect(np.vstack([moved, hull]))
                if sx < 0 or sy < 0 or sx + sw > w_img or sy + sh > h_img:
                    sides.append(0.0)
                    continue
                strip = np.zeros((sh, sw), np.uint8)
                cv2.fillConvexPoly(strip, moved - [sx, sy], 1)
                cv2.fillConvexPoly(strip, hull - [sx, sy], 0)
                sides.append(mask_[sy:sy + sh, sx:sx + sw][strip > 0].mean() if strip.any() else 0.0)
            best[k] = max(best[k], score - SIDE_PENALTY * min(sides))
    return best


def map_holds(images: dict, cameras: dict, shape: np.ndarray, score_min: float = SCORE_MIN, mask=hold_mask,
              exclude: dict | None = None) -> Wall:
    """The holds seen in the images, as a Wall (no start holds): the best-scoring grid positions, without overlaps.
    ``shape``: ``hold_shape`` (the hold's 3D shape); ``mask``: ``hold_mask``, ``dark_mask`` or ``colour_mask``;
    ``exclude``: image polygons to ignore per camera (``robot_region``)."""
    holds = candidates()
    s = scores(images, cameras, shape, holds, mask, exclude)
    taken: set = set()
    wall = Wall(name="mapped")
    for k in np.argsort(s)[::-1]:
        if s[k] < score_min:
            break
        h = holds[k]
        if any((h.col + dc, h.row + dr) in taken for dc, dr in OVERLAP):
            continue
        taken.add((h.col, h.row))
        wall.holds.append(h)
    return wall


def compare(mapped: Wall, truth: Wall) -> dict:
    """Holds found at their exact grid position, missed, and extra (false), against the true wall."""
    found = {(h.col, h.row) for h in mapped.holds}
    real = {(h.col, h.row) for h in truth.holds}
    return {"exact": len(found & real), "missed": sorted(real - found), "extra": sorted(found - real),
            "total": len(real)}
