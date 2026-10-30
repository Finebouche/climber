"""The map of a real wall from one camera image (``climber.perception.capture`` saves them): the camera's pose from
the corner markers, the holds on the grid (darker than the wall or coloured: ``colour_mask``), the wall written as JSON
(open it in ``climbing-sim-wall-editor``), and a picture of what was found.  The robot is left out: by the space it
takes around its marker (ID 5) when the image shows it, and anyway as a dark patch much bigger than a hold.  A colour image (any colour camera: the
competition's USB cameras, a webcam) or the D435i's infrared one.

    uv run python -m climber.perception.map_image --compare N1           # the last capture, against a wall's JSON
    uv run python -m climber.perception.map_image outputs/perception/camera0.png   # a given image

In the infrared, holds as light as the wall are missed: a colour image shows the coloured ones.
"""
import argparse
import pathlib

import cv2
import numpy as np
from climbing_sim import Camera, Wall
from climbing_sim.wall import WALL_HEIGHT, WALL_WIDTH, world_to_wall

from ..common import make_sim, output_dir
from .holds import colour_mask, compare, hold_shape, map_holds, robot_region
from .markers import camera_pose, focal_for
from .robot import RobotMarker, robot_pose

parser = argparse.ArgumentParser()
parser.add_argument("image", nargs="?", help="default: the last capture (s in climber.perception.capture)")
parser.add_argument("--focal", type=float, help="px (default: fitted to the 4 corner markers, or remembered)")
parser.add_argument("--score", type=float, default=0.4, help="share of a hold's silhouette that must look like one")
parser.add_argument("--compare", help="a wall (N1, N2, N3 or a JSON file) to compare the map with")
args = parser.parse_args()

if args.image is None:
    captures = sorted(output_dir("perception").glob("camera*_[0-9]*.png"), key=lambda p: p.stat().st_mtime)
    captures = [p for p in captures if not p.stem.endswith(("_map", "_wall"))]
    if not captures:
        raise SystemExit("no capture in outputs/perception: press s in climber.perception.capture")
    args.image = str(captures[-1])
    print(args.image)
image = cv2.imread(args.image)
if image is None:
    raise SystemExit(f"cannot read {args.image}")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
    rgb = rgb[..., 0]                         # a grey image (infrared) saved in three channels
h, w = rgb.shape[:2]
if args.focal:
    focal = args.focal
else:
    focal, source = focal_for(rgb, output_dir("perception") / "camera_focal.json")
    print(f"focal {focal:.0f} px (horizontal field {2 * np.degrees(np.arctan(w / 2 / focal)):.1f} deg): {source}")
camera = Camera("real", w, h, focal, focal, w / 2, h / 2)
pose = camera_pose(rgb, camera)
if pose is None:
    raise SystemExit("fewer than two corner markers found")
camera.rvec, camera.tvec = list(pose[0]), list(pose[1])
R = cv2.Rodrigues(np.asarray(pose[0]))[0]
print(f"markers {pose[2]}; camera at {(-R.T @ np.asarray(pose[1])).round(3)} m (wall frame: x right, y up, z out)")

sim = make_sim("N1")                          # only for the hold's 3D shape
shape = hold_shape(sim.model, sim.data)
sim.close()
exclude = None
robot = robot_pose({"real": rgb}, {"real": camera}, RobotMarker())
if robot is not None:                         # its marker seen: the space it takes is left out
    exclude = {"real": [robot_region(camera, robot[0], robot[1])]}
    print(f"robot marker seen: trunk at {robot[0].round(3)} m (world frame), its space left out")
wall = map_holds({"real": rgb}, {"real": camera}, shape, args.score, mask=colour_mask, exclude=exclude)
stem = pathlib.Path(args.image).stem
path = output_dir("perception") / f"{stem}_wall.json"
wall.to_json(path)
print(f"{len(wall.holds)} holds -> {path}")
if args.compare:
    r = compare(wall, Wall.load(args.compare))
    print(f"against {args.compare}: {r['exact']} / {r['total']} at their grid position, missed {r['missed']}, "
          f"extra {r['extra']}")

shown = image.copy()
for polygon in (exclude or {}).get("real", []):
    cv2.polylines(shown, [polygon], True, (0, 140, 255), 2)
corners = np.array([[0, 0, 0], [WALL_WIDTH, 0, 0], [WALL_WIDTH, WALL_HEIGHT, 0], [0, WALL_HEIGHT, 0]], float)
cv2.polylines(shown, [np.round(camera.project_wall(corners)).astype(np.int32)], True, (255, 0, 0), 1)
for hold in wall.holds:
    pts = camera.project_wall(world_to_wall(hold.socket) + shape)
    cv2.polylines(shown, [cv2.convexHull(np.round(pts).astype(np.int32))], True, (0, 220, 0), 1)
path = output_dir("perception") / f"{stem}_map.png"
cv2.imwrite(str(path), shown)
print(path)
