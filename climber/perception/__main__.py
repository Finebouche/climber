"""The perception chain on a simulated wall, each step against the simulator's truth: the cameras' poses from the
corner markers, the map of the empty wall (robot aside) against the wall's JSON, then the robot hanging on it and its
trunk's pose from its marker."""
import copy
import sys
import time

import cv2
import mujoco
import numpy as np

from .holds import compare, hold_shape, map_holds
from .markers import camera_pose
from .robot import RobotMarker, marked_sim, robot_pose

sim = marked_sim(sys.argv[1] if len(sys.argv) > 1 else "N2")
sim.reset(trunk_pos=[-0.8, -0.6, 0.3])        # the robot aside: the wall is empty, as before a run

# 1. the cameras, from the corner markers
images = {name: sim.render(name) for name in sim.cameras}
cameras = {}
for name, cam in sim.cameras.items():
    rvec, tvec, ids = camera_pose(images[name], cam)
    cameras[name] = copy.copy(cam)
    cameras[name].rvec, cameras[name].tvec = list(rvec), list(tvec)
    R = cv2.Rodrigues(np.asarray(rvec))[0]
    error = np.linalg.norm(-R.T @ np.asarray(tvec) - cam.position_wall())
    print(f"{name}: markers {ids}, position {1000 * error:.1f} mm from the truth")

# 2. the map of the wall
t = time.time()
mapped = map_holds(images, cameras, hold_shape(sim.model, sim.data))
r = compare(mapped, sim.wall)
print(f"map: {r['exact']} / {r['total']} holds at their grid position, missed {r['missed']}, extra {r['extra']} "
      f"({time.time() - t:.1f} s)")

# 3. the robot on the wall, its pose from its marker
sim.reset_hanging()
images = {name: sim.render(name) for name in sim.cameras}
t = time.time()
est = robot_pose(images, cameras, RobotMarker())
pos, quat = sim.trunk_pose()
if est is None:
    print("robot: its marker is not seen")
else:
    qd, qi, e = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(qi, quat)
    mujoco.mju_mulQuat(qd, est[1], qi)
    mujoco.mju_quat2Vel(e, qd, 1.0)
    print(f"robot: trunk at {est[0].round(3)} m ({est[2]} cameras), {1000 * np.linalg.norm(est[0] - pos):.1f} mm and "
          f"{np.degrees(np.linalg.norm(e)):.1f} deg from the truth ({1000 * (time.time() - t):.0f} ms)")
sim.close()
