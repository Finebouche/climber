"""Estimate each simulated camera's pose from its image and compare it with the pose the simulator used."""
import cv2
import numpy as np

from ..common import make_sim
from .markers import camera_pose

sim = make_sim("N2", cameras=None)            # the two default cameras
sim.reset_hanging()
for name, cam in sim.cameras.items():
    est = camera_pose(sim.render(name), cam)
    if est is None:
        print(f"{name}: fewer than two markers found")
        continue
    rvec, tvec, ids = est
    R, R_est = cam.R_wall_to_cam(), cv2.Rodrigues(rvec)[0]
    err_pos = np.linalg.norm(-R_est.T @ tvec - cam.position_wall())
    err_rot = np.degrees(np.linalg.norm(cv2.Rodrigues(R_est @ R.T)[0]))
    print(f"{name}: markers {ids}, position error {1000 * err_pos:.1f} mm, orientation error {err_rot:.2f} deg")
sim.close()
