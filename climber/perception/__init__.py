"""Perception: where the cameras are (the four ArUco corner markers), then where the holds and the robot are.

Not on the toolbox slide, asked by the rulebook.  Tools: OpenCV (``cv2.aruco``, ``cv2.solvePnP``), the RealSense
D435i cameras; the simulator renders their images.  Needs the ``perception`` extra.

    python -m climber.perception        # the pose of the simulated cameras from their images, against the truth
"""
from .markers import camera_pose, detect, marker_corners_wall

__all__ = ["camera_pose", "detect", "marker_corners_wall"]
