"""Perception: where the cameras are, where the holds are, where the robot is.

* ``markers.py``: each camera's pose from the four ArUco corner markers (IDs 1-4);
* ``holds.py``: the map of the wall (which grid positions hold a hold), from both cameras: best done before the
  robot is on the wall (it hides the holds behind it);
* ``robot.py``: the trunk's pose, live, from the robot's own marker (ID 5), triangulated by both cameras;
* ``capture.py``: a real camera, live, with the markers it sees; ``map_image.py``: the map of a real wall from one of
  its images (holds darker than the wall, or coloured).

Not on the toolbox slide, asked by the rulebook (challenge 1, which the markers make easier).  Tools: OpenCV
(``cv2.aruco``, ``cv2.solvePnP``, ``cv2.triangulatePoints``), the RealSense D435i cameras; the simulator renders their
images.  Needs the ``perception`` extra.

    python -m climber.perception [N2]   # cameras, the map of the empty wall, the robot's pose: against the truth
"""
from .holds import colour_mask, compare, dark_mask, hold_mask, hold_shape, map_holds
from .markers import camera_pose, detect, estimate_focal, marker_corners_wall
from .robot import RobotMarker, marked_sim, robot_pose

__all__ = ["RobotMarker", "camera_pose", "colour_mask", "compare", "dark_mask", "detect", "estimate_focal", "hold_mask", "hold_shape",
           "map_holds", "marked_sim", "marker_corners_wall", "robot_pose"]
