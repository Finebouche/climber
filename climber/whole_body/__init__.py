"""Whole-body dynamics: the forces and torques of climbing (the balls in the holds as contacts).

Tools: TSID (task-space inverse dynamics), Crocoddyl (optimal control), both on Pinocchio.  Their pip wheels do not
work with the Pinocchio that placo (the robot's) pins: install them from conda-forge in a separate environment when
needed.  Here we start with what the robot measures, the servos' currents (``loads.py``).

    python -m climber.whole_body        # the most loaded servos, hanging at the start, at kp 400 and 125
"""
from .loads import MAX_CURRENT, holding_currents, most_loaded

__all__ = ["MAX_CURRENT", "holding_currents", "most_loaded"]
