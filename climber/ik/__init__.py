"""Inverse kinematics: the joint angles of a stance (the four balls in four holds), the trunk free.

Tools: placo (``solver.py``: a QP on tasks, on the climbing URDF; Rhoban's, on the robot too), pink, mink.  The trunk
is placo's floating base: the solver chooses its position and tilt.  MuJoCo checks that the body stays off the wall.

    python -m climber.ik [N2]          # the start stance and the stances one hold up: reachable or not
"""
from .solver import ClimberIK
from .stance import CLEARANCE_MIN, REACH_TOL, Posture, ball_targets, stance_posture, trunk_for

__all__ = ["CLEARANCE_MIN", "REACH_TOL", "ClimberIK", "Posture", "ball_targets", "stance_posture", "trunk_for"]
