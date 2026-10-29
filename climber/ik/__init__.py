"""Inverse kinematics: the joint angles of a stance (the four balls in four holds).

Tools: pink, placo, mink (task-based IK: several tasks with priorities, joint limits, collisions); here we start with
the simulator's own damped least squares (``sim.joint_angles_for``), the trunk pose searched on a small grid (a floating-base IK, e.g. placo, would choose it).

    python -m climber.ik [N2]          # the start stance and the stances one hold up: reachable or not
"""
from .stance import REACH_TOL, Posture, ball_targets, stance_posture, trunk_for

__all__ = ["REACH_TOL", "Posture", "ball_targets", "stance_posture", "trunk_for"]
