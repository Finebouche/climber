"""Postures of stances.  A stance gives each limb a hold: ``{limb: hold index in sim.wall.holds}``.

placo's IK with the trunk free (``solver.py``): the trunk's position and tilt come with the joints.  A posture is
reached when every ball is within REACH_TOL of its socket and the body stays CLEARANCE_MIN away from the wall and
the holds.
"""
from typing import NamedTuple

import numpy as np

from ..common import LIMBS
from .solver import ClimberIK

REACH_TOL = 0.003                            # m: a ball farther than this from its target is not reached
CLEARANCE_MIN = 0.003                        # m: the body (not the hands and feet) this far from the wall and holds
ABOVE = np.array([0.0, 0.0, 0.001])          # balls 1 mm above the socket centre, as reset_hanging() does


class Posture(NamedTuple):
    q: np.ndarray                            # 19 joint angles, order JOINTS
    trunk_pos: np.ndarray
    trunk_quat: np.ndarray                   # w x y z
    error: float                             # largest ball distance left (m)
    clearance: float                         # body to wall and holds (m; < 0: inside)

    @property
    def reached(self) -> bool:
        return self.error < REACH_TOL and self.clearance > CLEARANCE_MIN


def ball_targets(sim, stance: dict) -> dict:
    """limb -> world position of its ball in the stance."""
    return {l: sim.wall.holds[i].socket + ABOVE for l, i in stance.items()}


def trunk_for(sim, stance: dict):
    """A first guess of the trunk pose for the stance: the current one, moved by as much as the centroid of the
    balls."""
    pos, quat = sim.trunk_pose()
    now = np.mean([p for l, p in sim.ball_positions().items() if l in stance], axis=0)
    then = np.mean(list(ball_targets(sim, stance).values()), axis=0)
    return pos + then - now, quat


def stance_posture(sim, stance: dict, q_start=None, trunk_start=None) -> Posture:
    """The posture of a stance (all four limbs), from the simulation's current posture (or ``q_start`` with the
    trunk at ``trunk_start`` = (pos, quat); default: ``trunk_for``).  First the trunk free; if the body then hits the
    wall, again with the trunk kept at its distance from the wall.  The simulation is not touched."""
    assert set(stance) == set(LIMBS), f"a stance gives a hold to each of {LIMBS}"
    ik = ClimberIK.of(sim)
    q0 = sim.joint_positions() if q_start is None else q_start
    pos0, quat0 = trunk_for(sim, stance) if trunk_start is None else trunk_start
    targets = ball_targets(sim, stance)
    best = None
    for hold_y in (None, sim.trunk_pose()[0][1]):
        ik.set_state(q0, pos0, quat0)
        error = ik.solve(targets, hold_y=hold_y)
        q, pos, quat = ik.state()
        posture = Posture(q, pos, quat, error, ik.clearance(q, pos, quat))
        if posture.reached:
            return posture
        if best is None or (posture.error, -posture.clearance) < (best.error, -best.clearance):
            best = posture
    return best
