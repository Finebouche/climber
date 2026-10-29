"""Postures of stances.  A stance gives each limb a hold: ``{limb: hold index in sim.wall.holds}``."""
import itertools
from typing import NamedTuple

import numpy as np

from ..common import LIMBS

REACH_TOL = 0.003                            # m: a ball farther than this from its target is not reached
ABOVE = np.array([0.0, 0.0, 0.001])          # balls 1 mm above the socket centre, as reset_hanging() does
# trunk offsets tried around the guess (world frame, m; y towards the wall), nearest first.  The guess alone often
# fails: a hand 10 cm higher needs the trunk higher and closer to the wall.
TRUNK_GRID = sorted(itertools.product([0.0, -0.02, 0.02], [0.0, 0.03, 0.06, -0.03], np.arange(-0.02, 0.13, 0.02)),
                    key=lambda o: np.linalg.norm(o))
COARSE_GRID = sorted(itertools.product([0.0], [0.0, 0.03, 0.06], [0.0, 0.04, 0.08, 0.12]),
                     key=lambda o: np.linalg.norm(o))      # 12 offsets: for checking many stances (planning)


class Posture(NamedTuple):
    q: np.ndarray                            # 19 joint angles, order JOINTS
    trunk_pos: np.ndarray
    trunk_quat: np.ndarray
    error: float                             # largest ball distance left (m)

    @property
    def reached(self) -> bool:
        return self.error < REACH_TOL


def ball_targets(sim, stance: dict) -> dict:
    """limb -> world position of its ball in the stance."""
    return {l: sim.wall.holds[i].socket + ABOVE for l, i in stance.items()}


def trunk_for(sim, stance: dict):
    """A guess of the trunk pose for the stance: the current one, moved by as much as the centroid of the balls."""
    pos, quat = sim.trunk_pose()
    now = np.mean([p for l, p in sim.ball_positions().items() if l in stance], axis=0)
    then = np.mean(list(ball_targets(sim, stance).values()), axis=0)
    return pos + then - now, quat


def stance_posture(sim, stance: dict, trunk_pos=None, trunk_quat=None, q_start=None, search: bool = True,
                   grid=TRUNK_GRID) -> Posture:
    """Joint angles putting the balls in the stance's holds, the trunk at the given pose (default: ``trunk_for``,
    same orientation).  ``search``: if that fails, the trunk moved by the offsets of ``grid``, the first that
    reaches (else the best).  The simulation is not touched."""
    assert set(stance) <= set(LIMBS)
    if trunk_pos is None:
        trunk_pos, trunk_quat = trunk_for(sim, stance)
    targets = ball_targets(sim, stance)
    best = None
    for offset in grid if search else [(0.0, 0.0, 0.0)]:
        pos = np.asarray(trunk_pos) + offset
        q, err = sim.joint_angles_for(targets, pos, trunk_quat, q_start=q_start, iters=200)
        if best is None or err < best.error:
            best = Posture(q, pos, np.asarray(trunk_quat), float(err))
        if best.reached:
            break
    return best
