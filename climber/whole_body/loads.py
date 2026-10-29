"""The servos' loads: which joints a stance or a motion asks the most of (motor currents, as the robot measures them).

A climbing robot hangs on its servos: a posture that needs too much current heats them and sags (a joint under
load stops short of its target).  The next step: plan the motions with the dynamics (TSID: torques under contact
constraints; Crocoddyl: optimal control of a limb transfer), the balls in the holds as contacts.
"""
import numpy as np

from ..common import JOINTS

MAX_CURRENT = 1.75                           # A, the XL330 firmware's current limit (as in microban_sim)


def holding_currents(sim, seconds: float = 1.0) -> np.ndarray:
    """Mean |current| (A) of each joint while the servos hold the current targets for ``seconds``."""
    target = sim.joint_positions()
    currents = [np.abs(sim.step(target)["current"]) for _ in range(int(round(seconds / sim.control_dt)))]
    return np.nanmean(currents, axis=0)


def most_loaded(currents: np.ndarray, n: int = 5) -> list[tuple[str, float]]:
    """The n joints with the highest currents: (name, A); compare with MAX_CURRENT."""
    order = np.argsort(currents)[::-1][:n]
    return [(JOINTS[i], float(currents[i])) for i in order]
