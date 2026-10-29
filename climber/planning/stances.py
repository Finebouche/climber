"""Hold sequences: A* (or Dijkstra) over stances, one limb moving at a time.

A stance is a tuple of four hold indices (in ``wall.holds``), in the order of LIMBS.  A move takes one limb to
another free hold; a stance is kept if it passes the geometric filter (cheap), and, with an IK check, if
``stance_posture`` reaches it (checked when the stance is expanded, cached).  The goal: a hand on one of the wall's
highest holds.
"""
import heapq
import itertools

import numpy as np

from ..common import LIMBS
from ..ik.stance import COARSE_GRID, stance_posture
from ..robot import top_holds

LH, RH, LF, RF = range(4)
MAX_STEP = 0.11                              # m, longest limb move (the routes never need more than 105 mm)
HANDS_ABOVE_FEET = (0.20, 0.40)              # m, lowest hand - highest foot (start stance: 0.315)
HAND_SPREAD = (0.08, 0.25)                   # m, right hand x - left hand x (start: 0.165)
FOOT_SPREAD = (0.03, 0.15)                   # m, right foot x - left foot x (start: 0.075)
MAX_HAND_DZ = 0.11                           # m, height difference between the hands
MAX_FOOT_DZ = 0.11                           # m, between the feet


def start_stance(wall) -> tuple:
    return tuple(wall.holds.index(wall.start[l]) for l in LIMBS)


def as_dict(stance: tuple) -> dict:
    return dict(zip(LIMBS, stance))


def plausible(sockets: np.ndarray, stance: tuple) -> bool:
    """The geometric filter: limbs not crossed, hands above the feet, spreads and height differences within
    bounds."""
    p = sockets[list(stance)]
    gap = min(p[LH, 2], p[RH, 2]) - max(p[LF, 2], p[RF, 2])
    return (HAND_SPREAD[0] <= p[RH, 0] - p[LH, 0] <= HAND_SPREAD[1]
            and FOOT_SPREAD[0] <= p[RF, 0] - p[LF, 0] <= FOOT_SPREAD[1]
            and abs(p[LH, 2] - p[RH, 2]) <= MAX_HAND_DZ and abs(p[LF, 2] - p[RF, 2]) <= MAX_FOOT_DZ
            and HANDS_ABOVE_FEET[0] <= gap <= HANDS_ABOVE_FEET[1])


def neighbours(sockets: np.ndarray, stance: tuple):
    """(limb index, new stance) for every plausible move of one limb."""
    for k, i in enumerate(stance):
        near = np.flatnonzero(np.linalg.norm(sockets - sockets[i], axis=1) <= MAX_STEP)
        for j in near:
            if j not in stance:
                new = stance[:k] + (int(j),) + stance[k + 1:]
                if plausible(sockets, new):
                    yield k, new


def plan(sim, start: tuple | None = None, ik: bool = False, astar: bool = True, max_expansions: int = 50_000):
    """The shortest hold sequence (fewest moves) to the top: a list of (limb, hold index, stance), or None.
    ``ik``: keep only stances the IK reaches with the trunk on COARSE_GRID (from the simulation's current state: call
    it on the robot hanging at the start).  ``astar=False``: Dijkstra."""
    wall = sim.wall
    sockets = wall.sockets()
    start = start or start_stance(wall)
    tops = set(top_holds(wall))
    z_top = sockets[list(tops), 2].max()

    def h(s):                                # each move raises the highest hand by MAX_STEP at most: admissible
        return max(0.0, (z_top - max(sockets[s[LH], 2], sockets[s[RH], 2])) / MAX_STEP) if astar else 0.0

    feasible = {}
    tie = itertools.count()
    frontier = [(h(start), next(tie), 0, start)]
    parent = {start: None}
    cost = {start: 0}
    for _ in range(max_expansions):
        if not frontier:
            return None
        _, _, g, s = heapq.heappop(frontier)
        if g > cost[s]:
            continue
        if ik and s != start:
            if s not in feasible:
                feasible[s] = stance_posture(sim, as_dict(s), grid=COARSE_GRID).reached
            if not feasible[s]:
                continue
        if {s[LH], s[RH]} & tops:
            path = []
            while parent[s] is not None:
                k, prev = parent[s]
                path.append((LIMBS[k], s[k], s))
                s = prev
            return path[::-1]
        for k, new in neighbours(sockets, s):
            if g + 1 < cost.get(new, np.inf):
                cost[new] = g + 1
                parent[new] = (k, s)
                heapq.heappush(frontier, (g + 1 + h(new), next(tie), g + 1, new))
    return None
