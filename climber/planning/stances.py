"""Hold sequences: A* (or Dijkstra) over stances, one limb moving at a time.

A stance is a tuple of four hold indices (in ``wall.holds``), in the order of LIMBS.  A move takes one limb to
another free hold; a stance is kept if it passes the geometric filter (cheap), and, with an IK check, if
``stance_posture`` (placo, the trunk free, the body off the wall) reaches it: checked when the stance is expanded,
cached.  The goal: a hand on one of the wall's
highest holds.
"""
import heapq
import itertools

import numpy as np

from ..common import LIMBS
from ..ik import stance_posture
from ..robot import top_holds

LH, RH, LF, RF = range(4)
MAX_STEP = 0.11                              # m, longest limb move (the routes never need more than 105 mm)
HANDS_ABOVE_FEET = (0.20, 0.40)              # m, lowest hand - highest foot (start stance: 0.315)
HAND_SPREAD = (0.08, 0.25)                   # m, right hand x - left hand x (start: 0.165)
FOOT_SPREAD = (0.03, 0.15)                   # m, right foot x - left foot x (start: 0.075)
MAX_HAND_DZ = 0.11                           # m, height difference between the hands
MAX_FOOT_DZ = 0.11                           # m, between the feet
MAX_DROP = 0.005                             # m: a limb never moves down more than this (climbing goes up)


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
    """(limb index, new stance) for every plausible move of one limb (not downwards)."""
    for k, i in enumerate(stance):
        near = np.flatnonzero((np.linalg.norm(sockets - sockets[i], axis=1) <= MAX_STEP)
                              & (sockets[:, 2] >= sockets[i, 2] - MAX_DROP))
        for j in near:
            if j not in stance:
                new = stance[:k] + (int(j),) + stance[k + 1:]
                if plausible(sockets, new):
                    yield k, new


def plan(sim, start: tuple | None = None, ik: bool = False, astar: bool = True, greed: float = 1.0,
         max_expansions: int = 50_000):
    """The shortest hold sequence (fewest moves) to the top: a list of (limb, hold index, stance), or None.
    ``ik``: keep only stances the IK reaches with the body off the wall, each solved from the posture of the stance
    it is reached from (call it on the robot hanging at the start).  ``astar=False``: Dijkstra.  ``greed`` > 1:
    weighted A* (the heuristic times greed), much faster, plans at most greed times longer."""
    wall = sim.wall
    sockets = wall.sockets()
    start = start or start_stance(wall)
    tops = set(top_holds(wall))
    z_top = sockets[list(tops), 2].max()

    # at the goal a hand is at z_top, so the other is at least MAX_HAND_DZ lower, a foot at most HANDS_ABOVE_FEET
    # lower than it, the other foot MAX_FOOT_DZ lower; a move raises one limb by MAX_STEP at most: admissible
    need = np.array([z_top, z_top - MAX_HAND_DZ])
    need_feet = need[1] - HANDS_ABOVE_FEET[1] - np.array([0.0, MAX_FOOT_DZ])

    def h(s):
        if not astar:
            return 0.0
        z = sockets[list(s), 2]
        hands, feet = np.sort(z[[LH, RH]])[::-1], np.sort(z[[LF, RF]])[::-1]
        return float(np.ceil(np.clip(need - hands, 0, None) / MAX_STEP).sum()
                     + np.ceil(np.clip(need_feet - feet, 0, None) / MAX_STEP).sum())

    postures = {start: stance_posture(sim, as_dict(start))} if ik else {}
    tie = itertools.count()
    frontier = [(greed * h(start), next(tie), 0, start)]
    parent = {start: None}
    cost = {start: 0}
    for _ in range(max_expansions):
        if not frontier:
            return None
        _, _, g, s = heapq.heappop(frontier)
        if g > cost[s]:
            continue
        if ik and s != start:
            if s not in postures:            # from the posture it comes from, the trunk moved with the balls' centroid
                k, prev = parent[s]
                p = postures[prev]
                shift = (sockets[s[k]] - sockets[prev[k]]) / len(LIMBS)
                postures[s] = stance_posture(sim, as_dict(s), q_start=p.q, trunk_start=(p.trunk_pos + shift, p.trunk_quat))
            if not postures[s].reached:
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
                heapq.heappush(frontier, (g + 1 + greed * h(new), next(tie), g + 1, new))
    return None
