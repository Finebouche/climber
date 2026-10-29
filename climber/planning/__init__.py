"""Planning: which holds, in which order (stances), then how to move a limb from one hold to the next.

Tools: Dijkstra, A* (over stances, ``stances.py``), RRT, RRT-Connect (limb motions in joint space, to come; the
first version of ``motion.py`` follows straight waypoints with the IK).

    python -m climber.planning [N2] [--ik] [--execute]   # plan to the top, check it with the IK, play it in the sim
"""
from .motion import ball_path, execute, transfer
from .stances import as_dict, neighbours, plan, plausible, start_stance

__all__ = ["as_dict", "ball_path", "execute", "neighbours", "plan", "plausible", "start_stance", "transfer"]
