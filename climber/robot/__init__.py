"""Robot & CAD: the robot, the walls and the goal of the competition.

Tools: Microban (the robot, Rhoban/microban, branch dev_v2), Onshape and onshape-to-robot (its CAD -> URDF / MJCF,
already done: ``microban_model``).  Here: the competition walls N1-N3, more routes for training (seed 0 of each level
is the competition wall: keep it for evaluation), and the rule's goal (touch one of the wall's highest holds).

    python -m climber.robot            # the walls, their holds, their top
"""
from .walls import COMPETITION_WALLS, reached_top, save_wall, top_holds, training_walls

__all__ = ["COMPETITION_WALLS", "reached_top", "save_wall", "top_holds", "training_walls"]
