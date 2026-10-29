"""Modelling: the robot's kinematics and dynamics, outside the simulation.

Tools: Pinocchio (``model.py``, on the URDF of microban_model), MuJoCo (the simulator's model: ``sim.model``,
``sim.ball_positions_at``).  Needs the ``modelling`` extra.

    python -m climber.modelling         # Pinocchio's balls against the simulator's, the mass, the centre of mass
"""
from .model import RobotModel

__all__ = ["RobotModel"]
