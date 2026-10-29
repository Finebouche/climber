"""Identification: make the simulator match the robot (and know where it does not).

Tools: BAM (Rhoban's servo identification: the servos of microban_sim are simulated by it), logs of the same motion
in the simulator and on the robot (``logs.py``).

    python -m climber.identification     # a knee sine hanging on the wall, its log, the servos' tracking error
"""
from .logs import compare, load, record, save, tracking_error

__all__ = ["compare", "load", "record", "save", "tracking_error"]
