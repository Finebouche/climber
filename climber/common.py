"""Shared parts: paths, the simulator with the project's defaults, the output folders."""
from pathlib import Path

from climbing_sim import JOINTS, LIMBS, ClimbingSim

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
MOVES = ROOT / "moves"                       # our moves, copied to microban/src/moves to run on the robot
MICROBAN = ROOT.parent / "microban"          # the robot's program (Rhoban/microban, branch dev_v2)

KP_PLANNING = 400.0                          # the XL330's factory gain: planning, kinematics
KP_LEARNING = 125.0                          # the gain the robot runs learned policies at

__all__ = ["JOINTS", "LIMBS", "ROOT", "OUTPUTS", "MOVES", "MICROBAN", "KP_PLANNING", "KP_LEARNING", "make_sim",
           "output_dir"]


def make_sim(wall="N2", learning: bool = False, cameras=(), **kwargs) -> ClimbingSim:
    """A ClimbingSim on ``wall`` (name, JSON file or Wall), without cameras by default (faster; None: the two
    default cameras), at the servo gain of planning or of learning."""
    cameras = None if cameras is None else list(cameras)
    kwargs.setdefault("kp_fw", KP_LEARNING if learning else KP_PLANNING)
    return ClimbingSim(wall, cameras=cameras, **kwargs)


def output_dir(*parts: str) -> Path:
    """outputs/<parts...>, created if needed."""
    d = OUTPUTS.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
