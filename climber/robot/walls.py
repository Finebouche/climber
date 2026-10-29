"""The walls: competition walls, generated routes for training, the top of a wall."""
import numpy as np
from climbing_sim import Wall
from climbing_sim.routes import generate_wall

from ..common import output_dir

COMPETITION_WALLS = ("N1", "N2", "N3")       # the rulebook's levels 1-3 (= generate_wall(level, seed=0))


def training_walls(levels=(1, 2, 3), seeds=range(1, 9)) -> list[Wall]:
    """Generated routes (seed 0 left out: it is the competition wall of each level)."""
    return [generate_wall(level, seed) for level in levels for seed in seeds if seed != 0]


def top_holds(wall: Wall) -> list[int]:
    """Indices (in ``wall.holds``) of the wall's highest holds: touching one of them is reaching the top."""
    rows = np.array([h.row for h in wall.holds])
    return [int(i) for i in np.flatnonzero(rows == rows.max())]


def reached_top(sim, tol: float = 0.02) -> bool:
    """True when a ball is within ``tol`` m of the socket of one of the highest holds (ground truth, sim only).
    Approximates the rule ("touching" a top hold); check the rulebook's exact wording before relying on it."""
    tops = sim.wall.sockets()[top_holds(sim.wall)]
    return any(np.linalg.norm(tops - p, axis=1).min() < tol for p in sim.ball_positions().values())


def save_wall(wall: Wall, name: str | None = None):
    """Write the wall to outputs/walls/<name>.json (open it in climbing-sim-wall-editor)."""
    path = output_dir("walls") / f"{name or wall.name}.json"
    wall.to_json(path)
    return path
