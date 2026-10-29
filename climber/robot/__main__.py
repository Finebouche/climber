"""The competition walls and a few training routes: holds, start stance, height of the top."""
from climbing_sim import LIMBS, Wall

from .walls import COMPETITION_WALLS, top_holds, training_walls

for wall in [Wall.load(n) for n in COMPETITION_WALLS] + training_walls(levels=(2, 3), seeds=range(1, 3)):
    tops = top_holds(wall)
    z_top = wall.holds[tops[0]].socket[2]
    start = ", ".join(f"{l} {h.col},{h.row}" for l, h in ((l, wall.start[l]) for l in LIMBS))
    print(f"{wall.name:>8}: {len(wall.holds):3d} holds, top at {z_top:.3f} m ({len(tops)} holds) | start {start}")
