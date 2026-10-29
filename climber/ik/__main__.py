"""IK of the start stance, then of each stance with one limb moved to a hold 2-11 cm higher."""
import sys

from ..common import LIMBS, make_sim
from .stance import stance_posture

sim = make_sim(sys.argv[1] if len(sys.argv) > 1 else "N2")
sim.reset_hanging()
holds, sockets = sim.wall.holds, sim.wall.sockets()
start = {l: holds.index(sim.wall.start[l]) for l in LIMBS}
p = stance_posture(sim, start)
print(f"start stance: error {1000 * p.error:.2f} mm, reached {p.reached}")
for limb in LIMBS:
    s = sockets[start[limb]]
    for i in range(len(holds)):
        if 0.02 < sockets[i, 2] - s[2] < 0.11 and abs(sockets[i, 0] - s[0]) < 0.06:
            p = stance_posture(sim, {**start, limb: i})
            moved = (1000 * (p.trunk_pos - sim.trunk_pose()[0])).round()
            print(f"  {limb:>10} -> hold {holds[i].col},{holds[i].row}: error {1000 * p.error:6.2f} mm, "
                  f"reached {p.reached}, trunk moved {moved} mm")
