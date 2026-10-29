"""The most loaded servos of the robot hanging on N2's start holds, at the planning and the learning gains."""
from ..common import make_sim
from .loads import holding_currents, most_loaded

for learning in (False, True):
    sim = make_sim("N2", learning=learning)
    sim.reset_hanging()
    currents = holding_currents(sim, 2.0)
    loads = ", ".join(f"{name} {a:.2f} A" for name, a in most_loaded(currents))
    print(f"kp {sim.servos.kp:.0f}: total {currents.sum():.2f} A | {loads}")
