"""Check Pinocchio's model against the simulator: the balls of the hanging robot, then its mass and centre of mass."""
import numpy as np

from ..common import make_sim
from .model import RobotModel

sim = make_sim("N2")
obs = sim.reset_hanging()
robot = RobotModel()
config = robot.configuration(sim.joint_positions(), *sim.trunk_pose())
pin_balls, sim_balls = robot.ball_positions(config), sim.ball_positions()
for limb in pin_balls:
    print(f"{limb:>10}: Pinocchio - simulator {1000 * np.linalg.norm(pin_balls[limb] - sim_balls[limb]):.2f} mm")
com = robot.center_of_mass(config)
print(f"mass {robot.mass:.3f} kg, centre of mass {com.round(3)} m, {1000 * -com[1]:.0f} mm from the wall")
