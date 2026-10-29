"""Limb transfers in the simulator: one ball out of its hold and into the next, the three others staying.

A first version: the trunk goes to the pose of the next stance's posture, then the ball follows a list of waypoints
(up out of the socket, away from the wall, over, in, down into the next socket), the IK giving the joint targets
every 20 ms.  To improve: an RRT / RRT-Connect in joint space against collisions, balance (the
centre of mass over the support), feedback from the IMU and the joint currents.
"""
import numpy as np

from ..ik import ball_targets, stance_posture

LIFT_UP = 0.02                               # m: a ball leaves a socket almost vertically
LIFT_OUT = 0.03                              # m: then away from the wall
STEPS_PER_SEGMENT = 15                       # control steps (20 ms) per waypoint segment
TRUNK_STEPS = 30
SETTLE_STEPS = 25


def ball_path(p0, p1) -> list:
    """Waypoints of a ball from socket p0 to socket p1 (world frame; y < 0 is away from the wall)."""
    up, out = np.array([0.0, 0.0, LIFT_UP]), np.array([0.0, -LIFT_OUT, 0.0])
    return [p0, p0 + up, p0 + up + out, p1 + up + out, p1 + up, p1]


def transfer(sim, stance: dict, limb: str, hold: int, record: list | None = None) -> bool:
    """Move ``limb`` from its hold in ``stance`` to ``hold``: first the trunk to the next stance's pose, the four balls
    in their holds, then the ball along its path, the IK done each step from the trunk's measured pose (ground
    truth here; on the robot: the IMU and the kinematics of the hooked limbs).  True when its ball ends in that
    hold and the others are still in theirs.  ``record``: the joint targets sent are appended to it."""
    goal = stance_posture(sim, {**stance, limb: hold})
    if not goal.reached:
        return False
    q = sim.joint_positions()

    def send(target, steps=1):
        for _ in range(steps):
            sim.step(target)
            if record is not None:
                record.append(target.copy())

    balls = ball_targets(sim, stance)
    trunk_pos, trunk_quat = sim.trunk_pose()
    for s in range(1, TRUNK_STEPS + 1):                     # the trunk, on four holds
        trunk = trunk_pos + s / TRUNK_STEPS * (goal.trunk_pos - trunk_pos)
        q, _ = sim.joint_angles_for(balls, trunk, trunk_quat, q_start=q, iters=50)
        send(q)
    send(q, SETTLE_STEPS)
    path = ball_path(balls[limb], ball_targets(sim, {limb: hold})[limb])
    for k in range(len(path) - 1):                          # the ball, on the three other holds
        for s in range(1, STEPS_PER_SEGMENT + 1):
            balls[limb] = path[k] + s / STEPS_PER_SEGMENT * (path[k + 1] - path[k])
            q, _ = sim.joint_angles_for(balls, *sim.trunk_pose(), q_start=q, iters=50)
            send(q)
    send(q, SETTLE_STEPS)
    hooked = sim.hooked()
    return all(hooked[l] == i for l, i in {**stance, limb: hold}.items())


def execute(sim, plan: list, record: list | None = None) -> int:
    """Play a plan of ``stances.plan`` from the current state; the number of moves that succeeded (stops at the
    first failure)."""
    stance = {l: i for l, i in sim.hooked().items()}
    for done, (limb, hold, _) in enumerate(plan):
        if None in stance.values() or not transfer(sim, stance, limb, hold, record):
            return done
        stance[limb] = hold
    return len(plan)
