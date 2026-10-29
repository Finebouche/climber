"""Limb transfers in the simulator: one ball out of its hold and into the next, the three others staying.

First the trunk goes to the pose of the next stance's posture (placo's IK, the trunk free), the four balls in their
holds; then the ball follows a list of waypoints (up out of the socket, away from the wall, over, in, down into the
next socket), the three others held in theirs.  Every 20 ms placo's IK, started from its previous solution, gives
the joint targets.  To improve: an RRT / RRT-Connect in joint space against collisions, balance (the centre of
mass over the support), feedback from the IMU and the joint currents.
"""
import numpy as np

from ..ik import ClimberIK, ball_targets, stance_posture

LIFT_UP = 0.02                               # m: a ball leaves a socket almost vertically
LIFT_OUT = 0.03                              # m: then away from the wall
STEPS_PER_SEGMENT = 15                       # control steps (20 ms) per waypoint segment
TRUNK_STEPS = 30
SETTLE_STEPS = 25
IK_ITERS = 5                                 # placo iterations per control step (warm-started)
TRUNK_WEIGHT = 0.01                          # the trunk's pull towards the next posture (the balls: 10)


def ball_path(p0, p1) -> list:
    """Waypoints of a ball from socket p0 to socket p1 (world frame; y < 0 is away from the wall)."""
    up, out = np.array([0.0, 0.0, LIFT_UP]), np.array([0.0, -LIFT_OUT, 0.0])
    return [p0, p0 + up, p0 + up + out, p1 + up + out, p1 + up, p1]


def slerp(q0, q1, a):
    """Quaternion (w x y z) a of the way from q0 to q1."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if q0 @ q1 < 0:
        q1 = -q1
    angle = np.arccos(np.clip(q0 @ q1, -1.0, 1.0))
    if angle < 1e-6:
        return q1
    return (np.sin((1 - a) * angle) * q0 + np.sin(a * angle) * q1) / np.sin(angle)


def transfer_frames(sim, ik, stance: dict, limb: str, hold: int, goal):
    """The configurations of a transfer, one per control step, from placo's current state (``ik``) to the posture
    ``goal`` of the next stance: (q, trunk_pos, trunk_quat, ball error left in m).  The settling pauses repeat the
    last one."""
    balls = ball_targets(sim, stance)
    _, trunk_pos, trunk_quat = ik.state()
    for s in range(1, TRUNK_STEPS + 1):                     # the trunk, on four holds
        a = s / TRUNK_STEPS
        trunk = (trunk_pos + a * (goal.trunk_pos - trunk_pos), slerp(trunk_quat, goal.trunk_quat, a))
        error = ik.solve(balls, trunk=trunk, trunk_weight=TRUNK_WEIGHT, iters=IK_ITERS)
        yield (*ik.state(), error)
    yield from [(*ik.state(), error)] * SETTLE_STEPS
    path = ball_path(balls[limb], ball_targets(sim, {limb: hold})[limb])
    trunk = (goal.trunk_pos, goal.trunk_quat)
    for k in range(len(path) - 1):                          # the ball, on the three other holds
        for s in range(1, STEPS_PER_SEGMENT + 1):
            balls[limb] = path[k] + s / STEPS_PER_SEGMENT * (path[k + 1] - path[k])
            error = ik.solve(balls, trunk=trunk, trunk_weight=TRUNK_WEIGHT, iters=IK_ITERS)
            yield (*ik.state(), error)
    yield from [(*ik.state(), error)] * SETTLE_STEPS


def transfer(sim, stance: dict, limb: str, hold: int, record: list | None = None) -> bool:
    """Move ``limb`` from its hold in ``stance`` to ``hold``.  True when its ball ends in that hold and the others
    are still in theirs.  ``record``: the joint targets sent are appended to it."""
    goal = stance_posture(sim, {**stance, limb: hold})
    if not goal.reached:
        return False
    ik = ClimberIK.of(sim)
    ik.set_state(sim.joint_positions(), *sim.trunk_pose())
    for q, *_ in transfer_frames(sim, ik, stance, limb, hold, goal):
        sim.step(q)
        if record is not None:
            record.append(q.copy())
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
