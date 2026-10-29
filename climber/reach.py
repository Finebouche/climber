"""An arm that moves to a point: the robot hangs on a wall's start holds, one hand leaves its hold (up out of the
socket, then away from the wall), goes to the point, comes back and hooks again; the three other limbs stay.  The IK
gives the joint targets every 20 ms, along straight segments, from the trunk's measured pose.

    uv run mjpython -m climber.reach                                # N1, right hand, 3 cm right, 6 cm from the wall
    uv run mjpython -m climber.reach --limb left_hand --offset -0.03 -0.06 0.0
    uv run python -m climber.reach --no-view                        # without a window: prints how it went

The offset is from the hand's ball in its hold, world frame (m): x right, y into the wall, z up.  Hanging, the arms are
already stretched upwards: a point more than about 2 cm above the hold is out of reach (checked before moving).  On N1
both hands come back into their hold; on N2 (start holds 15 mm higher, the arms more stretched) the right hand does,
the left one stops about 2 cm short (2 cm above the socket at the wall is at the edge of its reach).
"""
import argparse

import numpy as np

from .common import LIMBS, make_sim

LIFT = np.array([0.0, 0.0, 0.02])            # up out of the socket (a ball leaves it almost vertically)
OUT = np.array([0.0, -0.04, 0.0])            # then away from the wall
SEGMENT_S = 0.8                              # s per straight segment
PAUSE_S = 0.3                                # s held at each waypoint
REACH_TOL = 0.005                            # m, IK error accepted for the point


def move_ball(sim, limb, path, q, viewer=None):
    """Move the limb's ball along the waypoints; the last joint targets, and per waypoint where the ball got to."""
    n, reached = int(round(SEGMENT_S / sim.control_dt)), []
    for a, b in zip(path[:-1], path[1:]):
        for t in range(1, n + 1):
            q, _ = sim.joint_angles_for({limb: a + t / n * (b - a)}, *sim.trunk_pose(), q_start=q, iters=50)
            sim.step(q)
        for _ in range(int(round(PAUSE_S / sim.control_dt))):
            sim.step(q)
        reached.append(sim.ball_positions()[limb])
        if viewer is not None and not viewer.is_running():
            break
    return q, reached


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall", default="N1")
    parser.add_argument("--limb", default="right_hand", choices=LIMBS[:2])
    parser.add_argument("--offset", type=float, nargs=3, default=[0.03, -0.06, 0.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--no-view", action="store_true", help="no window")
    args = parser.parse_args()

    sim = make_sim(args.wall)
    sim.reset_hanging()
    hold = sim.hooked()[args.limb]
    p0 = sim.ball_positions()[args.limb]
    point = p0 + args.offset
    _, err = sim.joint_angles_for({args.limb: point}, *sim.trunk_pose(), iters=200)
    if err > REACH_TOL:
        raise SystemExit(f"{np.round(point, 3)} is out of reach of the {args.limb} ({1000 * err:.0f} mm short): "
                         f"try a smaller offset")
    viewer = None if args.no_view else sim.view()        # every step() now shows in the viewer, at real time
    out = [p0, p0 + LIFT, p0 + LIFT + OUT, point]
    q, there = move_ball(sim, args.limb, out, sim.joint_positions(), viewer)
    print(f"{args.limb} at the point: {1000 * np.linalg.norm(there[-1] - point):.1f} mm from it; hooked {sim.hooked()}")
    if viewer is None or viewer.is_running():
        back = [there[-1], p0 + LIFT + OUT, p0 + LIFT, p0 + [0.0, 0.0, 0.001]]
        q, again = move_ball(sim, args.limb, back, q, viewer)
        print(f"back in its hold: {sim.hooked()[args.limb] == hold} "
              f"({1000 * np.linalg.norm(again[-1] - p0):.1f} mm from the socket); hooked {sim.hooked()}")
    if viewer is not None:
        while viewer.is_running():                         # keep the window until it is closed
            sim.step(q)
    sim.close()
