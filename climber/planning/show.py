"""Watch a plan, without physics: the postures of placo's IK, one per control step, from the start stance to the top
(the trunk moves to the next stance's posture, then the limb goes over to its next hold), shown in MuJoCo's viewer.
It checks the plan's geometry (the IK reaches every stance, the body stays off the wall), not that the robot can do
it: the servos, gravity and contacts are not simulated (``python -m climber.planning --execute`` does that).

    uv run mjpython -m climber.planning.show N2            # the plan of outputs/planning/N2_plan.json
    uv run mjpython -m climber.planning.show N2 --speed 3  # 3 times faster
    uv run python -m climber.planning.show N2 --no-view    # without a window: checks every posture

Without a plan file, it plans first (``plan(ik=True, greed=3)``, about half a minute).
"""
import argparse
import json
import time

import mujoco
import numpy as np

from ..common import OUTPUTS, make_sim
from ..ik import REACH_TOL, ClimberIK, stance_posture
from .motion import transfer_frames
from .stances import as_dict, plan, start_stance


def load_plan(sim, wall_name: str):
    """The moves of outputs/planning/<wall>_plan.json as (limb, hold index), or a new plan checked by the IK."""
    path = OUTPUTS / "planning" / f"{wall_name}_plan.json"
    if path.exists():
        index = {(h.col, h.row): i for i, h in enumerate(sim.wall.holds)}
        return [(m["limb"], index[m["col"], m["row"]]) for m in json.loads(path.read_text())["moves"]]
    print(f"no {path}: planning (about half a minute)...")
    return [(limb, hold) for limb, hold, _ in plan(sim, ik=True, greed=3.0)]


def frames(sim, moves):
    """(move number, q, trunk_pos, trunk_quat, ball error) per control step of the whole plan, and the postures of
    the stances (from the start's).  The simulator must hang at the start (``reset_hanging``): its posture is the
    first guess."""
    ik = ClimberIK.of(sim)
    stance = as_dict(start_stance(sim.wall))
    posture = stance_posture(sim, stance)
    postures = [posture]
    ik.set_state(posture.q, posture.trunk_pos, posture.trunk_quat)
    out = [(0, posture.q, posture.trunk_pos, posture.trunk_quat, posture.error)]
    for n, (limb, hold) in enumerate(moves, 1):
        p = postures[-1]
        shift = (sim.wall.holds[hold].socket - sim.wall.holds[stance[limb]].socket) / len(stance)
        goal = stance_posture(sim, {**stance, limb: hold}, q_start=p.q, trunk_start=(p.trunk_pos + shift, p.trunk_quat))
        postures.append(goal)
        out += [(n, *f) for f in transfer_frames(sim, ik, stance, limb, hold, goal)]
        stance[limb] = hold
    return out, postures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wall", nargs="?", default="N2")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (1: as on the robot, 50 Hz)")
    parser.add_argument("--no-view", action="store_true", help="no window: check the plan's postures")
    args = parser.parse_args()

    sim = make_sim(args.wall)
    sim.reset_hanging()
    moves = load_plan(sim, args.wall)
    t = time.time()
    steps, postures = frames(sim, moves)
    reached = sum(p.reached for p in postures[1:])
    errors = np.array([f[4] for f in steps])
    worst = min(postures, key=lambda p: p.clearance)
    print(f"{args.wall}: {len(moves)} moves, {len(steps)} steps ({len(steps) * sim.control_dt:.0f} s at 50 Hz), "
          f"computed in {time.time() - t:.0f} s")
    print(f"stances reached by the IK: {reached} / {len(moves)}; closest to the wall: {1000 * worst.clearance:.1f} mm; "
          f"balls off their targets along the way: {np.mean(errors > REACH_TOL):.0%} of the steps "
          f"(max {1000 * errors.max():.1f} mm)")
    print(f"trunk from z {postures[0].trunk_pos[2]:.3f} to {postures[-1].trunk_pos[2]:.3f} m")
    if not args.no_view:
        import mujoco.viewer
        ik = ClimberIK.of(sim)
        data = mujoco.MjData(sim.model)
        with mujoco.viewer.launch_passive(sim.model, data) as viewer:
            while viewer.is_running():
                for n, q, pos, quat, _ in steps:
                    t0 = time.time()
                    data.qpos[:3], data.qpos[3:7], data.qpos[ik.qadr] = pos, quat, q
                    mujoco.mj_forward(sim.model, data)
                    viewer.sync()
                    if not viewer.is_running():
                        break
                    time.sleep(max(0.0, sim.control_dt / args.speed - (time.time() - t0)))
                time.sleep(1.0)                              # then again from the start
    sim.close()
