"""Plan a hold sequence to the top, optionally checked with the IK, optionally played in the simulator.  Saves the
plan (and the joint targets played) to outputs/planning/<wall>_plan.json: moves/replay.py replays them on the robot.

    python -m climber.planning [N2] [--ik] [--execute]
"""
import argparse
import json
import time

from ..common import make_sim, output_dir
from .motion import execute
from .stances import plan

parser = argparse.ArgumentParser()
parser.add_argument("wall", nargs="?", default="N2")
parser.add_argument("--ik", action="store_true", help="keep only stances the IK reaches")
parser.add_argument("--max-expansions", type=int, default=50_000)
parser.add_argument("--greed", type=float, default=1.0, help="weighted A*: > 1 is faster, the plan longer")
parser.add_argument("--execute", action="store_true", help="play the plan in the simulator")
args = parser.parse_args()

sim = make_sim(args.wall)
sim.reset_hanging()
t = time.time()
moves = plan(sim, ik=args.ik, greed=args.greed, max_expansions=args.max_expansions)
if moves is None:
    raise SystemExit(f"no plan to the top of {args.wall}")
print(f"{len(moves)} moves ({time.time() - t:.1f} s):")
holds = sim.wall.holds
for limb, i, _ in moves:
    print(f"  {limb:>10} -> {holds[i].col},{holds[i].row}  (z {holds[i].socket[2]:.3f} m)")

out = {"wall": args.wall, "moves": [{"limb": l, "col": holds[i].col, "row": holds[i].row} for l, i, _ in moves]}
if args.execute:
    targets = []
    done = execute(sim, moves, record=targets)
    print(f"played {done} / {len(moves)} moves, {len(targets) * sim.control_dt:.1f} s; hooked {sim.hooked()}")
    out.update(played=done, control_dt=sim.control_dt, targets=[q.round(5).tolist() for q in targets])
path = output_dir("planning") / f"{args.wall}_plan.json"
path.write_text(json.dumps(out))
print(path)
