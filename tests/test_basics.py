"""Quick checks of the parts that need no extra: robot, ik, planning, identification, whole_body, moves."""
import json
import sys

import numpy as np
import pytest
from climbing_sim import Wall, load_moves, run_move

from climber.common import LIMBS, MOVES, make_sim
from climber.identification import compare, record, tracking_error
from climber.ik import stance_posture
from climber.planning import as_dict, plan, plausible, start_stance
from climber.robot import reached_top, top_holds, training_walls
from climber.whole_body import holding_currents


@pytest.fixture(scope="module")
def sim():
    s = make_sim("N2")
    s.reset_hanging()
    return s


def test_walls():
    wall = Wall.load("N2")
    tops = top_holds(wall)
    assert tops and all(wall.holds[i].row == max(h.row for h in wall.holds) for i in tops)
    assert all(not w.name.endswith("seed0") for w in training_walls(seeds=range(0, 2)))


def test_start(sim):
    assert all(i is not None for i in sim.hooked().values())
    assert not reached_top(sim)


def test_start_stance_reached(sim):
    stance = {l: sim.wall.holds.index(sim.wall.start[l]) for l in LIMBS}
    assert stance_posture(sim, stance).reached


@pytest.mark.parametrize("wall", ["N1", "N2", "N3"])
def test_geometric_plan(wall):
    s = make_sim(wall)
    moves = plan(s)
    assert moves
    sockets, prev = s.wall.sockets(), start_stance(s.wall)
    for limb, hold, stance in moves:
        assert sum(a != b for a, b in zip(prev, stance)) == 1 and as_dict(stance)[limb] == hold
        assert plausible(sockets, stance)
        prev = stance
    assert {prev[0], prev[1]} & set(top_holds(s.wall))


def test_identification_log(sim):
    state = sim.get_state()
    targets = np.tile(sim.joint_positions(), (10, 1))
    log = record(sim, targets)
    sim.set_state(state)
    assert log["q"].shape == (10, 19) and len(log["time"]) == 10
    assert np.allclose(compare(log, log)["q_rms"], 0) and tracking_error(log).shape == (19,)


def test_holding_currents(sim):
    state = sim.get_state()
    assert holding_currents(sim, 0.1).shape == (19,)
    sim.set_state(state)


def test_replay_move(tmp_path):
    s = make_sim("N2")
    q0 = s.reset_hanging()["q"]
    (move,) = load_moves(str(MOVES / "replay.py")).values()
    module = sys.modules[type(move).__module__]      # load_moves imported the file as a module
    module.TRAJECTORY = tmp_path / "trajectory.json"
    module.TRAJECTORY.write_text(json.dumps({"targets": np.tile(q0, (5, 1)).tolist()}))
    run_move(move, s, 0.2)
    assert move.index >= 5
