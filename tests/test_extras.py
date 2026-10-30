"""Quick checks of the parts that need an extra (skipped without it): modelling, perception, learning, the policy
and reach moves."""
import inspect
import sys

import numpy as np
import pytest
from climbing_sim import load_moves, run_move

from climber.common import MOVES, make_sim


def test_pinocchio_matches_simulator():
    pytest.importorskip("pinocchio")
    from climber.modelling import RobotModel
    sim = make_sim("N2")
    sim.reset_hanging()
    robot = RobotModel()
    balls = robot.ball_positions(robot.configuration(sim.joint_positions(), *sim.trunk_pose()))
    for limb, p in sim.ball_positions().items():
        assert np.linalg.norm(balls[limb] - p) < 1e-3


def test_camera_pose_from_markers():
    pytest.importorskip("cv2")
    from climber.perception import camera_pose
    sim = make_sim("N2", cameras=None)
    sim.reset_hanging()
    for name, cam in sim.cameras.items():
        rvec, tvec, ids = camera_pose(sim.render(name), cam)
        assert ids == [1, 2, 3, 4]
        assert np.linalg.norm(tvec - cam.tvec) < 0.01
    sim.close()


def test_climb_env():
    pytest.importorskip("gymnasium")
    from climber.learning import ClimbEnv
    env = ClimbEnv(episode_s=0.1)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert truncated and np.isfinite(reward)


def test_policy_move_matches_env():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("gymnasium")
    from climber.learning import ClimbEnv
    (move,) = load_moves(str(MOVES / "policy.py")).values()
    module = sys.modules[type(move).__module__]
    env = ClimbEnv(episode_s=0.1)
    obs, _ = env.reset(seed=0)
    assert np.allclose(module.joint_limits(), env.sim.joint_range, atol=1e-6)
    assert np.array_equal(module.observe(env.sim.observation())[0], env._observe(env.sim.observation()))
    assert module.MAX_STEP == inspect.signature(ClimbEnv).parameters["max_step"].default


def test_reach_move():
    pytest.importorskip("placo")
    sim = make_sim("N1")
    sim.reset_hanging()
    hooked = sim.hooked()
    (move,) = load_moves(str(MOVES / "reach.py")).values()
    run_move(move, sim, 8.0)
    assert move.reachable and sim.hooked() == hooked


def test_map_and_robot_pose():
    pytest.importorskip("cv2")
    from climber.perception import RobotMarker, compare, hold_shape, map_holds, marked_sim, robot_pose
    sim = marked_sim("N1")
    sim.reset(trunk_pos=[-0.8, -0.6, 0.3])          # the robot aside: the empty wall
    images = {n: sim.render(n) for n in sim.cameras}
    r = compare(map_holds(images, sim.cameras, hold_shape(sim.model, sim.data)), sim.wall)
    assert r["exact"] == r["total"] and not r["extra"]
    sim.reset_hanging()
    pos, quat, n = robot_pose({n: sim.render(n) for n in sim.cameras}, sim.cameras, RobotMarker())
    assert n == 2 and np.linalg.norm(pos - sim.trunk_pose()[0]) < 0.015
    sim.close()
