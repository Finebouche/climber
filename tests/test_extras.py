"""Quick checks of the parts that need an extra (skipped without it): modelling, perception, learning, the policy
move."""
import inspect
import sys

import numpy as np
import pytest
from climbing_sim import load_moves

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
