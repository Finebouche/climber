"""Learning: a climbing policy by reinforcement learning, then exported to ONNX for a move on the robot.

Two paths:

* here, on the CPU: Gymnasium (``env.py``, on ClimbingSim: the robot, its BAM servos, the wall) and Stable
  Baselines3 (PPO, ``train.py``), several processes; ``moves/policy.py`` runs the ONNX on the robot.  Needs the
  ``learning`` extra;
* on a GPU, thousands of robots at once: mjlab, as Rhoban's example ``../mjlab_microban`` (a walking task,
  ``Mjlab-Velocity-Microban``, its own uv environment).  A climbing task there = its velocity task's structure
  (``tasks/``, registered by an entry point) with the climbing robot and the wall (``climbing_sim.build_spec``).
  Its policies run on the robot as microban's ``src/moves/walk.py`` does (ONNX with ``joint_names`` and
  ``default_joint_pos`` in its metadata, ``OBSERVATION_DOF_ORDER``).

    python -m climber.learning          # a few random steps in the environment (quick check)
"""
from .env import ClimbEnv

__all__ = ["ClimbEnv"]
