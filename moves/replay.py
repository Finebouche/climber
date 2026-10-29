"""Replay joint targets computed offline: the planner's motion (python -m climber.planning N2 --execute), on the robot.

A move of microban (branch dev_v2): the same file runs in the simulator, in the robot's program and on the robot.
It reads ``trajectory.json`` next to it, a JSON with ``targets`` (one list of 19 angles per 20 ms step, order JOINTS):
the planner's output, ``outputs/planning/<wall>_plan.json``, copied there.

    cp outputs/planning/N2_plan.json moves/trajectory.json
    cp moves/replay.py moves/trajectory.json ../microban/src/moves/       # then, in microban/:
    make sim WALL=N2                                                    # dashboard: Moves -> replay -> Start
    make deploy HOST=radxa@microban-xxxx.local                          # on the robot

Open loop: what the robot does differs from the simulator (identification: climber/identification).
"""
import json
from pathlib import Path

import numpy as np

TRAJECTORY = Path(__file__).with_name("trajectory.json")


class ReplayMove:
    """Replay a joint trajectory computed offline (trajectory.json)."""

    variants = ("climber",)                  # the climbing robot only: it starts hanging on the wall
    options = {"speed": 1.0}
    option_docs = {"speed": "playback speed (1: as computed, < 1 slower)"}

    def start(self, obs):
        data = json.loads(TRAJECTORY.read_text()) if TRAJECTORY.exists() else {}
        self.targets = np.asarray(data.get("targets", []), float).reshape(-1, len(obs["q"]))
        self.index = 0.0
        if len(self.targets) == 0:
            print(f"replay: no targets in {TRAJECTORY}")

    def step(self, obs):
        i = int(self.index)
        if i >= len(self.targets):
            return None                      # done: the last targets stay
        self.index += self.options["speed"]
        return self.targets[i]
