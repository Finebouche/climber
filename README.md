<p align="center"><h1 align="center">climber</h1></p>
<p align="center">
	<em>Our climbing robot for the hackathon competition</em>
</p>
<br>

## 🔗 Table of Contents

- [📍 Overview](#-overview)
- [🗂 Repository structure](#-repository-structure)
- [🚀 Setup](#-setup)
- [▶️ Usage](#️-usage)

---

## 📍 Overview

Make Microban, in its climbing version (ball hands, climbing feet, 19 XL330 servos), climb the competition wall
(levels N1-N3 of the rulebook) until it touches one of the highest holds. The code is organised as the toolbox:
one subpackage per field (robot, modelling, inverse kinematics, planning, learning, identification, whole-body
dynamics, plus perception for the rulebook's cameras). Each subpackage has a quick example
(`python -m climber.<part>`). The moves in `moves/` are what runs on the robot.

It builds on two repositories cloned next to this one:

- [rock-climbing-robots/simulator](https://github.com/rock-climbing-robots/simulator) (`../simulator`): the
  MuJoCo simulator of the robot on the wall (`climbing_sim`), its walls and cameras;
- [Rhoban/microban](https://github.com/Rhoban/microban/tree/dev_v2), branch `dev_v2` (`../microban`): the robot's
  program, where moves run (`make sim WALL=N2`, `make dashboard HOST=127.0.0.1`, `make deploy`);
- [Rhoban/mjlab_microban](https://github.com/Rhoban/mjlab_microban) (`../mjlab_microban`, optional): Rhoban's example
  of RL on a GPU with mjlab (a walking task), the model for a climbing task trained on a GPU.

---
## 🗂 Repository structure

| Folder | Content | Toolbox |
|:--|:--|:--|
| `climber/common.py` | Paths, `make_sim` (the simulator with the project's defaults), `output_dir` | |
| `climber/demo.py` | A first simulation to watch: the robot hanging sways its trunk, leans back, turns its head | |
| `climber/robot` | Competition and training walls, the highest holds, `reached_top` | Microban, Onshape, onshape-to-robot |
| `climber/modelling` | Pinocchio model of the climbing URDF with a floating base: balls, centre of mass | Pinocchio, MuJoCo |
| `climber/ik` | Joint angles of a stance (one hold per limb), with a search over the trunk's pose | pink, placo, mink |
| `climber/planning` | A* / Dijkstra over stances (`stances.py`), limb transfers in the simulator (`motion.py`) | Dijkstra, A*, RRT, RRT-Connect |
| `climber/learning` | Gymnasium environment (`env.py`), PPO and ONNX export (`train.py`), on the CPU; on a GPU: mjlab, as `../mjlab_microban` | Gymnasium, Stable Baselines3, mjlab |
| `climber/identification` | Logs of a motion in the simulator and on the robot, their comparison | BAM |
| `climber/perception` | Camera pose from the four ArUco corner markers | (rulebook) |
| `climber/whole_body` | Servo loads (currents) | TSID, Crocoddyl |
| `moves` | Moves for the robot: `replay.py` (a planned trajectory), `policy.py` (an ONNX policy) | |
| `outputs/<part>` | Results: plans, logs, policies, walls | |
| `tests` | Quick tests (`test_extras.py` skips what needs a missing extra) | |
| `ETAT_DES_TRAVAUX.md` | State of the work (in French): what was tried, what works, what is next | |
| `1. Exploration.ipynb` | The walls, a plan, the simulator | |

---
## 🚀 Setup

Next to `simulator/` and `microban/` (see the simulator's README for git-lfs and uv):

```bash
cd hackathon
git clone https://github.com/rock-climbing-robots/simulator
git clone -b dev_v2 https://github.com/Rhoban/microban
git clone https://github.com/Rhoban/mjlab_microban           # optional: RL on a GPU
cd climber
uv sync --all-extras --group dev
```

Extras: `modelling` (Pinocchio), `ik` (mink, placo: the robot's version), `learning` (Gymnasium, Stable Baselines3,
ONNX), `perception` (OpenCV), `notebooks`. TSID and Crocoddyl are not included: their pip wheels do not work with the
Pinocchio that placo pins; take them from conda-forge in a separate environment.

---
## ▶️ Usage

```bash
uv run mjpython -m climber.demo N2          # watch: the robot hanging sways and leans (Linux: python; --no-view)
uv run python -m climber.robot              # the walls
uv run python -m climber.ik N2              # which holds the limbs reach from the start
uv run python -m climber.planning N2        # a hold sequence to the top (--ik: checked with the IK, --execute: played)
uv run python -m climber.modelling          # Pinocchio against the simulator
uv run python -m climber.perception         # the cameras' pose from the markers
uv run python -m climber.identification     # a log in the simulator
uv run python -m climber.whole_body         # the most loaded servos
uv run python -m climber.learning           # random steps in the environment
uv run python -m climber.learning.train --wall N2 --steps 2000000 --envs 8    # training (long)
uv run pytest
```

On the robot's program (in `../microban`), a move of `moves/` with the file it reads:

```bash
cp outputs/planning/N2_plan.json moves/trajectory.json
cp moves/replay.py moves/trajectory.json ../microban/src/moves/
cd ../microban && make sim WALL=N2          # macOS: make sim PYTHON=mjpython WALL=N2; then make dashboard HOST=127.0.0.1
```
