"""Reach: one hand leaves its hold, goes to a point and comes back into it, the robot hanging on the wall.

A move of microban (branch dev_v2): the same file runs in the simulator, in the robot's program and on the robot.
Only the arm's three joints are driven (the other limbs keep their targets: they stay in their holds); their angles
come from placo's IK on the climbing URDF, in the trunk frame.  The trunk moves a little meanwhile: the three hooked
balls (their measured joint angles) show how, and the path follows.  Up is the gravity the IMU measures, "out" is
away from the wall (the trunk's -x: its front faces the wall when it climbs).

    cd ../microban && make sim PYTHON=mjpython WALL=N1 MOVES=../climber/moves/reach.py
    make dashboard HOST=127.0.0.1         # Moves -> reach -> Start (its options: which hand, where the point is)

The ball leaves its socket upwards (2 cm), then away from the wall (4 cm), goes to the point, and comes back the same
way.  Hanging, the arms are already stretched upwards: a point more than about 2 cm above the hold is out of reach
(the move then stops at once and says so).
"""
import microban_model
import numpy as np
import placo
from microban_model import JOINTS

LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")
ARMS = {side: tuple(f"{side}_{j}" for j in ("shoulder_pitch", "shoulder_roll", "elbow")) for side in ("left", "right")}
LIFT_M = 0.02                                # up out of the socket (a ball leaves it almost vertically)
OUT_M = 0.04                                 # then away from the wall
SEGMENT_S = 0.8                              # s per straight segment
PAUSE_S = 0.3                                # s held at each waypoint
REACH_TOL_M = 0.005                          # IK error accepted for the point


class ReachMove:
    """One hand out of its hold to a point, then back into it (hanging on the wall)."""

    variants = ("climber",)
    options = {"left_hand": False, "sideways_m": 0.03, "out_m": 0.06, "up_m": 0.0}
    option_docs = {"left_hand": "the left hand (else the right one)",
                   "sideways_m": "the point, from the hold: towards the hand's side (m)",
                   "out_m": "away from the wall (m)",
                   "up_m": "up (m; more than about 0.02 is out of reach)"}

    def __init__(self):
        self.model = placo.RobotWrapper(str(microban_model.urdf(climb=True)))
        self.solver = placo.KinematicsSolver(self.model)
        self.solver.mask_fbase(True)                       # the trunk stays at the origin: IK in the trunk frame
        self.solver.enable_joint_limits(True)
        self.solver.add_regularization_task(1e-6)
        self.task = None
        self.measured = placo.RobotWrapper(str(microban_model.urdf(climb=True)))   # forward kinematics of obs["q"]

    def start(self, obs):
        side = "left" if self.options["left_hand"] else "right"
        self.joints = ARMS[side]
        for name in JOINTS:
            (self.solver.unmask_dof if name in self.joints else self.solver.mask_dof)(name)
        self._set(obs["q"])
        frame = f"{side}_hand_ball"
        p0 = self.model.get_T_world_frame(frame)[:3, 3].copy()

        g = np.asarray(obs["gravity"], float)
        up = -g / np.linalg.norm(g) if np.all(np.isfinite(g)) else np.array([0.0, 0.0, 1.0])
        front = np.array([1.0, 0.0, 0.0]) - up[0] * up     # towards the wall, horizontal
        front /= np.linalg.norm(front)
        outward = np.cross(front, up) * (1.0 if side == "right" else -1.0)  # towards the hand's side (x × z: right)
        point = p0 + self.options["sideways_m"] * outward - self.options["out_m"] * front + self.options["up_m"] * up
        lifted, away = p0 + LIFT_M * up, p0 + LIFT_M * up - OUT_M * front
        self.path = [p0, lifted, away, point, away, lifted, p0]

        if self.task is not None:
            self.solver.remove_task(self.task)
        self.task = self.solver.add_position_task(frame, point)
        self.task.configure("reach", "soft", 1.0)
        for _ in range(100):                              # can the hand get there?
            self.solver.solve(True)
            self.model.update_kinematics()
        error = np.linalg.norm(self.model.get_T_world_frame(frame)[:3, 3] - point)
        self.reachable = error < REACH_TOL_M
        if not self.reachable:
            print(f"reach: the point is out of reach ({1000 * error:.0f} mm short), try a smaller offset")
        self._set(obs["q"])
        self.anchors = [f"{l}_ball" for l in LIMBS if not l.startswith(side + "_hand")]
        self.anchors_start = self._anchors(obs["q"])
        self.shift = (np.eye(3), np.zeros(3))
        self.t0 = obs["time"]

    def _set(self, q):
        for name, angle in zip(JOINTS, q):
            self.model.set_joint(name, float(angle))
        self.model.update_kinematics()

    def _anchors(self, q) -> np.ndarray:
        """Trunk-frame positions of the three hooked balls, from the measured angles."""
        for name, angle in zip(JOINTS, q):
            self.measured.set_joint(name, float(angle))
        self.measured.update_kinematics()
        return np.array([self.measured.get_T_world_frame(f)[:3, 3] for f in self.anchors])

    def target(self, t):
        """The ball's target t s after the start, None at the end of the path."""
        k, s = divmod(t, SEGMENT_S + PAUSE_S)
        k = int(k)
        if k >= len(self.path) - 1:
            return None
        a, b = self.path[k], self.path[k + 1]
        return a + min(s / SEGMENT_S, 1.0) * (b - a)

    def step(self, obs):
        p = self.target(obs["time"] - self.t0) if self.reachable else None
        if p is None:
            return None                                    # done: the arm keeps its last targets
        # the trunk moves a little: the hooked balls, fixed on the wall, show how (rigid fit, in the trunk frame), and the
        # path, fixed on the wall too, moves with them
        if np.all(np.isfinite(obs["q"])):
            self.shift = rigid_fit(self.anchors_start, self._anchors(obs["q"]))
        R, t = self.shift
        self.task.target_world = R @ p + t
        for _ in range(3):
            self.solver.solve(True)
            self.model.update_kinematics()
        return {name: self.model.get_joint(name) for name in self.joints}


def rigid_fit(a: np.ndarray, b: np.ndarray):
    """(R, t) of the rigid motion closest to taking the points a to b (N x 3): b ~ R a + t."""
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, cb - R @ ca
