"""placo's IK on the climbing robot, the trunk free (a floating base): the solver chooses the trunk's position and
tilt with the joints, as in the onboarding's example (a task per ball).  Plus a collision check against the wall
(MuJoCo distances), since the IK alone happily puts the trunk into the wall.

The placo model and the simulator share the world frame: the trunk is placo's floating base, set from
``sim.trunk_pose()`` (its balls then coincide with the simulator's).
"""
import microban_model
import mujoco
import numpy as np
import placo

from ..common import JOINTS, LIMBS

DISTAL_BODIES = ("radius_right", "radius_left", "foot_climb", "foot_climb_2")   # the bodies with the balls
CLEARANCE_MAX = 0.05                         # m: distances are computed up to this


class ClimberIK:
    """The robot in placo (``state``, ``set_state``, ``solve``) and its clearance from the wall in MuJoCo."""

    def __init__(self, sim):
        self.robot = placo.RobotWrapper(str(microban_model.urdf(climb=True)))
        m = self.model = sim.model
        self.data = mujoco.MjData(m)
        self.qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS]
        robot_bodies = {m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")]}
        robot_bodies |= {b for b in range(m.nbody) if m.body_rootid[b] in robot_bodies}
        distal = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in DISTAL_BODIES}
        collides = lambda g: m.geom_contype[g] or m.geom_conaffinity[g]
        name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        self.body_geoms = np.array([g for g in range(m.ngeom)
                                    if m.geom_bodyid[g] in robot_bodies - distal and collides(g)])
        self.wall_geoms = np.array([g for g in range(m.ngeom) if name(g) in ("wall", "wall_front")
                                    or (m.geom_bodyid[g] not in robot_bodies and m.geom_bodyid[g] != 0
                                        and collides(g))])

    @classmethod
    def of(cls, sim) -> "ClimberIK":
        """The ClimberIK of a simulator (made once)."""
        if getattr(sim, "_climber_ik", None) is None:
            sim._climber_ik = cls(sim)
        return sim._climber_ik

    def set_state(self, q, trunk_pos, trunk_quat):
        T = np.eye(4)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(trunk_quat, float))
        T[:3, :3], T[:3, 3] = R.reshape(3, 3), trunk_pos
        self.robot.set_T_world_fbase(T)
        for n, a in zip(JOINTS, q):
            self.robot.set_joint(n, float(a))
        self.robot.update_kinematics()

    def state(self):
        """(q, trunk_pos, trunk_quat w x y z) of the placo model."""
        T = self.robot.get_T_world_fbase()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T[:3, :3].flatten())
        return np.array([self.robot.get_joint(n) for n in JOINTS]), T[:3, 3].copy(), quat

    def balls(self) -> dict:
        return {l: self.robot.get_T_world_frame(f"{l}_ball")[:3, 3].copy() for l in LIMBS}

    def solve(self, targets: dict, weights: dict | None = None, trunk=None, trunk_weight=1.0, hold_y=None,
              iters: int = 100) -> float:
        """From the current placo state, the joints and the trunk that put the balls at ``targets`` (limb -> world
        xyz; ``weights``: limb -> weight, default 10); ``trunk``: (pos, quat) to stay near (soft, ``trunk_weight``);
        ``hold_y``: keep the trunk at this distance from the wall (y, soft).  Returns the largest ball distance
        left (m); the solution is the new placo state."""
        solver = placo.KinematicsSolver(self.robot)
        solver.enable_joint_limits(True)
        for l, p in targets.items():
            solver.add_position_task(f"{l}_ball", np.asarray(p, float)).configure(l, "soft", (weights or {}).get(l, 10.0))
        if trunk is not None:
            T = np.eye(4)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, np.asarray(trunk[1], float))
            T[:3, :3], T[:3, 3] = R.reshape(3, 3), trunk[0]
            solver.add_frame_task("trunk", T).configure("trunk", "soft", trunk_weight, trunk_weight)
        if hold_y is not None:
            task = solver.add_position_task("trunk", np.array([0.0, hold_y, 0.0]))
            task.mask.set_axises("y")
            task.configure("trunk_y", "soft", 1.0)
        solver.add_regularization_task(1e-4)
        for _ in range(iters):
            solver.solve(True)
            self.robot.update_kinematics()
        balls = self.balls()
        return max(float(np.linalg.norm(balls[l] - np.asarray(p))) for l, p in targets.items())

    def clearance(self, q=None, trunk_pos=None, trunk_quat=None) -> float:
        """Smallest distance (m, up to CLEARANCE_MAX; < 0: inside) between the robot's body (all but the four
        distal parts, with the balls) and the wall or the holds, for that configuration (default: placo's)."""
        if q is None:
            q, trunk_pos, trunk_quat = self.state()
        m, d = self.model, self.data
        d.qpos[:3], d.qpos[3:7], d.qpos[self.qadr] = trunk_pos, trunk_quat, q
        mujoco.mj_kinematics(m, d)
        a, b = self.body_geoms, self.wall_geoms
        gap = (np.linalg.norm(d.geom_xpos[a][:, None] - d.geom_xpos[b][None], axis=2)
               - m.geom_rbound[a][:, None] - m.geom_rbound[b][None])
        ia, ib = np.nonzero(gap < CLEARANCE_MAX)
        best, fromto = CLEARANCE_MAX, np.zeros(6)
        for ga, gb in zip(a[ia], b[ib]):
            best = min(best, mujoco.mj_geomDistance(m, d, ga, gb, CLEARANCE_MAX, fromto))
        return float(best)
