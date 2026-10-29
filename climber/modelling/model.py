"""The robot's model in Pinocchio: the climbing URDF of microban_model with a floating base (the trunk).

Kinematics and dynamics without the simulation: the balls' positions, the centre of mass, Jacobians, gravity
torques...  A configuration of the simulator (19 angles in the order JOINTS, the trunk pose with a w x y z
quaternion) becomes Pinocchio's with ``configuration``.
"""
import microban_model
import numpy as np
import pinocchio as pin

from ..common import JOINTS, LIMBS


class RobotModel:
    def __init__(self):
        self.model = pin.buildModelFromUrdf(str(microban_model.urdf(climb=True)), pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.q_index = [self.model.joints[self.model.getJointId(n)].idx_q for n in JOINTS]
        self.ball_frames = {l: self.model.getFrameId(f"{l}_ball") for l in LIMBS}

    @property
    def mass(self) -> float:
        return float(sum(inertia.mass for inertia in self.model.inertias))

    def configuration(self, q, trunk_pos, trunk_quat) -> np.ndarray:
        """Pinocchio's configuration: [x y z, qx qy qz qw, joints in the model's order]."""
        w, x, y, z = trunk_quat
        config = pin.neutral(self.model)
        config[:7] = [*trunk_pos, x, y, z, w]
        config[self.q_index] = q
        return config

    def ball_positions(self, config) -> dict:
        """limb -> world position of its ball."""
        pin.framesForwardKinematics(self.model, self.data, config)
        return {l: self.data.oMf[f].translation.copy() for l, f in self.ball_frames.items()}

    def center_of_mass(self, config) -> np.ndarray:
        return pin.centerOfMass(self.model, self.data, config).copy()
