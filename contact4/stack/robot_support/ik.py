"""Differential inverse kinematics of the arm, built on Pink and Pinocchio.

One call advances the joint configuration by one control step toward a Cartesian target of
the end-effector frame, subject to the joint position, velocity and acceleration limits of
the URDF. The configuration persists between calls, so successive targets give a smooth
joint trajectory.
"""
import numpy as np
import pinocchio as pin
import pink
import qpsolvers
from pink.limits import AccelerationLimit, ConfigurationLimit, VelocityLimit
from pink.tasks import FrameTask
from scipy.spatial.transform import Rotation as R

# Robot type reported by the robot server -> (URDF path, name of the end-effector frame).
MODELS = {}

TASK_GAIN = 0.9
POSITION_COST = 1.0
ORIENTATION_COST = 0.2
LM_DAMPING = 0.1
VELOCITY_SCALE = 0.5          # joint velocity limit as a fraction of the URDF value
ACCELERATION_SCALE = 50.0     # joint acceleration limit = this factor times the joint velocity limit above


class BrakingAccelerationLimit(AccelerationLimit):
    """Acceleration limit whose braking bound stays finite for a joint on or beyond a position limit."""

    def compute_qp_inequalities(self, configuration, dt):
        if self.projection_matrix is None:
            return None
        model, q = self.model, configuration.q
        room_up = np.maximum(pin.difference(model, q, model.upperPositionLimit)[self.indices], 0.0)
        room_down = np.maximum(pin.difference(model, model.lowerPositionLimit, q)[self.indices], 0.0)
        dt_sq = dt * dt
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([
            np.minimum(self.a_max * dt_sq + self.Delta_q_prev, dt * np.sqrt(2 * self.a_max * room_up)),
            np.minimum(self.a_max * dt_sq - self.Delta_q_prev, dt * np.sqrt(2 * self.a_max * room_down)),
        ])
        return G, h


def qp_solver_name():
    if not qpsolvers.available_solvers:
        raise RuntimeError("No QP solver is installed; install daqp")
    return "daqp" if "daqp" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]


class DifferentialIk:
    """Pink differential IK for one arm. Targets are 4x4 transforms of the end-effector frame in the base frame."""

    def __init__(self, robot_type):
        if robot_type not in MODELS:
            raise ValueError("No kinematic model registered for robot type %r" % (robot_type,))
        urdf_path, self.frame_name = MODELS[robot_type]
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.task = FrameTask(self.frame_name, position_cost=POSITION_COST, orientation_cost=ORIENTATION_COST,
                              lm_damping=LM_DAMPING, gain=TASK_GAIN)
        velocity = np.asarray(self.model.velocityLimit, dtype=np.float64).copy() * VELOCITY_SCALE
        self.model.velocityLimit = velocity
        self.acceleration_limit = BrakingAccelerationLimit(self.model, velocity * ACCELERATION_SCALE)
        self.limits = [ConfigurationLimit(self.model, config_limit_gain=1.0), VelocityLimit(self.model),
                       self.acceleration_limit]
        self.qp_solver = qp_solver_name()
        middle = 0.5 * (self.model.lowerPositionLimit + self.model.upperPositionLimit)
        self.configuration = pink.Configuration(self.model, self.data, middle)
        self.task.set_target_from_configuration(self.configuration)
        self._needs_reset = True

    def reset(self, joints):
        """Restart from the measured joint positions."""
        joints = np.asarray(joints, dtype=float)
        if joints.shape != self.configuration.q.shape:
            raise ValueError("joint vector has the wrong length")
        self.configuration = pink.Configuration(self.model, self.data, joints)
        if self.acceleration_limit.projection_matrix is not None:
            self.acceleration_limit.Delta_q_prev.fill(0.0)
        self._needs_reset = False

    def solve(self, target, joints, dt):
        """Advance one step of length dt toward target. Returns (joint positions, ok)."""
        if self._needs_reset:
            self.reset(joints)
        try:
            target = np.asarray(target, dtype=np.float64)
            rpy = R.from_matrix(target[:3, :3]).as_euler("xyz", degrees=False)
            x, y, z, w = R.from_euler("xyz", np.asarray(rpy, dtype=np.float64), degrees=False).as_quat()
            quat = np.array([w, x, y, z], dtype=np.float64)
            quat = quat / np.linalg.norm(quat)
            rotation = pin.Quaternion(quat[0], quat[1], quat[2], quat[3]).toRotationMatrix()
            self.task.transform_target_to_world = pin.SE3(rotation, np.asarray(target[:3, 3], dtype=np.float64))
            dt = float(dt)
            velocity = pink.solve_ik(self.configuration, [self.task], dt, solver=self.qp_solver, limits=self.limits)
            self.configuration.integrate_inplace(velocity, dt)
            self.acceleration_limit.set_last_integration(velocity, dt)
            result = self.configuration.q.copy().tolist()
            return result, bool(np.all(np.isfinite(result)))
        except Exception as exc:  # noqa: BLE001  the caller reports ik_failed
            print("[DifferentialIk] step failed: %s" % exc)
            return np.asarray(joints, dtype=float).tolist(), False
