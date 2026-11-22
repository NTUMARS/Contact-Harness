#!/usr/bin/env python3
"""Cartesian impedance robot server for a Franka arm with the Franka gripper.

Target publication: 50 Hz. Existing CartesianImpedanceController: ~1000 Hz torque loop.
The measured pose seeds the target ONCE. Thereafter: target_next = target_previous + delta.
Planner interface: POST /actions {"actions": [[dx, dy, dz, rx, ry, rz], ...]}.
Use one row for a single delta. The planner decomposes larger actions; this server
consumes ONE row per 20 ms tick, FIFO, and holds the last target when empty.
Translation is in meters; rotation is a rotation vector in radians, both in the
robot BASE frame: p_next = p_target + delta_xyz; R_next = Exp(delta_rotvec)*R_target.
HTTP 202 means queued, not physically reached. Serialize planner requests to
preserve order; retrying an accepted request adds the same deltas again.
GET /info reports queue_depth/applied_actions; POST /actions/clear drops pending
actions and holds the current target. See robot_control/README.md.
Agent functions (HTTP, all on the same server):
  POST /home {} -> 202; GET /home until state is succeeded/failed.
    Clears queued deltas, requires a stationary arm, temporarily uses JOINT
    POSITION control to the start script's HOME_JOINT_TARGET (seven angles in radians), then restores Cartesian
    impedance and explicitly seeds its target from the reached measured pose.
    New deltas/homing/gripper moves return 409 while homing. No automatic recovery.
  POST /gripper {"width_m": 0.0, "force_n": 40.0} -> 202: native force grasp.
    Width is total finger opening (0..0.08 m); force_n is passed directly to
    Franka GraspGoal.force (30..70 N), with no additional force cap or local force loop.
    Positive force selects "grasp"; mode="grasp" without force_n uses the start script's
    GRIPPER_GRASP_FORCE default. Optional speed_m_s defaults to 0.05 (0.005..0.2).
    Grasp epsilon_inner_m defaults to 0.005 m and epsilon_outer_m to 0.08 m:
    contact before reaching the closing width is accepted across the finger stroke.
    Set tighter tolerances explicitly when an exact object width matters. Franka
    checks (width_m-epsilon_inner_m, width_m+epsilon_outer_m) for grasp success.
  POST /gripper {"width_m": 0.08, "force_n": 0} -> 202: native Move to open/release.
    Omitting both force_n and mode also preserves position-only Move. Explicit
    mode="move" accepts only omitted/zero force; mode="grasp" requires positive force.
    Poll GET /gripper for the result and commanded force_n (not measured force).
    The installed native API has no in-place holding-force update. After a confirmed
    successful grasp, another grasp returns HTTP 409 without sending a new goal.
    An explicit successful Move clears this guard. This uses command history,
    not live contact/force feedback, and cannot detect grasps made before startup.
    Grasp errors remain failures. A grasp timeout/lost result sets outcome_unknown;
    further commands wait for the original result. No automatic Stop/open/regrasp:
    Stop removes holding force. A late result is reported separately, never as a
    replacement success for a timed-out request. Unknown results may require
    deliberate operator intervention; restarting cannot prove a grasp was released.
Both operations are asynchronous: 202 means accepted, NOT completed. Poll status
instead of resending on timeout. Homing has a 90 s deadline; failure stops the
server. The home is a direct joint-space move, not a collision-free path planner.
No automatic error recovery.
"""
import argparse
from collections import deque
import copy
import json
import logging
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

TARGET_HZ = 50.0
TARGET_PERIOD = 1.0 / TARGET_HZ
MAX_QUEUED_ACTIONS = 500  # 10 seconds at 50 Hz; reject overflow, never silently drop actions.
MAX_TRANSLATION_INCREMENT = .0005  # m per tick, vector norm
MAX_ROTATION_INCREMENT = .005  # rad per tick, rotation-vector norm
POSE_TOPIC = '/cartesian_impedance_controller/equilibrium_pose'
CONTROLLER = 'franka_controllers/CartesianImpedanceController'
HOME_JOINT_TARGET = (0., 0., 0., -1.9, 0., 2., 0.7854)  # Radians; joint 7 is approximately 45 degrees.
HOME_CONTROLLER = 'home_controller'
HOME_CONTROLLER_TYPE = 'franka_controllers/JointPositionFrankaController'
HOME_TIMEOUT = 90.
DEFAULT_GRIPPER_GRASP_FORCE = 50.  # N.
# Default profile; --eef-impedance and --translational-clip replace the matching entries.
IMPEDANCE_PROFILE = {
    'translational_stiffness': 1500., 'translational_damping': 89.,
    'rotational_stiffness': 150., 'rotational_damping': 7.,
    'nullspace_stiffness': .2, 'joint1_nullspace_stiffness': 100.,
    'translational_Ki': 0., 'rotational_Ki': 0.,
    **{'translational_clip_'+s: .01 for s in ['x','y','z','neg_x','neg_y','neg_z']},
    **{'rotational_clip_'+s: .05 for s in ['x','y','z','neg_x','neg_y','neg_z']},
}
EEF_IMPEDANCE_FIELDS = (
    'translational_stiffness', 'translational_damping',
    'rotational_stiffness', 'rotational_damping',
)
# Match the existing franka_controllers/cfg/compliance_param.cfg bounds.
EEF_IMPEDANCE_MAX = (6000., 400., 300., 30.)
COLLISION_TORQUE_BASELINE = (20., 20., 18., 18., 16., 14., 12.)  # Joints 1-7, Nm.
LOCAL_LAUNCH = Path(__file__).resolve().parent / 'robot_control' / 'impedance.launch'


def parse_home_joint_target(value):
    try:
        values = tuple(float(item) for item in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Home target must be seven comma-separated angles in radians') from exc
    if len(values) != 7 or not all(math.isfinite(number) for number in values):
        raise argparse.ArgumentTypeError('Home target must contain seven finite angles in radians')
    return values  # Actual robot joint limits are checked before switching controllers.


def parse_gripper_grasp_force(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError('Gripper grasp force must be a number in N') from exc
    # Franka Hand product manual, technical specifications: continuous force 30-70 N.
    if isinstance(value, bool) or not math.isfinite(number) or not 30. <= number <= 70.:
        raise argparse.ArgumentTypeError('Gripper grasp force must be finite and in [30, 70] N')
    return number


def parse_eef_impedance(value):
    try:
        values = tuple(float(item) for item in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('EEF impedance must be four comma-separated numbers') from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError('EEF impedance requires K_translation,D_translation,K_rotation,D_rotation')
    for name, number, maximum in zip(EEF_IMPEDANCE_FIELDS, values, EEF_IMPEDANCE_MAX):
        if not math.isfinite(number) or not 0 <= number <= maximum:
            raise argparse.ArgumentTypeError('%s must be finite and in [0, %g]' % (name, maximum))
    return values


def parse_collision_force_threshold(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Collision force threshold must be a number in N') from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Collision force threshold must be finite and > 0 N')
    return number


def scaled_collision_torques(scale):
    return [threshold * scale for threshold in COLLISION_TORQUE_BASELINE]


def parse_collision_torque_scale(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Collision torque scale must be a number') from exc
    if not math.isfinite(number) or number < 1:
        raise argparse.ArgumentTypeError('Collision torque scale must be finite and >= 1.0')
    if not all(math.isfinite(threshold) for threshold in scaled_collision_torques(number)):
        raise argparse.ArgumentTypeError('Scaled collision torque thresholds must remain finite')
    return number


def parse_translational_clip(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Translation error clip must be a number in meters') from exc
    # The existing controller's dynamic reconfigure upper bound is 0.1 m.
    if not math.isfinite(number) or not 0 < number <= .1:
        raise argparse.ArgumentTypeError('Translation error clip must be finite and in (0, 0.1] m')
    return number


def make_impedance_profile(eef_impedance, translational_clip=.01):
    profile = dict(IMPEDANCE_PROFILE)
    profile.update(zip(EEF_IMPEDANCE_FIELDS, eef_impedance))
    for name in profile:
        if name.startswith('translational_clip_'):
            profile[name] = translational_clip
    return profile


def launch_command(args):
    # The copied launch/config changes only this server's driver startup parameters.
    # A decimal point in scientific notation keeps all finite values numeric in ROS YAML 1.1.
    torque_array = '['+', '.join(format(value, '.17e') for value in
                                 scaled_collision_torques(args.collision_torque_scale))+']'
    return ['roslaunch', '-p', str(args.ros_port), str(LOCAL_LAUNCH),
            'robot_ip:='+args.robot_ip, 'load_gripper:=true',
            'collision_force_threshold:='+str(args.collision_force_threshold),
            'collision_torque_thresholds:='+torque_array]


def finite_vector(value, length, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (length,) or not np.all(np.isfinite(value)):
        raise ValueError(name+' must contain %d finite numbers' % length)
    return value


def validate_increment(delta_xyz, delta_rotvec):
    try:
        delta_xyz = finite_vector(delta_xyz, 3, 'delta_xyz_m')
        delta_rotvec = finite_vector(delta_rotvec, 3, 'delta_rotvec_rad')
    except (TypeError, OverflowError) as exc:
        raise ValueError('Delta components must be finite numbers') from exc
    if np.linalg.norm(delta_xyz) > MAX_TRANSLATION_INCREMENT+1e-12:
        raise ValueError('Translation increment exceeds 0.5 mm per tick')
    if np.linalg.norm(delta_rotvec) > MAX_ROTATION_INCREMENT+1e-12:
        raise ValueError('Rotation increment exceeds 0.005 rad per tick')
    return delta_xyz, delta_rotvec


def valid_pose(value):
    value = finite_vector(value, 7, 'pose').copy()
    norm = np.linalg.norm(value[3:])
    if norm < 1e-8:
        raise ValueError('Invalid zero quaternion')
    value[3:] /= norm
    return value


class CommandBusy(RuntimeError):
    pass


class FrankaServer:
    """Target accumulator, bounded command FIFO and latest state."""
    def __init__(self, increment_provider=None, clock=time.monotonic,
                 impedance_profile=None, collision_force_threshold=20., collision_torque_scale=1.,
                 home_joint_target=HOME_JOINT_TARGET, gripper_grasp_force=DEFAULT_GRIPPER_GRASP_FORCE):
        self.lock = threading.RLock()
        self.clock = clock
        self.impedance_profile = dict(IMPEDANCE_PROFILE if impedance_profile is None else impedance_profile)
        self.collision_force_threshold = collision_force_threshold
        self.collision_torque_scale = collision_torque_scale
        self.increment_provider = increment_provider  # Optional in-process provider for offline use.
        self.action_queue = deque()
        self.accepted_actions = 0
        self.applied_actions = 0
        self.cleared_actions = 0
        self.state = None
        self.target_pose = None
        self.last_robot_time = None
        self.received = None
        self.state_error = None
        self.published_count = 0
        self.first_publish_time = None
        self.last_publish_time = None
        self.command_backend = None  # Attached only after ROS/controller startup.
        self.home_joint_target = tuple(finite_vector(home_joint_target, 7, 'home_joint_target'))
        self.gripper_grasp_force = parse_gripper_grasp_force(gripper_grasp_force)
        self.home = {'state': 'idle', 'target_q': list(self.home_joint_target)}
        self.gripper = {'state': 'idle'}
        self.last_successful_grasp = None  # Command history, not live is_grasped/force feedback.
        self.home_settled_since = None

    def home_busy(self):
        return self.home['state'] in ('pending', 'moving', 'restoring')

    def gripper_busy(self):
        return (self.gripper['state'] in ('pending', 'moving', 'stopping') or
                self.gripper.get('outcome_unknown', False))

    def command_status(self, kind):
        with self.lock:
            return copy.deepcopy(self.home if kind == 'home' else self.gripper)

    def request_home(self):
        """Agent-callable home request. Only the main loop may switch arm controllers."""
        with self.lock:
            if self.home_busy() or self.gripper_busy():
                raise CommandBusy('Another home/gripper operation is active')
            if self.command_backend is None or self.target_pose is None:
                raise RuntimeError('Robot command backend is not ready')
            state = self.latest_state()
            if np.max(np.abs(finite_vector(state['dq'], 7, 'dq'))) > .02:
                raise CommandBusy('Wait for the arm to be stationary before homing')
            cleared = self.clear_actions()['cleared']
            self.home = {'state': 'pending', 'target_q': list(self.home_joint_target),
                         'cleared_actions': cleared, 'requested_at': self.clock(), 'error': None}
            self.home_settled_since = None
            return copy.deepcopy(self.home)

    def request_gripper(self, width_m, speed_m_s=.05, mode=None,
                        epsilon_inner_m=None, epsilon_outer_m=None, *, force_n=None):
        """Native grasp with width_m/force_n; zero or omitted force preserves Move."""
        if force_n is not None and (
                isinstance(force_n, bool) or not isinstance(force_n, (int, float)) or
                not 0. <= force_n <= 70. or not math.isfinite(force_n)):
            raise ValueError('force_n must be a finite number: 0 for Move, 30..70 N for Grasp')
        if mode is None:
            mode = 'grasp' if force_n is not None and force_n > 0. else 'move'
        if mode not in ('move', 'grasp'):
            raise ValueError('mode must be "move" or "grasp"')
        fields = [('width_m', width_m, 0., .08), ('speed_m_s', speed_m_s, .005, .2)]
        if mode == 'grasp':
            try:
                force_n = parse_gripper_grasp_force(self.gripper_grasp_force if force_n is None else force_n)
            except argparse.ArgumentTypeError as exc:
                raise ValueError(str(exc)) from exc
            epsilon_inner_m = .005 if epsilon_inner_m is None else epsilon_inner_m
            epsilon_outer_m = .08 if epsilon_outer_m is None else epsilon_outer_m
            fields += [('epsilon_inner_m', epsilon_inner_m, 0., .08),
                       ('epsilon_outer_m', epsilon_outer_m, 0., .08)]
        elif force_n not in (None, 0):
            raise ValueError('mode="move" accepts only omitted/zero force_n; use "grasp" for force')
        elif epsilon_inner_m is not None or epsilon_outer_m is not None:
            raise ValueError('Width tolerances apply only to mode "grasp"')
        for name, value, lower, upper in fields:
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not lower <= value <= upper or not math.isfinite(value)):
                raise ValueError('%s must be a finite number in [%g, %g]' % (name, lower, upper))
        with self.lock:
            if self.home_busy() or self.gripper_busy():
                raise CommandBusy('Another home/gripper operation is active')
            if self.command_backend is None or self.target_pose is None:
                raise RuntimeError('Robot command backend is not ready')
            self.latest_state()
            if mode == 'grasp' and self.last_successful_grasp is not None:
                raise CommandBusy(
                    'Native Franka holding-force/width updates are unsupported: an earlier grasp '
                    'succeeded and may still be holding. No command sent. Only an explicit '
                    'successful Move clears this guard; it can release the object.')
            self.gripper = {'state': 'pending', 'width_m': float(width_m),
                            'speed_m_s': float(speed_m_s), 'mode': mode,
                            'requested_at': self.clock(), 'error': None}
            if mode == 'grasp':
                self.gripper.update(force_n=force_n,
                                    epsilon_inner_m=float(epsilon_inner_m),
                                    epsilon_outer_m=float(epsilon_outer_m))
            return copy.deepcopy(self.gripper)

    def step_home(self):
        """Main-loop state machine. True suppresses Cartesian target publication this tick."""
        with self.lock:
            phase = self.home['state']
            if not self.home_busy():
                return False
        try:
            state = self.latest_state()
            if self.clock()-self.home['requested_at'] > HOME_TIMEOUT:
                raise RuntimeError('Homing timed out')
            if phase == 'pending':
                if np.max(np.abs(finite_vector(state['dq'], 7, 'dq'))) > .02:
                    raise RuntimeError('Arm started moving before the home controller switch')
                # No lock across ROS services: feedback callbacks must remain able to run.
                self.command_backend.start_home()
                with self.lock:
                    self.home['state'] = 'moving'
            elif phase == 'moving':
                self.command_backend.publish_home()
                reached = (np.max(np.abs(finite_vector(state['q'], 7, 'q')-self.home_joint_target)) < .005
                           and np.max(np.abs(finite_vector(state['dq'], 7, 'dq'))) < .01)
                self.home_settled_since = (self.home_settled_since if self.home_settled_since is not None
                                           else self.clock()) if reached else None
                if reached and self.clock()-self.home_settled_since >= .3:
                    self.command_backend.restore_impedance()
                    with self.lock:
                        self.home.update(state='restoring', restore_robot_time=self.last_robot_time)
            elif phase == 'restoring' and state['robot_time_s'] > self.home['restore_robot_time']:
                # The controller seeds itself when restarted; sync the server's target too.
                if np.max(np.abs(finite_vector(state['dq'], 7, 'dq'))) > .02:
                    return True
                if np.max(np.abs(finite_vector(state['q'], 7, 'q')-self.home_joint_target)) >= .005:
                    raise RuntimeError('Arm moved away from home during controller restoration')
                with self.lock:
                    self.target_pose = valid_pose(state['pose'])
                    self.home.update(state='succeeded', completed_at=self.clock())
            return True
        except Exception as exc:
            with self.lock:
                self.home.update(state='failed', error=str(exc))
                self.state_error = 'Homing failed: '+str(exc)
            # Do not resume queued Cartesian motion after a failed switch or robot fault.
            raise RuntimeError(self.state_error) from exc

    def step_gripper(self):
        """Poll action results without blocking the 50 Hz arm loop."""
        with self.lock:
            phase = self.gripper['state']
            if not self.gripper_busy():
                return
            grasp = self.gripper['mode'] == 'grasp'
        if phase == 'pending':
            try:
                self.latest_state()
                if self.gripper['mode'] == 'grasp':
                    self.command_backend.start_gripper(
                        self.gripper['width_m'], self.gripper['speed_m_s'],
                        force_n=self.gripper['force_n'], epsilon_inner_m=self.gripper['epsilon_inner_m'],
                        epsilon_outer_m=self.gripper['epsilon_outer_m'])
                else:
                    self.command_backend.start_gripper(self.gripper['width_m'], self.gripper['speed_m_s'])
            except Exception as exc:
                with self.lock:
                    self.gripper.update(state='failed', error=str(exc))
                    if grasp and self.command_backend.gripper_active:
                        self.gripper['outcome_unknown'] = True
                        return  # Do not release a possible grasp after uncertain delivery.
                if self.command_backend.gripper_active:
                    raise RuntimeError('Gripper command delivery failed: '+str(exc)) from exc
                return
            with self.lock:
                self.gripper.update(state='moving', deadline=self.clock()+.08/self.gripper['speed_m_s']+5.)
        else:
            try:
                result = self.command_backend.gripper_result(stopping=phase == 'stopping')
            except Exception as exc:
                if not grasp:
                    raise
                with self.lock:
                    if not self.gripper.get('outcome_unknown'):
                        self.gripper.update(state='failed', error=str(exc), outcome_unknown=True)
                    self.gripper['result_error'] = str(exc)
                return  # Lost result is not permission to Stop (which removes holding force).
            if result is not None:
                success, error = result
                with self.lock:
                    if success and phase != 'stopping':
                        self.last_successful_grasp = (
                            {key: self.gripper[key] for key in
                             ('width_m', 'force_n', 'epsilon_inner_m', 'epsilon_outer_m')}
                            if grasp else None)
                    if self.gripper.get('outcome_unknown'):
                        self.gripper.update(outcome_unknown=False,
                                            late_result={'success': bool(success), 'error': error})
                        # Keep state=failed and the original error; a late result is separate.
                    elif phase == 'stopping':
                        self.gripper.update(state='failed', error='Gripper command timed out; stop '+
                                            ('completed' if success else 'failed: '+error))
                    else:
                        self.gripper.update(state='succeeded' if success else 'failed', error=error)
                if phase == 'stopping' and not success:
                    raise RuntimeError('Gripper stop failed: '+error)
            elif not self.gripper.get('outcome_unknown') and self.clock() > self.gripper['deadline']:
                if grasp:
                    with self.lock:
                        if not self.gripper.get('outcome_unknown'):
                            self.gripper.update(state='failed', outcome_unknown=True,
                                                error='Native grasp timed out; outcome unknown. '
                                                      'No Stop/open sent because it may release the object. '
                                                      'Waiting for the original result; new commands are blocked.')
                    return
                if phase == 'stopping':
                    raise RuntimeError('Gripper stop timed out')
                self.command_backend.stop_gripper()
                with self.lock:
                    self.gripper.update(state='stopping', deadline=self.clock()+3.)

    def update_state(self, state):
        """Feedback is observational. This function NEVER overwrites target_pose."""
        state = copy.deepcopy(state)
        state['pose'] = valid_pose(state['pose']).tolist()
        robot_time = float(state['robot_time_s'])
        if not math.isfinite(robot_time):
            raise ValueError('Invalid robot time')
        with self.lock:
            if self.last_robot_time is not None:
                if robot_time < self.last_robot_time:
                    self.state_error = 'Robot time moved backwards; restart deliberately'
                    return
                if robot_time == self.last_robot_time:
                    return  # repeated states do not refresh their receipt time
            self.last_robot_time = robot_time
            self.state = state
            self.received = self.clock()

    def latest_state(self):
        with self.lock:
            if self.state_error:
                raise RuntimeError(self.state_error)
            if self.state is None or self.clock()-self.received > .5:
                raise RuntimeError('No fresh advancing robot state')
            state = copy.deepcopy(self.state)
        if state.get('current_errors'):
            raise RuntimeError('Robot error: '+', '.join(state['current_errors']))
        if state.get('robot_mode') not in (1, 2):
            raise RuntimeError('Robot is not in idle/move mode')
        return state

    def initialize_target_once(self):
        state = self.latest_state()
        if np.max(np.abs(finite_vector(state['dq'], 7, 'dq'))) > .02:
            raise RuntimeError('Wait for the arm to be stationary before starting')
        with self.lock:
            if self.target_pose is not None:
                raise RuntimeError('Target is already initialized; no automatic re-seeding')
            self.target_pose = valid_pose(state['pose'])
            return self.target_pose.copy()

    def enqueue_actions(self, actions):
        """Atomically append a validated batch; never change the target in an HTTP thread."""
        if not isinstance(actions, list) or not actions:
            raise ValueError('actions must be a non-empty list of 6D delta rows')
        if len(actions) > MAX_QUEUED_ACTIONS:
            raise OverflowError('Action queue capacity is %d' % MAX_QUEUED_ACTIONS)
        checked = []
        for action in actions:
            if (not isinstance(action, list) or len(action) != 6 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in action)):
                raise ValueError('Each action must contain six numbers: dx,dy,dz,rx,ry,rz')
            delta_xyz, delta_rotvec = validate_increment(action[:3], action[3:])
            checked.append((delta_xyz.copy(), delta_rotvec.copy()))
        with self.lock:
            if self.home_busy():
                raise CommandBusy('Homing is active; wait for GET /home to succeed')
            if self.increment_provider is not None:
                raise RuntimeError('HTTP queue is unavailable with a custom increment provider')
            if self.target_pose is None:
                raise RuntimeError('Target has not been initialized')
            self.latest_state()  # Reject new motion when feedback is stale or faulted.
            if len(self.action_queue)+len(checked) > MAX_QUEUED_ACTIONS:
                raise OverflowError('Action queue capacity is %d' % MAX_QUEUED_ACTIONS)
            self.action_queue.extend(checked)
            self.accepted_actions += len(checked)
            return {'accepted': len(checked), 'queue_depth': len(self.action_queue)}

    def clear_actions(self):
        with self.lock:
            count = len(self.action_queue)
            self.action_queue.clear()
            self.cleared_actions += count
            return {'cleared': count, 'queue_depth': 0}

    def next_target(self):
        state = self.latest_state()
        with self.lock:
            if self.home_busy():
                raise CommandBusy('Cartesian targets are paused during homing')
            if self.target_pose is None:
                raise RuntimeError('Target has not been initialized')
            previous = self.target_pose.copy()
            queued = self.increment_provider is None
            if queued:
                if not self.action_queue:
                    return previous
                delta_xyz, delta_rotvec = self.action_queue[0]  # Consume only after validation.
            else:
                delta_xyz, delta_rotvec = self.increment_provider(previous.copy(), copy.deepcopy(state))
            delta_xyz, delta_rotvec = validate_increment(delta_xyz, delta_rotvec)
            candidate = previous.copy()
            if np.any(delta_xyz) or np.any(delta_rotvec):
                candidate[:3] += delta_xyz
                candidate[3:] = (Rotation.from_rotvec(delta_rotvec)*Rotation.from_quat(previous[3:])).as_quat()
                measured = valid_pose(state['pose'])
                if np.linalg.norm(candidate[:3]-measured[:3]) > .03:
                    raise ValueError('New increment would exceed 3 cm target lead')
                angle = (Rotation.from_quat(candidate[3:])*Rotation.from_quat(measured[3:]).inv()).magnitude()
                if angle > .2:
                    raise ValueError('New increment would exceed 0.2 rad target lead')
            self.target_pose = candidate
            if queued:
                self.action_queue.popleft()
                self.applied_actions += 1  # Includes zero increments; each row occupies one tick.
            return candidate.copy()

    def note_publish(self):
        with self.lock:
            now = self.clock()
            if self.first_publish_time is None:
                self.first_publish_time = now
            self.last_publish_time = now
            self.published_count += 1

    def info(self):
        with self.lock:
            span = ((self.last_publish_time-self.first_publish_time)
                    if self.first_publish_time is not None else 0.)
            return {'server': 'franka_robot_server', 'target_publish_hz': TARGET_HZ,
                    'controller_type': CONTROLLER, 'controller_nominal_hz': 1000,
                    'target_rule': 'previous_target_plus_increment',
                    'increment_source': 'http_fifo' if self.increment_provider is None else 'custom_provider',
                    'queue_depth': len(self.action_queue), 'queue_capacity': MAX_QUEUED_ACTIONS,
                    'accepted_actions': self.accepted_actions, 'applied_actions': self.applied_actions,
                    'cleared_actions': self.cleared_actions,
                    'home': copy.deepcopy(self.home), 'gripper': copy.deepcopy(self.gripper),
                    'gripper_grasp_force_n': self.gripper_grasp_force,
                    'gripper_holding_force_update_supported': False,
                    'last_successful_grasp': copy.deepcopy(self.last_successful_grasp),
                    'max_translation_increment_m': MAX_TRANSLATION_INCREMENT,
                    'max_rotation_increment_rad': MAX_ROTATION_INCREMENT,
                    'increment_frame': 'robot_base', 'rotation_increment': 'rotation_vector_rad_left_multiply',
                    'target_pose': None if self.target_pose is None else self.target_pose.tolist(),
                    'published_count': self.published_count,
                    'observed_publish_hz': (self.published_count-1)/span if span > 0 else None,
                    'impedance_parameters': dict(self.impedance_profile),
                    'collision_force_threshold_n': self.collision_force_threshold,
                    'collision_contact_threshold_n': min(20., self.collision_force_threshold),
                    'collision_torque_scale': self.collision_torque_scale,
                    'collision_torque_thresholds_nm': scaled_collision_torques(self.collision_torque_scale),
                }


def create_app(server):
    from flask import Flask, jsonify, request
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 128 * 1024  # Allows batches up to the bounded FIFO capacity.

    @app.route('/info', methods=['GET'])
    def info():
        return jsonify(server.info())

    @app.route('/getstate', methods=['GET', 'POST'])
    def get_state():
        try:
            return jsonify(server.latest_state())
        except RuntimeError as exc:
            return jsonify({'error': str(exc)}), 503

    @app.route('/actions', methods=['POST'])
    def actions():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or set(data) != {'actions'}:
            return jsonify({'error': 'Expected JSON: {"actions": [[dx,dy,dz,rx,ry,rz], ...]}'}), 400
        try:
            return jsonify(server.enqueue_actions(data['actions'])), 202
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        except OverflowError as exc:
            return jsonify({'error': str(exc)}), 429
        except CommandBusy as exc:
            return jsonify({'error': str(exc)}), 409
        except RuntimeError as exc:
            return jsonify({'error': str(exc)}), 503

    @app.route('/actions/clear', methods=['POST'])
    def clear_actions():
        return jsonify(server.clear_actions())

    @app.route('/home', methods=['GET', 'POST'])
    def home():
        if request.method == 'GET':
            return jsonify(server.command_status('home'))
        if request.get_data() and request.get_json(silent=True) != {}:
            return jsonify({'error': 'POST /home accepts an empty body or {} only'}), 400
        try:
            return jsonify(server.request_home()), 202
        except CommandBusy as exc:
            return jsonify({'error': str(exc)}), 409
        except RuntimeError as exc:
            return jsonify({'error': str(exc)}), 503

    @app.route('/gripper', methods=['GET', 'POST'])
    def gripper():
        if request.method == 'GET':
            return jsonify(server.command_status('gripper'))
        data = request.get_json(silent=True)
        if (not isinstance(data, dict) or 'width_m' not in data or
                set(data)-{'width_m', 'force_n', 'speed_m_s', 'mode', 'epsilon_inner_m', 'epsilon_outer_m'}):
            return jsonify({'error': 'Expected width_m, optional force_n (0 for Move, 30..70 N for Grasp), '
                                    'speed_m_s, mode (move/grasp), '
                                    'epsilon_inner_m and epsilon_outer_m (grasp only)'}), 400
        if any(data[key] is None for key in ('force_n', 'mode', 'epsilon_inner_m', 'epsilon_outer_m') if key in data):
            return jsonify({'error': 'Optional gripper fields must be omitted or valid values, not null'}), 400
        try:
            return jsonify(server.request_gripper(
                data['width_m'], data.get('speed_m_s', .05), mode=data.get('mode'),
                force_n=data.get('force_n'), epsilon_inner_m=data.get('epsilon_inner_m'),
                epsilon_outer_m=data.get('epsilon_outer_m'))), 202
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        except CommandBusy as exc:
            return jsonify({'error': str(exc)}), 409
        except RuntimeError as exc:
            return jsonify({'error': str(exc)}), 503

    # No /pose alias: input is relative to the previous target, never a measured-pose reset.
    return app


class RosRobotCommands:
    """ROS adapter for this server only; configures the home controller in memory."""
    def __init__(self, home_joint_target=HOME_JOINT_TARGET):
        import actionlib
        import rospy
        from controller_manager_msgs.srv import LoadController, UnloadController, SwitchController, ListControllers
        from franka_gripper.msg import MoveAction, MoveGoal, StopAction, StopGoal, GraspAction, GraspGoal
        from franka_controllers.msg import JointFranka
        self.home_joint_target = tuple(finite_vector(home_joint_target, 7, 'home_joint_target'))
        self.rospy, self.JointFranka = rospy, JointFranka
        self.MoveGoal, self.StopGoal = MoveGoal, StopGoal
        self.GraspGoal = GraspGoal
        self.load = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)
        self.unload = rospy.ServiceProxy('/controller_manager/unload_controller', UnloadController)
        self.switch = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        self.list_controllers = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)
        self.joint_publisher = rospy.Publisher('/'+HOME_CONTROLLER+'/joint_info', JointFranka,
                                               queue_size=1, tcp_nodelay=True)
        self.move_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
        self.grasp_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.active_gripper_client = self.move_client
        self.stop_client = actionlib.SimpleActionClient('/franka_gripper/stop', StopAction)
        self.gripper_active = False

    def start_home(self):
        import xml.etree.ElementTree as ET
        rospy = self.rospy
        joints = rospy.get_param('/cartesian_impedance_controller/joint_names')
        model = ET.fromstring(rospy.get_param('/robot_description'))
        limits = {joint.attrib['name']: joint.find('limit') for joint in model.findall('joint')}
        if len(joints) != 7:
            raise RuntimeError('Expected seven arm joints')
        for name, target in zip(joints, self.home_joint_target):
            limit = limits.get(name)
            if (limit is None or not math.isfinite(target) or
                    not float(limit.attrib['lower']) < target < float(limit.attrib['upper'])):
                raise RuntimeError('Home target is outside the joint limits: '+name)
        # This plugin reads this optional legacy GLOBAL parameter during init.
        # Reject it so no unrelated startup target can move the arm during switching.
        if rospy.has_param('/target_joint_positions'):
            raise RuntimeError('Unexpected /target_joint_positions parameter; restart the private server')
        if any(c.name == HOME_CONTROLLER for c in self.list_controllers().controller):
            raise RuntimeError('Private home controller is already loaded')
        rospy.set_param('/'+HOME_CONTROLLER, {
            'type': HOME_CONTROLLER_TYPE, 'joint_names': joints,
            'filter_params': .005, 'max_joint_velocity': .1,
            'max_joint_acceleration': .2, 'hybrid_switch_threshold': .012,
            'command_timeout': .5})
        if not self.load(HOME_CONTROLLER).ok:
            raise RuntimeError('Could not load the joint home controller')
        if not self.switch(start_controllers=[HOME_CONTROLLER],
                           stop_controllers=['cartesian_impedance_controller'],
                           strictness=2, start_asap=False, timeout=5.).ok:
            raise RuntimeError('Could not switch to joint position control')

    def publish_home(self):
        message = self.JointFranka()
        message.joint_angle = list(self.home_joint_target)
        message.d_joint_angle = [0.]*7
        self.joint_publisher.publish(message)

    def restore_impedance(self):
        if not self.switch(start_controllers=['cartesian_impedance_controller'],
                           stop_controllers=[HOME_CONTROLLER], strictness=2,
                           start_asap=False, timeout=5.).ok:
            raise RuntimeError('Could not restore Cartesian impedance')
        if not self.unload(HOME_CONTROLLER).ok:
            raise RuntimeError('Could not unload the joint home controller')

    def start_gripper(self, width, speed, force_n=None, epsilon_inner_m=.005, epsilon_outer_m=.08):
        # Short availability checks only; movement completion is polled each tick.
        command_client = self.move_client if force_n is None else self.grasp_client
        for client in (command_client, self.stop_client):
            if not client.wait_for_server(self.rospy.Duration(.01)):
                raise RuntimeError('Franka gripper action server is unavailable')
        if force_n is None:
            goal = self.MoveGoal(width=width, speed=speed)
        else:
            goal = self.GraspGoal(width=width, speed=speed, force=force_n)
            goal.epsilon.inner, goal.epsilon.outer = epsilon_inner_m, epsilon_outer_m
        self.active_gripper_client = command_client
        self.gripper_active = True
        command_client.send_goal(goal)

    def gripper_result(self, stopping=False):
        client = self.stop_client if stopping else self.active_gripper_client
        status = client.get_state()
        if status == 9:
            # Keep uncertain execution active. The core blocks competing commands;
            # grasp cleanup must not send Stop, which would remove holding force.
            raise RuntimeError('Gripper action connection lost')
        if status not in (2, 3, 4, 5, 8, 9):  # actionlib terminal states (including LOST)
            return None
        result = client.get_result()
        if result is None and client is self.grasp_client:
            raise RuntimeError('Native grasp finished without a result; outcome unknown')
        success = status == 3 and result is not None and result.success
        # A failed stop leaves motion uncertain; cleanup must retry the stop.
        if not stopping or success:
            self.gripper_active = False
        return bool(success), (None if success else (getattr(result, 'error', '') or
                                                    'Gripper action failed (status %s)' % status))

    def stop_gripper(self):
        # The existing gripper move server does not implement cancel_goal as a stop.
        self.stop_client.send_goal(self.StopGoal())

    def close(self):
        try:
            # Native Stop removes holding force. Never use it as automatic grasp
            # cleanup or as a force-update workaround (libfranka issue #93).
            if self.gripper_active and self.active_gripper_client is self.move_client:
                self.stop_gripper()
                self.stop_client.wait_for_result(self.rospy.Duration(2.))
        finally:
            self.joint_publisher.unregister()


def next_deadline(previous, completed):
    planned = previous+TARGET_PERIOD
    return planned if planned > completed else completed+TARGET_PERIOD


class ShutdownRequested(KeyboardInterrupt):
    def __init__(self, signum):
        super().__init__('Shutdown signal %d' % signum)
        self.signum = signum


def stop_owned_launch(process):
    if process is None or process.poll() is not None:
        return
    for sig, timeout in [(signal.SIGINT, 8), (signal.SIGTERM, 3), (signal.SIGKILL, 2)]:
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            pass


def run(args):
    # Cold start only: do not attach to another server's controller/master.
    try:
        connection = socket.create_connection(('127.0.0.1', args.ros_port), timeout=.5)
    except ConnectionRefusedError:
        pass
    else:
        connection.close()
        raise RuntimeError('ROS port is already in use. Stop the other ROS master first; this server does not kill existing processes.')
    os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:%d' % args.ros_port
    import rospy
    import rosgraph
    from controller_manager_msgs.srv import ListControllers
    from dynamic_reconfigure.client import Client as ReconfClient
    from franka_msgs.msg import FrankaState
    from geometry_msgs.msg import PoseStamped
    from werkzeug.serving import make_server

    core = FrankaServer(impedance_profile=make_impedance_profile(args.eef_impedance, args.translational_clip),
                        collision_force_threshold=args.collision_force_threshold,
                        collision_torque_scale=args.collision_torque_scale,
                        home_joint_target=args.home_joint_target,
                        gripper_grasp_force=args.gripper_grasp_force)
    http = make_server('127.0.0.1', args.flask_port, create_app(core), threaded=True)
    http_thread = None
    launch = None
    subscriber = publisher = None
    stop = threading.Event()
    def stop_foreground(signum, frame):
        stop.set()
        # Interrupt startup waits as well as the running publish loop. Merely
        # setting the Event cannot interrupt rospy.wait_for_service().
        raise ShutdownRequested(signum)

    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    for sig in old_handlers:
        signal.signal(sig, stop_foreground)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    try:
        launch = subprocess.Popen(launch_command(args), start_new_session=True)
        deadline = time.monotonic()+20
        master = rosgraph.Master('/franka_control_api')
        while not stop.is_set():
            if launch.poll() is not None:
                raise RuntimeError('Impedance roslaunch exited during startup')
            try:
                master.getPid()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError('ROS master startup timed out')
                stop.wait(.1)
        if stop.is_set():
            return 130
        rospy.init_node('franka_control_api', disable_signals=True, disable_rosout=True)

        def state_callback(msg):
            try:
                matrix = np.asarray(msg.O_T_EE).reshape(4, 4, order='F')
                pose = np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]
                errors = [name for name in msg.current_errors.__slots__ if getattr(msg.current_errors, name)]
                age = (rospy.Time.now()-msg.header.stamp).to_sec()
                if not -.1 <= age <= .5:
                    raise RuntimeError('Robot state source timestamp is stale')
                core.update_state({'pose': pose.tolist(), 'q': list(msg.q), 'dq': list(msg.dq),
                                   'robot_time_s': msg.time, 'robot_mode': msg.robot_mode,
                                   'current_errors': errors,
                                   'wrench_K': list(msg.K_F_ext_hat_K)})
            except Exception as exc:
                with core.lock:
                    core.state_error = str(exc)

        subscriber = rospy.Subscriber('/franka_state_controller/franka_states', FrankaState,
                                      state_callback, queue_size=1, tcp_nodelay=True)
        rospy.wait_for_service('/controller_manager/list_controllers', timeout=20)
        list_controllers = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)

        def check_controller():
            controllers = list_controllers().controller
            joint_mode = core.command_status('home')['state'] == 'moving'
            name = HOME_CONTROLLER if joint_mode else 'cartesian_impedance_controller'
            controller_type = HOME_CONTROLLER_TYPE if joint_mode else CONTROLLER
            interface = ('hardware_interface::PositionJointInterface' if joint_mode else
                         'hardware_interface::EffortJointInterface')
            matching = [c for c in controllers if c.name == name
                        and c.state == 'running' and c.type == controller_type]
            if len(matching) != 1 or not any(r.hardware_interface == interface
                                           for r in matching[0].claimed_resources):
                raise RuntimeError('Expected controller is not running: '+name)
            foreign = [node for topic, nodes in master.getSystemState()[0] if topic == POSE_TOPIC
                       for node in nodes if node != rospy.get_name()]
            if foreign:
                raise RuntimeError('Another equilibrium-pose publisher exists: '+str(foreign))

        deadline = time.monotonic()+15
        while not stop.is_set():
            try:
                check_controller()
                core.latest_state()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                stop.wait(.1)
        if stop.is_set():
            return 130
        # These parameters were consumed by FrankaHW before opening the controller.
        # Verify the launch wiring; this is not a live hardware-threshold readback.
        for phase in ('acceleration', 'nominal'):
            for bound, force in (('lower', min(20., args.collision_force_threshold)),
                                 ('upper', args.collision_force_threshold)):
                name = '/franka_control/collision_config/%s_force_thresholds_%s' % (bound, phase)
                if rospy.get_param(name) != [force, force, force, 25., 25., 25.]:
                    raise RuntimeError('Collision startup parameter mismatch: '+name)
            for bound, thresholds in (('lower', list(COLLISION_TORQUE_BASELINE)),
                                      ('upper', scaled_collision_torques(args.collision_torque_scale))):
                name = '/franka_control/collision_config/%s_torque_thresholds_%s' % (bound, phase)
                if rospy.get_param(name) != thresholds:
                    raise RuntimeError('Collision startup parameter mismatch: '+name)
        reconf = ReconfClient('/cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node', timeout=5)
        applied = reconf.update_configuration(dict(core.impedance_profile))
        if any(not math.isclose(float(applied[k]), v, abs_tol=1e-8) for k, v in core.impedance_profile.items()):
            raise RuntimeError('Cartesian impedance parameter mismatch')
        core.initialize_target_once()
        core.command_backend = RosRobotCommands(core.home_joint_target)
        publisher = rospy.Publisher(POSE_TOPIC, PoseStamped, queue_size=1, tcp_nodelay=True)
        http_thread = threading.Thread(target=http.serve_forever, daemon=True)
        http_thread.start()
        print('robot_server_ready: http://127.0.0.1:%d | target 50 Hz | POST /actions | FIFO deltas' %
              args.flask_port, flush=True)
        deadline = time.monotonic()
        last_check = deadline
        while not stop.is_set() and not rospy.is_shutdown():
            if launch.poll() is not None:
                raise RuntimeError('Impedance roslaunch exited')
            if time.monotonic()-last_check >= 1:
                check_controller()
                last_check = time.monotonic()
            core.step_gripper()
            if not core.step_home():
                # Serialize request_home with target publication, so no old Cartesian
                # command can slip through after a home request has been accepted.
                with core.lock:
                    if not core.home_busy():
                        pose = core.next_target()
                        message = PoseStamped()
                        message.header.frame_id = 'panda_link0'
                        message.header.stamp = rospy.Time.now()
                        message.pose.position.x, message.pose.position.y, message.pose.position.z = pose[:3]
                        (message.pose.orientation.x, message.pose.orientation.y,
                         message.pose.orientation.z, message.pose.orientation.w) = pose[3:]
                        publisher.publish(message)
                        core.note_publish()
            deadline = next_deadline(deadline, time.monotonic())
            stop.wait(max(0., deadline-time.monotonic()))
    finally:
        # A second Ctrl+C must not interrupt cleanup and strand the robot driver.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        print('Stopping the robot server and its control processes; wait for the terminal to return.', flush=True)
        try:
            try:
                if http_thread is not None and http_thread.is_alive():
                    http.shutdown()
                    http_thread.join(timeout=2)
                http.server_close()
            finally:
                try:
                    if core.command_backend is not None:
                        core.command_backend.close()
                    if publisher is not None:
                        publisher.unregister()
                    if subscriber is not None:
                        subscriber.unregister()
                    if rospy.core.is_initialized():
                        rospy.signal_shutdown('robot server stopped')
                finally:
                    stop_owned_launch(launch)
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        print('robot_server_stopped', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-ip', '--robot_ip', default='FRANKA_ROBOT_IP')
    parser.add_argument('--flask-port', type=int, default=5003)
    parser.add_argument('--ros-port', type=int, default=11311)
    parser.add_argument('--home-joint-target', type=parse_home_joint_target, default=HOME_JOINT_TARGET,
                        metavar='Q1,Q2,Q3,Q4,Q5,Q6,Q7', help='home joint angles in radians')
    parser.add_argument('--gripper-grasp-force', type=parse_gripper_grasp_force,
                        default=DEFAULT_GRIPPER_GRASP_FORCE, metavar='NEWTONS',
                        help='default native grasp force when force_n is omitted, in [30, 70] N')
    parser.add_argument('--eef-impedance', type=parse_eef_impedance, default='1500,89,150,7',
                        metavar='KT,DT,KR,DR', help='translation stiffness/damping, rotation stiffness/damping')
    parser.add_argument('--translational-clip', type=parse_translational_clip, default=.01,
                        metavar='METERS', help='symmetric X/Y/Z position error clip, in (0, 0.1] m')
    parser.add_argument('--collision-force-threshold', type=parse_collision_force_threshold, default=20.,
                        metavar='NEWTONS', help='Cartesian X/Y/Z collision stop threshold for both motion phases')
    parser.add_argument('--collision-torque-scale', type=parse_collision_torque_scale, default=1.,
                        metavar='RATIO', help='joint collision stop threshold multiplier, finite and >= 1.0')
    parser.add_argument('--check-only', action='store_true', help='print plan without ROS/hardware')
    args = parser.parse_args(argv)
    if not 1 <= args.flask_port <= 65535 or not 1 <= args.ros_port <= 65535:
        parser.error('Invalid port')
    if not LOCAL_LAUNCH.is_file():
        parser.error('Missing launch file: '+str(LOCAL_LAUNCH))
    if args.check_only:
        print(json.dumps({'server': 'franka_robot_server', 'target_publish_hz': TARGET_HZ,
                          'controller_nominal_hz': 1000, 'increment_source': 'http_fifo',
                          'queue_capacity': MAX_QUEUED_ACTIONS,
                          'max_translation_increment_m': MAX_TRANSLATION_INCREMENT,
                          'max_rotation_increment_rad': MAX_ROTATION_INCREMENT,
                          'port': args.flask_port, 'hardware_opened': False,
                          'home_joint_target_rad': list(args.home_joint_target),
                          'gripper_grasp_force_n': args.gripper_grasp_force,
                          'impedance_parameters': make_impedance_profile(args.eef_impedance, args.translational_clip),
                          'collision_force_threshold_n': args.collision_force_threshold,
                          'collision_contact_threshold_n': min(20., args.collision_force_threshold),
                          'collision_torque_scale': args.collision_torque_scale,
                          'collision_torque_thresholds_nm': scaled_collision_torques(args.collision_torque_scale),
                          'launch_command': launch_command(args)}, ensure_ascii=False))
        return 0
    return run(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ShutdownRequested as exc:
        raise SystemExit(128+exc.signum)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print('robot server stopped: '+str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
