#!/usr/bin/env python3
"""Guarded Franka motion server using the 50 Hz delta controller of the robot server.

The HTTP API and job queue are the same as in the motion server of the simulation harness. The robot server
franka_robot_server.py runs next to this process. This process sends bounded base-frame increments at a
nominal maximum of 50 Hz and checks fresh state and force before each increment.
The controller publishes at 50 Hz. Late acknowledgements extend execution;
the client never catches up by batching or enlarging overdue increments.

Only the existing endpoints of the robot server are used. Startup does not move the arm;
home is explicit. Native home/gripper requests are polled until completion.
Their fixed API has no cancellation endpoint: stop waits for them to finish.
Cartesian stop clears the FIFO and holds the target, without stopping the torque
loop. Use this motion server as the sole command producer for that controller.

The Panda arm model is embedded locally. The constant flange-to-EE transform is
inferred once from stationary joint/pose feedback unless --flange-to-ee is given.
The stiffness frame defaults to EE; configure --ee-to-stiffness if the robot uses
a custom K frame.
"""
from __future__ import annotations

import argparse
import http.client
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from numbers import Real
from pathlib import Path
import signal
import sys
import threading
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation as R

LOG = logging.getLogger("motion_server")

# Workspace half-width on each robot-base axis, centred on the initial TCP.
WORKSPACE_HALF_EXTENT_M = 0.50
NATIVE_OPERATION_TIMEOUT_S = 100.0  # Includes the controller's 90 s home deadline.


# Panda link0-to-link8 kinematics, extracted from the available franka_description URDF.
PANDA_ARM_URDF = '<robot name="panda"><link name="panda_link0" /><link name="panda_link1" /><link name="panda_link2" /><link name="panda_link3" /><link name="panda_link4" /><link name="panda_link5" /><link name="panda_link6" /><link name="panda_link7" /><link name="panda_link8" /><joint name="panda_joint1" type="revolute">\n    <origin rpy="0 0 0" xyz="0 0 0.333" />\n    <parent link="panda_link0" />\n    <child link="panda_link1" />\n    <axis xyz="0 0 1" />\n    <limit effort="87.0" lower="-2.8973" upper="2.8973" velocity="2.175" />\n    </joint>\n  \n  <joint name="panda_joint2" type="revolute">\n    <origin rpy="-1.5707963267948966 0 0" xyz="0 0 0" />\n    <parent link="panda_link1" />\n    <child link="panda_link2" />\n    <axis xyz="0 0 1" />\n    <limit effort="87.0" lower="-1.7628" upper="1.7628" velocity="2.175" />\n    </joint>\n  \n  <joint name="panda_joint3" type="revolute">\n    <origin rpy="1.5707963267948966 0 0" xyz="0 -0.316 0" />\n    <parent link="panda_link2" />\n    <child link="panda_link3" />\n    <axis xyz="0 0 1" />\n    <limit effort="87.0" lower="-2.8973" upper="2.8973" velocity="2.175" />\n    </joint>\n  \n  <joint name="panda_joint4" type="revolute">\n    <origin rpy="1.5707963267948966 0 0" xyz="0.0825 0 0" />\n    <parent link="panda_link3" />\n    <child link="panda_link4" />\n    <axis xyz="0 0 1" />\n    <limit effort="87.0" lower="-3.0718" upper="-0.0698" velocity="2.175" />\n    </joint>\n  \n  <joint name="panda_joint5" type="revolute">\n    <origin rpy="-1.5707963267948966 0 0" xyz="-0.0825 0.384 0" />\n    <parent link="panda_link4" />\n    <child link="panda_link5" />\n    <axis xyz="0 0 1" />\n    <limit effort="12.0" lower="-2.8973" upper="2.8973" velocity="2.61" />\n    </joint>\n  \n  <joint name="panda_joint6" type="revolute">\n    <origin rpy="1.5707963267948966 0 0" xyz="0 0 0" />\n    <parent link="panda_link5" />\n    <child link="panda_link6" />\n    <axis xyz="0 0 1" />\n    <limit effort="12.0" lower="-0.0175" upper="3.7525" velocity="2.61" />\n    </joint>\n  \n  <joint name="panda_joint7" type="revolute">\n    <origin rpy="1.5707963267948966 0 0" xyz="0.088 0 0" />\n    <parent link="panda_link6" />\n    <child link="panda_link7" />\n    <axis xyz="0 0 1" />\n    <limit effort="12.0" lower="-2.8973" upper="2.8973" velocity="2.61" />\n    </joint>\n  <joint name="panda_joint8" type="fixed">\n    <origin rpy="0 0 0" xyz="0 0 0.107" />\n    <parent link="panda_link7" />\n    <child link="panda_link8" />\n  </joint>\n  </robot>'


class MotionError(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(detail or reason)


class PlanRejected(MotionError):
    """A proposed Cartesian path was rejected before submitting its targets."""


def number(value, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def vector(value, size: int, label: str) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != size:
        raise ValueError(f"{label} must contain {size} numbers")
    return np.array([number(v, label) for v in value], dtype=float)


def json_safe(value):
    """Expose invalid sensor values as null while keeping diagnostics readable."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def pose_T(pose):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def T_pose(T):
    return np.r_[T[:3, 3], R.from_matrix(T[:3, :3]).as_euler("xyz")]


def pose_error(a, b):
    return (float(np.linalg.norm(a[:3] - b[:3])),
            float((R.from_euler("xyz", a[3:]).inv() *
                   R.from_euler("xyz", b[3:])).magnitude()))


def advance(pose, dp, dr):
    return np.r_[pose[:3] + dp,
                 (R.from_rotvec(dr) * R.from_euler("xyz", pose[3:])).as_euler("xyz")]


def constrain_translation(dp, force, threshold):
    """Both vectors in base frame. Negative F*dp pushes against contact."""
    result = dp.copy()
    blocked = (np.abs(force) > threshold) & (force * dp < 0)
    result[blocked] = 0.0
    return result, blocked


@dataclass
class Settings:
    address: str = "http://127.0.0.1:5003"
    mode: str = "cartesian_impedance"
    hz: float = 50.0
    joint_min: np.ndarray | None = None
    joint_max: np.ndarray | None = None
    joint_margin_rad: float = math.radians(5)
    soft_force_n: float = 10.0
    hard_force_n: float = 50.0
    force_bias_n: np.ndarray = field(default_factory=lambda: np.zeros(3))
    linear_speed: float = 0.05  # m/s; each step must also fit the robot server limit
    angular_speed: float = 0.08
    joint_speed: float = 0.70  # rad/s, estimated from consecutive IK targets
    joint_acceleration: float = 8.0  # rad/s^2, estimated from consecutive IK targets
    max_translation: float = 0.15
    max_rotation: float = math.pi / 2
    state_timeout: float = 0.1
    rpc_timeout: float = 0.4
    ik_m: float = 0.003
    ik_rad: float = 0.03
    queue_size: int = 16
    max_jobs: int = 1000  # Finished results retained, in addition to active/queued work.
    force_sign: float = -1.0  # Negate libfranka's robot-on-environment-positive wrench.
    world_from_base: np.ndarray | None = field(default_factory=lambda: np.eye(3))
    ee_to_stiffness: np.ndarray = field(default_factory=lambda: np.zeros(6))
    flange_to_ee: np.ndarray | None = None
    robot_urdf: str | None = None

    def validate(self):
        for name in ("hz", "joint_margin_rad", "soft_force_n", "hard_force_n",
                     "linear_speed", "angular_speed", "joint_speed", "joint_acceleration",
                     "state_timeout", "rpc_timeout"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.hz != 50.0:
            raise ValueError("This controller publishes targets at exactly 50 Hz")
        if self.mode != "cartesian_impedance":
            raise ValueError("The supplied Franka controller supports only cartesian_impedance mode")
        if self.force_sign not in (-1.0, 1.0):
            raise ValueError("force_sign must be -1 or +1")
        if self.world_from_base is not None:
            m = np.asarray(self.world_from_base, dtype=float)
            if m.shape != (3, 3) or not np.all(np.isfinite(m)) or not np.allclose(m.T @ m, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(m), 1, atol=1e-5):
                raise ValueError("world_from_base must be a proper 3x3 rotation")
            self.world_from_base = m
        if self.hard_force_n > 50:
            raise ValueError("Hard force must not exceed 50 N")
        self.ee_to_stiffness = vector(self.ee_to_stiffness, 6, "ee_to_stiffness")
        if self.flange_to_ee is not None:
            self.flange_to_ee = vector(self.flange_to_ee, 6, "flange_to_ee")
        self.force_bias_n = vector(self.force_bias_n, 3, "force_bias_n")
        self.soft_force_n = self.validate_soft_force(self.soft_force_n)

    def validate_soft_force(self, value):
        value = number(value, "soft_force_n")
        if not 0 < value < self.hard_force_n:
            raise ValueError(f"Require 0 < soft force < hard force ({self.hard_force_n:g} N)")
        if np.linalg.norm(self.force_bias_n) >= value:
            raise ValueError("Force bias must be below soft threshold; calibrate only without contact")
        return value


def validate_command(payload, cfg: Settings):
    if not isinstance(payload, dict):
        raise ValueError("Command must be a JSON object")
    kind = payload.get("type")
    common = {"type"} if kind in ("home", "gripper") else {"type", "duration_s"}
    extra = {"delta": {"translation_m", "rotation_vector_rad", "frame"},
             "pose": {"position_m", "quaternion_xyzw", "frame"},
             "gripper": {"position", "width_m", "speed_m_s", "force_n"}, "home": set()}
    if kind not in extra:
        raise ValueError("type must be delta, pose, gripper, or home")
    if set(payload) - common - extra[kind]:
        raise ValueError(f"Unknown fields: {sorted(set(payload) - common - extra[kind])}")
    cmd = dict(payload)
    if kind == "home":
        return cmd  # The robot server executes the joint home trajectory.
    if kind == "gripper":
        if ("position" in cmd) == ("width_m" in cmd):
            raise ValueError("Specify exactly one of gripper position or width_m")
        if "position" in cmd:
            position = number(cmd.pop("position"), "position")
            if not 0 <= position <= 1:
                raise ValueError("gripper position must be between 0 (closed) and 1 (open)")
            cmd["width_m"] = .08 * position
        width = number(cmd["width_m"], "width_m")
        speed = number(cmd.get("speed_m_s", .05), "speed_m_s")
        if not 0 <= width <= .08 or not .005 <= speed <= .2:
            raise ValueError("Require 0 <= gripper width <= 0.08 m and 0.005 <= speed <= 0.2 m/s")
        cmd.update(width_m=width, speed_m_s=speed)
        if "force_n" in cmd:
            force = number(cmd["force_n"], "force_n")
            if force != 0 and not 30 <= force <= 70:
                raise ValueError("force_n must be 0 for Move or 30..70 N for Grasp")
            cmd["force_n"] = force
        return cmd
    duration = number(cmd.get("duration_s"), "duration_s")
    if not 1 / cfg.hz <= duration <= 120:
        raise ValueError(f"duration_s must be between {1/cfg.hz:g} and 120")
    cmd["duration_s"] = duration
    if kind == "delta":
        dp = vector(cmd.get("translation_m", [0, 0, 0]), 3, "translation_m")
        dr = vector(cmd.get("rotation_vector_rad", [0, 0, 0]), 3, "rotation_vector_rad")
        frame = cmd.setdefault("frame", "base")
        if frame not in ("world", "base", "tool"):
            raise ValueError("frame must be world, base or tool (fixed at command start)")
        cmd.update(translation_m=dp.tolist(), rotation_vector_rad=dr.tolist())
        check_distance(dp, dr, duration, cfg)
    elif kind == "pose":
        if cmd.setdefault("frame", "base") not in ("world", "base"):
            raise ValueError("absolute pose frame must be world or base")
        cmd["position_m"] = vector(cmd.get("position_m"), 3, "position_m").tolist()
        quat = vector(cmd.get("quaternion_xyzw"), 4, "quaternion_xyzw")
        if abs(np.linalg.norm(quat) - 1) > 1e-3:
            raise ValueError("quaternion_xyzw must have unit norm")
        cmd["quaternion_xyzw"] = (quat / np.linalg.norm(quat)).tolist()
    return cmd


def validate_actions(payload, cfg: Settings):
    """Normalize at the API boundary; execution always receives an action list."""
    if isinstance(payload, dict) and "actions" in payload:
        if set(payload) != {"actions"}:
            raise ValueError("A sequence must contain only actions")
        actions = payload["actions"]
    else:
        actions = [payload]  # Existing single-command HTTP clients.
    if not isinstance(actions, list) or not actions:
        raise ValueError("actions must be a nonempty list")
    if len(actions) > cfg.queue_size:
        raise ValueError(f"A sequence can contain at most {cfg.queue_size} actions")
    validated = []
    for index, action in enumerate(actions, 1):
        try:
            command = validate_command(action, cfg)
            if command["type"] == "home" and index != 1:
                raise ValueError("home must be the first action because it is a priority reset")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Action {index}: {exc}") from exc
        validated.append(command)
    return validated


def check_distance(dp, dr, duration, cfg):
    distance, angle = np.linalg.norm(dp), np.linalg.norm(dr)
    if distance > cfg.max_translation or angle > cfg.max_rotation:
        raise ValueError("Command displacement exceeds configured limit")
    for label, amount, limit, unit in (("translation", distance, cfg.linear_speed, "m/s"),
                                       ("rotation", angle, cfg.angular_speed, "rad/s")):
        speed = amount / duration
        if speed > limit + 1e-12:
            raise ValueError(f"Requested {label} speed {speed:.6g} {unit} exceeds "
                             f"{limit:g} {unit}; increase duration_s to at least {amount/limit:.6g} s")


@dataclass
class Plan:
    start: np.ndarray
    goal: np.ndarray
    dp: np.ndarray
    dr: np.ndarray
    count: int
    dt: float


def plan_command(cmd, reference, cfg):
    start = reference.copy()
    if cmd["type"] == "delta":
        dp, dr = np.array(cmd["translation_m"]), np.array(cmd["rotation_vector_rad"])
        if cmd["frame"] == "world":
            # Convert at the command boundary so IK, native feedback and force
            # clipping all continue to use the same controller base frame.
            if cfg.world_from_base is None:
                raise MotionError("world_frame_unconfigured")
            dp, dr = cfg.world_from_base.T @ dp, cfg.world_from_base.T @ dr
        elif cmd["frame"] == "tool":
            rotation = R.from_euler("xyz", start[3:])
            dp, dr = rotation.apply(dp), rotation.apply(dr)
        goal = advance(start, dp, dr)
    elif cmd["type"] == "pose":
        position = np.array(cmd["position_m"])
        rotation = R.from_quat(cmd["quaternion_xyzw"])
        if cmd["frame"] == "world":
            # Origins coincide. An absolute orientation maps tool -> world,
            # so change only its parent frame: R_base_tool = R_base_world R_world_tool.
            if cfg.world_from_base is None:
                raise MotionError("world_frame_unconfigured")
            position = cfg.world_from_base.T @ position
            rotation = R.from_matrix(cfg.world_from_base.T) * rotation
        goal = np.r_[position, rotation.as_euler("xyz")]
        dp = goal[:3] - start[:3]
        dr = (R.from_euler("xyz", goal[3:]) * R.from_euler("xyz", start[3:]).inv()).as_rotvec()
    else:
        raise ValueError("Only delta and pose commands use the trajectory planner")
    check_distance(dp, dr, cmd["duration_s"], cfg)
    count = math.ceil(cmd["duration_s"] * cfg.hz)
    return Plan(start, goal, dp / count, dr / count, count, 1 / cfg.hz)


class Kinematics:
    """Panda URDF FK and bounded differential IK for preflight checks."""

    def __init__(self, urdf, tool, base="panda_link0", flange="panda_link8"):
        root = ET.fromstring(urdf)
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], flange
        while link != base:
            if link not in by_child or len(chain) > 32:
                raise ValueError("Robot description lacks the Panda base/flange chain")
            joint = by_child[link]
            origin = joint.find("origin")
            xyz = [0., 0., 0.] if origin is None else list(map(float, origin.get("xyz", "0 0 0").split()))
            rpy = [0., 0., 0.] if origin is None else list(map(float, origin.get("rpy", "0 0 0").split()))
            transform = pose_T(xyz + rpy)
            axis = joint.find("axis")
            axis = np.array([1., 0., 0.]) if axis is None else np.fromstring(axis.get("xyz"), sep=" ")
            kind = joint.get("type")
            if kind not in ("fixed", "revolute"):
                raise ValueError("Panda chain must contain fixed/revolute joints")
            limit = joint.find("limit")
            bounds = None if kind == "fixed" else (float(limit.get("lower")), float(limit.get("upper")))
            chain.append((transform, axis, bounds))
            link = joint.find("parent").get("link")
        self.chain = chain[::-1]
        limits = [entry[2] for entry in self.chain if entry[2] is not None]
        if len(limits) != 7:
            raise ValueError("Expected seven Panda arm joints")
        self.lower, self.upper = np.array(limits).T
        self.tool = np.asarray(tool, dtype=float)
        if self.tool.shape != (4, 4) or not np.all(np.isfinite(self.tool)):
            raise ValueError("Invalid flange-to-EE transform")

    def transform_jacobian(self, joints):
        q = vector(joints, 7, "joints")
        transform, axes, origins, index = np.eye(4), [], [], 0
        for origin, axis, bounds in self.chain:
            transform = transform @ origin
            if bounds is not None:
                origins.append(transform[:3, 3].copy())
                axes.append(transform[:3, :3] @ axis)
                rotation = np.eye(4)
                rotation[:3, :3] = R.from_rotvec(axis * q[index]).as_matrix()
                transform = transform @ rotation
                index += 1
        transform = transform @ self.tool
        jacobian = np.array([np.r_[np.cross(axis, transform[:3, 3]-origin), axis]
                             for axis, origin in zip(axes, origins)]).T
        return transform, jacobian

    def fk(self, joints):
        return T_pose(self.transform_jacobian(joints)[0])

    def solve(self, pose, joints, dt):
        target = pose_T(pose)
        q = vector(joints, 7, "joints").copy()
        for _ in range(60):
            transform, jacobian = self.transform_jacobian(q)
            error = np.r_[target[:3, 3]-transform[:3, 3],
                          R.from_matrix(target[:3, :3] @ transform[:3, :3].T).as_rotvec()]
            if np.linalg.norm(error[:3]) < 1e-6 and np.linalg.norm(error[3:]) < 1e-5:
                return q
            u, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
            dq = vt.T @ ((singular / (singular * singular + 1e-6)) * (u.T @ error))
            q = np.clip(q + min(1., .04 / max(np.max(np.abs(dq)), 1e-12)) * dq,
                        self.lower + 1e-6, self.upper - 1e-6)
        raise MotionError("ik_failed", "Panda IK did not converge")


class FrankaClient:
    """One HTTP request per operation; never retries an ambiguous POST."""

    def __init__(self, address, timeout):
        url = urlsplit(address)
        if (url.scheme != "http" or not url.hostname or url.username or url.password or
                url.path not in ("", "/") or url.query or url.fragment):
            raise ValueError("Franka address must be http://HOST:PORT")
        self.host = url.hostname
        self.port = url.port or 80
        self.timeout = timeout

    def request(self, method, path, payload=None, timeout=None):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        try:
            headers = {"Content-Type": "application/json"}
            connection.request(method, path, body, headers)
            response = connection.getresponse()
            raw = response.read()
            status = response.status
        except (OSError, http.client.HTTPException) as exc:
            raise MotionError("robot_transport_unknown",
                              f"{method} {path} outcome unknown: {exc}") from exc
        finally:
            connection.close()
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise MotionError("robot_protocol_error", f"{path} returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise MotionError("robot_protocol_error", f"{path} returned a non-object")
        if not 200 <= status < 300:
            raise MotionError("robot_server_rejected", result.get("error", f"HTTP {status}"))
        return result


class RobotBackend:
    """HTTP client for the franka_robot_server controller.

    There is no ownership/session API. Use one producer; changes in the
    controller's counters and operation timestamps are checked locally.
    """

    BUSY = ("pending", "moving", "restoring", "stopping")

    def __init__(self, cfg):
        self.cfg = cfg
        self.client = FrankaClient(cfg.address, cfg.rpc_timeout)
        # Read-only state requests overlap /info on the worker. Each request has
        # its own connection; command POSTs stay serialized on the worker.
        self._state_reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="franka-state")
        self.connected = False
        self.active_mode = None
        self.native_operation = None
        self.target_seq = 0
        self.target_pose = None
        self.applied_actions = None
        self.accepted_actions = None
        self.max_translation_increment = None
        self.max_rotation_increment = None
        self.kin = None
        self.ambiguous = False
        self._counts = None
        self._known_operations = None
        self._robot_time = None
        self._last_advance = None

    def _info(self):
        info = self.client.request("GET", "/info")
        if info.get("server") != "franka_robot_server":
            raise MotionError("franka_service_incompatible", "Unexpected controller on this address")
        counts = {}
        for key in ("published_count", "accepted_actions", "applied_actions", "queue_depth"):
            value = info.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MotionError("robot_protocol_error", "Invalid counter: " + key)
            counts[key] = value
        if self._counts is not None:
            for key in ("published_count", "accepted_actions", "applied_actions"):
                if counts[key] < self._counts[key]:
                    raise MotionError("robot_server_restarted", "Controller counters moved backwards")
        operations = {kind: info.get(kind, {}).get("requested_at") for kind in ("home", "gripper")}
        if self._known_operations is not None:
            for kind in operations:
                if kind != self.native_operation and operations[kind] != self._known_operations[kind]:
                    raise MotionError("another_controller_client", "Unexpected " + kind + " request")
        else:
            self._known_operations = operations
        self._counts = counts
        return info

    def _raw(self):
        began = time.monotonic()
        pending_state = self._state_reader.submit(self.client.request, "GET", "/getstate")
        try:
            info = self._info()
        except Exception:
            pending_state.cancel()
            raise
        raw = pending_state.result()
        robot_time = number(raw.get("robot_time_s"), "robot time")
        if self._robot_time is not None and robot_time < self._robot_time:
            raise MotionError("robot_restarted", "Robot time moved backwards")
        if self._robot_time is None or robot_time > self._robot_time:
            self._robot_time, self._last_advance = robot_time, began
        received = time.monotonic()
        # The fixed API has no source-age field. Track advancing robot time and
        # conservatively include this HTTP round trip; do not invent a source timestamp.
        raw.update(_received=received, state_age_s=received-self._last_advance,
                   _controller=info)
        return raw

    def _normalize(self, raw):
        try:
            pose = vector(raw["pose"], 7, "Franka EE pose")
            joints = vector(raw["q"], 7, "Franka joints")
            velocity = vector(raw["dq"], 7, "Franka joint velocity")
            native_wrench = vector(raw["wrench_K"], 6, "Franka K wrench")
            if np.linalg.norm(pose[3:]) < 1e-8:
                raise ValueError("Invalid zero quaternion")
            quaternion = pose[3:] / np.linalg.norm(pose[3:])
            base_to_ee = R.from_quat(quaternion).as_matrix()
            ee_to_k = pose_T(self.cfg.ee_to_stiffness)
            base_to_k = base_to_ee @ ee_to_k[:3, :3]
            force = self.cfg.force_sign * (base_to_k @ native_wrench[:3])
            moment = (self.cfg.force_sign * (base_to_k @ native_wrench[3:]) +
                      np.cross(base_to_ee @ ee_to_k[:3, 3], force))
            info = raw["_controller"]
        except (KeyError, ValueError, TypeError) as exc:
            raise MotionError("invalid_feedback", str(exc)) from exc
        native_busy = any(info.get(kind, {}).get("state") in self.BUSY for kind in ("home", "gripper"))
        return dict(mode=self.active_mode, target_seq=self.target_seq,
                    state_age_s=raw["state_age_s"], _received=raw["_received"], _controller=info,
                    feedback_age_basis="robot_time_progress_and_http_round_trip",
                    joint_pos_real=joints.tolist(), joint_vel_real=velocity.tolist(),
                    cart_pos_real=np.r_[pose[:3], R.from_quat(quaternion).as_euler("xyz")].tolist(),
                    ext_wrench=np.r_[force, moment].tolist(), raw_wrench_K=native_wrench.tolist(),
                    wrench_reference="EE", wrench_frame="base",
                    ee_to_stiffness=self.cfg.ee_to_stiffness.tolist(),
                    in_realtime_loop=info.get("target_pose") is not None,
                    fault=None, stop_confirmed=info["queue_depth"] == 0 and not native_busy,
                    native_stop_confirmed=False, stop_mode="hold_target",
                    applied_actions=info["applied_actions"], published_count=info["published_count"],
                    queue_depth=info["queue_depth"], robot_time_s=raw["robot_time_s"],
                    home=info.get("home"), gripper=info.get("gripper"))

    def state(self):
        return self._normalize(self._raw())

    @staticmethod
    def _pose_matches(first, second, position_tolerance=1e-6, angle_tolerance=1e-5):
        return (np.linalg.norm(first[:3]-second[:3]) <= position_tolerance and
                (R.from_quat(first[3:])*R.from_quat(second[3:]).inv()).magnitude() <= angle_tolerance)

    def start(self):
        info = self._info()
        if number(info.get("target_publish_hz"), "target_publish_hz") != self.cfg.hz:
            raise MotionError("frequency_mismatch", "Expected the controller's 50 Hz target loop")
        if not all(isinstance(info.get(kind), dict) for kind in ("home", "gripper")):
            raise MotionError("franka_service_incompatible", "Expected the downloaded home/gripper controller")
        # Contact reporting and collision stopping have separate thresholds.
        collision_limit = number(info["collision_force_threshold_n"], "collision threshold")
        if self.cfg.hard_force_n > collision_limit:
            raise MotionError("force_configuration_mismatch",
                              f"Software hard limit {self.cfg.hard_force_n:g} N exceeds "
                              f"controller collision threshold {collision_limit:g} N")
        raw = self._raw()
        first_time = raw["robot_time_s"]
        deadline = time.monotonic() + 5.
        while True:
            info = raw["_controller"]
            if any(info[kind].get("state") in self.BUSY for kind in ("home", "gripper")) or info["queue_depth"]:
                raise MotionError("controller_busy", "Wait for the controller's existing work to finish")
            if raw["state_age_s"] > self.cfg.state_timeout:
                raise MotionError("stale_feedback")
            stationary = np.max(np.abs(vector(raw["dq"], 7, "joint velocity"))) <= .02
            if stationary and raw["robot_time_s"] > first_time:
                break
            if time.monotonic() >= deadline:
                raise MotionError("robot_moving", "Arm did not settle with advancing feedback")
            time.sleep(1/self.cfg.hz)
            raw = self._raw()
        measured = vector(raw["pose"], 7, "measured EE pose")
        target = vector(info["target_pose"], 7, "controller target")
        if not self._pose_matches(target, measured, .01, .1):
            raise MotionError("target_measured_mismatch", "Controller target is too far from measured pose")
        if self.kin is None:
            urdf = PANDA_ARM_URDF if self.cfg.robot_urdf is None else Path(self.cfg.robot_urdf).read_text()
            kin = Kinematics(urdf, np.eye(4))
            if self.cfg.flange_to_ee is None:
                measured_T = np.eye(4)
                measured_T[:3, :3] = R.from_quat(measured[3:]).as_matrix()
                measured_T[:3, 3] = measured[:3]
                flange_T = kin.transform_jacobian(raw["q"])[0]
                kin.tool = np.linalg.inv(flange_T) @ measured_T
                LOG.info("Inferred constant flange-to-EE transform from stationary q/pose: %s",
                         T_pose(kin.tool).tolist())
            else:
                kin.tool = pose_T(self.cfg.flange_to_ee)
            self.kin = kin
        self.max_translation_increment = number(info["max_translation_increment_m"], "translation limit")
        self.max_rotation_increment = number(info["max_rotation_increment_rad"], "rotation limit")
        self.target_pose = target
        self.applied_actions = info["applied_actions"]
        self.accepted_actions = info["accepted_actions"]
        self.target_seq = 0
        self.ambiguous = False
        self.connected, self.active_mode = True, "cartesian_impedance"
        return self._normalize(raw)

    def check_held(self, state):
        """Check an observed idle controller still holds our acknowledged target."""
        if (not self.connected or self.target_pose is None or self.ambiguous or
                self.native_operation is not None):
            raise MotionError("control_not_ready")
        info = state["_controller"]
        if (info["queue_depth"] or info["applied_actions"] != self.applied_actions or
                info["accepted_actions"] != self.accepted_actions or
                any(info[k].get("state") in self.BUSY for k in ("home", "gripper")) or
                not self._pose_matches(vector(info["target_pose"], 7, "held target"), self.target_pose)):
            raise MotionError("another_controller_client", "Robot server targets or FIFO changed unexpectedly")
        return info

    def send(self, pose, joints, dt, state):
        if not self.connected or self.target_pose is None:
            raise MotionError("control_not_ready")
        target = vector(pose, 6, "target EE pose")
        next_target = np.r_[target[:3], R.from_euler("xyz", target[3:]).as_quat()]
        translation = next_target[:3]-self.target_pose[:3]
        rotation = (R.from_quat(next_target[3:])*R.from_quat(self.target_pose[3:]).inv()).as_rotvec()
        if (np.linalg.norm(translation) > self.max_translation_increment+1e-12 or
                np.linalg.norm(rotation) > self.max_rotation_increment+1e-12):
            raise MotionError("franka_increment_limit")
        before = self.check_held(state)
        try:
            result = self.client.request("POST", "/actions",
                                         {"actions": [np.r_[translation, rotation].tolist()]})
        except MotionError as exc:
            self.ambiguous = exc.reason != "robot_server_rejected"
            raise
        if result.get("accepted") != 1 or result.get("queue_depth") not in (0, 1):
            self.ambiguous = True
            raise MotionError("target_not_acknowledged")
        deadline = time.monotonic()+self.cfg.rpc_timeout
        while time.monotonic() < deadline:
            # Collect the next force/state sample while checking publication.
            # This sample guards the next increment, avoiding a second pair of
            # feedback RPCs between consecutive 20 ms controller ticks.
            raw = self._raw()
            info = raw["_controller"]
            if (info["accepted_actions"] != self.accepted_actions+1 or
                    info["applied_actions"] not in (self.applied_actions, self.applied_actions+1) or
                    info["queue_depth"] not in (0, 1)):
                raise MotionError("another_controller_client", "Unexpected action counters")
            if info["applied_actions"] == self.applied_actions+1 and info["published_count"] > before["published_count"]:
                # The downloaded run loop holds core.lock across next_target(),
                # publish() and note_publish(); /info takes the same lock.
                actual = vector(info["target_pose"], 7, "published target")
                if not self._pose_matches(actual, next_target) or info["queue_depth"]:
                    raise MotionError("target_mismatch")
                self.target_pose = next_target
                self.applied_actions, self.accepted_actions = info["applied_actions"], info["accepted_actions"]
                self.target_seq += 1
                acknowledged = self._normalize(raw)
                acknowledged["accepted"] = True
                return acknowledged
            time.sleep(.002)
        self.ambiguous = True
        raise MotionError("target_publication_unknown", "No controller publication acknowledgement")

    def _operation(self, kind, payload, timeout, on_feedback=None):
        if not self.connected:
            raise MotionError("control_not_ready")
        before = self._info()
        if before["queue_depth"] or any(before[k].get("state") in self.BUSY for k in ("home", "gripper")):
            raise MotionError("controller_busy")
        self.native_operation = kind
        if kind == "home":
            self.active_mode = "joint_position"
        try:
            accepted = self.client.request("POST", "/"+kind, payload)
        except MotionError as exc:
            self.ambiguous = exc.reason != "robot_server_rejected"
            if not self.ambiguous:
                self.native_operation = None
                self.active_mode = "cartesian_impedance"
            raise
        stamp = accepted.get("requested_at")
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            self.ambiguous = True
            raise MotionError("operation_not_acknowledged", "Missing native request timestamp")
        self._known_operations[kind] = stamp
        deadline = time.monotonic()+timeout
        success_publish_count, guard_error = None, None
        while time.monotonic() < deadline:
            result = self.client.request("GET", "/"+kind)
            if result.get("requested_at") != stamp:
                raise MotionError("another_controller_client", "Native operation was replaced")
            phase = result.get("state")
            if phase == "failed":
                self.native_operation = None
                self.active_mode = "cartesian_impedance"
                raise MotionError(kind+"_failed", result.get("error") or "Native operation failed")
            raw = self._raw()
            state = self._normalize(raw)
            if on_feedback is not None:
                try:
                    on_feedback(state)
                except MotionError as exc:
                    guard_error = guard_error or exc
            if phase == "succeeded":
                if kind == "home":
                    info = raw["_controller"]
                    if success_publish_count is None:
                        success_publish_count = info["published_count"]
                    if info["published_count"] <= success_publish_count:
                        time.sleep(1/self.cfg.hz)
                        continue
                    self.target_pose = vector(info["target_pose"], 7, "post-home target")
                    self.applied_actions, self.accepted_actions = info["applied_actions"], info["accepted_actions"]
                self.native_operation = None
                self.active_mode = "cartesian_impedance"
                if guard_error is not None:
                    raise guard_error
                state = self._normalize(raw)
                state["operation"] = result
                return state
            if phase not in self.BUSY:
                raise MotionError("robot_protocol_error", "Unexpected "+kind+" state: "+str(phase))
            # The fixed API has no cancellation endpoint for either operation.
            # Keep polling to completion before honoring stop or further commands.
            time.sleep(1/self.cfg.hz)
        self.ambiguous = True
        raise MotionError(kind+"_outcome_unknown", "Native completion deadline exceeded")

    def send_gripper(self, width_m, speed_m_s, on_feedback=None, *, force_n=None):
        payload = {"width_m": width_m, "speed_m_s": speed_m_s}
        if force_n is not None:
            payload["force_n"] = force_n
        result = self._operation("gripper", payload, .08/speed_m_s+9., on_feedback)
        if force_n is not None and force_n > 0:
            operation = result["operation"]
            if operation.get("mode") != "grasp" or operation.get("force_n") != force_n:
                raise MotionError("gripper_protocol_error", "Controller did not confirm requested force grasp")
        return result

    def home(self, on_feedback=None):
        deadline = time.monotonic()+5.
        while np.max(np.abs(self.state()["joint_vel_real"])) > .02:
            if time.monotonic() >= deadline:
                raise MotionError("robot_moving", "Arm did not settle before home")
            time.sleep(1/self.cfg.hz)
        return self._operation("home", {}, 95., on_feedback)

    def stop(self):
        info = self._info()
        if any(info[k].get("state") in self.BUSY for k in ("home", "gripper")):
            raise MotionError("stop_unconfirmed", "Native home/gripper is still active; FIFO clear cannot stop it")
        self.client.request("POST", "/actions/clear", {})
        info = self._info()
        if info["queue_depth"] or any(info[k].get("state") in self.BUSY for k in ("home", "gripper")):
            raise MotionError("stop_unconfirmed", "Controller is still busy")
        # A lost POST response leaves the operation ID unknown. Once the fixed
        # controller confirms it is idle, explicit stop/recovery may adopt its
        # final timestamp; the lost operation is never retried or called successful.
        if self.native_operation is not None:
            kind = self.native_operation
            self._known_operations[kind] = info[kind].get("requested_at")
        self.connected, self.active_mode = False, None
        self.target_pose = None
        self.native_operation = None
        state = self.state()
        state.update(stop_confirmed=True, native_stop_confirmed=False,
                     stop_mode="hold_target", previous_command_outcome_unknown=self.ambiguous)
        return state

    def close(self):
        self._state_reader.shutdown(wait=True, cancel_futures=True)


class MotionService:
    """One worker owns backend calls; HTTP threads queue work or update idle settings."""
    def __init__(self, cfg):
        cfg.validate()
        self.cfg = cfg
        self.startup_soft_force_n = cfg.soft_force_n
        self.backend = None
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.pending = deque()
        self.jobs = {}
        self.active = None
        self.fault = None
        self.stopping = False
        self.latest = None
        self.last_result = None
        self.reference = None
        self.joint_reference = None
        self.lower = self.upper = None
        self.workspace_origin = None
        self.workspace_min = None
        self.workspace_max = None
        self.cancel = threading.Event()
        self.shutdown_event = threading.Event()
        self.control_request = None
        self.control_active = None
        self.homed = False  # Internal readiness flag also used for measured-pose startup.
        self.home_completed = False
        self.home_pending = None
        self.startup_home_id = None
        self.worker = threading.Thread(target=self._run, name="motion-worker", daemon=False)

    def start(self):
        self.worker.start()

    @staticmethod
    def _new_job(commands):
        # A fresh reference tracks this submission; it never deduplicates motion.
        actions = [dict(index=index, command=cmd, status="queued", steps_sent=0,
                        clipped_steps=0, discarded_translation_m=[0.0]*3)
                   for index, cmd in enumerate(commands, 1)]
        return dict(id=str(uuid.uuid4()), actions=actions, status="queued",
                    active_action=None, actions_completed=0, submitted_at=time.time())

    def enqueue_command(self, payload):
        commands = validate_actions(payload, self.cfg)
        job = self._new_job(commands)
        with self.condition:
            if (not self.worker.is_alive() or self.fault or self.control_request or self.control_active or
                    self.shutdown_event.is_set()):
                raise MotionError("not_ready", self.fault or "Stop/recovery in progress")
            if commands[0]["type"] == "home":
                if self.backend is not None and self.backend.native_operation == "home":
                    raise MotionError("busy", "A native home is already running")
                # A reset is a priority request, not a trajectory queued behind work.
                self._discard_pending("home_requested")
                self._discard_home("superseded_by_home")
                self.jobs[job["id"]] = job
                self.home_pending = job["id"]
                self.homed = False
                self.cancel.set()
            else:
                if not self.homed or self.home_pending or self.cancel.is_set() or self.stopping:
                    raise MotionError("not_ready", "Franka measured-pose initialization has not completed")
                queued_actions = sum(len(self.jobs[ident]["actions"]) for ident in self.pending)
                if queued_actions + len(commands) > self.cfg.queue_size:
                    raise MotionError("queue_full")
                self.jobs[job["id"]] = job
                self.pending.append(job["id"])
            self.condition.notify_all()
            return self._public(job)

    def set_soft_force(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"soft_force_n"}:
            raise ValueError("Expected a JSON object containing only soft_force_n")
        with self.condition:
            value = self.cfg.validate_soft_force(payload["soft_force_n"])
            if (not self.worker.is_alive() or not self.homed or self.fault or
                    self.home_pending is not None or self.control_request is not None or self.control_active is not None or
                    self.cancel.is_set() or self.stopping or self.shutdown_event.is_set()):
                raise MotionError("not_ready", "Soft force can change only when the motion server is ready and idle")
            if self.active is not None or self.pending:
                raise MotionError("busy", "Wait for all active and queued actions to finish before changing soft force")
            previous = self.cfg.soft_force_n
            self.cfg.soft_force_n = value
            LOG.info("Soft force threshold changed from %g N to %g N", previous, value)
            return dict(status="updated", previous_soft_force_n=previous, soft_force_n=value,
                        startup_soft_force_n=self.startup_soft_force_n, hard_force_n=self.cfg.hard_force_n)

    def _discard_home(self, reason):
        if self.home_pending is not None:
            job = self.jobs[self.home_pending]
            self._cancel_remaining(job, reason)
            job.update(status="cancelled", reason=reason, finished_at=time.time())
            self.home_pending = None
            self._prune_history()

    def _prune_history(self):
        """Expire old finished results without blocking new work or removing live jobs."""
        with self.lock:
            finished = [ident for ident, job in self.jobs.items()
                        if ident != self.active and
                        job["status"] in ("completed", "failed", "cancelled")]
            for ident in finished[:max(0, len(finished) - self.cfg.max_jobs)]:
                del self.jobs[ident]

    @staticmethod
    def _public(job):
        return json.loads(json.dumps(json_safe(job)))

    def status(self, ident=None):
        with self.lock:
            if ident is not None:
                if ident not in self.jobs:
                    raise KeyError(ident)
                return self._public(self.jobs[ident])
            return dict(mode=self.cfg.mode, robot_address=self.cfg.address, target_hz=self.cfg.hz,
                        model_source=self.cfg.robot_urdf or "embedded_panda_arm",
                        tool_transform_source="configured" if self.cfg.flange_to_ee is not None else "inferred_at_start",
                        ee_to_stiffness=self.cfg.ee_to_stiffness.tolist(),
                        soft_force_n=self.cfg.soft_force_n,
                        startup_soft_force_n=self.startup_soft_force_n,
                        hard_force_n=self.cfg.hard_force_n,
                        linear_speed_limit_m_s=self.cfg.linear_speed,
                        controller_linear_speed_limit_m_s=(
                            None if self.backend is None or self.backend.max_translation_increment is None
                            else self.backend.max_translation_increment*self.cfg.hz),
                        joint_speed_limit_rad_s=self.cfg.joint_speed,
                        joint_acceleration_limit_rad_s2=self.cfg.joint_acceleration,
                        active_mode=None if self.backend is None else self.backend.active_mode,
                        homed=self.home_completed, initialized_from_measured_pose=self.homed,
                        native_operation=None if self.backend is None else self.backend.native_operation,
                        native_operations_interruptible=False,
                        controller_requires_single_producer=True,
                        native_stop_available=False, stop_mode="hold_target",
                        ready=self.worker.is_alive() and self.homed and self.reference is not None and not (self.home_pending or self.fault or self.control_active or
                              self.cancel.is_set() or self.stopping or self.shutdown_event.is_set()),
                        pending_home=self.home_pending, startup_home_id=self.startup_home_id,
                        active=self.active, queued=list(self.pending), fault=self.fault,
                        stop_requested=bool(self.cancel.is_set() or self.control_request or self.control_active), stopping=self.stopping,
                        worker_alive=self.worker.is_alive(),
                        last_result=self.last_result,
                        feedback=self.latest,
                        reference_frame="base",
                        world_from_base=None if self.cfg.world_from_base is None else self.cfg.world_from_base.tolist(),
                        reference=None if self.reference is None else self.reference.tolist(),
                        workspace=None if self.workspace_origin is None else {
                            "frame": "base",
                            "origin_m": self.workspace_origin.tolist(),
                            "min_m": self.workspace_min.tolist(),
                            "max_m": self.workspace_max.tolist(),
                        },
                        job_ids_retained=len(self.jobs))

    def request_stop(self, recover=False):
        with self.condition:
            self.cancel.set()
            self.control_request = "recover" if recover else "stop"
            self._discard_pending("recovery_requested" if recover else "stop_requested")
            self._discard_home("recovery_requested" if recover else "stop_requested")
            self.condition.notify_all()
        return {"status": "requested", "action": "recover" if recover else "stop"}

    def _discard_pending(self, reason):
        while self.pending:
            job = self.jobs[self.pending.popleft()]
            self._cancel_remaining(job, reason)
            job.update(status="cancelled", reason=reason, finished_at=time.time())
        self._prune_history()

    def _cancel_remaining(self, job, reason):
        with self.lock:
            for action in job["actions"]:
                if action["status"] not in ("completed", "failed", "cancelled"):
                    action.update(status="cancelled", reason=reason, finished_at=time.time())

    def _update(self, job, **values):
        with self.lock:
            job.update(values)

    def _read(self):
        state = self.backend.state()
        self._record_feedback(state)
        return state

    def _record_feedback(self, state):
        with self.lock:
            self.latest = json_safe({k: v for k, v in state.items() if not k.startswith("_")})
            self.latest["observed_at"] = time.time()

    def _check_state(self, state, *, cancel=True):
        if cancel and self.cancel.is_set():
            raise MotionError("cancelled")
        age = state.get("state_age_s")
        if (age is None or not math.isfinite(age) or age < 0 or
                age + time.monotonic() - state["_received"] > self.cfg.state_timeout):
            raise MotionError("stale_feedback")
        if not state.get("in_realtime_loop") or state.get("fault"):
            raise MotionError("control_loop_stopped")
        try:
            q = vector(state["joint_pos_real"], 7, "joint feedback")
            pose = vector(state["cart_pos_real"], 6, "TCP feedback")
            force = vector(state["ext_wrench"], 6, "wrench feedback")
        except (ValueError, KeyError, TypeError) as exc:
            raise MotionError("invalid_feedback", str(exc)) from exc
        return pose, q, force

    def _initialize_workspace(self, pose):
        # Capture at initialization or successful explicit home. Stop/recover keeps the box.
        with self.lock:
            if self.workspace_origin is not None:
                return
            self.workspace_origin = pose[:3].copy()
            self.workspace_min = self.workspace_origin - WORKSPACE_HALF_EXTENT_M
            self.workspace_max = self.workspace_origin + WORKSPACE_HALF_EXTENT_M
        LOG.info("Workspace fixed around initial TCP %s: min %s, max %s (metres)",
                 self.workspace_origin.tolist(), self.workspace_min.tolist(),
                 self.workspace_max.tolist())

    def _bounds(self, pose, joints):
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(joints)):
            raise MotionError("nonfinite_target")
        if self.workspace_min is None or self.workspace_max is None:
            raise MotionError("workspace_uninitialized")
        if np.any(pose[:3] < self.workspace_min) or np.any(pose[:3] > self.workspace_max):
            raise MotionError("cartesian_boundary")
        if joints.shape != (7,) or np.any(joints < self.lower) or np.any(joints > self.upper):
            raise MotionError("joint_boundary")

    def _hard(self, state, candidate=None, candidate_q=None, *, cancel=True):
        pose, q, wrench = self._check_state(state, cancel=cancel)
        # Raw force ALWAYS participates: a software bias cannot conceal a hard stop.
        force = wrench[:3]
        if max(np.linalg.norm(force), np.linalg.norm(force-self.cfg.force_bias_n)) >= self.cfg.hard_force_n:
            raise MotionError("hard_force_limit")
        self._bounds(pose, q)
        if candidate is not None:
            self._bounds(candidate, candidate_q)
        # Impedance/contact can separate measured pose from the reference.
        # Bounds apply to each independently; their difference is not an abort.
        return pose, q, wrench

    def _ensure_context(self):
        if self.cancel.is_set():
            raise MotionError("cancelled")
        if not self.backend.connected or self.reference is None or self.joint_reference is None:
            if not self.backend.connected:
                self.backend.start()
            # Wait for actual native samples; a loop-off cache is not initialization.
            deadline = time.monotonic() + 1.0
            while True:
                if self.cancel.is_set():
                    raise MotionError("cancelled")
                state = self._read()
                if state.get("state_age_s") is not None:
                    pose, _, _ = self._check_state(state)
                    break
                if time.monotonic() > deadline:
                    raise MotionError("no_fresh_feedback")
                self.cancel.wait(0.01)
            kin = self.backend.kin
            self.lower = kin.lower + self.cfg.joint_margin_rad
            self.upper = kin.upper - self.cfg.joint_margin_rad
            if self.cfg.joint_min is not None:
                self.lower = np.maximum(self.lower, vector(self.cfg.joint_min, 7, "joint_min"))
            if self.cfg.joint_max is not None:
                self.upper = np.minimum(self.upper, vector(self.cfg.joint_max, 7, "joint_max"))
            if np.any(self.lower >= self.upper):
                raise MotionError("invalid_joint_limits")
            self._initialize_workspace(pose)
            pose, q, _ = self._hard(state)
            error = pose_error(kin.fk(q), pose)
            if error[0] > self.cfg.ik_m or error[1] > self.cfg.ik_rad:
                raise MotionError("model_tool_mismatch", f"FK vs feedback error: {error}")
            # Delta accumulation belongs to the controller's held target.
            # Contact may leave measured EE behind it; initialization/recovery
            # must not turn that difference into an unrequested first increment.
            held = self.backend.target_pose
            reference = np.r_[held[:3], R.from_quat(held[3:]).as_euler("xyz")]
            reference_q = kin.solve(reference, q, 1/self.cfg.hz)
            self._hard(self._read(), reference, reference_q)
            with self.lock:
                self.reference, self.joint_reference = reference, reference_q
        state = self._read()
        self._hard(state, self.reference, self.joint_reference)
        return state

    def _solve(self, candidate, q_previous, dt, previous_velocity=None):
        q = self.backend.kin.solve(candidate, q_previous, dt)
        self._bounds(candidate, q)
        error = pose_error(self.backend.kin.fk(q), candidate)
        if error[0] > self.cfg.ik_m or error[1] > self.cfg.ik_rad:
            raise MotionError("ik_residual", f"IK residual {error}")
        velocity = (q - q_previous) / dt
        fastest = int(np.argmax(np.abs(velocity)))
        if abs(velocity[fastest]) > self.cfg.joint_speed:
            raise MotionError("joint_speed_limit",
                              f"joint_speed_limit: IK joint {fastest+1}, "
                              f"{abs(velocity[fastest]):.6f} > {self.cfg.joint_speed:g} rad/s "
                              f"(dt={dt:.6f} s)")
        if previous_velocity is not None:
            # This is the finite difference of predicted IK joint targets,
            # not acceleration measured by the robot controller.
            acceleration = (velocity - previous_velocity) / dt
            fastest = int(np.argmax(np.abs(acceleration)))
            if abs(acceleration[fastest]) > self.cfg.joint_acceleration:
                raise MotionError("joint_acceleration_limit",
                                  f"joint_acceleration_limit: IK joint {fastest+1}, "
                                  f"{abs(acceleration[fastest]):.6f} > {self.cfg.joint_acceleration:g} rad/s^2 "
                                  f"(velocity {previous_velocity[fastest]:.6f} -> {velocity[fastest]:.6f} rad/s, "
                                  f"dt={dt:.6f} s)")
        return q, velocity

    def _preflight(self, plan):
        # Reject an unsupported plan before any action POST. The robot server may
        # have a lower increment cap than this motion server's speed cap.
        for label, delta, limit, unit in (
                ("translation", plan.dp, self.backend.max_translation_increment, "m"),
                ("rotation", plan.dr, self.backend.max_rotation_increment, "rad")):
            increment = float(np.linalg.norm(delta))
            if increment > limit + 1e-12:
                minimum_duration = increment*plan.count/(limit*self.cfg.hz)
                raise PlanRejected(
                    "franka_increment_limit",
                    f"Planned {label} increment {increment:.6g} {unit} exceeds the "
                    f"robot controller limit {limit:g} {unit}/tick "
                    f"({limit*self.cfg.hz:g} {unit}/s at {self.cfg.hz:g} Hz); "
                    f"increase duration_s to at least {minimum_duration:.6g} s "
                    "or update the limit of the robot server")
        pose, q = plan.start.copy(), self.joint_reference.copy()
        velocity = np.zeros(7)
        last_check = 0.0
        # Check every proposed point, including the final one; keep force/stop
        # monitoring and feedback checks active during expensive planning.
        for index in range(plan.count):
            if self.cancel.is_set():
                raise MotionError("cancelled")
            if time.monotonic() - last_check >= 1/self.cfg.hz:
                self._hard(self._read(), self.reference, self.joint_reference)
                last_check = time.monotonic()
            pose = advance(pose, plan.dp, plan.dr)
            try:
                q, velocity = self._solve(pose, q, plan.dt, velocity)
            except MotionError as exc:
                raise PlanRejected(exc.reason, f"Preflight step {index+1}/{plan.count}: {exc}") from exc

    def _wait_until(self, deadline):
        if self.cancel.wait(max(0, deadline-time.monotonic())):
            raise MotionError("cancelled")

    def _dispatch(self, dp, dr, dt, previous_velocity, state=None):
        # The preceding publication check supplies a new measured sample for
        # each next step. Its original timestamp is retained and checked after
        # any scheduling wait and IK; action boundaries fetch another sample.
        if state is None:
            state = self._read()
        _, _, wrench = self._check_state(state)
        # Soft constraint precedes final hard checks. Removed increments are
        # discarded; the next step starts at the accepted constrained reference.
        safe_dp, blocked = constrain_translation(dp, wrench[:3]-self.cfg.force_bias_n,
                                                   self.cfg.soft_force_n)
        candidate = advance(self.reference, safe_dp, dr)
        self._hard(state)
        q, velocity = self._solve(candidate, self.joint_reference, dt, previous_velocity)
        self._hard(state, candidate, q)  # Includes age again after IK.
        if self.cancel.is_set():
            raise MotionError("cancelled")
        result = self.backend.send(candidate, q, dt, state)
        # Commit reference only after accepted ACK; uncertainty aborts, never retries.
        with self.lock:
            self.reference, self.joint_reference = candidate.copy(), q.copy()
        self._last_ack = result
        self._record_feedback(result)
        return dp-safe_dp, bool(np.any(blocked)), velocity

    def _execute(self, job):
        self._update(job, status="planning", started_at=time.time(), endpoint_frame="base")
        self._ensure_context()
        cmd = job["command"]
        try:
            plan = plan_command(cmd, self.reference, self.cfg)
        except (ValueError, MotionError) as exc:
            reason = exc.reason if isinstance(exc, MotionError) else "invalid_plan"
            raise PlanRejected(reason, str(exc)) from exc
        self._preflight(plan)
        self._update(job, status="running", steps_total=plan.count, step_duration_s=plan.dt,
                     nominal_duration_s=plan.count*plan.dt, timing_policy="extend_without_catch_up",
                     scheduling_delay_s=0., max_step_lateness_s=0.,
                     requested_endpoint=plan.goal.tolist())
        start = time.monotonic()
        velocity = np.zeros(7)
        feedback = None
        discarded = np.zeros(3)
        clipped = 0
        scheduling_delay = max_lateness = 0.
        # Each step covers one nominal interval. Schedule from its actual start,
        # so RPC/OS jitter extends the action instead of causing catch-up bursts
        # or a software fault. Feedback age and communication deadlines still
        # apply, and every single increment must be acknowledged before the next.
        due = start
        for index in range(plan.count):
            self._wait_until(due)
            step_started = time.monotonic()
            lateness = max(0., step_started-due)
            scheduling_delay += lateness
            max_lateness = max(max_lateness, lateness)
            try:
                removed, blocked, velocity = self._dispatch(plan.dp, plan.dr, plan.dt, velocity, feedback)
                feedback = self._last_ack
            except MotionError as exc:
                raise MotionError(exc.reason, f"Execution step {index+1}/{plan.count}: {exc}") from exc
            discarded += removed
            clipped += int(blocked)
            self._update(job, steps_sent=index+1, clipped_steps=clipped,
                         discarded_translation_m=discarded.tolist(),
                         scheduling_delay_s=scheduling_delay, max_step_lateness_s=max_lateness)
            self._hard(self._last_ack, self.reference, self.joint_reference)
            due = step_started + plan.dt
        self._wait_until(due)
        # Completion is exhaustion of the nominal steps, including clipped ones.
        # Check protection once more after the final interval, without waiting
        # for measured pose or rotation arrival.
        measured, _, _ = self._hard(self._read(), self.reference, self.joint_reference)
        self._update(job, status="completed", completion_basis="timed_steps",
                     execution_elapsed_s=time.monotonic()-start, finished_at=time.time(),
                     accepted_endpoint=self.reference.tolist(), measured_endpoint=measured.tolist())
        LOG.info("Action %d completed: %d steps; %d force-constrained; %.3f s scheduling delay",
                 job["index"], plan.count, clipped, scheduling_delay)

    def _execute_gripper(self, job):
        self._update(job, status="running", started_at=time.time(), steps_total=1)
        self._ensure_context()
        self._hard(self._read(), self.reference, self.joint_reference)
        cmd = job["command"]
        def observe(state):
            self._record_feedback(state)
            self._hard(state, self.reference, self.joint_reference, cancel=False)
        result = self.backend.send_gripper(cmd["width_m"], cmd["speed_m_s"], observe,
                                           force_n=cmd.get("force_n"))
        self._update(job, steps_sent=1, controller_operation=result["operation"])
        self._hard(result, self.reference, self.joint_reference)
        self._update(job, status="completed", completion_basis="native_gripper_result",
                     width_m=cmd["width_m"], finished_at=time.time())
        LOG.info("Action %d completed: native gripper %s, requested width %.4f m",
                 job["index"], result["operation"].get("mode", "move"), cmd["width_m"])

    def _execute_home(self, job):
        self._update(job, status="running", started_at=time.time(),
                     control_mode="joint_position", method="controller_joint_home")
        self._ensure_context()
        self._hard(self._read(), self.reference, self.joint_reference)
        self.home_completed = False
        LOG.info("Requesting native joint home; a stop request waits for this home to finish.")
        result = self.backend.home(self._record_feedback)
        target = vector(result["operation"]["target_q"], 7, "home target")
        measured = vector(result["joint_pos_real"], 7, "home measured joints")
        error = float(np.max(np.abs(target - measured)))
        if error >= .005:
            raise MotionError("home_target_mismatch", "Native success did not match measured joints")
        self._update(job, home_joint_target_rad=target.tolist(),
                     measured_joints_rad=measured.tolist(), home_joint_error_rad=error,
                     controller_operation=result["operation"])
        with self.lock:
            self.reference = self.joint_reference = None
            self.workspace_origin = self.workspace_min = self.workspace_max = None
            self.home_completed = True
        if self.cancel.is_set():
            raise MotionError("cancelled", "Home finished; queued stop prevents subsequent motion")
        self._ensure_context()
        with self.lock:
            self.homed = True
        self._update(job, status="completed", completion_basis="native_home_result", finished_at=time.time())
        LOG.info("Home complete; Cartesian reference synchronized and workspace centered at home")

    def _run_job(self, job):
        with self.condition:
            self.active = job["id"]
            job.update(status="running", started_at=time.time())
        execute = {"delta": self._execute, "pose": self._execute,
                   "gripper": self._execute_gripper, "home": self._execute_home}
        action = None
        try:
            for action in job["actions"]:
                with self.condition:
                    job["active_action"] = action["index"]
                    if action["command"]["type"] == "home":
                        self.homed = False
                if self.cancel.is_set():
                    raise MotionError("cancelled")
                LOG.info("Job %s: action %d/%d (%s)", job["id"], action["index"],
                         len(job["actions"]), action["command"]["type"])
                execute[action["command"]["type"]](action)
                self._update(job, actions_completed=action["index"])
            self._update(job, status="completed", active_action=None, finished_at=time.time())
        except Exception as exc:
            planning_rejection = isinstance(exc, PlanRejected)
            if planning_rejection:
                # Only candidate-path failures take this route. A zero step
                # count alone does not prove that no POST reached the robot.
                # Recheck measured protection and the acknowledged held target
                # before leaving the server available for a new submission.
                try:
                    state = self._read()
                    self._hard(state, self.reference, self.joint_reference)
                    self.backend.check_held(state)
                except Exception as feedback_error:
                    exc = feedback_error
                    planning_rejection = False
            reason = exc.reason if isinstance(exc, MotionError) else "execution_error"
            LOG.warning("Job %s aborted: %s", job["id"], exc)
            with self.condition:
                self._discard_pending("previous_command_failed")
            confirmed = True if planning_rejection else self._stop()
            if not confirmed:
                with self.condition:
                    self._discard_home("stop_unconfirmed")
            failure = dict(status="cancelled" if reason == "cancelled" else "failed",
                           reason=reason, detail=str(exc), stop_confirmed=confirmed,
                           rejected_before_execution=planning_rejection,
                           recovery_required=not planning_rejection, finished_at=time.time())
            if action is not None:
                self._update(action, **failure)
            self._cancel_remaining(job, "sequence_cancelled" if reason == "cancelled" else "previous_action_failed")
            self._update(job, **failure, active_action=None,
                         failed_action=None if action is None else action["index"])
            if planning_rejection:
                LOG.info("Proposed trajectory discarded before execution; fresh feedback confirms "
                         "the held target. New commands remain available.")
            elif confirmed:
                LOG.info("Trajectory discarded; Franka FIFO cleared. The impedance controller holds its last target.")
                with self.condition:
                    self.fault = (None if reason == "cancelled" and self.home_pending is not None
                                  else f"Motion aborted ({reason}: {exc}); Franka controller "
                                       "holds its target. Recover explicitly.")
            else:
                LOG.error("Trajectory discarded; stop unconfirmed. "
                          "Status, stop and recover remain available; new motion waits for recovery.")
        finally:
            with self.condition:
                self.last_result = self._public(job)
                self.active = None
                self._prune_history()

    def _stop(self, force=False):
        with self.lock:
            self.stopping = True
        confirmed = True
        detail = None
        try:
            # A failed backend constructor must not require restarting HTTP.
            # Explicit stop/recover may recreate the client, without starting motion.
            if force and self.backend is None:
                self.backend = RobotBackend(self.cfg)
            if self.backend is not None and (force or self.backend.connected):
                state = self.backend.stop()
                self._record_feedback(state)
            else:
                confirmed = False
                detail = "No established controller connection; hold was not verified"
        except Exception as exc:
            confirmed, detail = False, str(exc)
        with self.lock:
            self.reference = self.joint_reference = None
            self.stopping = False
            self.fault = None if confirmed else f"Stop unconfirmed: {detail}"
            self.last_result = dict(action="stop", confirmed=confirmed, native_stop_confirmed=False,
                                    stop_mode="hold_target", detail=detail, at=time.time())
        return confirmed

    def _work_once(self):
        """Handle one outer-loop iteration; no transport is closed here."""
        job = None
        with self.condition:
            control = self.control_request
            self.control_request = None
            self.control_active = control
            if not control and not self.fault:
                if self.home_pending is not None:
                    job = self.jobs[self.home_pending]
                    self.home_pending = None
                    self.cancel.clear()  # Old work has stopped; start this explicit reset.
                elif self.pending and self.homed:
                    job = self.jobs[self.pending.popleft()]
                if job is not None:
                    # Reserve before releasing the queue lock: an idle-settings
                    # request must not slip between dequeue and _run_job.
                    self.active = job["id"]
        if control:
            try:
                confirmed = self._stop(force=True)
                if confirmed and control == "recover":
                    try:
                        self.cancel.clear()
                        self._ensure_context()
                        with self.condition:
                            self.homed = True
                            self.fault = None
                    except Exception as exc:
                        with self.condition:
                            self.fault = f"Recovery failed: {exc}"
                elif confirmed:
                    with self.condition:
                        self.fault = "Franka FIFO cleared; controller holds its target. Recover before new motion."
            finally:
                with self.condition:
                    self.control_active = None
                    if self.control_request is None:
                        self.cancel.clear()
            return
        if job is not None:
            self._run_job(job)
            return
        if self.backend is not None and self.backend.connected and not self.fault and not self.cancel.is_set():
            try:
                self._hard(self._read(), self.reference, self.joint_reference)
            except Exception as exc:
                if self.cancel.is_set():
                    return  # The queued stop/recovery request owns this transition.
                with self.condition:
                    self._discard_pending("idle_guard_failed")
                confirmed = self._stop()
                with self.condition:
                    if not confirmed:
                        self._discard_home("stop_unconfirmed")
                    self.last_result = dict(status="failed", reason="idle_guard_failed",
                                            detail=str(exc), stop_confirmed=confirmed)
                with self.condition:
                    self.fault = self.fault or "Idle guard failed; Franka holds its target. Recover explicitly."
                LOG.warning("Idle protection cleared the Franka FIFO: %s", exc)
        with self.condition:
            self.condition.wait(timeout=1/self.cfg.hz)

    def _worker_error(self, exc):
        """Contain an unexpected worker error without disabling communication."""
        LOG.exception("Worker operation failed; keeping the server available for recovery")
        with self.condition:
            self._discard_pending("worker_error")
            self._discard_home("worker_error")
        confirmed = self._stop()
        with self.condition:
            # Software/infrastructure errors require explicit recovery, even if
            # the motion stop succeeded. Never replay the interrupted operation.
            self.fault = self.fault or f"Worker error; recover required: {exc}"
            result = dict(status="failed", reason="worker_error", detail=str(exc),
                          stop_confirmed=confirmed, finished_at=time.time())
            if self.active is not None:
                job = self.jobs[self.active]
                self._cancel_remaining(job, "worker_error")
                job.update(result)
                job["active_action"] = None
                self.last_result = self._public(job)
                self.active = None
            else:
                self.last_result = result
            self._prune_history()

    def _run(self):
        try:
            try:
                self.backend = RobotBackend(self.cfg)
                # Startup does not move the arm. Homing is an explicit command.
                # Initialize against fresh measured state and its current held target.
                self._ensure_context()
                with self.condition:
                    self.homed = True
                    self.last_result = dict(status="ready", initialization="measured_pose", at=time.time())
            except Exception as exc:
                self._worker_error(exc)
            while not self.shutdown_event.is_set():
                try:
                    self._work_once()
                except Exception as exc:
                    self._worker_error(exc)
                    with self.condition:
                        self.condition.wait(timeout=1/self.cfg.hz)
        finally:
            # Only explicit service shutdown ends the worker/robot client.
            if self.backend is not None and self.backend.connected:
                self._stop()
            if self.backend is not None:
                self.backend.close()

    def close(self):
        self.request_stop()
        self.shutdown_event.set()
        if self.worker.ident is not None:
            self.worker.join(timeout=NATIVE_OPERATION_TIMEOUT_S + 10)
        if self.worker.is_alive():
            raise RuntimeError("Worker did not confirm shutdown")


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, fmt, *args):
            LOG.debug(fmt, *args)

        def respond(self, status, body):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            try:
                if self.path == "/status":
                    result = service.status()
                elif self.path.startswith("/commands/"):
                    result = service.status(self.path[len("/commands/"):])
                else:
                    return self.respond(404, {"error": "unknown route"})
                self.respond(200, result)
            except KeyError:
                self.respond(404, {"error": "unknown command id"})

        def do_POST(self):
            try:
                # Local API: reject browser-origin submissions and non-loopback Host
                # headers; commands come from a separate local HTTP client.
                if self.headers.get("Origin") is not None:
                    raise ValueError("Browser-origin control requests are disabled")
                host = self.headers.get("Host", "").split(":")[0]
                if host not in ("127.0.0.1", "localhost"):
                    raise ValueError("Only loopback Host headers are accepted")
                size = int(self.headers.get("Content-Length", "0"))
                if size < 0 or size > 16384 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("Invalid request size/encoding")
                data = self.rfile.read(size)
                if len(data) != size:
                    raise ValueError("Incomplete request")
                payload = json.loads(data or b"{}")
                if self.path == "/commands":
                    self.respond(202, service.enqueue_command(payload))
                elif self.path == "/settings/soft-force":
                    self.respond(200, service.set_soft_force(payload))
                elif self.path in ("/stop", "/recover"):
                    if payload != {}:
                        raise ValueError("Stop/recover expects an empty JSON object")
                    self.respond(202, service.request_stop(recover=self.path == "/recover"))
                else:
                    self.respond(404, {"error": "unknown route"})
            except (ValueError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
            except MotionError as exc:
                self.respond(409, {"error": exc.reason, "detail": str(exc)})
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--robot-address", default=Settings.address)
    parser.add_argument("--mode", choices=["cartesian_impedance"], default=Settings.mode)
    parser.add_argument("--hz", type=float, default=50)
    parser.add_argument("--joint-min", type=float, nargs=7, help="Additional lower bounds, radians")
    parser.add_argument("--joint-max", type=float, nargs=7, help="Additional upper bounds, radians")
    parser.add_argument("--soft-force-n", type=float, default=Settings.soft_force_n)
    parser.add_argument("--hard-force-n", type=float, default=Settings.hard_force_n)
    parser.add_argument("--force-sign", type=float, choices=(-1.0, 1.0), default=-1.0,
                        help="Sign converting Franka K wrench to environment-on-robot (default -1); verify on hardware")
    parser.add_argument("--robot-urdf", help="Optional local Panda URDF; embedded arm model is the default")
    parser.add_argument("--flange-to-ee", type=float, nargs=6, metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                        help="Fixed flange-to-EE xyz/RPY in m/rad; default inferred once from stationary feedback")
    parser.add_argument("--ee-to-stiffness", type=float, nargs=6, default=[0.]*6,
                        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                        help="EE-to-K xyz/RPY in m/rad; default identity (K=EE)")
    parser.add_argument("--world-from-base", type=float, nargs=9,
                        help="Row-major 3x3 world-from-base rotation; default identity (world=base)")
    parser.add_argument("--force-bias-n", type=float, nargs=3, default=[0, 0, 0],
                        help="Measured no-contact bias in base frame; default zero")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Settings(address=args.robot_address, mode=args.mode, hz=args.hz,
                   joint_min=args.joint_min, joint_max=args.joint_max,
                   robot_urdf=args.robot_urdf, flange_to_ee=args.flange_to_ee,
                   ee_to_stiffness=args.ee_to_stiffness,
                   soft_force_n=args.soft_force_n, hard_force_n=args.hard_force_n,
                   force_bias_n=np.array(args.force_bias_n), force_sign=args.force_sign,
                   world_from_base=(np.eye(3) if args.world_from_base is None else
                                    np.array(args.world_from_base).reshape(3, 3)))
    cfg.validate()
    # Bind before creating the worker: port conflicts cannot acquire robot control.
    http = ThreadingHTTPServer(("127.0.0.1", args.port), BaseHTTPRequestHandler)
    http.daemon_threads = True
    service = MotionService(cfg)
    http.RequestHandlerClass = handler_for(service)
    stop = threading.Event()

    def signal_stop(signum, frame):
        if service.backend is not None and service.backend.native_operation is not None:
            LOG.warning("Shutdown waits for the current native home/gripper operation to finish.")
        service.request_stop()
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal_stop)
    LOG.info("Listening on http://127.0.0.1:%s; robot endpoint %s",
             http.server_port, cfg.address)
    http.timeout = 0.1
    try:
        # Install HTTP handlers and interruption handling before initialization.
        if not stop.is_set():
            service.start()
        while not stop.is_set():
            http.handle_request()
    finally:
        try:
            service.close()
        finally:
            http.server_close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, MotionError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
