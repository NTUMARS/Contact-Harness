#!/usr/bin/env python3
"""Persistent HTTP server for guarded robot action sequences.

Start the robot server, then run python motion_server.py. Startup homes the arm
once, fixes the workspace around the measured TCP, and enables the control mode.

POST /commands: {"actions": [{"type": "delta", "translation_m": [0.1, 0, 0],
                            "duration_s": 4},
                           {"type": "delta", "translation_m": [0, 0.05, 0],
                            "duration_s": 2}]}
One action uses the same list format. Legacy single command objects are accepted.
All actions are validated before queueing. One job ID covers the entire sequence;
GET /commands/ID reports actions, active_action (1-based) and actions_completed.
Actions execute consecutively in the existing timed loop, without interleaving
other jobs. Each plans from the accepted reference left by the previous action.
Failure/stop cancels the remaining actions and queued jobs. Timing and geometric
guards apply per action; this does not blend trajectories across action boundaries.

The pending queue holds at most 16 actions by default. A home action must come
first; a sequence starting with home retains the priority reset behavior, stopping
active work and discarding queued jobs. Stop/recover remain separate HTTP routes.

Delta and pose default to world (same origin as base). Tool axes are fixed at
each action's start. Feedback, IK and transmitted targets use native base axes.
Guards use the explicit flange wrench transferred to the configured TCP in base
axes. Default force thresholds are 10 N soft and 50 N hard; there is no explicit
torque constraint.

POST /settings/soft-force: {"soft_force_n": 25} updates the soft threshold only
while ready and idle (no active/queued actions). GET /status reports the active,
startup and hard thresholds. Updates last until changed or the server restarts;
they neither command a contact force nor change the hard protection.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from numbers import Real
import signal
import sys
import threading
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation as R

from get_observation import transform_wrench

LOG = logging.getLogger("motion_server")

# Workspace half-width on each robot-base axis, centred on the initial TCP.
WORKSPACE_HALF_EXTENT_M = 0.30
HOME_RPC_TIMEOUT_S = 60.0  # Response deadline, not a motion duration or stop timer.

# Measured mounting orientation: native +X -> world +X, native +Y -> world -Z,
# native +Z -> world +Y. This is R_world_from_base = Rx(-90 degrees).
# Its transpose maps requested world vectors to native [x, -z, y]. Apply it
# equally to translation and axis-angle vectors; Euler angles are not vectors.
WORLD_FROM_BASE = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, -1.0, 0.0]])


class MotionError(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(detail or reason)


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
    address: str = "ipc:///tmp/robot_server_5555"
    mode: str = "cartesian_impedance"
    hz: float = 20.0
    joint_min: np.ndarray | None = None
    joint_max: np.ndarray | None = None
    joint_margin_rad: float = math.radians(5)
    soft_force_n: float = 10.0
    hard_force_n: float = 50.0
    force_bias_n: np.ndarray = field(default_factory=lambda: np.zeros(3))
    linear_speed: float = 0.035
    angular_speed: float = 0.08
    joint_speed: float = 0.35
    joint_acceleration: float = 4.0  # rad/s^2, estimated from consecutive IK targets
    max_translation: float = 0.15
    max_rotation: float = math.pi / 2
    state_timeout: float = 0.1
    rpc_timeout: float = 0.15
    ik_m: float = 0.003
    ik_rad: float = 0.03
    queue_size: int = 16
    max_jobs: int = 1000  # Finished results retained, in addition to active/queued work.

    def validate(self):
        for name in ("hz", "joint_margin_rad", "soft_force_n", "hard_force_n",
                     "linear_speed", "angular_speed", "joint_speed", "joint_acceleration",
                     "state_timeout", "rpc_timeout"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 10 <= self.hz <= 100:
            raise ValueError("frequency must be between 10 and 100 Hz")
        if self.hard_force_n > 50:
            raise ValueError("Hard force must not exceed 50 N")
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
             "gripper": {"position"}, "home": set()}
    if kind not in extra:
        raise ValueError("type must be delta, pose, gripper, or home")
    if set(payload) - common - extra[kind]:
        raise ValueError(f"Unknown fields: {sorted(set(payload) - common - extra[kind])}")
    cmd = dict(payload)
    if kind == "home":
        return cmd  # MoveJ plans its own timing; no software home trajectory.
    if kind == "gripper":
        position = number(cmd.get("position"), "position")
        if not 0 <= position <= 1:
            raise ValueError("gripper position must be between 0 (closed) and 1 (open)")
        cmd["position"] = position
        return cmd
    duration = number(cmd.get("duration_s"), "duration_s")
    if not 1 / cfg.hz <= duration <= 120:
        raise ValueError(f"duration_s must be between {1/cfg.hz:g} and 120")
    cmd["duration_s"] = duration
    if kind == "delta":
        dp = vector(cmd.get("translation_m", [0, 0, 0]), 3, "translation_m")
        dr = vector(cmd.get("rotation_vector_rad", [0, 0, 0]), 3, "rotation_vector_rad")
        frame = cmd.setdefault("frame", "world")
        if frame not in ("world", "base", "tool"):
            raise ValueError("frame must be world, base or tool (fixed at command start)")
        cmd.update(translation_m=dp.tolist(), rotation_vector_rad=dr.tolist())
        check_distance(dp, dr, duration, cfg)
    elif kind == "pose":
        if cmd.setdefault("frame", "world") not in ("world", "base"):
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
    if distance / duration > cfg.linear_speed + 1e-12 or angle / duration > cfg.angular_speed + 1e-12:
        raise ValueError("Requested duration exceeds speed limits; increase duration_s")


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
            dp, dr = WORLD_FROM_BASE.T @ dp, WORLD_FROM_BASE.T @ dr
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
            position = WORLD_FROM_BASE.T @ position
            rotation = R.from_matrix(WORLD_FROM_BASE.T) * rotation
        goal = np.r_[position, rotation.as_euler("xyz")]
        dp = goal[:3] - start[:3]
        dr = (R.from_euler("xyz", goal[3:]) * R.from_euler("xyz", start[3:]).inv()).as_rotvec()
    else:
        raise ValueError("Only delta and pose commands use the trajectory planner")
    check_distance(dp, dr, cmd["duration_s"], cfg)
    count = math.ceil(cmd["duration_s"] * cfg.hz)
    return Plan(start, goal, dp / count, dr / count, count, cmd["duration_s"] / count)


class Kinematics:
    """Forward kinematics and differential IK of the arm, from the URDF registered for the robot type."""
    def __init__(self, robot_type, tool_pose):
        import pinocchio as pin
        from robot_support.ik import DifferentialIk
        self.pin = pin
        self.solver = DifferentialIk(robot_type)
        self.model = self.solver.model
        self.data = self.model.createData()
        self.frame = self.model.getFrameId(self.solver.frame_name)
        self.tool = pose_T(tool_pose)
        self.tool_inverse = np.linalg.inv(self.tool)
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()

    def reset(self, joints):
        self.solver.reset(joints)

    def fk(self, joints):
        self.pin.framesForwardKinematics(self.model, self.data, joints)
        return T_pose(self.data.oMf[self.frame].homogeneous @ self.tool)

    def solve(self, pose, joints, dt):
        target = pose_T(pose) @ self.tool_inverse
        result, ok = self.solver.solve(target, joints, dt)
        if not ok:
            raise MotionError("ik_failed")
        return np.asarray(result, dtype=float)


class RobotBackend:
    def __init__(self, cfg):
        from robot_support.client import RobotClient
        self.cfg = cfg
        self.client = RobotClient(cfg.address, timeout=cfg.rpc_timeout)
        self.owner = str(uuid.uuid4())
        self.owned = False
        self.active_mode = None
        self.server_id = None
        self.target_seq = 0
        self.kin = None
        self.rejections = None

    def _rpc(self, method, params=None, timeout=None):
        begin = time.monotonic()
        result = self.client.request(method, params, timeout=timeout)
        if isinstance(result, dict):
            # Include RPC transit time; do not mistake response receipt for sample time.
            result["_received"] = begin
        return result

    def _identity(self, state, *, wrench=True):
        if state.get("protocol") != 1:
            raise MotionError("wrapper_incompatible", "Restart the robot server with this project's updated wrapper")
        if self.server_id is not None and state.get("server_id") != self.server_id:
            raise MotionError("robot_server_restarted")
        if self.owned:
            if (state.get("owner") != self.owner or state.get("mode") != self.active_mode or
                    state.get("target_seq") != self.target_seq or state.get("fault")):
                raise MotionError("control_session_changed")
            counts = (state["stale_target_rejections"], state["jump_target_rejections"])
            if self.rejections is not None and counts != self.rejections:
                raise MotionError("native_target_dropped")
        if wrench and state.get("in_realtime_loop"):
            try:
                sample = state.get("wrench")
                state["ext_wrench"] = transform_wrench(sample, reference="tcp", frame="base")
                age = number(sample["age_s"], "wrench age")
                if age < 0:
                    raise ValueError("Negative wrench age")
                if state.get("state_age_s") is not None:
                    state["state_age_s"] = max(state["state_age_s"], age)
            except (ValueError, TypeError, KeyError) as exc:
                raise MotionError("invalid_wrench_feedback", str(exc)) from exc
        return state

    def state(self):
        return self._identity(self._rpc("motion_state", {"owner": self.owner, "include_wrench": True}))

    def start(self):
        mode = self.cfg.mode
        before = self.state()
        if before["in_realtime_loop"] or before["owner"] or not before["stop_confirmed"] or before["fault"]:
            raise MotionError("control_already_active", "Use stop/recover explicitly before taking control")
        info = self.client.get_robot_info()
        if info["joint_num"] != 7:
            raise MotionError("unsupported_robot", "This server requires the copied seven-axis URDF/Pink model")
        tool = self.client.get_tool_info()
        if not np.allclose(vector(tool["ref_pos"], 6, "tool reference"), 0, atol=1e-9):
            raise MotionError("unsupported_tool_reference", "The base-frame path requires zero tool ref_pos")
        self.kin = Kinematics(info["type"], vector(tool["end_pos"], 6, "tool end pose"))
        # Mark ownership before the RPC: a timeout can still mean native start executed.
        self.owned = True
        self.active_mode = mode
        self.server_id = before["server_id"]
        self.target_seq = 0
        self.rejections = None
        result = self._rpc("motion_start", {"owner": self.owner, "mode": mode}, 5.0)
        self._identity(result, wrench=False)  # First fresh read follows native startup.
        self.rejections = (result["stale_target_rejections"], result["jump_target_rejections"])
        return result

    def send(self, pose, joints, dt):
        kind = "cart_pos" if self.active_mode == "cartesian_impedance" else "joint_pos"
        value = pose if kind == "cart_pos" else joints
        return self._send_target(kind, value.tolist(), interpolate_time=dt)

    def send_gripper(self, position):
        return self._send_target("gripper", [position])

    def _send_target(self, kind, value, **params):
        seq = self.target_seq + 1
        result = self._rpc("motion_target", dict(owner=self.owner, server_id=self.server_id,
                           target_seq=seq, action_type=kind, action_value=value, **params))
        if not result.get("accepted") or result.get("target_seq") != seq:
            raise MotionError("target_not_acknowledged")
        self.target_seq = seq
        return self._identity(result)

    def home(self):
        before = self.state()
        if before["in_realtime_loop"] or before["owner"] or not before["stop_confirmed"] or before["fault"]:
            raise MotionError("control_already_active", "Stop/recover before homing")
        # Mark this operation before sending: a timed-out RPC may still be moving.
        self.owned = True
        self.active_mode = "sdk_movej"
        self.server_id = before["server_id"]
        self.target_seq = 0
        self.rejections = None
        result = self._identity(self._rpc("motion_home", {"owner": self.owner}, HOME_RPC_TIMEOUT_S))
        self.stop()  # Release ownership only after the native reset returned.
        return result

    def stop(self):
        result = self._rpc("motion_stop", timeout=2.0)
        if result["in_realtime_loop"] or not result["stop_confirmed"] or result["fault"]:
            raise MotionError("stop_unconfirmed")
        self.owned = False
        self.active_mode = None
        self.server_id = None
        self.rejections = None
        return result

    def close(self):
        self.client.close()


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
        self.homed = False
        self.home_pending = None
        startup = self._new_job(validate_actions({"actions": [{"type": "home"}]}, cfg))
        self.startup_home_id = startup["id"]
        self.jobs[startup["id"]] = startup
        self.active = startup["id"]
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
            if (not self.worker.is_alive() or self.fault or self.control_request or
                    self.shutdown_event.is_set()):
                raise MotionError("not_ready", self.fault or "Stop/recovery in progress")
            if commands[0]["type"] == "home":
                # A reset is a priority request, not a trajectory queued behind work.
                self._discard_pending("home_requested")
                self._discard_home("superseded_by_home")
                self.jobs[job["id"]] = job
                self.home_pending = job["id"]
                self.homed = False
                self.cancel.set()
            else:
                if not self.homed or self.home_pending or self.cancel.is_set() or self.stopping:
                    raise MotionError("not_ready", "Homing must complete before normal commands")
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
                    self.home_pending is not None or self.control_request is not None or
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
            return dict(mode=self.cfg.mode,
                        soft_force_n=self.cfg.soft_force_n,
                        startup_soft_force_n=self.startup_soft_force_n,
                        hard_force_n=self.cfg.hard_force_n,
                        active_mode=None if self.backend is None else self.backend.active_mode,
                        homed=self.homed,
                        ready=self.worker.is_alive() and self.homed and not (self.home_pending or self.fault or
                              self.cancel.is_set() or self.stopping or self.shutdown_event.is_set()),
                        pending_home=self.home_pending, startup_home_id=self.startup_home_id,
                        active=self.active, queued=list(self.pending), fault=self.fault,
                        stop_requested=self.cancel.is_set(), stopping=self.stopping,
                        worker_alive=self.worker.is_alive(),
                        last_result=self.last_result,
                        feedback=self.latest,
                        reference_frame="base",
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
        # Capture once per server process. Stop/recover must not move the box.
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

    def _hard(self, state, candidate=None, candidate_q=None):
        pose, q, wrench = self._check_state(state)
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
        if not self.backend.owned:
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
            with self.lock:
                self.reference, self.joint_reference = pose.copy(), q.copy()
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
        pose, q = plan.start.copy(), self.joint_reference.copy()
        self.backend.kin.reset(q)
        velocity = np.zeros(7)
        last_check = 0.0
        # Check every proposed point, including the final one; keep force/stop
        # monitoring and the native watchdog alive during expensive planning.
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
                raise MotionError(exc.reason, f"Preflight step {index+1}/{plan.count}: {exc}") from exc
        self.backend.kin.reset(self.joint_reference)

    def _wait_until(self, deadline):
        if self.cancel.wait(max(0, deadline-time.monotonic())):
            raise MotionError("cancelled")

    def _dispatch(self, dp, dr, dt, previous_velocity, hold=False):
        state = self._read()
        _, _, wrench = self._check_state(state)
        # Soft constraint precedes final hard checks. Removed increments are
        # discarded; the next step starts at the accepted constrained reference.
        safe_dp, blocked = constrain_translation(dp, wrench[:3]-self.cfg.force_bias_n,
                                                   self.cfg.soft_force_n)
        candidate = advance(self.reference, safe_dp, dr)
        self._hard(state)
        if not hold:
            q, velocity = self._solve(candidate, self.joint_reference, dt, previous_velocity)
        else:
            q, velocity = self.joint_reference.copy(), np.zeros(7)
        self._hard(state, candidate, q)  # Includes age again after IK.
        if self.cancel.is_set():
            raise MotionError("cancelled")
        result = self.backend.send(candidate, q, dt)
        # Commit reference only after accepted ACK; uncertainty aborts, never retries.
        with self.lock:
            self.reference, self.joint_reference = candidate.copy(), q.copy()
        self._last_ack = result
        return dp-safe_dp, bool(np.any(blocked)), velocity

    def _execute(self, job):
        self._update(job, status="planning", started_at=time.time(), endpoint_frame="base")
        self._ensure_context()
        cmd = job["command"]
        plan = plan_command(cmd, self.reference, self.cfg)
        self._preflight(plan)
        self._update(job, status="running", steps_total=plan.count, step_duration_s=plan.dt,
                     requested_endpoint=plan.goal.tolist())
        start = time.monotonic()
        velocity = np.zeros(7)
        discarded = np.zeros(3)
        clipped = 0
        # First target starts immediately; it covers [0, dt]. Never batch overdue
        # targets. Stretch small scheduling delays, abort if more than one tick late.
        due = start
        for index in range(plan.count):
            self._wait_until(due)
            if time.monotonic() - due > plan.dt:
                raise MotionError("execution_overrun")
            dispatched_at = time.monotonic()
            try:
                removed, blocked, velocity = self._dispatch(plan.dp, plan.dr, plan.dt, velocity)
            except MotionError as exc:
                raise MotionError(exc.reason, f"Execution step {index+1}/{plan.count}: {exc}") from exc
            discarded += removed
            clipped += int(blocked)
            self._update(job, steps_sent=index+1, clipped_steps=clipped,
                         discarded_translation_m=discarded.tolist())
            self._hard(self._last_ack, self.reference, self.joint_reference)
            due = dispatched_at + plan.dt
        self._wait_until(due)
        # Legacy joint callback consumes only when >=2 targets are queued. A
        # checked duplicate releases the last real point of this finite sequence.
        if self.cfg.mode != "cartesian_impedance":
            self._dispatch(np.zeros(3), np.zeros(3), plan.dt, None, hold=True)
            self._hard(self._last_ack, self.reference, self.joint_reference)
            self._wait_until(time.monotonic() + plan.dt)
        # Completion is exhaustion of the timed steps, including clipped ones.
        # Check protection once more after the final interval, without waiting
        # for measured pose or rotation arrival.
        measured, _, _ = self._hard(self._read(), self.reference, self.joint_reference)
        self._update(job, status="completed", completion_basis="timed_steps",
                     execution_elapsed_s=time.monotonic()-start, finished_at=time.time(),
                     accepted_endpoint=self.reference.tolist(), measured_endpoint=measured.tolist())
        LOG.info("Action %d completed: %d timed steps; %d steps constrained by force",
                 job["index"], plan.count, clipped)

    def _execute_gripper(self, job):
        self._update(job, status="running", started_at=time.time(), steps_total=1)
        self._ensure_context()
        self._hard(self._read(), self.reference, self.joint_reference)
        if self.cancel.is_set():
            raise MotionError("cancelled")
        result = self.backend.send_gripper(job["command"]["position"])
        self._update(job, steps_sent=1)
        self._hard(result, self.reference, self.joint_reference)
        self._update(job, status="completed", completion_basis="acknowledged", finished_at=time.time())
        LOG.info("Action %d completed: gripper command acknowledged", job["index"])

    def _execute_home(self, job):
        self._update(job, status="running", started_at=time.time(),
                     control_mode="joint_position", method="SDK MoveJ")
        if self.cancel.is_set():
            raise MotionError("cancelled")
        if self.backend.owned:
            self.backend.stop()
        with self.lock:
            self.reference = self.joint_reference = None
        if self.cancel.is_set():
            raise MotionError("cancelled")
        LOG.info("Homing via the robot server's reset_position / SDK MoveJ. "
                 "This blocking reset cannot be interrupted through the RPC channel.")
        result = self.backend.home()
        self._update(job, home_joint_target_rad=result["home_joint_pos_rad"],
                     measured_joints_rad=result["home_measured_joints_rad"],
                     home_joint_error_rad=result["home_joint_error_rad"])
        if self.cancel.is_set():
            raise MotionError("cancelled")
        # MoveJ has finished; only now start the ordinary impedance/position loop.
        self._ensure_context()
        if self.cancel.is_set():
            raise MotionError("cancelled")
        with self.lock:
            self.homed = True
        self._update(job, status="completed", finished_at=time.time())
        LOG.info("Home complete; ready for commands in %s", self.cfg.mode)

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
            reason = exc.reason if isinstance(exc, MotionError) else "execution_error"
            LOG.warning("Job %s aborted: %s", job["id"], exc)
            with self.condition:
                self._discard_pending("previous_command_failed")
            confirmed = self._stop()
            if not confirmed:
                with self.condition:
                    self._discard_home("stop_unconfirmed")
            failure = dict(status="cancelled" if reason == "cancelled" else "failed",
                           reason=reason, detail=str(exc), stop_confirmed=confirmed, finished_at=time.time())
            if action is not None:
                self._update(action, **failure)
            self._cancel_remaining(job, "sequence_cancelled" if reason == "cancelled" else "previous_action_failed")
            self._update(job, **failure, active_action=None,
                         failed_action=None if action is None else action["index"])
            if confirmed:
                LOG.info("Trajectory discarded; native motion stopped. "
                         "Communication remains open; waiting for the next command.")
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
            if self.backend is not None and (force or self.backend.owned):
                state = self.backend.stop()
                self._record_feedback(state)
        except Exception as exc:
            confirmed, detail = False, str(exc)
        with self.lock:
            self.reference = self.joint_reference = None
            self.stopping = False
            self.fault = None if confirmed else f"Stop unconfirmed: {detail}"
            self.last_result = dict(action="stop", confirmed=confirmed, detail=detail, at=time.time())
        return confirmed

    def _work_once(self):
        """Handle one outer-loop iteration; no transport is closed here."""
        job = None
        with self.condition:
            control = self.control_request
            self.control_request = None
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
            self._stop(force=True)
            with self.condition:
                if self.control_request is None:
                    self.cancel.clear()
            return
        if job is not None:
            self._run_job(job)
            return
        if self.backend is not None and self.backend.owned and not self.fault:
            try:
                self._hard(self._read(), self.reference, self.joint_reference)
            except Exception as exc:
                with self.condition:
                    self._discard_pending("idle_guard_failed")
                confirmed = self._stop()
                with self.condition:
                    if not confirmed:
                        self._discard_home("stop_unconfirmed")
                    self.last_result = dict(status="failed", reason="idle_guard_failed",
                                            detail=str(exc), stop_confirmed=confirmed)
                LOG.warning("Idle protection stopped motion: %s. Communication remains open.", exc)
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
                # Home once before entering the normal command-processing loop.
                self._run_job(self.jobs[self.startup_home_id])
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
            if self.backend is not None and self.backend.owned:
                self._stop()
            if self.backend is not None:
                self.backend.close()

    def close(self):
        self.request_stop()
        self.shutdown_event.set()
        if self.worker.ident is not None:
            self.worker.join(timeout=HOME_RPC_TIMEOUT_S + 10)
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
                # Local API: reject browser-origin submissions and remote Host
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
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--robot-address", default=Settings.address)
    parser.add_argument("--mode", choices=["cartesian_impedance", "joint_impedance", "joint_position"],
                        default=Settings.mode)
    parser.add_argument("--hz", type=float, default=20)
    parser.add_argument("--joint-min", type=float, nargs=7, help="Additional lower bounds, radians")
    parser.add_argument("--joint-max", type=float, nargs=7, help="Additional upper bounds, radians")
    parser.add_argument("--soft-force-n", type=float, default=Settings.soft_force_n)
    parser.add_argument("--hard-force-n", type=float, default=50)
    parser.add_argument("--force-bias-n", type=float, nargs=3, default=[0, 0, 0],
                        help="Measured no-contact bias in base frame; default zero")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Settings(address=args.robot_address, mode=args.mode, hz=args.hz,
                   joint_min=args.joint_min, joint_max=args.joint_max,
                   soft_force_n=args.soft_force_n, hard_force_n=args.hard_force_n,
                   force_bias_n=np.array(args.force_bias_n))
    cfg.validate()
    # Bind before creating the worker: port conflicts cannot acquire robot control.
    http = ThreadingHTTPServer(("127.0.0.1", args.port), BaseHTTPRequestHandler)
    http.daemon_threads = True
    service = MotionService(cfg)
    http.RequestHandlerClass = handler_for(service)
    stop = threading.Event()

    def signal_stop(signum, frame):
        if service.backend is not None and service.backend.active_mode == "sdk_movej":
            LOG.warning("Shutdown requested during blocking MoveJ; waiting for its return. "
                        "Use the robot's physical stop to interrupt this home.")
        service.request_stop()
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal_stop)
    LOG.info("Listening on http://127.0.0.1:%s; robot endpoint %s",
             http.server_port, cfg.address)
    http.timeout = 0.1
    try:
        # Install HTTP handlers and interruption handling before automatic homing.
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
