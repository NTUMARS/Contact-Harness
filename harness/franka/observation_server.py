#!/usr/bin/env python3
"""Continuous Franka observations, without any robot-controller changes.

  python observation_server.py --list-devices
  python observation_server.py
  python get_observation.py --observations state wrist.rgb third_view.rgb

Configure cameras and transforms in config/observation.yaml. The HTTP service
binds localhost:8082; robot GET requests use localhost:5003 and motion status
uses localhost:8081. No robot command, home, stop or gripper POST is issued.
Cameras acquire continuously; only requested snapshots are saved by the client.
third_view is the right-side external camera; side remains a disabled placeholder.
No tactile hardware/SDK is used.

Keep the robot still and free of contact during the one-second startup wrench
tare. This is observation-only zeroing, not payload calibration. Re-tare with
get_observation.py --calibrate. Motion guards keep their unzeroed feedback.
Both force and torque come from K_F_ext_hat_K. They are sign-adjusted to
environment-on-robot, rotated and shifted to the requested reference point.
The robot server API omits EE_T_K: configure ee_to_stiffness if K differs from EE.
Measured gripper width is unavailable; gripper_status reports commands only.

Freshness uses advancing robot_time_s plus HTTP timing. The fixed controller
rejects feedback older than 0.5 seconds but does not expose its true source age.
Camera and robot sources are asynchronous, with no hardware synchronization.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, suppress
from copy import deepcopy
import fcntl
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit
import zipfile

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from motion_server import Kinematics, PANDA_ARM_URDF, pose_T, T_pose

LOG = logging.getLogger("observation_server")
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "observation.yaml"
SCALAR_KEYS = ("state", "motion_status", "gripper_status")
STATE_FIELDS = ("force", "eef_pose", "joint_angles", "joint_velocities", "torque", "wrench")
CAMERA_KINDS = ("rgb", "depth")
MAX_BUNDLE_BYTES = 128 * 1024 * 1024
WRENCH_ZERO_SECONDS = 1.0



def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def positive(value, label="value"):
    if not valid_number(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def positive_number(value):
    try:
        return positive(float(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc


def vector(value, size, label):
    if (not isinstance(value, (list, tuple, np.ndarray)) or len(value) != size or
            any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)) or
                not math.isfinite(v) for v in value)):
        raise ValueError(f"{label} must contain {size} finite numbers")
    return np.asarray(value, dtype=float)


def sensor_configs(config):
    return {name: ("cameras", value) for name, value in config["cameras"].items()
            if value.get("enabled", True)}


def expand_observations(requested, config):
    available = [*SCALAR_KEYS, *STATE_FIELDS, "gripper"]
    available += [f"{name}.{kind}" for name in config["cameras"] for kind in CAMERA_KINDS]
    if not isinstance(requested, list) or not requested or not all(isinstance(k, str) for k in requested):
        raise ValueError("observations must be a nonempty list of names")
    if "all" in requested:
        if requested != ["all"]:
            raise ValueError("use all by itself")
        return list(SCALAR_KEYS) + [f"{name}.{kind}" for name in sensor_configs(config) for kind in CAMERA_KINDS]
    unknown = set(requested)-set(available)
    if unknown:
        raise ValueError(f"Unknown observations: {sorted(unknown)}. Available: {available}")
    return list(dict.fromkeys(requested))


def load_config(path=CONFIG_PATH):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("cameras"), dict):
        raise ValueError("Configuration needs a cameras mapping")
    if config.get("force_sign", -1) not in (-1, 1) or isinstance(config.get("force_sign"), bool):
        raise ValueError("force_sign must be -1 or +1")
    config.setdefault("force_sign", -1.)
    config.setdefault("tare_on_start", True)
    if type(config["tare_on_start"]) is not bool:
        raise ValueError("tare_on_start must be true or false")
    config["ee_to_stiffness"] = vector(config.get("ee_to_stiffness", [0.]*6), 6, "ee_to_stiffness").tolist()
    if config.get("flange_to_ee") is not None:
        config["flange_to_ee"] = vector(config["flange_to_ee"], 6, "flange_to_ee").tolist()
    rotation = np.asarray(config.get("world_from_base"), dtype=float)
    if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)) or
            not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(rotation), 1.)):
        raise ValueError("world_from_base must be a proper 3x3 rotation")
    offsets = config.setdefault("wrench_reference_points_m", {"ee": [0., 0., 0.], "fingertip": None, "connector": None})
    if not isinstance(offsets, dict) or set(offsets) != {"ee", "fingertip", "connector"}:
        raise ValueError("wrench_reference_points_m needs ee, fingertip and connector")
    for name, offset in offsets.items():
        if offset is not None:
            offsets[name] = vector(offset, 3, name+" offset").tolist()
    if offsets["ee"] != [0., 0., 0.]:
        raise ValueError("The ee reference must be [0, 0, 0]")
    serials = set()
    for name, value in config["cameras"].items():
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or
                name in (*SCALAR_KEYS, *STATE_FIELDS, "gripper") or not isinstance(value, dict)):
            raise ValueError("Camera names must be unique identifiers distinct from scalar keys")
        if type(value.get("enabled", True)) is not bool:
            raise ValueError("Camera enabled must be true or false")
        if not value.get("enabled", True):
            continue
        serial = value.get("serial")
        if not isinstance(serial, str) or not serial or serial in serials:
            raise ValueError("Enabled cameras require unique, explicit serial strings")
        serials.add(serial)
        if value.get("type") not in ("realsense", "orbbec"):
            raise ValueError("Camera type must be realsense or orbbec")
        for key in ("width", "height", "fps"):
            if type(value.get(key)) is not int or value[key] <= 0:
                raise ValueError(f"Camera {key} must be a positive integer")
        if "color_gain" in value and "color_exposure" not in value:
            raise ValueError("color_gain requires manual color_exposure")
        for key in ("color_exposure", "color_gain"):
            if key in value:
                if value["type"] != "realsense" or not valid_number(value[key]) or value[key] < 0:
                    raise ValueError(f"{key} requires a nonnegative RealSense option value")
                if key == "color_exposure" and value[key] == 0:
                    raise ValueError("color_exposure must be positive")
    expand_observations(config["default_observations"], config)
    ReadOnlyRobotHTTP(config["robot_address"], .5)
    MotionStatusReader(config["motion_server"], .5)
    return config




@contextmanager
def sdk_stdout_to_stderr():
    """Redirect native SDK printf as well as Python print, preserving JSON stdout."""
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


class RealSenseSource:
    def __init__(self, config, timeout):
        import pyrealsense2 as rs
        self.rs, self.timeout = rs, int(timeout * 1000)
        self.pipeline = rs.pipeline()
        settings = rs.config()
        settings.enable_device(config["serial"])
        for stream, fmt in ((rs.stream.color, rs.format.rgb8), (rs.stream.depth, rs.format.z16)):
            settings.enable_stream(stream, config["width"], config["height"], fmt, config["fps"])
        self.profile = self.pipeline.start(settings)
        self.last_frame = -1
        self.align = rs.align(rs.stream.color)
        self.scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        try:
            self.configure_exposure(config)
            for index in range(config.get("warmup_frames", 10)):
                # D455 devices can take more than two seconds for their first
                # frame, especially when several cameras start together.
                self.pipeline.wait_for_frames(max(self.timeout, 5000) if index == 0 else self.timeout)
        except BaseException:
            self.pipeline.stop()
            raise

    def configure_exposure(self, config):
        if "color_exposure" not in config:
            return  # Preserve device auto-exposure when no manual setting is supplied.
        rs = self.rs
        # D405 color belongs to its Stereo Module; D435i has a separate RGB sensor.
        sensors = [sensor for sensor in self.profile.get_device().query_sensors()
                   if any(p.stream_type() == rs.stream.color for p in sensor.get_stream_profiles())]
        if len(sensors) != 1:
            raise ValueError("Expected exactly one RealSense sensor providing color")
        sensor = sensors[0]
        options = [(rs.option.exposure, config["color_exposure"])]
        if "color_gain" in config:
            options.append((rs.option.gain, config["color_gain"]))
        for option, value in [(rs.option.enable_auto_exposure, 0), *options]:
            if not sensor.supports(option):
                raise ValueError(f"Color sensor does not support {option}")
            bounds = sensor.get_option_range(option)
            if not bounds.min <= value <= bounds.max:
                raise ValueError(f"{option} must be in [{bounds.min}, {bounds.max}]")
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        for option, value in options:
            sensor.set_option(option, value)
        LOG.info("Camera %s manual exposure=%s gain=%s (SDK units)",
                 config["serial"], sensor.get_option(rs.option.exposure), sensor.get_option(rs.option.gain))

    def capture(self, kinds):
        import numpy as np
        # Drop queued frames before the next sample; bound the drain operation.
        for _ in range(32):
            if not self.pipeline.poll_for_frames():
                break
        deadline = time.monotonic() + self.timeout / 1000
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("No new RealSense color frame before timeout")
            frames = self.align.process(self.pipeline.wait_for_frames(max(1, int(remaining*1000))))
            received = time.time()
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            number = color.get_frame_number()
            if number < self.last_frame:
                raise RuntimeError(f"RealSense frame counter reset: {number} < {self.last_frame}")
            # A depth update can repeat the previous RGB frame in a frameset.
            # Wait for advancing color; do not publish it with a new timestamp.
            if number > self.last_frame:
                break
        self.last_frame = color.get_frame_number()
        meta = {"received_at_unix_s": received, "frame_number": self.last_frame}
        if (color.get_frame_timestamp_domain() == self.rs.timestamp_domain.global_time and
                depth.get_frame_timestamp_domain() == self.rs.timestamp_domain.global_time):
            age = received - min(color.get_timestamp(), depth.get_timestamp()) / 1000.0
            if age < -0.1:
                raise RuntimeError("RealSense global clock is ahead of host; freshness is unknown")
            meta["sample_age_s_at_receive"] = max(0.0, age)
        arrays = {}
        if "rgb" in kinds:
            arrays["rgb"] = np.asanyarray(color.get_data()).copy()
        if "depth" in kinds:
            arrays["depth"] = np.asanyarray(depth.get_data()).astype(np.float32) * self.scale
        return meta, arrays

    def close(self):
        self.pipeline.stop()


class OrbbecSource:
    def __init__(self, config, timeout):
        import pyorbbecsdk as ob
        self.ob, self.timeout = ob, int(timeout * 1000)
        self.context = ob.Context()
        device = self.context.query_devices().get_device_by_serial_number(config["serial"])
        self.pipeline = ob.Pipeline(device)
        settings = ob.Config()
        for sensor, formats in ((ob.OBSensorType.COLOR_SENSOR, (ob.OBFormat.MJPG, ob.OBFormat.RGB)),
                                (ob.OBSensorType.DEPTH_SENSOR, (ob.OBFormat.Y16,))):
            profiles = self.pipeline.get_stream_profile_list(sensor)
            profile = None
            for fmt in formats:
                try:
                    profile = profiles.get_video_stream_profile(config["width"], config["height"], fmt, config["fps"])
                    break
                except Exception:
                    continue
            if profile is None:
                raise RuntimeError(f"No matching Orbbec profile for {sensor}")
            settings.enable_stream(profile)
        settings.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        self.pipeline.start(settings)
        self.align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
        self.last_frame = -1
        try:
            for _ in range(config.get("warmup_frames", 10)):
                if self.pipeline.wait_for_frames(self.timeout) is None:
                    raise TimeoutError("Orbbec warmup timed out")
        except BaseException:
            self.pipeline.stop()
            raise

    def capture(self, kinds):
        import cv2
        import numpy as np
        # Read beyond the current queue so a new request cannot reuse a cached frame.
        for _ in range(32):
            if self.pipeline.wait_for_frames(1) is None:
                break
        frames = self.pipeline.wait_for_frames(self.timeout)
        if frames is None:
            raise TimeoutError("Orbbec frame timeout")
        frames = self.align.process(frames).as_frame_set()
        received = time.time()
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("Incomplete Orbbec RGB-D frameset")
        frame_number = color.get_index()
        if frame_number <= self.last_frame:
            raise RuntimeError("Orbbec color frame did not advance")
        self.last_frame = frame_number
        scale = depth.get_depth_scale() / 1000.0
        meta = {"received_at_unix_s": received, "frame_number": frame_number}
        arrays = {}
        if "rgb" in kinds:
            raw = np.asanyarray(color.get_data(), dtype=np.uint8)
            if color.get_format() == self.ob.OBFormat.MJPG:
                bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError("Orbbec JPEG decode failed")
                arrays["rgb"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                arrays["rgb"] = raw.reshape(color.get_height(), color.get_width(), 3).copy()
        if "depth" in kinds:
            arrays["depth"] = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(
                depth.get_height(), depth.get_width()).astype(np.float32) * scale
        return meta, arrays

    def close(self):
        self.pipeline.stop()


def sensor_worker(connection, category, config, timeout):
    """Isolate native drivers and their startup/hang/failure from the caller."""
    source = None
    device_lock = None
    # The terminal sends Ctrl+C to the whole process group. Let the server
    # coordinate shutdown so workers can release SDKs without tracebacks.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Permanent redirection includes C/C++ logs and destructor output.
    os.dup2(2, 1)
    try:
        # Keep exclusive ownership for the camera worker's lifetime.
        lock_name = hashlib.sha256(config["serial"].encode()).hexdigest()[:24]
        device_lock = open(f"/tmp/agentic_observation_{lock_name}.lock", "a")
        try:
            fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Sensor {config['serial']} is owned by another observation server") from exc
        cls = {"realsense": RealSenseSource, "orbbec": OrbbecSource}[config["type"]]
        source = cls(config, timeout)
        connection.send({"ok": True})
        while True:
            kinds = connection.recv()
            if kinds is None:
                break
            meta, arrays = source.capture(kinds)
            meta["received_monotonic_s"] = time.monotonic() - max(0.0, time.time() - meta["received_at_unix_s"])
            connection.send({"ok": True, "meta": meta, "arrays": arrays})
    except EOFError:
        pass
    except Exception as exc:
        with suppress(Exception):
            connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if source is not None:
            with suppress(Exception):
                source.close()
        if device_lock is not None:
            device_lock.close()
        connection.close()


class SensorWorker:
    def __init__(self, category, config, timeout, startup_timeout):
        ctx = mp.get_context("spawn")
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=sensor_worker, args=(child, category, config, timeout), daemon=True)
        self.process.start()
        child.close()
        self.timeout = timeout + 1.0
        try:
            self.receive(startup_timeout)
        except Exception:
            self.close()
            raise

    def receive(self, timeout):
        if not self.connection.poll(timeout):
            raise TimeoutError(f"Sensor worker did not reply within {timeout:g}s")
        result = self.connection.recv()
        if not result["ok"]:
            raise RuntimeError(result["error"])
        return result

    def capture(self, kinds):
        self.connection.send(kinds)
        return self.receive(self.timeout)

    def close(self):
        with suppress(Exception):
            self.connection.send(None)
        self.process.join(1.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(1.0)
        self.connection.close()


def list_devices():
    result = {"realsense": [], "orbbec": [], "errors": {}}
    with sdk_stdout_to_stderr():
        try:
            import pyrealsense2 as rs
            result["realsense"] = [{"serial": d.get_info(rs.camera_info.serial_number),
                                    "name": d.get_info(rs.camera_info.name)} for d in rs.context().query_devices()]
        except Exception as exc:
            result["errors"]["realsense"] = str(exc)
        try:
            import pyorbbecsdk as ob
            ctx = ob.Context()
            devices = ctx.query_devices()
            result["orbbec"] = [{"serial": devices.get_device_serial_number_by_index(i)} for i in range(devices.get_count())]
        except Exception as exc:
            result["errors"]["orbbec"] = str(exc)
    return result


class MotionStatusReader:
    """Read only /status from the existing local motion service."""
    def __init__(self, server, timeout):
        url = urlsplit(server)
        if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost") or
                url.username is not None or url.password is not None or
                url.path not in ("", "/") or url.query or url.fragment):
            raise ValueError("motion_server must be http://127.0.0.1:PORT or http://localhost:PORT")
        self.host, self.port = url.hostname, 80 if url.port is None else url.port
        if not 1 <= self.port <= 65535:
            raise ValueError("motion_server port must be between 1 and 65535")
        self.timeout = timeout

    def status(self):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            connection.request("GET", "/status")
            response = connection.getresponse()
            raw = response.read(1024 * 1024 + 1)
            if response.status != 200 or len(raw) > 1024 * 1024:
                raise RuntimeError(f"Invalid motion status response: HTTP {response.status}")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("Motion status must be a JSON object")
            return result
        finally:
            connection.close()


class ReadOnlyRobotHTTP:
    """No write method exists on this observation transport."""
    def __init__(self, address, timeout):
        url = urlsplit(address)
        if (url.scheme != "http" or not url.hostname or url.username or url.password or
                url.path not in ("", "/") or url.query or url.fragment):
            raise ValueError("robot_address must be http://HOST:PORT")
        self.host, self.port = url.hostname, url.port or 80
        if not 1 <= self.port <= 65535:
            raise ValueError("Invalid robot port")
        self.timeout = timeout

    def get(self, path):
        if path not in ("/getstate", "/info", "/gripper"):
            raise ValueError("Only read-only observation endpoints are allowed")
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            raw = response.read(1024*1024+1)
            if len(raw) > 1024*1024:
                raise ValueError("Robot response too large")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Expected a robot JSON object")
            if response.status != 200:
                raise RuntimeError(data.get("error") or f"Robot HTTP {response.status}")
            return data
        finally:
            connection.close()


class RobotReader:
    def __init__(self, config, timeout):
        self.http = ReadOnlyRobotHTTP(config["robot_address"], timeout)
        self.config = config
        self.last_robot_time = None
        self.last_progress = None
        self.progress_seen = False
        self.generation = str(uuid.uuid4())
        self.tool = None if config.get("flange_to_ee") is None else pose_T(config["flange_to_ee"])
        self.kin = Kinematics(PANDA_ARM_URDF, np.eye(4)) if self.tool is None else None
        self.verified = False

    def read(self):
        if not self.verified:
            info = self.http.get("/info")
            if info.get("server") != "franka_robot_server":
                raise ValueError("Expected the franka_robot_server controller")
            self.verified = True
        begin = time.monotonic()
        raw = self.http.get("/getstate")
        now = time.monotonic()
        robot_time = raw.get("robot_time_s")
        if not valid_number(robot_time):
            raise ValueError("Robot feedback lacks robot_time_s")
        if self.last_robot_time is not None and robot_time < self.last_robot_time:
            self.generation = str(uuid.uuid4())
            self.progress_seen = False
            self.last_robot_time = None
        if self.last_robot_time is None or robot_time > self.last_robot_time:
            if self.last_robot_time is not None:
                self.progress_seen = True
            self.last_progress = now
            self.last_robot_time = robot_time
        age = now-self.last_progress+(now-begin)
        pose = vector(raw.get("pose"), 7, "EE pose")
        joints = vector(raw.get("q"), 7, "joint positions")
        velocity = vector(raw.get("dq"), 7, "joint velocities")
        wrench = vector(raw.get("wrench_K"), 6, "Franka K wrench")
        norm = np.linalg.norm(pose[3:])
        if abs(norm-1.) > .01:
            raise ValueError("Invalid EE quaternion")
        pose[3:] /= norm
        euler_pose = np.r_[pose[:3], R.from_quat(pose[3:]).as_euler("xyz")]
        if self.tool is None and np.max(np.abs(velocity)) <= .02 and self.progress_seen:
            self.tool = np.linalg.inv(pose_T(self.kin.fk(joints))) @ pose_T(euler_pose)
            LOG.info("Inferred flange-to-EE transform from stationary q/pose: %s", T_pose(self.tool).tolist())
        errors = []
        if not self.progress_seen:
            errors.append("Waiting for advancing robot timestamps")
        if raw.get("current_errors"):
            errors.append("Robot reports an error")
        if raw.get("robot_mode") not in (1, 2):
            errors.append("Robot is not in idle/move mode")
        return {"pose": euler_pose, "q": joints, "dq": velocity, "native_wrench": wrench,
                "robot_time_s": robot_time, "generation": self.generation,
                "age_s": age, "received_monotonic_s": now, "errors": errors,
                "tool": None if self.tool is None else self.tool.copy()}


def wrench_values(native, pose, tool, config, mode, reference="ee"):
    """Move the wrench from K to EE/flange, then express it in the chosen axes.

    M_A = R_A_K M_K + p_A_K cross F_A. Sign -1 makes external contact
    environment-on-robot positive, matching this project's motion feedback.
    """
    ee_to_k = pose_T(config["ee_to_stiffness"])
    force = config["force_sign"] * (ee_to_k[:3, :3] @ native[:3])
    torque = config["force_sign"] * (ee_to_k[:3, :3] @ native[3:]) + np.cross(ee_to_k[:3, 3], force)
    if reference not in ("ee", "fingertip", "connector"):
        raise ValueError("wrench_reference must be ee, fingertip or connector")
    if mode == "wrist" and reference != "ee":
        raise ValueError("wrist mode fixes the reference at the flange; use raw or tcp for fingertip/connector")
    offset = config["wrench_reference_points_m"][reference]
    if offset is None:
        raise ValueError(f"{reference} geometry is not configured; set its wrench_reference_points_m offset in EE axes")
    if mode != "wrist":
        torque = torque - np.cross(np.asarray(offset), force)
    base_rotation = R.from_euler("xyz", pose[3:]).as_matrix()
    if mode == "raw":
        return np.r_[base_rotation@force, base_rotation@torque].tolist()
    if mode == "tcp":
        rotation = np.asarray(config["world_from_base"]) @ base_rotation
        return np.r_[rotation@force, rotation@torque].tolist()
    if mode != "wrist":
        raise ValueError("wrench must be raw, wrist or tcp")
    if tool is None:
        raise ValueError("Flange transform unavailable; hold still for inference or configure flange_to_ee")
    flange_force = tool[:3, :3] @ force
    flange_torque = tool[:3, :3] @ torque + np.cross(tool[:3, 3], flange_force)
    return np.r_[flange_force, flange_torque].tolist()


class WrenchBaseline:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.bias = None if enabled else np.zeros(6)
        self.samples, self.times = [], []
        self.started = self.anchor = self.generation = self.last_sequence = None
        self.fault = None
        self.sample_count = 0
        self.calibration_id = None
        self.completed_at_unix_s = None

    def interrupt(self):
        if self.bias is None:
            self.samples.clear()
            self.times.clear()
            self.started = self.anchor = self.last_sequence = None

    def status(self):
        return {"ready": self.bias is not None and self.fault is None, "enabled": self.enabled,
                "error": self.fault, "duration_s": WRENCH_ZERO_SECONDS,
                "sample_count": self.sample_count, "calibration_id": self.calibration_id,
                "completed_at_unix_s": self.completed_at_unix_s,
                "bias_wrench_K": None if self.bias is None else self.bias.tolist()}

    def apply(self, sample):
        if self.fault:
            raise ValueError(self.fault)
        if sample["errors"] or sample["age_s"] > .1:
            self.interrupt()
            raise ValueError("Wrench tare needs fresh advancing feedback (progress age <= 0.1 s)")
        if self.generation is not None and self.generation != sample["generation"]:
            self.fault = "Robot clock reset; re-tare robot_wrench while stationary and unloaded"
            raise ValueError(self.fault)
        self.generation = sample["generation"]
        if self.bias is None:
            now = time.monotonic()
            if self.anchor is None:
                if np.max(np.abs(sample["dq"])) > .02:
                    raise ValueError("Waiting for stationary robot before wrench tare")
                self.anchor = (sample["pose"][:3].copy(), sample["q"].copy())
                self.started = now
            if (np.linalg.norm(sample["pose"][:3]-self.anchor[0]) > .001 or
                    np.max(np.abs(sample["q"]-self.anchor[1])) > np.deg2rad(.5) or
                    np.max(np.abs(sample["dq"])) > .02):
                self.fault = "Robot moved during tare; re-tare while stationary and unloaded"
                raise ValueError(self.fault)
            if sample["robot_time_s"] != self.last_sequence and now-self.started <= WRENCH_ZERO_SECONDS:
                self.samples.append(sample["native_wrench"].copy())
                self.times.append(now)
                self.last_sequence = sample["robot_time_s"]
            if now-self.started < WRENCH_ZERO_SECONDS:
                raise ValueError("Force/torque tare in progress; keep robot still and unloaded")
            if len(self.samples) < 10 or self.times[-1]-self.times[0] < .75*WRENCH_ZERO_SECONDS:
                self.fault = "Insufficient distinct feedback samples for one-second tare"
                raise ValueError(self.fault)
            self.bias = np.mean(self.samples, axis=0)
            self.sample_count = len(self.samples)
            self.calibration_id = str(uuid.uuid4())
            self.completed_at_unix_s = time.time()
            self.samples.clear()
            self.times.clear()
            LOG.info("Robot wrench tared from %d samples (K axes): %s", self.sample_count, self.bias.tolist())
        return sample["native_wrench"]-self.bias


def encode_array(key, kind, array):
    data = np.asarray(array)
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Empty or nonfinite camera array")
    if kind == "rgb":
        if data.dtype != np.uint8 or data.ndim != 3 or data.shape[2] != 3:
            raise ValueError("Expected uint8 HxWx3 RGB image")
        success, encoded = cv2.imencode(".png", cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("PNG encoding failed")
        return f"{key}.png", encoded.tobytes()
    if kind != "depth" or data.ndim != 2 or np.any(data < 0) or not np.any(data > 0):
        raise ValueError("Expected depth in metres with valid positive samples")
    stream = io.BytesIO()
    np.save(stream, data.astype(np.float32, copy=False), allow_pickle=False)
    return f"{key}.npy", stream.getvalue()




class ObservationService:
    def __init__(self, config, timeout=2., startup_timeout=30., state_hz=20., max_age=.5,
                 worker_factory=SensorWorker):
        self.config = deepcopy(config)
        self.devices = sensor_configs(config)
        self.timeout = positive(timeout, "timeout")
        self.startup_timeout = positive(startup_timeout, "startup_timeout")
        self.period = 1/positive(state_hz, "state_hz")
        self.max_age = positive(max_age, "max_age")
        self.worker_factory = worker_factory
        self.server_id = str(uuid.uuid4())
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.threads, self.latest = [], {}
        self.wrench_baseline = WrenchBaseline(config["tare_on_start"])
        self.calibration_job = None
        self.health = {name: {"phase": "starting", "error": None}
                       for name in [*self.devices, *SCALAR_KEYS]}
        for name, camera in config["cameras"].items():
            if name not in self.devices:
                self.health[name] = {"phase": "disabled", "error": camera.get("note", "Camera disabled")}

    def start(self):
        if self.threads:
            raise RuntimeError("Observation service already started")
        for name in [*self.devices, *SCALAR_KEYS]:
            target = self._sensor_loop if name in self.devices else self._scalar_loop
            thread = threading.Thread(target=target, args=(name,), name=f"observe-{name}", daemon=True)
            self.threads.append(thread)
            thread.start()

    def _publish(self, name, item, errors=()):
        with self.lock:
            first = self.health[name]["phase"] == "starting"
            self.latest[name] = item
            self.health[name] = {"phase": "running", "error": "; ".join(errors) if errors else None}
        if first:
            LOG.info("%s initialized; acquiring continuously", name)

    def _error(self, name, exc):
        message = f"{type(exc).__name__}: {exc}"
        with self.lock:
            changed = self.health[name].get("error") != message
            self.health[name] = {"phase": "error", "error": message}
            self.latest.pop(name, None)
        if changed:
            LOG.warning("%s: %s", name, message)

    def _sensor_loop(self, name):
        worker = None
        try:
            category, config = self.devices[name]
            worker = self.worker_factory(category, config, self.timeout, self.startup_timeout)
            while not self.stop_event.is_set():
                start = time.monotonic()
                sample = worker.capture(CAMERA_KINDS)
                meta = dict(sample["meta"])
                meta.setdefault("received_monotonic_s", time.monotonic())
                self._publish(name, {"meta": meta, "arrays": sample["arrays"]})
                self.stop_event.wait(max(0., 1/config["fps"]-(time.monotonic()-start)))
        except Exception as exc:
            if not self.stop_event.is_set():
                self._error(name, exc)
        finally:
            if worker is not None:
                worker.close()

    def _scalar_loop(self, key):
        if key == "state":
            reader = RobotReader(self.config, min(self.timeout, .5))
        elif key == "motion_status":
            reader = MotionStatusReader(self.config["motion_server"], min(self.timeout, .5))
        else:
            reader = ReadOnlyRobotHTTP(self.config["robot_address"], min(self.timeout, .5))
        while not self.stop_event.is_set():
            begin = time.monotonic()
            try:
                if key == "state":
                    sample = reader.read()
                    data = {"eef_pose": sample["pose"].tolist(), "joint_angles": sample["q"].tolist(),
                            "joint_velocities": sample["dq"].tolist()}
                    item = {"data": data, "received_monotonic_s": sample["received_monotonic_s"],
                            "sample_age_s_at_receive": sample["age_s"], "errors": sample["errors"],
                            "field_errors": {}, "sample": sample}
                    with self.lock:
                        job = self.calibration_job
                        if job is not None:
                            with suppress(ValueError):
                                job["candidate"].apply(sample)
                            status = job["candidate"].status()
                            if status["ready"] or status["error"]:
                                if status["ready"]:
                                    self.wrench_baseline = job["candidate"]
                                job["result"] = status
                                self.calibration_job = None
                                job["done"].set()
                        try:
                            corrected = self.wrench_baseline.apply(sample)
                            item["corrected_wrench_K"] = corrected
                            data["force"] = wrench_values(corrected, sample["pose"], sample["tool"], self.config, "raw")
                            item["calibration_id"] = self.wrench_baseline.calibration_id
                        except ValueError as exc:
                            item["field_errors"]["force"] = str(exc)
                        self._publish(key, item, sample["errors"]+list(item["field_errors"].values()))
                else:
                    raw = reader.status() if key == "motion_status" else reader.get("/gripper")
                    if key == "motion_status":
                        if (any(type(raw.get(name)) is not bool for name in ("ready", "stop_requested", "stopping")) or
                                not isinstance(raw.get("queued"), list) or "active" not in raw):
                            raise ValueError("Invalid motion-server status")
                        fields = ("ready", "homed", "active_mode", "active", "queued", "pending_home",
                                  "fault", "stop_requested", "stopping", "worker_alive")
                        data = {name: raw.get(name) for name in fields}
                    else:
                        if raw.get("state") not in ("idle", "pending", "moving", "stopping", "succeeded", "failed"):
                            raise ValueError("Invalid gripper command status")
                        width = raw.get("width_m")
                        if width is not None and (not valid_number(width) or not 0 <= width <= .08):
                            raise ValueError("Invalid commanded gripper width")
                        data = {"state": raw.get("state"), "target_width_m": width,
                                "speed_m_s": raw.get("speed_m_s"), "error": raw.get("error"),
                                "requested_at_controller_s": raw.get("requested_at"),
                                "measurement_available": False, "measured_width_m": None,
                                "source": "controller_command_status"}
                    item = {"data": clean_json(data), "received_monotonic_s": time.monotonic(),
                            "sample_age_s_at_receive": time.monotonic()-begin, "errors": [], "field_errors": {}}
                    self._publish(key, item)
            except Exception as exc:
                if key == "state":
                    with self.lock:
                        self.wrench_baseline.interrupt()
                        if self.calibration_job is not None:
                            self.calibration_job["candidate"].interrupt()
                self._error(key, exc)
            self.stop_event.wait(max(0., self.period-(time.monotonic()-begin)))

    @staticmethod
    def _age(item, now):
        baseline = item.get("sample_age_s_at_receive", 0.)
        if not valid_number(baseline) or baseline < 0:
            return math.inf
        return baseline+max(0., now-item["received_monotonic_s"])

    def status(self):
        with self.lock:
            health = deepcopy(self.health)
            now = time.monotonic()
            for name, status in health.items():
                item = self.latest.get(name)
                if item is None:
                    status["fresh"] = False
                    continue
                meta = item.get("meta", item)
                age = self._age(meta, now)
                status.update(age_s=age if math.isfinite(age) else None,
                              fresh=not status["error"] and age <= self.max_age,
                              frame_number=meta.get("frame_number"))
            required = [*self.devices, *SCALAR_KEYS]
            return {"protocol": 1, "server_id": self.server_id,
                    "ready": all(health[name]["fresh"] for name in required),
                    "sensors_ready": all(health[name]["fresh"] for name in self.devices),
                    "sources": health, "default_observations": self.config["default_observations"],
                    "calibration": {"robot_wrench": self.wrench_baseline.status(),
                                    "pending": ["robot_wrench"] if self.calibration_job else []},
                    "wrench": {"source": "Franka K_F_ext_hat_K", "force_sign": self.config["force_sign"],
                               "ee_to_stiffness": self.config["ee_to_stiffness"],
                               "stiffness_transform_source": "local_configuration_not_exposed_by_robot_API",
                               "flange_to_ee_source": "configured" if self.config.get("flange_to_ee") is not None else "inferred_from_stationary_q_and_pose",
                               "world_from_base": self.config["world_from_base"],
                               "reference_points_m_in_EE": self.config["wrench_reference_points_m"],
                               "units": ["N", "N", "N", "N.m", "N.m", "N.m"]},
                    "freshness_basis": "robot_time_progress_plus_http_round_trip; true robot sample age unavailable",
                    "hardware_synchronized": False, "measured_gripper_width_available": False}

    def calibrate(self, request):
        if not isinstance(request, dict) or request != {"target": "robot_wrench"}:
            raise ValueError("Only target: robot_wrench is supported; no tactile sensors are installed")
        from get_observation import ObservationClient
        ObservationClient(motion_server=self.config["motion_server"],
                          timeout=min(self.timeout, .5)).ensure_motion_finished()
        with self.lock:
            if self.calibration_job is not None:
                raise ValueError("Robot wrench calibration already running")
            job = {"candidate": WrenchBaseline(), "done": threading.Event()}
            self.calibration_job = job
        if not job["done"].wait(4.):
            with self.lock:
                if not job["done"].is_set():
                    self.calibration_job = None
                    raise ValueError("Calibration timed out; previous baseline retained")
        if job["result"]["error"]:
            raise ValueError(job["result"]["error"]+"; previous baseline retained")
        return {"target": "robot_wrench", **job["result"]}

    def snapshot(self, request):
        if not isinstance(request, dict) or set(request)-{"observations", "max_age_s", "wrench", "wrench_source", "wrench_reference"}:
            raise ValueError("Request accepts observations, max_age_s, wrench, wrench_source and wrench_reference")
        if request.get("wrench_source", "franka") != "franka":
            raise ValueError("Only the Franka wrench estimate is available; no tactile sensors are installed")
        mode = request.get("wrench", "raw")
        if mode not in ("raw", "wrist", "tcp"):
            raise ValueError("wrench must be raw, wrist or tcp")
        reference = request.get("wrench_reference", "ee")
        if reference not in ("ee", "fingertip", "connector"):
            raise ValueError("wrench_reference must be ee, fingertip or connector")
        if mode == "wrist" and reference != "ee":
            raise ValueError("wrist mode fixes the reference at the flange")
        requested = expand_observations(request.get("observations", self.config["default_observations"]), self.config)
        max_age = positive(request.get("max_age_s", self.max_age), "max_age_s")
        with self.lock:
            latest, health = dict(self.latest), deepcopy(self.health)
        selected = time.monotonic()
        result = {"protocol": 2, "observations": {}, "errors": {}, "ages": {}}
        files = {}
        for key in requested:
            try:
                if key == "gripper":
                    raise ValueError("Measured gripper width is not exposed by this controller; request gripper_status for command status")
                source = "state" if key in STATE_FIELDS else key.split(".")[0]
                if source not in latest:
                    raise ValueError(health.get(source, {}).get("error") or "Source is still initializing")
                item = latest[source]
                age = self._age(item.get("meta", item), selected)
                if age > max_age:
                    raise ValueError("Latest sample is stale")
                if item.get("errors"):
                    raise ValueError("; ".join(item["errors"]))
                if source in SCALAR_KEYS:
                    value = item["data"]
                    if key in ("state", "force", "torque", "wrench"):
                        if item["field_errors"].get("force"):
                            raise ValueError(item["field_errors"]["force"])
                        sample = item["sample"]
                        wrench = wrench_values(item["corrected_wrench_K"], sample["pose"], sample["tool"], self.config, mode, reference)
                        if key == "state":
                            value = {"force": wrench, "eef_pose": value["eef_pose"], "joint_angles": value["joint_angles"]}
                        elif key == "torque":
                            value = wrench[3:]
                        elif key == "force":
                            value = wrench
                        else:
                            value = {"force_n": wrench[:3], "torque_nm": wrench[3:],
                                     "source": "Franka K_F_ext_hat_K",
                                     "reference": "flange" if mode == "wrist" else ("EE" if reference == "ee" else reference),
                                     "frame": {"raw": "base", "wrist": "flange", "tcp": "world"}[mode],
                                     "force_sign": self.config["force_sign"],
                                     "calibration_id": item.get("calibration_id")}
                    elif key in STATE_FIELDS:
                        value = value[key]
                else:
                    kind = key.split(".")[1]
                    value, payload = encode_array(key, kind, item["arrays"][kind])
                    files[value] = payload
                result["observations"][key] = value
                result["ages"][key] = age
            except Exception as exc:
                result["errors"][key] = str(exc)
        return result, files

    def close(self):
        self.stop_event.set()
        deadline = time.monotonic()+self.startup_timeout+self.timeout+5
        for thread in self.threads:
            thread.join(max(0., deadline-time.monotonic()))




def bundle_snapshot(result, files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(result, allow_nan=False))
        for name, contents in files.items():
            archive.writestr(name, contents)
    if buffer.tell() > MAX_BUNDLE_BYTES:
        raise ValueError("Requested observation exceeds the bundle size limit")
    return buffer.getvalue()


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5.0)

        def log_message(self, *args):
            pass

        def respond(self, status, body, content_type="application/json"):
            if content_type == "application/json":
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if self.path == "/status":
                    self.respond(200, service.status())
                else:
                    self.respond(404, {"error": "Unknown endpoint"})
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            try:
                if self.path not in ("/observation", "/calibrate"):
                    self.respond(404, {"error": "Unknown endpoint"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("Request must have a Content-Length between 1 and 16384")
                request = json.loads(self.rfile.read(length))
                if self.path == "/calibrate":
                    self.respond(200, service.calibrate(request))
                    return
                result, files = service.snapshot(request)
                self.respond(200, bundle_snapshot(result, files), "application/zip")
            except (ValueError, UnicodeError) as exc:
                self.respond(400, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                LOG.exception("Observation request failed")
                self.respond(500, {"error": str(exc)})
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--robot-address", help="Override robot_address in the local config")
    parser.add_argument("--motion-server", help="Override motion_server in the local config")
    parser.add_argument("--timeout", type=positive_number, default=2.0)
    parser.add_argument("--startup-timeout", type=positive_number, default=30.0)
    parser.add_argument("--state-hz", type=positive_number, default=20.0)
    parser.add_argument("--max-age", type=positive_number, default=0.5)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.list_devices:
        print(json.dumps(list_devices(), allow_nan=False))
        return 0
    config = load_config(args.config)
    if args.robot_address:
        config["robot_address"] = args.robot_address
        ReadOnlyRobotHTTP(args.robot_address, args.timeout)
    if args.motion_server:
        config["motion_server"] = args.motion_server
        MotionStatusReader(args.motion_server, args.timeout)
    service = ObservationService(config, timeout=args.timeout, startup_timeout=args.startup_timeout,
                                 state_hz=args.state_hz, max_age=args.max_age)
    # Bind before any worker/device startup, preventing duplicate-port baseline resets.
    http = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(service))
    http.daemon_threads = True
    stopping = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopping.set())
    try:
        service.start()
        LOG.info("Observation server at http://127.0.0.1:%s; starting %s. GET /status reports readiness.",
                 http.server_port, ", ".join(service.health))
        http.timeout = 0.2
        while not stopping.is_set():
            http.handle_request()
    finally:
        http.server_close()
        service.close()
        LOG.info("Observation devices released")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
