#!/usr/bin/env python3
"""Start observation devices once and serve their latest samples on localhost.

Terminal 3, from the project directory:
  conda activate agentic_manip
  python observation_server.py

Edit CONFIG below to set device identities. Every configured camera and tactile
sensor starts immediately, without a client request. Keep tactile pads unloaded
at startup: DM-Tac initializes its baseline in Sensor(). Connections stay open;
failed sensors are never automatically reinitialized. Restart this server to
reconnect a failed sensor.
Keep the robot stationary and free of contact at startup: its first 1 second
of fresh feedback establish a fixed force/torque baseline for all wrench modes.
This is single-pose zeroing, not payload calibration; offsets can change with pose.

  --list-devices            Enumerate devices and exit, without starting sensors.

The server continuously captures every supported modality and keeps one latest
sample per source in RAM. Client selection affects only the response and saving.
It also polls the existing robot/motion services using read-only requests;
it never starts,
stops, homes, powers or commands the robot. Unavailable robot services do not
block camera acquisition. Devices and SDK dependencies belong to this server.

GET /status reports startup, freshness and errors. POST /observation accepts
{"observations":["state","left_wrist.rgb"],"max_age_s":0.5}; an empty object
selects the configured defaults. A response is a ZIP containing manifest.json
and only the requested PNG/NPY files. get_observation.py saves that response.
Optional "wrench": "tcp" or "wrist" includes flange data and calibration for
client conversion. Omit it for legacy base feedback. Only force is affected.
No sensor files are recorded until a client requests a snapshot. The selected
samples may predate the request; freshness is checked before returning data.
Only requested values and file names are returned, with per-observation errors.
Sources are not hardware synchronized. Ctrl+C releases observation devices only.
Combined tactile observations use the latest individually fresh samples;
cross-source timestamp skew is reported but does not reject an observation.

POST /calibrate {"target":"robot_wrench"|"tactile_wrench"} averages one second
of fresh samples. Tactile tare covers both wrenches AND deformation fields.
After closing on a connector, tare tactile while clear of external contact.
This software tare does not reset the SDK image baseline. Startup SDK reference
initialization still requires unloaded pads. Re-tare after changing the grasp.
Tactile .force is the legacy uncalibrated SDK six-vector; use .wrench for tare,
units and frame metadata. Unknown torque scale is reported as torque_nm=null.
--tactile-reference in get_observation selects fingertip (20 mm) or connector
(30 mm) ahead of the pad midpoint, with gripper or world axes.
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
import msgpack
import numpy as np
import zmq

from observation_support.tactile import (
    DEFAULT_GEOMETRY, PAD_NAMES, TIP_OFFSETS, TactileBaseline,
    combined_wrench, frame_context, pad_wrench, robot_at_tip, validate_geometry,
)

LOG = logging.getLogger("observation_server")

# Device setup lives here. Restart this server after changing it.
CONFIG = {
    "robot_address": "ipc:///tmp/robot_server_5555",
    "motion_server": "http://127.0.0.1:18080",
    # Request shorthand when the client omits --observations; never a startup filter.
    "default_observations": ["state", "motion_status", "gripper", "left_wrist.rgb", "left_wrist.depth"],
    "cameras": {
        "left_wrist": {"type": "realsense", "serial": "LEFT_WRIST_REALSENSE_SERIAL",
                       "width": 640, "height": 480, "fps": 30, "warmup_frames": 10},
        "right_wrist": {"type": "realsense", "serial": "RIGHT_WRIST_REALSENSE_SERIAL",
                        "width": 640, "height": 480, "fps": 30, "warmup_frames": 10},
        "head": {"type": "orbbec", "serial": "HEAD_ORBBEC_SERIAL",
                 "width": 640, "height": 480, "fps": 30, "warmup_frames": 10},
    },
    "tactile": {
        "tactile_left": {"serial": "LEFT_TACTILE_SERIAL", "backend": "cpu", "pc_port": 60000, "fps": 30},
        "tactile_right": {"serial": "RIGHT_TACTILE_SERIAL", "backend": "cpu", "pc_port": 60001, "fps": 30},
    },
    # Physical pad midpoint is independent of the robot's selectable TCP.
    # Set torque_nm_per_native only after validating the SDK scale.
    "tactile_geometry": deepcopy(DEFAULT_GEOMETRY),
}

SCALAR_KEYS = ("state", "gripper", "motion_status")
STATE_FIELDS = {"force": ("ext_wrench", 6), "eef_pose": ("cart_pos_real", 6),
                "joint_angles": ("joint_pos_real", 7)}
SENSOR_KINDS = {"cameras": ("rgb", "depth"), "tactile": ("image", "deformation", "force")}
OUTPUT_KINDS = {"cameras": SENSOR_KINDS["cameras"],
                "tactile": (*SENSOR_KINDS["tactile"], "deformation_raw", "wrench")}
TIP_KEYS = ("tactile_tip.wrench", "tactile_tip_world.wrench")
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


def sensor_configs(config):
    return {name: (category, value) for category in SENSOR_KINDS
            for name, value in config.get(category, {}).items()}


def expand_observations(requested, config):
    available = [*SCALAR_KEYS, *STATE_FIELDS, *TIP_KEYS]
    sensors = sensor_configs(config)
    for name, (category, _) in sensors.items():
        available.extend(f"{name}.{kind}" for kind in OUTPUT_KINDS[category])
    if not isinstance(requested, list) or not requested or not all(isinstance(k, str) for k in requested):
        raise ValueError("observations must be a nonempty list of names")
    if "all" in requested:
        if requested != ["all"]:
            raise ValueError("use all by itself")
        return list(SCALAR_KEYS) + [f"{n}.{k}" for n, (category, _) in sensors.items()
                                    for k in OUTPUT_KINDS[category]] + list(TIP_KEYS)
    # Accept old source names, but return canonical physical left/right names.
    requested = [k.replace("tactile_left_1.", "tactile_left.").replace(
        "tactile_left_2.", "tactile_right.") for k in requested]
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"Unknown observations: {sorted(unknown)}. Available: {available}")
    return list(dict.fromkeys(requested))


def load_config():
    """Validate and copy the configuration embedded in this server."""
    config = deepcopy(CONFIG)
    names = [name for category in SENSOR_KINDS for name in config.get(category, {})]
    if len(names) != len(set(names)) or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", n) or
                                         n in (*SCALAR_KEYS, *STATE_FIELDS) for n in names):
        raise ValueError("Sensor names must be unique identifiers, distinct from scalar keys")
    serials = set()
    for _, (category, value) in sensor_configs(config).items():
        serial = value.get("serial")
        if not isinstance(serial, str) or not serial or serial in serials:
            raise ValueError("Each sensor requires a unique, explicit serial string")
        serials.add(serial)
        positive(value.get("fps", 30), "fps")
        if category == "cameras" and value.get("type") not in ("realsense", "orbbec"):
            raise ValueError("Camera type must be realsense or orbbec")
    expand_observations(config["default_observations"], config)
    validate_geometry(config["tactile_geometry"])
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
        for _ in range(config.get("warmup_frames", 10)):
            self.pipeline.wait_for_frames(self.timeout)

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
        for _ in range(config.get("warmup_frames", 10)):
            if self.pipeline.wait_for_frames(self.timeout) is None:
                raise TimeoutError("Orbbec warmup timed out")

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
        scale = depth.get_depth_scale() / 1000.0
        meta = {"received_at_unix_s": received}
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


def sdk_value(result):
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("SDK did not return a (frame_id, value) pair")
    return int(result[0]), result[1]


class TactileSource:
    def __init__(self, config, timeout):
        from dmrobotics import Mode, Sensor, SensorOptions
        self.timeout = int(timeout * 1000)
        # Starting this configured sensor resets its baseline once at server startup.
        self.sensor = Sensor(SensorOptions(dev_id=config["serial"], backend=config.get("backend", "cpu"),
                             mode=Mode.STANDARD, max_fps=config.get("fps", 30),
                             pc_port=config["pc_port"], enable_raw=False, enable_deformation=True,
                             enable_depth=True, enable_force=True, enable_shear=False))
        for _ in range(config.get("warmup_frames", 5)):
            self.capture(["image", "deformation", "force"])

    def capture(self, kinds):
        import numpy as np
        last, _ = sdk_value(self.sensor.getInferImg())
        deadline = time.monotonic() + self.timeout / 1000
        while time.monotonic() < deadline:
            if not self.sensor.wait_for_new(last, timeout_ms=max(1, int((deadline-time.monotonic())*1000))):
                break
            status = self.sensor.getDevStatus()
            if status != 0:
                raise RuntimeError(f"DM-Tac status {status}")
            values, ids = {}, []
            if "image" in kinds:
                fid, value = sdk_value(self.sensor.getInferImg())
                ids.append(fid)
                values["image"] = np.asarray(value.img).copy()
            if "deformation" in kinds:
                a_id, xy = sdk_value(self.sensor.getDeformation2D())
                b_id, z = sdk_value(self.sensor.getDepth())
                ids.extend((a_id, b_id))
                xy, z = np.asarray(xy), np.asarray(z)
                if xy.ndim != 3 or xy.shape[2] != 2 or z.shape != xy.shape[:2]:
                    raise RuntimeError("Invalid DM-Tac deformation/depth shapes")
                values["deformation"] = np.dstack((xy, z)).astype(np.float32)
            if "force" in kinds:
                fid, value = sdk_value(self.sensor.getForce())
                ids.append(fid)
                values["force"] = np.asarray(value, dtype=np.float32).reshape(-1)
            # SDK getters can race an update; accept only matching, new frame IDs.
            if ids and min(ids) > last and len(set(ids)) == 1:
                if any(a.size == 0 or not np.isfinite(a).all() for a in values.values()):
                    raise RuntimeError("Invalid/nonfinite tactile data")
                return {"received_at_unix_s": time.time(), "frame_number": ids[0]}, values
            last = max([last] + ids)
        raise TimeoutError("No new consistent tactile frame before timeout")

    def close(self):
        self.sensor.disconnect()


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
        # A second server (even on a different HTTP port) must not reopen a pad
        # and reset its baseline. Keep the advisory lock for the worker lifetime.
        lock_name = hashlib.sha256(config["serial"].encode()).hexdigest()[:24]
        device_lock = open(f"/tmp/agentic_observation_{lock_name}.lock", "a")
        try:
            fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Sensor {config['serial']} is owned by another observation server") from exc
        cls = TactileSource if category == "tactile" else {
            "realsense": RealSenseSource, "orbbec": OrbbecSource}[config["type"]]
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
    result = {"realsense": [], "orbbec": [], "tactile": [], "errors": {}}
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
        # USB enumeration works without loading CUDA or initializing a sensor.
        for path in sorted(Path("/dev/v4l/by-id").glob("usb-Generic_N160MU2_*-video-index0")):
            result["tactile"].append({"serial": path.name.removeprefix("usb-Generic_N160MU2_").removesuffix("-video-index0"),
                                      "device": str(path.resolve()), "stable_path": str(path)})
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


class RobotReader:
    """One persistent read-only REQ socket, used and closed by its owning thread."""
    def __init__(self, address, method, timeout):
        if method not in ("motion_state", "motion_gripper_state"):
            raise ValueError("Only observation RPCs are allowed")
        self.address, self.method, self.timeout = address, method, timeout
        self.include_wrench = False
        self.context = zmq.Context()
        self.socket = None
        self._connect()

    def _connect(self):
        if self.socket is not None:
            self.socket.close()
        self.socket = self.context.socket(zmq.REQ)
        for setting, value in ((zmq.LINGER, 0), (zmq.SNDTIMEO, max(1, int(self.timeout*1000))),
                               (zmq.RCVTIMEO, max(1, int(self.timeout*1000)))):
            self.socket.setsockopt(setting, value)
        self.socket.connect(self.address)

    def read(self):
        try:
            # Read-only requests deliberately omit the motion owner's lease token.
            params = {"include_wrench": True} if self.include_wrench else {}
            self.socket.send(msgpack.packb({"method": self.method, "params": params}, use_bin_type=True))
            result = msgpack.unpackb(self.socket.recv(), raw=False)
            if not result.get("success") or not isinstance(result.get("result"), dict):
                raise RuntimeError(result.get("error", "Invalid robot observation reply"))
            return result["result"]
        except zmq.ZMQError as exc:
            self._connect()  # Restore REQ/REP after timeout; never resend this call.
            raise RuntimeError(f"{self.method} unavailable at {self.address}: {exc}") from exc

    def close(self):
        self.socket.close()
        self.context.term()


def scalar_item(key, data, elapsed):
    """Keep readings and internal freshness checks, never the full RPC payload."""
    errors = []
    field_errors = {}
    age = 0.0
    wrench = None
    if key == "state":
        wrench = data.get("wrench")
        if isinstance(wrench, dict):
            wrench = dict(wrench)
            wrench_age = wrench.get("age_s")
            wrench["age_s"] = wrench_age + elapsed if valid_number(wrench_age) and wrench_age >= 0 else None
        raw_age = data.get("state_age_s")
        age = raw_age + elapsed if valid_number(raw_age) and raw_age >= 0 else None
        if data.get("protocol") != 1:
            errors.append("Robot wrapper lacks observation protocol 1")
        if age is None:
            errors.append("Native feedback has no valid sample timestamp")
        if data.get("in_realtime_loop") is not True:
            errors.append("Realtime loop is off; native feedback is cached")
        readings = {}
        for name, (field, size) in STATE_FIELDS.items():
            values = data.get(field)
            if not isinstance(values, list) or len(values) != size or not all(valid_number(v) for v in values):
                field_errors[name] = f"Invalid {field}"
            else:
                readings[name] = values
        data = readings
    elif key == "gripper":
        raw_age = data.get("age_s")
        age = raw_age + elapsed if valid_number(raw_age) and raw_age >= 0 else None
        if data.get("valid") is not True or not valid_number(data.get("raw")) or age is None:
            errors.append("Gripper feedback invalid or has no valid timestamp")
        data = data.get("raw")
    else:
        data = {"ready": data.get("ready"), "homed": data.get("homed"),
                "active_mode": data.get("active_mode"), "active": data.get("active"),
                "queued": data.get("queued"), "fault": data.get("fault"),
                "stop_requested": data.get("stop_requested"), "stopping": data.get("stopping")}
    item = {"received_monotonic_s": time.monotonic(), "sample_age_s_at_receive": age,
            "data": clean_json(data), "errors": errors, "field_errors": field_errors}
    if key == "state":
        item["wrench"] = clean_json(wrench)  # Transport context; the client saves only converted values.
    return item, errors + list(field_errors.values())


def encode_array(key, kind, array):
    """Encode sensor arrays in the server so the client needs only Python stdlib."""
    data = np.asarray(array)
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Empty or nonfinite sensor array")
    if kind == "force":
        if data.shape != (6,):
            raise ValueError("Expected six tactile force components")
        return data.tolist(), None
    if kind in ("rgb", "image"):
        if data.dtype != np.uint8:
            raise ValueError("Expected uint8 image")
        pixels = cv2.cvtColor(data, cv2.COLOR_RGB2BGR) if kind == "rgb" else data
        success, encoded = cv2.imencode(".png", pixels)
        if not success:
            raise RuntimeError("PNG encoding failed")
        return f"{key}.png", encoded.tobytes()
    if kind == "depth":
        if data.ndim != 2 or np.any(data < 0) or not np.any(data > 0):
            raise ValueError("Depth contains no valid positive samples or has invalid shape/values")
    else:
        if data.ndim != 3 or data.shape[2] != 3:
            raise ValueError("Expected HxWx3 tactile deformation")
    stream = io.BytesIO()
    np.save(stream, data.astype(np.float32, copy=False), allow_pickle=False)
    return f"{key}.npy", stream.getvalue()


class WrenchBaseline:
    """One tare window, in flange axes for explicit wrench and base for legacy."""
    def __init__(self):
        self.samples = []
        self.sample_times = []
        self.started = None
        self.anchor = None
        self.identity = None
        self.bias = None
        self.fault = None
        self.last_sequence = None
        self.sample_count = 0
        self.calibration_id = None
        self.completed_at_unix_s = None

    def interrupt(self):
        if self.bias is None:
            self.samples.clear()
            self.sample_times.clear()
            self.started = None
            self.anchor = None
            self.last_sequence = None

    def status(self):
        return {"ready": self.bias is not None and not self.fault, "error": self.fault,
                "duration_s": WRENCH_ZERO_SECONDS, "sample_count": self.sample_count,
                "calibration_id": self.calibration_id, "completed_at_unix_s": self.completed_at_unix_s}

    def apply(self, state, elapsed=0.0):
        if self.fault:
            raise ValueError(self.fault)
        context = state.get("wrench")
        try:
            if not isinstance(context, dict) or context.get("error"):
                raise ValueError("No explicit flange wrench for startup zeroing")
            ages = (state.get("state_age_s"), context.get("age_s"))
            if (state.get("protocol") != 1 or state.get("in_realtime_loop") is not True or
                    any(not valid_number(a) or a < 0 or a + elapsed > 0.1 for a in ages)):
                raise ValueError("Startup zeroing needs fresh realtime feedback (age <= 0.1 s)")

            def vector(value, size):
                if not isinstance(value, list) or len(value) != size or not all(valid_number(v) for v in value):
                    raise ValueError("Invalid wrench, pose or joint feedback for startup zeroing")
                return np.asarray(value, dtype=float)

            readings = np.stack((vector(context.get("flange"), 6), vector(state.get("ext_wrench"), 6)))
            pose = vector(context.get("tcp_pose_base"), 6)
            joints = vector(state.get("joint_pos_real"), 7)
            identity = (state.get("server_id"), tuple(vector(context.get("tool_end"), 6)))
            if not isinstance(identity[0], str) or not identity[0]:
                raise ValueError("Robot server identity missing for startup zeroing")
        except ValueError:
            # Before zeroing completes, require a continuous window of fresh data.
            self.interrupt()
            raise

        if self.identity is not None and identity != self.identity:
            self.fault = "Robot server/tool changed; request robot_wrench calibration or restart unloaded"
            raise ValueError(self.fault)
        if self.bias is None:
            now = time.monotonic()
            if self.anchor is None:
                self.anchor = (pose[:3].copy(), joints.copy())
                self.identity = identity
            if (np.linalg.norm(pose[:3] - self.anchor[0]) > 0.001 or
                    np.max(np.abs(joints - self.anchor[1])) > np.deg2rad(0.5)):
                self.fault = "Robot moved during zeroing; request robot_wrench calibration or restart unloaded and still"
                raise ValueError(self.fault)
            if self.started is None:
                self.started = now
            sequence = state.get("state_seq")
            if (sequence is None or sequence != self.last_sequence) and now-self.started <= WRENCH_ZERO_SECONDS:
                self.samples.append(readings)
                self.sample_times.append(now)
                self.last_sequence = sequence
            if now - self.started < WRENCH_ZERO_SECONDS:
                raise ValueError("Force/torque zeroing in progress; keep robot still and unloaded")
            if len(self.samples) < 10 or self.sample_times[-1]-self.sample_times[0] < .75*WRENCH_ZERO_SECONDS:
                self.fault = "Insufficient distinct robot samples in one second; request calibration again"
                raise ValueError(self.fault)
            self.bias = np.mean(self.samples, axis=0)
            self.sample_count = len(self.samples)
            self.calibration_id = str(uuid.uuid4())
            self.completed_at_unix_s = time.time()
            LOG.info("Force/torque zeroed from %d samples: flange [N,N.m]=%s; legacy base=%s",
                     len(self.samples), self.bias[0].tolist(), self.bias[1].tolist())
            self.samples.clear()
            self.sample_times.clear()
        return readings - self.bias


class ObservationService:
    def __init__(self, config, timeout=2.0, startup_timeout=30.0,
                 state_hz=20.0, max_age=0.5, worker_factory=SensorWorker):
        self.config = config
        self.devices = sensor_configs(config)
        self.timeout = positive(timeout, "timeout")
        self.startup_timeout = positive(startup_timeout, "startup_timeout")
        self.period = 1 / positive(state_hz, "state_hz")
        self.max_age = positive(max_age, "max_age")
        self.worker_factory = worker_factory
        self.server_id = str(uuid.uuid4())
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.threads = []
        self.latest = {}
        self.wrench_baseline = WrenchBaseline()
        self.tactile_baseline = TactileBaseline()
        self.geometry = deepcopy(config.get("tactile_geometry", DEFAULT_GEOMETRY))
        validate_geometry(self.geometry)
        self.calibration_jobs = {}
        self.health = {name: {"phase": "starting", "error": None} for name in [*self.devices, *SCALAR_KEYS]}

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
            if name in PAD_NAMES:
                now = time.monotonic()
                self.tactile_baseline.offer(self.latest, now)
                job = self.calibration_jobs.get("tactile_wrench")
                if job is not None:
                    job["candidate"].offer(self.latest, now)
                    self._finish_calibration("tactile_wrench", job)
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
            if category == "tactile":
                LOG.info("%s: initializing tactile baseline; pads must be unloaded", name)
            worker = self.worker_factory(category, config, self.timeout, self.startup_timeout)
            period = 1 / config.get("fps", 30)
            while not self.stop_event.is_set():
                start = time.monotonic()
                sample = worker.capture(SENSOR_KINDS[category])
                meta = dict(sample["meta"])
                meta.setdefault("received_monotonic_s", time.monotonic())
                self._publish(name, {"meta": meta, "arrays": sample["arrays"]})
                self.stop_event.wait(max(0.0, period - (time.monotonic()-start)))
        except Exception as exc:
            # Never reconnect a tactile pad mid-task: SDK startup resets baseline.
            if not self.stop_event.is_set():
                self._error(name, exc)
        finally:
            if worker is not None:
                worker.close()

    def _scalar_loop(self, key):
        reader = None
        try:
            reader = (MotionStatusReader(self.config["motion_server"], min(self.timeout, 0.5)) if key == "motion_status" else
                      RobotReader(self.config["robot_address"], "motion_state" if key == "state" else "motion_gripper_state",
                                  min(self.timeout, 0.5)))
            if key == "state":
                reader.include_wrench = True
                LOG.info("Zeroing force/torque over %.1f s; keep robot stationary and unloaded", WRENCH_ZERO_SECONDS)
            while not self.stop_event.is_set():
                begin = time.monotonic()
                try:
                    data = reader.status() if key == "motion_status" else reader.read()
                    elapsed = time.monotonic()-begin
                    item, errors = scalar_item(key, data, elapsed)
                    if key == "state":
                        item["raw_wrench"] = deepcopy(item.get("wrench"))
                        item["state_seq"] = data.get("state_seq")
                        item["robot_server_id"] = data.get("server_id")
                        with self.lock:
                            job = self.calibration_jobs.get("robot_wrench")
                            if job is not None:
                                try:
                                    job["candidate"].apply(data, elapsed)
                                except ValueError:
                                    pass  # Candidate status distinguishes collection from failure.
                                self._finish_calibration("robot_wrench", job)
                            try:
                                flange, legacy = self.wrench_baseline.apply(data, elapsed)
                                item["wrench"]["flange"] = flange.tolist()
                                item["wrench"]["calibration_id"] = self.wrench_baseline.calibration_id
                                item["data"]["force"] = legacy.tolist()
                            except ValueError as exc:
                                item["data"].pop("force", None)
                                item["field_errors"]["force"] = str(exc)
                                item["wrench"] = {"error": str(exc)}
                            # Publication and baseline commit must be atomic to readers.
                            errors = item["errors"] + list(item["field_errors"].values())
                            self._publish(key, item, errors)
                        self.stop_event.wait(max(0.0, self.period - (time.monotonic()-begin)))
                        continue
                    self._publish(key, item, errors)
                except Exception as exc:
                    if key == "state":
                        with self.lock:
                            self.wrench_baseline.interrupt()
                            if "robot_wrench" in self.calibration_jobs:
                                self.calibration_jobs["robot_wrench"]["candidate"].interrupt()
                    self._error(key, exc)
                self.stop_event.wait(max(0.0, self.period - (time.monotonic()-begin)))
        except Exception as exc:
            self._error(key, exc)
        finally:
            if isinstance(reader, RobotReader):
                reader.close()

    def _copy_cache(self):
        with self.lock:
            # Published arrays are immutable snapshots; replacement never mutates
            # arrays already held by a request. Encoding runs outside this lock.
            return dict(self.latest), {k: dict(v) for k, v in self.health.items()}

    @staticmethod
    def _age(item, now):
        baseline = item.get("sample_age_s_at_receive")
        baseline = baseline if valid_number(baseline) and baseline >= 0 else 0.0
        return baseline + max(0.0, now - item["received_monotonic_s"])

    def status(self):
        latest, health = self._copy_cache()
        now = time.monotonic()
        for name, status in health.items():
            if name in latest:
                item = latest[name].get("meta", latest[name])
                status["age_s"] = self._age(item, now)
                status["fresh"] = not status["error"] and status["age_s"] <= self.max_age
                status["frame_number"] = item.get("frame_number")
            else:
                status["fresh"] = False
        calibration = self.calibration_status()
        tactile_ready = (not any(n in self.devices for n in PAD_NAMES) or
                         calibration["tactile_wrench"]["ready"])
        return {"protocol": 1, "server_id": self.server_id,
                "ready": all(s["fresh"] for s in health.values()) and tactile_ready,
                "sensors_ready": all(health[n]["fresh"] for n in self.devices),
                "sources": health,
                "calibration": calibration,
                "tactile_geometry": deepcopy(self.geometry),
                "default_observations": self.config["default_observations"]}

    def calibration_status(self):
        with self.lock:
            return {"robot_wrench": self.wrench_baseline.status(),
                    "tactile_wrench": self.tactile_baseline.status(),
                    "pending": list(self.calibration_jobs)}

    def _finish_calibration(self, target, job):
        candidate = job["candidate"]
        if candidate.bias is None and not candidate.fault:
            return
        if not candidate.fault:
            if target == "robot_wrench":
                self.wrench_baseline = candidate
            else:
                self.tactile_baseline = candidate
        job["result"] = candidate.status()
        self.calibration_jobs.pop(target, None)
        job["done"].set()

    def calibrate(self, request):
        if not isinstance(request, dict) or set(request) != {"target"}:
            raise ValueError("Calibration requires only target: robot_wrench or tactile_wrench")
        target = request["target"]
        if target == "tactile":
            target = "tactile_wrench"
        if target not in ("robot_wrench", "tactile_wrench"):
            raise ValueError("target must be robot_wrench or tactile_wrench")
        with self.lock:
            if target in self.calibration_jobs:
                raise ValueError(f"{target} calibration is already running")
            if target == "tactile_wrench" and not all(n in self.devices for n in PAD_NAMES):
                raise ValueError("Both tactile_left and tactile_right must be configured")
            job = {"candidate": WrenchBaseline() if target == "robot_wrench" else TactileBaseline(),
                   "done": threading.Event()}
            self.calibration_jobs[target] = job
        # Leave acquisition threads running. The data window is exactly 1 s;
        # this extra timeout allows for waiting for the first usable samples.
        if not job["done"].wait(4.0):
            with self.lock:
                if not job["done"].is_set():
                    self.calibration_jobs.pop(target, None)
                    raise ValueError("Calibration timed out waiting for fresh samples; previous baseline retained")
        if job["result"]["error"]:
            raise ValueError(job["result"]["error"] + "; previous baseline retained")
        return {"target": target, **job["result"]}

    def _fresh(self, name, latest, health, now, max_age):
        if name not in latest:
            raise ValueError(health.get(name, {}).get("error") or f"{name} is still initializing")
        item = latest[name]
        # Robot force tare may be pending while its pose is already usable.
        error = ("; ".join(item.get("errors", [])) if name == "state" else
                 health.get(name, {}).get("error"))
        if error:
            raise ValueError(error)
        age = self._age(item.get("meta", item), now)
        if age > max_age:
            raise ValueError(f"{name} sample is stale")
        return item, age

    def _tactile_result(self, latest, health, baseline, now, max_age, reference, world=False):
        pads, times, ids = {}, {}, {}
        ages = []
        for name in PAD_NAMES:
            item, age = self._fresh(name, latest, health, now, max_age)
            pads[name] = pad_wrench(name, item["arrays"]["force"], baseline, self.geometry)
            times[name] = now-age
            ids[name] = item["meta"]["frame_number"]
            ages.append(age)
        gripper, age = self._fresh("gripper", latest, health, now, max_age)
        times["gripper"] = now-age
        ages.append(age)
        R_wg = None
        if world:
            state, age = self._fresh("state", latest, health, now, max_age)
            context = state.get("raw_wrench") or state.get("wrench")
            if not isinstance(context, dict) or context.get("error"):
                raise ValueError("World tactile wrench needs fresh robot TCP pose/tool context")
            R_wg = frame_context(context, self.geometry)[2]
            times["robot_pose"] = now-age
            ages.append(age)
        skew = max(times.values())-min(times.values())
        result = combined_wrench(pads, gripper["data"], self.geometry, reference, R_wg)
        result.update(frame_numbers=ids, sample_skew_s=skew,
                      sample_times_monotonic_s=times, timestamp_basis="host_receipt_with_reported_age")
        return result, max(ages)

    def comparison(self, request):
        """Read-only, small JSON sample for the manual torque experiment."""
        if not isinstance(request, dict) or set(request)-{"reference", "max_age_s"}:
            raise ValueError("Comparison accepts reference and max_age_s only")
        reference = request.get("reference", "connector")
        if reference not in TIP_OFFSETS:
            raise ValueError("reference must be fingertip or connector")
        max_age = positive(request.get("max_age_s", self.max_age), "max_age_s")
        with self.lock:
            latest, health = self._copy_cache()
            baseline = self.tactile_baseline
        now = time.monotonic()
        result = {"server_id": self.server_id, "recorded_at_unix_s": time.time(),
                  "selected_monotonic_s": now, "reference": reference,
                  "geometry": deepcopy(self.geometry), "errors": {}}
        # Raw pad samples remain recordable even before software tare succeeds.
        result["pads"] = {}
        for name in PAD_NAMES:
            try:
                item, age = self._fresh(name, latest, health, now, max_age)
                raw = np.asarray(item["arrays"]["force"], dtype=float)
                if raw.shape != (6,) or not np.isfinite(raw).all():
                    raise ValueError("Invalid raw tactile wrench")
                result["pads"][name] = {"raw_native": raw.tolist(), "meta": dict(item["meta"]), "age_s": age}
                result["pads"][name]["calibrated"] = pad_wrench(name, raw, baseline, self.geometry)
            except Exception as exc:
                result["errors"][name] = str(exc)
        try:
            result["tactile"], _ = self._tactile_result(latest, health, baseline, now, max_age, reference)
        except Exception as exc:
            result["errors"]["tactile"] = str(exc)
        try:
            state, age = self._fresh("state", latest, health, now, max_age)
            context = state.get("raw_wrench") or state.get("wrench")
            if not isinstance(context, dict) or context.get("error"):
                raise ValueError("No fresh explicit robot wrench")
            wa = context.get("age_s")
            if not valid_number(wa) or wa < 0:
                raise ValueError("Robot wrench has no sample timestamp")
            age = max(age, wa+max(0., now-state["received_monotonic_s"]))
            if age > max_age:
                raise ValueError("Robot wrench is stale")
            result["robot"] = {"raw_context": context, "raw_tip": robot_at_tip(context, self.geometry, reference),
                               "age_s": age, "sample_time_monotonic_s": now-age,
                               "state_seq": state.get("state_seq"), "server_id": state.get("robot_server_id")}
            compensated = state.get("wrench", {})
            if compensated and not compensated.get("error"):
                result["robot"]["calibrated_tip"] = robot_at_tip(compensated, self.geometry, reference)
                result["robot"]["calibration_id"] = compensated.get("calibration_id")
        except Exception as exc:
            result["errors"]["robot"] = str(exc)
        if "tactile" in result and "robot" in result:
            times = [*result["tactile"]["sample_times_monotonic_s"].values(),
                     result["robot"]["sample_time_monotonic_s"]]
            result["sample_skew_s"] = max(times)-min(times)
            result["synchronized"] = result["sample_skew_s"] <= self.geometry["max_sample_skew_s"]
        else:
            result["synchronized"] = False
        return result

    def snapshot(self, request):
        if not isinstance(request, dict) or set(request)-{"observations", "max_age_s", "wrench", "tactile_reference"}:
            raise ValueError("Request accepts observations, max_age_s, wrench and tactile_reference")
        reference = request.get("tactile_reference", "fingertip")
        if reference not in TIP_OFFSETS:
            raise ValueError("tactile_reference must be fingertip or connector")
        wrench_mode = request.get("wrench", "raw")
        if wrench_mode not in ("raw", "wrist", "tcp"):
            raise ValueError("wrench must be raw, wrist or tcp")
        requested = expand_observations(request.get("observations", self.config["default_observations"]),
                                        self.config)
        max_age = positive(request.get("max_age_s", self.max_age), "max_age_s")
        with self.lock:
            latest, health = self._copy_cache()
            baseline = self.tactile_baseline
        selected = time.monotonic()
        # Ages are transport-only: the client includes transfer/save time in its
        # freshness check. No diagnostic manifest is generated or saved.
        result = {"protocol": 2, "observations": {}, "errors": {}, "ages": {}}
        files = {}
        for key in requested:
            if key in TIP_KEYS:
                try:
                    value, age = self._tactile_result(latest, health, baseline, selected, max_age, reference,
                                                      world=key == "tactile_tip_world.wrench")
                    result["observations"][key], result["ages"][key] = value, age
                except Exception as exc:
                    result["errors"][key] = str(exc)
                continue
            source = "state" if key in STATE_FIELDS else key.split(".")[0]
            if source not in latest:
                result["errors"][key] = health[source]["error"] or "Source is still initializing"
                continue
            sample = latest[source]
            try:
                field_errors = dict(sample.get("field_errors", {}))
                data = sample.get("data")
                age = self._age(sample.get("meta", sample), selected)
                if key in ("force", "state") and wrench_mode != "raw":
                    context = sample.get("wrench")
                    if not isinstance(context, dict) or context.get("error"):
                        raise ValueError(context.get("error") if isinstance(context, dict) else
                                         "Missing explicit flange wrench; restart the robot server")
                    wrench_age = context.get("age_s")
                    if not valid_number(wrench_age) or wrench_age < 0:
                        raise ValueError("No valid wrench sample timestamp")
                    age = max(age, wrench_age + max(0.0, selected - sample["received_monotonic_s"]))
                    data = dict(data, force=context)
                    field_errors.pop("force", None)
                if source in SCALAR_KEYS:
                    errors = list(sample["errors"])
                    if key in STATE_FIELDS:
                        if key in field_errors:
                            errors.append(field_errors[key])
                    else:
                        errors.extend(field_errors.values())
                    if errors:
                        result["errors"][key] = "; ".join(errors)
                        continue
                elif health[source]["error"]:
                    result["errors"][key] = health[source]["error"]
                    continue
                if age > max_age:
                    result["errors"][key] = "Latest sample is stale"
                    continue
                if source in SCALAR_KEYS:
                    value = data[key] if key in STATE_FIELDS else data
                else:
                    kind = key.split(".")[1]
                    if kind == "wrench":
                        value = pad_wrench(source, sample["arrays"]["force"], baseline, self.geometry)
                        value.update(frame_number=sample["meta"]["frame_number"],
                                     sample_time_monotonic_s=selected-age)
                    else:
                        array = sample["arrays"]["deformation" if kind == "deformation_raw" else kind]
                        if kind == "deformation":
                            array = baseline.subtract(source, "deformation", array)
                        value, payload = encode_array(key, kind, array)
                        if payload is not None:
                            files[value] = payload
                result["observations"][key] = value
                result["ages"][key] = age
            except Exception as exc:
                result["errors"][key] = f"{type(exc).__name__}: {exc}"
        return result, files

    def close(self):
        self.stop_event.set()
        deadline = time.monotonic() + self.startup_timeout + self.timeout + 5
        for thread in self.threads:
            thread.join(max(0.0, deadline-time.monotonic()))


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
                if self.path not in ("/observation", "/calibrate", "/wrench-comparison"):
                    self.respond(404, {"error": "Unknown endpoint"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("Request must have a Content-Length between 1 and 16384")
                request = json.loads(self.rfile.read(length))
                if self.path == "/calibrate":
                    self.respond(200, service.calibrate(request))
                    return
                if self.path == "/wrench-comparison":
                    self.respond(200, service.comparison(request))
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
    parser.add_argument("--port", type=int, default=18081)
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
    config = load_config()
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
