#!/usr/bin/env python3
"""Run the UNMODIFIED stack/observation_server.py against the simulator.

Only device drivers are substituted (no file of the real stack is edited):
  RealSenseSource / OrbbecSource -> SimCameraSource  (renders via the sim robot server)
  TactileSource                  -> SimTactileSource (DM-Tac-like arrays from pad contacts)
and CONFIG is pointed at the simulated robot endpoint. Everything else - request handling,
freshness checks, wrench zeroing, tactile tare, combined tip wrench, ZIP bundles - is the
real server's code. Patches sit at module top level so the spawn()ed sensor workers, which
re-import this file as __mp_main__, get them too.

Tactile model (simulation-only approximation of the DM-Tac SDK outputs):
  force   = sum of contact forces on the pad in SDK axes [N]; torque_native = moment about
            the SDK fixed corner in units of 0.01 N.m (the SDK's unverified '10*e-2 N*m').
  deformation (H x W x 3: dx, dy, depth) = Gaussian imprints at contact points, depth from
            normal force, dx/dy from shear; image = shaded rendering of that field.
"""
from __future__ import annotations

import builtins
import os
from pathlib import Path
import sys
import time

import msgpack
import numpy as np
import zmq

HERE = Path(__file__).resolve().parent
STACK = HERE / "stack"
SIM_ADDRESS = os.environ.get("SIM_ROBOT_ADDRESS", "tcp://127.0.0.1:5655")          # robot RPCs
SENSOR_ADDRESS = os.environ.get("SIM_SENSOR_ADDRESS", "tcp://127.0.0.1:5656")     # sim_* renders
LOCK_DIR = HERE / "runs" / ".locks"
CAMERA_FPS = float(os.environ.get("SIM_CAMERA_FPS", "5"))
TACTILE_SHAPE = (180, 240)            # rows = SDK y (18 mm), cols = SDK x (24 mm), 0.1 mm/px
PAD_SIZE_M = (0.024, 0.018)

sys.path.insert(0, str(STACK))
import observation_server as obs  # noqa: E402


def sim_rpc(sock_holder, method, params=None, timeout=2.0):
    sock = sock_holder.get("sock")
    if sock is None:
        sock = zmq.Context.instance().socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        sock.setsockopt(zmq.SNDTIMEO, int(timeout * 1000))
        sock.connect(SENSOR_ADDRESS)
        sock_holder["sock"] = sock
    try:
        sock.send(msgpack.packb({"method": method, "params": params or {}, "token": os.environ.get("SIM_RPC_TOKEN", "")},
                                use_bin_type=True))
        r = msgpack.unpackb(sock.recv(), raw=False)
    except zmq.ZMQError:
        sock.close()
        sock_holder["sock"] = None
        raise
    if not r.get("success"):
        raise RuntimeError(r.get("error"))
    return r["result"]


class SimCameraSource:
    def __init__(self, config, timeout):
        self.config, self.name, self.rpc = config, config["sim_name"], {}
        self.frame = 0
        for _ in range(config.get("warmup_frames", 1)):
            self.capture(["rgb"])

    def capture(self, kinds):
        r = sim_rpc(self.rpc, "sim_camera", {"name": self.name, "width": self.config["width"],
                                             "height": self.config["height"]}, timeout=5.0)
        h, w, _ = r["shape"]
        self.frame += 1
        arrays = {}
        if "rgb" in kinds:
            arrays["rgb"] = np.frombuffer(r["rgb"], np.uint8).reshape(h, w, 3).copy()
        if "depth" in kinds:
            arrays["depth"] = np.frombuffer(r["depth"], np.float32).reshape(h, w).copy()
        return {"received_at_unix_s": time.time(), "frame_number": self.frame}, arrays

    def close(self):
        pass


class SimTactileSource:
    def __init__(self, config, timeout):
        self.name, self.rpc = config["sim_name"], {}
        self.rng = np.random.default_rng(abs(hash(self.name)) % 2**32)
        ys, xs = np.mgrid[0:TACTILE_SHAPE[0], 0:TACTILE_SHAPE[1]]
        self.gx = (xs + 0.5) * PAD_SIZE_M[0] / TACTILE_SHAPE[1]
        self.gy = (ys + 0.5) * PAD_SIZE_M[1] / TACTILE_SHAPE[0]
        self.last = -1
        for _ in range(config.get("warmup_frames", 5)):
            self.capture(["image", "deformation", "force"])

    def capture(self, kinds):
        r = sim_rpc(self.rpc, "sim_tactile", timeout=2.0)
        self.last = r["frame"]
        pts = r["pads"][self.name]
        force, torque = np.zeros(3), np.zeros(3)
        dx = np.zeros(TACTILE_SHAPE, np.float32)
        dy = np.zeros(TACTILE_SHAPE, np.float32)
        dz = np.zeros(TACTILE_SHAPE, np.float32)
        sigma = 0.0025
        for c in pts:
            p, f = np.asarray(c["p"]), np.asarray(c["f"])
            force += f
            torque += np.cross(p, f)
            g = np.exp(-((self.gx - p[0]) ** 2 + (self.gy - p[1]) ** 2) / (2 * sigma ** 2)).astype(np.float32)
            dz += g * max(0.0, -f[2]) * 0.08         # mm of indentation per N of normal load
            dx += g * f[0] * 0.04
            dy += g * f[1] * 0.04
        noise = self.rng.normal(0, 0.02, 6)
        values = {}
        if "force" in kinds:
            values["force"] = np.r_[force + noise[:3], torque / 0.01 + noise[3:] * 0.5].astype(np.float32)
        if "deformation" in kinds:
            values["deformation"] = np.dstack((dx, dy, dz)).astype(np.float32)
        if "image" in kinds:
            gy_, gx_ = np.gradient(dz)
            shade = np.clip(0.55 + 6.0 * (gx_ + gy_), 0, 1)
            base = np.array([70, 110, 160], np.float32)
            img = base * (0.6 + 0.4 * shade[..., None]) + np.clip(dz, 0, 1.5)[..., None] * np.array([60, 40, -30])
            # DM-Tac SDK images are BGR; observation_server encodes "image" without conversion.
            values["image"] = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8)[..., ::-1])
        return {"received_at_unix_s": time.time(), "frame_number": int(self.last)}, values

    def close(self):
        pass


def _open(path, *args, **kwargs):
    """Keep the server's per-device advisory lock files inside this folder."""
    path = str(path)
    if path.startswith("/tmp/agentic_observation_"):
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        path = str(LOCK_DIR / Path(path).name)
    return builtins.open(path, *args, **kwargs)


obs.RealSenseSource = SimCameraSource
obs.OrbbecSource = SimCameraSource
obs.TactileSource = SimTactileSource
obs.open = _open
obs.CONFIG["robot_address"] = SIM_ADDRESS
for cam in ("left_wrist", "right_wrist", "head"):
    obs.CONFIG["cameras"][cam].update(serial=f"SIM-{cam}", sim_name=cam, fps=CAMERA_FPS, warmup_frames=1)
for pad in ("tactile_left", "tactile_right"):
    obs.CONFIG["tactile"][pad].update(serial=f"SIM-{pad}", sim_name=pad, fps=30, warmup_frames=3)

# Ablations (simulation only). SIM_ABLATION, set by episode_ctl.py for the servers only, withholds
# one feedback source from the agent's replies; the devices keep running and compliant execution keeps using the wrench.
# Unset (all other runs): nothing below is active.
ABLATION = os.environ.get("SIM_ABLATION", "")
UNAVAILABLE = "Not available in this configuration"
HIDDEN = {"no_force": lambda k: k in ("force", "state"),
          "no_force_tactile": lambda k: k in ("force", "state") or k.startswith("tactile"),
          "no_tactile": lambda k: k.startswith("tactile"),
          "no_side_camera": lambda k: k.startswith("right_wrist.")}.get(ABLATION)
if HIDDEN is not None:
    _snapshot = obs.ObservationService.snapshot

    def snapshot(self, request):
        result, files = _snapshot(self, request)
        for key in [k for k in list(result["observations"]) + list(result["errors"]) if HIDDEN(k)]:
            value = result["observations"].pop(key, None)
            result["ages"].pop(key, None)
            if isinstance(value, str):
                files.pop(value, None)
            result["errors"][key] = (UNAVAILABLE + " (force is withheld; request eef_pose and joint_angles)"
                                     if key == "state" else UNAVAILABLE)
        return result, files

    obs.ObservationService.snapshot = snapshot


if __name__ == "__main__":
    cfg = sim_rpc({}, "sim_config", timeout=10.0)
    obs.CONFIG["tactile_geometry"].update(cfg["tactile_geometry"])
    os.chdir(STACK)
    raise SystemExit(obs.main(sys.argv[1:]))
