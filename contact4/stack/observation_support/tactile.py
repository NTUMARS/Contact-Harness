"""DM-Tac software tare and wrench geometry. All lengths are metres.

Manufacturer references:
  hardware manual v2, section 2.3: fixed corner origin, x along pad length,
  y across pad width, z outward; NOT a moving contact-centroid origin.
  https://www.dmrobot.com/en/devices/dm-tac-w2.html : W2M sensing area 24 x 18 mm.
  SDK manual v2, section 1.2.2.8: force N; torque printed as '10*e-2 N*m'.
The torque unit is therefore not unambiguous. Native moments must not be added to N.m
lever-arm moments until an explicit scale is configured.
"""
from __future__ import annotations

import time
import uuid

import numpy as np


PAD_NAMES = ("tactile_left", "tactile_right")
# SIMULATION COPY ONLY: the tactile reference sits at the simulated gripper's fingertip, i.e. the
# physical end of the Panda finger collision mesh, 12.85 mm beyond the pad centre (measured on robosuite's
# panda_gripper.xml: pad centre 93.40 mm, finger end 106.25 mm from the flange).
# Both reference names map to the fingertip in simulation.
TIP_OFFSETS = {"fingertip": 0.01285, "connector": 0.01285}
# Columns are SDK x/y/z expressed in the gripper axes. Both x axes point
# forward; opposing pad normals require proper rotations, not reflections.
PAD_ROTATIONS = {
    "tactile_left": np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]]),
    "tactile_right": np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]),
}
WORLD_FROM_BASE = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
DEFAULT_GEOMETRY = {
    "gripper_in_flange": [0., 0., 0.230, 0., 0., np.pi / 6],  # m, radians
    # gripper feedback 0..1000 maps to a 0..95 mm opening; each side lies at +/- gap/2.
    "opening_m": 0.095,
    "pad_center_in_sensor_m": [0.012, 0.009, 0.],
    # Comparison-recording synchronization label only; not an observation gate.
    "max_sample_skew_s": 0.10,
    "torque_nm_per_native": {"tactile_left": None, "tactile_right": None},
}


def vec(value, size):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"Expected {size} finite components")
    return a


def rotation(rpy):
    x, y, z = vec(rpy, 3)
    sx, sy, sz = np.sin([x, y, z])
    cx, cy, cz = np.cos([x, y, z])
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]])


def validate_geometry(geometry):
    vec(geometry["gripper_in_flange"], 6)
    vec(geometry["pad_center_in_sensor_m"], 3)
    for key in ("opening_m", "max_sample_skew_s"):
        if not np.isfinite(geometry[key]) or geometry[key] <= 0:
            raise ValueError(f"Invalid tactile geometry {key}")
    for name in PAD_NAMES:
        scale = geometry["torque_nm_per_native"][name]
        if scale is not None and (isinstance(scale, bool) or not np.isfinite(scale) or scale == 0):
            raise ValueError("Torque scales must be nonzero finite N.m/native-unit or null")


class TactileBaseline:
    """One shared one-second window; sums only advancing frames from each pad.

    A completed object is immutable. Recalibration builds another object and
    the service swaps it in only after successful completion.
    """
    def __init__(self, duration=1.0):
        self.duration = duration
        self.started = None
        self.last_ids = {}
        self.counts = {n: 0 for n in PAD_NAMES}
        self.sums = {}
        self.first_times = {}
        self.last_times = {}
        self.bias = None
        self.fault = None
        self.calibration_id = None
        self.completed_at_unix_s = None
        self.gap_raw = None

    def offer(self, latest, now):
        if self.bias is not None or self.fault:
            return
        if self.started is None:
            if not all(n in latest and 0 <= now-latest[n]["meta"]["received_monotonic_s"] <= .15
                       for n in PAD_NAMES):
                return
            self.started = now
            gripper = latest.get("gripper", {})
            if not gripper.get("errors") and isinstance(gripper.get("data"), (int, float)):
                self.gap_raw = float(gripper["data"])
        try:
            gripper = latest.get("gripper", {})
            if (self.gap_raw is not None and not gripper.get("errors") and
                    isinstance(gripper.get("data"), (int, float)) and
                    abs(gripper["data"]-self.gap_raw) > 10):
                raise ValueError("Gripper moved during tactile calibration; keep the grasp fixed")
            for name in PAD_NAMES:
                sample = latest.get(name)
                if sample is None:
                    continue
                meta, arrays = sample["meta"], sample["arrays"]
                fid, captured = meta["frame_number"], meta["received_monotonic_s"]
                if name in self.last_ids and fid < self.last_ids[name]:
                    raise ValueError("Tactile frame counter reset during calibration")
                if (fid == self.last_ids.get(name) or captured < self.started or
                        captured > self.started+self.duration):
                    continue
                force = vec(arrays["force"], 6)
                deformation = np.asarray(arrays["deformation"], dtype=np.float64)
                if deformation.ndim != 3 or deformation.shape[-1] != 3 or not np.isfinite(deformation).all():
                    raise ValueError("Invalid deformation during calibration")
                if name not in self.sums:
                    self.sums[name] = [np.zeros(6), np.zeros_like(deformation)]
                if deformation.shape != self.sums[name][1].shape:
                    raise ValueError("Deformation shape changed during calibration")
                self.sums[name][0] += force
                self.sums[name][1] += deformation
                self.last_ids[name] = fid
                self.counts[name] += 1
                self.first_times.setdefault(name, captured)
                self.last_times[name] = captured
            if now-self.started >= self.duration:
                if any(self.counts[n] < 10 or self.last_times[n]-self.first_times[n] < .75*self.duration
                       for n in PAD_NAMES):
                    raise ValueError("Insufficient distinct tactile frames in the one-second calibration window")
                self.bias = {n: {"force": self.sums[n][0]/self.counts[n],
                                 "deformation": (self.sums[n][1]/self.counts[n]).astype(np.float32)}
                             for n in PAD_NAMES}
                self.sums.clear()
                self.calibration_id = str(uuid.uuid4())
                self.completed_at_unix_s = time.time()
        except (ValueError, KeyError, TypeError) as exc:
            self.fault = str(exc)
            self.sums.clear()

    def subtract(self, name, kind, value):
        if self.bias is None:
            raise ValueError(self.fault or "Tactile calibration in progress; keep pads/grasp still")
        a = np.asarray(value)
        b = self.bias[name][kind]
        if a.shape != b.shape:
            raise ValueError("Tactile shape changed; recalibrate tactile")
        # Never modify the raw frame: deformation_raw and SDK inference reuse it.
        return a-b

    def status(self):
        return {"ready": self.bias is not None, "error": self.fault, "duration_s": self.duration,
                "sample_counts": dict(self.counts), "calibration_id": self.calibration_id,
                "completed_at_unix_s": self.completed_at_unix_s}


def pad_wrench(name, raw, baseline, geometry):
    delta = baseline.subtract(name, "force", vec(raw, 6))
    scale = geometry["torque_nm_per_native"][name]
    return {"force_n": delta[:3].tolist(), "torque_native": delta[3:].tolist(),
            "torque_nm": None if scale is None else (delta[3:]*scale).tolist(),
            "torque_nm_per_native": scale, "torque_scale_verified": scale is not None,
            "native_torque_unit": "SDK manual: 10*e-2 N*m (unverified notation)",
            "frame": name, "origin": "sdk_fixed_corner", "calibration_id": baseline.calibration_id}


def combined_wrench(pads, gap_raw, geometry, reference, world_from_gripper=None):
    if reference not in TIP_OFFSETS:
        raise ValueError("tactile_reference must be fingertip or connector")
    if isinstance(gap_raw, bool) or not np.isfinite(gap_raw) or not 0 <= gap_raw <= 1000:
        raise ValueError("AG95 actual position feedback must be in [0,1000]")
    gap = gap_raw/1000*geometry["opening_m"]
    target = np.array([0., 0., TIP_OFFSETS[reference]])
    R_out = np.eye(3) if world_from_gripper is None else world_from_gripper
    forces, native, moments = {}, {}, {}
    for name, sign in zip(PAD_NAMES, (-1, 1)):
        R_gs = PAD_ROTATIONS[name]
        center = np.array([sign*gap/2, 0., 0.])
        origin = center - R_gs @ vec(geometry["pad_center_in_sensor_m"], 3)
        force_g = R_gs @ vec(pads[name]["force_n"], 3)
        forces[name] = R_out @ force_g
        native[name] = R_out @ R_gs @ vec(pads[name]["torque_native"], 3)
        moments[name] = R_out @ np.cross(origin-target, force_g)
    force = sum(forces.values())
    lever = sum(moments.values())
    scales = geometry["torque_nm_per_native"]
    resolved = all(scales[n] is not None for n in PAD_NAMES)
    torque = lever + sum(native[n]*scales[n] for n in PAD_NAMES) if resolved else None
    gripper = vec(geometry["gripper_in_flange"], 6)
    return {"force_n": force.tolist(), "torque_nm": None if torque is None else torque.tolist(),
            "torque_scale_verified": resolved,
            "torque_native_by_pad": {n: v.tolist() for n, v in native.items()},
            "lever_arm_torque_nm": lever.tolist(),
            "lever_arm_torque_nm_by_pad": {n: v.tolist() for n, v in moments.items()},
            "torque_nm_per_native": dict(scales),
            "frame": "gripper" if world_from_gripper is None else "world",
            "reference": reference, "pad_gap_m": gap,
            "origin_in_flange_m": (gripper[:3]+rotation(gripper[3:])@target).tolist(),
            "calibration_id": pads[PAD_NAMES[0]]["calibration_id"],
            "status": "ready" if resolved else "force_ready_torque_scale_unverified"}


def frame_context(context, geometry):
    """Recover flange axes using the ACTUAL configured robot TCP, then gripper."""
    tcp, tool = vec(context["tcp_pose_base"], 6), vec(context["tool_end"], 6)
    R_bf = rotation(tcp[3:]) @ rotation(tool[3:]).T
    R_fg = rotation(vec(geometry["gripper_in_flange"], 6)[3:])
    return R_bf, R_fg, WORLD_FROM_BASE @ R_bf @ R_fg


def robot_at_tip(context, geometry, reference):
    R_bf, R_fg, _ = frame_context(context, geometry)
    gripper = vec(geometry["gripper_in_flange"], 6)
    tip = gripper[:3]+R_fg@np.array([0., 0., TIP_OFFSETS[reference]])
    wrench = vec(context["flange"], 6)
    force, torque = wrench[:3], wrench[3:]-np.cross(tip, wrench[:3])
    return {"gripper": np.r_[R_fg.T@force, R_fg.T@torque].tolist(),
            "world": np.r_[WORLD_FROM_BASE@R_bf@force, WORLD_FROM_BASE@R_bf@torque].tolist(),
            "origin_in_flange_m": tip.tolist(), "reference": reference}
