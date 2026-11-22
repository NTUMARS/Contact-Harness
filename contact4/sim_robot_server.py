#!/usr/bin/env python3
"""Simulated robot server: the hardware seam of the harness, backed by MuJoCo.

Speaks the msgpack REQ/REP protocol of the real robot server,
for exactly the calls that motion_server.py / observation_server.py use:
  get_robot_info, get_tool_info, get_base_frame, is_in_realtime_loop, stop_realtime_loop,
  motion_state / motion_start / motion_target / motion_home / motion_stop / motion_gripper_state.
Session semantics (owner lease, server_id, target_seq, fault, 50 N hard reject, 5 cm / 30 deg
jump guard, latest-only target consumed after the previous interpolation) mirror
the real robot server. The native 1 kHz Cartesian impedance loop is replaced
by a torque-level impedance controller with the stiffness [400 400 400 50 50 50].

The flange wrench is the sum of contact forces the environment exerts on the tool subtree
(gripper + anything it touches through the fingers), about the flange origin, flange axes,
i.e. what the SDK's getEndTorque(flange) reports after setLoad compensation.
"""
from __future__ import annotations

import argparse
from collections import deque
import logging
import math
import os
from pathlib import Path
import threading
import time
import uuid

import msgpack
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import zmq

HERE = Path(__file__).resolve().parent
LOG = logging.getLogger("sim_robot_server")
SIM_RPC_TOKEN = os.environ.get("SIM_RPC_TOKEN", "")   # per-episode secret for sim_* RPCs (empty: no check)
BEHIND_YIELD_S = 0.0003   # pause per tick while the physics loop is behind real time
# "w/o soft force limit": without the force-limiting layer a robot executes every motion to
# completion, so in that ablation run (SIM_ABLATION=no_force_limit, set by episode_ctl.py) the simulator does not refuse
# targets at the 50 N hard limit either. Every other run: unchanged.
NO_FORCE_LIMIT = os.environ.get("SIM_ABLATION", "") in ("no_force_limit", "no_compliance_limit")
# "w/o compliant execution and soft force limit" (compliance and the soft force limit ablated
# together; stiffness 3000 N/m, 300 Nm/rad, the maximum Cartesian stiffness of a Franka's impedance controller,
# i.e. close to position control). Only in that ablation run (SIM_ABLATION=no_compliance_limit); damping keeps the same
# 0.9-of-critical rule. Every other run: 400 N/m, 50 Nm/rad.
STIFF_EXECUTION = os.environ.get("SIM_ABLATION", "") == "no_compliance_limit"

ROBOT_TYPE = "AR5-5_07L-W4C4A2"
HOME_DEG = [152, 48, -100, 97, 30, 24.5, 31]          # home joint angles
TOOL_END = [0.0, 0.0, 0.250, 0.0, 0.0, math.radians(30)]  # tool end pose (m, rad)
CART_IMPEDANCE = [400, 400, 400, 50, 50, 50]          # Cartesian impedance
TOOL_MASS, TOOL_COG = 1.55, [0.0, 0.0, 0.075]
CONTROL_PERIOD = 0.001
TARGET_MAX_T, TARGET_MAX_R, TARGET_MAX_AGE = 0.05, math.pi / 6, 0.2
HARD_FORCE_N = 50.0
# Alternative scheme: SIM_HARD_LIMIT=staged judges the hard limit on held force instead of the instantaneous
# reading. Contact force >= HARD_FORCE_N for HL_HOLD_S -> the reference backs off to the midpoint between the measured pose and
# the commanded target (new targets dropped for HL_HOLD_S); if the force stays >= HARD_FORCE_N for that whole window -> the
# limit fires (targets refused as before; the client resets to the measured pose). Otherwise normal tracking resumes, the
# first Cartesian target after the window ramping back at <= HL_RESYNC_V. Default (unset): the instantaneous rule.
HL_STAGED = os.environ.get("SIM_HARD_LIMIT") == "staged"
HL_HOLD_S, HL_RESYNC_V = 0.2, 0.25
WRENCH_TAU = 0.01                    # s, external-wrench low-pass (~16 Hz), see _update_state
FINGER_STROKE = 0.0475
# observation_support/tactile.py PAD_ROTATIONS: columns are DM-Tac SDK x/y/z in gripper axes
# (x across the opening, left pad at -x; z forward). SDK z is the outward pad normal.
PAD_ROTATIONS = {
    "tactile_left": np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]]),
    "tactile_right": np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]),
}
PAD_CENTER_IN_SENSOR = np.array([0.012, 0.009, 0.0])   # DEFAULT_GEOMETRY, W2M 24 x 18 mm


def pose_of(pos, mat):
    return np.r_[pos, R.from_matrix(mat).as_euler("xyz")]


class Sim:
    """Physics + native-controller emulation. One thread steps physics in wall-clock time."""

    def __init__(self, xml, realtime=True, wrench_noise=0.0, start_deg=None):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.substeps = max(1, round(CONTROL_PERIOD / self.m.opt.timestep))
        self.success = None
        self.realtime = realtime
        self.noise = wrench_noise
        self.rng = np.random.default_rng(0)
        m = self.m
        self.base = m.body("base").id
        self.flange = m.body("flange").id
        self.tcp = m.site("tcp").id
        self.arm_dof = np.array([m.joint(f"j{i}").dofadr[0] for i in range(1, 8)])
        self.arm_qpos = np.array([m.joint(f"j{i}").qposadr[0] for i in range(1, 8)])
        self.grip_act = m.actuator("gripper").id
        self.finger_qpos = m.joint("finger_l").qposadr[0]
        self.tool_bodies = {b for b in range(m.nbody) if self._is_desc(b, self.flange)}
        self.robot_type, self.tool_end = ROBOT_TYPE, list(TOOL_END)
        self.tool_mass, self.tool_cog = TOOL_MASS, list(TOOL_COG)
        self.dt = CONTROL_PERIOD
        self.home = np.deg2rad(HOME_DEG)
        q0 = self.home if start_deg is None else np.deg2rad(start_deg)
        self.d.qpos[self.arm_qpos] = q0
        self.d.qpos[self.finger_qpos] = FINGER_STROKE
        self.d.qpos[m.joint("finger_r").qposadr[0]] = FINGER_STROKE
        self.d.ctrl[self.grip_act] = FINGER_STROKE
        mujoco.mj_forward(m, self.d)
        self.cam_map = {"left_wrist": "left_wrist", "right_wrist": "overview", "head": "head"}
        # gripper axes = tool axes; finger_r sits at tool -x, so it is the "left" pad
        self.pad_geoms = {"tactile_left": m.geom("pad_r").id, "tactile_right": m.geom("pad_l").id}
        self.pad_half_thickness = 0.0015
        self.R_flange_G = R.from_euler("xyz", TOOL_END[3:]).as_matrix()
        self.tactile_geometry = {"gripper_in_flange": [0., 0., 0.230, 0., 0., TOOL_END[5]],
                                 "opening_m": 2 * FINGER_STROKE}
        Rb = self.d.xmat[self.base].reshape(3, 3).copy()
        self.R_wb, self.p_wb = Rb, self.d.xpos[self.base].copy()
        self._init_runtime(q0)

    def _init_runtime(self, q0):
        """Controller/session state shared by all backends (needs m, d, frames, R_wb, p_wb)."""
        self._tool_geom_mask = np.array([self.m.geom_bodyid[g] in self.tool_bodies for g in range(self.m.ngeom)])
        self.lock = threading.RLock()
        # native-controller state
        self.mode = None                  # None (brakes) | cartesian_impedance | joint_* | movej
        self.in_rt = False
        self.stop_confirmed = True
        self.hold_q = q0.copy()
        self.cart_ref = None              # interpolated reference (base frame pose)
        self.target_buf = deque(maxlen=1)
        self.interp_remain = 0
        self.interp_n = 0
        self.interp_start = None
        self.interp_target = None
        self.stale_rejects = 0
        self.jump_rejects = 0
        self.joint_buf = deque()
        self.movej = None
        self.null_q = q0.copy()
        self.grip_cmd = 1.0
        self.snapshot = None
        self.snap_time = None
        self.snap_seq = 0
        self.sim_time = 0.0
        self.render_req = deque()
        self.hooks = []                   # callables(sim) run each control tick (task logic)
        self.tactile_frame = 0
        K = np.array([3000, 3000, 3000, 300, 300, 300] if STIFF_EXECUTION else CART_IMPEDANCE, float)
        if STIFF_EXECUTION:
            LOG.info("ablation no_compliance_limit: stiff execution %s (instead of %s)", K.tolist(), CART_IMPEDANCE)
        self.Kt, self.Kr = K[:3], K[3:]
        self.Dt, self.Dr = 2 * 0.9 * np.sqrt(self.Kt * 4.0), 2 * 0.9 * np.sqrt(self.Kr * 0.06)
        self._hl_reset()
        self.hl_log = []                  # staged hard limit events (sim time, event, force)
        # The impedance reference is ONE target shared by agent, shim and simulator. A staged
        # back-off moves it; this counter (reported in state()) lets the shim and the agent follow the moved reference.
        self.hl_backoff_seq = 0
        self._update_state()

    def _hl_reset(self):
        self.hl_state, self.hl_hi_since, self.hl_t0, self.hl_all_high, self.hl_resync = "normal", None, None, True, False

    def _hl_backoff(self):
        """Reference to the midpoint between the measured pose and the commanded target; pending targets dropped."""
        if self.mode == "cartesian_impedance":
            tgt = self.target_buf[-1][0] if self.target_buf else (self.interp_target if self.interp_remain > 0 else self.cart_ref)
            cur = self.tcp_pose_base()
            r0 = R.from_euler("xyz", cur[3:])
            dr = (r0.inv() * R.from_euler("xyz", tgt[3:])).as_rotvec()
            self.cart_ref = np.r_[0.5 * (cur[:3] + tgt[:3]), (r0 * R.from_rotvec(0.5 * dr)).as_euler("xyz")]
            self.interp_remain = 0
            self.target_buf.clear()
        else:
            tgt = self.joint_buf[-1] if self.joint_buf else self.hold_q
            self.hold_q = 0.5 * (self.d.qpos[self.arm_qpos].copy() + tgt)
            self.joint_buf.clear()
        self.hl_backoff_seq += 1

    def _hl_tick(self):
        wf = getattr(self, "_wf_filt", None)
        f = 0.0 if wf is None else float(np.linalg.norm(wf[:3]))
        t, high = self.sim_time, f >= HARD_FORCE_N
        if self.hl_state == "normal":
            if not high:
                self.hl_hi_since = None
                return
            self.hl_hi_since = t if self.hl_hi_since is None else self.hl_hi_since
            if t - self.hl_hi_since >= HL_HOLD_S:
                self._hl_backoff()
                self.hl_state, self.hl_t0, self.hl_all_high = "backoff", t, True
                self.hl_log.append((round(t, 3), "backoff", round(f, 1)))
                LOG.warning("staged hard limit: %.1f N held %.2f s -> reference to the midpoint", f, HL_HOLD_S)
        elif self.hl_state == "backoff":
            self.hl_all_high = self.hl_all_high and high
            self.joint_buf.clear()
            self.target_buf.clear()
            if t - self.hl_t0 >= HL_HOLD_S:
                if self.hl_all_high:
                    self.hl_state = "limit"
                    self.hl_log.append((round(t, 3), "limit", round(f, 1)))
                    LOG.warning("staged hard limit: still >= %.0f N at the midpoint -> hard force limit", HARD_FORCE_N)
                else:
                    self.hl_state, self.hl_hi_since, self.hl_resync = "normal", None, True
                    self.hl_log.append((round(t, 3), "resume", round(f, 1)))
                    LOG.warning("staged hard limit: force below %.0f N at the midpoint -> tracking resumes", HARD_FORCE_N)

    def _is_desc(self, b, root):
        while b > 0:
            if b == root:
                return True
            b = self.m.body_parentid[b]
        return False

    # ---- frames -------------------------------------------------------------------------
    def tcp_world(self):
        return self.d.site_xpos[self.tcp].tolist()

    def tcp_pose_base(self):
        p = self.R_wb.T @ (self.d.site_xpos[self.tcp] - self.p_wb)
        Rm = self.R_wb.T @ self.d.site_xmat[self.tcp].reshape(3, 3)
        return pose_of(p, Rm)

    def flange_wrench(self):
        """Environment-on-tool wrench about the flange origin, flange axes ([F, M])."""
        m, d = self.m, self.d
        F, M = np.zeros(3), np.zeros(3)
        pf = d.xpos[self.flange]
        n = d.ncon
        if n:
            con = d.contact  # struct-of-arrays fields (per-element access is broken in mujoco 2.3.x)
            g1, g2 = con.geom1[:n], con.geom2[:n]
            in1 = self._tool_geom_mask[g1]
            in2 = self._tool_geom_mask[g2]
            idx = np.nonzero(in1 != in2)[0]
            if len(idx):
                f6 = np.zeros(6)
                fc = np.empty((len(idx), 3))
                for k, i in enumerate(idx):
                    mujoco.mj_contactForce(m, d, int(i), f6)
                    fc[k] = f6[:3]
                frames = con.frame[idx].reshape(-1, 3, 3)
                fw = np.einsum("kji,kj->ki", frames, fc)       # frame.T @ f: force on geom2, world
                fw[in1[idx]] *= -1.0                            # flip when the tool is geom1
                F = fw.sum(0)
                M = np.cross(con.pos[idx] - pf, fw).sum(0)
        Rf = d.xmat[self.flange].reshape(3, 3)
        w = np.r_[Rf.T @ F, Rf.T @ M]
        if self.noise:
            w = w + self.rng.normal(0, self.noise, 6) * np.r_[1, 1, 1, .02, .02, .02]
        return w

    def tactile_contacts(self):
        """Per pad: contact points/forces (environment on pad) in DM-Tac SDK axes, origin at the
        SDK fixed corner. Raw physics only; the observation-side source builds SDK-like arrays."""
        m, d = self.m, self.d
        Rf = d.xmat[self.flange].reshape(3, 3)
        out = {}
        f6 = np.zeros(6)
        for name, gid in self.pad_geoms.items():
            R_sw = Rf @ self.R_flange_G @ PAD_ROTATIONS[name]
            surface = d.geom_xpos[gid] + R_sw[:, 2] * self.pad_half_thickness
            corner = surface - R_sw @ PAD_CENTER_IN_SENSOR
            pts = []
            con = d.contact
            for i in range(d.ncon):
                g1, g2 = con.geom1[i], con.geom2[i]
                if gid not in (g1, g2):
                    continue
                other = g2 if g1 == gid else g1
                if m.geom_bodyid[other] in self.tool_bodies:
                    continue
                mujoco.mj_contactForce(m, d, i, f6)
                fw = con.frame[i].reshape(3, 3).T @ f6[:3]      # on geom2
                if g1 == gid:
                    fw = -fw
                pts.append({"p": (R_sw.T @ (con.pos[i] - corner)).tolist(), "f": (R_sw.T @ fw).tolist()})
            out[name] = pts
        for name in PAD_ROTATIONS:          # tools without fingers report unloaded pads
            out.setdefault(name, [])
        self.tactile_frame += 1
        return {"frame": self.tactile_frame, "pads": out}

    def _update_state(self):
        d = self.d
        q = d.qpos[self.arm_qpos].copy()
        dq = d.qvel[self.arm_dof].copy()
        # external wrench as a robot reports it: low-pass filtered (first order, WRENCH_TAU). The raw sum of
        # contact forces at the 2 ms control step carries solver/contact ringing (e.g. a 20 N plug bottoming
        # read 46-108 N for one step) that a joint-torque-based estimate on the real arm never shows.
        raw = self.flange_wrench()
        prev = getattr(self, "_wf_filt", None)
        a = self.dt / (WRENCH_TAU + self.dt)
        wf = raw if prev is None else prev + a * (raw - prev)
        self._wf_filt = wf
        Rf = self.R_wb.T @ d.xmat[self.flange].reshape(3, 3)
        ext = np.r_[Rf @ wf[:3], Rf @ wf[3:]]
        snap = {"joint_pos_real": q.tolist(), "cart_pos_real": self.tcp_pose_base().tolist(),
                "joint_vel_real": dq.tolist(), "ext_wrench": ext.tolist()}
        self.snapshot, self.snap_time, self.flange_w = snap, time.monotonic(), wf
        self.snap_seq += 1

    # ---- control --------------------------------------------------------------------------
    def _arm_dynamics(self):
        m, d = self.m, self.d
        Jp, Jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, Jp, Jr, self.tcp)
        J = np.r_[Jp[:, self.arm_dof], Jr[:, self.arm_dof]]
        return J, d.qfrc_bias[self.arm_dof].copy()

    def _joint_pd(self, qd, kp=600.0, kd=40.0, dqd=None):
        d = self.d
        q, dq = d.qpos[self.arm_qpos], d.qvel[self.arm_dof]
        kp = np.array([kp, kp, kp * .7, kp * .7, kp * .3, kp * .3, kp * .3])
        kd = np.array([kd, kd, kd * .7, kd * .7, kd * .3, kd * .3, kd * .3])
        v = np.zeros(7) if dqd is None else dqd
        return kp * (qd - q) + kd * (v - dq) + d.qfrc_bias[self.arm_dof]

    def _target_tick(self):
        if self.interp_remain <= 0 and self.target_buf:
            pos, steps, recv = self.target_buf.popleft()
            if time.monotonic() - recv > TARGET_MAX_AGE:
                self.stale_rejects += 1
            else:
                self.interp_target, self.interp_n, self.interp_remain = pos, steps, steps
                self.interp_start = self.cart_ref.copy()
                ra = R.from_euler("xyz", self.interp_start[3:])
                self._interp_ra = ra
                self._interp_dr = (ra.inv() * R.from_euler("xyz", pos[3:])).as_rotvec()   # slerp axis-angle
        if self.interp_remain > 0:
            k = self.interp_n - self.interp_remain + 1
            r = min(k / self.interp_n, 1.0)
            a, b = self.interp_start, self.interp_target
            rot = self._interp_ra * R.from_rotvec(r * self._interp_dr)
            self.cart_ref = np.r_[a[:3] + r * (b[:3] - a[:3]), rot.as_euler("xyz")]
            self.interp_remain -= 1

    def _cart_impedance(self):
        d = self.d
        J, bias = self._arm_dynamics()
        ref = self.cart_ref
        pd_w = self.R_wb @ ref[:3] + self.p_wb
        Rd_w = self.R_wb @ R.from_euler("xyz", ref[3:]).as_matrix()
        p = d.site_xpos[self.tcp]
        Rm = d.site_xmat[self.tcp].reshape(3, 3)
        ep = pd_w - p
        er = R.from_matrix(Rd_w @ Rm.T).as_rotvec()
        v = J @ d.qvel[self.arm_dof]
        # stiffness is specified in tool axes (iso here); rotate to world for generality
        Kt = Rm @ np.diag(self.Kt) @ Rm.T
        Kr = Rm @ np.diag(self.Kr) @ Rm.T
        F = np.r_[Kt @ ep - self.Dt * v[:3], Kr @ er - self.Dr * v[3:]]
        tau = J.T @ F + bias
        # null-space posture keeps the redundant elbow near its start configuration
        Jp = np.linalg.pinv(J)
        N = np.eye(7) - J.T @ Jp.T
        q, dq = d.qpos[self.arm_qpos], d.qvel[self.arm_dof]
        tau += N @ (8.0 * (self.null_q - q) - 1.5 * dq)
        return tau

    def control_tick(self):
        with self.lock:
            mode = self.mode
            if HL_STAGED and mode in ("cartesian_impedance", "joint_impedance", "joint_position"):
                self._hl_tick()
            if mode == "cartesian_impedance":
                self._target_tick()
                tau = self._cart_impedance()
            elif mode in ("joint_impedance", "joint_position"):
                if self.joint_buf:
                    self.hold_q = self.joint_buf.popleft()
                tau = self._joint_pd(self.hold_q, kp=300 if mode == "joint_impedance" else 1500)
            elif mode == "movej":
                t = self.sim_time - self.movej["t0"]
                T = self.movej["T"]
                s = min(t / T, 1.0)
                sm = 10 * s**3 - 15 * s**4 + 6 * s**5
                qd = self.movej["q0"] + sm * (self.movej["q1"] - self.movej["q0"])
                tau = self._joint_pd(qd, kp=1500, kd=80)
                if t > T + 0.6:
                    self.movej["done"].set()
            else:   # brakes engaged
                tau = self._joint_pd(self.hold_q, kp=3000, kd=120)
            self._apply(tau)
            for _ in range(self.substeps):
                mujoco.mj_step(self.m, self.d)
            self.sim_time += self.dt
            for h in self.hooks:
                h(self)
            self._update_state()

    def _render_one(self, cam, w, h, depth=False):
        """Render in the physics thread (it owns the GL context). Returns rgb or (rgb, depth_m)."""
        r = getattr(self, "_renderer", None)
        if r is None or (r.width, r.height) != (w, h):
            r = self._renderer = mujoco.Renderer(self.m, h, w)
        r.update_scene(self.d, camera=cam)
        rgb = r.render().copy()
        if not depth:
            return rgb
        r.enable_depth_rendering()
        r.update_scene(self.d, camera=cam)
        dep = r.render().copy()
        r.disable_depth_rendering()
        return rgb, dep

    def run(self, stop_event):
        next_t = time.perf_counter()
        while not stop_event.is_set():
            self.control_tick()
            # at most ONE camera render per control tick: physics and the state snapshot keep updating
            # between the renders of a multi-camera observation (all at once could exceed the motion
            # server's 0.1 s stale-feedback limit when the GPU is shared with other jobs)
            if self.render_req:
                cam, w, h, out = self.render_req.popleft()
                try:
                    with self.lock:
                        res = self._render_one(cam, w, h, out.get("depth", False))
                    if isinstance(res, tuple):
                        out["rgb"], out["depth"] = res
                    else:
                        out["rgb"] = res
                except Exception as exc:
                    out["error"] = str(exc)
                out["event"].set()
            if self.realtime:
                next_t += self.dt
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    # behind real time (Insert press fit: ~35 contacts, ~2.7 ms per 2 ms tick): a real pause, not
                    # sleep(0), so the RPC threads can take the lock. With sleep(0) this loop re-took the lock at once
                    # and motion_state requests waited 0.1-0.7 s (motion_server stale_feedback / motion_state timeouts).
                    # Physics, control and task settings are unchanged; only the wall-clock pacing while behind.
                    time.sleep(BEHIND_YIELD_S)
                    if delay < -0.05:     # fell behind: resync instead of bursting
                        next_t = time.perf_counter()

    # ---- native API used by the motion RPCs ----------------------------------------------
    def state(self):
        with self.lock:
            r = dict(self.snapshot)
            r.update(state_seq=self.snap_seq, state_age_s=time.monotonic() - self.snap_time,
                     in_realtime_loop=self.in_rt, stop_confirmed=self.stop_confirmed,
                     callback_mode=("cart_pos" if self.mode == "cartesian_impedance" else
                                    "joint_pos" if self.in_rt else None),
                     stale_target_rejections=self.stale_rejects,
                     jump_target_rejections=self.jump_rejects,
                     joint_queue_size=len(self.joint_buf),
                     # shared reference: the current impedance reference (base-B pose),
                     # the staged-limit state and the back-off counter, so clients follow a moved reference
                     cart_ref=(None if self.cart_ref is None else [float(x) for x in self.cart_ref]),
                     hl_state=getattr(self, "hl_state", "normal"), hl_backoff_seq=self.hl_backoff_seq)
            return r

    def start(self, mode):
        with self.lock:
            self.cart_ref = self.tcp_pose_base()
            self.target_buf.clear()
            self.interp_remain = 0
            self.joint_buf.clear()
            self.hold_q = self.d.qpos[self.arm_qpos].copy()
            self.null_q = self.hold_q.copy()
            self.mode, self.in_rt, self.stop_confirmed = mode, True, False
            self._hl_reset()

    def stop(self):
        with self.lock:
            self.hold_q = self.d.qpos[self.arm_qpos].copy()
            self.mode, self.in_rt, self.stop_confirmed = None, False, True
            self.target_buf.clear()
            self.joint_buf.clear()
            self._hl_reset()

    def set_target(self, pose, dt):
        n = max(1, int(round(dt / self.dt)))
        with self.lock:
            ref = self.cart_ref
            if HL_STAGED and self.hl_state == "backoff":
                return True                                       # dropped during the back-off window
            if HL_STAGED and self.hl_resync:
                self.hl_resync = False
                dist = float(np.linalg.norm(np.asarray(pose[:3], float) - ref[:3]))
                n = max(n, int(math.ceil(dist / HL_RESYNC_V / self.dt)))
                self.target_buf.append((np.asarray(pose, float).copy(), n, time.monotonic()))
                return True
            jt = float(np.linalg.norm(pose[:3] - ref[:3]))
            jr = float((R.from_euler("xyz", ref[3:]).inv() * R.from_euler("xyz", pose[3:])).magnitude())
            if jt > TARGET_MAX_T or jr > TARGET_MAX_R:
                self.target_buf.clear()
                self.jump_rejects += 1
                return False
            self.target_buf.append((np.asarray(pose, float).copy(), n, time.monotonic()))
            return True

    def set_joint(self, q, dt):
        with self.lock:
            self.joint_buf.append(np.asarray(q, float).copy())

    def movej_home(self, timeout=60.0):
        with self.lock:
            q0 = self.d.qpos[self.arm_qpos].copy()
            T = max(1.5, float(np.max(np.abs(self.home - q0))) / 0.5)
            ev = threading.Event()
            self.movej = dict(q0=q0, q1=self.home.copy(), T=T, t0=self.sim_time, done=ev)
            self.mode, self.stop_confirmed = "movej", False
        if not ev.wait(timeout):
            raise RuntimeError("Simulated MoveJ timed out")
        with self.lock:
            self.hold_q = self.home.copy()
            self.mode, self.stop_confirmed = None, True
            return self.d.qpos[self.arm_qpos].copy()

    def _apply(self, tau):
        self.d.ctrl[self.grip_act] = self.grip_cmd * FINGER_STROKE
        self.d.qfrc_applied[self.arm_dof] = tau

    def gripper_opening(self):
        return float(self.d.qpos[self.finger_qpos]) / FINGER_STROKE

    def gripper_feedback(self):
        with self.lock:
            q = self.gripper_opening()
        return {"raw": round(1000.0 * q), "valid": True, "age_s": 0.002,
                "target": round(1000.0 * self.grip_cmd)}

    def render(self, cam, w=640, h=480, timeout=2.0, depth=False):
        out = {"event": threading.Event(), "depth": depth}
        self.render_req.append((cam, w, h, out))
        if not out["event"].wait(timeout):
            raise RuntimeError("render timeout")
        if "error" in out:
            raise RuntimeError(out["error"])
        return (out["rgb"], out["depth"]) if depth else out["rgb"]


def geom_points(m, d, p):
    """Pose oracle for the GenCHiP baseline: points on the geoms
    of the named bodies (box / sphere / cylinder / capsule / ellipsoid: the 8 corners of the geom's local bounding box;
    mesh: its vertices) and the frames of those bodies, all in the Franka base frame (robot0_link0). Read only.
    p: bodies (names), prefix (body-name prefix, e.g. "contact" for the Wipe dirt markers), geoms (names),
    visible_only (skip geoms with alpha 0)."""
    L = m.body("robot0_link0").id
    Rl, pl = d.xmat[L].reshape(3, 3), d.xpos[L]
    names = set(p.get("bodies") or [])
    pre = p.get("prefix")
    gnames = set(p.get("geoms") or [])
    vis = bool(p.get("visible_only", True))
    sign = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    pts, frames, used = [], {}, []
    for g in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not (bn in names or gn in gnames or (pre and bn.startswith(pre))):
            continue
        if vis and m.geom_rgba[g][3] <= 0.0:
            continue
        t, sz = int(m.geom_type[g]), m.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            a = m.mesh_vertadr[m.geom_dataid[g]]
            loc = m.mesh_vert[a:a + m.mesh_vertnum[m.geom_dataid[g]]].astype(float)
        elif t == mujoco.mjtGeom.mjGEOM_BOX or t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            loc = sign * sz[:3]
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            loc = sign * sz[0]
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            loc = sign * np.array([sz[0], sz[0], sz[1]])
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            loc = sign * np.array([sz[0], sz[0], sz[1] + sz[0]])
        else:
            continue
        w = loc @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        pts.append((w - pl) @ Rl)
        used.append(gn)
        b = m.geom_bodyid[g]
        if bn not in frames:
            frames[bn] = {"pos": ((d.xpos[b] - pl) @ Rl).tolist(), "rot": (Rl.T @ d.xmat[b].reshape(3, 3)).tolist()}
    return {"points": np.concatenate(pts).tolist() if pts else [], "frames": frames, "geoms": used}


class SimZmqServer:
    """Protocol layer, mirroring the motion_* session logic of the real robot server."""

    def __init__(self, sim: Sim, address):
        self.sim = sim
        self.address = address
        self.server_id = str(uuid.uuid4())
        self.owner = None
        self.deadline = 0.0
        self.fault = None
        self.mode = None
        self.seq = 0
        self.wrench_sample = None

    def _stop(self):
        self.sim.stop()
        self.owner, self.deadline, self.fault, self.mode = None, 0.0, None, None

    def _lease(self):
        if self.owner and not self.fault and time.monotonic() > self.deadline:
            LOG.warning("Motion-client lease expired: stopping")
            self._stop()

    def _snapshot(self):
        r = self.sim.state()
        r.update(protocol=1, server_id=self.server_id, owner=self.owner, mode=self.mode,
                 target_seq=self.seq, fault=self.fault)
        if self.wrench_sample is not None:
            sample, captured = self.wrench_sample[:2]
            r["wrench"] = dict(sample, age_s=max(0.0, time.monotonic() - captured))
        return r

    def _sample_wrench(self):
        # Sampled on every state request. A 50 ms cache would let the reported wrench age
        # reach 50 ms; motion_server takes max(state age, wrench age) and then adds the RPC round trip and
        # the IK time before comparing with its 0.1 s state_timeout, so half of the budget would be spent here
        # and an ordinary 30-40 ms scheduling hiccup would produce stale_feedback.
        # The real robot server reports the age of a continuously sampled wrench, a few milliseconds.
        with self.sim.lock:
            st = self.sim.state()
            captured = time.monotonic() - st["state_age_s"]
            sample = {"flange": self.sim.flange_w.tolist(), "tcp_pose_base": st["cart_pos_real"],
                      "tool_end": list(self.sim.tool_end)}
        self.wrench_sample = (sample, captured, time.monotonic())

    def _idle_check(self, owner):
        st = self.sim.state()
        if self.owner or self.fault or st["in_realtime_loop"] or not st["stop_confirmed"]:
            raise RuntimeError("Control already active or unresolved; explicitly stop first")
        if not isinstance(owner, str) or not (1 <= len(owner) <= 128):
            raise ValueError("Invalid motion owner")

    def motion(self, method, p):
        if method == "motion_stop":
            self._stop()
            return self._snapshot()
        if method == "motion_home":
            self._idle_check(p.get("owner"))
            self.owner, self.mode, self.seq = p["owner"], "sdk_movej", 0
            try:
                measured = self.sim.movej_home()
                err = float(np.max(np.abs(measured - self.sim.home)))
                if err > np.deg2rad(1.0):
                    raise RuntimeError(f"SDK MoveJ did not reach home: joint error {err:g} rad")
            except Exception as exc:
                self.fault = f"SDK home failed; stop required: {exc}"
                raise
            self.deadline = time.monotonic() + 1.0
            r = self._snapshot()
            r.update(home_joint_pos_rad=self.sim.home.tolist(),
                     home_measured_joints_rad=measured.tolist(), home_joint_error_rad=err)
            return r
        if method == "motion_gripper_state":
            return self.sim.gripper_feedback()
        if method == "motion_state":
            if p.get("owner") and p["owner"] == self.owner:
                self.deadline = time.monotonic() + 1.0
            if p.get("include_wrench") and self.sim.in_rt:
                self._sample_wrench()
            return self._snapshot()
        if method == "motion_start":
            self._idle_check(p.get("owner"))
            mode = p.get("mode")
            if mode not in ("cartesian_impedance", "joint_impedance", "joint_position"):
                raise ValueError("Unsupported motion mode")
            self.owner, self.mode, self.seq, self.wrench_sample = p["owner"], mode, 0, None
            self.sim.start(mode)
            self.deadline = time.monotonic() + 1.0
            return self._snapshot()
        if method != "motion_target":
            raise ValueError("Unknown motion RPC")
        st = self._snapshot()
        if (not self.owner or p.get("owner") != self.owner or p.get("server_id") != self.server_id
                or self.fault or not st["in_realtime_loop"]):
            raise RuntimeError("Motion session mismatch")
        if p.get("target_seq") != self.seq + 1:
            raise RuntimeError("Out-of-order or repeated target; do not retry motion")
        if st["state_age_s"] is None or st["state_age_s"] > 0.1:
            raise RuntimeError("Native feedback is stale")
        if HL_STAGED and not NO_FORCE_LIMIT:
            if self.sim.hl_state == "limit":
                raise RuntimeError("Invalid feedback or hard force limit")
        elif not NO_FORCE_LIMIT and np.linalg.norm(np.asarray(st["ext_wrench"])[:3]) >= HARD_FORCE_N:
            raise RuntimeError("Invalid feedback or hard force limit")
        kind = p.get("action_type")
        if kind in ("cart_pos", "joint_pos"):
            dt = float(p["interpolate_time"])
            if not np.isfinite(dt) or not 0.005 <= dt <= 0.2:
                raise ValueError("Invalid interpolation time")
        value = np.asarray(p.get("action_value"), float)
        if not np.all(np.isfinite(value)):
            raise ValueError("Target must be finite")
        if kind == "cart_pos" and self.mode == "cartesian_impedance":
            if not self.sim.set_target(value, dt):
                raise RuntimeError("Native Cartesian jump guard rejected target")
        elif kind == "joint_pos" and self.mode in ("joint_impedance", "joint_position"):
            if value.shape != (7,):
                raise ValueError("Invalid joint target shape")
            if st["joint_queue_size"] >= 3:
                raise RuntimeError("Joint target queue is falling behind")
            self.sim.set_joint(value, dt)
        elif kind == "gripper":
            if value.shape != (1,) or not 0 <= value[0] <= 1:
                raise ValueError("Invalid or unavailable gripper")
            self.sim.grip_cmd = float(value[0])
        else:
            raise ValueError("Target does not match motion mode")
        self.seq += 1
        self.deadline = time.monotonic() + 1.0
        r = self._snapshot()
        r["accepted"] = True
        return r

    def handle(self, req):
        method, p = req.get("method"), req.get("params") or {}
        self._lease()
        if isinstance(method, str) and method.startswith("motion_"):
            return self.motion(method, p)
        if method == "stop_realtime_loop":
            self._stop()
            return None
        if method == "is_in_realtime_loop":
            return self.sim.in_rt
        if method == "get_robot_info":
            return {"id": "sim", "version": "sim-mujoco-" + mujoco.__version__, "type": self.sim.robot_type,
                    "joint_num": 7, "mac": ""}
        if method == "get_tool_info":
            return {"mass": self.sim.tool_mass, "center_of_mass": list(self.sim.tool_cog), "inertia_tensor": [0, 0, 0],
                    "end_pos": list(self.sim.tool_end), "ref_pos": [0.0] * 6}
        if method == "get_base_frame":
            return [0.0] * 6
        if method == "get_state":
            st = self.sim.state()
            g = self.sim.gripper_feedback()
            return {k: v for k, v in {
                "joint_pos_real": st["joint_pos_real"], "cart_pos_real": st["cart_pos_real"],
                "ext_wrench": st["ext_wrench"], "psi": 0.0, "gripper_pos": [self.sim.grip_cmd],
                "gripper_pos_feedback_raw": [float(g["raw"])], "gripper_pos_feedback_valid": [1.0],
            }.items() if k in (p.get("quantities") or [])}
        # simulation-only extras (never used by the real stack). They expose simulator state (sim_info: success,
        # object poses), so when the server was started with SIM_RPC_TOKEN only our own processes (which get the
        # token) may call them, not the agent under test
        if isinstance(method, str) and method.startswith("sim_") and SIM_RPC_TOKEN and req.get("token") != SIM_RPC_TOKEN:
            raise ValueError(f"Unsupported method in simulation: {method}")
        if method == "sim_render":
            rgb = self.sim.render(p.get("camera", "overview"), int(p.get("width", 640)), int(p.get("height", 480)))
            return {"shape": list(rgb.shape), "data": rgb.tobytes()}
        if method == "sim_camera":
            cam = self.sim.cam_map.get(p["name"], p["name"])
            rgb, dep = self.sim.render(cam, int(p.get("width", 640)), int(p.get("height", 480)), depth=True)
            return {"shape": list(rgb.shape), "rgb": rgb.tobytes(), "depth": dep.astype(np.float32).tobytes(),
                    "time_unix": time.time()}
        if method == "sim_tactile":
            with self.sim.lock:
                return self.sim.tactile_contacts()
        if method == "sim_set_impedance":
            # Polymetis-style gains for baselines that choose their own (e.g. Show-Harness Kx/Kxd)
            kx, kxd = np.asarray(p["kx"], float), np.asarray(p["kxd"], float)
            with self.sim.lock:
                self.sim.Kt, self.sim.Kr = kx[:3].copy(), kx[3:].copy()
                self.sim.Dt, self.sim.Dr = kxd[:3].copy(), kxd[3:].copy()
            return {"kx": kx.tolist(), "kxd": kxd.tolist()}
        if method == "sim_gripper":
            # a baseline's own gripper device (Show-Harness's Franka hand via the NUC shim): on a Franka the
            # hand is a separate device from the arm controller, so this is not gated by the arm's motion
            # session or its force limit (our method's gripper still goes through motion_target)
            v = float(p["value"])
            if not 0.0 <= v <= 1.0:
                raise ValueError("Invalid gripper value")
            with self.sim.lock:
                self.sim.grip_cmd = v
            return {"target": v}
        if method == "sim_dump_recording":
            if not hasattr(self.sim, "dump_recording"):
                raise ValueError("Recording not supported by this backend")
            return self.sim.dump_recording(p["path"])
        if method == "sim_config":
            return {"tactile_geometry": self.sim.tactile_geometry, "cam_map": self.sim.cam_map,
                    "robot_type": self.sim.robot_type, "task": getattr(self.sim, "task_info", None)}
        if method == "sim_info":
            sim = self.sim
            return {"sim_time": sim.sim_time, "tcp_world": sim.tcp_world(),
                    "success": sim.success, "success_time": getattr(sim, "success_time", None),
                    "progress": getattr(sim, "progress", None), "metrics": getattr(sim, "metrics", None),
                    "task": getattr(sim, "task_info", None)}
        if method == "sim_geom_points":
            # GenCHiP pose oracle (geom_points above); gated by SIM_RPC_TOKEN like every sim_* call
            with self.sim.lock:
                return geom_points(self.sim.m, self.sim.d, p)
        raise ValueError(f"Unsupported method in simulation: {method}")

    def serve(self, stop_event, address=None, sensors_only=False):
        """One REP loop. The control endpoint mirrors the real robot server; a second,
        sensors_only endpoint serves sim_* rendering/tactile so slow renders never queue
        in front of motion_server's 0.15 s state/target RPCs."""
        address = address or self.address
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(address)
        LOG.info("Simulated robot server (%s) listening on %s", "sensors" if sensors_only else "control", address)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while not stop_event.is_set():
            if not poller.poll(100):
                continue
            req = msgpack.unpackb(sock.recv(), raw=False)
            try:
                if sensors_only and not str(req.get("method", "")).startswith("sim_"):
                    raise ValueError("Sensor endpoint serves sim_* methods only")
                resp = {"success": True, "result": self.handle(req)}
            except Exception as exc:
                LOG.warning("%s failed: %s", req.get("method"), exc)
                resp = {"success": False, "error": str(exc)}
            sock.send(msgpack.packb(resp, use_bin_type=True))
        sock.close()
        ctx.term()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", default="tcp://127.0.0.1:5655")
    ap.add_argument("--scene", type=Path, default=HERE / "assets" / "scene.xml")
    ap.add_argument("--wrench-noise", type=float, default=0.0, help="N (torque scaled by 0.02 m)")
    ap.add_argument("--start-deg", type=float, nargs=7, help="Initial joints (default: home)")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    sim = Sim(a.scene, wrench_noise=a.wrench_noise, start_deg=a.start_deg)
    stop = threading.Event()
    phys = threading.Thread(target=sim.run, args=(stop,), daemon=True)
    phys.start()
    try:
        SimZmqServer(sim, a.address).serve(stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
