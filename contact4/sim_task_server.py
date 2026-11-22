#!/usr/bin/env python3
"""robosuite-family backends for the simulated robot server (standard Panda).

Benchmarks:
  wipe    robosuite Wipe (contact-rich wiping; success = all dirt markers wiped). The marker
          wiping rule is robosuite's own Wipe.reward(), evaluated every 20 ms.
  factory_peg  Factory PegInsert re-hosted in MuJoCo (peg_tasks.py).
  pivot   PivotUp: stand a flat box up against a wall by pushing (pivot_tasks.py).
  stack   robosuite Stack: red cube onto the green cube (pick-and-place reference task).
  place   put the bowl on the plate (place_tasks.py, our robosuite scene with LIBERO's meshes).
  write   WriteBoard: write "ROBOT" on a whiteboard lying on the table with a grasped pen, adapted from
          ManiSkill3 DrawSVG (write_tasks.py).
Each env is created and settled with its stock controller (10 steps, zero motion); then the
stock controller is bypassed and the same torque-level Cartesian impedance emulation as
sim_robot_server.py drives the Panda's torque actuators, so the unmodified
motion_server.py / CLI see the same robot-server protocol as on hardware.

Frames: the reported "native base" is a virtual frame B at the Panda base with
R_world_B = Rx(-90 deg), the real stack's measured mounting (motion_server.WORLD_FROM_BASE),
so the agent's world frame equals the simulator's world axes (z up), origin at the Panda base.
The IK model handed to motion_server (assets/sim_panda.urdf) is generated from this MuJoCo
model with the same root rotation; run_motion_server.py registers it.

Episode metrics (sim_info RPC): success / first-success time, task progress, peak contact
force on the tool, time above 15/30/60 N, and arm (non-tool) collisions with the scene.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

from sim_robot_server import Sim, SimZmqServer, LOG

HERE = Path(__file__).resolve().parent
ROBOT_TYPE = "SimPanda-LIBERO"
URDF_OUT = HERE / "assets" / "sim_panda.urdf"
TCP_LINK = "sim_panda_tcp"
R_WORLD_BASE = R.from_euler("x", -90, degrees=True).as_matrix()
SETTLE_STEPS = 10
GRIP_SPEED = 0.05                # m/s per finger (Franka Hand: 0.1 m/s opening rate)

# Show-Harness's own Franka cameras, added to every
# scene for the Show-Harness baseline only; our method keeps its cameras (cam_map). Cameras do not
# affect the physics.
#  sh_front: its real D435, calibrated extrinsics relative to the Franka base (link0): position
#            FRONT_CAM_POS, OpenCV camera-to-base FRONT_CAM_R (columns: image right, image down, view
#            direction) -> MuJoCo xyaxes = (right, -down). Intrinsics FRONT_CAM_K (fy 607.719, principal
#            point (348.961, 270.486) in 640x480): rendered 698x541 with the principal point at the centre
#            and cropped back to 640x480 by the baseline runner (exact pinhole).
#  sh_wrist: on the hand, 3.5 cm along hand +x and 3.6 cm along +z of Isaac's panda_hand (whose origin is
#            6 mm farther from the fingers than robosuite's right_hand: fingers at z 0.0584 vs 0.0524),
#            looking down the grasp axis, oriented as its wrist contract after wrist_rotation_degrees 270
#            (fingertips at the top, hand +y = left finger on the image left). 90 deg horizontal FOV
#            (ManiSkill/RoboLab); at the 4:3 frames the RealSense driver reads, fovy = 2 atan(0.75).
SH_FRONT_POS = (1.1002696, -0.00701879, 0.2589829)
SH_FRONT_R = np.array([[0.02816316, 0.21788680, -0.97556762],
                       [0.99959024, -0.00114196, 0.02860160],
                       [0.00511786, -0.97597338, -0.21782968]])
SH_FRONT_FOVY = math.degrees(2 * math.atan(270.486 / 607.719))
SH_WRIST_POS = (0.035, 0.0, 0.036 - 0.006)
# With robosuite's wiping pad (Wipe) that mount sits level with the pad's face (pad 0-3 cm along the grasp axis),
# 5 mm beside it: the pad is never in view, and with the pad on the table the camera is within the 1.1 cm near clip of the
# table or under its surface, so it shows the floor below the table.
# For the pad the camera sits 3 cm farther back along the grasp axis: the pad's edge at the top of the image, the table in view
# when pressed. Every other tool keeps SH_WRIST_POS.
SH_WRIST_POS_PAD = (0.035, 0.0, 0.0)
SH_WRIST_FOVY = math.degrees(2 * math.atan(0.75))


def _add_sh_cameras(xml):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    bodies = {b.get("name"): b for b in root.iter("body")}
    if "robot0_link0" not in bodies or "robot0_right_hand" not in bodies:
        return xml
    right, down = SH_FRONT_R[:, 0], SH_FRONT_R[:, 1]
    fmt = lambda v: " ".join("%.8g" % x for x in v)
    ET.SubElement(bodies["robot0_link0"], "camera", name="sh_front", mode="fixed", pos=fmt(SH_FRONT_POS),
                  xyaxes=fmt(np.r_[right, -down]), fovy="%.6g" % SH_FRONT_FOVY)
    wrist = SH_WRIST_POS_PAD if any("wiping_surface" in (g.get("name") or "") for g in root.iter("geom")) else SH_WRIST_POS
    ET.SubElement(bodies["robot0_right_hand"], "camera", name="sh_wrist", mode="fixed", pos=fmt(wrist),
                  xyaxes="0 -1 0 -1 0 0", fovy="%.6g" % SH_WRIST_FOVY)
    return ET.tostring(root, encoding="unicode")


def _install_sh_cameras():
    from robosuite.environments.base import MujocoEnv
    if getattr(MujocoEnv, "_sh_cameras", False):
        return
    orig = MujocoEnv._initialize_sim

    def _initialize_sim(self, xml_string=None):
        orig(self, _add_sh_cameras(xml_string if xml_string else self.model.get_xml()))
    MujocoEnv._initialize_sim, MujocoEnv._sh_cameras = _initialize_sim, True


def write_ik_urdf(m, path):
    """Mesh-free URDF of robot0_link0..right_hand from the live MuJoCo model, rooted at frame B."""
    def rpy(q):
        return " ".join(f"{v:.9g}" for v in R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz"))

    def xyz(v):
        return " ".join(f"{x:.9g}" for x in v)

    def inertial(b):
        return (f'<inertial><origin xyz="{xyz(m.body_ipos[b])}" rpy="{rpy(m.body_iquat[b])}"/>'
                f'<mass value="{max(m.body_mass[b], 1e-4):.6g}"/>'
                f'<inertia ixx="{max(m.body_inertia[b][0], 1e-6):.6g}" iyy="{max(m.body_inertia[b][1], 1e-6):.6g}" '
                f'izz="{max(m.body_inertia[b][2], 1e-6):.6g}" ixy="0" ixz="0" iyz="0"/></inertial>')

    out = ['<?xml version="1.0"?>', '<robot name="sim_panda_libero">', '<link name="sim_base"/>',
           f'<link name="robot0_link0">{inertial(m.body("robot0_link0").id)}</link>',
           '<joint name="mount" type="fixed"><parent link="sim_base"/><child link="robot0_link0"/>'
           f'<origin xyz="0 0 0" rpy="{math.pi / 2:.9g} 0 0"/></joint>']
    parent = "robot0_link0"
    for i in range(1, 8):
        b = m.body(f"robot0_link{i}").id
        j = m.joint(f"robot0_joint{i}").id
        lo, hi = m.jnt_range[j]
        out.append(f'<link name="robot0_link{i}">{inertial(b)}</link>')
        out.append(f'<joint name="robot0_joint{i}" type="revolute"><parent link="{parent}"/>'
                   f'<child link="robot0_link{i}"/><origin xyz="{xyz(m.body_pos[b])}" rpy="{rpy(m.body_quat[b])}"/>'
                   f'<axis xyz="{xyz(m.jnt_axis[j])}"/><limit lower="{lo:.9g}" upper="{hi:.9g}" effort="87" velocity="2.1"/></joint>')
        parent = f"robot0_link{i}"
    b = m.body("robot0_right_hand").id
    out.append(f'<link name="{TCP_LINK}"/>')
    out.append(f'<joint name="flange" type="fixed"><parent link="{parent}"/><child link="{TCP_LINK}"/>'
               f'<origin xyz="{xyz(m.body_pos[b])}" rpy="{rpy(m.body_quat[b])}"/></joint>')
    out.append("</robot>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))


def level_tool(env, iters=600):
    """Turn the hand so the wiping pad is level (tool axis straight down), keeping the pad's position.
    robosuite's default Panda pose tilts the hand 8.3 deg, which leaves one pad edge ~1 cm above the
    table; its wipe rule only counts dirt under a pad pressed flat. The real pad starts level."""
    m, d = env.sim.model._model, env.sim.data._data
    robot = env.robots[0]
    qidx, didx = np.array(robot._ref_joint_pos_indexes), np.array(robot._ref_joint_vel_indexes)
    site = m.site("gripper0_grip_site").id
    mujoco.mj_forward(m, d)
    pos, Rs = d.site_xpos[site].copy(), d.site_xmat[site].reshape(3, 3).copy()
    x = Rs[:, 0] - Rs[:, 0][2] * np.array([0, 0, 1.0]); x /= np.linalg.norm(x)
    z = np.array([0, 0, -1.0]); target = np.c_[x, np.cross(z, x), z]
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        ep = pos - d.site_xpos[site]
        er = R.from_matrix(target @ d.site_xmat[site].reshape(3, 3).T).as_rotvec()
        if np.linalg.norm(ep) < 1e-5 and np.linalg.norm(er) < 1e-5:
            break
        mujoco.mj_jacSite(m, d, jp, jr, site)
        J = np.r_[jp[:, didx], jr[:, didx]]
        d.qpos[qidx] += J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), np.r_[ep, er])
    d.qvel[didx] = 0
    mujoco.mj_forward(m, d)
    robot.controller.update(force=True)
    robot.controller.reset_goal()


def build_wipe(a):
    import robosuite as suite
    np.random.seed(a.seed)          # robosuite samples the dirt path with the global numpy RNG
    env = suite.make("Wipe", robots="Panda", has_renderer=False, has_offscreen_renderer=True,
                     use_camera_obs=False, control_freq=20, ignore_done=True, hard_reset=False)
    np.random.seed(a.seed)
    env.reset()
    level_tool(env)
    for _ in range(SETTLE_STEPS):
        env.step(np.zeros(env.action_dim))
    info = {"benchmark": "wipe", "seed": a.seed, "num_markers": env.num_markers,
            "language": "wipe all the dirt off the table"}

    def progress(inner):
        inner.reward()             # robosuite's marker-wiping rule (updates inner.wiped_markers)
        return {"wiped_markers": len(inner.wiped_markers), "num_markers": inner.num_markers,
                "fraction_wiped": len(inner.wiped_markers) / inner.num_markers}
    return env, env, info, progress


def strong_grasp(env, pin=None, zero_gravity=False):
    """Close on the part already placed in the hand, with the real setup's grasp (shared by tasks).
    pin: optional callable re-placing the part during the first closing steps (setup only).
    zero_gravity: gravity off during the first closing steps (setup only), as in Factory's reset, which
    closes the hand with gravity disabled; otherwise the part drops ~1 cm per 50 ms step before the
    fingers reach it."""
    # Strong grasp as in the real setup (high gripper force): kp 2e4 N/m, saturating at 70 N per
    # finger (Franka hand continuous grasp force); robosuite's default fingers give only ~4 N.
    # Joint damping (integrated implicitly) keeps the stiff fingers stable.
    m = env.sim.model._model
    for name in ("gripper0_gripper_finger_joint1", "gripper0_gripper_finger_joint2"):
        a_id = m.actuator(name).id
        m.actuator_gainprm[a_id][0] = 2e4
        m.actuator_biasprm[a_id][1] = -2e4
        m.actuator_forcerange[a_id] = [-70.0, 70.0]
    for name in ("gripper0_finger_joint1", "gripper0_finger_joint2"):
        m.dof_damping[m.joint(name).dofadr[0]] = 40.0
    g = m.opt.gravity.copy()
    for k in range(25):            # close on the part with the stock controller holding the arm
        if pin is not None and k < 8:
            pin()
        m.opt.gravity[:] = 0.0 if (zero_gravity and k < 8) else g
        env.step(np.r_[np.zeros(env.action_dim - 1), 1.0])
    m.opt.gravity[:] = g
    env._sim_initial_grip = 0.0    # the episode starts with the part grasped
    env._sim_physics_dt = 0.0005   # stiff part contacts: 4 physics substeps per 2 ms tick
    # Static friction must hold the grasped part (as on the real gripper); MuJoCo's soft friction
    # otherwise lets it creep/twist under sustained load. Elliptic cones + impratio 100 + noslip
    # reduce in-hand rotation to ~0.05 deg per 20 N slide (was ~0.8 deg at impratio 10).
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    m.opt.impratio = 100.0
    m.opt.noslip_iterations = 50


def build_write(a):
    import robosuite as suite
    import write_tasks  # noqa: F401  (registers WriteBoard with robosuite)
    env = suite.make("WriteBoard", robots="Panda", seed=a.seed, has_renderer=False,
                     has_offscreen_renderer=True, use_camera_obs=False, control_freq=20,
                     ignore_done=True, hard_reset=False)
    env.reset()
    strong_grasp(env, pin=env.pin_pen)      # fingers close sideways: hold the pen until they have closed
    env._sim_cam_map = {"head": "write_head", "left_wrist": "robot0_eye_in_hand", "right_wrist": "write_side"}
    env._sim_tick_hook = env.ink_tick
    info = {"benchmark": "write", "seed": a.seed, "write_init": env.write_init, "word": write_tasks.WORD,
            "language": "write the letters ROBOT on the whiteboard",
            "note": "WriteBoard adapted from ManiSkill3 DrawSVG (grasped pen, force-gated ink, vertical board)"}
    return env, env, info, lambda inner: inner.write_state()


def build_factory(a):
    import robosuite as suite
    import peg_tasks  # noqa: F401  (registers the Factory re-hosted envs with robosuite)
    env = suite.make("FactoryPegInsert", robots="Panda", seed=a.seed, has_renderer=False,
                     has_offscreen_renderer=True, use_camera_obs=False, control_freq=20,
                     ignore_done=True, hard_reset=False)
    env.reset()
    strong_grasp(env, zero_gravity=True)
    env._sim_cam_map = {"head": "task_head", "left_wrist": "robot0_eye_in_hand", "right_wrist": "task_side"}
    env._sim_tick_hook = env.press_tick           # connector press fit (peg_tasks.PRESS_F)
    info = {"benchmark": "factory_peg_insert", "seed": a.seed, "factory_init": env.factory_init,
            "language": "insert the peg held in the gripper fully into the hole",
            "note": "Factory PegInsert re-hosted in MuJoCo (dimensions/randomization/success from Isaac Lab Factory)"}
    return env, env, info, lambda inner: inner.factory_state()


def build_pivot(a):
    import robosuite as suite
    import pivot_tasks  # noqa: F401  (registers PivotUpBox with robosuite)
    env = suite.make("PivotUpBox", robots="Panda", seed=a.seed, has_renderer=False,
                     has_offscreen_renderer=True, use_camera_obs=False, control_freq=20,
                     ignore_done=True, hard_reset=False)
    env.reset()
    # same finger servo as the grasping tasks (real gripper force), fingers closed: the hand is a pusher
    m = env.sim.model._model
    for name in ("gripper0_gripper_finger_joint1", "gripper0_gripper_finger_joint2"):
        a_id = m.actuator(name).id
        m.actuator_gainprm[a_id][0] = 2e4
        m.actuator_biasprm[a_id][1] = -2e4
        m.actuator_forcerange[a_id] = [-70.0, 70.0]
    for name in ("gripper0_finger_joint1", "gripper0_finger_joint2"):
        m.dof_damping[m.joint(name).dofadr[0]] = 40.0
    for _ in range(SETTLE_STEPS):
        env.step(np.r_[np.zeros(env.action_dim - 1), 1.0])
    env._sim_initial_grip = 0.0
    env._sim_physics_dt = 0.001        # 1 ms physics step (Zhou & Held's robosuite setup)
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    m.opt.impratio = 100.0
    m.opt.noslip_iterations = 50
    env._sim_cam_map = {"head": "pivot_head", "left_wrist": "robot0_eye_in_hand", "right_wrist": "pivot_side"}
    env._sim_tick_hook = env.pivot_tick
    info = {"benchmark": "pivot", "seed": a.seed, "pivot_init": env.pivot_init,
            "language": "stand the box up against the wall",
            "note": "PivotUp: flat box pushed up against a wall (extrinsic dexterity; after Zhou & Held 2022, "
                    "Shirai 2026, ACP, FACTR); our scene and success test"}
    return env, env, info, lambda inner: inner.pivot_state()


def _real_fingers(m):
    """The finger servo of the grasping tasks (real Franka hand force: kp 2e4 N/m, 70 N per finger, damped)."""
    for name in ("gripper0_gripper_finger_joint1", "gripper0_gripper_finger_joint2"):
        a_id = m.actuator(name).id
        m.actuator_gainprm[a_id][0] = 2e4
        m.actuator_biasprm[a_id][1] = -2e4
        m.actuator_forcerange[a_id] = [-70.0, 70.0]
    for name in ("gripper0_finger_joint1", "gripper0_finger_joint2"):
        m.dof_damping[m.joint(name).dofadr[0]] = 40.0


def _register_stack_level():
    """robosuite Stack with its table top at 0.907 m, the height of the Pivot table, 5 mm below the robot base.
    robosuite's Stack table is 0.112 m below the base; the baselines' front camera (sh_front) is
    calibrated relative to the base as on the real robot, and at that table height the cubes fell below its view.
    Raising the table keeps every camera calibration valid; robosuite's scene, sampler and success test otherwise
    unchanged (its success test measures the lift from the table, so it follows the new height)."""
    from robosuite.environments.manipulation.stack import Stack
    from robosuite.environments.base import REGISTERED_ENVS
    if "StackLevel" in REGISTERED_ENVS:
        return

    class StackLevel(Stack):
        def _load_model(self):
            self.table_offset = np.array((0, 0, 0.907))
            super()._load_model()
    REGISTERED_ENVS["StackLevel"] = StackLevel


def build_stack(a):
    """Stack (pick-and-place reference task): robosuite's own Stack scene with the standard Panda,
    its placement sampler (seeded through the global numpy RNG, as Wipe) and its success test (the red 4 cm cube lifted
    above the table, touching the green 5 cm cube and not grasped). Gripper open at the start; the same finger servo
    and grasp friction settings as the other tasks with a held part."""
    import robosuite as suite
    _register_stack_level()
    np.random.seed(a.seed)
    env = suite.make("StackLevel", robots="Panda", has_renderer=False, has_offscreen_renderer=True,
                     use_camera_obs=False, control_freq=20, ignore_done=True, hard_reset=False)
    np.random.seed(a.seed)
    env.reset()
    m = env.sim.model._model
    _real_fingers(m)
    for _ in range(SETTLE_STEPS):
        env.step(np.r_[np.zeros(env.action_dim - 1), -1.0])
    env._sim_initial_grip = 1.0
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    m.opt.impratio = 100.0
    m.opt.noslip_iterations = 50
    info = {"benchmark": "stack", "seed": a.seed, "language": "stack the red cube on the green cube",
            "note": "robosuite Stack (standard scene and success test); pick-and-place reference task"}

    def progress(inner):
        d = inner.sim.data
        pa, pb = d.body_xpos[inner.cubeA_body_id], d.body_xpos[inner.cubeB_body_id]
        _, _, r_stack = inner.staged_rewards()
        return {"red_above_green_m": round(float(pa[2] - pb[2]), 4),
                "xy_offset_m": round(float(np.linalg.norm(pa[:2] - pb[:2])), 4), "success": bool(r_stack > 0)}
    return env, env, info, progress


def build_place(a):
    """Place: put the bowl on the plate in our robosuite scene (place_tasks.PlaceBowl): the Stack /
    Pivot table height, LIBERO's bowl and plate meshes, positions sampled per seed (global numpy RNG, as Stack),
    LIBERO's "On" success test. Gripper open; the finger servo and grasp friction of the other tasks."""
    import robosuite as suite
    import place_tasks  # noqa: F401  (registers PlaceBowl with robosuite)
    np.random.seed(a.seed)
    env = suite.make("PlaceBowl", robots="Panda", has_renderer=False, has_offscreen_renderer=True,
                     use_camera_obs=False, control_freq=20, ignore_done=True, hard_reset=False)
    np.random.seed(a.seed)
    env.reset()
    m = env.sim.model._model
    _real_fingers(m)
    for _ in range(SETTLE_STEPS):
        env.step(np.r_[np.zeros(env.action_dim - 1), -1.0])
    env._sim_initial_grip = 1.0
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    m.opt.impratio = 100.0
    m.opt.noslip_iterations = 50
    info = {"benchmark": "place", "seed": a.seed, "language": "put the bowl on the plate",
            "note": "PlaceBowl (place_tasks.py): our robosuite scene with LIBERO's bowl and plate; pick-and-place reference"}
    return env, env, info, lambda inner: inner.place_state()


BUILDERS = {"wipe": build_wipe, "factory_peg": build_factory, "write": build_write,
            "pivot": build_pivot, "stack": build_stack, "place": build_place}


STABLE_HOLD_S = 10.0              # Stack / Place success: test held continuously this long
HOLD_TASKS = ("stack", "place")


class RobosuiteSim(Sim):
    def __init__(self, a, realtime=True, wrench_noise=0.0):
        _install_sh_cameras()
        env, inner, self.task_info, self.progress_fn = BUILDERS[a.benchmark](a)
        self.env, self.inner = env, inner
        self.m, self.d = env.sim.model._model, env.sim.data._data
        m, d = self.m, self.d
        self.dt = float(m.opt.timestep)    # control period = the benchmark's native timestep
        self.substeps = 1
        pdt = getattr(inner, "_sim_physics_dt", None)
        if pdt:
            self.substeps = max(1, int(round(self.dt / pdt)))
            m.opt.timestep = self.dt / self.substeps
        self.success = bool(self.inner._check_success())
        self.realtime, self.noise = realtime, wrench_noise
        self.rng = np.random.default_rng(0)
        self.base = m.body("robot0_base").id
        self.flange = m.body("robot0_right_hand").id
        self.tcp = m.site("gripper0_grip_site").id
        self.arm_dof = np.array([m.joint(f"robot0_joint{i}").dofadr[0] for i in range(1, 8)])
        self.arm_qpos = np.array([m.joint(f"robot0_joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_act = np.array([m.actuator(f"robot0_torq_j{i}").id for i in range(1, 8)])
        self.arm_lim = m.actuator_ctrlrange[self.arm_act].copy()
        names = {m.actuator(i).name for i in range(m.nu)}
        self.has_fingers = "gripper0_gripper_finger_joint1" in names
        if self.has_fingers:
            self.grip_acts = [m.actuator("gripper0_gripper_finger_joint1").id,
                              m.actuator("gripper0_gripper_finger_joint2").id]
            self.finger_q = [m.joint("gripper0_finger_joint1").qposadr[0], m.joint("gripper0_finger_joint2").qposadr[0]]
        self.tool_bodies = {b for b in range(m.nbody) if self._is_desc(b, self.flange)}
        self.robot_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith("robot0_")} - self.tool_bodies
        # tool_end = flange -> TCP (grip_site), as the controller's toolset would report it
        Tf = np.eye(4); Tf[:3, :3] = d.xmat[self.flange].reshape(3, 3); Tf[:3, 3] = d.xpos[self.flange]
        Tt = np.eye(4); Tt[:3, :3] = d.site_xmat[self.tcp].reshape(3, 3); Tt[:3, 3] = d.site_xpos[self.tcp]
        Tft = np.linalg.inv(Tf) @ Tt
        self.robot_type = ROBOT_TYPE
        self.tool_end = np.r_[Tft[:3, 3], R.from_matrix(Tft[:3, :3]).as_euler("xyz")].round(9).tolist()
        self.tool_mass = float(sum(m.body_mass[b] for b in self.tool_bodies))
        self.tool_cog = [0.0, 0.0, 0.0]
        self.R_wb = d.xmat[self.base].reshape(3, 3) @ R_WORLD_BASE
        self.p_wb = d.xpos[self.base].copy()
        # Observation mapping: robosuite's standard wrist + agent views, plus its side view.
        self.cam_map = {"left_wrist": "robot0_eye_in_hand", "head": "agentview", "right_wrist": "sideview"}
        # Simulated DM-Tac pads on the Panda finger pads (none on the Wipe tool). Gripper axes G:
        # x across the opening with the left pad (leftfinger, flange +y) at -x, z forward -> Rz(-90).
        self.R_flange_G = R.from_euler("z", -90, degrees=True).as_matrix()
        self.pad_geoms, pad_z = {}, 0.0934
        if self.has_fingers:
            self.pad_geoms = {"tactile_left": m.geom("gripper0_finger1_pad_collision").id,
                              "tactile_right": m.geom("gripper0_finger2_pad_collision").id}
            self.pad_half_thickness = float(m.geom_size[self.pad_geoms["tactile_left"]][1])
            pad_z = float((d.xmat[self.flange].reshape(3, 3).T @ (d.geom_xpos[self.pad_geoms["tactile_left"]] - d.xpos[self.flange]))[2])
        self.tactile_geometry = {"gripper_in_flange": [0., 0., round(pad_z, 4), 0., 0., -math.pi / 2],
                                 "opening_m": 0.08}
        q0 = d.qpos[self.arm_qpos].copy()
        self.home = q0.copy()      # the benchmark's initial arm configuration plays the role of home
        write_ik_urdf(m, URDF_OUT)
        self._init_runtime(q0)
        self.grip_cmd = getattr(inner, "_sim_initial_grip", 1.0 if self.has_fingers else 0.0)
        self.cam_map.update(getattr(inner, "_sim_cam_map", {}))
        self.success_time = None
        self._ok_since = None     # Stack / Place: start of the current uninterrupted success-test interval
        self._ticks = 0
        self.progress = None
        self.metrics = {"peak_tool_force_n": 0.0, "time_above_15n_s": 0.0, "time_above_30n_s": 0.0,
                        "time_above_60n_s": 0.0, "arm_collision_s": 0.0}
        if getattr(inner, "_sim_tick_hook", None) is not None:      # per-tick task physics (ink)
            self.hooks.append(lambda sim: sim.inner._sim_tick_hook())
        self.hooks.append(RobosuiteSim._success_hook)
        self.hooks.append(RobosuiteSim._metrics_hook)
        # 30 Hz state recording for offline video rendering (cheap: no rendering in the loop)
        self.rec_every = max(1, int(round((1 / 30) / self.dt)))
        self.rec = {"t": [], "qpos": [], "force": [], "progress": [], "wiped": [], "ink": []}
        self.hooks.append(RobosuiteSim._record_hook)
        # robosuite draws the robot's collision meshes (debug green/blue/yellow) on top of the visual
        # meshes in this MuJoCo build; make them transparent for rendering only (physics unchanged).
        for g in range(m.ngeom):
            if (m.geom(g).name.startswith(("robot0_", "gripper0_", "mount0_")) and m.geom_group[g] == 0
                    and (m.geom_contype[g] or m.geom_conaffinity[g])):
                m.geom_rgba[g][3] = 0.0
        # Warm up every observation camera once (the first offscreen render creates GL buffers and can
        # block the physics thread for >100 ms, which motion_server would see as stale feedback).
        for cam in sorted(set(self.cam_map.values())):
            self._render_one(cam, 640, 480, depth=True)
        if mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "sh_front") >= 0:   # Show-Harness cameras
            self._render_one("sh_front", 698, 541, depth=True)                     # (grows the buffer once)
            self._render_one("sh_wrist", 640, 480, depth=True)

    @staticmethod
    def _record_hook(sim):
        if sim._ticks % sim.rec_every:
            return
        r = sim.rec
        r["t"].append(sim.sim_time)
        r["qpos"].append(sim.d.qpos.copy())
        r["force"].append(float(np.linalg.norm(sim.flange_w[:3])))
        r["progress"].append(sim.progress)
        r["wiped"].append(len(getattr(sim.inner, "wiped_markers", [])))
        r["ink"].append(len(getattr(sim.inner, "ink_pts", [])))

    def dump_recording(self, path):
        """Write the recorded trajectory (npz) plus marker wipe order (wipe) for render_video.py."""
        with self.lock:
            r = {k: list(v) for k, v in self.rec.items()}
            order = [mk.visual_geoms[0] for mk in getattr(self.inner, "wiped_markers", [])]
            ink_pts = list(getattr(self.inner, "ink_pts", []))
        np.savez_compressed(path, t=np.array(r["t"]), qpos=np.array(r["qpos"]), force=np.array(r["force"]),
                            wiped=np.array(r["wiped"]), progress=json.dumps(r["progress"]),
                            wipe_order=json.dumps(order), ink=np.array(r["ink"]), ink_pts=np.array(ink_pts).reshape(-1, 2))
        return {"frames": len(r["t"]), "path": str(path)}

    @staticmethod
    def _metrics_hook(sim):
        f = float(np.linalg.norm(sim.flange_w[:3])) if getattr(sim, "flange_w", None) is not None else 0.0
        mt = sim.metrics
        mt["peak_tool_force_n"] = max(mt["peak_tool_force_n"], f)
        for th in (15, 30, 60):
            if f > th:
                mt[f"time_above_{th}n_s"] += sim.dt
        if sim._ticks % 10 == 0:   # arm links (not the tool) touching the scene
            con, m = sim.d.contact, sim.m
            for i in range(sim.d.ncon):
                b1, b2 = m.geom_bodyid[con.geom1[i]], m.geom_bodyid[con.geom2[i]]
                if (b1 in sim.robot_bodies) != (b2 in sim.robot_bodies) and not (
                        b1 in sim.tool_bodies or b2 in sim.tool_bodies):
                    mt["arm_collision_s"] += 10 * sim.dt
                    break

    @staticmethod
    def _success_hook(sim):
        sim._ticks += 1
        if sim._ticks % 10:        # 50 Hz at dt = 2 ms
            return
        if sim.progress_fn is not None:
            sim.progress = sim.progress_fn(sim.inner)
        ok = bool(sim.inner._check_success())
        # On Stack and Place the test must hold continuously for
        # STABLE_HOLD_S of sim time after the release (the test already requires the object not to be held); a cube that
        # only grazes the green cube while falling, or slides off soon after the release, is not a success.
        if sim.task_info.get("benchmark") in HOLD_TASKS:
            if ok:
                sim._ok_since = sim.sim_time if sim._ok_since is None else sim._ok_since
                ok = sim.sim_time - sim._ok_since >= STABLE_HOLD_S
            else:
                sim._ok_since = None
        if ok and not sim.success:
            sim.success_time = sim.sim_time
            LOG.info("Task success at sim t=%.2f s", sim.sim_time)
        sim.success = sim.success or ok   # first success is the episode outcome

    def _apply(self, tau):
        self.d.ctrl[self.arm_act] = np.clip(tau, self.arm_lim[:, 0], self.arm_lim[:, 1])
        if self.has_fingers:
            # finger set-point moves toward the command at a real gripper's speed (Franka Hand / libfranka:
            # max 0.1 m/s opening rate = 0.05 m/s per finger), not in one step: the stiff finger servo
            # otherwise snaps open in < 0.1 s and the fingers strike what they rest on (97 N in one
            # peg run, 37 N with the rate limit)
            w = 0.04 * float(np.clip(self.grip_cmd, 0, 1))
            cur = getattr(self, "_grip_w", None)
            if cur is None:
                cur = float(self.d.qpos[self.finger_q[0]])          # start from the actual finger position
            step = GRIP_SPEED * self.dt
            self._grip_w = cur + float(np.clip(w - cur, -step, step))
            self.d.ctrl[self.grip_acts[0]], self.d.ctrl[self.grip_acts[1]] = self._grip_w, -self._grip_w

    def gripper_opening(self):
        if not self.has_fingers:
            return 0.0                 # fixed tool, reported as a closed grasp
        return float(np.clip((self.d.qpos[self.finger_q[0]] - self.d.qpos[self.finger_q[1]]) / 0.08, 0, 1))

    def _render_one(self, cam, w, h, depth=False):
        sim = self.env.sim
        ctx = getattr(sim, "_render_context_offscreen", None)
        if ctx is not None and ctx.scn.maxgeom < 8000:   # robosuite's 1000-geom scene is too small for ink
            ctx.scn = mujoco.MjvScene(sim.model._model, maxgeom=8000)
        if not depth:
            return sim.render(camera_name=cam, width=w, height=h)[::-1].copy()
        rgb, dep = sim.render(camera_name=cam, width=w, height=h, depth=True)
        # OpenGL z-buffer -> metres (same formula as robosuite camera_utils.get_real_depth_map)
        ext = self.m.stat.extent
        near, far = self.m.vis.map.znear * ext, self.m.vis.map.zfar * ext
        metres = near / (1.0 - dep * (1.0 - near / far))
        return rgb[::-1].copy(), metres[::-1].astype(np.float32).copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=sorted(BUILDERS), default="wipe")
    ap.add_argument("--seed", type=int, default=0, help="wipe: dirt layout seed")
    ap.add_argument("--address", default="tcp://127.0.0.1:5655")
    ap.add_argument("--sensor-address", default="tcp://127.0.0.1:5656")
    ap.add_argument("--wrench-noise", type=float, default=0.0)
    ap.add_argument("--info-out", type=Path, help="Write task info JSON here once ready")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    sim = RobosuiteSim(a, wrench_noise=a.wrench_noise)
    info = dict(sim.task_info, robot_type=sim.robot_type, tool_end=sim.tool_end, urdf=str(URDF_OUT),
                home=sim.home.tolist(), success_at_start=sim.success)
    LOG.info("Task: %s", json.dumps(info))
    if a.info_out:
        a.info_out.write_text(json.dumps(info, indent=2))
    stop = threading.Event()
    zs = SimZmqServer(sim, a.address)
    for addr, sensors in ((a.address, False), (a.sensor_address, True)):
        threading.Thread(target=zs.serve, args=(stop, addr, sensors), daemon=True).start()
    try:
        sim.run(stop)       # physics + rendering stay in the thread that owns the GL context
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
