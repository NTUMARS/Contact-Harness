"""Factory tasks re-hosted in MuJoCo (robosuite, standard Panda).

Faithful to Isaac Lab's Factory (isaaclab_tasks/direct/factory, v2.3.2) in everything that
defines the task: the ORIGINAL part meshes (the USDs Isaac Lab loads, converted by
factory_assets.py), masses and friction, the initial-state randomization procedure and ranges,
and the success test (_get_curr_successes). Physics engine, contact model and controller differ,
so this is NOT the original Isaac environment and must be reported as
"Factory tasks re-hosted in MuJoCo".

PegInsert (Factory: Peg8mm / Hole8mm)
  peg   diameter 7.986 mm, length 50 mm, 19 g, friction 0.75. Gravity ON (deviation: Factory disables the
        held asset's gravity because its RL policies never release it; our protocol releases and pushes)
  hole  original mesh: bore 9.0 mm (cfg 'diameter' field says 8.1 mm but is not the collision
        geometry), 1 mm entry chamfer, depth 25 mm; 40 x 40 mm block; fixed asset.
        The COLLISION bore is the cfg's 8.1 mm (radial clearance ~0.06 mm), 1 mm chamfer
  peg   original mesh has 0.5 mm chamfers at both ends
  init  hole at robot-base + (0.6, 0, 0.05) +- U(0.05, 0.05, 0.05), yaw U(0, 360 deg)
        fingertip at hole tip + (0, 0, 0.047) +- U(0.02, 0.02, 0.01), facing down, yaw +- 0.785
        peg in hand: base 50 - 17.6 mm (finger-pad length) below the fingertip (= physical finger
        end), +- U(3, 0, 3) mm
  press fit (addition, not in Factory: the real task inserts a USB-A cable, whose contact springs need a
        push to seat and hold the plug by friction, no latch; Factory's peg is a 0.5 mm clearance fit
        that slides in under its own weight). Once the peg base is 3 mm inside the hole (tilt < 5 deg) it
        rides a slide along the hole axis with Coulomb friction PRESS_F = 15 N all the way to the bottom;
        the socket shell guides it coaxial and centred over the first 2 mm of advance (passive: a plug that
        is not pushed is not straightened)
        (USB 2.0: USB-A insertion force <= 35 N, extraction >= 10 N; 15 N lies between the real prompt's
        10 N search limit and its 20 N insertion limit): it advances only while pushed with more than
        15 N and resists withdrawal the same way; pulled back to 1 mm inside the rim, it is released.
  success  peg-base xy distance < 2.5 mm and height above hole bottom < 0.04 * 25 mm = 1 mm
  engaged  height above hole bottom < 0.9 * 25 mm (Factory engage_threshold)
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "peg_hole"
FINGERPAD_LENGTH = 0.017608          # Factory RobotCfg.franka_fingerpad_length
PRESS_F = 15.0                       # N, connector insertion force (press fit, see docstring)
PRESS_ENGAGE = 0.003                 # m inside the rim where the press fit starts
PRESS_RELEASE = 0.001                # m inside the rim where a withdrawn peg is released
PRESS_TILT_DEG = 5.0
FLOOR_PAD_TOP = 0.0009               # m: compliant pad on the hole floor, peg-only (plastic plug/socket give)
FLOOR_PAD_SOLREF = "0.01 1"          # time constant 10 ms, critically damped (mass-normalised: a direct
                                     # "-5e4 -400" was unstable on the free 19 g peg and ejected it)
PAD_SOLREF = (0.01, 3.0)             # finger pads: robosuite's (0.01, 0.5) is underdamped; the peg held between the
                                     # pads rang at ~16 Hz on the hole chamfer while the arm stood still (0-9 N force
                                     # chatter); rubber/gel pads are heavily damped.
                                     # Only the damping changes: a contact averages both geoms' solref, so pad-peg contacts
                                     # keep their time constant (0.006 s with the grasp patch, 0.0065 s with the peg) and
                                     # get damping ratio 2 instead of 0.75
PRESS_TORQUESCALE = 1.0             # weld rotational weight (1 = as stiff as translation)
PRESS_ALIGN_TRAVEL = 0.002           # m of axial advance over which the socket shell brings the plug coaxial
                                     # (passive guidance: no advance, no alignment; time-based 0.2 s / 1 s
                                     # alignment pulled the plug straight even against a braked arm, 45-80 N)

PEG_INSERT = dict(
    name="peg_insert",
    held=dict(diameter=0.007986, height=0.050, mass=0.019, friction=0.75),
    fixed=dict(diameter=0.0081, height=0.025, base_height=0.0, friction=0.75, ),
    hand_init_pos=[0.0, 0.0, 0.047], hand_init_pos_noise=[0.02, 0.02, 0.01],
    hand_init_orn=[np.pi, 0.0, 0.0], hand_init_orn_noise=[0.0, 0.0, 0.785],
    fixed_asset_init_pos_noise=[0.05, 0.05, 0.05],
    fixed_asset_init_orn_deg=0.0, fixed_asset_init_orn_range_deg=360.0,
    held_asset_pos_noise=[0.003, 0.0, 0.003],
    success_threshold=0.04, engage_threshold=0.9,
    fixed_default_pos=[0.6, 0.0, 0.05],          # relative to the robot base (Factory init_state)
)


def look_at_quat(cam_pos, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera quaternion (wxyz) looking from cam_pos at target (camera looks along -z)."""
    f = np.asarray(target, float) - np.asarray(cam_pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return R.from_matrix(np.c_[x, y, -f]).as_quat()[[3, 0, 1, 2]]


class FactoryPegInsert(SingleArmEnv):
    """Factory PegInsert re-hosted in MuJoCo."""

    TASK = PEG_INSERT

    def __init__(self, robots, seed=0, table_full_size=(0.8, 1.0, 0.05), **kwargs):
        self.table_full_size = table_full_size
        self.table_offset = np.array((0.0, 0.0, 0.907))   # table top just below the robot base height
        self.rng = np.random.default_rng(seed)
        kwargs.setdefault("gripper_types", "default")
        super().__init__(robots=robots, mount_types="default", **kwargs)

    # ---- model ------------------------------------------------------------------------------
    def _load_model(self):
        super()._load_model()
        self.robots[0].robot_model.set_base_xpos(
            self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
        arena = TableArena(table_full_size=self.table_full_size, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots])
        t = self.TASK
        held, fixed = t["held"], t["fixed"]
        # Original Isaac Lab Factory geometry (factory_assets.py): visual = original mesh,
        # collision = convex pieces (hole) / convex hull of the original vertices (peg).
        import json
        def vstr(vs):
            return " ".join("%.7f" % x for v in vs for x in v)
        hole_pieces = json.load(open(ASSET_DIR / "hole_8mm_collision.json"))
        peg_hull = json.load(open(ASSET_DIR / "peg_8mm_collision.json"))[0]
        A = self.model.asset
        A.append(ET.Element("mesh", name="hole_visual", file=str(ASSET_DIR / "hole_8mm_visual.obj")))
        A.append(ET.Element("mesh", name="peg_visual", file=str(ASSET_DIR / "peg_8mm_visual.obj")))
        A.append(ET.Element("mesh", name="peg_collision", vertex=vstr(peg_hull)))
        for i, pc in enumerate(hole_pieces):
            A.append(ET.Element("mesh", name=f"hole_col_{i:02d}", vertex=vstr(pc)))
        A.append(ET.Element("material", name="factory_steel", rgba="0.62 0.64 0.68 1", specular="0.6", shininess="0.6"))
        A.append(ET.Element("material", name="factory_peg", rgba="0.85 0.55 0.2 1", specular="0.4", shininess="0.4"))
        fr = fixed["friction"]
        contact = 'solref="0.003 1" solimp="0.95 0.99 0.0005"'
        geoms = "".join(
            f'<geom name="hole_col_{i:02d}" type="mesh" mesh="hole_col_{i:02d}" friction="{fr} 0.005 0.0001" '
            f'{contact} contype="1" conaffinity="9" rgba="0 0 0 0" group="3"/>' for i in range(len(hole_pieces)))
        H = fixed["height"]
        press = (f'<body name="press_carrier"><inertial pos="0 0 0" mass="0.005" diaginertia="1e-7 1e-7 1e-7"/>'
                 f'<joint name="press_slide" type="slide" axis="0 0 1" limited="true" range="0 {H - PRESS_RELEASE + 0.0002}" '
                 f'damping="0.5" armature="0.01" frictionloss="{PRESS_F}" solreffriction="0.0005 1"/></body>')
        # compliant pad on the hole floor (only the peg touches it, via the pair below): bottoming a plug
        # at ~20 mm/s on the rigid floor plate gave 90-108 N one-frame spikes. The floor plate
        # (hole_col_00, 40 x 40 x 3 mm under the whole block) itself stays rigid: softening it let a peg
        # pushed down beside the block sink under it and be ejected.
        pad = (f'<geom name="hole_floor_pad" type="cylinder" size="0.0044 {FLOOR_PAD_TOP / 2}" '
               f'pos="0 0 {FLOOR_PAD_TOP / 2}" contype="0" conaffinity="0" rgba="0 0 0 0" group="3"/>')
        fixed_xml = (f'<body name="fixed_asset" pos="0 0 1">{geoms}{pad}{press}'
                     f'<geom name="hole_visual" type="mesh" mesh="hole_visual" material="factory_steel" '
                     f'contype="0" conaffinity="0" group="1"/>'
                     # pedestal from the block base down to the table: solid like the block, so a peg cannot pass
                     # under the raised block and stand in line with the hole
                     f'<geom name="pedestal" type="box" size="0.016 0.016 0.05" pos="0 0 -0.053" '
                     f'rgba="0.3 0.3 0.33 1" {contact} contype="1" conaffinity="9" group="1"/>'
                     f'<site name="fixed_origin" size="0.001" rgba="0 0 0 0"/></body>')
        h, rr = held["height"], held["diameter"] / 2
        held_xml = (f'<body name="held_asset" pos="0 0 1.3"><freejoint name="held_asset_joint"/>'
                    f'<inertial pos="0 0 {h / 2}" mass="{held["mass"]}" '
                    f'diaginertia="{held["mass"] * (3 * rr**2 + h**2) / 12} {held["mass"] * (3 * rr**2 + h**2) / 12} {held["mass"] * rr**2 / 2}"/>'
                    f'<geom name="peg" type="mesh" mesh="peg_collision" '
                    f'friction="{held["friction"]} 0.02 0.0001" {contact} '
                    f'condim="4" contype="8" conaffinity="8" rgba="0 0 0 0" group="3"/>'
                    f'<geom name="peg_visual" type="mesh" mesh="peg_visual" material="factory_peg" '
                    f'contype="0" conaffinity="0" group="1"/>'
                    # grasp patch: a box inscribed in the peg with faces on its surface along the opening
                    # axis; collides ONLY with the finger pads (group 16), giving a multi-point patch
                    # contact like PhysX/SDF instead of one point per pad.
                    f'<geom name="peg_grip" type="box" size="{rr} {rr} 0.015" pos="0 0 {h - 0.015}" '
                    f'friction="{held["friction"]} 0.1 0.01" solref="0.002 1" solimp="0.95 0.99 0.0005" '
                    f'contype="16" conaffinity="0" condim="6" rgba="0 0 0 0" group="3"/>'
                    f'<site name="held_base" size="0.001" rgba="0 0 0 0"/></body>')
        self.model.worldbody.append(ET.fromstring(fixed_xml))
        self.model.worldbody.append(ET.fromstring(held_xml))
        eq = self.model.root.find("equality")
        if eq is None:
            eq = ET.SubElement(self.model.root, "equality")
        contact = self.model.root.find("contact")
        if contact is None:
            contact = ET.SubElement(self.model.root, "contact")
        contact.append(ET.Element("pair", geom1="peg", geom2="hole_floor_pad", condim="3", solref=FLOOR_PAD_SOLREF,
                                  friction="0.75 0.75 0.005 0.0001 0.0001"))
        eq.append(ET.Element("weld", name="press_weld", body1="press_carrier", body2="held_asset", active="false",
                             solref="0.002 1"))
        # task cameras: a close side view of the socket (role of the real 'right wrist' side view)
        # and a head view from above the robot base (role of the real head camera)
        self.model.worldbody.append(ET.Element("camera", name="task_side", mode="fixed",
                                               pos="0 0 1", xyaxes="1 0 0 0 0 1", fovy="40"))
        self.model.worldbody.append(ET.Element("camera", name="task_head", mode="fixed",
                                               pos="0 0 1", xyaxes="1 0 0 0 0 1", fovy="45"))

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model._model
        self.fixed_body = m.body("fixed_asset").id
        self.held_body = m.body("held_asset").id
        self.held_qpos = m.joint("held_asset_joint").qposadr[0]
        self.held_dof = m.joint("held_asset_joint").dofadr[0]
        self.side_cam = m.camera("task_side").id
        self.head_cam = m.camera("task_head").id
        self.pedestal = m.geom("pedestal").id
        self.press_weld = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "press_weld")   # PegInsert only
        if self.press_weld >= 0:
            pass
            self.press_carrier = m.body("press_carrier").id
            self.press_q = m.joint("press_slide").qposadr[0]
            self.press_dof = m.joint("press_slide").dofadr[0]
        # Collisions as in reality: every gripper part (pads, fingers incl. fingertips, palm) is solid
        # against the peg (group 8), the pads additionally press on the peg's grasp patch (16) for an
        # area contact. Fingertips/pads are rubber/gel: very high friction (pads 3.0, fingers 2.0) so a
        # grasped part practically cannot slip (peg-hole stays metal-metal at Factory's 0.75).
        for g in ("gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"):
            m.geom_conaffinity[m.geom(g).id] |= 16 | 8
        for g in ("gripper0_finger1_collision", "gripper0_finger2_collision", "gripper0_hand_collision"):
            m.geom_conaffinity[m.geom(g).id] |= 8
        for g in ("gripper0_finger1_collision", "gripper0_finger2_collision"):
            m.geom_friction[m.geom(g).id][0] = 2.0
        for g in ("gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"):
            # rubber/gel pad pressed at ~70 N: sliding 3.0, plus moderate torsional/rolling friction
            m.geom_friction[m.geom(g).id] = [3.0, 0.1, 0.01]
            m.geom_condim[m.geom(g).id] = 6
            m.geom_solref[m.geom(g).id] = PAD_SOLREF
        for g in range(m.ngeom):
            if m.geom(g).name in ("table_collision", "floor"):   # a dropped part lands on the table / floor
                m.geom_conaffinity[g] |= 8

    def _setup_observables(self):
        return super()._setup_observables()

    # ---- Factory initial-state randomization -----------------------------------------------
    def _reset_internal(self):
        super()._reset_internal()
        self._factory_randomize()

    def _ik(self, pos, rotm, iters=300):
        """Damped least-squares IK of the grip site; returns position/orientation errors."""
        m, d = self.sim.model._model, self.sim.data._data
        robot = self.robots[0]
        qidx = np.array(robot._ref_joint_pos_indexes)
        didx = np.array(robot._ref_joint_vel_indexes)
        site = m.site("gripper0_grip_site").id
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            ep = pos - d.site_xpos[site]
            er = R.from_matrix(rotm @ d.site_xmat[site].reshape(3, 3).T).as_rotvec()
            if np.linalg.norm(ep) < 1e-5 and np.linalg.norm(er) < 1e-4:
                break
            mujoco.mj_jacSite(m, d, jp, jr, site)
            J = np.r_[jp[:, didx], jr[:, didx]]
            e = np.r_[ep, er]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = d.qpos[qidx] + np.clip(dq, -0.2, 0.2)
            lo, hi = m.jnt_range[robot._ref_joint_indexes].T
            d.qpos[qidx] = np.clip(q, lo, hi)
        mujoco.mj_forward(m, d)
        return np.linalg.norm(ep), np.linalg.norm(er)

    def _finger_end_offset(self, origin, axis):
        """Distance from origin to the farthest finger-collision vertex along axis (world)."""
        m, d = self.sim.model._model, self.sim.data._data
        best = 0.0
        for name in ("gripper0_finger1_collision", "gripper0_finger2_collision"):
            g = m.geom(name).id
            md = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[md]:m.mesh_vertadr[md] + m.mesh_vertnum[md]]
            w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
            best = max(best, float(((w - origin) @ axis).max()))
        return best

    def _factory_randomize(self):
        t, rng = self.TASK, self.rng
        m, d = self.sim.model._model, self.sim.data._data
        if self.press_weld >= 0:
            self._press_release()
            d.qpos[self.press_q] = t["fixed"]["height"] - PRESS_RELEASE
        base = d.xpos[m.body("robot0_base").id].copy()
        # (1) fixed asset pose: default + U[-1,1]*noise, yaw = init + range * U[0,1]
        fpos = base + np.array(t["fixed_default_pos"]) + (2 * rng.random(3) - 1) * np.array(t["fixed_asset_init_pos_noise"])
        yaw = np.deg2rad(t["fixed_asset_init_orn_deg"]) + np.deg2rad(t["fixed_asset_init_orn_range_deg"]) * rng.random()
        m.body_pos[self.fixed_body] = fpos
        m.body_quat[self.fixed_body] = R.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
        # pedestal from the hole floor down to the table top
        drop = max(0.005, fpos[2] - 0.003 - self.table_offset[2])
        m.geom_size[self.pedestal][2] = drop / 2
        m.geom_pos[self.pedestal][2] = -0.003 - drop / 2
        tip_c = fpos + np.array([0, 0, t["fixed"]["height"]])
        # side camera: 25 cm to the side of the socket, looking at it horizontally
        m.cam_pos[self.side_cam] = fpos + np.array([0.0, -0.25, 0.03])
        m.cam_quat[self.side_cam] = look_at_quat(m.cam_pos[self.side_cam], tip_c)
        # head camera: front-left above the socket, facing the robot (avoids arm occlusion)
        m.cam_pos[self.head_cam] = fpos + np.array([0.35, 0.22, 0.28])   # front-left, above (agentview-like)
        m.cam_quat[self.head_cam] = look_at_quat(m.cam_pos[self.head_cam], tip_c)
        mujoco.mj_forward(m, d)
        tip = fpos + np.array([0, 0, t["fixed"]["height"] + t["fixed"]["base_height"]])
        # (2) fingertip above the fixed tip, facing down with random yaw; retry like Factory.
        # Factory's fingertip frame has the orientation of Isaac's panda_hand (fingers open along its y);
        # robosuite's grip site is turned -90 deg about its z from its hand (= Isaac's panda_hand axes, fingers
        # along y), so the grip-site target is rotm @ Rz(-90). Factory's hand_init_pos is for the fingertip midpoint = the
        # physical finger ends; robosuite's grip site lies inside the fingers, so the IK target is shifted
        # back by the finger-end offset (two passes, as the offset depends on the pose).
        site = m.site("gripper0_grip_site").id
        RZ90 = R.from_euler("z", -np.pi / 2).as_matrix()
        for _ in range(20):
            target = tip + np.array(t["hand_init_pos"]) + (2 * rng.random(3) - 1) * np.array(t["hand_init_pos_noise"])
            eul = np.array(t["hand_init_orn"]) + (2 * rng.random(3) - 1) * np.array(t["hand_init_orn_noise"])
            rotm = R.from_euler("xyz", eul).as_matrix()
            for _k in range(2):
                end = self._finger_end_offset(d.site_xpos[site], d.site_xmat[site].reshape(3, 3)[:, 2])
                ep, er = self._ik(target - rotm[:, 2] * end, rotm @ RZ90)
            if ep < 1e-3 and er < 1e-3:
                break
        # (3) held asset in hand: Factory's flipped fingertip frame (180 deg about y), base (height - pad
        # length) below, + noise in that frame (Isaac axes: x = across the jaw, y = finger opening axis)
        tip_pos, tip_rot = d.site_xpos[site].copy(), d.site_xmat[site].reshape(3, 3).copy()
        tip_pos = tip_pos + tip_rot[:, 2] * self._finger_end_offset(tip_pos, tip_rot[:, 2])
        fingertip = tip_rot @ RZ90.T                                         # Isaac fingertip axes
        flipped = fingertip @ R.from_euler("y", np.pi).as_matrix()          # z up
        rel = np.array([0.0, 0.0, t["held"]["height"] - FINGERPAD_LENGTH])
        noise = (2 * rng.random(3) - 1) * np.array(t["held_asset_pos_noise"])
        held_pos = tip_pos + flipped @ (-rel + noise)
        d.qpos[self.held_qpos:self.held_qpos + 3] = held_pos
        d.qpos[self.held_qpos + 3:self.held_qpos + 7] = R.from_matrix(flipped).as_quat()[[3, 0, 1, 2]]
        d.qvel[self.held_dof:self.held_dof + 6] = 0
        # open fingers around the peg, then close on it while the arm holds still
        robot = self.robots[0]
        gq = np.array(robot._ref_gripper_joint_pos_indexes)
        d.qpos[gq] = [0.02, -0.02]
        mujoco.mj_forward(m, d)
        self.factory_init = {"fixed_pos": fpos.tolist(), "fixed_yaw": float(yaw), "hand_target": target.tolist(),
                             "ik_err": [float(ep), float(er)], "held_noise": noise.tolist()}

    # ---- press fit (called every control tick by the simulator) -------------------------------
    def press_tick(self):
        m, d = self.sim.model._model, self.sim.data._data
        H = self.TASK["fixed"]["height"]
        Rf, pf = d.xmat[self.fixed_body].reshape(3, 3), d.xpos[self.fixed_body]
        base = Rf.T @ (d.xpos[self.held_body] - pf)                   # peg base in the hole frame
        if not self.press_engaged:
            ax = Rf.T @ d.xmat[self.held_body].reshape(3, 3)[:, 2]
            tilt = np.degrees(np.arccos(np.clip(ax[2], -1, 1)))
            # base[2] > -1 mm: the peg is in the bore (entered from the top), not below the block
            if -0.001 < base[2] <= H - PRESS_ENGAGE and np.hypot(base[0], base[1]) < 0.001 and tilt < PRESS_TILT_DEG:
                # the slide starts with the plug's current axial velocity: a plug already moving in is
                # slowed by the 15 N friction, not stopped dead (a carrier at rest made an impact, 70-80 N)
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.held_body, vel, 0)
                d.qpos[self.press_q], d.qvel[self.press_dof] = base[2], float(vel[3:] @ Rf[:, 2])
                mujoco.mj_kinematics(m, d)
                Rc, pc = d.xmat[self.press_carrier].reshape(3, 3), d.xpos[self.press_carrier]
                Rn, pn = d.xmat[self.held_body].reshape(3, 3), d.xpos[self.held_body]
                # the socket shell guides the plug: the weld's relative pose goes from the current pose
                # (constraint satisfied, no jump) to coaxial and centred as the plug advances PRESS_ALIGN_TRAVEL
                rel0 = R.from_matrix(Rc.T @ Rn)
                z = rel0.as_matrix()[:, 2]
                c = np.cross(z, [0.0, 0.0, 1.0])
                align = R.from_rotvec(c / np.linalg.norm(c) * np.arccos(np.clip(z[2], -1, 1))) if np.linalg.norm(c) > 1e-9 else R.identity()
                self._press_rel = (Rc.T @ (pn - pc), np.zeros(3), Slerp([0, 1], R.concatenate([rel0, align * rel0])))
                self._press_q0, self._press_a = float(base[2]), 0.0
                self._press_set_rel(0.0)
                m.eq_data[self.press_weld][:3] = 0
                m.eq_data[self.press_weld][10] = PRESS_TORQUESCALE
                m.eq_active[self.press_weld] = 1
                self.press_engaged = True
            return
        q = d.qpos[self.press_q]
        a = max(self._press_a, min(1.0, (self._press_q0 - float(q)) / PRESS_ALIGN_TRAVEL))   # never un-aligns
        if a > self._press_a:
            self._press_set_rel(a)
            self._press_a = a
        if q >= H - PRESS_RELEASE:                                       # pulled back out of the press fit
            self._press_release()

    def _press_set_rel(self, a):
        p0, p1, slerp = self._press_rel
        m = self.sim.model._model
        m.eq_data[self.press_weld][3:6] = (1 - a) * p0 + a * p1
        m.eq_data[self.press_weld][6:10] = slerp([a])[0].as_quat()[[3, 0, 1, 2]]

    def _press_release(self):
        m = self.sim.model._model
        m.eq_active[self.press_weld] = 0
        m.dof_frictionloss[self.press_dof] = PRESS_F
        self.press_engaged = False

    # ---- Factory success test ----------------------------------------------------------------
    def factory_state(self):
        t = self.TASK
        m, d = self.sim.model._model, self.sim.data._data
        held_base = d.xpos[self.held_body]                     # held frame origin = peg base
        target = d.xpos[self.fixed_body] + d.xmat[self.fixed_body].reshape(3, 3) @ np.zeros(3)
        xy = float(np.linalg.norm(held_base[:2] - target[:2]))
        z = float(held_base[2] - target[2])
        h = t["fixed"]["height"]
        # z > -1 mm: the peg base is in the bore, not under the raised block in line with the hole (Factory needs no
        # lower bound because its socket sits on the table). Depth is reported only for a peg in the bore.
        in_bore = bool(xy < t["fixed"]["diameter"] / 2 + 0.0005 and z > -0.001)
        return {"xy_dist_m": xy, "z_disp_m": z,
                "success": bool(xy < 0.0025 and -0.001 < z < h * t["success_threshold"]),
                "engaged": bool(xy < 0.0025 and -0.001 < z < h * t["engage_threshold"]),
                "insertion_depth_m": float(min(h, max(0.0, h - z))) if in_bore else 0.0,
                "press_engaged": bool(getattr(self, "press_engaged", False))}

    def _check_success(self):
        return self.factory_state()["success"]

    def reward(self, action=None):
        return float(self._check_success())
