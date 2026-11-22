"""PivotUp: stand a flat box up against a wall by non-prehensile pushing (robosuite, standard Panda).

Task family "pivot against a wall / flip-up against a fixture" (extrinsic dexterity), after
  Zhou & Held, CoRL 2022 (robosuite, Panda, flat box ~15x20x5 cm, 50-345 g, 0-2 cm from the wall;
    the box is too wide for the gripper, so it must be rotated against the wall first),
  Shirai et al., RA-L 2026 (MuJoCo + robosuite, box 0.16 m, 50/110/300 g, wrist F/T),
  ACP (Hou et al. 2024; flip-up > 70 deg), FACTR (Liu et al. 2025; 90 deg and standing),
  ImplicitRDP (Chen et al., RA-L 2026; box flipping fails if the applied force exceeds 14 N).
The scene and success test are ours (no public asset/test exists for this exact task).

Scene (seeded randomization; robot-base frame = world axes, x away from the robot)
  wall    fixed block, face toward the robot at x = 0.64 +- 0.02 m, 0.40 m long (y), 0.07 m tall (taller than
          the lying box; low so that the Show-Harness front camera across the table sees over it),
          friction mu_w ~ U(0.2, 0.45) (cardboard on a painted wall; above ~0.5 the wall's friction can
          carry a light box pinched by the finger, so it lifts off the table instead of pivoting)
  table   friction mu_t ~ U(0.3, 0.6) (the table top's friction is set per seed)
  box     lying flat: depth D (x) 0.10, width W (y) 0.16, thickness H (z) 0.05 m, each +- 5 mm;
          mass U(0.05, 0.30) kg; its far face 0-2 cm from the wall, centre y +- 4 cm, yaw +- 8 deg.
          W and D exceed the Panda's 8 cm opening: it cannot be picked up flat
  hand    fingers closed, pointing down, 6 cm in front of the box's near face, 10 +- 1 cm above the table;
          finger friction 1.0 (rubber; Shirai: 0.8)
Success  the box stands up: its thickness axis is >= 80 deg from vertical (it rests on a narrow face),
         within 3 cm of the wall, its lowest corner on the table, and it has stayed still (< 1 cm/s,
         < 0.2 rad/s) without any robot contact for 2 s.
         strict_success additionally: the robot never pushed the box with more than 14 N (ImplicitRDP).
Physics: elliptic friction cones, noslip iterations, 1 ms physics step (Zhou & Held: 1 ms, noslip 20).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask

from peg_tasks import look_at_quat

WALL_X = 0.64               # wall face, from the robot base (m)
WALL_SIZE = (0.03, 0.40, 0.07)   # low enough for the Show-Harness front camera (across the table) to see the box
BOX = (0.10, 0.16, 0.05)    # depth (x), width (y), thickness (z) when lying flat
STAND_DEG = 80.0
STILL_S = 2.0
NEAR_WALL = 0.03
F_STRICT = 14.0             # N (ImplicitRDP box flipping)
FINGER_MU = 1.0


class PivotUpBox(SingleArmEnv):
    """Stand a flat box up against a wall by pushing (non-prehensile)."""

    def __init__(self, robots, seed=0, table_full_size=(0.8, 1.2, 0.05), **kwargs):
        self.table_full_size = table_full_size
        self.table_offset = np.array((0.0, 0.0, 0.907))
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
        A = self.model.asset
        A.append(ET.Element("material", name="cardboard", rgba="0.72 0.55 0.35 1", specular="0.1", shininess="0.05"))
        A.append(ET.Element("material", name="wallmat", rgba="0.55 0.6 0.66 1", specular="0.1"))
        wx, wy, wz = (s / 2 for s in WALL_SIZE)
        wall = (f'<body name="wall" pos="0 0 1"><geom name="wall" type="box" size="{wx} {wy} {wz}" '
                'material="wallmat" friction="0.5 0.005 0.0001" group="1"/></body>')
        # box geometry is resized per seed (_randomize); label stripes mark the faces for the cameras:
        # dark stripe on the top face (+z), red stripe on the near face (-x) when lying flat
        box = ('<body name="box" pos="0 0 1.2"><freejoint name="box_joint"/>'
               '<inertial pos="0 0 0" mass="0.15" diaginertia="1e-4 1e-4 1e-4"/>'
               '<geom name="box" type="box" size="0.05 0.08 0.025" material="cardboard" '
               'friction="0.1 0.005 0.0001" condim="4" group="1"/>'
               '<geom name="box_top_mark" type="box" size="0.01 0.06 0.0005" pos="0 0 0.025" '
               'rgba="0.2 0.15 0.1 1" contype="0" conaffinity="0" group="1"/>'
               '<geom name="box_near_mark" type="box" size="0.0005 0.06 0.006" pos="-0.05 0 0" '
               'rgba="0.75 0.15 0.12 1" contype="0" conaffinity="0" group="1"/>'
               '</body>')
        for x in (wall, box):
            self.model.worldbody.append(ET.fromstring(x))
        for name in ("pivot_head", "pivot_side"):
            self.model.worldbody.append(ET.Element("camera", name=name, mode="fixed", pos="0 0 1",
                                                   xyaxes="1 0 0 0 0 1", fovy="45"))

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model._model
        self.wall_body, self.wall_geom = m.body("wall").id, m.geom("wall").id
        self.box_body, self.box_geom = m.body("box").id, m.geom("box").id
        self.box_top_mark, self.box_near_mark = m.geom("box_top_mark").id, m.geom("box_near_mark").id
        self.box_qpos = m.joint("box_joint").qposadr[0]
        self.box_dof = m.joint("box_joint").dofadr[0]
        self.head_cam, self.side_cam = m.camera("pivot_head").id, m.camera("pivot_side").id
        self.table_geoms = [g for g in range(m.ngeom) if m.geom(g).name.startswith("table")]
        self.pv_robot_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(("robot0_", "gripper0_"))}
        # rubber finger surfaces (the box geom's own friction is low, so contact friction = the other geom's)
        for g in range(m.ngeom):
            if m.geom(g).name.startswith("gripper0_") and (m.geom_contype[g] or m.geom_conaffinity[g]):
                m.geom_friction[g][0] = FINGER_MU

    def _setup_observables(self):
        return super()._setup_observables()

    # ---- initial state ----------------------------------------------------------------------
    def _reset_internal(self):
        super()._reset_internal()
        self._randomize()

    def _ik(self, pos, rotm, iters=600):
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
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[ep, er])
            lo, hi = m.jnt_range[robot._ref_joint_indexes].T
            d.qpos[qidx] = np.clip(d.qpos[qidx] + np.clip(dq, -0.2, 0.2), lo, hi)
        mujoco.mj_forward(m, d)
        return float(np.linalg.norm(ep)), float(np.linalg.norm(er))

    def _finger_end_offset(self, origin, axis):
        m, d = self.sim.model._model, self.sim.data._data
        best = 0.0
        for name in ("gripper0_finger1_collision", "gripper0_finger2_collision"):
            g = m.geom(name).id
            md = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[md]:m.mesh_vertadr[md] + m.mesh_vertnum[md]]
            w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
            best = max(best, float(((w - origin) @ axis).max()))
        return best

    def _randomize(self):
        rng = self.rng
        m, d = self.sim.model._model, self.sim.data._data
        base = d.xpos[m.body("robot0_base").id].copy()
        u = lambda a: (2 * rng.random() - 1) * a          # noqa: E731
        top = self.table_offset[2]
        # wall
        wall_x = WALL_X + u(0.02)
        wall_c = base + np.array([wall_x + WALL_SIZE[0] / 2, u(0.03), 0.0])
        wall_c[2] = top + WALL_SIZE[2] / 2
        m.body_pos[self.wall_body] = wall_c
        self.mu_wall, self.mu_table = 0.2 + 0.25 * rng.random(), 0.3 + 0.3 * rng.random()
        m.geom_friction[self.wall_geom][0] = self.mu_wall
        for g in self.table_geoms:
            m.geom_friction[g][0] = self.mu_table
        # box: size, mass, pose
        self.box_size = np.array(BOX) + np.array([u(0.005), u(0.005), u(0.005)])
        Dx, Wy, Hz = self.box_size
        mass = 0.05 + 0.25 * rng.random()
        m.geom_size[self.box_geom] = self.box_size / 2
        m.body_mass[self.box_body] = mass
        m.body_inertia[self.box_body] = mass / 12 * np.array([Wy**2 + Hz**2, Dx**2 + Hz**2, Dx**2 + Wy**2])
        m.geom_pos[self.box_top_mark] = [0, 0, Hz / 2]
        m.geom_size[self.box_top_mark] = [0.01, Wy * 0.38, 0.0005]
        m.geom_pos[self.box_near_mark] = [-Dx / 2, 0, 0]
        m.geom_size[self.box_near_mark] = [0.0005, Wy * 0.38, Hz * 0.12]
        yaw = math.radians(u(8.0))
        gap = 0.02 * rng.random()
        # far face (worst corner after yaw) gap..gap from the wall
        reach = Dx / 2 * math.cos(yaw) + Wy / 2 * abs(math.sin(yaw))
        box_c = np.array([base[0] + wall_x - gap - reach, wall_c[1] + u(0.04), top + Hz / 2 + 1e-4])
        d.qpos[self.box_qpos:self.box_qpos + 3] = box_c
        d.qpos[self.box_qpos + 3:self.box_qpos + 7] = R.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
        d.qvel[self.box_dof:self.box_dof + 6] = 0
        self.box_mass, self.box_yaw, self.box_gap = mass, yaw, gap
        self.wall_face_x = base[0] + wall_x
        # cameras: head = front-left above (looks at box and wall from the robot's left), side = profile
        # along the wall (looks along -y at the box/wall/table cross-section)
        tgt = box_c + np.array([0.02, 0.0, 0.03])
        m.cam_pos[self.head_cam] = box_c + np.array([-0.05, 0.42, 0.32])
        m.cam_quat[self.head_cam] = look_at_quat(m.cam_pos[self.head_cam], tgt)
        m.cam_pos[self.side_cam] = box_c + np.array([0.0, -0.45, 0.06])
        m.cam_quat[self.side_cam] = look_at_quat(m.cam_pos[self.side_cam], box_c + np.array([0.02, 0.0, 0.05]))
        mujoco.mj_forward(m, d)
        # hand: pointing down, finger opening axis along world x (closed fingers = a 2 cm wide pusher face
        # toward the box), fingertips 6 cm in front of the near face, 10 +- 1 cm above the table
        rotm = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        robot = self.robots[0]
        qidx = np.array(robot._ref_joint_pos_indexes)
        q_seed = np.array([0.0, 0.2, 0.0, -2.2, 0.0, 2.4, math.pi / 4])
        site = m.site("gripper0_grip_site").id
        gq = np.array(robot._ref_gripper_joint_pos_indexes)
        d.qpos[gq] = [0.0, 0.0]                      # closed
        near_x = box_c[0] - Dx / 2
        tip = np.array([near_x - 0.06, box_c[1] + u(0.02), top + 0.10 + u(0.01)])
        for _ in range(20):
            d.qpos[qidx] = q_seed
            mujoco.mj_forward(m, d)
            end = self._finger_end_offset(d.site_xpos[site], d.site_xmat[site].reshape(3, 3)[:, 2])
            ep, er = self._ik(tip - rotm[:, 2] * end, rotm)
            if ep < 1e-3 and er < 1e-3:
                break
        mujoco.mj_forward(m, d)
        # episode state
        self.peak_force, self.force_now, self.last_touch_t, self.still_since = 0.0, 0.0, -1e9, None
        self.max_angle = 0.0
        self.pivot_init = {"wall_face_x_from_base": round(wall_x, 4), "box_size": self.box_size.round(4).tolist(),
                           "box_mass": round(mass, 3), "box_yaw_deg": round(math.degrees(yaw), 2),
                           "gap_m": round(gap, 4), "mu_wall": round(self.mu_wall, 3), "mu_table": round(self.mu_table, 3),
                           "finger_mu": FINGER_MU, "ik_err": [ep, er], "q0": d.qpos[qidx].round(4).tolist()}

    # ---- per-tick task state (called every control tick by the simulator) -------------------
    def pivot_tick(self):
        m, d = self.sim.model._model, self.sim.data._data
        f, c6 = 0.0, np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            if (b1 == self.box_body and b2 in self.pv_robot_bodies) or (b2 == self.box_body and b1 in self.pv_robot_bodies):
                mujoco.mj_contactForce(m, d, i, c6)
                f += float(np.linalg.norm(c6[:3]))
        t = float(d.time)
        self.force_now = f
        self.peak_force = max(self.peak_force, f)
        if f > 1e-6:
            self.last_touch_t = t
        v = d.qvel[self.box_dof:self.box_dof + 6]
        still = np.linalg.norm(v[:3]) < 0.01 and np.linalg.norm(v[3:]) < 0.2 and f <= 1e-6
        if still:
            if self.still_since is None:
                self.still_since = t
        else:
            self.still_since = None
        self.max_angle = max(self.max_angle, self._angle())

    def _angle(self):
        zb = self.sim.data._data.xmat[self.box_body].reshape(3, 3)[:, 2]     # thickness axis
        return math.degrees(math.acos(float(np.clip(abs(zb[2]), -1.0, 1.0))))

    def _corners(self):
        d = self.sim.data._data
        Rb, p = d.xmat[self.box_body].reshape(3, 3), d.xpos[self.box_body]
        s = self.box_size / 2
        return np.array([p + Rb @ (s * np.array(c)) for c in
                         [(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]])

    def pivot_state(self):
        d = self.sim.data._data
        ang = self._angle()
        cs = self._corners()
        on_table = bool(cs[:, 2].min() < self.table_offset[2] + 0.004)
        wall_gap = float(self.wall_face_x - cs[:, 0].max())
        still_s = 0.0 if self.still_since is None else float(d.time) - self.still_since
        ok = ang >= STAND_DEG and on_table and wall_gap <= NEAR_WALL and still_s >= STILL_S
        return {"angle_deg": round(ang, 1), "max_angle_deg": round(self.max_angle, 1),
                "box_wall_gap_m": round(wall_gap, 4), "box_on_table": on_table,
                "robot_box_force_n": round(self.force_now, 2), "peak_robot_box_force_n": round(self.peak_force, 2),
                "still_untouched_s": round(still_s, 2),
                "strict_success": bool(ok and self.peak_force <= F_STRICT), "success": bool(ok)}

    def _check_success(self):
        return self.pivot_state()["success"]

    def reward(self, action=None):
        return float(self._check_success())


def _register():
    # robosuite registers environments through its metaclass on class definition; nothing else to do
    return PivotUpBox


_register()
