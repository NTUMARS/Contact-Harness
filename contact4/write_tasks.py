"""WriteBoard: write letters on a whiteboard lying flat on the table with a grasped pen (robosuite, standard Panda).

Adapted from ManiSkill3 DrawSVG / DrawTriangle (mani_skill/envs/tasks/drawing): the target is a set of
stroke paths (here the uppercase block word "ROBOT"), its placement is randomized, ink is a pool of
small "dot" geoms placed where the tool touches the canvas, and success compares the drawn points with
the target path points by distance. Changes (to match the real robot task and make it contact-rich):
  * the pen is GRASPED by the parallel gripper (ManiSkill: a stick rigidly attached to the arm);
  * ink is deposited only when the pen tip actually CONTACTS the board with normal force >= 0.5 N
    (ManiSkill: whenever the tool is within a distance of the canvas, no force involved);
    pushing harder than 20 N crushes the felt tip (episode failure);
  * the canvas is a whiteboard lying flat on the table, like ManiSkill's canvas (the real board is
    vertical). Flat so that the Show-Harness baseline's own camera contract holds: its front camera
    (real D435 pose) faces the robot from across the table, and a standing board would hide the
    writing area from it;
  * success needs per-stroke coverage and precision within 5 mm (ManiSkill: 5 cm), plus the pen off the
    board at the end (real prompt: "Finish with the pen clear of the board").

Scene (seeded randomization)
  board   0.70 x 0.45 m whiteboard, 2 cm thick, lying flat on the table (writing face up; letters read
          from the robot: up = away from the robot); board-frame origin (not drawn) at robot-base +
          (0.55 +- 0.01, 0 +- 0.03) m on the face
  area    no marked area: the word may
          be written anywhere on the board, only the letter height is requested
  word    "ROBOT", letter height 5 cm, letter widths 3.25-3.5 cm, 1.5 cm gaps (23.25 cm wide)
  pen     marker 9 cm long, 14 mm diameter, 15 g, gravity on; tip sphere r 2.5 mm, 6 cm beyond the
          finger ends; tip-board contact: compliant felt (MuJoCo solref -20000 -200, ~1 mm at 3 N), friction 0.3
  start   pen perpendicular to the board, pointing down (tool +Z = into the board, tool +X = world +Y,
          tool +Y = up along the board = world +X), tip 4 +- 1 cm above the board, over the frame origin +- 2 cm
Success  after the best alignment of the target word (translation anywhere on the board, found around the
         ink centroid; scale 0.85-1.15):
         EVERY stroke's recall (its target points with ink within 5 mm) >= 0.8 AND precision (ink dots
         within 5 mm of the target) >= 0.8, the pen tip not touching the board, and the tip never crushed.
"""
from __future__ import annotations

import math
import threading
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as R

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask

from peg_tasks import look_at_quat

WORD = "ROBOT"
LETTER_H = 0.05
GAP = 0.015
BOX = (0.27, 0.075)         # nominal word area used for the scorer raster
N_DOTS = 4000
DOT_R = 0.0018
DOT_CELL = 0.0012           # one dot per 1.2 mm cell
F_INK = 0.5                 # N, minimum tip normal force that leaves ink
F_CRUSH = 20.0              # N, felt tip crushed
TOL = 0.005                 # success distance tolerance (m)
THRESH = 0.8                # recall and precision thresholds
PEN_L, PEN_R, TIP_R = 0.09, 0.007, 0.0025
TILT_DEG = 90.0             # board tilt back from vertical: 90 = lying flat on the table
BOARD_T = 0.02              # board thickness (face to table)
TIP_OUT = 0.06              # pen tip beyond the finger ends


def _arc(cx, cy, r, a0, a1, n=24):
    a = np.linspace(math.radians(a0), math.radians(a1), n)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


# Block letters in letter units (height 1). Each letter: (width, [polyline, ...]).
LETTERS = {
    "R": (0.7, [np.array([[0, 0], [0, 1]]),
                np.vstack([[[0, 1], [0.4, 1]], _arc(0.4, 0.75, 0.25, 90, -90), [[0, 0.5]]]),
                np.array([[0.25, 0.5], [0.7, 0]])]),
    "O": (0.7, [np.c_[0.35 + 0.35 * np.cos(np.linspace(0, 2 * np.pi, 64)),
                      0.5 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, 64))]]),
    "B": (0.65, [np.array([[0, 0], [0, 1]]),
                 np.vstack([[[0, 1], [0.35, 1]], _arc(0.35, 0.75, 0.25, 90, -90), [[0, 0.5]]]),
                 np.vstack([[[0, 0.5], [0.4, 0.5]], _arc(0.4, 0.25, 0.25, 90, -90), [[0, 0]]])]),
    "T": (0.7, [np.array([[0, 1], [0.7, 1]]), np.array([[0.35, 1], [0.35, 0]])]),
}


def word_polylines(word=WORD, h=LETTER_H, gap=GAP):
    """Polylines of the word in board coordinates (u to the robot's right, v up), centred at 0."""
    total = sum(LETTERS[c][0] * h for c in word) + gap * (len(word) - 1)
    u0, out = -total / 2, []
    for c in word:
        w, strokes = LETTERS[c]
        for s in strokes:
            out.append(np.c_[u0 + s[:, 0] * h, (s[:, 1] - 0.5) * h])
        u0 += w * h + gap
    return out, total


def sample_polylines(polys, step=0.002):
    pts = []
    for p in polys:
        for a, b in zip(p[:-1], p[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            pts.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
        pts.append(p[-1:])
    return np.vstack(pts)


class WriteBoard(SingleArmEnv):
    """Write "ROBOT" on a whiteboard lying on the table with a grasped pen."""

    def __init__(self, robots, seed=0, table_full_size=(0.8, 1.2, 0.05), **kwargs):
        self.table_full_size = table_full_size
        self.table_offset = np.array((0.0, 0.0, 0.907))
        self.rng = np.random.default_rng(seed)
        self.polys, self.word_w = word_polylines()
        self.template = sample_polylines(self.polys)
        self.stroke_id = np.concatenate([np.full(len(sample_polylines([p])), i) for i, p in enumerate(self.polys)])
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
        A.append(ET.Element("material", name="whiteboard", rgba="0.97 0.97 0.96 1", specular="0.05", shininess="0.0", emission="0.55"))
        A.append(ET.Element("material", name="pen_body", rgba="0.15 0.15 0.18 1", specular="0.3"))
        # board (body origin = centre of the writing face); collides with the pen shaft (group 8) and the
        # robot; the felt tip contacts it only through the explicit pair below (compliant felt).
        board = (
            '<body name="board" pos="0 0 1">'
            '<geom name="board" type="box" size="0.01 0.35 0.225" pos="0.01 0 0" material="whiteboard" '
            'friction="0.3 0.005 0.0001" contype="1" conaffinity="9" group="1"/>'
            '<geom name="board_frame" type="box" size="0.012 0.36 0.235" pos="0.014 0 0" rgba="0.55 0.56 0.6 1" '
            'contype="0" conaffinity="0" group="1"/>'
            '</body>')
        stand = ('<body name="board_stand" pos="0 0 1">'
                 '<geom name="stand_post" type="box" size="0.02 0.02 0.2" rgba="0.4 0.4 0.45 1" contype="0" conaffinity="0" group="1"/>'
                 '<geom name="stand_foot" type="box" size="0.12 0.2 0.01" rgba="0.4 0.4 0.45 1" contype="0" conaffinity="0" group="1"/>'
                 '</body>')
        # ink pool: flat disks (axis along X), hidden inside the board until used
        dots = "".join(
            f'<geom name="ink{i}" type="cylinder" size="{DOT_R} 0.0001" pos="0.01 0 0" euler="0 1.5707963 0" '
            f'rgba="0.05 0.08 0.35 0" contype="0" conaffinity="0" group="1"/>' for i in range(N_DOTS))
        ink = f'<body name="ink" pos="0 0 1">{dots}</body>'
        m, h = 0.015, PEN_L
        pen = (
            '<body name="pen" pos="0 0 1.5"><freejoint name="pen_joint"/>'
            f'<inertial pos="0 0 {h / 2}" mass="{m}" diaginertia="{m * (3 * PEN_R**2 + h**2) / 12} {m * (3 * PEN_R**2 + h**2) / 12} {m * PEN_R**2 / 2}"/>'
            # shaft (back end at z=0, tip at z=PEN_L); a conical nib leads to the felt tip
            f'<geom name="pen_shaft" type="capsule" size="{PEN_R} {(h - 0.02) / 2 - PEN_R}" pos="0 0 {(h - 0.02) / 2}" '
            'material="pen_body" friction="0.75 0.02 0.0001" condim="4" contype="8" conaffinity="8" group="1"/>'
            f'<geom name="pen_nib" type="cylinder" size="0.004 0.008" pos="0 0 {h - 0.012}" rgba="0.85 0.85 0.85 1" '
            'contype="0" conaffinity="0" group="1"/>'
            f'<geom name="pen_tip" type="sphere" size="{TIP_R}" pos="0 0 {h - TIP_R}" rgba="0.05 0.08 0.35 1" '
            'contype="32" conaffinity="0" group="1"/>'
            # grasp patch (pads only, group 16), as for the Factory peg
            f'<geom name="pen_grip" type="box" size="{PEN_R} {PEN_R * 0.75} 0.009" pos="0 0 {PEN_L - TIP_OUT - 0.009}" '
            'friction="0.75 0.1 0.01" solref="0.002 1" solimp="0.95 0.99 0.0005" contype="16" conaffinity="0" '
            'condim="6" rgba="0 0 0 0" group="3"/>'
            '</body>')
        for x in (board, stand, ink, pen):
            self.model.worldbody.append(ET.fromstring(x))
        # felt tip on the whiteboard: compliant (~1 mm at 3 N), low friction
        contact = self.model.root.find("contact")
        if contact is None:
            contact = ET.SubElement(self.model.root, "contact")
        contact.append(ET.Element("pair", geom1="pen_tip", geom2="board", condim="3",
                                  friction="0.3 0.3 0.005 0.0001 0.0001", solref="-20000 -200"))
        for name in ("write_head", "write_side"):
            self.model.worldbody.append(ET.Element("camera", name=name, mode="fixed", pos="0 0 1",
                                                   xyaxes="1 0 0 0 0 1", fovy="45"))

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model._model
        self.board_body = m.body("board").id
        self.board_geom = m.geom("board").id
        self.pen_body = m.body("pen").id
        self.tip_geom = m.geom("pen_tip").id
        self.pen_qpos = m.joint("pen_joint").qposadr[0]
        self.pen_dof = m.joint("pen_joint").dofadr[0]
        self.dot_geoms = np.array([m.geom(f"ink{i}").id for i in range(N_DOTS)])
        self.ink_body, self.stand_body = m.body("ink").id, m.body("board_stand").id
        self.stand_post, self.stand_foot = m.geom("stand_post").id, m.geom("stand_foot").id
        self.head_cam, self.side_cam = m.camera("write_head").id, m.camera("write_side").id
        # gripper as in the Factory re-host: solid fingers/pads against the pen (8), pads on its grasp
        # patch (16), rubber pads with high friction
        for g in ("gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"):
            m.geom_conaffinity[m.geom(g).id] |= 16 | 8
            m.geom_friction[m.geom(g).id] = [3.0, 0.1, 0.01]
            m.geom_condim[m.geom(g).id] = 6
        for g in ("gripper0_finger1_collision", "gripper0_finger2_collision", "gripper0_hand_collision"):
            m.geom_conaffinity[m.geom(g).id] |= 8
        for g in ("gripper0_finger1_collision", "gripper0_finger2_collision"):
            m.geom_friction[m.geom(g).id][0] = 2.0
        for g in range(m.ngeom):
            if m.geom(g).name.startswith("table"):
                m.geom_conaffinity[g] |= 8

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
        # board frame: origin = a point on the face (nothing is drawn there); x = into the board (world -Z), y = world +Y,
        # z = up along the board (world +X, away from the robot)
        self.Rb = R.from_euler("y", TILT_DEG, degrees=True).as_matrix()
        self.box_c = base + np.array([0.55 + u(0.01), u(0.03), 0.0])
        self.box_c[2] = self.table_offset[2] + BOARD_T
        qb = R.from_matrix(self.Rb).as_quat()[[3, 0, 1, 2]]
        for b in (self.board_body, self.ink_body):
            m.body_pos[b], m.body_quat[b] = self.box_c, qb
        # no stand: the board lies on the table (visual-only stand geoms hidden)
        m.geom_rgba[[self.stand_post, self.stand_foot], 3] = 0.0
        # ink pool hidden behind the face
        m.geom_pos[self.dot_geoms] = [0.01, 0.0, 0.0]
        m.geom_rgba[self.dot_geoms, 3] = 0.0
        self.ink_pts, self.ink_cells = [], set()
        self.tip_force, self.tip_force_peak, self.crushed = 0.0, 0.0, False
        self._state, self._state_n, self._busy = None, -1, False
        # cameras: head in front-left of the board, above, looking back at the frame origin (as the Factory task_head);
        # side view along the board face
        m.cam_pos[self.head_cam] = self.box_c + np.array([0.30, 0.35, 0.40])
        m.cam_quat[self.head_cam] = look_at_quat(m.cam_pos[self.head_cam], self.box_c)
        m.cam_pos[self.side_cam] = self.box_c + self.Rb @ np.array([-0.10, -0.42, 0.05])
        m.cam_quat[self.side_cam] = look_at_quat(m.cam_pos[self.side_cam], self.box_c + self.Rb @ np.array([-0.05, 0, 0]))
        mujoco.mj_forward(m, d)
        # hand: tool +Z = into the board, tool +X = world +Y, tool +Y = up along the board
        rotm = self.Rb @ np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        robot = self.robots[0]
        qidx = np.array(robot._ref_joint_pos_indexes)
        q_seed = np.array([0.0, 0.2, 0.0, -2.2, 0.0, 2.4, math.pi / 4])     # hand pointing down
        site = m.site("gripper0_grip_site").id
        for _ in range(20):
            d.qpos[qidx] = q_seed
            mujoco.mj_forward(m, d)
            end = self._finger_end_offset(d.site_xpos[site], d.site_xmat[site].reshape(3, 3)[:, 2])
            tip_local = np.array([-(0.04 + u(0.01)), u(0.02), u(0.02)])
            tip_target = self.box_c + self.Rb @ tip_local
            target = tip_target - rotm[:, 2] * (end + TIP_OUT)
            ep, er = self._ik(target, rotm)
            if ep < 1e-3 and er < 1e-3:
                break
        tool = d.site_xmat[site].reshape(3, 3).copy()
        end = self._finger_end_offset(d.site_xpos[site], tool[:, 2])
        tip = d.site_xpos[site] + tool[:, 2] * (end + TIP_OUT)
        pen_pos = tip - tool[:, 2] * PEN_L                   # pen frame: z along the pen, origin at the back
        d.qpos[self.pen_qpos:self.pen_qpos + 3] = pen_pos
        d.qpos[self.pen_qpos + 3:self.pen_qpos + 7] = R.from_matrix(tool).as_quat()[[3, 0, 1, 2]]
        d.qvel[self.pen_dof:self.pen_dof + 6] = 0
        # fingers start just around the pen (they close horizontally, so the pen must not drop through
        # the finger gap before the grasp is established)
        gq = np.array(robot._ref_gripper_joint_pos_indexes)
        d.qpos[gq] = [PEN_R + 0.0008, -(PEN_R + 0.0008)]
        mujoco.mj_forward(m, d)
        self._pen_pose0 = d.qpos[self.pen_qpos:self.pen_qpos + 7].copy()
        self.write_init = {"box_centre": self.box_c.tolist(), "tilt_deg": TILT_DEG, "tip_local": tip_local.tolist(),
                           "ik_err": [ep, er], "q0": d.qpos[qidx].round(4).tolist()}

    def pin_pen(self):
        """Setup only: keep the pen at its in-hand pose while the fingers close (they close sideways,
        so gravity would otherwise pull the pen out through the finger gap during the ~150 ms close)."""
        d = self.sim.data._data
        d.qpos[self.pen_qpos:self.pen_qpos + 7] = self._pen_pose0
        d.qvel[self.pen_dof:self.pen_dof + 6] = 0

    # ---- ink (called every control tick by the simulator) ------------------------------------
    def ink_tick(self):
        m, d = self.sim.model._model, self.sim.data._data
        fn, pts = 0.0, []
        f6 = np.zeros(6)
        con = d.contact                      # read as arrays: d.contact[i].pos is unreliable in this binding
        g1, g2 = con.geom1[:d.ncon], con.geom2[:d.ncon]
        for i in np.flatnonzero(((g1 == self.tip_geom) & (g2 == self.board_geom)) |
                                ((g2 == self.tip_geom) & (g1 == self.board_geom))):
            mujoco.mj_contactForce(m, d, int(i), f6)
            fn += abs(f6[0])
            pts.append(con.pos[i].copy())
        self.tip_force = float(fn)
        self.tip_force_peak = max(self.tip_force_peak, float(fn))
        if fn > F_CRUSH:
            self.crushed = True
        if fn < F_INK or not pts or len(self.ink_pts) >= N_DOTS:
            return
        loc = self.Rb.T @ (np.mean(pts, axis=0) - self.box_c)      # board frame (y, z on the face)
        cell = (int(round(loc[1] / DOT_CELL)), int(round(loc[2] / DOT_CELL)))
        if cell in self.ink_cells:
            return
        self.ink_cells.add(cell)
        self.place_dot(len(self.ink_pts), loc[1], loc[2])
        self.ink_pts.append((float(loc[1]), float(loc[2])))

    def place_dot(self, k, y, z):
        """Show ink dot k at board-frame (y, z) on the face."""
        m = self.sim.model._model
        g = self.dot_geoms[k]
        m.geom_pos[g] = [-0.00015, y, z]
        m.geom_rgba[g][3] = 1.0

    # ---- evaluation -----------------------------------------------------------------------
    def _uv(self, yz):
        """Board-frame (y, z) -> word coordinates (u to the robot's right = -y, v up the board)."""
        yz = np.asarray(yz, float).reshape(-1, 2)
        return np.c_[-yz[:, 0], yz[:, 1]]

    def _score(self, tol=None, ink_pts=None):
        """Best alignment (translation anywhere on the board, scale 0.85-1.15) of the target word to the ink; vectorized.
        The word may sit anywhere. The search is centred on the shift that
        matches the template centroid to the ink centroid and covers +- 40 mm around it (coarse 2.5 mm), which is
        wider than the centroid error of a partially drawn word; the raster covers ink and template extents.
        The success rule holds if SOME allowed alignment gives every stroke recall >= 0.8 and precision >= 0.8, so
        the alignment maximises min(worst-stroke recall, precision) (ties: recall + precision). Coarse grid
        (2.5 mm, 5 scales), then refined around the 5 best coarse alignments (0.5 mm, scale step 0.01).
        A search that stops at the coarse grid and picks the alignment by recall + precision
        can miss an alignment that satisfies the rule (up to ~4 mm misfit at the word ends)."""
        tol = TOL if tol is None else tol
        pts = self.ink_pts if ink_pts is None else ink_pts
        if len(pts) == 0:
            return {"recall": 0.0, "precision": 0.0, "min_stroke_recall": 0.0, "shift_mm": [0.0, 0.0], "scale": 1.0}
        ink = self._uv(pts)
        res, pad, half = 0.001, 0.08, 0.04
        c0 = ink.mean(0) - self.template.mean(0)                                    # centroid-matching shift
        # raster must hold the ink (looked up as template * sc + shift) and the unshifted template (looked up as
        # ink - shift); points outside it read as misses
        lo = np.minimum(ink.min(0), self.template.min(0) * 1.15) - pad
        hi = np.maximum(ink.max(0), self.template.max(0) * 1.15) + pad
        shape = tuple(int(v) for v in np.ceil((hi - lo) / res) + 1)

        def edt(p):
            occ = np.ones(shape, bool)
            ij = np.round((p - lo) / res).astype(int)
            ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
            occ[ij[ok, 0], ij[ok, 1]] = False
            return ndimage.distance_transform_edt(occ) * res

        def lookup(dmap, p):                  # p: (..., 2) -> distances (inf outside the raster)
            ij = np.round((p - lo) / res).astype(int)
            ok = (ij >= 0).all(-1) & (ij[..., 0] < shape[0]) & (ij[..., 1] < shape[1])
            out = np.full(p.shape[:-1], np.inf)
            out[ok] = dmap[ij[ok][:, 0], ij[ok][:, 1]]
            return out

        d_ink = edt(ink)
        onehot = np.eye(int(self.stroke_id.max()) + 1)[self.stroke_id]                 # (Nt, strokes)
        n_stroke = onehot.sum(0)

        def evaluate(sc, shifts):             # -> list of (key, recall, precision, worst stroke, shift, scale)
            if ink_pts is not None:
                time.sleep(0.002)             # background thread: yield the GIL to the physics thread
            tpl = self.template * sc
            hit = lookup(d_ink, tpl[None] + shifts[:, None]) <= tol                    # (S, Nt)
            pre = (lookup(edt(tpl), ink[None] - shifts[:, None]) <= tol).mean(1)       # (S,)
            rec = hit.mean(1)
            worst = (hit @ onehot / n_stroke).min(1)
            return [((min(worst[k], pre[k]), rec[k] + pre[k]), float(rec[k]), float(pre[k]), float(worst[k]),
                     shifts[k], float(sc)) for k in range(len(shifts))]

        g = np.arange(-half, half + 1e-9, 0.0025)
        coarse = []
        for sc in (0.85, 0.92, 1.0, 1.08, 1.15):
            coarse += evaluate(sc, c0 + np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2))
        coarse.sort(key=lambda c: c[0], reverse=True)
        best = coarse[0]
        f = np.arange(-0.0025, 0.00251, 0.0005)
        for c in coarse[:5]:
            xs, ys = (np.clip(c[4][i] + f, c0[i] - half, c0[i] + half) for i in (0, 1))
            shifts = np.unique(np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2).round(4), axis=0)
            for sc in np.unique(np.clip(np.round(c[5] + np.arange(-0.04, 0.0401, 0.01), 2), 0.85, 1.15)):
                best = max([best] + evaluate(sc, shifts), key=lambda c: c[0])
        return {"recall": best[1], "precision": best[2], "min_stroke_recall": best[3],
                "shift_mm": (np.asarray(best[4]) * 1e3).round(1).tolist(), "scale": best[5]}

    def _bg_score(self, pts, n):
        try:
            self._state, self._state_n = self._score(ink_pts=pts), n
        finally:
            self._busy = False

    def write_state(self):
        """Task progress. The alignment score runs in a background thread whenever new ink appeared
        (never in the physics thread, which must keep real-time feedback); success needs an up-to-date
        score, the pen off the board and an intact tip."""
        n = len(self.ink_pts)
        if self._state is None:
            self._state, self._state_n, self._busy, self._state_t = self._score(ink_pts=[]), 0, False, -1e9
        t = float(self.sim.data.time)
        if n != self._state_n and not self._busy and t - self._state_t >= 1.0:   # at most once per sim second
            self._busy, self._state_t = True, t
            threading.Thread(target=self._bg_score, args=(list(self.ink_pts), n), daemon=True).start()
        sc = self._state
        touching = bool(self.tip_force > 0.05)
        ok = sc["min_stroke_recall"] >= THRESH and sc["precision"] >= THRESH and not touching and not self.crushed \
            and n == self._state_n
        return {"ink_dots": n, "recall": round(sc["recall"], 3), "min_stroke_recall": round(sc["min_stroke_recall"], 3),
                "precision": round(sc["precision"], 3),
                "align_shift_mm": sc["shift_mm"], "align_scale": sc["scale"], "tip_force_n": round(self.tip_force, 2),
                "tip_force_peak_n": round(self.tip_force_peak, 2), "tip_crushed": bool(self.crushed),
                "pen_on_board": touching, "success": bool(ok)}

    def _check_success(self):
        return self.write_state()["success"]

    def reward(self, action=None):
        return float(self._check_success())
