"""PlaceBowl: put the bowl on the plate, a pick-and-place reference task built in robosuite.

The scene uses:
  table   robosuite TableArena, top at 0.907 m (the Pivot / Stack height, 5 mm below the robot base), 0.8 x 0.8 m
  objects LIBERO's own meshes for the bowl (akita_black_bowl) and the plate (libero.libero.envs.objects)
  layout  side by side in front of the robot (bowl on the robot's left, +y; plate on its right, -y), positions and
          yaw sampled per seed (robosuite's UniformRandomSampler, global numpy RNG seeded with the seed, as Stack)
  cameras robosuite's agentview / sideview / eye-in-hand (as Wipe and Stack) + the baselines' sh_front / sh_wrist
Success (LIBERO's "On" predicate, as in libero_goal): the bowl touches the plate, its centre is above the plate and
within the plate's radius of the plate centre, and the gripper does not hold it.
Objects when LIBERO is not installed: the same robosuite object
class LIBERO builds (GoogleScannedObject) from byte-identical copies of its two asset folders in assets/objects;
the compiled scene is identical. With LIBERO installed, LIBERO itself is used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from robosuite.environments.manipulation.stack import Stack
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

TABLE_Z = 0.907
BOWL_X, BOWL_Y = (-0.06, 0.04), (0.10, 0.16)      # table-centre frame (robot base at x = -0.56)
PLATE_X, PLATE_Y = (-0.06, 0.04), (-0.16, -0.08)
PLATE_RADIUS_OK = 0.06                              # bowl centre within this of the plate centre counts as "on"
VENDORED = Path(__file__).resolve().parent / "assets" / "objects"


def scanned_object(obj_name, name):
    """LIBERO's object obj_name (libero/libero/envs/objects/google_scanned_objects.py). LIBERO when installed; otherwise
    the identical class (MujocoXMLObject, free joint with damping 0.0005, obj_type "all", no duplicated collision geoms,
    LIBERO's rotation attributes) built from the copied asset folder."""
    try:
        from libero.libero.envs.objects import get_object_fn
    except ImportError:
        from robosuite.models.objects import MujocoXMLObject
        obj = MujocoXMLObject(str(VENDORED / obj_name / f"{obj_name}.xml"), name=name,
                              joints=[dict(type="free", damping="0.0005")], obj_type="all",
                              duplicate_collision_geoms=False)
        obj.category_name = obj_name
        obj.rotation, obj.rotation_axis = (np.pi / 2, np.pi / 2), "x"
        obj.object_properties = {"vis_site_names": {}}
        return obj
    return get_object_fn(obj_name)(name=name)


class PlaceBowl(Stack):
    """Stack's robot, table and controller setup with the bowl and plate instead of the cubes."""

    def _load_model(self):
        SingleArmEnv._load_model(self)
        self.robots[0].robot_model.set_base_xpos(
            self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0]))
        self.table_offset = np.array((0, 0, TABLE_Z))
        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        self.bowl = scanned_object("akita_black_bowl", "bowl")
        self.plate = scanned_object("plate", "plate")
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(UniformRandomSampler(
            name="PlateSampler", mujoco_objects=self.plate, x_range=list(PLATE_X), y_range=list(PLATE_Y),
            rotation=(-np.pi, np.pi), ensure_object_boundary_in_range=False, ensure_valid_placement=True,
            reference_pos=self.table_offset, z_offset=0.01))
        self.placement_initializer.append_sampler(UniformRandomSampler(
            name="BowlSampler", mujoco_objects=self.bowl, x_range=list(BOWL_X), y_range=list(BOWL_Y),
            rotation=(-np.pi, np.pi), ensure_object_boundary_in_range=False, ensure_valid_placement=True,
            reference_pos=self.table_offset, z_offset=0.01))
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots],
                                      mujoco_objects=[self.plate, self.bowl])

    def _setup_references(self):
        SingleArmEnv._setup_references(self)
        self.bowl_body_id = self.sim.model.body_name2id(self.bowl.root_body)
        self.plate_body_id = self.sim.model.body_name2id(self.plate.root_body)

    def _setup_observables(self):
        return SingleArmEnv._setup_observables(self)

    def _reset_internal(self):
        SingleArmEnv._reset_internal(self)
        if not self.deterministic_reset:
            for obj_pos, obj_quat, obj in self.placement_initializer.sample().values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _contact(self, a, b):
        m, d = self.sim.model._model, self.sim.data._data
        ga = {g for g in range(m.ngeom) if m.geom_bodyid[g] == a or self._is_child(m.geom_bodyid[g], a)}
        gb = {g for g in range(m.ngeom) if m.geom_bodyid[g] == b or self._is_child(m.geom_bodyid[g], b)}
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in ga and c.geom2 in gb) or (c.geom1 in gb and c.geom2 in ga):
                return True
        return False

    def _is_child(self, body, root):
        m = self.sim.model._model
        while body > 0:
            if body == root:
                return True
            body = m.body_parentid[body]
        return False

    def place_state(self):
        d = self.sim.data._data
        pb, pp = d.xpos[self.bowl_body_id], d.xpos[self.plate_body_id]
        touching = self._contact(self.bowl_body_id, self.plate_body_id)
        held = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.bowl)
        xy = float(np.linalg.norm(pb[:2] - pp[:2]))
        ok = touching and pb[2] > pp[2] and xy < PLATE_RADIUS_OK and not held
        return {"bowl_above_plate_m": round(float(pb[2] - pp[2]), 4), "xy_offset_m": round(xy, 4),
                "touching": bool(touching), "held": bool(held), "success": bool(ok)}

    def _check_success(self):
        return self.place_state()["success"]

    def reward(self, action=None):
        return 0.0
