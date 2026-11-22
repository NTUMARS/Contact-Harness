"""Offline frame checks. No servers, sockets, cameras or robot motion."""
from copy import deepcopy
from pathlib import Path
import re
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation as R

import motion_server as motion
import observation_server as obs
import get_observation as client


class FrameConventions(unittest.TestCase):
    def setUp(self):
        self.config = obs.load_config()
        self.cfg = motion.Settings()
        self.cfg.validate()

    def plan(self, payload, start, cfg=None):
        cfg = cfg or self.cfg
        return motion.plan_command(motion.validate_command(payload, cfg), start, cfg)

    def test_default_world_base_and_status_agree(self):
        np.testing.assert_array_equal(self.config["world_from_base"], np.eye(3))
        np.testing.assert_array_equal(self.cfg.world_from_base, np.eye(3))
        np.testing.assert_array_equal(motion.MotionService(self.cfg).status()["world_from_base"], np.eye(3))
        np.testing.assert_array_equal(obs.ObservationService(self.config).status()["wrench"]["world_from_base"], np.eye(3))

    def test_cli_default_and_explicit_world_rotation_without_binding(self):
        class BeforeBind(Exception):
            pass
        for args, expected in (([], np.eye(3)), (
                ["--world-from-base", "0", "-1", "0", "1", "0", "0", "0", "0", "1"],
                np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]))):
            with self.subTest(args=args):
                constructor = motion.Settings
                with patch.object(motion, "Settings", wraps=constructor) as settings:
                    # argparse accesses scalar class defaults before constructing Settings.
                    settings.address = constructor.address
                    settings.mode = constructor.mode
                    settings.soft_force_n = constructor.soft_force_n
                    settings.hard_force_n = constructor.hard_force_n
                    with patch.object(motion, "ThreadingHTTPServer", side_effect=BeforeBind):
                        with self.assertRaises(BeforeBind):
                            motion.main(args)
                    np.testing.assert_array_equal(settings.call_args.kwargs["world_from_base"], expected)

    def test_world_and_base_delta_translation_and_rotation_match(self):
        start = np.array([.4, .1, .3, .4, -.3, .8])
        for axis in np.eye(3):
            for sign in (-1., 1.):
                payload = dict(type="delta", translation_m=(axis*.004*sign).tolist(),
                               rotation_vector_rad=(axis*.02*sign).tolist(), duration_s=1.)
                base = self.plan(dict(payload, frame="base"), start)
                world = self.plan(dict(payload, frame="world"), start)
                np.testing.assert_allclose(world.goal, base.goal, atol=1e-12)
                np.testing.assert_allclose(world.dp, base.dp, atol=1e-12)
                np.testing.assert_allclose(world.dr, base.dr, atol=1e-12)

    def test_absolute_pose_orientation_matches_in_world_and_base(self):
        start = np.array([.4, .1, .3, .4, -.3, .8])
        target = R.from_euler("xyz", [.45, -.28, .82])
        payload = dict(type="pose", position_m=[.41, .11, .31],
                       quaternion_xyzw=target.as_quat().tolist(), duration_s=2.)
        for frame in ("base", "world"):
            plan = self.plan(dict(payload, frame=frame), start)
            np.testing.assert_allclose(plan.goal[:3], payload["position_m"], atol=1e-12)
            np.testing.assert_allclose(R.from_euler("xyz", plan.goal[3:]).as_matrix(),
                                       target.as_matrix(), atol=1e-12)

    def test_tool_delta_uses_reference_orientation_and_base_rotation_premultiplication(self):
        start = np.array([.4, .1, .3, .3, -.4, .7])
        rotation = R.from_euler("xyz", start[3:])
        local_dp, local_dr = np.array([.004, 0., 0.]), np.array([0., .02, 0.])
        plan = self.plan(dict(type="delta", translation_m=local_dp.tolist(),
                             rotation_vector_rad=local_dr.tolist(), frame="tool", duration_s=1.), start)
        np.testing.assert_allclose(plan.goal[:3], start[:3] + rotation.apply(local_dp), atol=1e-12)
        expected = rotation * R.from_rotvec(local_dr)
        np.testing.assert_allclose(R.from_euler("xyz", plan.goal[3:]).as_matrix(),
                                   expected.as_matrix(), atol=1e-12)

    def test_wrench_base_world_and_motion_normalization_agree(self):
        config = deepcopy(self.config)
        config["ee_to_stiffness"] = [.03, -.02, .01, .2, -.3, .4]
        cfg = motion.Settings(ee_to_stiffness=np.array(config["ee_to_stiffness"]))
        backend = motion.RobotBackend(cfg)
        pose = np.array([.4, .1, .3, .4, -.6, .8])
        native = np.array([1., 2., -3., .2, -.4, .6])
        ee_k = motion.pose_T(config["ee_to_stiffness"])
        base_ee = R.from_euler("xyz", pose[3:]).as_matrix()
        force_ee = -ee_k[:3, :3] @ native[:3]
        moment_ee = -ee_k[:3, :3] @ native[3:] + np.cross(ee_k[:3, 3], force_ee)
        expected = np.r_[base_ee @ force_ee, base_ee @ moment_ee]
        raw = dict(pose=np.r_[pose[:3], R.from_euler("xyz", pose[3:]).as_quat()].tolist(),
                   q=[0.]*7, dq=[0.]*7, wrench_K=native.tolist(),
                   state_age_s=0., _received=time.monotonic(), robot_time_s=1.,
                   _controller=dict(target_pose=[0.]*7, queue_depth=0,
                                    applied_actions=0, published_count=1))
        try:
            normalized = backend._normalize(raw)
        finally:
            backend.close()
        np.testing.assert_allclose(normalized["ext_wrench"], expected, atol=1e-12)
        for mode in ("raw", "tcp"):
            np.testing.assert_allclose(obs.wrench_values(native, pose, np.eye(4), config, mode),
                                       expected, atol=1e-12)

    def test_flange_reference_changes_moment_and_client_round_trip(self):
        pose = np.array([.4, .1, .3, .4, -.6, .8])
        tool = motion.pose_T([.02, -.03, .1, .1, .2, -.4])
        native = np.array([1., 2., -3., .2, -.4, .6])
        flange = obs.wrench_values(native, pose, tool, self.config, "wrist")
        expected_force = tool[:3, :3] @ -native[:3]
        expected_moment = tool[:3, :3] @ -native[3:] + np.cross(tool[:3, 3], expected_force)
        np.testing.assert_allclose(flange, np.r_[expected_force, expected_moment], atol=1e-12)
        context = dict(flange=flange, tool_end=motion.T_pose(tool).tolist(),
                       tcp_pose_base=pose.tolist(), world_from_base=self.config["world_from_base"])
        expected = obs.wrench_values(native, pose, tool, self.config, "raw")
        for frame in ("base", "world"):
            np.testing.assert_allclose(client.transform_wrench(context, reference="tcp", frame=frame),
                                       expected, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "geometry is not configured"):
            obs.wrench_values(native, pose, tool, self.config, "tcp", "connector")

    def test_snapshot_force_torque_shapes_and_frames(self):
        service = obs.ObservationService(self.config)
        pose = np.array([.4, .1, .3, .4, -.6, .8])
        native = np.array([1., 2., -3., .2, -.4, .6])
        service.latest["state"] = dict(
            received_monotonic_s=time.monotonic(), errors=[], field_errors={},
            data=dict(eef_pose=pose.tolist(), joint_angles=[0.]*7),
            sample=dict(pose=pose, tool=np.eye(4)), corrected_wrench_K=native)
        for mode in ("raw", "tcp"):
            result, _ = service.snapshot(dict(observations=["force", "torque", "wrench", "eef_pose"], wrench=mode))
            self.assertFalse(result["errors"])
            values = result["observations"]
            self.assertEqual(len(values["force"]), 6)
            self.assertEqual(len(values["torque"]), 3)
            np.testing.assert_allclose(values["torque"], values["force"][3:])
            np.testing.assert_allclose(values["wrench"]["torque_nm"], values["torque"])
            self.assertEqual(values["wrench"]["reference"], "EE")
            self.assertEqual(values["wrench"]["frame"], "base" if mode == "raw" else "world")
            np.testing.assert_allclose(values["eef_pose"], pose)

    def test_camera_basis_and_rotation_examples_in_prompt(self):
        # User-supplied nominal orientation: optical Z down, image up base +X.
        forward, up = np.array([0., 0., -1.]), np.array([1., 0., 0.])
        right = np.cross(forward, up)
        np.testing.assert_array_equal(-right, [0., 1., 0.])
        camera = np.column_stack([right, -up, forward])
        np.testing.assert_allclose(camera.T @ camera, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(camera), 1.)
        text = Path("docs/task_prompt_cable_insertion.txt").read_text()
        for axis, expected in zip("XYZ", np.eye(3)):
            line = next(line for line in text.splitlines() if "World-" + axis + " robot torque error:" in line)
            command = re.search(r"--rotation-deg ([\d. -]+) --duration", line).group(1)
            np.testing.assert_allclose([float(v) for v in command.split()], expected)


if __name__ == "__main__":
    unittest.main()
