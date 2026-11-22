"""Offline soft-force checks: real planning/IK/step loop, no sockets or servers."""
import time
import unittest
from unittest.mock import patch
import numpy as np
import motion_server as motion


class FeedbackBackend:
    connected = True
    max_translation_increment = .0005
    max_rotation_increment = .01

    def __init__(self, force_at_step):
        self.kin = motion.Kinematics(motion.PANDA_ARM_URDF, np.eye(4))
        self.q = np.array([0., -.4, 0., -2., 0., 1.5, .8])
        self.pose = self.kin.fk(self.q)
        self.force_at_step = force_at_step
        self.sent = []

    def state(self):
        return dict(state_age_s=0., _received=time.monotonic(),
                    in_realtime_loop=True, fault=None,
                    joint_pos_real=self.q.tolist(), cart_pos_real=self.pose.tolist(),
                    ext_wrench=[*self.force_at_step(len(self.sent)), 0., 0., 0.])

    def send(self, candidate, q, dt, state):
        self.sent.append(candidate.copy())
        self.pose, self.q = candidate.copy(), q.copy()
        return self.state()


class SoftForceTruncation(unittest.TestCase):
    def service(self, force_at_step):
        service = motion.MotionService(motion.Settings())
        service.backend = FeedbackBackend(force_at_step)
        service.reference = service.backend.pose.copy()
        service.joint_reference = service.backend.q.copy()
        service.lower = service.backend.kin.lower + service.cfg.joint_margin_rad
        service.upper = service.backend.kin.upper - service.cfg.joint_margin_rad
        service._initialize_workspace(service.reference)
        return service

    def execute(self, service, translation, rotation=None):
        command = motion.validate_command(dict(
            type="delta", translation_m=translation,
            rotation_vector_rad=rotation or [0., 0., 0.],
            duration_s=.2, frame="base"), service.cfg)
        action = service._new_job([command])["actions"][0]
        with patch.object(service, "_wait_until"):
            service._execute(action)
        self.assertEqual(action["status"], "completed")
        self.assertEqual(action["steps_sent"], 10)
        return action

    def test_directional_rule_on_each_axis_and_sign(self):
        for axis in range(3):
            for sign in (-1., 1.):
                with self.subTest(axis=axis, sign=sign):
                    force = np.zeros(3)
                    force[axis] = sign * 15
                    into = np.full(3, .0001)
                    into[axis] = -sign * .0001
                    safe, blocked = motion.constrain_translation(into, force, 10)
                    expected = into.copy()
                    expected[axis] = 0
                    np.testing.assert_array_equal(safe, expected)
                    self.assertEqual(np.flatnonzero(blocked).tolist(), [axis])
                    away = -into
                    np.testing.assert_array_equal(
                        motion.constrain_translation(away, force, 10)[0], away)

    def test_all_blocked_steps_consumed_and_next_action_runs(self):
        service = self.service(lambda step: [-15., 0., 0.])
        start = service.reference.copy()
        action = self.execute(service, [.002, 0., 0.])
        self.assertEqual(action["clipped_steps"], 10)
        np.testing.assert_allclose(action["discarded_translation_m"], [.002, 0., 0.])
        np.testing.assert_allclose(service.backend.sent, np.tile(start, (10, 1)))
        self.assertIsNone(service.fault)
        retreat = self.execute(service, [-.002, 0., 0.])
        self.assertEqual(retreat["clipped_steps"], 0)
        np.testing.assert_allclose(service.reference[:3], start[:3] + [-.002, 0., 0.])

    def test_force_appears_then_clears_without_replaying_steps(self):
        service = self.service(lambda step: [-15., 0., 0.] if 3 <= step < 7 else [0., 0., 0.])
        start = service.reference.copy()
        action = self.execute(service, [.002, 0., 0.])
        self.assertEqual(action["clipped_steps"], 4)
        increments = np.diff(np.vstack([start, *service.backend.sent]), axis=0)
        np.testing.assert_allclose(increments[:, 0], [.0002]*3 + [0.]*4 + [.0002]*3, atol=1e-12)
        np.testing.assert_allclose(action["discarded_translation_m"], [.0008, 0., 0.])
        np.testing.assert_allclose(service.reference[:3], start[:3] + [.0012, 0., 0.])

    def test_unblocked_axis_and_rotation_continue(self):
        service = self.service(lambda step: [-15., 0., 0.])
        start = service.reference.copy()
        action = self.execute(service, [.001, .001, 0.], [0., 0., .001])
        self.assertEqual(action["clipped_steps"], 10)
        expected = motion.advance(start, np.array([0., .001, 0.]), np.array([0., 0., .001]))
        np.testing.assert_allclose(service.reference, expected, atol=1e-12)

    def test_force_bias_applied_before_directional_check(self):
        service = self.service(lambda step: [-11., 0., 0.])
        service.cfg.force_bias_n = np.array([-2., 0., 0.])
        action = self.execute(service, [.002, 0., 0.])
        self.assertEqual(action["clipped_steps"], 0)

    def test_hard_limit_still_rejects(self):
        service = self.service(lambda step: [-51., 0., 0.])
        with self.assertRaises(motion.MotionError) as raised:
            self.execute(service, [.002, 0., 0.])
        self.assertEqual(raised.exception.reason, "hard_force_limit")
        self.assertEqual(service.backend.sent, [])

    def test_default_hard_limit_and_force_magnitude_boundary(self):
        service = self.service(lambda step: [0., 0., 0.])
        self.assertEqual(service.cfg.hard_force_n, 50.)
        self.assertEqual(service.cfg.soft_force_n, 10.)
        for force, trips in (([21., 0., 0.], False), ([49.9, 0., 0.], False),
                             ([50., 0., 0.], True), ([30., 40., 0.], True)):
            with self.subTest(force=force):
                state = service.backend.state()
                state["ext_wrench"] = [*force, 0., 0., 0.]
                if trips:
                    with self.assertRaises(motion.MotionError) as raised:
                        service._hard(state)
                    self.assertEqual(raised.exception.reason, "hard_force_limit")
                else:
                    service._hard(state)

    def test_soft_targets_below_50_accepted(self):
        for soft in (10., 20., 25., 49.):
            motion.Settings(soft_force_n=soft).validate()
        for settings in (motion.Settings(soft_force_n=50.), motion.Settings(hard_force_n=51.)):
            with self.assertRaises(ValueError):
                settings.validate()

    def test_startup_uses_collision_threshold_not_contact_threshold(self):
        class ReachedFeedback(Exception):
            pass
        for server_limit in (20., 50.):
            backend = motion.RobotBackend(motion.Settings())
            info = dict(target_publish_hz=50., home={}, gripper={},
                        collision_force_threshold_n=server_limit,
                        collision_contact_threshold_n=20.)
            try:
                with patch.object(backend, "_info", return_value=info), \
                        patch.object(backend, "_raw", side_effect=ReachedFeedback):
                    if server_limit == 50.:
                        with self.assertRaises(ReachedFeedback):
                            backend.start()
                    else:
                        with self.assertRaises(motion.MotionError) as raised:
                            backend.start()
                        self.assertEqual(raised.exception.reason, "force_configuration_mismatch")
            finally:
                backend.close()



if __name__ == "__main__":
    unittest.main()
