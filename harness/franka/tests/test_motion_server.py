"""Offline integration checks against the unchanged downloaded controller.

Run locally: python -m unittest discover -s tests -v
No ROS or robot connection is used. The actual Flask routes and controller state
machines run over loopback; only the ROS hardware backend is simulated.
"""
import contextlib
import hashlib
import io
import json
import logging
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer

import numpy as np
from scipy.spatial.transform import Rotation as R
from flask import request
from werkzeug.serving import make_server

import franka_robot_server as controller
import motion_server as motion
import terminal_command as terminal


def wait_until(predicate, timeout=5.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.005)
    raise AssertionError("Timed out waiting for test condition")


class SimulatedCommands:
    def __init__(self, runtime):
        self.runtime = runtime
        self.gripper_active = False
        self.gripper_delay = .08
        self.fail_gripper = False
        self.fail_home = False
        self.widths = []
        self.grasp_parameters = []
        self.home_starts = 0

    def start_home(self):
        self.home_starts += 1
        if self.fail_home:
            raise RuntimeError("simulated home controller failure")

    def publish_home(self):
        self.runtime.q = np.array(controller.HOME_JOINT_TARGET)

    def restore_impedance(self):
        pass

    def start_gripper(self, width, speed, **kwargs):
        self.widths.append((width, speed))
        self.grasp_parameters.append(kwargs)
        self.gripper_active = True
        self.finish_at = time.monotonic() + self.gripper_delay

    def gripper_result(self, stopping=False):
        if time.monotonic() < self.finish_at:
            return None
        self.gripper_active = False
        return (False, "simulated blocked gripper") if self.fail_gripper else (True, None)

    def stop_gripper(self):
        self.finish_at = time.monotonic()


class Runtime:
    def __init__(self, held_offset=None, rpc_delay_s=0., initial_q=None):
        self.kin = motion.Kinematics(motion.PANDA_ARM_URDF, np.eye(4))
        self.q = np.array([0., -.4, 0., -2., 0., 1.5, .8] if initial_q is None else initial_q)
        self.force = np.zeros(6)
        self.core = controller.FrankaServer(collision_force_threshold=motion.Settings.hard_force_n)
        self.driver = SimulatedCommands(self)
        self.core.command_backend = self.driver
        self.paths = []
        self.action_requests = []
        self.response_delay = None
        self.loop_error = None
        self.follow_target = held_offset is None
        self.pause_feedback = False
        self.alive = threading.Event()
        self.feed()
        self.core.initialize_target_once()
        if held_offset is not None:
            self.core.target_pose[:3] += np.asarray(held_offset)
        app = controller.create_app(self.core)

        @app.before_request
        def record_request():
            if rpc_delay_s:
                time.sleep(rpc_delay_s)
            self.paths.append((request.method, request.path))
            if request.method == "POST" and request.path == "/actions":
                self.action_requests.append((time.monotonic(), request.get_json()))
            if request.headers.get("X-Motion-Owner") is not None:
                raise AssertionError("Unexpected adapter header")

        @app.after_request
        def delay_response(response):
            if self.response_delay is not None:
                delay = self.response_delay(request.method, request.path)
                if delay:
                    time.sleep(delay)
            return response

        self.robot_server = make_server("127.0.0.1", 0, app, threaded=True)
        self.service = motion.MotionService(motion.Settings(
            address="http://127.0.0.1:%d" % self.robot_server.server_port))
        self.local = ThreadingHTTPServer(("127.0.0.1", 0), motion.handler_for(self.service))
        self.client = terminal.MotionClient("http://127.0.0.1:%d" % self.local.server_port)

    def feed(self):
        pose = self.kin.fk(self.q)
        self.core.update_state(dict(
            pose=np.r_[pose[:3], R.from_euler("xyz", pose[3:]).as_quat()].tolist(),
            q=self.q.tolist(), dq=[0.]*7, robot_time_s=time.monotonic(),
            robot_mode=2, current_errors=[], wrench_K=self.force.tolist()))

    def tick(self):
        deadline = time.monotonic()
        while self.alive.is_set():
            try:
                if not self.pause_feedback:
                    self.feed()
                self.core.step_gripper()
                if not self.core.step_home():
                    # Match the fixed controller's atomic publish/info ordering.
                    with self.core.lock:
                        if not self.core.home_busy():
                            target = self.core.next_target()
                            self.core.note_publish()
                            pose = np.r_[target[:3], R.from_quat(target[3:]).as_euler("xyz")]
                            if self.follow_target:
                                self.q = self.kin.solve(pose, self.q, .02)
                deadline = controller.next_deadline(deadline, time.monotonic())
                time.sleep(max(0., deadline-time.monotonic()))
            except Exception as exc:
                self.loop_error = exc
                return

    def __enter__(self):
        self.alive.set()
        self.tick_thread = threading.Thread(target=self.tick, daemon=True)
        self.tick_thread.start()
        self.robot_server_thread = threading.Thread(target=self.robot_server.serve_forever, daemon=True)
        self.robot_server_thread.start()
        self.local_thread = threading.Thread(target=self.local.serve_forever, daemon=True)
        self.local_thread.start()
        self.service.start()
        wait_until(lambda: self.service.status()["ready"] or self.service.status()["fault"])
        if not self.service.status()["ready"]:
            self.__exit__(None, None, None)
            raise AssertionError(self.service.status()["fault"])
        return self

    def __exit__(self, *args):
        self.service.close()
        self.local.shutdown()
        self.local.server_close()
        self.robot_server.shutdown()
        self.robot_server.server_close()
        self.alive.clear()
        self.tick_thread.join(timeout=2)

    def cli(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = terminal.main(["--server", self.client_url, *args])
        return code, json.loads(output.getvalue())

    @property
    def client_url(self):
        return "http://127.0.0.1:%d" % self.local.server_port


class DirectControllerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        cls.controller_path = Path(controller.__file__)
        cls.original_hash = hashlib.sha256(cls.controller_path.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        assert hashlib.sha256(cls.controller_path.read_bytes()).hexdigest() == cls.original_hash

    def test_home_gripper_motion_sequence_through_terminal(self):
        with Runtime() as rt:
            code, job = rt.cli(
                "home", "--next", "gripper", "--width-m", ".04", "--speed-m-s", ".05",
                "--next", "delta", "--translation-m", ".001", "0", "0",
                "--duration", ".2", "--frame", "base", "--wait")
            self.assertEqual(code, 0, job)
            self.assertEqual(job["status"], "completed", job)
            self.assertEqual(job["actions_completed"], 3)
            self.assertEqual(rt.driver.home_starts, 1)
            self.assertEqual(rt.driver.widths, [(.04, .05)])
            self.assertEqual(rt.core.applied_actions, 10)
            self.assertLess(job["actions"][2]["execution_elapsed_s"], .35)
            self.assertTrue(rt.service.status()["homed"])
            self.assertEqual(job["actions"][0]["completion_basis"], "native_home_result")
            self.assertEqual(job["actions"][1]["completion_basis"], "native_gripper_result")
            self.assertIsNone(rt.loop_error)
            self.assertLessEqual(set(path for _, path in rt.paths),
                                 {"/info", "/getstate", "/actions", "/actions/clear", "/home", "/gripper"})

    def test_force_grasp_and_release_through_terminal(self):
        with Runtime() as rt:
            code, job = rt.cli("gripper", "--position", "0", "--force-n", "30", "--wait")
            self.assertEqual(code, 0, job)
            op = job["actions"][0]["controller_operation"]
            self.assertEqual(op["mode"], "grasp")
            self.assertEqual(op["force_n"], 30)
            self.assertEqual(rt.driver.grasp_parameters[-1], dict(force_n=30, epsilon_inner_m=.005, epsilon_outer_m=.08))
            code, job = rt.cli("gripper", "--position", "1", "--force-n", "0", "--wait")
            self.assertEqual(code, 0, job)
            self.assertEqual(job["actions"][0]["controller_operation"]["mode"], "move")
            self.assertEqual(rt.driver.grasp_parameters[-1], {})

    def test_invalid_grasp_force_rejected_before_hardware(self):
        with Runtime() as rt:
            for force in (-1, 20, 71, True, float("nan")):
                with self.subTest(force=force):
                    with self.assertRaises(Exception):
                        rt.client.submit({"type": "gripper", "width_m": 0., "force_n": force})
            self.assertEqual(rt.driver.widths, [])

    def test_normalized_gripper_and_explicit_native_failure(self):
        with Runtime() as rt:
            code, job = rt.cli("gripper", "--position", ".5", "--wait")
            self.assertEqual(code, 0, job)
            self.assertEqual(rt.driver.widths[-1][0], .04)
            rt.driver.fail_gripper = True
            submitted = rt.client.submit([
                {"type": "gripper", "width_m": .02},
                {"type": "delta", "translation_m": [.001, 0, 0], "duration_s": .2}])
            result = rt.client.wait_for_command(submitted["id"], 5)
            self.assertEqual(result["status"], "failed", result)
            self.assertEqual(result["reason"], "gripper_failed")
            self.assertEqual(result["actions"][1]["status"], "cancelled")
            self.assertEqual(rt.core.accepted_actions, 0)

    def test_stop_waits_for_native_gripper_then_recover(self):
        with Runtime() as rt:
            rt.driver.gripper_delay = .3
            job = rt.client.submit([
                {"type": "gripper", "width_m": .04},
                {"type": "delta", "translation_m": [.001, 0, 0], "duration_s": .2}])
            wait_until(lambda: rt.core.command_status("gripper")["state"] == "moving")
            rt.client.stop()
            time.sleep(.04)
            self.assertTrue(rt.service.status()["stop_requested"])
            result = rt.client.wait_for_command(job["id"], 5)
            self.assertEqual(result["status"], "cancelled", result)
            wait_until(lambda: not rt.service.status()["stop_requested"])
            self.assertTrue(rt.service.status()["last_result"]["confirmed"])
            self.assertEqual(rt.core.accepted_actions, 0)
            rt.client.recover()
            wait_until(lambda: rt.service.status()["ready"])
            self.assertTrue(rt.service.status()["ready"])

    def test_stop_during_home_waits_and_discards_following_gripper(self):
        with Runtime() as rt:
            job = rt.client.submit([{"type": "home"}, {"type": "gripper", "width_m": .04}])
            wait_until(lambda: rt.core.home_busy())
            rt.client.stop()
            result = rt.client.wait_for_command(job["id"], 5)
            self.assertEqual(result["status"], "cancelled", result)
            self.assertEqual(rt.core.command_status("home")["state"], "succeeded")
            self.assertEqual(rt.driver.widths, [])
            wait_until(lambda: not rt.service.status()["stop_requested"])

    def test_soft_force_clips_without_replaying_discarded_motion(self):
        with Runtime() as rt:
            base_ee = R.from_euler("xyz", rt.kin.fk(rt.q)[3:]).as_matrix()
            rt.force[:3] = base_ee.T @ np.array([15., 0., 0.])
            time.sleep(.04)
            start = rt.core.target_pose.copy()
            code, result = rt.cli("delta", "--translation-m", ".001", "0", "0",
                                  "--duration", ".2", "--wait")
            self.assertEqual(code, 0, result)
            self.assertEqual(result["actions"][0]["clipped_steps"], 10)
            np.testing.assert_allclose(rt.core.target_pose, start, atol=1e-9)

    def test_home_preempts_cartesian_motion(self):
        with Runtime() as rt:
            previous = rt.client.submit({"type": "delta", "translation_m": [.005, 0, 0],
                                         "duration_s": 1.})
            wait_until(lambda: rt.core.applied_actions >= 2)
            home = rt.client.submit({"type": "home"})
            result = rt.client.wait_for_command(home["id"], 5)
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(rt.client.status(previous["id"])["status"], "cancelled")
            self.assertEqual(rt.driver.home_starts, 1)
            self.assertTrue(rt.service.status()["ready"])

    def test_home_failure_never_runs_following_motion(self):
        with Runtime() as rt:
            rt.driver.fail_home = True
            job = rt.client.submit([
                {"type": "home"},
                {"type": "delta", "translation_m": [.001, 0, 0], "duration_s": .2}])
            result = rt.client.wait_for_command(job["id"], 5)
            self.assertEqual(result["status"], "failed", result)
            self.assertIn("simulated home controller failure", result["detail"])
            self.assertEqual(result["actions"][1]["status"], "cancelled")
            self.assertEqual(rt.core.accepted_actions, 0)
            self.assertFalse(rt.service.status()["ready"])

    def test_hard_force_discards_sequence_and_queued_work(self):
        with Runtime() as rt:
            job = rt.client.submit([
                {"type": "delta", "translation_m": [.005, 0, 0], "duration_s": 1.},
                {"type": "gripper", "width_m": .04}])
            wait_until(lambda: rt.core.applied_actions >= 2)
            queued = rt.client.submit({"type": "gripper", "width_m": .02})
            rt.force[:3] = [51., 0., 0.]
            result = rt.client.wait_for_command(job["id"], 5)
            self.assertEqual(result["reason"], "hard_force_limit", result)
            self.assertEqual(result["actions"][1]["status"], "cancelled")
            self.assertEqual(rt.client.status(queued["id"])["status"], "cancelled")
            self.assertEqual(rt.driver.widths, [])
            self.assertLess(rt.core.applied_actions, 50)
            self.assertFalse(rt.service.status()["ready"])

    def test_stale_feedback_stops_before_remaining_steps(self):
        with Runtime() as rt:
            job = rt.client.submit({"type": "delta", "translation_m": [.005, 0, 0],
                                    "duration_s": 1.})
            wait_until(lambda: rt.core.applied_actions >= 2)
            rt.pause_feedback = True
            result = rt.client.wait_for_command(job["id"], 5)
            self.assertEqual(result["reason"], "stale_feedback", result)
            self.assertLess(rt.core.applied_actions, 50)
            self.assertFalse(rt.service.status()["ready"])

    def test_workspace_preflight_rejects_without_sending_any_steps(self):
        with Runtime() as rt:
            with rt.service.lock:
                rt.service.workspace_max[0] = rt.service.reference[0]+.0005
            job = rt.client.submit({"type": "delta", "translation_m": [.001, 0, 0],
                                    "duration_s": .2})
            result = rt.client.wait_for_command(job["id"], 5)
            self.assertEqual(result["reason"], "cartesian_boundary", result)
            self.assertEqual(rt.core.accepted_actions, 0)
            self.assertTrue(result["rejected_before_execution"])
            self.assertFalse(result["recovery_required"])
            self.assertTrue(rt.service.status()["ready"])
            code, gripper = rt.cli("gripper", "--position", "0", "--wait")
            self.assertEqual(code, 0, gripper)

    def test_acceleration_rejection_cancels_sequence_and_allows_new_commands(self):
        with Runtime() as rt:
            normal_limit = rt.service.cfg.joint_acceleration
            rt.service.cfg.joint_acceleration = .01
            entered, release = threading.Event(), threading.Event()
            original_preflight = rt.service._preflight
            def blocked_preflight(plan):
                entered.set()
                if not release.wait(3):
                    raise AssertionError("Test preflight timed out")
                original_preflight(plan)
            rt.service._preflight = blocked_preflight
            try:
                job = rt.client.submit([
                    {"type": "delta", "translation_m": [.001, 0, 0], "duration_s": .2},
                    {"type": "gripper", "width_m": .04}])
                self.assertTrue(entered.wait(2))
                queued = rt.client.submit({"type": "gripper", "width_m": .02})
            finally:
                release.set()
            result = rt.client.wait_for_command(job["id"], 5)
            self.assertEqual(result["reason"], "joint_acceleration_limit", result)
            self.assertEqual(result["actions"][0]["steps_sent"], 0)
            self.assertTrue(result["rejected_before_execution"])
            self.assertFalse(result["recovery_required"])
            self.assertEqual(result["actions"][1]["status"], "cancelled")
            self.assertEqual(rt.client.status(queued["id"])["status"], "cancelled")
            self.assertEqual(rt.core.accepted_actions, 0)
            self.assertEqual(rt.driver.widths, [])
            self.assertNotIn(("POST", "/actions/clear"), rt.paths)
            self.assertTrue(rt.service.status()["ready"])
            self.assertIsNone(rt.service.status()["fault"])
            code, gripper = rt.cli("gripper", "--position", "0", "--wait")
            self.assertEqual(code, 0, gripper)
            rt.service.cfg.joint_acceleration = normal_limit
            code, move = rt.cli("delta", "--translation-m", ".001", "0", "0",
                                "--duration", ".2", "--wait")
            self.assertEqual(code, 0, move)

    def test_real_force_failure_during_planning_still_requires_recovery(self):
        with Runtime() as rt:
            original_preflight = rt.service._preflight
            def force_during_preflight(plan):
                rt.force[:3] = [51., 0., 0.]
                rt.feed()
                original_preflight(plan)
            rt.service._preflight = force_during_preflight
            job = rt.client.submit({"type": "delta", "translation_m": [.001, 0, 0],
                                    "duration_s": .2})
            result = rt.client.wait_for_command(job["id"], 5)
            self.assertEqual(result["reason"], "hard_force_limit", result)
            self.assertEqual(result["actions"][0]["steps_sent"], 0)
            self.assertFalse(result["rejected_before_execution"])
            self.assertTrue(result["recovery_required"])
            self.assertFalse(rt.service.status()["ready"])
            with self.assertRaises(terminal.SubmissionError):
                rt.client.submit({"type": "gripper", "width_m": 0.})

    def test_lost_cartesian_response_is_not_a_planning_rejection(self):
        with Runtime() as rt:
            original_request = rt.service.backend.client.request
            posts = []
            def lose_response(method, path, *args, **kwargs):
                result = original_request(method, path, *args, **kwargs)
                if method == "POST" and path == "/actions":
                    posts.append(result)
                    raise motion.MotionError("robot_transport_unknown", "lost Cartesian response")
                return result
            rt.service.backend.client.request = lose_response
            job = rt.client.submit({"type": "delta", "translation_m": [.001, 0, 0],
                                    "duration_s": .2})
            result = rt.client.wait_for_command(job["id"], 5)
            self.assertEqual(result["actions"][0]["steps_sent"], 0)
            self.assertFalse(result["rejected_before_execution"])
            self.assertTrue(result["recovery_required"])
            self.assertEqual(len(posts), 1)
            self.assertFalse(rt.service.status()["ready"])
            with self.assertRaises(terminal.SubmissionError):
                rt.client.submit({"type": "gripper", "width_m": 0.})

    def test_doubled_acceleration_limit_accepts_100mm_move_from_home(self):
        with Runtime(initial_q=controller.HOME_JOINT_TARGET) as rt:
            self.assertEqual(rt.service.status()["joint_acceleration_limit_rad_s2"], 8.)
            initial = rt.core.target_pose.copy()
            code, result = rt.cli("delta", "--translation-m", "-.10", "0", "0",
                                  "--duration", "4", "--wait")
            self.assertEqual(code, 0, result)
            self.assertEqual(rt.core.applied_actions, 200)
            np.testing.assert_allclose(rt.core.target_pose[:3], initial[:3]+[-.10, 0, 0], atol=1e-8)

    def test_controller_speed_limit_rejects_before_motion_without_blocking_gripper(self):
        with Runtime() as rt:
            code, result = rt.cli("delta", "--translation-m", ".10", "0", "0",
                                  "--duration", "3", "--frame", "base", "--wait")
            self.assertEqual(code, 1, result)
            self.assertEqual(result["reason"], "franka_increment_limit", result)
            self.assertIn("0.025 m/s at 50 Hz", result["detail"])
            self.assertIn("at least 4 s", result["detail"])
            self.assertTrue(result["rejected_before_execution"])
            self.assertFalse(result["recovery_required"])
            self.assertEqual(rt.core.accepted_actions, 0)
            self.assertEqual(rt.action_requests, [])
            status = rt.service.status()
            self.assertEqual(status["linear_speed_limit_m_s"], .05)
            self.assertEqual(status["controller_linear_speed_limit_m_s"], .025)
            self.assertTrue(status["ready"])
            code, gripper = rt.cli("gripper", "--position", "0", "--wait")
            self.assertEqual(code, 0, gripper)

    def test_faster_translation_with_matching_controller_limit(self):
        # Emulate the user updating the constant of the robot server; the copied file stays intact.
        home = list(controller.HOME_JOINT_TARGET)
        home[-1] = np.pi/4
        with patch.object(controller, "MAX_TRANSLATION_INCREMENT", .001):
            for distance, duration, initial_q in ((.10, 3., home), (.02, .4, None)):
                with self.subTest(distance=distance, duration=duration), Runtime(initial_q=initial_q) as rt:
                    initial = rt.core.target_pose.copy()
                    code, result = rt.cli("delta", "--translation-m", str(distance), "0", "0",
                                          "--duration", str(duration), "--frame", "base", "--wait")
                    self.assertEqual(code, 0, result)
                    self.assertEqual(rt.core.applied_actions, round(duration*50))
                    np.testing.assert_allclose(rt.core.target_pose[:3], initial[:3]+[distance, 0, 0], atol=1e-8)
                    for _, request_body in rt.action_requests:
                        self.assertLessEqual(np.linalg.norm(request_body["actions"][0][:3]), .001+1e-12)
                    self.assertEqual(rt.service.status()["controller_linear_speed_limit_m_s"], .05)

    def test_joint_boundary_faults_on_measured_state(self):
        with Runtime() as rt:
            with rt.service.lock:
                rt.service.lower[0] = rt.q[0]+.01
            wait_until(lambda: rt.service.status()["fault"])
            result = rt.service.status()["last_result"]
            self.assertEqual(result["detail"], "joint_boundary", result)
            self.assertEqual(rt.core.accepted_actions, 0)

    def test_two_second_motion_uses_one_hundred_ticks(self):
        with Runtime() as rt:
            start = rt.core.target_pose.copy()
            code, result = rt.cli("delta", "--translation-m", ".01", "0", "0",
                                  "--duration", "2", "--wait")
            self.assertEqual(code, 0, result)
            action = result["actions"][0]
            self.assertEqual(rt.core.applied_actions, 100)
            self.assertGreaterEqual(action["execution_elapsed_s"], 2.)
            self.assertLess(action["execution_elapsed_s"], 2.3)
            np.testing.assert_allclose(rt.core.target_pose[:3], start[:3]+[.01, 0, 0], atol=1e-8)

    def test_two_second_motion_with_four_ms_request_delay(self):
        # Each request to the robot server takes a few milliseconds.
        with Runtime(rpc_delay_s=.004) as rt:
            initial = rt.core.target_pose.copy()
            code, result = rt.cli("delta", "--translation-m", ".01", "0", "0",
                                  "--duration", "2", "--wait")
            self.assertEqual(code, 0, result)
            self.assertEqual(rt.core.applied_actions, 100)
            self.assertEqual(sum(method == "POST" and path == "/actions"
                                 for method, path in rt.paths), 100)
            self.assertLess(result["actions"][0]["execution_elapsed_s"], 2.2)
            np.testing.assert_allclose(rt.core.target_pose[:3], initial[:3]+[.01, 0, 0], atol=1e-8)

    def test_four_second_motion_survives_late_responses_without_catch_up(self):
        with Runtime(rpc_delay_s=.004) as rt:
            delays = []
            def delayed_response(method, path):
                if method == "POST" and path == "/actions" and len(rt.action_requests) in (41, 140):
                    delays.append(len(rt.action_requests))
                    return .045
                return 0.
            rt.response_delay = delayed_response
            initial = rt.core.target_pose.copy()
            code, result = rt.cli("delta", "--translation-m", ".05", "0", "0",
                                  "--duration", "4", "--wait")
            self.assertEqual(code, 0, result)
            action = result["actions"][0]
            self.assertEqual(delays, [41, 140])
            self.assertEqual(rt.core.applied_actions, 200)
            self.assertEqual(len(rt.action_requests), 200)
            for _, payload in rt.action_requests:
                self.assertEqual(len(payload["actions"]), 1)
                np.testing.assert_allclose(payload["actions"][0], [.00025, 0, 0, 0, 0, 0], atol=1e-9)
            self.assertEqual(action["timing_policy"], "extend_without_catch_up")
            self.assertGreater(action["scheduling_delay_s"], .04)
            self.assertGreater(action["execution_elapsed_s"], 4.04)
            self.assertEqual(action["clipped_steps"], 0)
            np.testing.assert_allclose(rt.core.target_pose[:3], initial[:3]+[.05, 0, 0], atol=1e-8)
            self.assertTrue(rt.service.status()["ready"])

    def test_terminal_accepts_negative_scientific_notation(self):
        with Runtime() as rt:
            initial = rt.core.target_pose.copy()
            code, result = rt.cli("delta", "--translation-m", "-1e-3", "0", "0",
                                  "--duration", ".2", "--wait")
            self.assertEqual(code, 0, result)
            np.testing.assert_allclose(rt.core.target_pose[:3], initial[:3]+[-.001, 0, 0], atol=1e-8)

    def test_rotation_tool_delta_and_absolute_pose(self):
        with Runtime() as rt:
            initial = rt.core.target_pose.copy()
            rotation = R.from_rotvec([0, 0, .01])*R.from_quat(initial[3:])
            code, result = rt.cli(
                "delta", "--rotation-rad", "0", "0", ".01", "--duration", ".5",
                "--next", "delta", "--translation-m", ".001", "0", "0",
                "--frame", "tool", "--duration", ".2", "--wait")
            self.assertEqual(code, 0, result)
            np.testing.assert_allclose(rt.core.target_pose[:3],
                                       initial[:3]+rotation.apply([.001, 0, 0]), atol=1e-8)
            self.assertLess((R.from_quat(rt.core.target_pose[3:])*rotation.inv()).magnitude(), 1e-8)
            code, result = rt.cli(
                "pose", "--position-m", *map(str, initial[:3]),
                "--quaternion-xyzw", *map(str, initial[3:]), "--duration", ".5", "--wait")
            self.assertEqual(code, 0, result)
            np.testing.assert_allclose(rt.core.target_pose[:3], initial[:3], atol=1e-8)

    def test_invalid_sequence_is_rejected_before_native_commands(self):
        with Runtime() as rt:
            with self.assertRaises(terminal.SubmissionError):
                rt.client.submit([{"type": "gripper", "width_m": .04},
                                  {"type": "delta", "translation_m": [.001, 0, 0],
                                   "duration_s": 0}])
            self.assertEqual(rt.driver.widths, [])
            self.assertEqual(rt.core.accepted_actions, 0)

    def test_start_and_recover_preserve_held_target_under_contact(self):
        with Runtime(held_offset=[.002, 0, 0]) as rt:
            held = rt.core.target_pose.copy()
            np.testing.assert_allclose(rt.service.reference[:3], held[:3], atol=1e-8)
            code, job = rt.cli("delta", "--translation-m", ".001", "0", "0",
                               "--duration", ".2", "--wait")
            self.assertEqual(code, 0, job)
            np.testing.assert_allclose(rt.core.target_pose[:3], held[:3]+[.001, 0, 0], atol=1e-8)
            rt.client.stop()
            rt.client.wait_for_stop(5)
            rt.client.recover()
            wait_until(lambda: rt.service.status()["ready"])
            np.testing.assert_allclose(rt.service.reference[:3], rt.core.target_pose[:3], atol=1e-8)
            self.assertEqual(rt.core.applied_actions, 10)

    def test_recovery_remains_pending_until_initialization_finishes(self):
        with Runtime() as rt:
            rt.client.stop()
            rt.client.wait_for_stop(5)
            entered, release = threading.Event(), threading.Event()
            original_start = rt.service.backend.start
            def delayed_start():
                entered.set()
                if not release.wait(3):
                    raise AssertionError("Recovery test timed out")
                return original_start()
            rt.service.backend.start = delayed_start
            try:
                rt.client.recover()
                self.assertTrue(entered.wait(2))
                status = rt.client.status()
                self.assertTrue(status["stop_requested"], status)
                self.assertFalse(status["ready"], status)
                with self.assertRaises(terminal.SubmissionError):
                    rt.client.submit({"type": "gripper", "width_m": .04})
            finally:
                release.set()
            result = rt.client.wait_for_stop(5)
            self.assertTrue(result["ready"], result)

    def test_lost_native_response_allows_deliberate_recovery(self):
        with Runtime() as rt:
            rt.driver.gripper_delay = .15
            original_request = rt.service.backend.client.request
            posts = []
            def lose_response(method, path, *args, **kwargs):
                result = original_request(method, path, *args, **kwargs)
                if method == "POST" and path == "/gripper":
                    posts.append(result)
                    raise motion.MotionError("robot_transport_unknown", "lost accepted response")
                return result
            rt.service.backend.client.request = lose_response
            job = rt.client.submit({"type": "gripper", "width_m": .04})
            result = rt.client.wait_for_command(job["id"], 5)
            self.assertEqual(result["status"], "failed")
            wait_until(lambda: rt.core.command_status("gripper")["state"] == "succeeded")
            rt.client.recover()
            result = rt.client.wait_for_stop(5)
            self.assertTrue(result["ready"], result)
            self.assertEqual(len(posts), 1)
            self.assertEqual(rt.driver.widths, [(.04, .05)])

    def test_unconnected_backend_cannot_confirm_stop(self):
        service = motion.MotionService(motion.Settings())
        service.backend = motion.RobotBackend(service.cfg)
        self.assertFalse(service._stop())
        self.assertFalse(service.status()["last_result"]["confirmed"])

    def test_lost_post_response_is_not_retried(self):
        backend = motion.RobotBackend(motion.Settings())
        class LostResponse:
            def __init__(self):
                self.posts = 0
            def request(self, method, path, payload=None):
                if method == "GET":
                    return dict(server="franka_robot_server", published_count=1,
                                applied_actions=0, accepted_actions=0, queue_depth=0,
                                home={"state": "idle"}, gripper={"state": "idle"})
                self.posts += 1
                raise motion.MotionError("robot_transport_unknown", "simulated lost response")
        client = LostResponse()
        backend.client, backend.connected = client, True
        with self.assertRaises(motion.MotionError):
            backend.send_gripper(.04, .05)
        self.assertEqual(client.posts, 1)
        self.assertTrue(backend.ambiguous)


if __name__ == "__main__":
    unittest.main()
