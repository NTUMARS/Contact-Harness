"""Offline observation migration checks. Real cameras/robots are never used here."""
import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

import observation_server as obs
import get_observation as client
from test_motion_server import Runtime, wait_until


class FakeCamera:
    instances = []
    def __init__(self, category, config, timeout, startup_timeout):
        self.config = config
        self.closed = False
        self.frame = 0
        self.instances.append(self)

    def capture(self, kinds):
        self.frame += 1
        return {"meta": {"received_monotonic_s": time.monotonic(),
                         "received_at_unix_s": time.time(), "frame_number": self.frame},
                "arrays": {"rgb": np.full((3, 4, 3), [10, 20, 30], dtype=np.uint8),
                           "depth": np.full((3, 4), .25, dtype=np.float32)}}

    def close(self):
        self.closed = True


@contextlib.contextmanager
def running_observations(rt, *, tare=True, worker=FakeCamera):
    config = obs.load_config()
    config["robot_address"] = "http://127.0.0.1:%d" % rt.robot_server.server_port
    config["motion_server"] = rt.client_url
    config["tare_on_start"] = tare
    service = obs.ObservationService(config, worker_factory=worker)
    http = ThreadingHTTPServer(("127.0.0.1", 0), obs.handler_for(service))
    http.daemon_threads = True
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    service.start()
    observer = client.ObservationClient("http://127.0.0.1:%d" % http.server_port,
                                        motion_server=rt.client_url)
    try:
        wait_until(lambda: service.status()["ready"])
        yield service, observer
    finally:
        service.close()
        http.shutdown()
        http.server_close()
        thread.join(2)


class ObservationMigration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        cls.protected = [Path("franka_robot_server.py"), Path("motion_server.py"),
                         Path("terminal_command.py")]
        cls.hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.protected]

    @classmethod
    def tearDownClass(cls):
        assert cls.hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.protected]

    def test_wrench_rotates_and_shifts_force_and_torque(self):
        config = obs.load_config()
        config["ee_to_stiffness"] = [.1, 0, 0, 0, np.pi/2, 0]
        pose = np.array([0., 0., 0., 0., 0., np.pi/2])
        tool = obs.pose_T([0, .2, 0, 0, 0, 0])
        native = np.array([1., 0., 0., 0., 1., 0.])
        base = obs.wrench_values(native, pose, tool, config, "raw")
        wrist = obs.wrench_values(native, pose, tool, config, "wrist")
        world = obs.wrench_values(native, pose, tool, config, "tcp")
        np.testing.assert_allclose(base, [0, 0, 1, 1.1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(wrist, [0, 0, 1, .2, -1.1, 0], atol=1e-12)
        np.testing.assert_allclose(world, [0, 0, 1, 1.1, 0, 0], atol=1e-12)
        context = {"flange": wrist, "tool_end": obs.T_pose(tool).tolist(),
                   "tcp_pose_base": pose.tolist(), "world_from_base": config["world_from_base"]}
        np.testing.assert_allclose(client.transform_wrench(context, reference="tcp", frame="base"), base, atol=1e-12)
        np.testing.assert_allclose(client.transform_wrench(context, reference="tcp", frame="world"), world, atol=1e-12)

    def test_configured_contact_reference_changes_moment_only(self):
        config = obs.load_config()
        pose = np.zeros(6)
        native = np.array([0., 0., -2., 0., 0., 0.])
        with self.assertRaisesRegex(ValueError, "geometry is not configured"):
            obs.wrench_values(native, pose, np.eye(4), config, "raw", "fingertip")
        config["wrench_reference_points_m"]["fingertip"] = [.1, 0, 0]
        shifted = obs.wrench_values(native, pose, np.eye(4), config, "raw", "fingertip")
        np.testing.assert_allclose(shifted, [0, 0, 2, 0, .2, 0])
        with self.assertRaises(ValueError):
            obs.wrench_values(native, pose, np.eye(4), config, "wrist", "fingertip")

    def test_camera_config_has_two_real_devices_and_one_placeholder(self):
        config = obs.load_config()
        self.assertEqual(list(obs.sensor_configs(config)), ["wrist", "third_view"])
        self.assertEqual(config["cameras"]["wrist"]["serial"], "WRIST_REALSENSE_SERIAL")
        self.assertFalse(config["cameras"]["side"]["enabled"])
        self.assertTrue(config["cameras"]["third_view"]["enabled"])
        self.assertNotIn("side.rgb", obs.expand_observations(["all"], config))
        with self.assertRaises(ValueError):
            obs.expand_observations(["tactile_left.wrench"], config)
        with self.assertRaises(ValueError):
            obs.expand_observations(["right_wrist.rgb"], config)

    def test_snapshot_force_torque_images_placeholders_and_atomic_replacement(self):
        with Runtime() as rt:
            rt.force[:] = [1., 2., 3., .1, .2, .3]
            with running_observations(rt) as (service, observer), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as output:
                bias = service.wrench_baseline.bias.copy()
                rt.force[:] = bias + [2., -1., 3., .4, .5, -.6]
                wait_until(lambda: np.allclose(service.latest["state"]["sample"]["native_wrench"], rt.force))
                result = observer.get_observation(
                    ["state", "force", "torque", "wrench", "joint_velocities",
                     "wrist.rgb", "wrist.depth", "side.rgb", "third_view.rgb", "gripper", "gripper_status"],
                    output, max_age=2.)
                self.assertEqual(set(result["errors"]), {"side.rgb", "gripper"})
                values = result["observations"]
                expected = obs.wrench_values(rt.force-bias, service.latest["state"]["sample"]["pose"],
                                             np.eye(4), service.config, "raw")
                np.testing.assert_allclose(values["force"], expected, atol=1e-8)
                np.testing.assert_allclose(values["state"]["force"], expected, atol=1e-8)
                np.testing.assert_allclose(values["torque"], expected[3:], atol=1e-8)
                self.assertEqual(set(values["state"]), {"force", "eef_pose", "joint_angles"})
                self.assertEqual(len(values["state"]["joint_angles"]), 7)
                self.assertEqual(values["wrench"]["reference"], "EE")
                self.assertFalse(values["gripper_status"]["measurement_available"])
                self.assertIsNone(values["gripper_status"]["measured_width_m"])
                image = cv2.imread(values["wrist.rgb"])
                np.testing.assert_array_equal(image[0, 0], [30, 20, 10])
                np.testing.assert_allclose(np.load(values["wrist.depth"], allow_pickle=False), .25)
                second = observer.get_observation(["force"], output, max_age=2., wrench="tcp")
                self.assertFalse(second["errors"])
                self.assertEqual({p.name for p in (Path(output)/"latest").iterdir()}, {"observation.json"})
                self.assertTrue(all(method == "GET" for method, path in rt.paths))
                self.assertFalse(any(path.startswith("/motion") for _, path in rt.paths))
            self.assertTrue(FakeCamera.instances[-1].closed)

    def test_tare_uses_new_samples_and_does_not_modify_motion_wrench(self):
        with Runtime() as rt, running_observations(rt) as (service, observer):
            rt.force[:] = [2., 0, 0, .3, 0, 0]
            wait_until(lambda: np.allclose(service.latest["state"]["sample"]["native_wrench"], rt.force))
            old_id = service.wrench_baseline.calibration_id
            result = observer.calibrate()
            self.assertTrue(result["ready"])
            self.assertGreaterEqual(result["sample_count"], 10)
            self.assertNotEqual(old_id, result["calibration_id"])
            result, _ = service.snapshot({"observations": ["force"]})
            np.testing.assert_allclose(result["observations"]["force"], np.zeros(6), atol=1e-10)
            np.testing.assert_allclose(rt.core.latest_state()["wrench_K"], rt.force)
            self.assertTrue(all(method == "GET" for method, _ in rt.paths))

    def test_missing_camera_does_not_block_robot_feedback(self):
        class MissingCamera:
            def __init__(self, *args):
                raise RuntimeError("camera not connected")
        with Runtime() as rt:
            config = obs.load_config()
            config.update(robot_address="http://127.0.0.1:%d" % rt.robot_server.server_port,
                          motion_server=rt.client_url, tare_on_start=False)
            service = obs.ObservationService(config, worker_factory=MissingCamera)
            try:
                service.start()
                wait_until(lambda: "state" in service.latest and not service.latest["state"]["errors"])
                result, files = service.snapshot({"observations": ["eef_pose", "wrist.rgb", "third_view.depth"]})
                self.assertIn("eef_pose", result["observations"])
                self.assertEqual(set(result["errors"]), {"wrist.rgb", "third_view.depth"})
                self.assertEqual(files, {})
            finally:
                service.close()

    def test_camera_acquisition_continues_with_robot_services_offline(self):
        config = obs.load_config()
        config.update(robot_address="http://127.0.0.1:1", motion_server="http://127.0.0.1:1")
        service = obs.ObservationService(config, timeout=.1, worker_factory=FakeCamera)
        try:
            service.start()
            wait_until(lambda: "wrist" in service.latest and service.health["state"]["phase"] == "error")
            result, files = service.snapshot({"observations": ["state", "wrist.rgb"]})
            self.assertEqual(set(result["observations"]), {"wrist.rgb"})
            self.assertIn("state", result["errors"])
            self.assertIn("wrist.rgb.png", files)
        finally:
            service.close()

    def test_frozen_robot_feedback_is_not_refreshed_by_polling(self):
        with Runtime() as rt, running_observations(rt, tare=False) as (service, observer):
            rt.pause_feedback = True
            time.sleep(.2)
            result, _ = service.snapshot({"observations": ["eef_pose"], "max_age_s": .08})
            self.assertNotIn("eef_pose", result["observations"])
            self.assertIn("eef_pose", result["errors"])

    def test_client_blocks_during_motion_and_preserves_latest(self):
        with Runtime() as rt, running_observations(rt, tare=False) as (service, observer):
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as output:
                observer.get_observation(["eef_pose"], output, max_age=2.)
                before = (Path(output)/"latest"/"observation.json").read_bytes()
                with rt.service.lock:
                    rt.service.active = "simulated-busy"
                try:
                    with self.assertRaisesRegex(client.ObservationError, "active or queued"):
                        observer.get_observation(["eef_pose"], output, max_age=2.)
                    self.assertEqual((Path(output)/"latest"/"observation.json").read_bytes(), before)
                finally:
                    with rt.service.lock:
                        rt.service.active = None

    def test_client_rejects_unsafe_bundle_paths_and_stale_data(self):
        def bundle(manifest, files):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as archive:
                archive.writestr("manifest.json", json.dumps(manifest))
                for name, value in files.items():
                    archive.writestr(name, value)
            return stream.getvalue()
        manifest = {"protocol": 2, "observations": {"wrist.rgb": "../escape.png"}, "errors": {}, "ages": {"wrist.rgb": 0}}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as output:
            with self.assertRaises(client.ObservationError):
                client.save_bundle(bundle(manifest, {"../escape.png": b"bad"}), output, time.monotonic(), 1.)
            manifest = {"protocol": 2, "observations": {"eef_pose": [0]*6}, "errors": {}, "ages": {"eef_pose": 2.}}
            result = client.save_bundle(bundle(manifest, {}), output, time.monotonic(), .5)
            self.assertEqual(result["observations"], {})
            self.assertIn("eef_pose", result["errors"])

    def test_robot_clock_reset_invalidates_tare(self):
        baseline = obs.WrenchBaseline(enabled=False)
        sample = {"errors": [], "age_s": 0., "generation": "first",
                  "native_wrench": np.ones(6)}
        baseline.apply(sample)
        sample["generation"] = "second"
        with self.assertRaisesRegex(ValueError, "clock reset"):
            baseline.apply(sample)

    def test_read_only_transport_rejects_control_routes(self):
        transport = obs.ReadOnlyRobotHTTP("http://127.0.0.1:1", .1)
        for path in ("/actions", "/actions/clear", "/home", "/recover"):
            with self.assertRaises(ValueError):
                transport.get(path)


if __name__ == "__main__":
    unittest.main()
