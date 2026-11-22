"""Robot-free tests for HTTP admission, FIFO execution and target semantics."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


spec = importlib.util.spec_from_file_location(
    'robot_server_queue', Path(__file__).resolve().parent.parent / 'franka_robot_server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class ActionQueueTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.state = {'pose': [.4, 0., .3, 0., 0., 0., 1.], 'q': [0.]*7, 'dq': [0.]*7,
                      'robot_time_s': 1., 'robot_mode': 2, 'current_errors': []}
        self.core = server.FrankaServer(clock=lambda: self.now)
        self.core.update_state(self.state)
        self.initial = self.core.initialize_target_once()
        self.client = server.create_app(self.core).test_client()

    def test_fifo_one_row_per_tick_and_no_repeat_when_empty(self):
        response = self.client.post('/actions', json={'actions': [
            [.0003, 0., 0., 0., 0., 0.], [0., -.0002, 0., 0., 0., 0.]]})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json, {'accepted': 2, 'queue_depth': 2})
        np.testing.assert_array_equal(self.core.target_pose, self.initial)
        first = self.core.next_target()
        np.testing.assert_allclose(first[:3], [.4003, 0., .3])
        self.assertEqual(self.core.info()['queue_depth'], 1)
        second = self.core.next_target()
        np.testing.assert_allclose(second[:3], [.4003, -.0002, .3])
        for _ in range(4):
            np.testing.assert_array_equal(self.core.next_target(), second)
        self.assertEqual(self.core.info()['applied_actions'], 2)

    def test_target_accumulates_independently_of_measured_pose(self):
        self.core.enqueue_actions([[.0002, 0., 0., 0., 0., 0.]]*2)
        self.core.next_target()
        self.state['pose'][0] = .395  # Feedback changes, but the command reference stays fixed.
        self.state['robot_time_s'] = 2.
        self.core.update_state(self.state)
        self.assertAlmostEqual(self.core.next_target()[0], .4004)

    def test_rotation_vector_is_left_multiplied_in_base_frame(self):
        initial_rotation = Rotation.from_euler('z', np.pi/2)
        self.state['pose'][3:] = initial_rotation.as_quat().tolist()
        core = server.FrankaServer(clock=lambda: self.now)
        core.update_state(self.state)
        core.initialize_target_once()
        core.enqueue_actions([[0., 0., 0., .004, 0., 0.]])
        actual = Rotation.from_quat(core.next_target()[3:]).as_matrix()
        expected = (Rotation.from_rotvec([.004, 0., 0.])*initial_rotation).as_matrix()
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_zero_action_consumes_a_tick_and_clear_holds_last_target(self):
        self.core.enqueue_actions([[0.]*6, [.0002, 0., 0., 0., 0., 0.], [.0002, 0., 0., 0., 0., 0.]])
        np.testing.assert_array_equal(self.core.next_target(), self.initial)
        self.assertEqual(self.core.info()['queue_depth'], 2)
        target = self.core.next_target()
        response = self.client.post('/actions/clear')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'cleared': 1, 'queue_depth': 0})
        np.testing.assert_array_equal(self.core.next_target(), target)
        info = self.client.get('/info').json
        self.assertEqual((info['accepted_actions'], info['applied_actions'], info['cleared_actions']), (3, 2, 1))

    def test_invalid_batch_is_rejected_atomically(self):
        valid = [.0001, 0., 0., 0., 0., 0.]
        invalid_rows = [[0.]*5, [0.]*7, ['x']*6, [True]*6, [float('nan')]*6,
                        [float('inf')]*6, [10**1000]*6, [.0004, .0004, 0., 0., 0., 0.],
                        [0., 0., 0., .004, .004, 0.]]
        for row in invalid_rows:
            with self.subTest(row=str(row)[:80]):
                response = self.client.post('/actions', json={'actions': [valid, row]})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(self.core.info()['queue_depth'], 0)
                self.assertEqual(self.core.info()['accepted_actions'], 0)
        for body in ({}, {'actions': []}, {'actions': None}, {'actions': valid}, [],
                     {'actions': [valid], 'unexpected': True}):
            self.assertEqual(self.client.post('/actions', json=body).status_code, 400)
        self.assertEqual(self.client.post('/actions', data='{', content_type='application/json').status_code, 400)

    def test_queue_overflow_rejects_whole_batch_without_dropping_existing_actions(self):
        self.core.enqueue_actions([[0.]*6]*(server.MAX_QUEUED_ACTIONS-1))
        response = self.client.post('/actions', json={'actions': [[0.]*6]*2})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.core.info()['queue_depth'], server.MAX_QUEUED_ACTIONS-1)
        self.assertEqual(self.client.post('/actions', json={'actions': [[0.]*6]}).status_code, 202)
        self.assertEqual(self.core.info()['queue_depth'], server.MAX_QUEUED_ACTIONS)

    def test_not_ready_or_stale_robot_rejects_new_actions(self):
        cold = server.create_app(server.FrankaServer()).test_client()
        self.assertEqual(cold.post('/actions', json={'actions': [[0.]*6]}).status_code, 503)
        self.now = .6
        self.assertEqual(self.client.post('/actions', json={'actions': [[0.]*6]}).status_code, 503)
        self.assertEqual(self.core.info()['queue_depth'], 0)

    def test_faulted_feedback_prevents_queued_motion(self):
        self.core.enqueue_actions([[.0002, 0., 0., 0., 0., 0.]])
        self.state.update(robot_time_s=2., current_errors=['joint_reflex'])
        self.core.update_state(self.state)
        with self.assertRaisesRegex(RuntimeError, 'joint_reflex'):
            self.core.next_target()
        np.testing.assert_array_equal(self.core.target_pose, self.initial)
        self.assertEqual(self.core.info()['applied_actions'], 0)
        self.assertEqual(self.client.post('/actions', json={'actions': [[0.]*6]}).status_code, 503)

    def test_execution_checks_target_lead_before_consuming_action(self):
        for rotation in (False, True):
            with self.subTest(rotation=rotation):
                self.core.target_pose = self.initial.copy()
                if rotation:
                    self.core.target_pose[3:] = Rotation.from_rotvec([0., 0., .199]).as_quat()
                    action = [0., 0., 0., 0., 0., .002]
                else:
                    self.core.target_pose[0] += .0299
                    action = [.0002, 0., 0., 0., 0., 0.]
                previous = self.core.target_pose.copy()
                self.core.enqueue_actions([action])
                with self.assertRaisesRegex(ValueError, 'target lead'):
                    self.core.next_target()
                np.testing.assert_array_equal(self.core.target_pose, previous)
                self.assertEqual(self.core.info()['queue_depth'], 1)
                self.assertEqual(self.core.info()['applied_actions'], 0)
                self.core.clear_actions()

    def test_concurrent_batches_remain_contiguous(self):
        actions = [[.0001, 0., 0., 0., 0., 0.], [0., .0001, 0., 0., 0., 0.]]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda action: self.core.enqueue_actions([action]*5), actions))
        self.assertEqual(sum(result['accepted'] for result in results), 10)
        previous = self.initial.copy()
        directions = []
        for _ in range(10):
            target = self.core.next_target()
            directions.append(int(np.argmax(np.abs(target[:3]-previous[:3]))))
            previous = target
        self.assertIn(directions, [[0]*5+[1]*5, [1]*5+[0]*5])
        self.assertEqual(self.core.info()['applied_actions'], 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
