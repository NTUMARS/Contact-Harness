"""Offline checks: start script -> server arguments -> expanded ROS startup parameters.

Run with the Python of the robot server. The tests that expand the launch file need the
catkin workspace (CATKIN_SETUP, default ~/catkin_ws/devel/setup.bash) and are
skipped without it. Launch files are only parsed; no ROS nodes are started.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml


SERVER_DIR = Path(__file__).resolve().parent.parent
CATKIN_SETUP = os.environ.get('CATKIN_SETUP', str(Path.home() / 'catkin_ws' / 'devel' / 'setup.bash'))
spec = importlib.util.spec_from_file_location('robot_server', SERVER_DIR / 'franka_robot_server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class ConfigurationTests(unittest.TestCase):
    def shell_plan(self, **overrides):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        for key in ('EEF_IMPEDANCE', 'COLLISION_FORCE_THRESHOLD', 'TRANSLATIONAL_CLIP', 'COLLISION_TORQUE_SCALE',
                    'HOME_JOINT_TARGET', 'GRIPPER_GRASP_FORCE', 'ROBOT_IP'):
            env.pop(key, None)
        env['ROBOT_PYTHON'] = sys.executable
        env.update(overrides)
        result = subprocess.run(['bash', str(SERVER_DIR / 'start_robot_server.sh'), '--check-only'],
                                cwd='/tmp', env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def ros_parameters(self, launch_command):
        # Parse with the real ROS loader, including xacro and YAML substitutions.
        # --dump-params never launches nodes or connects to the robot/master.
        if not Path(CATKIN_SETUP).is_file():
            self.skipTest('ROS workspace not found; set CATKIN_SETUP')
        command = list(launch_command)
        command.insert(1, '--dump-params')
        with tempfile.TemporaryDirectory(prefix='robot_server_ros_') as ros_home:
            env = dict(os.environ, ROS_HOME=ros_home, ROS_LOG_DIR=ros_home+'/log',
                       PYTHONDONTWRITEBYTECODE='1', CATKIN_SETUP=CATKIN_SETUP)
            env['PATH'] = str(Path(sys.executable).parent)+':'+env['PATH']
            result = subprocess.run(
                ['bash', '-c', 'source "$CATKIN_SETUP"\nexec "$@"',
                 'ros-configuration-check', *command],
                cwd='/tmp', env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        parameters = yaml.safe_load(result.stdout)
        # roslaunch can exit 0 on parsing failure, so inspect the output too.
        self.assertIsInstance(parameters, dict, result.stderr)
        return parameters

    def test_reference_settings_select_default_profile_and_local_launch(self):
        # Start script defaults are user-editable; compare against the explicit Python defaults.
        plan = self.shell_plan(EEF_IMPEDANCE='1500,89,150,7', COLLISION_FORCE_THRESHOLD='20',
                               TRANSLATIONAL_CLIP='0.01', COLLISION_TORQUE_SCALE='1.0')
        self.assertEqual(plan['impedance_parameters'], server.IMPEDANCE_PROFILE)
        self.assertEqual(plan['collision_force_threshold_n'], 20.)
        self.assertFalse(plan['hardware_opened'])
        self.assertEqual(plan['launch_command'][3], str(server.LOCAL_LAUNCH))
        params = self.ros_parameters(plan['launch_command'])
        prefix = '/franka_control/collision_config/'
        for phase in ('acceleration', 'nominal'):
            for bound in ('lower', 'upper'):
                self.assertEqual(params[prefix+bound+'_force_thresholds_'+phase], [20.]*3 + [25.]*3)
                self.assertEqual(params[prefix+bound+'_torque_thresholds_'+phase],
                                 [20., 20., 18., 18., 16., 14., 12.])

    def test_custom_settings_reach_driver_namespace(self):
        for threshold in (12.5, 25.):
            with self.subTest(threshold=threshold):
                plan = self.shell_plan(EEF_IMPEDANCE='800,55,100,5',
                                       COLLISION_FORCE_THRESHOLD=str(threshold), COLLISION_TORQUE_SCALE='1.0')
                profile = plan['impedance_parameters']
                self.assertEqual([profile[key] for key in server.EEF_IMPEDANCE_FIELDS], [800., 55., 100., 5.])
                params = self.ros_parameters(plan['launch_command'])
                prefix = '/franka_control/collision_config/'
                for phase in ('acceleration', 'nominal'):
                    self.assertEqual(params[prefix+'upper_force_thresholds_'+phase],
                                     [threshold]*3 + [25.]*3)
                    self.assertEqual(params[prefix+'lower_force_thresholds_'+phase],
                                     [min(20., threshold)]*3 + [25.]*3)
                    for bound in ('lower', 'upper'):
                        self.assertEqual(params[prefix+bound+'_torque_thresholds_'+phase],
                                         [20., 20., 18., 18., 16., 14., 12.])

    def test_invalid_settings_fail_before_startup(self):
        invalid = {
            '--eef-impedance': ['1,2,3', '1,2,3,4,5', 'bad', 'nan,89,150,7',
                                '1500,inf,150,7', '-1,89,150,7', '6001,89,150,7',
                                '1500,401,150,7', '1500,89,301,7', '1500,89,150,31'],
            '--collision-force-threshold': ['nan', 'inf', '-inf', '0', '-1', 'bad'],
            '--translational-clip': ['nan', 'inf', '-inf', '0', '-0.01', '0.10001', 'bad'],
            '--collision-torque-scale': ['nan', 'inf', '-inf', '0', '-1', '0.999', '1e308', 'bad'],
        }
        with patch.object(server, 'run') as run:
            for option, values in invalid.items():
                for value in values:
                    with self.subTest(option=option, value=value), contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            server.main([option+'='+value])
                        self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_check_only_does_not_contact_robot_or_start_processes(self):
        with patch.object(server, 'run') as run, patch.object(server.socket, 'create_connection') as connect, \
                patch.object(server.subprocess, 'Popen') as popen, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(server.main(['--check-only']), 0)
            run.assert_not_called()
            connect.assert_not_called()
            popen.assert_not_called()

    def test_info_reports_selected_profile_without_mutating_defaults(self):
        original = dict(server.IMPEDANCE_PROFILE)
        profile = server.make_impedance_profile((800., 55., 100., 5.), .03)
        core = server.FrankaServer(impedance_profile=profile, collision_force_threshold=12.5,
                                   collision_torque_scale=1.5)
        response = server.create_app(core).test_client().get('/info')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['impedance_parameters'], profile)
        for axis in ('x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z'):
            self.assertEqual(response.json['impedance_parameters']['translational_clip_'+axis], .03)
        self.assertEqual(response.json['collision_force_threshold_n'], 12.5)
        self.assertEqual(response.json['collision_torque_scale'], 1.5)
        self.assertEqual(response.json['collision_torque_thresholds_nm'], [30., 30., 27., 27., 24., 21., 18.])
        profile['translational_stiffness'] = 1.
        self.assertEqual(core.info()['impedance_parameters']['translational_stiffness'], 800.)
        self.assertEqual(server.FrankaServer().info()['impedance_parameters'], original)

    def test_shell_clip_updates_all_directions_and_preserves_other_gains(self):
        for clip in ('0.03', '0.1'):
            with self.subTest(clip=clip):
                plan = self.shell_plan(TRANSLATIONAL_CLIP=clip, EEF_IMPEDANCE='1500,89,150,7')
                profile = plan['impedance_parameters']
                for name, original in server.IMPEDANCE_PROFILE.items():
                    expected = float(clip) if name.startswith('translational_clip_') else original
                    self.assertEqual(profile[name], expected, name)

    def test_torque_scale_reaches_both_driver_stop_thresholds(self):
        baseline = [20., 20., 18., 18., 16., 14., 12.]
        for scale in (1.0, 1.5, 2.0, 1e20):
            with self.subTest(scale=scale):
                # The large value checks YAML numeric encoding only; no hardware is opened.
                plan = self.shell_plan(COLLISION_TORQUE_SCALE=str(scale), COLLISION_FORCE_THRESHOLD='70')
                expected = [value * scale for value in baseline]
                self.assertEqual(plan['collision_torque_scale'], scale)
                self.assertEqual(plan['collision_torque_thresholds_nm'], expected)
                params = self.ros_parameters(plan['launch_command'])
                prefix = '/franka_control/collision_config/'
                for phase in ('acceleration', 'nominal'):
                    self.assertEqual(params[prefix+'upper_torque_thresholds_'+phase], expected)
                    self.assertEqual(params[prefix+'lower_torque_thresholds_'+phase], baseline)
                    self.assertEqual(params[prefix+'upper_force_thresholds_'+phase], [70.]*3 + [25.]*3)
                    self.assertEqual(params[prefix+'lower_force_thresholds_'+phase], [20.]*3 + [25.]*3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
