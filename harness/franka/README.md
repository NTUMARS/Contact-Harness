# Contact Harness on Franka

Harness for a Franka Panda arm with the Franka gripper. The agent sees a wrist camera and a third-view camera and reads the robot's estimated external wrench. There is no tactile sensor on this platform.

## Files

| File | Purpose |
|---|---|
| `franka_robot_server.py` | Robot server. Cartesian impedance control at 1 kHz with a 50 Hz target loop, joint-position homing, gripper actions, state feedback |
| `start_robot_server.sh` | Starts the robot server with our settings |
| `robot_control/` | Launch file and driver configuration that the robot server starts, and the description of its interface |
| `motion_server.py` | Guarded motion. Sends bounded base-frame increments and checks fresh state and force before each increment |
| `observation_server.py` | Cameras and robot state, served on request |
| `terminal_command.py` | Client the agent uses to send motion and gripper commands |
| `get_observation.py` | Client the agent uses to read observations |
| `config/observation.yaml` | Camera serial numbers and frames |
| `docs/` | Agent prompt and task skills |
| `requirements.txt` | Python packages |
| `tests/` | Offline tests |

## Requirements

One computer is connected to the robot and to the cameras and runs the three services and the agent.

- A real-time kernel and the Franka Control Interface, as required by `franka_ros`
- ROS 1 Noetic with `franka_ros`. We used `franka_ros` 0.10.2 without modification
- The controller package in `third_party/franka_controllers/`, built in the same catkin workspace. It holds the Cartesian impedance controller and the joint position controller used for homing
- Python 3.10 with the packages in `requirements.txt`
- RealSense cameras
- A terminal coding agent

## Setup

```bash
cp -r third_party/franka_controllers ~/catkin_ws/src/      # next to franka_ros
cd ~/catkin_ws && catkin_make -DCMAKE_BUILD_TYPE=Release
pip install -r harness/franka/requirements.txt             # in the Python 3.10 environment
```

1. Set `CATKIN_SETUP` and `ROBOT_PYTHON` at the top of `start_robot_server.sh` if they differ from the defaults, and replace the placeholder `FRANKA_ROBOT_IP` in `ROBOT_IP` with the address of your Franka controller.
2. Put the serial numbers of your cameras in `config/observation.yaml`. `python observation_server.py --list-devices` prints the connected devices.
3. If your end effector frame or stiffness frame differs from the default, set `ee_to_stiffness` and `flange_to_ee` in the config and the matching options of `motion_server.py`.

## Start

Unlock the joints and activate FCI in Franka Desk first, and keep the arm still. Start the three services in three terminals and keep them running.

```bash
bash start_robot_server.sh --check-only   # prints the settings and the launch command, starts nothing
bash start_robot_server.sh                # terminal 1, foreground, Ctrl+C stops it
```

The robot server starts the Franka driver and the Cartesian impedance controller, applies the impedance and collision settings, takes the measured pose as its first target and prints `robot_server_ready`. It does not move the arm at startup. The script sets a stiffness of 2000 N/m and a collision force threshold of 50 N. `motion_server.py` refuses to start when the collision threshold is below its hard force limit of 50 N, so start the server through the script or pass the same options by hand. `robot_control/README.md` lists all settings and the interface.

```bash
python motion_server.py          # terminal 2, does not move the arm at startup
python observation_server.py     # terminal 3
```

Keep the robot still and free of contact while the observation server starts. It averages the wrench for one second to set the zero point.

Then open the agent in this folder and give it the task instruction.

## Commands

```bash
python terminal_command.py delta --translation-m 0.01 0 0 --duration 2 --wait
python terminal_command.py delta --translation-m 0.1 0 0 --duration 4 --next delta --translation-m 0 0.05 0 --duration 2 --wait
python terminal_command.py gripper --position 0 --force-n 30 --wait
python terminal_command.py status
python terminal_command.py stop --wait
python terminal_command.py home --wait
python get_observation.py --observations force eef_pose wrist.rgb third_view.rgb --wrench tcp
python get_observation.py --calibrate
```

Units are metres, radians and seconds. One move is limited to 0.15 m and 90 degrees. The soft force threshold starts at 10 N and can be changed while the server is idle.

```bash
python -c "from terminal_command import MotionClient; print(MotionClient().set_soft_force(15))"
```

`docs/agent_prompt.txt` has the full description of commands, observations, units and frames.

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

77 tests, about one minute, no hardware needed. They need `flask` and `pyyaml`. Seven of them expand the launch file with ROS and are skipped on a machine without the catkin workspace.
