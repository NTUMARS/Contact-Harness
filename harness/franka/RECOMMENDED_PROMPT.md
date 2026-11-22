# Recommended prompt: Franka real robot, single machine

The robot server, the motion server and the observation server all run on the Franka control PC. Open a terminal coding agent on the control PC, or on a machine with key-based SSH to it, and paste the prompt below. First fill in the `<...>` fields.

Verified on 2026-11-22: Franka Panda, Ubuntu 20.04 with an RT kernel, ROS Noetic and franka_ros 0.10.1 already installed, one D435i camera. The +10 cm move ran 400 of 400 steps in 8.04 s for 8 s nominal.

**Before pasting (by a person):**
- In Franka Desk, unlock the joints and activate FCI.
- Keep at least 10 cm free above the gripper.
- Keep someone at the user stop.
- Plug the camera(s) into the control PC over USB 3.

```text
Deploy Contact Harness on the Franka control PC as a single-machine setup: robot server, motion server and observation server all on that PC. The code is in <REPO on the control PC>, the robot is at <ROBOT_IP>, and ROS Noetic with franka_ros is already installed. [If you are not on the control PC: reach it with ssh <user>@<host>.] Do three things:
1. Deploy the code and its controller (third_party/franka_controllers) as described in harness/franka/README.md. Do not modify the existing franka_ros workspace; build the controller in an overlay workspace.
2. A RealSense camera is connected to the control PC. Detect its serial number automatically, configure it in config/observation.yaml, and check that the observation server reads all data correctly.
3. When everything is running, command the arm to move up 10 cm and check that it executes.
Write every problem that needs a person into a markdown file in the project folder.
```

## If the agent gets stuck

These are the points our run had to work out:

- **`conda create` hangs:** create a venv from an existing Python 3.10 instead (`python3.10 -m venv .venv`).
- **Timing tests fail, or motion is slower than nominal:** check each core's frequency. On our PC, CPUs 0–3 and 8–11 were stuck at 400 MHz. Pin the robot and motion servers to the fast cores with `taskset -c 4-7`.
- **A service stays tied to the ssh session:** start the services in the background with `setsid -f ... < /dev/null`.
- **The robot server stops (User Stop, reflex):** restart all three services in order. The motion server takes its startup pose as its reference.

To run a real task afterwards, open the agent in `harness/franka/` and use the official instruction from README.md. The task prompts also need a wrist camera.
