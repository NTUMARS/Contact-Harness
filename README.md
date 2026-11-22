# Contact Harness

Contact Harness lets a frozen vision-language model carry out contact-rich manipulation on a real robot. The model reads images, force and, where the robot has tactile sensors, tactile feedback, decides one short motion at a time, and each motion is executed under compliant control with force guards.

This repository contains the harness for a Franka robot and Contact-4, a simulated benchmark of contact-rich manipulation.

| Folder | Content |
|---|---|
| `harness/franka/` | Harness for a Franka Panda with the Franka gripper. Wrist and third-view RGB-D, estimated external wrench |
| `contact4/` | Contact-4 benchmark. Four simulated contact-rich tasks, Insert, Wipe, Write and Pivot, with two pick-and-place reference tasks and 50 demonstrations per task |
| `harness/rokae/` | Prompts of the whiteboard wiping and writing tasks on the Rokae arm |
| `third_party/` | Controller package for the Franka robot |

The real robot and the simulation expose the same command line interface and the same prompt format, so a skill written for one is easy to move to the other.

## How it works

Three services run in the background. On the real robot, we recommend starting them in three separate terminals, by hand or by an agent, and keeping them running in the background for the whole task.

- **Robot server.** Low-level hardware access. It is specific to the robot and is described in `harness/franka/README.md`.
- **`motion_server.py`.** Turns each command into guarded motion. It validates the command, limits speed, step size and workspace, and watches the measured force during execution. When the force along an axis exceeds the soft threshold, further motion into the contact along that axis is clipped. When it exceeds the hard threshold, motion stops.
- **`observation_server.py`.** Keeps the latest sample from every camera and sensor and serves them on request.

The agent is a terminal coding agent that can run shell commands and view images. It works in the harness folder and repeats one loop.

```
get observations  ->  choose command arguments  ->  execute with --wait  ->  repeat
```

```bash
python get_observation.py --observations force eef_pose wrist.rgb
python terminal_command.py delta --translation-m 0 0 -0.01 --duration 2 --wait
```

## Setup

The quickest way to set up either environment is to let a terminal coding agent (Claude Code, Codex, ...) do it from one prompt. We provide a recommended prompt for each environment, together with the points our own runs had to work out:

| Environment | Recommended prompt | Requirements |
|---|---|---|
| Contact-4 simulation | [`contact4/RECOMMENDED_PROMPT.md`](contact4/RECOMMENDED_PROMPT.md) | Linux with an NVIDIA GPU (EGL), `ffmpeg`, Python 3.10 |
| Franka real robot, single machine | [`harness/franka/RECOMMENDED_PROMPT.md`](harness/franka/RECOMMENDED_PROMPT.md) | Franka control PC with an RT kernel, ROS Noetic, franka_ros, one RealSense camera |

Fill in the `<...>` fields, open the agent at the repository root (simulation) or on the control PC (real robot), paste the prompt, and the agent installs the dependencies, starts the services and runs a first check. For the real robot, unlock the joints and activate FCI in Franka Desk first, and keep someone at the user stop.

Each file ends with an "If the agent gets stuck" list.

The detailed deployment instructions are kept in the README of each environment, not here: [`contact4/README.md`](contact4/README.md) for the simulation and [`harness/franka/README.md`](harness/franka/README.md) for the real robot. They are what the recommended prompts point the agent to, and they are the reference for setting up by hand.

## Prompts and skills

The harness has a `docs/` folder that the agent reads at the start of a task.

- `agent_prompt.txt` describes the system, the commands, the observations with their units and frames, and general operating rules.
- `task_prompt_*.txt` is a textual skill for one task. It gives the goal, the observations to use, the required order of phases, and how to interpret feedback in each phase.

| Where | Skills |
|---|---|
| `harness/franka/docs/` | cable insertion, peeling, pivoting |
| `contact4/prompts/` | peg insertion, wiping, writing, pivoting, and the two pick-and-place reference tasks |

A task is started by opening the agent in the harness folder and typing one instruction, for example

```
Do not use any skill in the system. Please read only the instruction in agent_prompt
and task_prompt_cable_insertion in the docs/ folder, then insert the USB cable.
```

The experiments used the Codex CLI with GPT-6 Astra at high reasoning effort.

## Contact-4 benchmark

`contact4/` runs the harness in MuJoCo, so the method can be tried without a robot.

```bash
cd contact4
bash setup.sh
CODEX_HOME="$PWD/.codex_home" codex login
bash run_codex_episode.sh peg_insertion 0
```

The benchmark also serves learned policies, which train on its demonstrations. See `contact4/README.md` for the tasks, the success conditions, the evaluation protocol and the demonstrations.

## Tests

The Franka harness comes with offline tests that need no robot or camera.

```bash
cd harness/franka && PYTHONPATH=. python -m unittest discover -s tests   # 77 tests
```

The tests need `flask` and `pyyaml`.

## Safety

The harness moves a real robot. The force guards limit motion into contact but do not guarantee a force cap. Keep the emergency stop within reach, check the workspace before each task, and read `harness/franka/README.md` before the first run.

## License

Our code, prompts and skills are released under the MIT License, see `LICENSE`. The demonstrations are released under CC BY 4.0.

Material from other projects keeps its own license, and the license text sits next to the files.

| Files | Origin | License |
|---|---|---|
| `third_party/franka_controllers/` | Adapted from an open-source Franka controller package, reduced, with one controller added | MIT |
| `harness/franka/robot_control/impedance.launch`, `franka_control_node.yaml` | Adapted from [franka_ros](https://github.com/frankaemika/franka_ros) 0.10.2 | Apache-2.0 |
| `contact4/assets/peg_hole/` | Peg and hole of the Factory tasks of NVIDIA, converted to MuJoCo meshes | BSD-3-Clause |
| `contact4/assets/objects/` | Bowl and plate of [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | MIT |

## Acknowledgments

The simulation runs on [MuJoCo](https://mujoco.org) and [robosuite](https://robosuite.ai), which are installed as dependencies and not included here. Wipe and Stack are robosuite environments. Insert follows the Factory peg insertion task of Isaac Lab, Write is adapted from the DrawSVG task of ManiSkill3, and Place uses objects of LIBERO.
