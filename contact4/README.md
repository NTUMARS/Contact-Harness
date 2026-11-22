# Contact-4

Contact-4 is a simulated benchmark of four contact-rich manipulation tasks, Insert, Wipe, Write and Pivot. Two pick-and-place tasks, Place and Stack, ship with it as reference tasks. All six run in MuJoCo 2.3.2 and robosuite 1.4.0 on a Panda arm under Cartesian impedance control, with force feedback and synthesized tactile feedback.

The benchmark serves two kinds of methods.

- **Training-free agents** act through the observation and motion interface of Contact Harness. The simulation runs the motion and observation services of the harness, with a simulated robot behind them.
- **Learned policies** train on the demonstrations, 50 per task. See [Demonstrations](#demonstrations).

## Tasks

| Task | Name in commands | Goal | Success condition |
|---|---|---|---|
| Insert | `peg_insertion` | Insert the grasped peg into the hole of the block below it. The radial clearance is about 0.06 mm, and a 15 N press-fit resistance acts once the peg is 3 mm in | The horizontal offset between peg and hole is below 2.5 mm and the peg is within 1 mm of the hole bottom |
| Wipe | `wipe_table` | Wipe 100 dirt markers off the table with a pad mounted on the hand | All markers are removed |
| Write | `write_board` | Write the word ROBOT on a whiteboard with a pen held in the gripper. Ink appears where the tip presses with at least 0.5 N | A person can read the word ROBOT on the board |
| Pivot | `pivot_box` | Push a flat box against a low wall until it stands on its narrow side. The box is too wide to grasp | The box tilt is at least 80°, its lowest corner is less than 4 mm above the table, its nearest corner is at most 3 cm from the wall, and it stays still for 2 s without robot contact |
| Place | `place_bowl` | Pick up a bowl and put it on a plate | The bowl touches the plate, its center is within 6 cm of the plate center, and the gripper has released it |
| Stack | `stack_cubes` | Place the red cube on the green cube | The red cube is off the table, touches the green cube, and the gripper has released it |

The simulator checks the success condition during the whole run and reports it in `result.json`. For Write, the simulator reports a stroke-matching test as an indicator, and the benchmark label is a person's reading of the board in the final video.

Each task has 10 evaluation seeds, 0 to 9. The seed fixes every random quantity of the scene, so all methods start from the same initial state. A run ends when the agent stops or after 30 minutes. Success counts only if it is reached within 20 minutes for Place, Stack and Insert and within 30 minutes for Wipe, Write and Pivot.

## Requirements

- Linux with an NVIDIA GPU. Cameras are rendered through EGL. `start_episode.sh` selects the NVIDIA EGL vendor file `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` and device 0.
- Python 3.10
- `ffmpeg` for the episode videos
- A terminal coding agent. The experiments used Codex CLI 0.162.1 with GPT-6 Astra at high reasoning effort, and Claude Code with Claude Fable 5.1 and 5.2 for the comparison of agent models.

Our runs used Ubuntu 22.04 and an RTX 5090.

## Setup

```bash
cd contact4
bash setup.sh
```

The script creates two environments, `.venv` for the motion and observation services and `.venv_sim` for the simulator, and writes the agent settings with the path of this clone. `PYTHON=/path/to/python3.10 bash setup.sh` selects the interpreter.

## Run an episode with Codex

Log in once inside the isolated Codex home that `setup.sh` created.

```bash
CODEX_HOME="$PWD/.codex_home" codex login
```

Then run one task and seed.

```bash
bash run_codex_episode.sh peg_insertion 0
```

The script starts the simulator and the two services, opens a new Codex session in `stack/` with the opening prompt of the task, waits for the agent to finish or for the 30 minute limit, and closes the episode. `AGENT_MODEL` and `AGENT_EFFORT` select another model or reasoning effort.

Results are written to `runs/contact_harness/<task>/<task>_s<seed>_<time>/`.

| File | Content |
|---|---|
| `result.json` | Success, time, number of motion commands, final task progress, contact force metrics |
| `metrics.jsonl` | Success, progress and force sampled at 1 Hz |
| `episode.mp4` | Video of the run from several views |
| `trajectory.npz` | Simulator state recording at 30 Hz |
| `agent_stream.jsonl` | Event stream of the agent |
| `guard.jsonl` | Every tool request of the agent and whether it was allowed |
| `task.json` | Scene parameters of this seed |

## Run an episode with another agent

Any terminal agent that can run shell commands and view images can be used. Start the episode, open the agent in `stack/`, and close the episode when the agent is done.

```bash
.venv/bin/python episode_ctl.py start --task peg_insertion --seed 0

cd stack
TMPDIR="$PWD/tmp" CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 claude --settings ../agent_settings.json
# type the opening prompt, wait for the agent, then quit the window
cd ..

.venv/bin/python episode_ctl.py stop
```

`episode_ctl.py status` shows the state of the running episode, and `episode_ctl.py abort` discards it.

Only one episode runs at a time. The services need a few seconds to release their ports after `stop`, and a start within that time is refused with a `port busy` message.

The opening prompt has the same form for every task.

```
Do not read any previous conversations or any skills stored in the system; read only the prompts under this project path. Read only the instructions in agent_prompt and task_prompt_peg_insertion in docs, pay attention to the important notes, and insert the peg held in the gripper into the hole of the metal block below it.
```

`run_codex_episode.sh` lists the opening prompts of all six tasks.

## What the agent sees

The agent works in `stack/`. At the start of an episode `stack/docs/` receives `agent_prompt.txt` and the prompt of the current task from `prompts/`. The agent then uses the same commands as on the real robot.

```bash
python get_observation.py --observations force eef_pose head.rgb left_wrist.rgb right_wrist.rgb
python terminal_command.py delta --translation-m 0 0 -0.01 --duration 2 --wait
```

The camera names follow our real-robot setup. `head` is a fixed camera in front of the scene, `left_wrist` is the camera on the hand, and `right_wrist` is a fixed camera at the side of the scene.

The agent must not read the simulator state. `agent_guard.py` for Claude Code and `codex_guard.py` for Codex restrict the agent to `stack/`, and the simulator answers its internal requests only with a token that is created per episode and never given to the agent.

## Harness in simulation

`stack/` holds the motion and observation services and the command line clients of the harness, in the version used for the simulated experiments.

- The service ports are 18080 and 18081.
- Inverse kinematics and the client of the robot server are in `stack/robot_support/`, built on Pink and Pinocchio.
- The tactile reference point is the fingertip of the simulated Panda gripper.

`run_motion_server.py` and `run_observation_server.py` start the two services. They register the kinematic model of the Panda and replace the camera and tactile drivers with simulated sources. `sim_task_server.py` plays the role of the robot server.

The task prompts of Insert, Wipe and Write were ported from our real-robot prompts by substituting scene nouns, scene geometry and camera descriptions. The prompts of Pivot, Place and Stack were written for the simulation.

## Ablations

`episode_ctl.py start --ablation <name>` and `ABLATION=<name> bash run_codex_episode.sh <task> <seed>` run an episode with one component removed, for example `no_force_tactile`, `no_side_camera` or `no_compliance_limit`. `episode_ctl.py start --help` lists all names.

## Demonstrations

The benchmark provides 50 teleoperated demonstrations per task, 300 episodes and 93195 frames at 10 Hz in total, in LeRobot v2.1 format. They were collected with a SpaceMouse in the scenes used for evaluation, with seeds from 100 on, which do not overlap the evaluation seeds.

The demonstrations are distributed separately because of their size, 8.6 GB, of which the four contact-rich tasks take 5.6 GB. See `demos/README.md` for the download location and the data format.

## Files

| File | Purpose |
|---|---|
| `setup.sh`, `requirements-*.txt` | Environments and agent settings |
| `run_codex_episode.sh` | One episode with Codex, start to result |
| `episode_ctl.py` | Start, stop, abort and inspect an episode |
| `start_episode.sh`, `stop_sim.sh` | Start and stop the simulator and the two services |
| `sim_task_server.py`, `sim_robot_server.py` | Simulator process with the robot server interface, physics, contact force and tactile contacts, success tests |
| `peg_tasks.py`, `write_tasks.py`, `pivot_tasks.py`, `place_tasks.py` | Scenes of Insert, Write, Pivot and Place. Wipe and Stack use the robosuite environments |
| `run_motion_server.py`, `run_observation_server.py` | Launchers of the harness services on the simulator |
| `stack/` | Harness copy and working directory of the agent |
| `prompts/` | Agent prompt and task skills |
| `agent_guard.py`, `codex_guard.py`, `agent_cc.sh` | Agent isolation |
| `render_video.py`, `render.sh` | Video from a state recording |
| `run_episode.py`, `simctl.py` | Helpers used by the scripts above |
| `assets/` | Meshes and the Panda kinematic model |

## Third-party material

- `assets/peg_hole/` holds the peg and hole of the Isaac Lab Factory PegInsert task, converted to MuJoCo meshes. The collision bore of the hole is set to the nominal 8.1 mm diameter of the Factory configuration. NVIDIA distributes these parts under the BSD-3-Clause license, see `assets/peg_hole/LICENSE.txt`.
- `assets/objects/` holds the bowl and the plate of LIBERO, which is MIT licensed, see `assets/objects/LICENSE.txt`.
- Wipe and Stack are robosuite environments, and Write is adapted from the DrawSVG task of ManiSkill3.
