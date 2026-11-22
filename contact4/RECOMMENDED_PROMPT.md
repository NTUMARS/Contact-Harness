# Recommended prompt: Contact-4 simulation

Open a terminal coding agent (Claude Code, Codex, …) at the repository root and paste the prompt below. You need Linux with an NVIDIA GPU (EGL), `ffmpeg` and any Python 3.10 interpreter.

Verified on 2026-11-22 (RTX 5080, Claude Code): `place_bowl` seed 0 succeeded in about 8 min.

```text
This is the Contact Harness repository. Use the Contact-4 simulation in contact4/ to run the pick-and-place task place_bowl (seed 0), and give me a demo when it is done.
- Set it up as described in contact4/README.md. setup.sh needs Python 3.10; an existing interpreter, for example from a conda env, works: PYTHON=<python3.10> bash setup.sh.
- Start the episode with `.venv/bin/python episode_ctl.py start --task place_bowl --seed 0 --time-limit-min 90`. Then act as the agent yourself: work only in contact4/stack/, follow stack/docs/agent_prompt.txt and the task prompt there, and use only the harness CLI (get_observation.py, terminal_command.py, agent_request.py, with ../.venv/bin on PATH). Do not read the simulator's internal state.
- When the task is done, run `.venv/bin/python episode_ctl.py stop`, report result.json, and give me episode.mp4 plus a short version without the idle frames.
```

Change `place_bowl` to `stack_cubes`, `peg_insertion`, `wipe_table`, `write_board` or `pivot_box` to run another task.

## If the agent gets stuck

These are the points our run had to work out:

- One move is limited to 0.15 m. A `--next` chain with a longer leg is rejected as a whole.
- The wrist-camera intrinsics are not given. Estimate the focal length from two known hand moves (about 310 px), then locate objects from the wrist depth image.
- Lower the soft force threshold temporarily for contact: `agent_request.py set-soft-force --newtons 3`. Restore 10 afterwards.
- The video has a text overlay at the bottom, so `ffmpeg mpdecimate` drops nothing. Compare frames without the bottom strip.

For a benchmark-grade isolated agent instead, use `bash run_codex_episode.sh place_bowl 0` (see contact4/README.md). Credentials saved in `contact4/.codex_home/` and `contact4/.agent_cc/` must be left out when you copy or zip the folder.
