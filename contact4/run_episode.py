#!/usr/bin/env python3
"""Run one simulated episode of Contact Harness: real-robot prompts (ported, prompts/), unmodified
harness (stack/, ports isolated), frozen Codex agent, benchmark success measured by the simulator.

  python run_episode.py --task peg_insertion --seed 0 [--timeout-min 30] [--name RUN]

Outputs in runs/<RUN>/: task.json, agent.jsonl (agent event log: Claude Code stream-json or Codex json), metrics.jsonl
(1 Hz success/progress/force trace), result.json (episode summary), server logs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simctl import rpc  # noqa: E402

TASKS = {
    # task -> (benchmark args, task prompt file, opening phrase)
    "peg_insertion": (["--benchmark", "factory_peg"], "task_prompt_peg_insertion.txt", "start the peg insertion task"),
    "wipe_table": (["--benchmark", "wipe"], "task_prompt_wipe_table.txt", "start the wiping task"),
    "write_board": (["--benchmark", "write"], "task_prompt_write_board.txt",
                    "start the writing task: write the letters ROBOT in uppercase block letters about 5 cm tall "
                    "on the whiteboard"),
    # Pivot: no real-robot counterpart; its task prompt was written for simulation in the format of
    # the other task prompts (prompts/task_prompt_pivot_box.txt), not generated from a real prompt
    "pivot_box": (["--benchmark", "pivot"], "task_prompt_pivot_box.txt",
                  "stand the flat box up on its narrow side against the wall by pushing it with the closed fingers"),
    # Pick-and-place reference tasks: no real-robot counterpart; prompts written for simulation in
    # the Pivot format (prompts/task_prompt_stack_cubes.txt, task_prompt_place_bowl.txt)
    "stack_cubes": (["--benchmark", "stack"], "task_prompt_stack_cubes.txt",
                    "pick up the red cube and place it on top of the green cube"),
    "place_bowl": (["--benchmark", "place"], "task_prompt_place_bowl.txt", "pick up the bowl and put it on the plate"),
}
KICKOFF = ("please read only the instruction in agent_prompt and {prompt} in docs, pay attention to the "
           "important notes, and {phrase}. Do not use any skill in the system")


def monitor(run, stop, out):
    t0 = time.time()
    with open(run / "metrics.jsonl", "w") as f:
        while not stop.is_set():
            try:
                info = rpc("sim_info", timeout=2.0)
                info["wall_s"] = round(time.time() - t0, 2)
                f.write(json.dumps(info) + "\n")
                f.flush()
                out["last"] = info
                if info.get("success") and "first_success_wall_s" not in out:
                    out["first_success_wall_s"] = info["wall_s"]
            except Exception as exc:
                out.setdefault("monitor_errors", []).append(str(exc))
            stop.wait(1.0)


def agent_command(agent, model, kickoff, run):
    stack = str(HERE / "stack")
    if agent == "claude":
        # Claude Code, headless. No session files (--no-session-persistence), no skills/commands, no MCP,
        # no user-level settings/memory; tools: shell + file reading (images via Read); no edits, no web.
        cmd = ["claude", "-p", kickoff, "--output-format", "stream-json", "--verbose",
               "--no-session-persistence", "--disable-slash-commands", "--strict-mcp-config",
               "--setting-sources", "project", "--settings", '{"autoMemoryEnabled": false}',
               "--permission-mode", "dontAsk",
               "--allowedTools", "Bash", "Read", "Glob", "Grep",
               "--disallowedTools", "Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch", "Agent"]
        return cmd + (["--model", model] if model else [])
    # Codex. --ephemeral: no session files in ~/.codex/sessions.
    cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--json", "-C", stack,
           "-s", "workspace-write", "-c", "sandbox_workspace_write.network_access=true",
           "-c", "shell_environment_policy.inherit=all", "-o", str(run / "agent_last.txt"), kickoff]
    return cmd + (["-m", model] if model else [])


def agent_bash_commands(log_path):
    """Shell commands issued by the agent (Claude stream-json tool_use Bash, or Codex command_execution)."""
    out = []
    for line in open(log_path):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "assistant":
            for c in e.get("message", {}).get("content", []):
                if c.get("type") == "tool_use" and c.get("name") == "Bash":
                    out.append(c.get("input", {}).get("command", ""))
        it = e.get("item", {})
        if e.get("type") == "item.started" and it.get("type") == "command_execution":
            out.append(it.get("command", ""))
    return out


NO_RENDER_FLAG = HERE / "NO_RENDER"   # skip videos while this file exists


def render_video(run):
    """Offline multi-view video from the 30 Hz state recording (runs in the simulator venv). Skipped while
    NO_RENDER exists (faster runs); trajectory.npz is still saved, so the video can be
    rendered later with: .venv_sim/bin/python render_video.py <run>."""
    if NO_RENDER_FLAG.exists():
        (run / "render.log").write_text("video skipped: NO_RENDER exists; "
                                        "render later with render_video.py from trajectory.npz\n")
        return
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               MUJOCO_EGL_DEVICE_ID="0", __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if (run / "trajectory.npz").exists():
        subprocess.run([str(HERE / ".venv_sim" / "bin" / "python"), str(HERE / "render_video.py"), str(run)],
                       env=env, cwd=HERE, stdout=open(run / "render.log", "w"), stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-min", type=float, default=30.0)
    ap.add_argument("--name")
    ap.add_argument("--agent", choices=["claude", "codex"], default="claude")
    ap.add_argument("--model", default=None, help="claude: default claude-fable-5-1 (Fable 5.1); codex: config default")
    a = ap.parse_args()
    bench, prompt_file, phrase = TASKS[a.task]
    if a.agent == "claude" and not a.model:
        a.model = "claude-fable-5-1"
    name = a.name or f"ours_{a.task}_s{a.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run = HERE / "runs" / name
    # 1. prompts: place them in the isolated stack copy's docs
    docs = HERE / "stack" / "docs"
    shutil.copy(HERE / "prompts" / "agent_prompt.txt", docs / "agent_prompt.txt")
    shutil.copy(HERE / "prompts" / prompt_file, docs / prompt_file)
    # 2. simulator + unmodified motion/observation servers
    subprocess.run(["bash", str(HERE / "start_episode.sh"), name, "--", *bench, "--seed", str(a.seed)], check=True)
    stop, out = threading.Event(), {}
    mon = threading.Thread(target=monitor, args=(run, stop, out), daemon=True)
    mon.start()
    # 3. frozen agent with the opening phrase
    kickoff = KICKOFF.format(prompt=prompt_file.removesuffix(".txt"), phrase=phrase)
    env = dict(os.environ, PATH=f"{HERE / '.venv' / 'bin'}:{os.environ['PATH']}", PYTHONDONTWRITEBYTECODE="1")
    for k in [k for k in env if k.startswith(("CLAUDECODE", "CLAUDE_CODE_"))]:
        env.pop(k)                      # the agent is a fresh, independent Claude Code process
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"   # never write memory under ~/.claude
    cmd = agent_command(a.agent, a.model, kickoff, run)
    t0 = time.time()
    status = "finished"
    with open(run / "agent.jsonl", "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE / "stack",
                             stdin=subprocess.DEVNULL)
        try:
            p.wait(timeout=a.timeout_min * 60)
        except subprocess.TimeoutExpired:
            status = "timeout"
            p.terminate()
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
    wall = time.time() - t0
    time.sleep(2)
    stop.set()
    mon.join(5)
    final = out.get("last", {})
    n_cmd = sum(1 for c in agent_bash_commands(run / "agent.jsonl") if "terminal_command.py" in c)
    result = {"task": a.task, "seed": a.seed, "method": "ours", "agent": a.agent, "model": a.model, "agent_status": status, "agent_exit": p.returncode,
              "wall_s": round(wall, 1), "success": bool(final.get("success")),
              "first_success_wall_s": out.get("first_success_wall_s"), "progress": final.get("progress"),
              "metrics": final.get("metrics"), "motion_commands": n_cmd, "kickoff": kickoff,
              "monitor_errors": out.get("monitor_errors", [])[:5]}
    (run / "result.json").write_text(json.dumps(result, indent=2))
    try:
        rpc("sim_dump_recording", {"path": str(run / "trajectory.npz")}, timeout=30)
    except Exception as exc:
        print("recording dump failed:", exc)
    subprocess.run(["bash", str(HERE / "stop_sim.sh"), name])
    render_video(run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
