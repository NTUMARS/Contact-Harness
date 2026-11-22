#!/usr/bin/env python3
"""Episodes of Contact Harness in an interactive Claude Code window.

  python episode_ctl.py start --task peg_insertion --seed 0   # simulator + servers + prompts + 1 Hz monitor
  cd stack && claude --model claude-fable-5-1                 # (your terminal) a NEW Claude Code conversation
      -> type your opening prompt; when the agent is done, quit the window   (agent_cc.sh = isolated variant)
  python episode_ctl.py stop                                  # verdict, recording, video, transcript copy

The verdict is the simulator's success test. stop prints the next name for the conversation
(succeed<N> / fail<N>, numbered separately per task); renaming it (/rename in that session) is up to you.
Runs: runs/contact_harness/<task>/<run name>/ (task.json, metrics.jsonl, result.json, trajectory.npz, episode.mp4,
transcript.jsonl = copy of the agent's Claude Code conversation).
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from simctl import rpc  # noqa: E402
from run_episode import TASKS, agent_bash_commands, render_video  # noqa: E402

STATE = HERE / "runs" / "_current.json"          # the episode in progress
STOPPING = HERE / "runs" / "_stopping.json"     # claimed by the one stop in progress (a second stop is a no-op)
COUNTERS = HERE / "runs" / "_counters.json"      # {task: {"succeed": n, "fail": m}}
AGENT_CONFIG = HERE / ".agent_cc"                # CLAUDE_CONFIG_DIR of the agent window (agent_cc.sh)
STACK_DOCS = HERE / "stack" / "docs"
# the agent's private temporary directory (its TMPDIR; the guard allows no other /tmp path except its own Claude Code
# scratchpad): emptied at start, collected into the run at stop, so nothing passes between runs
AGENT_TMP = HERE / "stack" / "tmp"
# runs/<GROUP>/<task>/; set by the driver through SIM_RUN_GROUP
GROUP = os.environ.get("SIM_RUN_GROUP", "contact_harness")
# Ablations: the servers get SIM_ABLATION (run_motion_server.py, run_observation_server.py),
# never the agent
ABLATIONS = ("none", "no_force", "no_tactile", "no_side_camera", "no_soft_limit", "no_force_tactile",
             "soft_limit_stop", "no_force_limit", "no_compliance_limit")
# One sentence appended to the end of the task prompt, for these ablations only; every other run gets the
# prompts unchanged
ABLATION_NOTE = {
    "no_force_tactile": "In this configuration, force and tactile feedback are not available; continue the task with "
                        "the remaining observations.",
    "soft_limit_stop": "In this configuration, a motion stops as soon as the contact force exceeds the soft force "
                       "limit, instead of clipping translation into contact.",
    # same form as no_force_tactile
    "no_side_camera": "In this configuration, the right wrist camera (right_wrist.rgb, right_wrist.depth) is not "
                      "available; continue the task with the remaining observations.",
    # "w/o soft force limit": no force limit in execution
    "no_force_limit": "In this configuration, motions are executed to completion without any force limit.",
    # "w/o compliant execution and soft force limit"
    "no_compliance_limit": "In this configuration, motions are executed with stiff position control instead of "
                           "impedance control, and run to completion without any force limit.",
}
# Files left from a previous episode (docs copies, observation captures) are moved here; nothing is deleted
LEFTOVERS = HERE / "runs" / "_leftovers"


def _use_token(run):
    """sim_* RPCs need the episode's token (sim_robot_server.py); our own processes read it from the run directory."""
    p = Path(run) / ".sim_rpc_token"
    if p.exists():
        os.environ["SIM_RPC_TOKEN"] = p.read_text().strip()


def _platform_failure(paths):
    """The agent session ENDED on an API/platform error (quota, rate limit, outage), not on a task decision: the last
    stream result is an error, or the last assistant message is an API error. Such an episode is not an attempt."""
    last = None
    for p in paths:
        if not p or not Path(p).exists():
            continue
        for line in open(p, errors="ignore"):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "result" or (e.get("type") == "assistant" and isinstance(e.get("message"), dict)):
                last = e
    if last is None:
        return None
    if last.get("type") == "result":
        bad = bool(last.get("is_error")) or last.get("subtype") not in (None, "success")
        text = str(last.get("result") or last.get("subtype") or "")
    else:
        bad = bool(last.get("isApiErrorMessage") or last.get("error"))
        text = json.dumps(last.get("message", {}).get("content", ""))[:500]
    if bad:
        return text[:300] or "agent session ended with an error"
    return None


def _codex_stream(path):
    """Codex `exec --json` event stream: thread id, motion commands, and an API/platform error that ended the run
    (last turn failed or a top-level error event). None if the file is not a Codex stream."""
    path = Path(path)
    if not path.exists():
        return None
    tid, last, codex = None, None, False
    for line in open(path, errors="ignore"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        t = e.get("type", "")
        if t == "thread.started":
            tid, codex = e.get("thread_id"), True
        if t in ("turn.completed", "turn.failed", "error"):
            last = e
    if not codex:
        return None
    err = None
    if last is not None and last.get("type") in ("turn.failed", "error"):
        err = json.dumps(last.get("error") or last.get("message") or last)[:300]
    n = sum(1 for c in agent_bash_commands(path) if "terminal_command.py" in c)
    return {"thread_id": tid, "n_cmd": n, "error": err}


def _monitor_main(run, limit_s):
    """Detached 1 Hz trace of the simulator (success, progress, force) until the run is stopped; at the
    time limit it ends the episode itself (stop --reason timeout, output in <run>/stop.log)."""
    run = Path(run)
    _use_token(run)
    t0 = time.time()
    with open(run / "metrics.jsonl", "a") as f:
        while not (run / "STOP").exists():
            if time.time() - t0 >= limit_s:
                subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "stop", "--reason", "timeout"],
                                 stdout=open(run / "stop.log", "w"), stderr=subprocess.STDOUT,
                                 cwd=HERE, start_new_session=True)
                return
            try:
                info = rpc("sim_info", timeout=2.0)
                info["wall_s"] = round(time.time() - t0, 2)
                f.write(json.dumps(info) + "\n")
                f.flush()
            except Exception as exc:  # noqa: BLE001
                f.write(json.dumps({"monitor_error": str(exc), "wall_s": round(time.time() - t0, 2)}) + "\n")
            time.sleep(1.0)


def start(a):
    if STATE.exists():
        raise SystemExit(f"an episode is already running ({json.loads(STATE.read_text())['run']}); stop it first")
    if STOPPING.exists():
        raise SystemExit("an episode is still being stopped (runs/_stopping.json); wait, or abort it")
    bench, prompt_file, phrase = TASKS[a.task]
    name = f"{a.task}_s{a.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run = HERE / "runs" / GROUP / a.task / name                   # runs/<GROUP>/<task>/
    # prompts: the agent's docs/ holds ONLY this task's prompts
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for f in STACK_DOCS.glob("*.txt"):
        (LEFTOVERS / stamp / "docs").mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(LEFTOVERS / stamp / "docs" / f.name))
    shutil.copy(HERE / "prompts" / "agent_prompt.txt", STACK_DOCS / "agent_prompt.txt")
    shutil.copy(HERE / "prompts" / prompt_file, STACK_DOCS / prompt_file)
    if a.ablation in ABLATION_NOTE:
        text = (STACK_DOCS / prompt_file).read_text().rstrip("\n")
        (STACK_DOCS / prompt_file).write_text(text + "\n\n" + ABLATION_NOTE[a.ablation] + "\n")
    # no observations left over from the previous episode: the agent may only see its own captures
    for f in (HERE / "stack" / "observations" / "latest").glob("*"):
        if f.is_file():
            (LEFTOVERS / stamp / "observations").mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(LEFTOVERS / stamp / "observations" / f.name))
    # the agent's private temporary directory starts empty (leftovers from an interrupted run go to runs/_leftovers)
    if AGENT_TMP.exists() and any(AGENT_TMP.iterdir()):
        dst = LEFTOVERS / "agent_tmp" / time.strftime("%Y%m%d_%H%M%S")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(AGENT_TMP), str(dst))
    AGENT_TMP.mkdir(exist_ok=True)
    # per-episode token for the simulator's sim_* RPCs: given to the simulator and our own processes, never to the agent
    run.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    (run / ".sim_rpc_token").write_text(token + "\n")
    os.chmod(run / ".sim_rpc_token", 0o600)
    os.environ["SIM_RPC_TOKEN"] = token
    if a.ablation != "none":
        os.environ["SIM_ABLATION"] = a.ablation
        (run / "ablation.json").write_text(json.dumps({"ablation": a.ablation,
                                                       "prompt_note": ABLATION_NOTE.get(a.ablation)}) + "\n")
    # simulator + unmodified motion/observation servers (start_episode.sh takes a name under runs/)
    subprocess.run(["bash", str(HERE / "start_episode.sh"), f"{GROUP}/{a.task}/{name}", "--", *bench, "--seed", str(a.seed)],
                   check=True)
    # the token check must be active: a sim_* call without the token has to be refused
    saved = os.environ.pop("SIM_RPC_TOKEN")
    try:
        rpc("sim_info", timeout=5.0)
        leaked = True
    except Exception:  # noqa: BLE001
        leaked = False
    os.environ["SIM_RPC_TOKEN"] = saved
    if leaked:
        subprocess.run(["bash", str(HERE / "stop_sim.sh"), f"{GROUP}/{a.task}/{name}"])
        raise SystemExit("sim_* RPCs answer without the token: refusing to start (simulator not protected)")
    mon = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_monitor", str(run), str(a.time_limit_min * 60)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    STATE.write_text(json.dumps({"task": a.task, "seed": a.seed, "run": str(run), "monitor_pid": mon.pid,
                                 "started_unix": t0, "prompt_file": prompt_file,
                                 "time_limit_min": a.time_limit_min, "ablation": a.ablation}, indent=2))
    print(f"\nEpisode ready: {run}\n"
          f"Task prompt in the agent's docs/: {prompt_file} (+ agent_prompt.txt)\n"
          f"Opening phrase for this task: \"{phrase}\"\n"
          f"Now open a NEW Claude Code conversation in the agent's directory (model Claude Fable 5.1):\n"
          f"    cd \"{HERE / 'stack'}\" && TMPDIR=\"$PWD/tmp\" CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 claude --model claude-fable-5-1 --settings ../agent_settings.json\n"
          f"Time limit {a.time_limit_min:g} min: then the episode ends by itself (verdict in <run>/stop.log).\n"
          f"When the agent is done, quit that window and run:  .venv/bin/python {Path(__file__).name} stop")


def _newest_transcript(since):
    """The agent's conversation: the newest Claude Code session started in stack/ during this episode,
    from a normal Claude Code window (~/.claude/projects, read only) or agent_cc.sh (.agent_cc/projects)."""
    stack = str(HERE / "stack")
    files = []
    for proj in (Path.home() / ".claude" / "projects", AGENT_CONFIG / "projects"):
        if not proj.exists():
            continue
        for p in proj.glob("*/*.jsonl"):
            if p.stat().st_mtime < since:
                continue
            with open(p) as f:
                head = [next(f, "") for _ in range(20)]
            for x in head:
                try:
                    if x.strip().startswith("{") and json.loads(x).get("cwd") == stack:
                        files.append(p)
                        break
                except ValueError:          # a line still being written by a live session
                    continue
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def stop(a):
    # claim the episode atomically: the time-limit stop and the driver's stop can race; only one may count it
    try:
        os.rename(STATE, STOPPING)
    except FileNotFoundError:
        print("no episode in progress (already stopped or being stopped)")
        return
    st = json.loads(STOPPING.read_text())
    run, task = Path(st["run"]), st["task"]
    _use_token(run)
    try:
        final = rpc("sim_info", timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        final = {"error": str(exc)}
    (run / "STOP").touch()
    time.sleep(2)
    metrics = [json.loads(x) for x in open(run / "metrics.jsonl")] if (run / "metrics.jsonl").exists() else []
    first = next((m["wall_s"] for m in metrics if m.get("success")), None)
    success = bool(final.get("success"))
    tr = _newest_transcript(st["started_unix"])
    if tr is not None:
        shutil.copy(tr, run / "transcript.jsonl")
    if AGENT_TMP.exists() and any(AGENT_TMP.iterdir()):     # the agent's temporary files belong to this run
        shutil.move(str(AGENT_TMP), str(run / "agent_tmp"))
    AGENT_TMP.mkdir(exist_ok=True)
    platform = _platform_failure([run / "agent_stream.jsonl", run / "transcript.jsonl" if tr else None])
    n_cmd = sum(1 for c in agent_bash_commands(run / "transcript.jsonl") if "terminal_command.py" in c) if tr else None
    codex = _codex_stream(run / "agent_stream.jsonl") if tr is None else None
    if codex:                            # Codex run
        n_cmd, platform = codex["n_cmd"], platform or codex["error"]
    label = None
    if not platform and not GROUP.startswith("diag_"):   # a platform failure is not an attempt: no number, not
        # counted; diagnostic runs (diag_*) are never counted either
        counters = json.loads(COUNTERS.read_text()) if COUNTERS.exists() else {}
        c = counters.setdefault(task if st.get("ablation", "none") == "none" else f"{task}@{st['ablation']}",
                                {"succeed": 0, "fail": 0})
        key = "succeed" if success else "fail"
        c[key] += 1
        label = f"{key}{c[key]}"
        COUNTERS.write_text(json.dumps(counters, indent=2))
    result = {"task": task, "seed": st["seed"], "method": "ours", "mode": "interactive Claude Code session",
              "ended_by": a.reason, "time_limit_min": st.get("time_limit_min"),
              "label": label, "success": success, "wall_s": round(time.time() - st["started_unix"], 1),
              "first_success_wall_s": first, "progress": final.get("progress"), "metrics": final.get("metrics"),
              "motion_commands": n_cmd, "session_id": tr.stem if tr else (codex or {}).get("thread_id"),
              "transcript": str(tr) if tr else None, "platform_failure": platform,
              "ablation": st.get("ablation", "none")}
    (run / "result.json").write_text(json.dumps(result, indent=2))
    try:
        rpc("sim_dump_recording", {"path": str(run / "trajectory.npz")}, timeout=30)
    except Exception as exc:  # noqa: BLE001
        print("recording dump failed:", exc)
    subprocess.run(["bash", str(HERE / "stop_sim.sh"), str(run.relative_to(HERE / "runs"))])
    try:
        os.kill(st["monitor_pid"], 15)
    except OSError:
        pass
    (LEFTOVERS / "markers").mkdir(parents=True, exist_ok=True)   # the marker is moved, not deleted
    shutil.move(str(STOPPING), str(LEFTOVERS / "markers" / f"{run.name}_stopping.json"))
    render_video(run)
    print(json.dumps(result, indent=2))
    if platform:
        dst = HERE / "runs" / "_aborted"
        dst.mkdir(parents=True, exist_ok=True)
        (run / "ABORTED.txt").write_text(f"platform failure {time.strftime('%Y-%m-%d %H:%M:%S')}: {platform}\n")
        shutil.move(str(run), str(dst / run.name))
        print(f"\nPLATFORM FAILURE (not counted, moved to {dst / run.name}): {platform}")
        return
    if a.reason == "timeout":
        print(f"\nTime limit reached ({st.get('time_limit_min')} min): episode ended; quit the agent window.")
    print(f"\nVerdict: {'SUCCESS' if success else 'FAIL'}  ->  rename the conversation to: {label}"
          + (f"   (session {tr.stem}: in stack/, claude --resume {tr.stem}, then /rename {label})" if tr else
             "   (no Claude Code session started in stack/ found for this episode)"))


def abort(a):
    """Discard the episode in progress (setup problem, not an agent attempt): no verdict, no number.
    Its folder goes to runs/_aborted/."""
    state = STATE if STATE.exists() else STOPPING
    if not state.exists():
        raise SystemExit("no episode in progress")
    st = json.loads(state.read_text())
    run = Path(st["run"])
    (run / "STOP").touch()
    subprocess.run(["bash", str(HERE / "stop_sim.sh"), str(run.relative_to(HERE / "runs"))])
    try:
        os.kill(st["monitor_pid"], 15)
    except OSError:
        pass
    (LEFTOVERS / "markers").mkdir(parents=True, exist_ok=True)   # moved, not deleted
    shutil.move(str(state), str(LEFTOVERS / "markers" / f"{run.name}_current_aborted.json"))
    if AGENT_TMP.exists() and any(AGENT_TMP.iterdir()):
        shutil.move(str(AGENT_TMP), str(run / "agent_tmp"))
    AGENT_TMP.mkdir(exist_ok=True)
    dst = HERE / "runs" / "_aborted"
    dst.mkdir(parents=True, exist_ok=True)
    (run / "ABORTED.txt").write_text(f"aborted {time.strftime('%Y-%m-%d %H:%M:%S')}: {a.reason}\n")
    shutil.move(str(run), str(dst / run.name))
    print(f"aborted; moved to {dst / run.name} (not counted)")


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "_monitor":
        return _monitor_main(sys.argv[2], float(sys.argv[3]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--task", choices=sorted(TASKS), required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--time-limit-min", type=float, default=30.0)
    s.add_argument("--ablation", choices=ABLATIONS, default="none")
    sp = sub.add_parser("stop")
    sp.add_argument("--reason", default="user")
    ab = sub.add_parser("abort")
    ab.add_argument("--reason", default="")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "start":
        start(a)
    elif a.cmd == "stop":
        stop(a)
    elif a.cmd == "abort":
        abort(a)
    else:
        print(STATE.read_text() if STATE.exists() else "no episode in progress")
        if STATE.exists():
            _use_token(json.loads(STATE.read_text())["run"])
            print(json.dumps(rpc("sim_info", timeout=5.0), indent=2)[:1500])


if __name__ == "__main__":
    main()
