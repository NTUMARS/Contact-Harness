#!/usr/bin/env python3
"""PreToolUse hook for our method's agent when it runs as Codex (GPT-6 Astra).

Same policy as agent_guard.py (the Claude Code guard): the agent may use only the project directory stack/
(prompts, robot/observation clients, observations, its private TMPDIR stack/tmp). It may not reach simulator
internals, logs, other directories, the rest of /tmp, process information or previous conversations. The checks
themselves are agent_guard's; this file only maps Codex's tool names onto them:
  Bash (shell / unified exec)  -> agent_guard.check_bash on tool_input.command
  view_image                   -> the image path must be inside stack/
  update_plan, clocksleep      -> allowed (touch no files)
  anything else (apply_patch, MCP, agents, ...) -> refused (fails closed, like agent_guard)
Every request and decision is appended to $CODEX_GUARD_LOG when it is set (audit trail in the run directory).
Exit code 2 blocks the call and returns the message to the agent.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_guard as g  # noqa: E402


def allowed(tool, inp, cwd):
    if tool == "Bash":
        cmd = inp.get("command", "")
        if isinstance(cmd, list):
            cmd = " ".join(str(x) for x in cmd)
        return g.check_bash(str(cmd), "")
    if tool == "view_image":
        return g.inside(str(inp.get("path", "")), cwd, "")
    if tool in ("update_plan", "clocksleep"):   # clocksleep: Codex's wait tool, touches no files
        return True                            # (Bash `sleep` is allowed as well)
    return False


def main():
    data = {}
    try:
        data = json.load(sys.stdin)
        ok = allowed(data.get("tool_name", ""), data.get("tool_input") or {}, data.get("cwd") or g.STACK)
    except Exception:  # noqa: BLE001  fail closed: an unreadable request is refused
        ok = False
    log = os.environ.get("CODEX_GUARD_LOG")
    if log:
        try:
            with open(log, "a") as f:
                f.write(json.dumps({"t": time.time(), "ok": ok, "tool": data.get("tool_name"),
                                    "input": data.get("tool_input")}) + "\n")
        except OSError:
            pass
    if not ok:
        print(g.MSG, file=sys.stderr)
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
