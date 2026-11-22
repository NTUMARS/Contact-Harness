#!/usr/bin/env python3
"""PreToolUse hook for our method's agent (Claude Code in stack/): keep it on the robot interface.

The agent may use this project directory (stack/: prompts, robot/observation clients, observations), its
private temporary directory stack/tmp (its TMPDIR, emptied before every episode) and its own Claude Code
scratchpad. It may NOT reach simulator internals, which a real robot does not have: the episode's run directory
(ground-truth metrics, simulator logs with object poses), simulator sources, other project folders, the rest of
/tmp (files of earlier runs and of other programs), previous conversations (~/.claude) or process introspection.
Paths are resolved before checking, any absolute path outside the allowed roots is refused, every tool goes through this hook and
only the four tools of the experiment are allowed, and the hook fails closed. The simulator's sim_* RPCs are
additionally protected by a per-episode token (sim_robot_server.py), so a call that slips through cannot read
the simulator state either.
Exit code 2 blocks the call and returns the message to the agent.
"""
import json
import os
import re
import sys

STACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stack")
# the project may be reached through a symlink; Claude Code then reports and uses the real path, so the real path
# of the project is allowed as well
STACK_REAL = os.path.realpath(STACK)
TOOLS = {"Bash", "Read", "Glob", "Grep", "TodoWrite", "ToolSearch"}     # TodoWrite/ToolSearch touch no files
MSG = ("Blocked by the experiment setup: only this project directory is accessible; use ./tmp (your TMPDIR) or "
       "your own scratchpad for temporary files. Simulator internals, logs, other directories, the rest of /tmp, "
       "process information and previous conversations are not part of the robot interface.")
BIN_OK = ("/dev/null", "/dev/stdout", "/dev/stderr", "/usr/bin/", "/bin/", "/usr/local/bin/")
DENY = [
    r"(^|[^\w.])\.\.([^\w.]|$)", r"\bpardir\b", r"\bdirname\b", r"\brealpath\b", r"\breadlink\b",
    r"\$\{?(PWD|OLDPWD|HOME)\b", r"(^|[\s;&|=:])~(/|\s|$|[;&|)])", r"\bexpanduser\b", r"\bcd\s*($|[;&|)])", r"\bcd\s+-", r"\bcd\s+[\"']?[/$]",
    r"\b(pushd|popd|chdir)\b", r"\b(find|ls)\s+/(\s|$)", r"\b(Path|PurePath|listdir|scandir|walk|i?glob|open)\s*\(\s*f?['\"]/['\"]",
    r"\b(ps|pgrep|pkill|pwdx|lsof|ss|netstat|env|printenv|history|mount|ln|strace|gdb|locate|journalctl)\b(?!\s*=)",
    r"\b(environ|getenv)\b", r"\bfind\b[^|;&]*\s-L\b",
    r"\b(zmq|msgpack|socket|nc|ncat|telnet|curl|wget|urllib|requests|http\.client)\b",
    r"\b(mujoco|robosuite|importlib|__import__|base64|b64decode|codecs)\b", r"\b(exec|eval)\s*\(",
    r"(?<![\d.])(5655|5656|4242|18080|18081|18090)(?![\d.])", r"\bsim_\w+", r"\bsimctl\b",
    r"\.claude\b", r"SIM_STATUS", r"_current\.json", r"_counters\.json", r"metrics\.jsonl", r"sim_robot\.log",
    r"robosuite\.log", r"trajectory\.npz", r"task\.json", r"\.sim_rpc_token", r"SIM_RPC_TOKEN",
]


def roots(session_id):
    """Allowed directories: the project and the agent's own Claude Code scratchpad of this session."""
    r = [STACK]
    if session_id and re.fullmatch(r"[A-Za-z0-9-]+", session_id):
        tmp = os.environ.get("TMPDIR") or "/tmp"
        for slug in dict.fromkeys(re.sub(r"[^A-Za-z0-9]", "-", x) for x in (STACK, STACK_REAL)):
            r += [f"/tmp/claude-{os.getuid()}/{slug}/{session_id}", f"{tmp.rstrip('/')}/claude-{os.getuid()}/{slug}/{session_id}"]
    return r


def inside(path, cwd, session_id):
    p = os.path.realpath(os.path.join(cwd or STACK, os.path.expanduser(path)) if not os.path.isabs(path) else path)
    return any(p == r or p.startswith(r + "/") for r in (os.path.realpath(x) for x in roots(session_id)))


def check_bash(cmd, session_id):
    t = cmd.replace(STACK, "<STACK>").replace(STACK_REAL, "<STACK>")
    for r in roots(session_id)[1:]:
        t = t.replace(r, "<SCRATCH>")
    if re.search(r"<(STACK|SCRATCH)>/?\.\.", t) or any(re.search(p, t) for p in DENY):
        return False
    # every absolute path must be inside the allowed roots (already replaced) or a plain system binary / device
    for m in re.finditer(r"(?<![\w.:/<>-])/(?=[A-Za-z_~.*?\[])[^\s'\"`;|&)<>]*", t):
        if not m.group(0).startswith(BIN_OK):
            return False
    return True


def allowed(tool, inp, cwd=STACK, session_id=""):
    if tool not in TOOLS:
        return False
    if tool == "Bash":
        return check_bash(inp.get("command", ""), session_id)
    if tool == "Read":
        return inside(inp.get("file_path", ""), cwd, session_id)
    if tool in ("Glob", "Grep"):
        pat = [inp.get("glob") or ""] + ([inp.get("pattern") or ""] if tool == "Glob" else [])
        if any(re.search(r"(^|/)\.\.(/|$)", x) or x.startswith(("/", "~")) for x in pat):
            return False
        return inside(inp.get("path") or cwd or STACK, cwd, session_id)
    return True


def main():
    try:
        data = json.load(sys.stdin)
        ok = allowed(data.get("tool_name", ""), data.get("tool_input") or {}, data.get("cwd") or STACK,
                     str(data.get("session_id") or ""))
    except Exception:  # noqa: BLE001  fail closed: an unreadable request is refused
        ok = False
    if not ok:
        print(MSG, file=sys.stderr)
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
