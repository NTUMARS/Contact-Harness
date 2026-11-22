#!/usr/bin/env bash
# One-time setup of Contact-4: two Python environments and the agent settings.
#   bash setup.sh                                  uses python3.10 from PATH
#   PYTHON=/path/to/python3.10 bash setup.sh       uses the given interpreter
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3.10}"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 is required"'

# 1. motion and observation services
[ -d "$HERE/.venv" ] || "$PY" -m venv "$HERE/.venv"
"$HERE/.venv/bin/pip" install -r "$HERE/requirements-harness.txt"

# 2. simulator
[ -d "$HERE/.venv_sim" ] || "$PY" -m venv "$HERE/.venv_sim"
"$HERE/.venv_sim/bin/pip" install -r "$HERE/requirements-sim.txt"

# 3. folders of the agent's working directory that are filled at run time
mkdir -p "$HERE/stack/docs" "$HERE/stack/tmp"

# 4. agent settings, which need the absolute path of this clone
"$HERE/.venv/bin/python" - "$HERE" <<'PY'
import json
import sys
from pathlib import Path

here = Path(sys.argv[1])


def hook(script, matcher, **extra):
    return {"hooks": {"PreToolUse": [{"matcher": matcher, "hooks": [
        dict({"type": "command", "command": f'python3 "{here / script}"'}, **extra)]}]}}


# Claude Code: settings file passed with --settings
(here / "agent_settings.json").write_text(json.dumps(dict({"autoMemoryEnabled": False}, **hook("agent_guard.py", "*")), indent=2) + "\n")
# Codex: an isolated CODEX_HOME that holds only the model settings and the guard hook
home = here / ".codex_home"
home.mkdir(exist_ok=True)
(home / "hooks.json").write_text(json.dumps(hook("codex_guard.py", ".*", timeout=30), indent=2) + "\n")
if not (home / "config.toml").exists():
    stack = str(here / "stack").replace("\\", "\\\\").replace('"', '\\"')
    (home / "config.toml").write_text('model = "gpt-6-astra"\nmodel_reasoning_effort = "high"\nservice_tier = "default"\n\n'
                                      f'[projects."{stack}"]\ntrust_level = "trusted"\n')
print("agent settings written:", here / "agent_settings.json", "and", home)
PY
echo "Setup complete."
