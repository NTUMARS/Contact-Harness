#!/usr/bin/env bash
# The agent's Claude Code window for a manual-session episode (start one first: python episode_ctl.py start ...).
# Interactive Claude Code, isolated from everything stored on this machine:
#   * CLAUDE_CONFIG_DIR=contact4/.agent_cc: its own config/sessions/login (no history, no stored skills,
#     no memory, no user settings); its conversations are saved there (resume/rename them there)
#   * working directory = the harness copy contact4/stack (docs/ holds only this task's prompts)
#   * model Claude Fable 5.1; no MCP servers; auto memory off
#   * tools: Bash, Read, Glob, Grep (no file edits, no web, no subagents) -- same as the headless runs
# First use: log in once inside this window (/login, Claude subscription).
# Resume a past conversation (e.g. to /rename it):  bash agent_cc.sh --resume <session id>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export CLAUDE_CONFIG_DIR="$HERE/.agent_cc"
export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$HERE/.venv/bin:$PATH"
for v in $(env | grep -oE '^(CLAUDECODE|CLAUDE_CODE_[A-Z_]+)=' | tr -d '=' | grep -v '^CLAUDE_CODE_DISABLE_AUTO_MEMORY$'); do
  unset "$v"
done
mkdir -p "$CLAUDE_CONFIG_DIR"
cd "$HERE/stack"
exec claude --model claude-fable-5-1 \
  --strict-mcp-config --setting-sources project --settings '{"autoMemoryEnabled": false}' \
  --permission-mode dontAsk \
  --allowedTools Bash Read Glob Grep \
  --disallowedTools Edit Write NotebookEdit WebFetch WebSearch Agent \
  "$@"
