#!/bin/bash
# One episode of Contact Harness with Codex (default GPT-6 Astra, reasoning effort high), one seed:
# 1) episode_ctl start  2) a NEW agent session in stack/ with the opening prompt  3) when the agent ends:
# episode_ctl stop (or the 30-min limit stops it).
# Agent isolation:
#   * CODEX_HOME = contact4/.codex_home (written by setup.sh; log in there once): model settings only, no skills,
#     memories, rules or history; --ephemeral, nothing written to ~/.codex
#   * PreToolUse hook codex_guard.py (agent_guard.py's policy: only stack/; no simulator internals, logs, /proc,
#     other directories, previous conversations); every request logged to <run>/guard.jsonl
#   * tools: shell, view_image, update_plan (web search, apps, plugins, browser, computer use, image generation and
#     sub-agents disabled; apply_patch refused by the guard); private TMPDIR stack/tmp; no simulator token in its env
# Runs go to runs/contact_harness/<task>/ (RUN_GROUP=<name> selects another folder).
# Usage: run_codex_episode.sh <task> <seed>
# Ablations: ABLATION=<name> (the choices of episode_ctl.py start --ablation) runs the same flow with that ablation;
# runs go to runs/ablation/<ablation>/<task>/. The variable is removed from the agent's environment.
set -u
TASK="$1"; SEED="${2:-0}"
SIM="$(cd "$(dirname "$0")" && pwd)"
CH="${CODEX_HOME_DIR:-$SIM/.codex_home}"
MODEL="${AGENT_MODEL:-gpt-6-astra}"; EFFORT="${AGENT_EFFORT:-high}"
ABLATION="${ABLATION:-none}"
TAG=""; [ "$ABLATION" != none ] && TAG="${ABLATION}_"
mkdir -p "$SIM/runs/_logs"
LOG="$SIM/runs/_logs/${TAG}${TASK}_s${SEED}_$(date +%Y%m%d_%H%M%S).log"
export SIM_RUN_GROUP="${RUN_GROUP:-contact_harness}"   # RUN_GROUP: a separate folder for a set of runs
[ "$ABLATION" != none ] && export SIM_RUN_GROUP="ablation/$ABLATION"
case "$TASK" in
  wipe_table) PROMPT="Do not read any previous conversations or any skills stored in the system; read only the prompts under this project path. Read only the instructions in agent_prompt and task_prompt_wipe_table in docs, pay attention to the important notes, and wipe all the dirt marks off the table with the wiping pad mounted on the hand." ;;
  write_board) PROMPT="Do not read any previous conversations or any skills stored in the system; read only the prompts under this project path. Read only the instructions in agent_prompt and task_prompt_write_board in docs, pay attention to the important notes, and write the letters ROBOT in uppercase block letters about 5 cm tall on the whiteboard." ;;
  peg_insertion) PROMPT="Do not read any previous conversations or any skills stored in the system; read only the prompts under this project path. Read only the instructions in agent_prompt and task_prompt_peg_insertion in docs, pay attention to the important notes, and insert the peg held in the gripper into the hole of the metal block below it." ;;
  pivot_box) PROMPT="Do not read any previous conversations or any skills stored in the system; read only the prompts under this project path. Read only the instructions in agent_prompt and task_prompt_pivot_box in docs, pay attention to the important notes, and stand the flat box up on its narrow side against the wall by pushing it with the closed fingers." ;;
  stack_cubes) PROMPT="Do not read any previous conversations or any skills stored in the system; read only the prompts under this project path. Read only the instructions in agent_prompt and task_prompt_stack_cubes in docs, pay attention to the important notes, and pick up the red cube and place it on top of the green cube." ;;
  place_bowl) PROMPT="Do not read any previous conversations or any skills stored in the system; read only the prompts under this project path. Read only the instructions in agent_prompt and task_prompt_place_bowl in docs, pay attention to the important notes, and pick up the bowl and put it on the plate." ;;
  *) echo "unknown task"; exit 1 ;;
esac
# the isolated Codex home must hold only what we put there: no skills, memories files or rules. Codex itself installs
# its built-in system skills under skills/.system at startup (imagegen, review-agent, skill-creator, openai-docs,
# skill-installer; the real-robot ~/.codex has the same set), so only that folder is tolerated.
for d in skills memories rules sessions; do
  if [ -n "$(ls -A "$CH/$d" 2>/dev/null | grep -vx '.system')" ]; then echo "[$(date +%T)] CODEX HOME NOT CLEAN ($CH/$d); HALT" | tee -a "$LOG"; exit 2; fi
done
cd "$SIM"
echo "[$(date +%T)] START $TASK seed $SEED (Codex $MODEL, effort $EFFORT, ablation $ABLATION)" | tee -a "$LOG"
T0=$(date +%s)
.venv/bin/python episode_ctl.py start --task "$TASK" --seed "$SEED" --ablation "$ABLATION" >> "$LOG" 2>&1 || { echo "[$(date +%T)] START FAILED" | tee -a "$LOG"; exit 1; }
RUN=$(python3 -c "import json;print(json.load(open('runs/_current.json'))['run'])")
echo "[$(date +%T)] episode ready: $RUN" | tee -a "$LOG"
cd "$SIM/stack"
# own process group (setsid) so the agent and the commands it started can be ended together
env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT -u CLAUDE_CODE_SSE_PORT -u SIM_RPC_TOKEN -u SIM_RUN_GROUP -u ABLATION -u TAG -u RUN_GROUP \
  CODEX_HOME="$CH" CODEX_GUARD_LOG="$RUN/guard.jsonl" TMPDIR="$SIM/stack/tmp" \
  setsid codex exec --skip-git-repo-check --ephemeral --json -C "$SIM/stack" \
    -s workspace-write -c sandbox_workspace_write.network_access=true -c shell_environment_policy.inherit=all \
    -m "$MODEL" -c "model_reasoning_effort=\"$EFFORT\"" -c 'web_search="disabled"' \
    --disable multi_agent --disable apps --disable plugins --disable browser_use --disable computer_use \
    --disable image_generation --disable in_app_browser --dangerously-bypass-hook-trust \
    -o "$RUN/agent_last.txt" "$PROMPT" < /dev/null > "$RUN/agent_stream.jsonl" 2>&1 &
APID=$!
echo "[$(date +%T)] agent pid $APID" | tee -a "$LOG"
endagent() { kill -TERM -- -"$APID" 2>/dev/null || kill "$APID" 2>/dev/null; sleep 5; kill -KILL -- -"$APID" 2>/dev/null; }
cd "$SIM"
LIMIT_S=$(( ${TIME_LIMIT_MIN:-30} * 60 + 180 ))   # backstop if the time-limit stop itself fails
while kill -0 $APID 2>/dev/null; do
  [ -f runs/_current.json ] || { echo "[$(date +%T)] time limit reached: episode stopped, ending the agent" | tee -a "$LOG"; endagent; break; }
  [ $(( $(date +%s) - T0 )) -ge $LIMIT_S ] && { echo "[$(date +%T)] backstop: time limit + 3 min, ending the agent" | tee -a "$LOG"; endagent; break; }
  sleep 10
done
wait $APID 2>/dev/null
kill -KILL -- -"$APID" 2>/dev/null   # nothing the agent started may outlive it
echo "[$(date +%T)] agent ended" | tee -a "$LOG"
[ -f runs/_current.json ] && .venv/bin/python episode_ctl.py stop >> "$LOG" 2>&1
for i in $(seq 1 120); do [ ! -f runs/_current.json ] && [ ! -f runs/_stopping.json ] && break; sleep 5; done
[ -f "$RUN/stop.log" ] && cat "$RUN/stop.log" >> "$LOG"
if [ ! -f "$RUN/result.json" ]; then
  if [ -f "$SIM/runs/_aborted/$(basename "$RUN")/ABORTED.txt" ]; then
    echo "[$(date +%T)] PLATFORM FAILURE (API/quota error ended the agent; not counted): $(cat "$SIM/runs/_aborted/$(basename "$RUN")/ABORTED.txt"); HALT" | tee -a "$LOG"; exit 4; fi
  echo "[$(date +%T)] NO RESULT for $RUN; HALT" | tee -a "$LOG"; exit 5
fi
python3 - "$RUN" "$MODEL" "$EFFORT" <<'EOF' 2>/dev/null
import json, sys
run, model, effort = sys.argv[1:4]
p = run + "/result.json"; j = json.load(open(p))
usage = None
for l in open(run + "/agent_stream.jsonl", errors="ignore"):
    try:
        e = json.loads(l)
    except ValueError:
        continue
    if e.get("type") == "turn.completed":
        usage = e.get("usage")
g = [json.loads(l) for l in open(run + "/guard.jsonl")] if __import__("os").path.exists(run + "/guard.jsonl") else []
j.update(mode="Codex exec session, driven by run_codex_episode.sh", agent="codex", agent_model=model,
         agent_effort=effort, agent_settings="isolated CODEX_HOME + PreToolUse codex_guard.py",
         agent_usage=usage, guard_requests=len(g), guard_blocked=sum(1 for x in g if not x.get("ok")))
json.dump(j, open(p, "w"), indent=2)
EOF
# every run keeps its video: episode_ctl stop renders <run>/episode.mp4; render again once if missing
if [ ! -s "$RUN/episode.mp4" ] && [ ! -e "$SIM/NO_RENDER" ]; then   # no video while NO_RENDER exists
  echo "[$(date +%T)] episode.mp4 missing, rendering again" | tee -a "$LOG"
  (cd "$SIM" && .venv/bin/python -c "import sys; from pathlib import Path; from run_episode import render_video; render_video(Path(sys.argv[1]))" "$RUN")
  [ -s "$RUN/episode.mp4" ] || echo "[$(date +%T)] WARNING: no video for $RUN (see $RUN/render.log)" | tee -a "$LOG"
fi
echo "[$(date +%T)] DONE $TASK success=$(python3 -c "import json;print(json.load(open('$RUN/result.json'))['success'])") run=$RUN" | tee -a "$LOG"
