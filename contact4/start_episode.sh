#!/usr/bin/env bash
# Start one simulated episode: robosuite-family backend + unmodified motion/observation servers.
# Usage: bash start_episode.sh RUN_NAME -- [sim_task_server.py args, e.g. --benchmark wipe --seed 0]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="$HERE/runs/$1"; shift; [ "${1:-}" = "--" ] && shift
mkdir -p "$RUN"
# Never share ports with a real robot stack (8080/8081, ipc 5555): the sim uses 18080/18081 + tcp 5655/5656.
for port in 18080 18081 5655 5656; do
  if ss -ltn 2>/dev/null | grep -q "127.0.0.1:$port "; then echo "port $port busy: refusing to start"; exit 1; fi
done
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
ADDR="tcp://127.0.0.1:5655"
SIM_PY="$HERE/sim_task_server.py"
"$HERE/.venv_sim/bin/python" "$SIM_PY" --address "$ADDR" \
    --info-out "$RUN/task.json" "$@" > "$RUN/sim_robot.log" 2>&1 &
echo $! > "$RUN/sim_robot.pid"
for i in $(seq 1 120); do [ -f "$RUN/task.json" ] && break; sleep 1; done
[ -f "$RUN/task.json" ] || { echo "sim failed to start; see $RUN/sim_robot.log"; exit 1; }
sleep 1
"$HERE/.venv/bin/python" "$HERE/run_motion_server.py" --robot-address "$ADDR" > "$RUN/motion_server.log" 2>&1 &
echo $! > "$RUN/motion_server.pid"
for i in $(seq 1 60); do grep -q "ready for commands" "$RUN/motion_server.log" && break; sleep 1; done
grep -q "ready for commands" "$RUN/motion_server.log" || { echo "motion_server not ready; see $RUN/motion_server.log"; exit 1; }
SIM_ROBOT_ADDRESS="$ADDR" "$HERE/.venv/bin/python" "$HERE/run_observation_server.py" > "$RUN/observation_server.log" 2>&1 &
echo $! > "$RUN/observation_server.pid"
for i in $(seq 1 60); do
  "$HERE/.venv/bin/python" "$HERE/stack/get_observation.py" --status 2>/dev/null | grep -q '"ready": true' && break; sleep 1; done
# Final check: the motion server the sim CLIs reach must be backed by the simulator.
grep -q "robot endpoint tcp://127.0.0.1:5655" "$RUN/motion_server.log" || { echo "motion server is not the sim"; exit 1; }
echo "$RUN"
