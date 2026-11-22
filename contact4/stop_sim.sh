#!/usr/bin/env bash
# Stop processes started by start_sim.sh for RUN_NAME (default: newest run).
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="$HERE/runs/${1:-$(cd "$HERE/runs" && ls -td contact_harness/*/*/ 2>/dev/null | head -1)}"
for f in observation_server motion_server sim_robot; do
  [ -f "$RUN/$f.pid" ] && kill "$(cat "$RUN/$f.pid")" 2>/dev/null && echo "stopped $f"
done
