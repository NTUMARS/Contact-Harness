#!/usr/bin/env bash
# Starts the robot server. Runs in the foreground; Ctrl+C stops it.
set -eo pipefail

# Environment: edit the defaults here or override via environment.
CATKIN_SETUP="${CATKIN_SETUP:-$HOME/catkin_ws/devel/setup.bash}" # Workspace with franka_ros and franka_controllers.
ROBOT_PYTHON="${ROBOT_PYTHON:-python3}" # Python with requirements.txt installed.
ROBOT_IP="${ROBOT_IP:-FRANKA_ROBOT_IP}" # Address of the Franka controller.

# Adjustable settings: edit defaults here or override via environment; restart after edits.
HOME_JOINT_TARGET="${HOME_JOINT_TARGET:-0,0,0,-2.70158294,0,2.70158294,0.7853981633974483}" # Home J1-J7 (rad).
GRIPPER_GRASP_FORCE="${GRIPPER_GRASP_FORCE:-50}" # Default native grasp force (N, 30-70); API force_n overrides.
EEF_IMPEDANCE="${EEF_IMPEDANCE:-2000,89,150,7}" # Kt (N/m), Dt (Ns/m), Kr (Nm/rad), Dr (Nms/rad).
TRANSLATIONAL_CLIP="${TRANSLATIONAL_CLIP:-0.03}" # XYZ spring error clip (m, >0 to 0.1).
COLLISION_FORCE_THRESHOLD="${COLLISION_FORCE_THRESHOLD:-50}" # XYZ collision stop force (N, >0).
COLLISION_TORQUE_SCALE="${COLLISION_TORQUE_SCALE:-2.0}" # Joint collision stop multiplier (>=1).
# End of adjustable settings.

if [[ "$#" -gt 1 || ( "${1:---start}" != --start && "${1:---start}" != --check-only ) ]]; then
  echo 'Usage: bash start_robot_server.sh [--start|--check-only]' >&2
  exit 2
fi
server_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
robot_python="$(command -v -- "$ROBOT_PYTHON")" || { echo "Python not found: $ROBOT_PYTHON" >&2; exit 2; }
server_args=(
  "--robot-ip=$ROBOT_IP"
  "--home-joint-target=$HOME_JOINT_TARGET"
  "--gripper-grasp-force=$GRIPPER_GRASP_FORCE"
  "--eef-impedance=$EEF_IMPEDANCE"
  "--translational-clip=$TRANSLATIONAL_CLIP"
  "--collision-force-threshold=$COLLISION_FORCE_THRESHOLD"
  "--collision-torque-scale=$COLLISION_TORQUE_SCALE"
)
if [[ "${1:---start}" == --check-only ]]; then
  # Prints the selected settings and the launch command. Needs no ROS or robot.
  exec "$robot_python" -u "$server_dir/franka_robot_server.py" "${server_args[@]}" --check-only
fi
source "$CATKIN_SETUP"
export ROS_MASTER_URI=http://127.0.0.1:11311
# ROS tools such as xacro use /usr/bin/env python3; use the same environment.
export PATH="$(dirname -- "$robot_python"):$PATH"
# Owns only the new roslaunch process group. No pkill/wipe, no automatic recovery.
echo "Starting the robot server in the foreground. Keep this terminal open; press Ctrl+C to stop."
exec "$robot_python" -u "$server_dir/franka_robot_server.py" "${server_args[@]}"
