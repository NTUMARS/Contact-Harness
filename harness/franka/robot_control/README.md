# Robot server interface and settings

`franka_robot_server.py` starts the Franka driver and the Cartesian impedance controller through `impedance.launch` in this folder and serves a local interface. `motion_server.py` and `observation_server.py` are its only clients in the harness. This page describes the interface for anyone who wants to connect another client or another robot.

Address: `http://127.0.0.1:5003`.

## Motion interface

The client decomposes a motion into small increments and sends one or more rows to `POST /actions`.

```python
import requests

reply = requests.post(
    'http://127.0.0.1:5003/actions',
    json={'actions': [
        [0.0002, 0, 0, 0, 0, 0],
        [0.0002, 0, 0, 0, 0, 0.001],
    ]},
    timeout=2,
)
reply.raise_for_status()  # 202 = entire batch enqueued, not motion completed.
print(reply.json())  # {"accepted": 2, "queue_depth": ...}
```

Each row is `[dx, dy, dz, rx, ry, rz]`. Translation is in meters. Rotation is a rotation vector (axis times angle, radians), not Euler angles. Both are in the robot base frame. Targets accumulate as `p_next = p_previous_target + delta_xyz` and `R_next = Exp(delta_rotvec) * R_previous_target`. The measured pose seeds the target once at startup and never resets it afterwards.

The server consumes at most one row per tick at a nominal 50 Hz (20 ms). Late ticks delay execution and do not trigger a burst of catch-up commands. Every accepted row, including an all-zero row, occupies one tick. An empty queue holds the latest equilibrium target under impedance. This is the timing of target publication, not a guarantee that the robot reaches each target in 20 ms.

Client contract

- Each translation vector norm must be at most 0.0005 m and each rotation vector norm at most 0.005 rad. The server does not split, rescale or interpolate rows.
- The FIFO holds up to 500 pending rows, about 10 s at 50 Hz. A batch is accepted as a whole or rejected. HTTP 400 means invalid input, 429 a full queue, 503 unavailable or faulted feedback, 413 an oversized body, 409 an active homing.
- Requests from one client must be serialized. Batches execute in the order the server accepts them. There is no request deduplication, so a blind retry after a timeout can enqueue the same relative motion twice.
- `GET /info` reports `queue_depth`, `queue_capacity`, `accepted_actions`, `applied_actions` and `cleared_actions`. Applied means consumed into the target by the control loop, not measured arrival. Counts reset on restart.
- `GET /getstate` returns the current feedback: `pose` (position and quaternion x, y, z, w in the base frame), `q`, `dq`, `robot_time_s`, `robot_mode`, `current_errors` and `wrench_K` (estimated external wrench in the stiffness frame).
- The server stops on stale or faulted feedback and on a new target more than 0.03 m or 0.2 rad away from the measured pose. Pending commands are lost when the server exits.
- `POST /actions/clear` discards pending rows and holds the latest target. It is not an emergency stop. A step that was already consumed may still be tracking.

## Homing and gripper

Both operations are asynchronous. HTTP 202 means accepted, not completed. Poll the status instead of resending on a timeout.

| Request | Effect |
|---|---|
| `POST /home {}` | Clears queued increments, requires a stationary arm, switches to joint position control, moves to `HOME_JOINT_TARGET`, restores Cartesian impedance and seeds its target from the reached pose. The move is a direct joint-space move, not a collision-free plan. Deadline 90 s. A failure stops the server |
| `GET /home` | Status, `state` is `idle`, `pending`, `moving`, `restoring`, `succeeded` or `failed` |
| `POST /gripper {"width_m": 0.0, "force_n": 40.0}` | Native force grasp. `width_m` is the total finger opening, 0 to 0.08 m. `force_n` is 30 to 70 N |
| `POST /gripper {"width_m": 0.08, "force_n": 0}` | Native move, used to open or release |
| `GET /gripper` | Status and the commanded force, not a measured force |

Optional gripper fields are `speed_m_s` (default 0.05, range 0.005 to 0.2), `mode` (`move` or `grasp`), and for grasps `epsilon_inner_m` (default 0.005) and `epsilon_outer_m` (default 0.08). After a successful grasp a second grasp returns 409 until an explicit move has succeeded, because the native interface cannot update the holding force in place. A grasp whose result is lost or timed out is reported as `outcome_unknown`, and the server sends no stop or open, since either would release the object.

New increments, homing and gripper requests return 409 while homing is active.

## Startup settings

Edit the settings near the top of `../start_robot_server.sh` or override them through environment variables.

```bash
EEF_IMPEDANCE='1500,89,150,7' TRANSLATIONAL_CLIP=0.01 \
  COLLISION_FORCE_THRESHOLD=20 COLLISION_TORQUE_SCALE=1.0 \
  bash start_robot_server.sh --check-only
```

`--check-only` prints the selected settings and the launch command without ROS or robot. Without it the server starts with those settings. Changes take effect on the next start.

| Setting | Default in the script | Meaning |
|---|---|---|
| `EEF_IMPEDANCE` | `2000,89,150,7` | Translation stiffness in N/m (0 to 6000), translation damping in Ns/m (0 to 400), rotation stiffness in Nm/rad (0 to 300), rotation damping in Nms/rad (0 to 30). One value for the three translation axes and one for the three rotation axes |
| `TRANSLATIONAL_CLIP` | `0.03` | Position error clip in m, greater than 0 and at most 0.1, applied to both directions of X, Y and Z. At low speed the spring force per axis saturates at stiffness times clip |
| `COLLISION_FORCE_THRESHOLD` | `50` | Upper Cartesian X, Y, Z collision threshold in N for the nominal and acceleration phases. The lower contact thresholds are the smaller of 20 N and this value. Rotational thresholds stay at 25 Nm |
| `COLLISION_TORQUE_SCALE` | `2.0` | Multiplier, at least 1, of the joint collision torque thresholds `[20, 20, 18, 18, 16, 14, 12]` Nm. The lower contact thresholds keep their values |
| `HOME_JOINT_TARGET` | `0,0,0,-2.70158294,0,2.70158294,0.7853981633974483` | Home joint angles in rad |
| `GRIPPER_GRASP_FORCE` | `50` | Grasp force in N used when a grasp request has no `force_n` |

`motion_server.py` refuses to start when its hard force limit, 50 N by default, exceeds `COLLISION_FORCE_THRESHOLD`.

The impedance profile is applied through dynamic reconfigure at startup and the response is checked. The remaining impedance values, the rotational error clip and the nullspace gains, are in `IMPEDANCE_PROFILE` of the server. The collision thresholds are passed to `franka_control_node.yaml` through `impedance.launch` before the driver starts, and the server verifies the ROS parameters before it reports ready. `/info` reports the selected impedance and collision settings. The collision values describe the startup configuration and are not read back from the hardware.

Neither setting disables collision protection, and neither implements a force controller.

## Files

| File | Content |
|---|---|
| `impedance.launch` | Copy of `franka_control.launch` of `franka_ros` 0.10.2 with launch arguments for the collision thresholds, plus the spawner of the Cartesian impedance controller |
| `franka_control_node.yaml` | Copy of the driver configuration of `franka_ros` 0.10.2 with the collision thresholds taken from the launch arguments |

The server finds `impedance.launch` by its path relative to the server file, so this folder must stay next to it.
