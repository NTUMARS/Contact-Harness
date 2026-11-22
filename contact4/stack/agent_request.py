#!/usr/bin/env python3
"""Send observation-calibration or soft-force requests to the running servers.

  python agent_request.py calibrate-observation --target robot_wrench
  python agent_request.py calibrate-observation --target tactile_wrench
  python agent_request.py set-soft-force --newtons 10

Wait for motion to finish before either request. Calibration averages one second:
robot_wrench zeros robot force/torque without contact; tactile_wrench (alias
tactile) zeros both pads' wrench AND deformation for a still grasp without
external contact. It does not reset the SDK image reference.

set-soft-force changes translation clipping, not a commanded force or guaranteed
force cap. The server must be ready with no active/queued actions. The value must
be positive, below the configured hard threshold, and above the force-bias norm.
Both settings live in RAM; restarting restores the startup force threshold
(default 10 N). Neither request commands motion. Failed validation retains the
previous setting. If a force request times out, check terminal_command.py status
for soft_force_n before retrying.
"""
import argparse
import json

from get_observation import DEFAULT_MOTION_SERVER, DEFAULT_SERVER, ObservationClient, positive_number
from terminal_command import MotionClient


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="request", required=True)
    calibrate = commands.add_parser("calibrate-observation", help="Average a one-second observation baseline")
    calibrate.add_argument("--target", required=True, choices=("robot_wrench", "tactile_wrench", "tactile"))
    calibrate.add_argument("--server", default=DEFAULT_SERVER,
                           help="Observation-server URL (default: %(default)s)")
    force = commands.add_parser("set-soft-force", help="Update the idle motion server's soft force threshold")
    force.add_argument("--newtons", type=positive_number, required=True)
    for command in (calibrate, force):
        command.add_argument("--motion-server", default=DEFAULT_MOTION_SERVER,
                             help="Motion-server URL (default: %(default)s)")
        command.add_argument("--timeout", type=positive_number, default=6.,
                             help="HTTP request timeout in seconds (default: %(default)s)")
    args = parser.parse_args(argv)
    try:
        if args.request == "calibrate-observation":
            result = ObservationClient(args.server, args.timeout, motion_server=args.motion_server).calibrate(args.target)
        else:
            result = MotionClient(args.motion_server, args.timeout).set_soft_force(args.newtons)
        print(json.dumps(result, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
