#!/usr/bin/env python3
"""Send actions to motion_server.py (localhost:18080). Start robot and motion servers first.

One action:
  python terminal_command.py delta --translation-m 0.01 0 0 --duration 2 --wait

Sequence:
  python terminal_command.py delta --translation-m 0.1 0 0 --duration 4 --next delta --translation-m 0 0.05 0 --duration 2 --wait

--next separates actions with their own parameters. --wait waits for the whole
sequence; --wait-timeout SECONDS changes the default 300 s timeout.

Other commands: pose, gripper, home, status [JOB_ID], stop, recover.
status/stop/recover run separately. home cancels prior work and must be first.
Ctrl+C or timeout leaves work running: check status before retrying, or use stop.
stop cannot interrupt home. Verify actual motion in observations after --wait.

Use --help or COMMAND --help for options; see docs/agent_prompt.txt for units,
coordinate frames and the observation workflow.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import sys
import time
from urllib.parse import quote, urlsplit


class SubmissionError(RuntimeError):
    """The request failed or its result could not be determined."""


class MotionClient:
    """Small HTTP client, also importable by a future submission interface."""

    def __init__(self, server="http://127.0.0.1:18080", http_timeout=5.0):
        url = urlsplit(server)
        if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost") or
                url.username is not None or url.password is not None or
                url.path not in ("", "/") or url.query or url.fragment):
            raise ValueError("server must be http://127.0.0.1:PORT or http://localhost:PORT")
        self.host = url.hostname
        self.port = 80 if url.port is None else url.port
        if not 1 <= self.port <= 65535:
            raise ValueError("server port must be between 1 and 65535")
        self.http_timeout = positive_number(http_timeout)

    def _request(self, method, path, payload=None):
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.http_timeout)
        try:
            connection.request(method, path, body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read()
        except (OSError, http.client.HTTPException) as exc:
            note = " Request outcome is unknown; do not automatically resend." if method == "POST" else ""
            raise SubmissionError(f"HTTP request to {self.host}:{self.port} failed: {exc}.{note}") from exc
        finally:
            connection.close()
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise SubmissionError(f"HTTP {response.status}: server returned invalid JSON; request outcome is unknown") from exc
        if not isinstance(result, dict):
            raise SubmissionError("Server returned an unexpected response; request outcome is unknown")
        if not 200 <= response.status < 300:
            detail = result.get("detail") or result.get("error") or response.reason
            raise SubmissionError(f"HTTP {response.status}: {detail}")
        return result

    def submit(self, actions):
        """Submit one sequence once; a single dict is normalized at this boundary."""
        if isinstance(actions, dict):
            actions = [actions]
        return self._request("POST", "/commands", {"actions": actions})

    def status(self, reference=None):
        path = "/status" if reference is None else "/commands/" + quote(reference, safe="")
        return self._request("GET", path)

    def set_soft_force(self, newtons):
        return self._request("POST", "/settings/soft-force", {"soft_force_n": newtons})

    def stop(self):
        return self._request("POST", "/stop", {})

    def recover(self):
        return self._request("POST", "/recover", {})

    def wait_for_command(self, reference, timeout=300.0):
        deadline = time.monotonic() + positive_number(timeout)
        while True:
            result = self.status(reference)
            if result.get("status") in ("completed", "failed", "cancelled"):
                return result
            if time.monotonic() >= deadline:
                raise SubmissionError(f"Wait timed out for {reference}; the command may still be running. "
                                      "Check its status or explicitly request stop.")
            time.sleep(0.1)

    def wait_for_stop(self, timeout=300.0):
        deadline = time.monotonic() + positive_number(timeout)
        while True:
            result = self.status()
            if result.get("stop_requested") is False:
                return result
            if time.monotonic() >= deadline:
                raise SubmissionError("Wait timed out; native stop is not confirmed. Check server status.")
            time.sleep(0.1)


def finite_number(value):
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def positive_number(value):
    result = finite_number(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def connection_options(parser):
    parser.add_argument("--server", default="http://127.0.0.1:18080", help="Motion-server URL (default: %(default)s)")
    parser.add_argument("--http-timeout", type=positive_number, default=5.0, metavar="SECONDS",
                        help="Timeout per HTTP request (default: %(default)s)")


def wait_options(parser):
    parser.add_argument("--wait", action="store_true", help="Wait for the whole sequence (or stop/recovery)")
    parser.add_argument("--wait-timeout", type=positive_number, metavar="SECONDS",
                        help="Polling deadline with --wait (default: 300); does not cancel motion")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    connection_options(parser)
    sub = parser.add_subparsers(dest="action", required=True)
    for action, help_text in (
        ("delta", "Relative translation and/or rotation over a duration"),
        ("pose", "Absolute TCP pose in world coordinates (or explicit --frame base)"),
        ("gripper", "Set opening: 0 closed, 1 open"),
        ("home", "Reset to YAML home using the server's SDK MoveJ"),
        ("stop", "Cancel active/queued work; blocking SDK homing must finish first"),
        ("recover", "Confirm native stop and clear a resolved service fault"),
        ("status", "Read server status, or a result using its server-generated reference"),
    ):
        command = sub.add_parser(action, help=help_text, description=help_text, allow_abbrev=False)
        if action == "status":
            command.add_argument("reference", nargs="?", metavar="REFERENCE",
                                 help="Optional reference returned by the server; omit for server status")
            continue
        wait_options(command)
        if action in ("stop", "recover"):
            continue
        if action == "home":
            continue
        if action == "gripper":
            command.add_argument("--position", type=finite_number, required=True, metavar="OPENING",
                                 help="Normalized opening from 0 (closed) to 1 (open)")
            continue
        command.add_argument("--duration", type=positive_number, required=True, metavar="SECONDS",
                             help="Timed execution interval in seconds; no target-arrival requirement")
        if action == "delta":
            command.add_argument("--translation-m", nargs=3, type=finite_number, metavar=("X", "Y", "Z"))
            rotation = command.add_mutually_exclusive_group()
            rotation.add_argument("--rotation-rad", nargs=3, type=finite_number, metavar=("RX", "RY", "RZ"),
                                  help="Rotation vector in radians (axis times angle)")
            rotation.add_argument("--rotation-deg", nargs=3, type=finite_number, metavar=("RX", "RY", "RZ"),
                                  help="Same rotation vector expressed in degrees")
            command.add_argument("--frame", choices=("world", "base", "tool"), default="world",
                                 help="Frame fixed at command start (default: %(default)s)")
        elif action == "pose":
            command.add_argument("--frame", choices=("world", "base"), default="world",
                                 help="Absolute pose coordinate frame (default: %(default)s)")
            command.add_argument("--position-m", nargs=3, type=finite_number, required=True, metavar=("X", "Y", "Z"))
            command.add_argument("--quaternion-xyzw", nargs=4, type=finite_number, required=True,
                                 metavar=("QX", "QY", "QZ", "QW"), help="Unit quaternion in x,y,z,w order")
    return parser


def command_from_args(args):
    command = {"type": args.action}
    if args.action == "home":
        return command
    if args.action == "gripper":
        if not 0 <= args.position <= 1:
            raise ValueError("--position must be between 0 (closed) and 1 (open)")
        command["position"] = args.position
        return command
    command["duration_s"] = args.duration
    if args.action == "delta":
        if args.translation_m is None and args.rotation_rad is None and args.rotation_deg is None:
            raise ValueError("delta requires --translation-m, --rotation-rad or --rotation-deg")
        rotation = args.rotation_rad or [0.0, 0.0, 0.0]
        if args.rotation_deg is not None:
            rotation = [math.radians(value) for value in args.rotation_deg]
        command.update(translation_m=args.translation_m or [0.0, 0.0, 0.0],
                       rotation_vector_rad=rotation, frame=args.frame)
    elif args.action == "pose":
        if abs(math.hypot(*args.quaternion_xyzw) - 1.0) > 1e-3:
            raise ValueError("--quaternion-xyzw must have unit length")
        command.update(position_m=args.position_m, quaternion_xyzw=args.quaternion_xyzw, frame=args.frame)
    return command


def sequence_from_args(argv, parser):
    """Parse shared request options once and each --next-separated action alike."""
    shared = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    connection_options(shared)
    wait_options(shared)
    options, remaining = shared.parse_known_args(argv)
    groups = [[]]
    for token in remaining:
        if token == "--next":
            if not groups[-1]:
                raise ValueError("--next must separate complete actions")
            groups.append([])
        else:
            groups[-1].append(token)
    if len(groups) > 1 and not groups[-1]:
        raise ValueError("--next must be followed by an action")
    actions = [parser.parse_args(group) for group in groups]
    if any(action.action in ("status", "stop", "recover") for action in actions) and len(actions) != 1:
        raise ValueError("status, stop and recover are separate operations; do not combine them with --next")
    if options.wait_timeout is not None and not options.wait:
        raise ValueError("--wait-timeout requires --wait")
    if options.wait and actions[0].action == "status":
        raise ValueError("status does not accept --wait")
    options.wait_timeout = 300.0 if options.wait_timeout is None else options.wait_timeout
    return options, actions


def main(argv=None):
    parser = build_parser()
    try:
        args, actions = sequence_from_args(sys.argv[1:] if argv is None else argv, parser)
        operation = actions[0]
        client = MotionClient(args.server, args.http_timeout)
        commands = ([command_from_args(action) for action in actions]
                    if operation.action in ("delta", "pose", "gripper", "home") else None)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    reference = None
    try:
        if commands is not None:
            result = client.submit(commands)
            reference = result.get("id")
            if args.wait and result.get("status") not in ("completed", "failed", "cancelled"):
                if not isinstance(reference, str) or not reference:
                    raise SubmissionError("Server acknowledgement lacks a result reference; cannot wait")
                result = client.wait_for_command(reference, args.wait_timeout)
        elif operation.action == "status":
            result = client.status(operation.reference)
        else:
            result = client.stop() if operation.action == "stop" else client.recover()
            if args.wait:
                result = client.wait_for_stop(args.wait_timeout)
        print(json.dumps(result, indent=2, allow_nan=False))
        return int(result.get("status") in ("failed", "cancelled") or bool(result.get("fault")))
    except SubmissionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if commands is not None:
            lookup = f" {reference}" if reference else ""
            print(f"Inspect status with: python terminal_command.py status{lookup} "
                  "(using the same --server).", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nClient interrupted. Submitted commands are not cancelled; "
              "use terminal_command.py stop to request a stop.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
