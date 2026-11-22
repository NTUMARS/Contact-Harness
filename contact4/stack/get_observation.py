#!/usr/bin/env python3
r"""Fetch observations from the running observation server (localhost:18081).

Usage in agentic_manip:
  python get_observation.py --observations force eef_pose joint_angles left_wrist.rgb head.rgb
  python get_observation.py --observations all
  python get_observation.py --observations force --wrench tcp
  python get_observation.py --observations tactile_left.wrench tactile_right.wrench tactile_tip.wrench tactile_tip_world.wrench --tactile-reference connector
  python get_observation.py --observations tactile_left.deformation tactile_right.deformation
  python get_observation.py --status

All devices keep streaming; --observations selects only returned/saved data.
Omitting it uses the server's defaults. Stdout and observations/latest/observation.json
contain requested values/file paths and errors for unavailable or stale readings.
Each call replaces latest/; read its files before calling again.
Active/queued motion jobs reject the request: call again after completion.
The motion status is checked directly at localhost:18080 (--motion-server).
Wrench: raw (default) keeps legacy base feedback; wrist uses flange origin/axes;
tcp uses the configured TCP origin and world axes. Pose remains in base axes.
All modes subtract the observation server's fixed unloaded startup baseline.
Motion guards continue to use unzeroed feedback.
Tactile .wrench includes calibrated force_n and native torque with frame/units.
torque_nm is null until its scale is validated in observation_server.CONFIG.
.deformation is software-tared; optional .deformation_raw reuses the raw array.
Legacy tactile .force remains the uncalibrated SDK six-vector.
The tactile reference is fingertip (+20 mm) or connector (+30 mm), measured
from the pad midpoint; *_world rotates axes without changing that reference.

Python: from get_observation import get_observation; get_observation(["force"]).
Units and field names: docs/agent_prompt.txt. Connection/output options: --help.
Exit codes: 0 success, 1 incomplete data/service error, 2 invalid arguments.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import http.client
import io
import json
import math
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlsplit
import zipfile


ROOT = Path(__file__).resolve().parent
DEFAULT_SERVER = "http://127.0.0.1:18081"
DEFAULT_MOTION_SERVER = "http://127.0.0.1:18080"
MAX_BUNDLE_BYTES = 128 * 1024 * 1024


def valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def transform_wrench(sample, *, reference="tcp", frame="world"):
    """Shift a documented flange wrench to wrist/TCP, then rotate its axes.

    sample contains flange [F,M], measured tcp_pose_base and calibrated tool_end
    [xyz, xyz Euler], in N, N.m, metres and radians. R_base_flange = R_base_tcp
    @ R_flange_tcp.T; M_tcp = M_flange - cross(p_flange_tcp, F_flange).
    The project world frame maps base (x,y,z) to (x,z,-y).
    """
    if reference not in ("wrist", "tcp") or frame not in ("flange", "base", "world"):
        raise ValueError("Invalid wrench reference or frame")
    if not isinstance(sample, dict) or sample.get("error"):
        raise ValueError(sample.get("error", "Missing explicit flange wrench") if isinstance(sample, dict)
                         else "Missing explicit flange wrench; restart the robot/observation servers")

    def values(key):
        value = sample.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 6 or not all(valid_number(v) for v in value):
            raise ValueError(f"Invalid wrench {key}")
        return value

    def rotation(rpy):
        x, y, z = rpy
        sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
        cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
        return ((cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx),
                (sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx), (-sy, cy*sx, cy*cx))

    def apply(matrix, vector):
        return [sum(a*b for a, b in zip(row, vector)) for row in matrix]

    wrench = values("flange")
    force, torque = list(wrench[:3]), list(wrench[3:])
    if reference == "tcp":
        x, y, z = values("tool_end")[:3]
        fx, fy, fz = force
        moment = (y*fz-z*fy, z*fx-x*fz, x*fy-y*fx)
        torque = [m-offset for m, offset in zip(torque, moment)]
    if frame != "flange":
        base_tcp = rotation(values("tcp_pose_base")[3:])
        tcp_flange = tuple(zip(*rotation(values("tool_end")[3:])))
        force, torque = [apply(base_tcp, apply(tcp_flange, v)) for v in (force, torque)]
        if frame == "world":
            force, torque = [[v[0], v[2], -v[1]] for v in (force, torque)]
    result = force + torque
    if not all(valid_number(v) for v in result):
        raise ValueError("Nonfinite transformed wrench")
    return result


def positive(value, label="value"):
    if not valid_number(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def positive_number(value):
    try:
        return positive(float(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc


class ObservationError(RuntimeError):
    pass


@contextmanager
def latest_directory(output_dir):
    """Stage one result, then replace latest; serialize writers to this folder."""
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory, pending, previous = (root / name for name in ("latest", ".pending", ".previous"))
    with (root / ".observation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Recover an interrupted replacement. These are fixed working folders,
        # so even an interrupted client cannot accumulate snapshot directories.
        if previous.exists():
            if directory.exists():
                shutil.rmtree(previous)
            else:
                previous.replace(directory)
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir()
        try:
            yield pending, directory
            if directory.exists():
                directory.replace(previous)
            try:
                pending.replace(directory)
            except BaseException:
                if previous.exists():
                    previous.replace(directory)
                raise
            if previous.exists():
                shutil.rmtree(previous)
        finally:
            if pending.exists():
                shutil.rmtree(pending)


def save_bundle(raw, output_dir, request_started, max_age, *, wrench="raw"):
    """Validate the bundle and replace the latest saved observation as a set."""
    if len(raw) > MAX_BUNDLE_BYTES:
        raise ObservationError("Observation response exceeds size limit")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or sum(entry.file_size for entry in entries) > MAX_BUNDLE_BYTES:
            raise ObservationError("Duplicate archive names or excessive expanded size")
        if "manifest.json" not in names or archive.getinfo("manifest.json").file_size > 1024*1024:
            raise ObservationError("Missing or excessive observation manifest")
        result = json.loads(archive.read("manifest.json"))
        if (not isinstance(result, dict) or result.get("protocol") != 2 or
                set(result) != {"protocol", "observations", "errors", "ages"} or
                not all(isinstance(result[k], dict) for k in ("observations", "errors", "ages"))):
            raise ObservationError("Unsupported observation bundle; restart observation_server.py after updating it")
        result.pop("protocol")
        ages = result.pop("ages")
        if (set(ages) != set(result["observations"]) or
                set(result["errors"]) & set(result["observations"]) or
                not all(isinstance(error, str) for error in result["errors"].values())):
            raise ObservationError("Invalid observation readings/errors/ages")
        if wrench != "raw":
            for key in ("force", "state"):
                if key not in result["observations"]:
                    continue
                try:
                    value = result["observations"][key]
                    converted = transform_wrench(value if key == "force" else value["force"],
                                                 reference=wrench, frame="world" if wrench == "tcp" else "flange")
                    result["observations"][key] = converted if key == "force" else dict(value, force=converted)
                except (ValueError, TypeError, KeyError) as exc:
                    result["errors"][key] = f"Invalid wrench: {exc}"
                    del result["observations"][key], ages[key]
        required = {"manifest.json"}
        filenames = {}
        for key, value in result["observations"].items():
            match = re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.(rgb|depth|image|deformation|deformation_raw)", key)
            if match:
                suffix = ".png" if match[1] in ("rgb", "image") else ".npy"
                if value != key + suffix:
                    raise ObservationError("Invalid sensor filename in response")
                filenames[key] = value
                required.add(value)
            elif not isinstance(value, (list, dict)) and not valid_number(value):
                raise ObservationError("Invalid observation value")
        if set(names) != required:
            raise ObservationError("Missing or unexpected sensor files in response")
        with latest_directory(output_dir) as (pending, directory):
            for key, filename in filenames.items():
                (pending / filename).write_bytes(archive.read(filename))
                result["observations"][key] = str(directory / filename)
            elapsed = time.monotonic() - request_started
            for key, age in ages.items():
                if not valid_number(age) or age < 0:
                    result["errors"][key] = "Server returned no valid observation age"
                # The entire round trip bounds time since server selection,
                # including file writes and waiting for another client's lock.
                elif age + elapsed > max_age:
                    result["errors"][key] = "Observation is stale at client return"
                if key in result["errors"]:
                    del result["observations"][key]
                    if key in filenames:
                        (pending / filenames[key]).unlink()
            (pending / "observation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result


def server_address(server):
    url = urlsplit(server)
    if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost") or
            url.username is not None or url.password is not None or
            url.path not in ("", "/") or url.query or url.fragment):
        raise ValueError("server must be http://127.0.0.1:PORT or http://localhost:PORT")
    port = 80 if url.port is None else url.port
    if not 1 <= port <= 65535:
        raise ValueError("server port must be between 1 and 65535")
    return url.hostname, port


class ObservationClient:
    def __init__(self, server=DEFAULT_SERVER, timeout=5.0, *, motion_server=DEFAULT_MOTION_SERVER):
        self.host, self.port = server_address(server)
        self.motion_host, self.motion_port = server_address(motion_server)
        self.http_timeout = positive(timeout, "timeout")

    def ensure_motion_finished(self):
        """Check execution/queue state, independently of measured robot velocity."""
        connection = http.client.HTTPConnection(self.motion_host, self.motion_port, timeout=self.http_timeout)
        try:
            connection.request("GET", "/status")
            response = connection.getresponse()
            raw = response.read(1024 * 1024 + 1)
            if response.status != 200 or len(raw) > 1024 * 1024:
                raise ObservationError(f"Cannot verify motion completion: HTTP {response.status}")
            state = json.loads(raw)
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise ObservationError(f"Cannot verify motion completion at {self.motion_host}:{self.motion_port}; "
                                   "check the motion server and call again.") from exc
        finally:
            connection.close()
        fields = {"active", "queued", "pending_home", "stop_requested", "stopping"}
        if (not isinstance(state, dict) or not fields <= state.keys() or
                any(state[key] is not None and not isinstance(state[key], str)
                    for key in ("active", "pending_home")) or
                not isinstance(state["queued"], list) or
                not all(isinstance(job, str) for job in state["queued"]) or
                any(type(state[key]) is not bool for key in ("stop_requested", "stopping"))):
            raise ObservationError("Cannot verify motion completion: invalid motion-server status")
        if (state["active"] is not None or state["queued"] or state["pending_home"] is not None or
                state["stop_requested"] or state["stopping"]):
            raise ObservationError("Motion work is still active or queued; "
                                   "call get_observation.py again after completion.")

    def status(self):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.http_timeout)
        try:
            connection.request("GET", "/status")
            response = connection.getresponse()
            raw = response.read(1024 * 1024 + 1)
            if response.status != 200 or len(raw) > 1024 * 1024:
                raise ObservationError(f"Invalid observation status response: HTTP {response.status}")
            result = json.loads(raw)
        finally:
            connection.close()
        if not isinstance(result, dict) or result.get("protocol") != 1 or not isinstance(result.get("sources"), dict):
            raise ObservationError("Endpoint is not an observation server")
        return result

    def post_json(self, path, request):
        if path not in ("/calibrate", "/wrench-comparison"):
            raise ValueError("Unsupported observation JSON endpoint")
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.http_timeout)
        try:
            connection.request("POST", path, json.dumps(request, allow_nan=False), {"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read(1024*1024+1)
            if len(raw) > 1024*1024:
                raise ObservationError("JSON response exceeds size limit")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ObservationError("Expected observation JSON object")
            if response.status != 200:
                raise ObservationError(result.get("error", f"HTTP {response.status}"))
            return result
        except (OSError, http.client.HTTPException) as exc:
            raise ObservationError(f"Observation server unavailable: {exc}") from exc
        finally:
            connection.close()

    def calibrate(self, target):
        self.ensure_motion_finished()
        return self.post_json("/calibrate", {"target": target})

    def get_observation(self, observations=None, output_dir=ROOT / "observations", max_age=0.5, *,
                        wrench="raw", tactile_reference="fingertip"):
        max_age = positive(max_age, "max_age")
        if wrench not in ("raw", "wrist", "tcp"):
            raise ValueError("wrench must be raw, wrist or tcp")
        if tactile_reference not in ("fingertip", "connector"):
            raise ValueError("tactile_reference must be fingertip or connector")
        if observations is not None and (not isinstance(observations, list) or not observations or
                                        not all(isinstance(k, str) for k in observations)):
            raise ValueError("observations must be a nonempty list of names")
        request = {"max_age_s": max_age}
        if tactile_reference != "fingertip":
            request["tactile_reference"] = tactile_reference
        if wrench != "raw":
            request["wrench"] = wrench
        if observations is not None:
            request["observations"] = observations
        self.ensure_motion_finished()
        begin = time.monotonic()
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.http_timeout)
        try:
            connection.request("POST", "/observation", json.dumps(request), {"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read(MAX_BUNDLE_BYTES + 1)
            if response.status != 200:
                try:
                    detail = json.loads(raw).get("error", response.reason)
                except (ValueError, AttributeError):
                    detail = response.reason
                raise ObservationError(f"HTTP {response.status}: {detail}")
            if response.getheader("Content-Type", "").split(";")[0] != "application/zip":
                raise ObservationError("Expected an observation bundle from observation_server.py")
        except (OSError, http.client.HTTPException) as exc:
            raise ObservationError(f"Observation server unavailable at {self.host}:{self.port}: {exc}") from exc
        finally:
            connection.close()
        self.ensure_motion_finished()  # Reject if another job started during the fetch.
        return save_bundle(raw, output_dir, begin, max_age, wrench=wrench)


def get_observation(observations=None, *, server=DEFAULT_SERVER, output_dir=ROOT / "observations",
                    timeout=5.0, max_age=0.5, wrench="raw", tactile_reference="fingertip",
                    motion_server=DEFAULT_MOTION_SERVER):
    """Fetch and save only when the motion server has no unfinished work."""
    return ObservationClient(server, timeout, motion_server=motion_server).get_observation(
        observations, output_dir, max_age, wrench=wrench, tactile_reference=tactile_reference)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--motion-server", default=DEFAULT_MOTION_SERVER,
                        help="Motion service to check for unfinished work (default: %(default)s)")
    parser.add_argument("--timeout", type=positive_number, default=5.0)
    parser.add_argument("--max-age", type=positive_number, default=0.5)
    parser.add_argument("--wrench", choices=("raw", "wrist", "tcp"), default="raw",
                        help="force: legacy base (default), flange origin/axes, or TCP origin/world axes")
    parser.add_argument("--tactile-reference", choices=("fingertip", "connector"), default="fingertip",
                        help="Tactile wrench reference: 20 or 30 mm forward from the pad midpoint")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "observations",
                        help="Save under PATH/latest, replacing the previous result (default: %(default)s)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--observations", nargs="+", help="Select observations; omit for server defaults")
    action.add_argument("--status", action="store_true", help="Read server readiness without saving files")
    args = parser.parse_args(argv)
    try:
        client = ObservationClient(args.server, args.timeout, motion_server=args.motion_server)
        if args.status:
            print(json.dumps(client.status(), allow_nan=False))
            return 0
        result = client.get_observation(args.observations, args.output_dir, args.max_age,
                                        wrench=args.wrench, tactile_reference=args.tactile_reference)
        print(json.dumps(result, allow_nan=False))
        return 1 if result["errors"] else 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(json.dumps({"observations": {}, "errors": {"request": f"{type(exc).__name__}: {exc}"}}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
