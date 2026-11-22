#!/usr/bin/env python3
"""Fetch Franka observations from localhost:8082; motion status uses 8081.

  python get_observation.py --observations state wrist.rgb side.rgb
  python get_observation.py --observations force torque --wrench tcp
  python get_observation.py --observations all
  python get_observation.py --calibrate
  python get_observation.py --status

The server continuously captures cameras and polls robot state. Requests select
only the returned/saved data. Each call atomically replaces observations/latest.
Active or queued motion rejects a snapshot. Sources are not synchronized.
force remains [Fx,Fy,Fz,Tx,Ty,Tz]; torque selects its last three components.
raw: EE origin/base axes; wrist: flange origin/flange axes;
tcp: EE origin/configured world axes. All use the robot wrench estimate, with
the observation server's startup tare. No tactile SDK is used.
The client needs only the Python standard library.
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
DEFAULT_SERVER = "http://127.0.0.1:8082"
DEFAULT_MOTION_SERVER = "http://127.0.0.1:8081"
MAX_BUNDLE_BYTES = 128 * 1024 * 1024



def valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def transform_wrench(sample, *, reference="tcp", frame="world"):
    """Shift a documented flange wrench to wrist/TCP, then rotate its axes.

    sample contains flange [F,M], measured tcp_pose_base and calibrated tool_end
    [xyz, xyz Euler], in N, N.m, metres and radians. R_base_flange = R_base_tcp
    @ R_flange_tcp.T; M_tcp = M_flange - cross(p_flange_tcp, F_flange).
    The world rotation is explicit in sample['world_from_base'].
    """
    if reference not in ("wrist", "tcp") or frame not in ("flange", "base", "world"):
        raise ValueError("Invalid wrench reference or frame")
    if not isinstance(sample, dict) or sample.get("error"):
        raise ValueError(sample.get("error", "Missing explicit flange wrench") if isinstance(sample, dict)
                         else "Missing explicit flange wrench; check observation-server transform configuration")

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
            matrix = sample.get("world_from_base")
            if (not isinstance(matrix, list) or len(matrix) != 3 or
                    any(not isinstance(row, list) or len(row) != 3 or
                        not all(valid_number(v) for v in row) for row in matrix)):
                raise ValueError("Missing configured world_from_base rotation")
            force, torque = [apply(matrix, v) for v in (force, torque)]
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
        required = {"manifest.json"}
        filenames = {}
        for key, value in result["observations"].items():
            match = re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.(rgb|depth)", key)
            if match:
                suffix = ".png" if match[1] == "rgb" else ".npy"
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
        if path != "/calibrate":
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

    def calibrate(self, target="robot_wrench"):
        self.ensure_motion_finished()
        return self.post_json("/calibrate", {"target": target})

    def get_observation(self, observations=None, output_dir=ROOT / "observations", max_age=0.5, *,
                        wrench="raw", wrench_source="franka", wrench_reference="ee"):
        if wrench_source != "franka":
            raise ValueError("Only the Franka wrench estimate is available; tactile sensors are not installed")
        max_age = positive(max_age, "max_age")
        if wrench not in ("raw", "wrist", "tcp"):
            raise ValueError("wrench must be raw, wrist or tcp")
        if observations is not None and (not isinstance(observations, list) or not observations or
                                        not all(isinstance(k, str) for k in observations)):
            raise ValueError("observations must be a nonempty list of names")
        if wrench_reference not in ("ee", "fingertip", "connector"):
            raise ValueError("wrench_reference must be ee, fingertip or connector")
        if wrench == "wrist" and wrench_reference != "ee":
            raise ValueError("wrist mode fixes the reference at the flange")
        request = {"max_age_s": max_age, "wrench": wrench, "wrench_source": wrench_source,
                   "wrench_reference": wrench_reference}
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
        self.ensure_motion_finished()
        # Wrench coordinates were converted by the server before bundling.
        return save_bundle(raw, output_dir, begin, max_age)



def get_observation(observations=None, *, server=DEFAULT_SERVER, output_dir=ROOT / "observations",
                    timeout=5.0, max_age=0.5, wrench="raw",
                    motion_server=DEFAULT_MOTION_SERVER, wrench_source="franka", wrench_reference="ee"):
    """Fetch and save only when the motion server has no unfinished work."""
    return ObservationClient(server, timeout, motion_server=motion_server).get_observation(
        observations, output_dir, max_age, wrench=wrench, wrench_source=wrench_source, wrench_reference=wrench_reference)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--motion-server", default=DEFAULT_MOTION_SERVER)
    parser.add_argument("--timeout", type=positive_number, default=5.0)
    parser.add_argument("--max-age", type=positive_number, default=0.5)
    parser.add_argument("--wrench-source", choices=("franka",), default="franka")
    parser.add_argument("--wrench", choices=("raw", "wrist", "tcp"), default="raw",
                        help="Wrench origin/axes: EE/base, flange/flange, or EE/configured world")
    parser.add_argument("--wrench-reference", choices=("ee", "fingertip", "connector"), default="ee",
                        help="Wrench origin; fingertip/connector require measured offsets in the server config")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "observations")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--observations", nargs="+", help="Select observations; omit for server defaults")
    action.add_argument("--status", action="store_true", help="Read readiness without saving files")
    action.add_argument("--calibrate", action="store_true", help="Re-tare robot wrench for one second while still/unloaded")
    args = parser.parse_args(argv)
    try:
        client = ObservationClient(args.server, args.timeout, motion_server=args.motion_server)
        if args.status:
            result = client.status()
        elif args.calibrate:
            result = client.calibrate()
        else:
            result = client.get_observation(args.observations, args.output_dir, args.max_age,
                                            wrench=args.wrench, wrench_source=args.wrench_source, wrench_reference=args.wrench_reference)
        print(json.dumps(result, allow_nan=False))
        return 1 if result.get("errors") else 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(json.dumps({"observations": {}, "errors": {"request": f"{type(exc).__name__}: {exc}"}}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
