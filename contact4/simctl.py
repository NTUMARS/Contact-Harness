#!/usr/bin/env python3
"""Debug helper for the simulated robot (sim-only RPCs): state, wrench, renders."""
import argparse, os, sys
from pathlib import Path
import numpy as np, zmq, msgpack

def rpc(method, params=None, address=None, timeout=5.0):
    address = address or ("tcp://127.0.0.1:5656" if method.startswith("sim_") else "tcp://127.0.0.1:5655")
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0); s.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
    s.connect(address); s.send(msgpack.packb({"method": method, "params": params or {},
                                              "token": os.environ.get("SIM_RPC_TOKEN", "")}))
    r = msgpack.unpackb(s.recv(), raw=False); s.close()
    if not r["success"]: raise RuntimeError(r["error"])
    return r["result"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["state", "render"])
    ap.add_argument("--camera", nargs="+", default=["overview"]); ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--size", type=int, nargs=2, default=[640, 480])
    a = ap.parse_args()
    if a.cmd == "state":
        st = rpc("motion_state", {"include_wrench": False}); W = np.array([[1,0,0],[0,0,1],[0,-1,0]])
        print("tcp world", np.round(W @ np.array(st["cart_pos_real"][:3]), 4))
        print("ext force world", np.round(W @ np.array(st["ext_wrench"][:3]), 2))
        print("in_rt", st["in_realtime_loop"], "sim", rpc("sim_info"))
    else:
        import cv2
        a.out.mkdir(parents=True, exist_ok=True)
        for cam in a.camera:
            r = rpc("sim_render", {"camera": cam, "width": a.size[0], "height": a.size[1]})
            img = np.frombuffer(r["data"], np.uint8).reshape(r["shape"])
            cv2.imwrite(str(a.out / f"{cam}.png"), img[:, :, ::-1]); print(a.out / f"{cam}.png")
