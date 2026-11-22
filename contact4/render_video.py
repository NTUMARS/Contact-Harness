#!/usr/bin/env python3
"""Render an episode video from its 30 Hz state recording (runs/<RUN>/trajectory.npz).

Rebuilds the same benchmark/seed (deterministic initial scene), replays the recorded joint/object
states and renders head | left_wrist | right_wrist side by side with a status overlay, at real-time
speed (sim time). Run with .venv_sim:  python render_video.py runs/<RUN> [--speed 1] [--width 480]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import types

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sim_task_server import RobosuiteSim  # noqa: E402

BENCH = {"factory_peg_insert": "factory_peg", "wipe": "wipe", "write": "write",
         "pivot": "pivot", "stack": "stack", "place": "place"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed-up factor")
    ap.add_argument("--width", type=int, default=480)
    a = ap.parse_args()
    task = json.loads((a.run / "task.json").read_text())
    rec = np.load(a.run / "trajectory.npz")
    t, qpos, force = rec["t"], rec["qpos"], rec["force"]
    progress = json.loads(str(rec["progress"]))
    wipe_order = json.loads(str(rec["wipe_order"]))
    wiped = rec["wiped"]
    ink = rec["ink"] if "ink" in rec.files else np.zeros(len(t), int)
    ink_pts = rec["ink_pts"] if "ink_pts" in rec.files else np.zeros((0, 2))
    placed = 0
    sim = RobosuiteSim(types.SimpleNamespace(benchmark=BENCH[task["benchmark"]], seed=task.get("seed", 0)),
                       realtime=False)
    m, d = sim.m, sim.d
    marker_geoms = []
    if wipe_order:
        marker_geoms = [m.geom(g).id for g in wipe_order]
    cams = [("head", sim.cam_map["head"]), ("left_wrist", sim.cam_map["left_wrist"]),
            ("right_wrist", sim.cam_map["right_wrist"])]
    W = a.width
    H = int(W * 0.75)
    out = a.run / "episode.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W * 3}x{H + 40}", "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt",
                           "yuv420p", "-crf", "23", str(out)], stdin=subprocess.PIPE)
    step = max(1, int(round(a.speed)))
    for i in range(0, len(t), step):
        d.qpos[:] = qpos[i]
        d.qvel[:] = 0
        for k in range(int(wiped[i]) if len(marker_geoms) else 0):
            m.geom_rgba[marker_geoms[k]][3] = 0.0
        while placed < min(int(ink[i]), len(ink_pts)):
            sim.inner.place_dot(placed, *ink_pts[placed])
            placed += 1
        mujoco.mj_forward(m, d)
        tiles = []
        for label, cam in cams:
            img = sim._render_one(cam, W, H)[:, :, ::-1].copy()
            cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(img)
        frame = np.zeros((H + 40, W * 3, 3), np.uint8)
        frame[:H] = np.hstack(tiles)
        pr = progress[i] or {}
        if "turns_since_engage" in pr:
            status = "nut base %+.1f mm vs tip  xy %.1f mm  %s  turns %.2f%s" % (
                pr["nut_base_above_tip_mm"], pr["xy_dist_m"] * 1e3, "ENGAGED" if pr["engaged_threads"] else "free",
                pr["turns_since_engage"], "  SUCCESS" if pr.get("success") else "")
        elif "insertion_depth_m" in pr:
            status = "depth %.1f mm  xy %.1f mm  %s" % (pr["insertion_depth_m"] * 1e3, pr["xy_dist_m"] * 1e3,
                                                        "SUCCESS" if pr.get("success") else ("engaged" if pr.get("engaged") else ""))
        elif "recall" in pr:
            status = "ink %d  recall %.2f  precision %.2f  tip %.1f N%s" % (
                pr["ink_dots"], pr["recall"], pr["precision"], pr["tip_force_n"],
                "  SUCCESS" if pr.get("success") else ("  TIP CRUSHED" if pr.get("tip_crushed") else ""))
        elif "max_angle_deg" in pr:
            status = "box %.0f deg (max %.0f)  push %.1f N  peak %.1f N%s" % (
                pr["angle_deg"], pr["max_angle_deg"], pr["robot_box_force_n"], pr["peak_robot_box_force_n"],
                "  SUCCESS" if pr.get("success") else "")
        elif "red_above_green_m" in pr:                       # Stack
            status = "red over green %.1f mm  xy %.1f mm%s" % (pr["red_above_green_m"] * 1e3, pr["xy_offset_m"] * 1e3,
                                                             "  SUCCESS" if pr.get("success") else "")
        elif "bowl_above_plate_m" in pr:                      # Place
            status = "bowl over plate %.1f mm  xy %.1f mm%s" % (pr["bowl_above_plate_m"] * 1e3, pr["xy_offset_m"] * 1e3,
                                                              "  SUCCESS" if pr.get("success") else "")
        elif "fraction_wiped" in pr:
            status = "wiped %d/%d" % (pr["wiped_markers"], pr["num_markers"])
        else:
            status = ""
        txt = "t = %6.1f s   |F| = %5.1f N   %s" % (t[i] - t[0], force[i], status)
        cv2.putText(frame, txt, (10, H + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(out, "frames", len(range(0, len(t), step)))


if __name__ == "__main__":
    main()
