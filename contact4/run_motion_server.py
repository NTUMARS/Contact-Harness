#!/usr/bin/env python3
"""Run stack/motion_server.py with the kinematic model of the simulated robot registered.

motion_server builds its IK from a URDF looked up by the robot type that the robot server
reports. This launcher registers the URDF of the simulated Panda under that type and then
executes motion_server.py as __main__ with the given arguments.
"""
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
STACK = HERE / "stack"
SIM_MODELS = {"SimPanda-LIBERO": (str(HERE / "assets" / "sim_panda.urdf"), "sim_panda_tcp")}

sys.path.insert(0, str(STACK))
import robot_support.ik as ik  # noqa: E402

ik.MODELS.update(SIM_MODELS)

if __name__ == "__main__":
    os.chdir(STACK)
    sys.path.insert(0, str(STACK))
    sys.argv = [str(STACK / "motion_server.py")] + sys.argv[1:]
    override = os.environ.get("SIM_STATE_TIMEOUT_S")
    # Ablations (simulation only), set by episode_ctl.py for the servers only:
    #   no_force       the status reply withholds the measured wrench (execution keeps using it)
    #   no_soft_limit  no soft-force clipping (the setting is still accepted); only the hard limit stops execution
    #   no_force_tactile  as no_force (force and tactile ablated together; tactile is withheld by
    #                  run_observation_server.py)
    #   soft_limit_stop  same trigger as the soft limit (an axis over the threshold while the step
    #                  pushes against it), but the motion stops there (job failed, reason soft_force_limit) instead of
    #                  clipping that component and continuing; the agent issues the next command. Ablation runs only
    #   no_force_limit  ("w/o soft force limit") no force limit at all, as a robot without the
    #                  force-limiting layer: no soft-force clipping (set-soft-force still accepted, as no_soft_limit) and
    #                  no hard-force abort (workspace and joint bounds still checked); every motion runs to completion.
    #                  The simulator skips its 50 N target refusal too (sim_robot_server.py NO_FORCE_LIMIT).
    #   no_compliance_limit  ("w/o compliant execution and soft force limit") as
    #                  no_force_limit here; the simulator also replaces the 400 N/m impedance by stiff execution
    #                  (3000 N/m, 300 Nm/rad, sim_robot_server.py STIFF_EXECUTION).
    ablation = os.environ.get("SIM_ABLATION", "")
    if not override and ablation not in ("no_force", "no_force_tactile", "no_soft_limit", "soft_limit_stop",
                                         "no_force_limit", "no_compliance_limit"):
        runpy.run_path(str(STACK / "motion_server.py"), run_name="__main__")
    else:
        mod = runpy.run_path(str(STACK / "motion_server.py"), run_name="motion_server_sim")
        service_cls = mod["MotionService"]
        unavailable = "Not available in this configuration"
        if override:
            # Optional, simulation only: relax the idle-protection state_timeout (0.1 s on the real
            # robot) without editing the stack copy. Unset by default; the stack's own value then applies.
            timeout = float(override)
            original_init = service_cls.__init__

            def init(self, cfg, *a, **k):
                cfg.state_timeout = timeout
                print(f"[run_motion_server] state_timeout overridden to {timeout:g} s (SIM_STATE_TIMEOUT_S)",
                      file=sys.stderr, flush=True)
                original_init(self, cfg, *a, **k)

            service_cls.__init__ = init
        if ablation in ("no_force", "no_force_tactile"):
            status_force = service_cls.status

            def status(self, ident=None):
                reply = status_force(self, ident)
                fb = reply.get("feedback") if ident is None else None
                if isinstance(fb, dict):                   # measured wrench, also inside the flange wrench context
                    fb = dict(fb, ext_wrench=unavailable)
                    if isinstance(fb.get("wrench"), dict):
                        fb["wrench"] = dict(fb["wrench"], flange=unavailable)
                    reply["feedback"] = fb
                return reply

            service_cls.status = status
        if ablation == "no_soft_limit":
            # silent: the interface is unchanged (set-soft-force is accepted and reported as on the
            # full system) but execution never clips, so only the hard limit acts
            g = service_cls._dispatch.__globals__
            g["constrain_translation"] = lambda dp, force, threshold: (dp.copy(), g["np"].zeros(3, dtype=bool))
        if ablation in ("no_force_limit", "no_compliance_limit"):
            g = service_cls._dispatch.__globals__
            g["constrain_translation"] = lambda dp, force, threshold: (dp.copy(), g["np"].zeros(3, dtype=bool))

            def hard_bounds_only(self, state, candidate=None, candidate_q=None):
                pose, q, wrench = self._check_state(state)          # state validity as before
                self._bounds(pose, q)                                # workspace / joint bounds as before
                if candidate is not None:
                    self._bounds(candidate, candidate_q)
                return pose, q, wrench                               # no hard-force abort

            service_cls._hard = hard_bounds_only
        if ablation == "soft_limit_stop":
            g = service_cls._dispatch.__globals__
            clip, motion_error = g["constrain_translation"], g["MotionError"]

            def stop_at_limit(dp, force, threshold):
                result, blocked = clip(dp, force, threshold)
                if g["np"].any(blocked):
                    axes = "".join("xyz"[i] for i in range(3) if blocked[i])
                    raise motion_error("soft_force_limit", f"contact force {float(g['np'].abs(force[blocked]).max()):.1f} N "
                                       f"exceeds the soft force limit {threshold:g} N along base {axes}; motion stopped")
                return result, blocked

            g["constrain_translation"] = stop_at_limit
        print(f"[run_motion_server] ablation: {ablation or 'none'}", file=sys.stderr, flush=True)
        mod["main"]()
