"""Logs of a motion: the targets sent and what was measured, to compare the simulator with the robot.

A log is a dict of arrays: ``time`` (T), ``target``, ``q``, ``dq``, ``current`` (T x 19, order JOINTS), saved as
.npz.  Record the same targets in the simulator (``record``) and on the robot (a move that plays them and logs obs),
then ``compare``: where they differ is what to identify (servo model: BAM; friction, masses, latency).
"""
import numpy as np

KEYS = ("target", "q", "dq", "current")


def record(sim, targets) -> dict:
    """Play the joint targets (T x 19), one per control step, from the current state; the log."""
    log = {k: [] for k in ("time", *KEYS)}
    for target in targets:
        obs = sim.step(target)
        log["time"].append(obs["time"])
        log["target"].append(np.asarray(target, float))
        for k in KEYS[1:]:
            log[k].append(obs[k])
    return {k: np.asarray(v) for k, v in log.items()}


def save(log: dict, path):
    np.savez_compressed(path, **log)


def load(path) -> dict:
    with np.load(path) as f:
        return dict(f)


def compare(a: dict, b: dict) -> dict:
    """Per joint, RMS difference of the measured angles (rad) and currents (A) of two logs of the same targets."""
    n = min(len(a["q"]), len(b["q"]))
    return {"q_rms": np.sqrt(np.nanmean((a["q"][:n] - b["q"][:n]) ** 2, axis=0)),
            "current_rms": np.sqrt(np.nanmean((a["current"][:n] - b["current"][:n]) ** 2, axis=0))}


def tracking_error(log: dict) -> np.ndarray:
    """Per joint, RMS of target - measured angle (rad): how far the servos stay from their targets."""
    return np.sqrt(np.nanmean((log["target"] - log["q"]) ** 2, axis=0))
