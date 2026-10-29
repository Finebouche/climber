"""Log a sine of the right knee (robot hanging on N2) in the simulator, save it, print the tracking error."""
import numpy as np

from ..common import JOINTS, make_sim, output_dir
from .logs import record, save, tracking_error

sim = make_sim("N2")
q0 = sim.reset_hanging()["q"]
knee = JOINTS.index("right_knee")
t = np.arange(150) * sim.control_dt                          # 3 s
targets = np.tile(q0, (len(t), 1))
targets[:, knee] += 0.3 * np.sin(2 * np.pi * 1.0 * t)
log = record(sim, targets)
path = output_dir("identification") / "sim_knee_sine.npz"
save(log, path)
err = tracking_error(log)
print(f"right knee: tracking error {err[knee]:.3f} rad RMS, max current {np.nanmax(np.abs(log['current'][:, knee])):.2f} A")
print(path)
