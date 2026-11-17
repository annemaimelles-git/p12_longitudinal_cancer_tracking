#!/usr/bin/env python
"""Issue 2: a handful of mislocalised lesions dominate LesionLocator's mean error.

Per lesion, the initial estimate's error against LesionLocator's error. Points below the
diagonal are lesions LesionLocator placed more accurately; the few far above it are the
catastrophic mislocalisations. Only lesions where LesionLocator produced a mask are shown,
since the comparison needs both errors.

Usage:  python scripts/plot_catastrophes.py [evaluation_folder]
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")                       # no screen on the cluster: draw straight into files
import matplotlib.pyplot as plt
import numpy as np

EVAL = sys.argv[1] if len(sys.argv) > 1 else \
    "/gpfs/helios/home/melles/ML_project/work_v02/results/lesionlocator_val/evaluation"
FIGS = os.path.join(EVAL, "figures")
BIG_MM = 50                                 # a prediction this far from the truth counts as catastrophic
os.makedirs(FIGS, exist_ok=True)

rows = [r for r in csv.DictReader(open(os.path.join(EVAL, "per_lesion.csv")))
        if r["err_lesionlocator_mm"] and r["err_initial_mm"]]
init = np.array([float(r["err_initial_mm"]) for r in rows])
ll = np.array([float(r["err_lesionlocator_mm"]) for r in rows])
big = ll > BIG_MM

fig, ax1 = plt.subplots(figsize=(7.5, 6.5))
lims = (0.1, max(init.max(), ll.max()) * 1.6)
ax1.plot(lims, lims, "k--", lw=1)
ax1.fill_between(lims, lims, lims[0], color="tab:green", alpha=0.06)
ax1.text(0.16, lims[1] * 0.35, "LesionLocator worse", fontsize=9, color="tab:red")
ax1.text(lims[1] * 0.012, 0.16, "LesionLocator better", fontsize=9, color="tab:green")
ax1.scatter(np.maximum(init[~big], 0.1), np.maximum(ll[~big], 0.1), s=22, alpha=0.6,
            color="tab:green", label=f"error <= {BIG_MM} mm ({(~big).sum()})")
ax1.scatter(np.maximum(init[big], 0.1), np.maximum(ll[big], 0.1), s=45, alpha=0.85,
            color="tab:red", label=f"error > {BIG_MM} mm ({big.sum()})")
ax1.set_xscale("log"); ax1.set_yscale("log"); ax1.set_xlim(lims); ax1.set_ylim(lims)
ax1.set_xlabel("initial estimate: error (mm)")
ax1.set_ylabel("LesionLocator: error (mm)")
ax1.grid(alpha=0.3, which="both"); ax1.legend(loc="upper right", fontsize=9)

order = np.argsort(-ll)
share = np.cumsum(ll[order]) / ll.sum() * 100
k = big.sum()
mean_all, mean_rest = ll.mean(), ll[~big].mean()
ax1.set_title(f"Most lesions are placed more accurately, a few are placed far away\n"
              f"mean error {mean_all:.1f} mm over {len(ll)} predictions, "
              f"{mean_rest:.1f} mm without the {k} mislocalised ones", fontsize=11)
fig.tight_layout()
path = os.path.join(FIGS, "catastrophic_errors.png")
fig.savefig(path, dpi=120)
plt.close(fig)

print(f"{len(ll)} predictions; {k} with an error > {BIG_MM} mm")
print(f"  they carry {share[k - 1]:.0f}% of the total error")
print(f"  mean error: {mean_all:.2f} mm overall, {mean_rest:.2f} mm without them")
print(f"  initial estimate on the same lesions: {init.mean():.2f} mm overall, "
      f"{init[~big].mean():.2f} mm without them")
print(f"\nfigure: {path}")
