#!/usr/bin/env python
"""Why does LesionLocator miss 70 of 194 lesions? Two figures:

  1_volume_change.png  baseline volume vs follow-up volume, found and missed lesions.
                       Points far from the diagonal changed size dramatically between scans.
  2_by_organ.png       detection rate per lesion type, with the number of lesions per bar.

Usage:  python plot_misses.py [evaluation_folder]
"""
import csv
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")                       # no screen on the cluster: draw straight into files
import matplotlib.pyplot as plt
import numpy as np

DATA = "/gpfs/helios/home/melles/ML_project/work/data/Longitudinal_CT/inputsTr"
EVAL = sys.argv[1] if len(sys.argv) > 1 else \
    "/gpfs/helios/home/melles/ML_project/work_v02/results/lesionlocator_val/evaluation"
FIGS = os.path.join(EVAL, "figures")
os.makedirs(FIGS, exist_ok=True)

# lesions with a known follow-up centre, split into found (a mask was produced) and missed
rows = [r for r in csv.DictReader(open(os.path.join(EVAL, "per_lesion.csv"))) if r["true_x"]]
volumes = {}
for pid in {r["patient"] for r in rows}:
    for r in csv.DictReader(open(os.path.join(DATA, f"{pid}.csv"))):
        if r.get("volume_bl", "").strip() and r.get("volume_fu", "").strip():
            volumes[(pid, r["lesion_id"])] = (float(r["volume_bl"]), float(r["volume_fu"]))

found = [r for r in rows if r["pred_x"]]
missed = [r for r in rows if r["pred_voxels"] == "0"]
vol = lambda group: np.array([volumes[(r["patient"], r["lesion_id"])] for r in group
                              if (r["patient"], r["lesion_id"]) in volumes])

# ---------------------------------------------------------------- 1) volume change
fig, ax = plt.subplots(figsize=(7, 6.5))
lims = (1, 1e6)
ax.plot(lims, lims, "k--", lw=1, label="unchanged volume")
for f in (3, 1 / 3):                        # lines marking a three-fold change in either direction
    ax.plot(lims, [l * f for l in lims], color="gray", ls=":", lw=1)
for group, name, colour, size in ((found, "found", "tab:blue", 20), (missed, "missed", "tab:red", 32)):
    v = vol(group)
    ratio = v[:, 1] / v[:, 0]
    big = np.mean((ratio > 3) | (ratio < 1 / 3))
    ax.scatter(v[:, 0], v[:, 1], s=size, alpha=0.65, color=colour,
               label=f"{name} ({len(v)}); {big:.0%} changed > 3x")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("baseline volume (mm$^3$)")
ax.set_ylabel("follow-up volume (mm$^3$)")
ax.set_title("Lesions LesionLocator missed changed size far more\n"
             "(dashed: unchanged, dotted: three-fold change)")
ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "1_volume_change.png"), dpi=120); plt.close(fig)

# ---------------------------------------------------------------- 2) detection rate per organ
f_type, m_type = Counter(r["lesion_type"] for r in found), Counter(r["lesion_type"] for r in missed)
types = sorted(set(f_type) | set(m_type), key=lambda t: -(f_type[t] + m_type[t]))
rates = [100 * f_type[t] / (f_type[t] + m_type[t]) for t in types]
totals = [f_type[t] + m_type[t] for t in types]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(range(len(types)), rates, color="tab:blue", alpha=0.75)
overall = 100 * len(found) / (len(found) + len(missed))
ax.axhline(overall, color="black", ls="--", lw=1, label=f"overall {overall:.0f}%")
for i, (b, n) in enumerate(zip(bars, totals)):
    ax.text(i, b.get_height() + 2, f"n={n}", ha="center", fontsize=9)
ax.set_xticks(range(len(types)))
ax.set_xticklabels([t.replace(" / ", "/\n") for t in types], fontsize=9)
ax.set_ylabel("% of lesions for which a prediction was produced")
ax.set_ylim(0, 105)
ax.set_title("Detection rate per lesion type (validation)")
ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "2_by_organ.png"), dpi=120); plt.close(fig)

print(f"{len(found)} found, {len(missed)} missed")
for t in types:
    print(f"  {t:>18}: {f_type[t]:>3} found, {m_type[t]:>3} missed "
          f"({100 * f_type[t] / (f_type[t] + m_type[t]):.0f}% detected)")
print("\nfigures:", *sorted(os.listdir(FIGS)), sep="\n  ")