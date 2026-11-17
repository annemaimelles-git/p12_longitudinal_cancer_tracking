#!/usr/bin/env python
"""Why LesionLocator produces no prediction for some lesions. One figure, two panels:

  left   detection rate per lesion type (the share of lesions for which a mask was produced)
  right  baseline volume of found vs missed lesions (box plot, log scale)

Only lesions with an annotated follow-up centre and a propagated centre are used, so the
counts match the metrics tables (194 lesions on the validation set).

Usage:  python plot_detection.py [evaluation_folder]
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
MIN_PER_TYPE = 20                           # lesion types with fewer lesions are left out of the bar chart
os.makedirs(FIGS, exist_ok=True)

rows = [r for r in csv.DictReader(open(os.path.join(EVAL, "per_lesion.csv")))
        if r["true_x"] and r["init_x"]]     # same lesions as the metrics tables
volumes = {}
for pid in {r["patient"] for r in rows}:
    for r in csv.DictReader(open(os.path.join(DATA, f"{pid}.csv"))):
        if r.get("volume_bl", "").strip():
            volumes[(pid, r["lesion_id"])] = float(r["volume_bl"])

found = [r for r in rows if r["pred_x"]]
missed = [r for r in rows if not r["pred_x"]]
counts = Counter(r["lesion_type"] for r in rows)
common = lambda r: counts[r["lesion_type"]] >= MIN_PER_TYPE

f_type = Counter(r["lesion_type"] for r in found if common(r))
m_type = Counter(r["lesion_type"] for r in missed if common(r))
types = sorted(set(f_type) | set(m_type), key=lambda t: -(100 * f_type[t] / (f_type[t] + m_type[t])))
rates = [100 * f_type[t] / (f_type[t] + m_type[t]) for t in types]
totals = [f_type[t] + m_type[t] for t in types]
overall = 100 * len(found) / len(rows)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# ---- left: detection rate per lesion type
shades = plt.cm.RdPu(np.linspace(0.75, 0.3, len(types)))   # darkest pink = best rate
bars = ax1.bar(range(len(types)), rates, color=shades, edgecolor="white", linewidth=0.8)
ax1.axhline(overall, color="black", ls="--", lw=1, label=f"overall {overall:.0f}%")
for i, (b, n) in enumerate(zip(bars, totals)):
    ax1.text(i, b.get_height() + 2, f"n={n}", ha="center", fontsize=9)
ax1.set_xticks(range(len(types)))
ax1.set_xticklabels([t.replace(" / ", "/\n") for t in types], fontsize=9)
ax1.set_ylabel("lesions with a prediction (%)")
ax1.set_ylim(0, 105)
ax1.set_title(f"Detection rate by lesion type\n(types with at least {MIN_PER_TYPE} lesions; {sum(1 for r in rows if not common(r))} others not shown)")
ax1.grid(alpha=0.3, axis="y"); ax1.legend(fontsize=9, loc="lower left")

# ---- right: baseline volume, found vs missed (overlapping histograms)
vol = lambda g: np.array([volumes[(r["patient"], r["lesion_id"])] for r in g
                          if (r["patient"], r["lesion_id"]) in volumes])
v_found, v_missed = vol(found), vol(missed)
bins = np.logspace(np.log10(min(v_found.min(), v_missed.min())),
                   np.log10(max(v_found.max(), v_missed.max())), 25)
for v, name, colour in ((v_found, "found", "tab:green"), (v_missed, "missed", "tab:red")):
    ax2.hist(v, bins=bins, alpha=0.55, color=colour, edgecolor="white", linewidth=0.5,
             label=f"{name} (n={len(v)}), median {np.median(v):.0f} mm$^3$")
    ax2.axvline(np.median(v), color=colour, ls="--", lw=2)
ax2.set_xscale("log")
ax2.set_xlabel("baseline lesion volume (mm$^3$)")
ax2.set_ylabel("number of lesions")
ax2.set_title("Missed lesions are smaller, but the ranges overlap")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="y")

fig.suptitle(f"LesionLocator produced no prediction for {len(missed)} of {len(rows)} validation lesions")
fig.tight_layout()
path = os.path.join(FIGS, "detection_by_type_and_size.png")
fig.savefig(path, dpi=120)
plt.close(fig)

print(f"{len(rows)} lesions: {len(found)} found ({overall:.0f}%), {len(missed)} missed")
for t, r_, n in zip(types, rates, totals):
    print(f"  {t:>18}: {f_type[t]:>3} found, {m_type[t]:>3} missed ({r_:.0f}% detected)")
print(f"\nbaseline volume (mm3)  found: median {np.median(v_found):.0f} "
      f"(25th {np.percentile(v_found, 25):.0f}, 75th {np.percentile(v_found, 75):.0f})")
print(f"                      missed: median {np.median(v_missed):.0f} "
      f"(25th {np.percentile(v_missed, 25):.0f}, 75th {np.percentile(v_missed, 75):.0f})")
print(f"\nfigure: {path}")