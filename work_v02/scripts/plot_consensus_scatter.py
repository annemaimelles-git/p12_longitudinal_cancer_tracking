#!/usr/bin/env python
"""Scatter plot of the consensus check on the training predictions.

Horizontal: how far a prediction's implied displacement deviates from the median displacement
            of the patient's other lesions (computable at inference).
Vertical:   LesionLocator's actual error (known only because the data is annotated).

Usage:  python scripts/plot_consensus_scatter.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")                       # no screen on the cluster: draw straight into files
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

DATA = "/gpfs/helios/home/melles/ML_project/work/data/Longitudinal_CT/inputsTr"
TABLE = "results/lesionlocator_train/evaluation/per_lesion.csv"
OUT = "results/lesionlocator_train/evaluation/figures"
THRESHOLD, BAD_MM = 50, 50
os.makedirs(OUT, exist_ok=True)


def geometry(path):
    r = sitk.ImageFileReader(); r.SetFileName(path); r.ReadImageInformation()
    return (np.array(r.GetOrigin()), np.array(r.GetSpacing()),
            np.array(r.GetDirection()).reshape(3, 3))


baseline = {}
for pid in {r["patient"] for r in csv.DictReader(open(TABLE))}:
    for r in csv.DictReader(open(f"{DATA}/{pid}.csv")):
        if r["cog_bl"].strip() and r["img_id_bl"].strip():
            o, s, d = geometry(f"{DATA}/{pid}_BL_img_{int(float(r['img_id_bl'])):02d}.nii.gz")
            baseline[(pid, r["lesion_id"])] = o + d @ (np.array([float(v) for v in r["cog_bl"].split()]) * s)

rows = [r for r in csv.DictReader(open(TABLE))
        if r["pred_x"] and r["err_lesionlocator_mm"] and r["dist_to_init_mm"]]
pos = lambda r, p: np.array([float(r[f"{p}_{a}"]) for a in "xyz"])

by_patient = {}
for r in rows:
    key = (r["patient"], r["lesion_id"])
    if key in baseline:
        r["shift"] = pos(r, "pred") - baseline[key]
        by_patient.setdefault(r["patient"], []).append(r)

kept = []
for pid, group in by_patient.items():
    if len(group) < 3:                      # a median needs a few values
        continue
    for r in group:
        others = np.stack([o["shift"] for o in group if o is not r])
        r["dev"] = float(np.linalg.norm(r["shift"] - np.median(others, axis=0)))
        kept.append(r)

err = np.array([float(r["err_lesionlocator_mm"]) for r in kept])
dev = np.array([r["dev"] for r in kept])
bad = err > BAD_MM

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(np.maximum(dev[~bad], 0.1), np.maximum(err[~bad], 0.1), s=20, alpha=0.6,
           color="tab:green", label=f"error <= {BAD_MM} mm ({(~bad).sum()})")
ax.scatter(np.maximum(dev[bad], 0.1), np.maximum(err[bad], 0.1), s=40, alpha=0.85,
           color="tab:red", label=f"error > {BAD_MM} mm ({bad.sum()})")
ax.axvline(THRESHOLD, color="black", ls="--", lw=1.2)
ax.axhline(BAD_MM, color="gray", ls=":", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
# short horizontal label at the top of the threshold line, instead of rotated text
ax.annotate(f"reject beyond {THRESHOLD} mm", xy=(THRESHOLD, ax.get_ylim()[1]),
            xytext=(THRESHOLD * 1.15, ax.get_ylim()[1] * 0.55),
            fontsize=9, ha="left", va="top")
ax.set_xlabel("deviation from the patient's median displacement (mm)")
ax.set_ylabel("LesionLocator's error (mm)")
ax.grid(alpha=0.3, which="both")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
path = os.path.join(OUT, "consensus_scatter.png")
fig.savefig(path, dpi=130)
plt.close(fig)

print(f"{len(kept)} predictions, {bad.sum()} with an error > {BAD_MM} mm")
print(f"caught {np.mean(dev[bad] > THRESHOLD):.1%} of them, "
      f"rejected {np.mean(dev[~bad] > THRESHOLD):.1%} of the accurate ones")
print(f"figure: {path}")
