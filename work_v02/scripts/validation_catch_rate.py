#!/usr/bin/env python
"""Catch and rejection rates of the consensus check on the VALIDATION patients,
using the 50 mm threshold chosen on the training patients.

  caught   = share of predictions with an error > 50 mm that the check rejects
  rejected = share of accurate predictions (error <= 50 mm) that the check rejects

Usage:  python scripts/validation_catch_rate.py
"""
import csv

import numpy as np
import SimpleITK as sitk

DATA = "/gpfs/helios/home/melles/ML_project/work/data/Longitudinal_CT/inputsTr"
TABLE = "results/lesionlocator_val/evaluation/per_lesion.csv"
THRESHOLD, BAD_MM = 50, 50


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

rows = [r for r in csv.DictReader(open(TABLE)) if r["pred_x"] and r["err_lesionlocator_mm"]]
pos = lambda r, p: np.array([float(r[f"{p}_{a}"]) for a in "xyz"])

by_patient = {}
for r in rows:
    key = (r["patient"], r["lesion_id"])
    if key in baseline:
        r["shift"] = pos(r, "pred") - baseline[key]
        by_patient.setdefault(r["patient"], []).append(r)

kept = []
for pid, group in by_patient.items():
    if len(group) < 3:                      # no consensus without a few lesions
        continue
    for r in group:
        others = np.stack([o["shift"] for o in group if o is not r])
        r["dev"] = float(np.linalg.norm(r["shift"] - np.median(others, axis=0)))
        kept.append(r)

err = np.array([float(r["err_lesionlocator_mm"]) for r in kept])
dev = np.array([r["dev"] for r in kept])
bad = err > BAD_MM

print(f"{len(kept)} validation predictions with a consensus "
      f"({len(rows) - len(kept)} in patients with too few predictions)")
print(f"{bad.sum()} have an error > {BAD_MM} mm\n")
print(f"at the {THRESHOLD} mm threshold chosen on training:")
print(f"  caught   {np.mean(dev[bad] > THRESHOLD):6.1%} of the catastrophic predictions "
      f"({np.sum(dev[bad] > THRESHOLD)} of {bad.sum()})")
print(f"  rejected {np.mean(dev[~bad] > THRESHOLD):6.1%} of the accurate predictions "
      f"({np.sum(dev[~bad] > THRESHOLD)} of {(~bad).sum()})")
