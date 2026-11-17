#!/usr/bin/env python
"""Per-axis error of the consensus position on the lesions LesionLocator missed.

A box prompt centred on the consensus position must cover two things: how far that
position is from the true centre (per axis), and the lesion's own radius. This prints
both, as percentiles, so the box half-width can be chosen rather than guessed.
"""
import csv

import numpy as np
import SimpleITK as sitk

DATA = "/gpfs/helios/home/melles/ML_project/work/data/Longitudinal_CT/inputsTr"
TABLE = "results/lesionlocator_val/evaluation/per_lesion.csv"


def geometry(path):
    r = sitk.ImageFileReader(); r.SetFileName(path); r.ReadImageInformation()
    return (np.array(r.GetOrigin()), np.array(r.GetSpacing()),
            np.array(r.GetDirection()).reshape(3, 3))


rows = [r for r in csv.DictReader(open(TABLE)) if r["true_x"]]
pos = lambda r, p: np.array([float(r[f"{p}_{a}"]) for a in "xyz"]) if r[f"{p}_x"] else None

baseline, vol_fu = {}, {}
for pid in {r["patient"] for r in rows}:
    for r in csv.DictReader(open(f"{DATA}/{pid}.csv")):
        key = (pid, r["lesion_id"])
        if r["cog_bl"].strip() and r["img_id_bl"].strip():
            o, s, d = geometry(f"{DATA}/{pid}_BL_img_{int(float(r['img_id_bl'])):02d}.nii.gz")
            baseline[key] = o + d @ (np.array([float(v) for v in r["cog_bl"].split()]) * s)
        if r.get("volume_fu", "").strip():
            vol_fu[key] = float(r["volume_fu"])

by_patient = {}
for r in rows:
    by_patient.setdefault(r["patient"], []).append(r)

delta, radii, spacings = [], [], []
for pid, group in by_patient.items():
    shifts = [pos(r, "pred") - baseline[(pid, r["lesion_id"])]
              for r in group if r["pred_x"] and (pid, r["lesion_id"]) in baseline]
    if len(shifts) < 3:
        continue
    m = np.median(np.stack(shifts), axis=0)
    for r in group:
        key = (pid, r["lesion_id"])
        if r["pred_x"] or key not in baseline:          # only the missed lesions
            continue
        delta.append((baseline[key] + m) - pos(r, "true"))     # consensus minus truth, per axis
        if key in vol_fu:                                       # sphere-equivalent radius
            radii.append((3 * vol_fu[key] / (4 * np.pi)) ** (1 / 3))
        fu_id = int(float(r["img_id_fu"])) if r.get("img_id_fu", "").strip() else 0
        spacings.append(geometry(f"{DATA}/{pid}_FU_img_{fu_id:02d}.nii.gz")[1])

delta = np.abs(np.stack(delta))
radii = np.array(radii)
sp = np.median(np.stack(spacings), axis=0)

print(f"{len(delta)} missed lesions with a consensus position\n")
print("absolute error of the consensus position, per axis (mm):")
print(f"{'':>12}{'x':>9}{'y':>9}{'z':>9}")
for q in (50, 75, 90, 95, 100):
    v = np.percentile(delta, q, axis=0)
    name = "max" if q == 100 else f"p{q}"
    print(f"{name:>12}{v[0]:>9.1f}{v[1]:>9.1f}{v[2]:>9.1f}")

print(f"\nfollow-up lesion radius (sphere-equivalent, mm), n={len(radii)}:")
for q in (50, 75, 90, 95):
    print(f"  p{q}: {np.percentile(radii, q):.1f}")

print(f"\nmedian follow-up spacing (x, y, z): {np.round(sp, 2)} mm per voxel")
print("suggested half-widths = per-axis error percentile + lesion radius percentile:")
for q in (75, 90, 95):
    half = np.percentile(delta, q, axis=0) + np.percentile(radii, q)
    vox = half / sp
    print(f"  p{q}: {np.round(half, 0)} mm  =  {np.round(vox, 0)} voxels (x, y, z)")
