#!/usr/bin/env python
"""Write point and box prompt JSONs at the consensus position, for the lesions
LesionLocator's tracking missed.

The JSON format (documentation/prompting.md) is Z, Y, X order:
  point: ["z", "y", "x"]
  box:   [z_min, z_max, y_min, y_max, x_min, x_max], half-open
Everything else in this project is X, Y, Z, so the order is swapped only at the end.

Box half-widths are per axis, from the p90 consensus error plus the p90 lesion radius.
Usage:  python scripts/make_prompts.py [hx hy hz]      (half-widths in mm, x y z)
"""
import csv
import json
import os
import sys

import numpy as np
import SimpleITK as sitk

DATA = "/gpfs/helios/home/melles/ML_project/work/data/Longitudinal_CT/inputsTr"
TABLE = "results/lesionlocator_val/evaluation/per_lesion.csv"
OUT = "prompts"
HALF_MM = np.array([float(v) for v in sys.argv[1:4]]) if len(sys.argv) > 3 else np.array([32., 43., 51.])


def header(path):
    r = sitk.ImageFileReader(); r.SetFileName(path); r.ReadImageInformation()
    return r


def geometry(path):
    r = header(path)
    return (np.array(r.GetOrigin()), np.array(r.GetSpacing()),
            np.array(r.GetDirection()).reshape(3, 3))


rows = [r for r in csv.DictReader(open(TABLE)) if r["true_x"]]
pos = lambda r, p: np.array([float(r[f"{p}_{a}"]) for a in "xyz"]) if r[f"{p}_x"] else None

baseline = {}
for pid in {r["patient"] for r in rows}:
    for r in csv.DictReader(open(f"{DATA}/{pid}.csv")):
        if r["cog_bl"].strip() and r["img_id_bl"].strip():
            o, s, d = geometry(f"{DATA}/{pid}_BL_img_{int(float(r['img_id_bl'])):02d}.nii.gz")
            baseline[(pid, r["lesion_id"])] = o + d @ (np.array([float(v) for v in r["cog_bl"].split()]) * s)

by_patient = {}
for r in rows:
    by_patient.setdefault(r["patient"], []).append(r)

os.makedirs(f"{OUT}/point", exist_ok=True)
os.makedirs(f"{OUT}/box", exist_ok=True)
n_les = n_clipped = 0

for pid, group in by_patient.items():
    shifts = [pos(r, "pred") - baseline[(pid, r["lesion_id"])]
              for r in group if r["pred_x"] and (pid, r["lesion_id"]) in baseline]
    if len(shifts) < 3:
        continue                                            # no consensus for this patient
    m = np.median(np.stack(shifts), axis=0)

    points, boxes, fu_name = {}, {}, None
    for r in group:
        key = (pid, r["lesion_id"])
        if r["pred_x"] or key not in baseline:              # only the missed lesions
            continue
        fu_id = int(float(r["img_id_fu"])) if r.get("img_id_fu", "").strip() else 0
        fu_name = f"{pid}_FU_img_{fu_id:02d}"
        img = header(f"{DATA}/{fu_name}.nii.gz")
        size = np.array(img.GetSize())                      # (x, y, z) voxels
        spacing = np.array(img.GetSpacing())

        # consensus position in mm -> continuous voxel index (x, y, z)
        origin, sp, direction = geometry(f"{DATA}/{fu_name}.nii.gz")
        idx = np.linalg.inv(direction) @ (baseline[key] + m - origin) / sp
        half_vox = HALF_MM / spacing                        # mm -> voxels, per axis
        lo = np.maximum(0, np.round(idx - half_vox)).astype(int)
        hi = np.minimum(size, np.round(idx + half_vox) + 1).astype(int)   # half-open
        if np.any(lo == 0) or np.any(hi == size):
            n_clipped += 1

        lid = str(int(float(r["lesion_id"])))
        points[lid] = {"point": [f"{idx[2]:.2f}", f"{idx[1]:.2f}", f"{idx[0]:.2f}"]}   # z, y, x
        boxes[lid] = {"box": [str(lo[2]), str(hi[2]), str(lo[1]), str(hi[1]),
                              str(lo[0]), str(hi[0])]}                                  # z, y, x
        n_les += 1

    if points:
        json.dump(points, open(f"{OUT}/point/{fu_name}.json", "w"), indent=2)
        json.dump(boxes, open(f"{OUT}/box/{fu_name}.json", "w"), indent=2)

print(f"{n_les} missed lesions written, {n_clipped} with a box clipped at the image edge")
print(f"box half-widths (x, y, z): {HALF_MM} mm")
print(f"files: {OUT}/point/ and {OUT}/box/")
