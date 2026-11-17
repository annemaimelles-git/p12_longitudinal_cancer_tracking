#!/usr/bin/env python
"""How accurate is the consensus position for the lesions LesionLocator missed?

Compares, on the lesions with an empty mask, the consensus fallback with cog_propagated
(the reference estimate, not available at inference) and with the plain baseline centre.
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

baseline = {}
for pid in {r["patient"] for r in rows}:
    for r in csv.DictReader(open(f"{DATA}/{pid}.csv")):
        if r["cog_bl"].strip() and r["img_id_bl"].strip():
            o, s, d = geometry(f"{DATA}/{pid}_BL_img_{int(float(r['img_id_bl'])):02d}.nii.gz")
            baseline[(pid, r["lesion_id"])] = o + d @ (np.array([float(v) for v in r["cog_bl"].split()]) * s)

by_patient = {}
for r in rows:
    by_patient.setdefault(r["patient"], []).append(r)

cons, init, base, true = [], [], [], []
for pid, group in by_patient.items():
    shifts = [pos(r, "pred") - baseline[(pid, r["lesion_id"])]
              for r in group if r["pred_x"] and (pid, r["lesion_id"]) in baseline]
    if len(shifts) < 3:
        continue
    m = np.median(np.stack(shifts), axis=0)
    for r in group:
        key = (pid, r["lesion_id"])
        if r["pred_x"] or key not in baseline:         # only the lesions with an empty mask
            continue
        cons.append(baseline[key] + m)                 # consensus fallback
        base.append(baseline[key])                     # baseline centre, no shift
        init.append(pos(r, "init"))                    # cog_propagated, may be missing
        true.append(pos(r, "true"))

true = np.stack(true)
print(f"{len(true)} missed lesions in patients with a consensus\n")
for name, est in (("consensus fallback", cons), ("baseline centre", base),
                  ("cog_propagated", init)):
    ok = [i for i, e in enumerate(est) if e is not None]
    e = np.linalg.norm(np.stack([est[i] for i in ok]) - true[ok], axis=1)
    print(f"{name:>20}: n={len(e):>3}  mean {e.mean():6.2f}  median {np.median(e):6.2f}  "
          f"CPM@5 {100*np.mean(e<=5):5.1f}%  CPM@10 {100*np.mean(e<=10):5.1f}%  "
          f"CPM@20 {100*np.mean(e<=20):5.1f}%")
