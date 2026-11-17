"""How far is the initial estimate (cog_propagated) from the truth (cog_fu), per axis, on TRAINING patients?
The crop half-size must cover this distance, plus a margin for the lesion and the registration's context."""
import csv, json, math
import numpy as np
import SimpleITK as sitk

ROOT = "/gpfs/helios/home/melles/ML_project/work/data/Longitudinal_CT"
train = json.load(open(f"{ROOT}/data_split.json"))["train"]

def header(path):
    r = sitk.ImageFileReader(); r.SetFileName(path); r.ReadImageInformation()
    return np.array(r.GetOrigin()), np.array(r.GetSpacing()), np.array(r.GetDirection()).reshape(3, 3)

diffs = []
for pid in train:
    for r in csv.DictReader(open(f"{ROOT}/inputsTr/{pid}.csv")):
        if not (r["cog_fu"].strip() and r["cog_propagated"].strip() and r["img_id_fu"].strip()):
            continue
        origin, spacing, direction = header(f"{ROOT}/inputsTr/{pid}_FU_img_{int(float(r['img_id_fu'])):02d}.nii.gz")
        to_mm = lambda t: origin + direction @ (np.array([float(v) for v in t.split()]) * spacing)
        diffs.append(np.abs(to_mm(r["cog_propagated"]) - to_mm(r["cog_fu"])))
diffs = np.array(diffs)
print(f"{len(diffs)} training lesions; |initial estimate - truth| per axis (mm):")
for q in (50, 90, 95, 99):
    print(f"  {q}th percentile:  x {np.percentile(diffs[:, 0], q):6.1f}   y {np.percentile(diffs[:, 1], q):6.1f}   z {np.percentile(diffs[:, 2], q):6.1f}")
print(f"  maximum:          x {diffs[:, 0].max():6.1f}   y {diffs[:, 1].max():6.1f}   z {diffs[:, 2].max():6.1f}")
p99 = np.percentile(diffs, 99, axis=0).max()
print(f"\nSuggested crop half-size: 99th percentile ({p99:.0f} mm) + 30 mm margin, rounded up "
      f"= {math.ceil((p99 + 30) / 10) * 10} mm  (box = twice that per axis)")
