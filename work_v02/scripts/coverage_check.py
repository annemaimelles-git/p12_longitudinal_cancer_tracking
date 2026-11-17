"""How much of the body do the baseline and follow-up scans have in common (along z, head-feet)?
The shift between the two scans is estimated from the lesion pairs cog_bl -> cog_propagated."""
import csv
import numpy as np
import SimpleITK as sitk

D = "/gpfs/helios/home/melles/ML_project/work/data/Longitudinal_CT/inputsTr"
PATIENTS = [p.strip() for p in open("/gpfs/helios/home/melles/ML_project/work_v02/val_patients.txt") if p.strip()]
FAILURES = {"2ba100d604", "3988c7f88e", "0f49c89d1e", "e2c0be2456"}   # patients with the big errors

def header(path):
    r = sitk.ImageFileReader(); r.SetFileName(path); r.ReadImageInformation()
    return np.array(r.GetOrigin()), np.array(r.GetSpacing()), np.array(r.GetDirection()).reshape(3, 3), r.GetSize()

def to_mm(idx, h):
    origin, spacing, direction, _ = h
    return origin + direction @ (np.asarray(idx, float) * spacing)

def z_range(h):
    """z (mm) of the first and last slice."""
    z0, z1 = to_mm([0, 0, 0], h)[2], to_mm([0, 0, h[3][2] - 1], h)[2]
    return min(z0, z1), max(z0, z1)

print(f"{'patient':>11} {'BL length':>10} {'FU length':>10} {'overlap':>9} {'shared':>7}")
for pid in PATIENTS:
    bl, fu = header(f"{D}/{pid}_BL_img_00.nii.gz"), header(f"{D}/{pid}_FU_img_00.nii.gz")
    shifts = []
    for r in csv.DictReader(open(f"{D}/{pid}.csv")):
        if r["cog_bl"].strip() and r["cog_propagated"].strip() and r["img_id_bl"].strip() \
                and int(float(r["img_id_bl"])) == 0:
            z_bl = to_mm([float(v) for v in r["cog_bl"].split()], bl)[2]
            z_fu = to_mm([float(v) for v in r["cog_propagated"].split()], fu)[2]
            shifts.append(z_fu - z_bl)
    if not shifts:
        print(f"{pid:>11}  no lesion pairs to estimate the shift"); continue
    a, b = z_range(bl); c, d = z_range(fu); s = np.median(shifts)
    overlap = max(0.0, min(b + s, d) - max(a + s, c))
    shared = overlap / min(b - a, d - c)
    flag = "  <- had big errors" if pid in FAILURES else ""
    print(f"{pid:>11} {b - a:>8.0f}mm {d - c:>8.0f}mm {overlap:>7.0f}mm {shared:>6.0%}{flag}")
