"""Create cropped baseline/follow-up boxes for a list of lesions (step 1c).
Usage: python make_crops.py PATIENT:LESION [PATIENT:LESION ...]"""
import csv, os, sys
import numpy as np
import SimpleITK as sitk

DATA = "/gpfs/helios/home/melles/ML_project/work/data/Longitudinal_CT/inputsTr"
CROPS = "/gpfs/helios/home/melles/ML_project/work_v02/crops"
HALF_MM = (140, 140, 140)            # box half-size in mm along x, y, z


def box(image, centre):
    """Start voxel and size (in voxels) of the box around 'centre', clipped to the scan."""
    sp, n = image.GetSpacing(), image.GetSize()
    start = [max(0, int(round(c - h / s))) for c, h, s in zip(centre, HALF_MM, sp)]
    end = [min(k, int(round(c + h / s)) + 1) for c, h, s, k in zip(centre, HALF_MM, sp, n)]
    return start, [e - b for b, e in zip(start, end)]


for arg in sys.argv[1:]:
    pid, lid = arg.split(":")
    row = next(r for r in csv.DictReader(open(f"{DATA}/{pid}.csv")) if r["lesion_id"] == lid)
    if not row["cog_propagated"].strip():
        print(f"{pid} lesion {lid}: no cog_propagated, skipped"); continue
    bl_id = int(float(row["img_id_bl"]))
    fu_id = int(float(row["img_id_fu"])) if row["img_id_fu"].strip() else bl_id

    bl  = sitk.ReadImage(f"{DATA}/{pid}_BL_img_{bl_id:02d}.nii.gz")
    msk = sitk.ReadImage(f"{DATA}/{pid}_BL_mask_{bl_id:02d}.nii.gz")
    fu  = sitk.ReadImage(f"{DATA}/{pid}_FU_img_{fu_id:02d}.nii.gz")

    bl_start, bl_size = box(bl, [float(v) for v in row["cog_bl"].split()])
    fu_start, fu_size = box(fu, [float(v) for v in row["cog_propagated"].split()])

    bl_crop = sitk.RegionOfInterest(bl, bl_size, bl_start)
    one = sitk.Cast(sitk.RegionOfInterest(msk, bl_size, bl_start) == int(lid), sitk.sitkUInt8) * int(lid)
    fu_crop = sitk.RegionOfInterest(fu, fu_size, fu_start)

    n = int((sitk.GetArrayFromImage(one) > 0).sum())
    out = f"{CROPS}/{pid}_lesion{lid}"
    os.makedirs(out, exist_ok=True)
    sitk.WriteImage(bl_crop, f"{out}/{pid}_BL_img_{bl_id:02d}.nii.gz")
    sitk.WriteImage(one,     f"{out}/{pid}_BL_mask_{bl_id:02d}.nii.gz")
    sitk.WriteImage(fu_crop, f"{out}/{pid}_FU_img_{fu_id:02d}.nii.gz")
    print(f"{pid} lesion {lid}: BL box {bl_size}, FU box {fu_size}, "
          f"{n} lesion voxels ({n * float(np.prod(bl.GetSpacing())):.0f} mm3, CSV {float(row['volume_bl']):.0f})")
