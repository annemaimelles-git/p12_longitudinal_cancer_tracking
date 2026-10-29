#!/usr/bin/env python
"""
Affine registration baseline for the Longitudinal-CT dataset.

For each patient: register baseline (fixed) and follow-up (moving) CT,
propagate every baseline lesion centre into the follow-up scan, and
compare with the true follow-up centre.

Usage:
    python register_baseline.py count            # how many patients were found
    python register_baseline.py check <index>    # verify the coordinate convention on one patient
    python register_baseline.py run <index>      # process one patient (used by the Slurm array)
    python register_baseline.py summarize        # combine results and print metrics
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", os.path.expanduser("~/work/data/Longitudinal_CT"))
INPUT_DIR = os.path.join(DATA_DIR, "inputsTr")
OUT_DIR = os.environ.get("OUT_DIR", os.path.expanduser("~/work/results/registration"))

# cog_* columns look like voxel indices ("263.97 202.81 92.55").
# Run `python register_baseline.py check 0` to confirm, and set the axis order it reports.
COG_IN_VOXELS = True
COG_AXIS_ORDER = "xyz"      # "xyz" or "zyx"

N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(N_THREADS)


def patient_ids():
    csvs = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    return sorted(os.path.splitext(os.path.basename(c))[0] for c in csvs)


def ct_path(pid, timepoint, img_id):
    return os.path.join(INPUT_DIR, f"{pid}_{timepoint}_img_{int(img_id):02d}.nii.gz")


def mask_path(pid, img_id):
    return os.path.join(INPUT_DIR, f"{pid}_BL_mask_{int(img_id):02d}.nii.gz")


def parse_xyz(value, order=None):
    """'263.9 202.8 92.5' -> (x, y, z) as floats, or None if empty."""
    if pd.isna(value):
        return None
    parts = str(value).split()
    if len(parts) < 3:
        return None
    p = tuple(float(v) for v in parts[:3])
    return p[::-1] if (order or COG_AXIS_ORDER) == "zyx" else p


def to_mm(img, p):
    if p is None:
        return None
    return img.TransformContinuousIndexToPhysicalPoint(p) if COG_IN_VOXELS else p


def register(bl, fu):
    init = sitk.CenteredTransformInitializer(
        bl, fu, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.05, 42)          # fixed seed -> reproducible
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 200)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(init, inPlace=False)
    return reg.Execute(bl, fu)                         # fixed = baseline, moving = follow-up


def check(index):
    """Look up each baseline centre in the baseline mask to find the right axis order."""
    pid = patient_ids()[index]
    df = pd.read_csv(os.path.join(INPUT_DIR, f"{pid}.csv"))
    masks, hits_id, hits_any, n = {}, {"xyz": 0, "zyx": 0}, {"xyz": 0, "zyx": 0}, 0
    for _, r in df.iterrows():
        if pd.isna(r["cog_bl"]) or pd.isna(r["img_id_bl"]):
            continue
        i = int(r["img_id_bl"])
        if i not in masks:
            masks[i] = sitk.ReadImage(mask_path(pid, i))
        m = masks[i]
        n += 1
        for order in ("xyz", "zyx"):
            idx = [int(round(c)) for c in parse_xyz(r["cog_bl"], order)]
            if all(0 <= idx[k] < m.GetSize()[k] for k in range(3)):
                v = m.GetPixel(idx)
                hits_id[order] += int(v == r["lesion_id"])
                hits_any[order] += int(v > 0)
    m0 = next(iter(masks.values()))
    print(f"Patient {pid}: image size {m0.GetSize()}, spacing {tuple(round(s, 2) for s in m0.GetSpacing())}")
    print(f"{n} baseline lesions checked")
    for order in ("xyz", "zyx"):
        print(f"  order {order}: centre inside its own lesion {hits_id[order]}/{n}, inside any lesion {hits_any[order]}/{n}")


def run(index):
    pid = patient_ids()[index]
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Patient {pid} (index {index}), {N_THREADS} threads")
    df = pd.read_csv(os.path.join(INPUT_DIR, f"{pid}.csv"))

    images, transforms, rows = {}, {}, []

    def img(tp, i):
        if (tp, i) not in images:
            images[(tp, i)] = sitk.ReadImage(ct_path(pid, tp, i), sitk.sitkFloat32)
        return images[(tp, i)]

    for _, r in df.iterrows():
        out = r.to_dict()
        out["patient_id"] = pid
        p_vox = parse_xyz(r["cog_bl"])
        if p_vox is None or pd.isna(r["img_id_bl"]):
            rows.append(out)                           # e.g. newly appearing lesions
            continue

        # Each lesion says which volume it is in (multi-volume studies have _00, _01, ...).
        bl_id = int(r["img_id_bl"])
        fu_id = int(r["img_id_fu"]) if not pd.isna(r["img_id_fu"]) else bl_id
        if not os.path.exists(ct_path(pid, "FU", fu_id)):
            fu_id = 0
        bl, fu = img("BL", bl_id), img("FU", fu_id)

        if (bl_id, fu_id) not in transforms:
            print(f"Registering BL_{bl_id:02d} -> FU_{fu_id:02d}")
            transforms[(bl_id, fu_id)] = register(bl, fu)
            sitk.WriteTransform(transforms[(bl_id, fu_id)],
                                os.path.join(OUT_DIR, f"{pid}_BL{bl_id:02d}_FU{fu_id:02d}.tfm"))
        tx = transforms[(bl_id, fu_id)]

        p_bl = to_mm(bl, p_vox)
        p_pred = tx.TransformPoint(p_bl)
        out.update({"pred_x": p_pred[0], "pred_y": p_pred[1], "pred_z": p_pred[2]})

        p_true = to_mm(fu, parse_xyz(r["cog_fu"]))
        if p_true is not None:
            d = np.subtract(p_pred, p_true)
            out.update({"dx": d[0], "dy": d[1], "dz": d[2],
                        "err_reg": np.linalg.norm(d),
                        "err_identity": np.linalg.norm(np.subtract(p_bl, p_true))})
            p_prop = to_mm(fu, parse_xyz(r["cog_propagated"]))   # propagation provided in the dataset
            if p_prop is not None:
                out["err_dataset_prop"] = np.linalg.norm(np.subtract(p_prop, p_true))
        rows.append(out)

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"{pid}.csv"), index=False)
    print(f"Done: {len(rows)} lesions, {len(transforms)} registration(s)")


def metrics(err, dx=None, dy=None, dz=None):
    m = {"n": len(err), "mean": err.mean(), "median": err.median()}
    if dx is not None:
        m.update({"MEDx": dx.abs().mean(), "MEDy": dy.abs().mean(), "MEDz": dz.abs().mean()})
    for r in (5, 10, 20):
        m[f"CPM@{r}"] = (err <= r).mean() * 100
    return m


def summarize():
    files = [f for f in glob.glob(os.path.join(OUT_DIR, "*.csv")) if not f.endswith("all_results.csv")]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(OUT_DIR, "all_results.csv"), index=False)
    ok = df.dropna(subset=["err_reg"])          # drops lesions without a follow-up centre
    print(f"{len(files)} patients, {len(df)} lesions, {len(ok)} with a follow-up centre\n")
    results = [("Affine registration (ours)", metrics(ok.err_reg, ok.dx, ok.dy, ok.dz)),
               ("No registration", metrics(ok.err_identity))]
    if "err_dataset_prop" in ok:
        p = ok.dropna(subset=["err_dataset_prop"])
        results.append(("cog_propagated from dataset", metrics(p.err_dataset_prop)))
    for name, m in results:
        print(name + ":")
        for k, v in m.items():
            print(f"  {k:8s} {v:.2f}" if isinstance(v, float) else f"  {k:8s} {v}")
    q = ok.err_reg.quantile([0.9, 0.95, 0.99])
    print(f"\nRegistration error percentiles (mm) -> use to size the search region:\n{q.round(1)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "count":
        print(len(patient_ids()))
    elif cmd == "check":
        check(int(sys.argv[2]))
    elif cmd == "run":
        run(int(sys.argv[2]))
    elif cmd == "summarize":
        summarize()
    else:
        print(__doc__)