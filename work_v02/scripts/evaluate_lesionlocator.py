#!/usr/bin/env python
"""
Evaluate LesionLocator's tracked follow-up masks with the metrics agreed in the project.

For every baseline lesion of the validation patients:
  true centre       = cog_fu          (patient CSV; centre of gravity of the expert follow-up mask)
  initial estimate  = cog_propagated  (patient CSV; the registration-based guess)
  LesionLocator     = centre of gravity of the predicted mask <PID>_FU_img_XX_lesion_<id>.nii.gz

All positions are voxel indices (x, y, z); they are converted to millimetres with the follow-up
scan's geometry and compared with:
  mean, median    3D distance between predicted and true centre (mm)
  MEDx/MEDy/MEDz  median absolute error along each axis (mm)
  CPM@5/10/20     % of lesions whose predicted centre is at most 5 / 10 / 20 mm from the truth

Usage:
  python evaluate_lesionlocator.py [results_folder] [patient_list]
  defaults: work_v02/results/lesionlocator_val  and  work_v02/lists/val_patients.txt
Writes per_lesion.csv, metrics.csv and summary.txt into <results_folder>/evaluation/.

per_lesion.csv also holds the signals used by the fallback rules and the meta-model:
  dist_to_init_mm  how far LesionLocator moved from the initial estimate
  duplicate_mm     distance to the nearest OTHER lesion's prediction in the same patient
  volume_ratio     predicted volume / baseline volume (volume_bl from the CSV)
  label            1 if LesionLocator is closer to the truth than the initial estimate, else 0
"""
import csv
import os
import sys

import numpy as np
import SimpleITK as sitk

DATA = "/gpfs/helios/home/melles/ML_project/work/data/Longitudinal_CT/inputsTr"
PRED = sys.argv[1] if len(sys.argv) > 1 else \
    "/gpfs/helios/home/melles/ML_project/work_v02/results/lesionlocator_val"
PATIENTS = sys.argv[2] if len(sys.argv) > 2 else \
    "/gpfs/helios/home/melles/ML_project/work_v02/lists/val_patients.txt"
OUT = os.path.join(PRED, "evaluation")
CPM_MM = (5, 10, 20)


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------
def geometry(path):
    """Read only the header of a scan (origin, spacing, direction), not the image data."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    return (np.array(reader.GetOrigin()), np.array(reader.GetSpacing()),
            np.array(reader.GetDirection()).reshape(3, 3))


def to_mm(xyz, geo):
    """Voxel position (x, y, z), may be fractional -> physical position in mm.
    Same formula as SimpleITK's TransformContinuousIndexToPhysicalPoint."""
    origin, spacing, direction = geo
    return origin + direction @ (np.asarray(xyz, dtype=float) * spacing)


def parse(text):
    """CSV coordinate '112.2 342.3 144.3' -> array; empty field -> None."""
    return np.array([float(v) for v in text.split()]) if text.strip() else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(pred, true):
    """Project metrics from (N, 3) arrays of predicted and true positions in mm."""
    delta = np.asarray(pred) - np.asarray(true)
    err = np.linalg.norm(delta, axis=1)
    m = {"lesions": len(err), "mean": err.mean(), "median": np.median(err)}
    for i, axis in enumerate("xyz"):
        m[f"MED{axis}"] = np.median(np.abs(delta[:, i]))
    for k in CPM_MM:
        m[f"CPM@{k}"] = 100 * np.mean(err <= k)
    return m


def table(columns):
    """{method: metrics dict} -> aligned text table, one row per metric."""
    names = list(columns)
    keys = list(columns[names[0]])
    lines = [" " * 10 + "".join(f"{n:>18s}" for n in names)]
    for k in keys:
        cells = "".join(f"{columns[n][k]:>18d}" if k == "lesions" else f"{columns[n][k]:>18.2f}"
                        for n in names)
        lines.append(f"{k:<10s}" + cells)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Collect one row per baseline lesion
# ---------------------------------------------------------------------------
def collect():
    rows, missing_patients = [], []
    patients = [p.strip() for p in open(PATIENTS) if p.strip()]
    for pid in patients:
        pdir = os.path.join(PRED, pid)
        if not os.path.isdir(pdir):
            missing_patients.append(pid)
            continue
        for r in csv.DictReader(open(os.path.join(DATA, f"{pid}.csv"))):
            if not r["img_id_bl"].strip():           # newly appeared: no baseline lesion to track
                continue
            bl_id = int(float(r["img_id_bl"]))
            fu_id = int(float(r["img_id_fu"])) if r["img_id_fu"].strip() else bl_id
            lid = str(int(float(r["lesion_id"])))
            geo = geometry(os.path.join(DATA, f"{pid}_FU_img_{fu_id:02d}.nii.gz"))
            voxel_mm3 = float(np.prod(geo[1]))                     # volume of one follow-up voxel
            true, init = parse(r["cog_fu"]), parse(r["cog_propagated"])

            pred_file = os.path.join(pdir, f"{pid}_FU_img_{fu_id:02d}_lesion_{lid}.nii.gz")
            n_vox, pred_mm = None, None
            if os.path.exists(pred_file):
                mask = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))   # numpy order (z, y, x)
                z, y, x = np.nonzero(mask)
                n_vox = len(x)
                if n_vox:
                    pred_mm = to_mm([x.mean(), y.mean(), z.mean()], geo)  # centre of gravity

            rows.append({"patient": pid, "lesion_id": lid, "lesion_type": r["lesion_type"],
                         "topology": r["topology_class"], "pred_voxels": n_vox,
                         "volume_bl": float(r["volume_bl"]) if r.get("volume_bl", "").strip() else None,
                         "voxel_mm3": voxel_mm3,
                         "true": to_mm(true, geo) if true is not None else None,
                         "init": to_mm(init, geo) if init is not None else None,
                         "pred": pred_mm})
    return rows, missing_patients


def add_signals(rows):
    """Add the signals the fallback rules and the meta-model use."""
    by_patient = {}
    for r in rows:
        by_patient.setdefault(r["patient"], []).append(r)

    for pid, group in by_patient.items():
        for r in group:
            # how far LesionLocator moved away from the registration's guess
            r["dist_to_init_mm"] = (float(np.linalg.norm(r["pred"] - r["init"]))
                                    if r["pred"] is not None and r["init"] is not None else None)
            # distance to the nearest OTHER lesion's prediction of the same patient
            others = [float(np.linalg.norm(r["pred"] - o["pred"])) for o in group
                      if o is not r and o["pred"] is not None and r["pred"] is not None]
            r["duplicate_mm"] = min(others) if others else None
            # predicted volume compared with the baseline volume
            r["volume_ratio"] = (r["pred_voxels"] * r["voxel_mm3"] / r["volume_bl"]
                                 if r["pred_voxels"] and r.get("volume_bl") else None)
            # label: was LesionLocator closer to the truth than the initial estimate?
            if r["true"] is not None and r["init"] is not None:
                e_init = float(np.linalg.norm(r["init"] - r["true"]))
                e_ll = float(np.linalg.norm(r["pred"] - r["true"])) if r["pred"] is not None else None
                r["label"] = 1 if (e_ll is not None and e_ll < e_init) else 0
            else:
                r["label"] = None
    return rows


def write_per_lesion(rows, path):
    def err(a, b):
        return f"{np.linalg.norm(a - b):.2f}" if a is not None and b is not None else ""

    def xyz(p):
        return [f"{v:.2f}" for v in p] if p is not None else ["", "", ""]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "lesion_id", "lesion_type", "topology", "pred_voxels",
                    "true_x", "true_y", "true_z", "init_x", "init_y", "init_z",
                    "pred_x", "pred_y", "pred_z", "err_initial_mm", "err_lesionlocator_mm",
                    "dist_to_init_mm", "duplicate_mm", "volume_ratio", "label"])
        for r in rows:
            w.writerow([r["patient"], r["lesion_id"], r["lesion_type"], r["topology"],
                        "" if r["pred_voxels"] is None else r["pred_voxels"],
                        *xyz(r["true"]), *xyz(r["init"]), *xyz(r["pred"]),
                        err(r["init"], r["true"]), err(r["pred"], r["true"]),
                        *[f"{r[k]:.2f}" if r.get(k) is not None else ""
                          for k in ("dist_to_init_mm", "duplicate_mm", "volume_ratio")],
                        "" if r.get("label") is None else r["label"]])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rows, missing_patients = collect()
    add_signals(rows)
    os.makedirs(OUT, exist_ok=True)
    write_per_lesion(rows, os.path.join(OUT, "per_lesion.csv"))

    # lesions that have a true follow-up centre and an initial estimate
    scored = [r for r in rows if r["true"] is not None and r["init"] is not None]
    found = [r for r in scored if r["pred"] is not None]
    no_truth = [r for r in rows if r["true"] is None]
    lines = [f"Patients evaluated: {len({r['patient'] for r in rows})}"
             + (f" (no predictions yet for {len(missing_patients)}: {', '.join(missing_patients)})"
                if missing_patients else ""),
             f"Baseline lesions: {len(rows)}; with a true follow-up centre: {len(scored)}; "
             f"LesionLocator produced a non-empty mask for {len(found)} of them.",
             ""]

    if scored:
        true = np.stack([r["true"] for r in scored])
        init = np.stack([r["init"] for r in scored])
        # where LesionLocator's mask is empty or missing, fall back to the initial estimate
        ll_fb = np.stack([r["pred"] if r["pred"] is not None else r["init"] for r in scored])
        all_cols = {"initial estimate": metrics(init, true),
                    "LL + fallback": metrics(ll_fb, true)}
        lines += ["1) All lesions with a true follow-up centre (mm; CPM = % of lesions):",
                  "   'LL + fallback' uses the initial estimate where LesionLocator's mask is empty.",
                  table(all_cols), ""]
        write_metrics(all_cols, os.path.join(OUT, "metrics.csv"))

    if found:
        true_f = np.stack([r["true"] for r in found])
        found_cols = {"initial estimate": metrics(np.stack([r["init"] for r in found]), true_f),
                      "LesionLocator": metrics(np.stack([r["pred"] for r in found]), true_f)}
        lines += ["2) Only lesions where LesionLocator produced a mask (same lesions for both columns):",
                  table(found_cols), ""]
        write_metrics(found_cols, os.path.join(OUT, "metrics_found_only.csv"))

    if no_truth:
        empty = sum(1 for r in no_truth if r["pred_voxels"] == 0)
        lines.append(f"3) Lesions without a follow-up centre (e.g. DISAPPEARING): {len(no_truth)}; "
                     f"LesionLocator's mask was empty (= correctly 'gone') for {empty} of them.")

    lines += ["",
              "Reference from the old pipeline, same 30 validation patients (initial estimate):",
              "  194 lesions, median 10.29, MEDx 2.09, MEDy 2.72, MEDz 4.39, CPM@5 38.14, "
              "CPM@10 49.48, CPM@20 68.04",
              "  -> once all 30 patients are done, table 1's 'initial estimate' column should match this."]
    text = "\n".join(lines)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nFiles: {OUT}/per_lesion.csv, metrics.csv, summary.txt")


def write_metrics(columns, path):
    names = list(columns)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", *names])
        for k in columns[names[0]]:
            w.writerow([k, *[round(float(columns[n][k]), 2) for n in names]])


if __name__ == "__main__":
    main()