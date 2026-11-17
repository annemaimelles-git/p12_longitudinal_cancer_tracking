#!/usr/bin/env python
"""
Draw LesionLocator's predicted follow-up masks over the follow-up CT, one PNG per lesion.

Each figure has three panels:
  1. coronal overview of the whole scan, with a line marking where the axial slice is
  2. the full axial slice through the lesion's true centre
  3. a zoom of 80 x 80 mm around the true centre
Markers:  yellow outline = LesionLocator's predicted mask   yellow dot = its centre
          red outline    = expert follow-up mask (if found)  red x      = true centre (cog_fu)
          cyan +         = initial estimate (cog_propagated)

Usage:  python view_lesions.py <patient_id> [results_folder]
Writes  <results_folder>/<patient_id>/figures/lesion_<id>.png
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")                      # no screen on the cluster: draw straight into files
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

DATA = "/gpfs/helios/home/melles/ML_project/work/data/Longitudinal_CT"
PID = sys.argv[1]
PRED = sys.argv[2] if len(sys.argv) > 2 else \
    "/gpfs/helios/home/melles/ML_project/work_v02/results/lesionlocator_val"
WINDOWS = {"Lung": (-1000, 200)}           # HU window per lesion type; default: soft tissue
SOFT_TISSUE = (-160, 240)
ZOOM_MM = 40                               # half-width of the zoom panel


def parse(text):
    """CSV coordinate 'x y z' (voxels) -> array; empty -> None."""
    return np.array([float(v) for v in text.split()]) if text.strip() else None


def outline(ax, mask2d, color):
    """Draw the border of a 2D mask (if it has any voxels)."""
    if mask2d.any():
        ax.contour(mask2d.astype(float), levels=[0.5], colors=color, linewidths=1.2)


def main():
    out_dir = os.path.join(PRED, PID, "figures")
    os.makedirs(out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(DATA, "inputsTr", f"{PID}.csv"))))
    cache = {}                                                     # follow-up scans, read once each

    for r in rows:
        if not r["img_id_bl"].strip():
            continue                                               # newly appeared: nothing tracked
        lid = str(int(float(r["lesion_id"])))
        fu_id = int(float(r["img_id_fu"])) if r["img_id_fu"].strip() else int(float(r["img_id_bl"]))
        pred_file = os.path.join(PRED, PID, f"{PID}_FU_img_{fu_id:02d}_lesion_{lid}.nii.gz")
        if not os.path.exists(pred_file):
            continue
        if fu_id not in cache:
            img = sitk.ReadImage(os.path.join(DATA, "inputsTr", f"{PID}_FU_img_{fu_id:02d}.nii.gz"))
            truth_file = os.path.join(DATA, "targetsTr", f"{PID}_FU_mask_{fu_id:02d}.nii.gz")
            truth = sitk.GetArrayFromImage(sitk.ReadImage(truth_file)) if os.path.exists(truth_file) else None
            cache[fu_id] = (sitk.GetArrayFromImage(img), img.GetSpacing(), truth)
        ct, (sx, sy, sz), truth_all = cache[fu_id]                # ct is indexed [z, y, x]

        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_file)) > 0
        true_v, init_v = parse(r["cog_fu"]), parse(r["cog_propagated"])
        pz, py, px = np.nonzero(pred)
        pred_v = np.array([px.mean(), py.mean(), pz.mean()]) if len(px) else None
        centre = next(v for v in (true_v, pred_v, init_v) if v is not None)   # where to look
        z = int(round(centre[2]))
        truth = (truth_all == int(lid)) if truth_all is not None else None
        lo, hi = WINDOWS.get(r["lesion_type"], SOFT_TISSUE)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        # 1) coronal overview through the centre; the dashed line marks the axial slice
        y = int(round(centre[1]))
        axes[0].imshow(ct[:, y, :], cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect=sz / sx)
        axes[0].axhline(z, color="yellow", ls="--", lw=1)
        axes[0].set_title(f"coronal overview: axial slice {z + 1} of {ct.shape[0]}")
        # 2) and 3) axial slice, full and zoomed
        half = ZOOM_MM / sx
        for ax, zoom in ((axes[1], False), (axes[2], True)):
            ax.imshow(ct[z], cmap="gray", vmin=lo, vmax=hi)
            outline(ax, pred[z], "yellow")
            if truth is not None:
                outline(ax, truth[z], "red")
            for v, style, label in ((true_v, "rx", "truth"), (init_v, "c+", "initial estimate"),
                                    (pred_v, "yo", "LesionLocator centre")):
                if v is not None:
                    dz = (v[2] - z) * sz
                    ax.plot(v[0], v[1], style, ms=9, mew=2, mfc="none",
                            label=f"{label} ({dz:+.0f} mm from slice)")
            if zoom:
                ax.set_xlim(centre[0] - half, centre[0] + half)
                ax.set_ylim(centre[1] + half, centre[1] - half)
                ax.set_title(f"zoom {2 * ZOOM_MM} x {2 * ZOOM_MM} mm")
            else:
                ax.set_title(f"axial slice {z + 1} of {ct.shape[0]}")
                ax.legend(loc="lower left", fontsize=8)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])

        def err(v):
            if v is None or true_v is None:
                return "n/a"
            d = (v - true_v) * np.array([sx, sy, sz])
            return f"{np.linalg.norm(d):.1f} mm"
        fig.suptitle(f"{PID}  lesion {lid}  ({r['lesion_type']}, {r['topology_class']})   "
                     f"predicted voxels: {len(px)}   error: LesionLocator {err(pred_v)}, "
                     f"initial estimate {err(init_v)}")
        fig.tight_layout()
        path = os.path.join(out_dir, f"lesion_{lid}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print("saved", path)


if __name__ == "__main__":
    main()
