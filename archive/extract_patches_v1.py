#!/usr/bin/env python
"""
Steps 3 + 4: search region and 2.5D patch extraction for Longitudinal-CT.

Step 3: around the initial estimate in the follow-up scan, a 64 mm cube is
        sampled on a 4 mm grid (17 x 17 x 17 = 4,913 candidate points).
Step 4: at the baseline lesion centre and at every candidate, three orthogonal
        64 x 64 patches (axial, coronal, sagittal) are cut at 1 mm resolution
        and stacked as 3 channels.

Usage:
    python extract_patches.py coverage              # share of true centres inside the cube (reads headers only, fast)
    python extract_patches.py preview <index> [n]   # save preview images for patient <index> (first n lesions, default 3)

The functions baseline_patch(), search_region() and candidate_patches() are
meant to be imported later by the training code.
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", os.path.expanduser("~/work/data/Longitudinal_CT"))
INPUT_DIR = os.path.join(DATA_DIR, "inputsTr")
REG_DIR = os.environ.get("OUT_DIR", os.path.expanduser("~/work/results/registration"))
PATCH_DIR = os.environ.get("PATCH_DIR", os.path.expanduser("~/work/results/patches"))

SPACING = 1.0               # resample to 1 mm isotropic
PATCH = 64                  # patch size in voxels (= 64 mm)
CUBE_HALF = 32              # search cube: +-32 mm  -> 64 mm cube
GRID_STEP = 4               # candidate spacing in mm -> 17 points per axis
HU_WINDOW = (-1000, 1000)   # intensity window before scaling to 0..1
AIR = -1000.0               # padding value outside the scan

N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(N_THREADS)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def patient_ids():
    csvs = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    return sorted(os.path.splitext(os.path.basename(c))[0] for c in csvs)


def ct_path(pid, timepoint, img_id):
    return os.path.join(INPUT_DIR, f"{pid}_{timepoint}_img_{int(img_id):02d}.nii.gz")


def parse_xyz(value):
    if pd.isna(value):
        return None
    parts = str(value).split()
    return np.array([float(v) for v in parts[:3]]) if len(parts) >= 3 else None


def read_header(path):
    """Read only the image geometry (fast, no voxel data)."""
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()
    return r


def vox_to_mm(geom, p):
    """Voxel index (x, y, z) -> physical point in mm. geom: sitk.Image or header reader."""
    if p is None:
        return None
    direction = np.array(geom.GetDirection()).reshape(3, 3)
    return np.array(geom.GetOrigin()) + direction @ (np.array(geom.GetSpacing()) * p)


def lesion_list(pid):
    """Lesions with a known baseline and follow-up centre, excluding DISAPPEARING."""
    df = pd.read_csv(os.path.join(INPUT_DIR, f"{pid}.csv"))
    df = df[df["topology_class"] != "DISAPPEARING"]
    df = df.dropna(subset=["cog_bl", "cog_fu", "img_id_bl", "img_id_fu"])
    affine = None
    reg_csv = os.path.join(REG_DIR, f"{pid}.csv")
    if os.path.exists(reg_csv):
        affine = pd.read_csv(reg_csv).set_index("lesion_id")
    return df, affine


def initial_estimate_mm(row, fu_geom, affine):
    """cog_propagated from the dataset; falls back to our affine registration."""
    p = parse_xyz(row["cog_propagated"])
    if p is not None:
        return vox_to_mm(fu_geom, p), "propagated"
    if affine is not None and row["lesion_id"] in affine.index:
        a = affine.loc[row["lesion_id"]]
        if not pd.isna(a.get("pred_x")):
            return np.array([a["pred_x"], a["pred_y"], a["pred_z"]]), "affine"
    return None, None


# ---------------------------------------------------------------------------
# Step 3 + 4: patches
# ---------------------------------------------------------------------------
def resample_cube(img, center_mm, n):
    """Resample an n x n x n cube at 1 mm around center_mm, axis-aligned.
    Voxel index n // 2 lies exactly on center_mm. Returns a numpy array (z, y, x)."""
    origin = np.asarray(center_mm) - (n // 2) * SPACING
    ref = sitk.Image([n, n, n], sitk.sitkFloat32)
    ref.SetSpacing([SPACING] * 3)
    ref.SetOrigin(origin.tolist())
    ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    out = sitk.Resample(img, ref, sitk.Transform(), sitk.sitkLinear, AIR, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(out)


def normalize(x):
    lo, hi = HU_WINDOW
    return ((np.clip(x, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def orthogonal_patches(arr, c):
    """Three 64 x 64 patches crossing at numpy index c = (z, y, x), stacked as channels."""
    h = PATCH // 2
    z, y, x = c
    axial = arr[z, y - h:y + h, x - h:x + h]
    coronal = arr[z - h:z + h, y, x - h:x + h][::-1]      # flipped so the head is up
    sagittal = arr[z - h:z + h, y - h:y + h, x][::-1]
    return normalize(np.stack([axial, coronal, sagittal]))


def baseline_patch(bl_img, center_mm):
    """3 x 64 x 64 patch at the known lesion centre in the baseline scan."""
    arr = resample_cube(bl_img, center_mm, PATCH)
    return orthogonal_patches(arr, (PATCH // 2,) * 3)


def search_region(fu_img, init_mm):
    """Crop around the initial estimate, large enough that patches at the cube's
    edges are complete: 64 mm cube + 32 mm on each side = 128 mm."""
    return resample_cube(fu_img, init_mm, 2 * (CUBE_HALF + PATCH // 2))


def candidate_offsets():
    """Offsets (x, y, z) in mm of all candidates relative to the initial estimate."""
    o = np.arange(-CUBE_HALF, CUBE_HALF + 1, GRID_STEP)
    gx, gy, gz = np.meshgrid(o, o, o, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)      # (4913, 3)


def candidate_patches(region, offsets):
    """Patches for every candidate: array of shape (N, 3, 64, 64)."""
    c = region.shape[0] // 2
    return np.stack([orthogonal_patches(region, (c + dz, c + dy, c + dx))
                     for dx, dy, dz in offsets.astype(int)])


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def coverage():
    rows = []
    for pid in patient_ids():
        df, affine = lesion_list(pid)
        for _, r in df.iterrows():
            fu = read_header(ct_path(pid, "FU", r["img_id_fu"]))
            init, source = initial_estimate_mm(r, fu, affine)
            if init is None:
                continue
            d = vox_to_mm(fu, parse_xyz(r["cog_fu"])) - init
            rows.append({"source": source, "dx": d[0], "dy": d[1], "dz": d[2],
                         "dist": np.linalg.norm(d), "inside": bool(np.all(np.abs(d) <= CUBE_HALF))})
    t = pd.DataFrame(rows)
    print(f"{len(t)} lesions ({(t.source == 'propagated').sum()} with cog_propagated, "
          f"{(t.source == 'affine').sum()} with affine fallback)")
    print(f"True centre inside the {2 * CUBE_HALF} mm cube: {t.inside.mean() * 100:.1f}%")
    print(f"Outside per axis: x {(t.dx.abs() > CUBE_HALF).mean() * 100:.1f}%, "
          f"y {(t.dy.abs() > CUBE_HALF).mean() * 100:.1f}%, z {(t.dz.abs() > CUBE_HALF).mean() * 100:.1f}%")
    print(f"Candidates per lesion: {len(candidate_offsets())}")


def preview(index, n_lesions=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pid = patient_ids()[index]
    os.makedirs(PATCH_DIR, exist_ok=True)
    df, affine = lesion_list(pid)
    print(f"Patient {pid}: {len(df)} usable lesions, showing up to {n_lesions}")
    images = {}

    def img(tp, i):
        key = (tp, int(i))
        if key not in images:
            images[key] = sitk.ReadImage(ct_path(pid, tp, i), sitk.sitkFloat32)
        return images[key]

    rng = np.random.default_rng(0)
    offsets = candidate_offsets()
    for _, r in df.head(n_lesions).iterrows():
        bl, fu = img("BL", r["img_id_bl"]), img("FU", r["img_id_fu"])
        p_bl = vox_to_mm(bl, parse_xyz(r["cog_bl"]))
        p_true = vox_to_mm(fu, parse_xyz(r["cog_fu"]))
        init, source = initial_estimate_mm(r, fu, affine)
        if init is None:
            print(f"  lesion {r['lesion_id']}: no initial estimate, skipped")
            continue

        t0 = time.time()
        region = search_region(fu, init)
        cands = candidate_patches(region, offsets)
        dt = time.time() - t0

        rows = [("Baseline: known centre", baseline_patch(bl, p_bl)),
                ("Follow-up: TRUE centre", baseline_patch(fu, p_true)),
                (f"Follow-up: initial estimate ({source})", cands[len(offsets) // 2]),
                ("Follow-up: random candidate", cands[rng.integers(len(cands))])]
        fig, axes = plt.subplots(4, 3, figsize=(7.5, 10))
        for i, (title, patch) in enumerate(rows):
            for j, plane in enumerate(["axial", "coronal", "sagittal"]):
                axes[i, j].imshow(patch[j], cmap="gray", vmin=0, vmax=1)
                axes[i, j].axhline(PATCH / 2, color="r", lw=0.4)
                axes[i, j].axvline(PATCH / 2, color="r", lw=0.4)
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                if i == 0:
                    axes[i, j].set_title(plane)
            axes[i, 0].set_ylabel(title, fontsize=8)
        dist = np.linalg.norm(p_true - init)
        fig.suptitle(f"{pid} lesion {r['lesion_id']} ({r['lesion_type']}), "
                     f"initial estimate {dist:.1f} mm from true centre", fontsize=9)
        fig.tight_layout()
        out = os.path.join(PATCH_DIR, f"preview_{pid}_lesion{r['lesion_id']}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  lesion {r['lesion_id']}: {dist:.1f} mm off, candidates {cands.shape} "
              f"in {dt:.1f} s -> {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "coverage":
        coverage()
    elif cmd == "preview":
        preview(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        print(__doc__)