# Cluster guide: data and scripts

This guide describes what is stored on the UT HPC **Rocket** cluster for the Longitudinal Cancer Tracking project, how the data is organised, and what each script does, reads and writes.

## 1. Folder overview

All paths are in the home directory of user `melles` (`~` = `/gpfs/helios/home/melles`).

```
~/
├── ondemand/                           created by the Open OnDemand web portal (not used by the project)
├── venvs/lct/                          Python virtual environment used by the project (Python 3.14)
└── work/
    ├── data/
    │   ├── Longitudinal_CT/            the dataset
    │   │   ├── LongitudinalCT_demographics.csv
    │   │   ├── data_split.json         fixed train / validation / test split
    │   │   ├── inputsTr/               1,972 files: CT scans, baseline masks, lesion tables, JSON centres
    │   │   └── targetsTr/              336 files: follow-up masks (ground truth)
    │   ├── extract_patches_v3.py       steps 1–4: initial estimate, search region, 2.5D patches (current)
    │   ├── model.py                    steps 5–6: ResNet-18 encoder and localisation (not yet tested)
    │   ├── model_slurm                 Slurm job that runs model.py on a GPU
    │   ├── logs/                       Slurm logs of model_slurm (model_<jobid>.out)
    │   ├── show.py                     own helper script
    │   ├── preview.png                 own test image
    │   └── archive/                    scripts that are no longer run
    │       ├── register_baseline.py    affine registration baseline (step 2, first week)
    │       ├── register.slurm          Slurm job array that ran it for all patients
    │       ├── extract_patches_v1.py   old version (wide intensity window)
    │       ├── extract_patches_v2.py   old version (1 mm plane offset)
    │       └── logs/                   300 Slurm logs of the registration run
    └── results/
        ├── registration/               659 files, KEEP: affine results, still used as fallback (section 4)
        └── patches/                    preview and heatmap images
```

**Still to do:** test `model.py` on a GPU.

## 2. The dataset

Source: *A longitudinal whole-body CT dataset with manually annotated tumor lesions*, Scientific Data (2026). 300 patients, each with a baseline (BL) and a follow-up (FU) whole-body CT.

### 2.1 Files in `inputsTr/`

Every patient has a 10-character ID (e.g. `006f52e910`). Per patient:

| File | Content |
|---|---|
| `<id>.csv` | One row per lesion: centres, lesion type, how the lesion changed (see 2.3) |
| `<id>_BL_img_00.nii.gz` | Baseline CT scan |
| `<id>_BL_mask_00.nii.gz` | Baseline lesion mask; each voxel holds the lesion ID (0 = background) |
| `<id>_FU_img_00.nii.gz` | Follow-up CT scan |
| `<id>_BL_00.json`, `<id>_FU_00.json` | Lesion centres in JSON format (pixel coordinates) |

Some studies consist of several volumes, numbered `_00`, `_01`, ... (e.g. patient `01161aaa0b`). The CSV columns `img_id_bl` and `img_id_fu` say which volume each lesion is in.

### 2.2 Files in `targetsTr/`

Follow-up lesion masks (`<id>_FU_mask_00.nii.gz`, ...). These are **ground truth**: used only for training labels, evaluation and visual checks, never as model input.

### 2.3 Lesion table (`<id>.csv`)

| Column | Meaning |
|---|---|
| `lesion_id` | Lesion number within the patient (matches the mask values) |
| `cog_bl` | Lesion centre in the baseline scan |
| `cog_fu` | Lesion centre in the follow-up scan (**ground truth**) |
| `cog_propagated` | Baseline centre propagated into the follow-up scan by the dataset authors. Our initial estimate; **empty for 101 lesions** |
| `cog_backpropagated` | Propagation in the other direction (not used) |
| `img_id_bl`, `img_id_fu` | Which volume (`_00`, `_01`, ...) the lesion is in |
| `lesion_type` | Location, e.g. Lymph node, Lung, Others |
| `topology_class` | UNCHANGED, DISAPPEARING, MERGING or NEWLYAPPEARING |
| `merged_into` | For merging lesions: the lesion it merged into |
| `volume_bl`, `volume_fu` | Lesion volume at baseline and follow-up |
| `target_lesion`, `use_for_challenge`, `linking_unclear` | Flags; `linking_unclear = True` means the radiologists were unsure of the match |

**Coordinates:** all `cog_*` values are **voxel indices in x y z order** (verified: the baseline centre lies inside its own lesion in the baseline mask). They are converted to millimetres with the scan's header (origin, spacing, direction).

### 2.4 Image properties

- Native resolution: about 0.78 × 0.78 mm in-plane, **3 mm slice thickness**, e.g. 512 × 512 × 245 voxels.
- The scans are not resampled by the dataset authors; our patch extraction resamples small regions to 1 mm.
- Baseline and follow-up scans do **not** share a coordinate origin, so raw coordinates cannot be compared across scans without registration.

### 2.5 Rules we follow

- DISAPPEARING lesions are excluded (no follow-up location).
- For MERGING lesions the target is the centre of the merged follow-up lesion (`cog_fu`).
- Final results are reported on the 30 **test** patients from `data_split.json`.

## 3. Environment

```bash
source ~/venvs/lct/bin/activate     # activate the environment (needed in every new terminal)
```

Installed packages: SimpleITK, pandas, numpy, matplotlib (torch and torchvision still to be checked).

**Login node vs. compute node:** the login node (`login1`, `login2`) is for light work: editing, `ls`, `pip install`, submitting jobs, reading headers. Anything that loads full CT scans or uses a GPU runs on a compute node:

```bash
# interactive CPU session (prompt changes to e.g. ares5 / artemis8)
srun --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash

# interactive GPU session: the GPU type must be named
srun --partition=gpu --gres=gpu:a100-40g:1 --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash
```

GPU types on the `gpu` partition (`sinfo -o "%P %G %l" | grep -i gpu`): `a100-40g`, `a100-80g` (recommended), `h200-141g`, `b200-180g` (very powerful, often busy), `tesla` (older; may not be supported by recent PyTorch). Maximum job time 8 days. If a job waits long (`squeue -u $USER`, reason `Resources`), try another type.

After `srun`, activate the environment again and `cd` to the script folder. Type `exit` to leave the compute node, and cancel forgotten sessions with `scancel <jobid>`.

**Paths** can be changed with environment variables. Defaults:

| Variable | Default | Used for |
|---|---|---|
| `DATA_DIR` | `~/work/data/Longitudinal_CT` | the dataset |
| `OUT_DIR` | `~/work/results/registration` | registration results (written by the archived registration, read by `extract_patches_v3.py`) |
| `PATCH_DIR` | `~/work/results/patches` | preview and heatmap images |

Patients are referred to by **index** 0–299: their position in the alphabetically sorted list of patient IDs (0 = `006f52e910`, 1 = `01161aaa0b`, ...).

## 4. Initial estimate and the registration results

The search region is centred on an initial estimate of the lesion position in the follow-up scan. `initial_estimate_mm()` in `extract_patches_v3.py` decides per lesion:

1. **`cog_propagated`** from the dataset CSV, if it has a value: **2,571 lesions (96%)**.
2. Otherwise **our affine registration result** from `~/work/results/registration/<id>.csv` (columns `pred_x/y/z`, in mm): **101 lesions (4%)**, whose `cog_propagated` is empty in the dataset.
3. If neither exists, the lesion is skipped.

So the registration **scripts** are archived, but their **results are still read**. Do not delete or move `~/work/results/registration/`. To change the fallback, edit `initial_estimate_mm()`.

## 5. Scripts

### 5.1 `extract_patches_v3.py` — steps 1–4 (current)

For each lesion:

1. **Initial estimate:** see section 4.
2. **Search region:** a 64 mm cube around the initial estimate. A 129 mm region (cube + 32 mm margin for complete patches) is resampled to 1 mm; outside the scan it is padded with air (−1000 HU). Whole scans are never resampled.
3. **Candidates:** 17 × 17 × 17 = 4,913 grid points, 4 mm apart.
4. **Patches:** at each candidate and at the baseline centre, the axial, coronal and sagittal 64 × 64 slices crossing exactly at that point, clipped to −200…300 HU, scaled to 0–1 and stacked as 3 channels (3 × 64 × 64). Coronal and sagittal are flipped so the head is up.

DISAPPEARING lesions and lesions without a follow-up centre are skipped (`lesion_list`). All 4,913 candidate patches of one lesion take about 0.5 s.

| Command | What it does | Where |
|---|---|---|
| `python extract_patches_v3.py coverage` | Share of true centres inside the cube, plus mean/median, MEDx/y/z and CPM of the initial estimate, separately for `cog_propagated` and the affine fallback (reads headers only) | login node |
| `python extract_patches_v3.py preview 1` | Check images for patient 1 (first 3 lesions) | compute node |
| `python extract_patches_v3.py preview 5 10` | Patient 5, up to 10 lesions | compute node |

**Output:** `~/work/results/patches/preview_<id>_lesion<n>.png`. Rows: baseline patch at the known centre, follow-up patch at the true centre, at the initial estimate, and at a random candidate. Red outline = this lesion (from the masks, for checking only), yellow = other lesions, cyan = patch centre. In rows 1–2 the red outline should surround the centre.

**Results:** the true centre lies inside the 64 mm cube for 92.7% of lesions; most misses are in z (5.5%). The existing preview images were made with v2; the 1 mm offset fixed in v3 does not change them noticeably.

### 5.2 `model.py` — steps 5–6 (written, not yet tested, needs a GPU)

- **Encoder:** ImageNet-pretrained ResNet-18 from torchvision. The first max-pool is removed (keeps a 4 × 4 feature map for 64 × 64 input), the classifier is replaced, giving a 512-d feature vector, followed by a projection head to 128-d for training. Vectors are normalised to length 1.
- **Localisation:** cosine similarity between the baseline patch vector and all 4,913 candidate vectors gives a similarity heatmap; the best point is refined on a 1 mm grid (±4 mm, 729 points).
- Imports `extract_patches_v3.py` (or `extract_patches.py`) from the same folder.

| Command | What it does | Where |
|---|---|---|
| `python model.py info` | Parameter count, GPU name, speed for 4,913 patches | GPU node |
| `python model.py test 1` | Untrained (ImageNet-only) model on patient 1: errors and heatmaps | GPU node |

Instead of an interactive GPU session, the same checks can be submitted as a job with **`model.slurm`** (A100-40G, 4 CPUs, 16 GB, 30 min). It prints whether PyTorch sees the GPU, then runs `model.py info` and `model.py test 1`:

```bash
cd ~/work/data
mkdir -p logs                # needed once; the job writes its log here
sbatch model.slurm
squeue -u $USER              # PD = waiting for a GPU, R = running, gone = finished
cat logs/model_<jobid>.out   # the output
```

**Output:** `~/work/results/patches/heatmap_<id>_lesion<n>.png`. Bright = high similarity; red ✕ = true centre, white ✕ = prediction, cyan + = initial estimate.

Before the first run, install PyTorch and download the ResNet-18 weights on the login node (compute nodes have no internet):

```bash
pip install torch torchvision
python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.DEFAULT)"
```

Training (step 7) is not implemented yet.

### 5.3 `archive/` — no longer run

| File | What it was for |
|---|---|
| `register_baseline.py` | Step 2 in the first week: affine registration (SimpleITK, mutual information, 3-level multi-resolution) of each BL/FU volume pair, propagation of the lesion centres, error metrics. Masks are not used |
| `register.slurm` | Job array for all 300 patients (`--array=0-299%40`, 4 CPUs, 16 GB, 1 h per task); finished in under an hour |
| `logs/` | The 300 Slurm logs of that run (`reg_2961687_<task>.out`) |
| `extract_patches_v1.py`, `extract_patches_v2.py` | Earlier versions of the patch script |

**Registration output** (kept in `~/work/results/registration/`, see section 4):

- `<id>.csv` per patient: all original CSV columns plus `pred_x`, `pred_y`, `pred_z` (propagated centre, mm), `dx`, `dy`, `dz` (error per axis), `err_reg` (affine error, mm), `err_identity` (error without registration, not meaningful) and `err_dataset_prop` (error of `cog_propagated`, mm).
- `<id>_BLxx_FUyy.tfm`: the registration transforms (358 in total, one per volume pair).
- `all_results.csv`: all patients combined.

**Registration results** (2,672 lesions): affine median 11.75 mm, CPM@10 42.9%; `cog_propagated` median 4.70 mm, CPM@10 69.8%.

To rerun it (only needed if the results are lost), work inside the archive folder, because `register.slurm` expects the script and `logs/` next to it:

```bash
cd ~/work/data/archive
sbatch register.slurm
python register_baseline.py summarize
```

## 6. Typical workflows

**Check the patches of a patient**

```bash
srun --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash
source ~/venvs/lct/bin/activate
cd ~/work/data
python extract_patches_v3.py preview 3
exit
```

Then open the PNGs in VS Code (Remote-SSH, folder `~/work/results/patches`).

**Test the model on a GPU (interactive)**

```bash
srun --partition=gpu --gres=gpu:a100-40g:1 --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash
source ~/venvs/lct/bin/activate
cd ~/work/data
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python model.py info
python model.py test 1
exit
```

**Test the model on a GPU (as a job)**

```bash
cd ~/work/data
sbatch model.slurm
cat logs/model_*.out
```