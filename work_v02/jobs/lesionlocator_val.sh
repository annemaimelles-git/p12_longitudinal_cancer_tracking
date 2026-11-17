#!/bin/bash
#SBATCH --job-name=ll_val
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-29%5
#SBATCH --output=/gpfs/helios/home/melles/ML_project/work_v02/logs/%x_%A_%a.out

# LesionLocator tracking for the 30 validation patients, one array task per patient.
# --array=0-29%5: tasks 0..29 (one per line of val_patients.txt), at most 5 running at once.

set -e
module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate lesionlocator

DATA=/gpfs/helios/home/melles/ML_project/work/data/Longitudinal_CT/inputsTr
CKPT=/gpfs/helios/home/melles/ML_project/work_v02/checkpoints/LesionLocatorCheckpoint
OUT=/gpfs/helios/home/melles/ML_project/work_v02/results/lesionlocator_val
LIST=/gpfs/helios/home/melles/ML_project/work_v02/val_patients.txt

# patient for this task: line (task number + 1) of the list
PID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST)
echo "Task $SLURM_ARRAY_TASK_ID: patient $PID, started $(date) on $SLURM_JOB_NODELIST"

# which baseline scan goes with which follow-up scan, from the patient's CSV, e.g. "00:00 01:01"
PAIRS=$(python - "$DATA/$PID.csv" << 'PYEOF'
import csv, sys
pairs = set()
for r in csv.DictReader(open(sys.argv[1])):
    if not r["img_id_bl"].strip():          # lesion without a baseline (newly appeared): nothing to track
        continue
    bl = int(float(r["img_id_bl"]))
    fu = int(float(r["img_id_fu"])) if r["img_id_fu"].strip() else bl
    pairs.add(f"{bl:02d}:{fu:02d}")
print(" ".join(sorted(pairs)))
PYEOF
)
echo "Scan pairs (baseline:follow-up): $PAIRS"

for PAIR in $PAIRS; do
    BL=${PAIR%%:*}
    FU=${PAIR##*:}
    echo "--- baseline img_$BL -> follow-up img_$FU"
    # -p: the dataset's expert baseline mask; -t prev_mask: use it directly. No -f: all 5 folds.
    LesionLocator_track \
      -bl $DATA/${PID}_BL_img_${BL}.nii.gz \
      -fu $DATA/${PID}_FU_img_${FU}.nii.gz \
      -p  $DATA/${PID}_BL_mask_${BL}.nii.gz \
      -t prev_mask \
      -o $OUT/$PID \
      -m $CKPT
done

echo "Finished $PID at $(date)"
