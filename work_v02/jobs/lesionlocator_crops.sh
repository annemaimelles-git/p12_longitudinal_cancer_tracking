#!/bin/bash
#SBATCH --job-name=ll_crops
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-27%4
#SBATCH --output=/gpfs/helios/home/melles/ML_project/work_v02/logs/%x_%A_%a.out

set -e
module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate lesionlocator
cd /gpfs/helios/home/melles/ML_project/work_v02

CASE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" crop_list.txt)
[ -z "$CASE" ] && { echo "no case for task $SLURM_ARRAY_TASK_ID"; exit 0; }
PID=${CASE%%_lesion*}
echo "Task $SLURM_ARRAY_TASK_ID: $CASE, started $(date)"

LesionLocator_track \
  -bl crops/$CASE/${PID}_BL_img_*.nii.gz \
  -fu crops/$CASE/${PID}_FU_img_*.nii.gz \
  -p  crops/$CASE/${PID}_BL_mask_*.nii.gz \
  -t prev_mask \
  -o results/lesionlocator_crop/$CASE \
  -m checkpoints/LesionLocatorCheckpoint

echo "Finished $CASE at $(date)"
