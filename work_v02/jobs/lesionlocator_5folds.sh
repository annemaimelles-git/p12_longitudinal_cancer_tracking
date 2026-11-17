#!/bin/bash
#SBATCH --job-name=ll_5folds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/helios/home/melles/ML_project/work_v02/logs/%x_%j.out

# LesionLocator tracking for one patient, all 5 folds (the README default).
# Inputs:  baseline CT, follow-up CT, and the dataset's expert baseline mask.
# Output:  one predicted follow-up mask per baseline lesion (..._lesion_N.nii.gz).

set -e
module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate lesionlocator

DATA=/gpfs/helios/home/melles/ML_project/work/data/Longitudinal_CT/inputsTr
CKPT=/gpfs/helios/home/melles/ML_project/work_v02/checkpoints/LesionLocatorCheckpoint
OUT=/gpfs/helios/home/melles/ML_project/work_v02/results/lesionlocator_5folds
PID=5f93f98352

echo "Starting at $(date) on $SLURM_JOB_NODELIST"
nvidia-smi --query-gpu=name,memory.total --format=csv

# -p passes the dataset's expert baseline mask; -t prev_mask tells LesionLocator to use it directly.
# No -f option, so all 5 folds are used and their predictions combined.
LesionLocator_track \
  -bl $DATA/${PID}_BL_img_00.nii.gz \
  -fu $DATA/${PID}_FU_img_00.nii.gz \
  -p  $DATA/${PID}_BL_mask_00.nii.gz \
  -t prev_mask \
  -o $OUT/$PID \
  -m $CKPT

echo "Finished at $(date)"
