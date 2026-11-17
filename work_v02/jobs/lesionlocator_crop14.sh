#!/bin/bash
#SBATCH --job-name=ll_crop14
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/helios/home/melles/ML_project/work_v02/logs/%x_%j.out

# LesionLocator tracking on the cropped boxes of one lesion (step 1c test).
set -e
module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate lesionlocator

DATA=/gpfs/helios/home/melles/ML_project/work_v02/crops/2ba100d604_lesion14
CKPT=/gpfs/helios/home/melles/ML_project/work_v02/checkpoints/LesionLocatorCheckpoint
OUT=/gpfs/helios/home/melles/ML_project/work_v02/results/lesionlocator_crop/2ba100d604_lesion14
PID=2ba100d604

echo "Starting at $(date) on $SLURM_JOB_NODELIST"

# same settings as the validation run: expert mask as prompt, all 5 folds (no -f)
LesionLocator_track \
  -bl $DATA/${PID}_BL_img_00.nii.gz \
  -fu $DATA/${PID}_FU_img_00.nii.gz \
  -p  $DATA/${PID}_BL_mask_00.nii.gz \
  -t prev_mask \
  -o $OUT \
  -m $CKPT

echo "Finished at $(date)"
