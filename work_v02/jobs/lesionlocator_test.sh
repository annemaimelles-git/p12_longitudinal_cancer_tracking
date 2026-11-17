#!/bin/bash
#SBATCH --job-name=ll_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/helios/home/melles/ML_project/work_v02/logs/%x_%j.out

set -e
module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate lesionlocator

DATA=/gpfs/helios/home/melles/ML_project/work/data/Longitudinal_CT/inputsTr
PID=5f93f98352

echo "Starting at $(date) on $SLURM_JOB_NODELIST"
nvidia-smi --query-gpu=name,memory.total --format=csv

LesionLocator_track \
  -bl $DATA/${PID}_BL_img_00.nii.gz \
  -fu $DATA/${PID}_FU_img_00.nii.gz \
  -p  $DATA/${PID}_BL_mask_00.nii.gz \
  -t prev_mask \
  -f 0 \
  -o /gpfs/helios/home/melles/ML_project/work_v02/results/lesionlocator_test/$PID \
  -m /gpfs/helios/home/melles/ML_project/work_v02/checkpoints/LesionLocatorCheckpoint

echo "Finished at $(date)"
