#!/bin/bash
#SBATCH --job-name=ll_seg_box
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200-180g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/helios/home/melles/ML_project/work_v02/logs/%x_%j.out

# Segmentation mode on one patient: point prompts at the consensus position,
# for the lesions that tracking missed.
set -e
module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate lesionlocator
cd /gpfs/helios/home/melles/ML_project/work_v02

echo "Starting at $(date) on $SLURM_JOB_NODELIST"
LesionLocator_segment \
  -i images_one \
  -p prompts/box_one \
  -t box \
  -o results/lesionlocator_segment_box \
  -m checkpoints/LesionLocatorCheckpoint \
  --disable_progress_bar
echo "Finished at $(date)"
