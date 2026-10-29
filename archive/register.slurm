#!/bin/bash
#SBATCH --job-name=reg-baseline
#SBATCH --partition=main            # check partition names with: sinfo
#SBATCH --array=0-299%40            # one task per patient, max 40 at once (set to count-1)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/reg_%A_%a.out

source ~/venvs/lct/bin/activate

export DATA_DIR=$HOME/work/data/Longitudinal_CT
export OUT_DIR=$HOME/work/results/registration

python register_baseline.py run "$SLURM_ARRAY_TASK_ID"
