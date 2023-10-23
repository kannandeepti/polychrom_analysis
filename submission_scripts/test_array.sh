#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -o test_polychrom.sh.log-%A-%a
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-2

source /etc/profile
module load anaconda/2023b

echo $SLURM_ARRAY_TASK_ID
echo $SLURM_ARRAY_TASK_COUNT

source activate polychrom
python simulation_scripts/activity_sweep.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
conda deactivate
