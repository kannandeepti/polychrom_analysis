#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -o test_polychrom.sh.log-%A-%a
#SBATCH --gres=gpu:volta:1
#SBATCH --array=1-2

# change above array value to something
source /etc/profile
module load anaconda/2023b

echo $SLURM_ARRAY_TASK_ID
echo $SLURM_ARRAY_TASK_COUNT

export PATH=/home/gridsan/lchan/.conda/envs/polychrom/bin:$PATH
echo $PATH
path_to_chr1=data/ABidentities_chr21_Su2020_artificial1.csv
path_to_chr2=data/ABidentities_chr21_Su2020_artificial2.csv
path_to_chr3=data/ABidentities_chr21_Su2020_blocky1.csv
path_to_chr4=data/ABidentities_chr21_Su2020_blocky2.csv
path_to_chr5=data/ABidentities_chr21_Su2020_2perlocus.csv
path_to_chr6=data/ABidentities_chr21_Su2020_simple1.csv

output_folder1=./artificial_chr/artificial1
output_folder2=./artificial_chr/artificial2
output_folder3=./artificial_chr/blocky1
output_folder4=./artificial_chr/blocky2
output_folder5=./artificial_chr/normal
output_folder12=./artificial_chr/simple1

output_folder6=./artificial_chr/artificial1_logclustered
output_folder7=./artificial_chr/artificial2_logclustered
output_folder8=./artificial_chr/blocky1_logclustered
output_folder9=./artificial_chr/blocky2_logclustered
output_folder10=./artificial_chr/normal_logclustered
output_folder11=./artificial_chr/simple1_logclustered

source activate polychrom
python simulation_scripts/sticky_active.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT $path_to_chr6 $output_folder12 # test one chromosome at a time
conda deactivate
