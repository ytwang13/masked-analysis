#!/bin/bash

#SBATCH --job-name=Array1d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=00:01:00
#SBATCH --array=0-7

## check example /scratch/work/public/examples/slurm/jobarry
#msk we test with mask_ratios 8 selections
mask_ratios=(0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.10)

echo "maskratios:" ${mask_ratios[$SLURM_ARRAY_TASK_ID]}

# module purge
# cd /scratch/yw6594/cf/mmcl
# source /vast/yw6594/miniconda/bin/activate cl
# # MMCL_KDrank_emamsk_res18_cifar10
# bash /scratch/yw6594/cf/mmcl/tools/cl_trails/test_res_msk.sh