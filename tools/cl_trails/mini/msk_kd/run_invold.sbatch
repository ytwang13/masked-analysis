#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=2:10:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7 # here replace with other interation
#SBATCH --job-name=MMCL_KDranks_mskinvold
#SBATCH --output=MMCL_v2_mskinvoldcssgf_res18_cifar10-%j-%a.out # set the output file name for each task. The %j is replaced by the job ID and %a is replaced by the task ID.

#sgf not working? later try v1?
#msk we test with mask_ratios 8 selections
mask_ratios=(0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.05) #8
# ema_ratios=(0.9 0.8 0.5 0.2 0.1 0.05 0.01) #7

echo "maskratios:" $mask_ratios[$SLURM_ARRAY_TASK_ID]
# echo "ema_ratios:" ${ema_ratios[$SLURM_ARRAY_TASK_ID]}
module purge
cd /scratch/yw6594/cf/mmcl
source /vast/yw6594/miniconda/bin/activate cl
# jobname: MMCL_KDrank_emamsk_res18_cifar10 
# test_res_msk_var.sh  ${mask_ratios[$SLURM_ARRAY_TASK_ID]} v1
bash /--your-own-dir/tools/cl_trails/msk_kd/test_res_msk_var.sh ${mask_ratios[$SLURM_ARRAY_TASK_ID]} 'v2'

#v1: cssgf csa  a |
#v2: csa csagf |cssgf ?
# ----
#v3: mask only use previous classifier as target?
#