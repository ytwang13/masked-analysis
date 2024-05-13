#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=3:30:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-10 # here replace with other interation
#SBATCH --job-name=MMCL_kdrankemaratio
#SBATCH --output=MMCL_cssfmxrankemaratio_clso_24-%j.out

ema_ratios=(0.9 0.7 0.5 0.3 0.1 0.07 0.05 0.01 0.007 0.005 0.001) #7
# ema_ratios=(0.3 0.1 0.007 0.005 0.001) #5

echo "ema_ratios:" $ema_ratios[$SLURM_ARRAY_TASK_ID]
module purge
cd /scratch/yw6594/cf/mmcl
source /vast/yw6594/miniconda/bin/activate cl
# MMCL_kdrank_res18_cifar10
# bash /scratch/yw6594/cf/mmcl/tools/cl_trails/baseline_selfKD/test_res_selfkd_var.sh ${ema_ratios[$SLURM_ARRAY_TASK_ID]}
#### 4.30 try classifier-only ema and kd?
bash /scratch/yw6594/cf/mmcl/tools/cl_trails/cls_only/test_selfkd_var.sh ${ema_ratios[$SLURM_ARRAY_TASK_ID]}