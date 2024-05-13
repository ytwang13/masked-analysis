#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=1:50:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7 # here replace with other interation
#SBATCH --job-name=MMCL_KDrank_emamsk_ema
#SBATCH --output=MMCL_KDrank_emaratiomsk_res18_cifar10-%j-%a.out # set the output file name for each task. The %j is replaced by the job ID and %a is replaced by the task ID.


#msk we test with mask_ratios 8 selections
# mask_ratios=(0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.10) #8
# mask_ratios=(0.18 0.13 0.12 0.09 0.05 0.03 0.01) #7
ema_ratios=(0.9 0.5 0.2 0.15 0.1 0.09 0.05 0.01) #8

echo "maskratios:" $mask_ratios[$SLURM_ARRAY_TASK_ID]
# echo "ema_ratios:" ${ema_ratios[$SLURM_ARRAY_TASK_ID]}
module purge
cd /scratch/yw6594/cf/mmcl
source /vast/yw6594/miniconda/bin/activate cl
# jobname: MMCL_KDrank_emamsk_res18_cifar10 
# test_res_mskema_var.sh  ${ema_ratios[$SLURM_ARRAY_TASK_ID]} ${mask_ratios[$SLURM_ARRAY_TASK_ID]} 
bash /scratch/yw6594/cf/mmcl/tools/cl_trails/msk_kd/test_res_mskema_var.sh ${ema_ratios[$SLURM_ARRAY_TASK_ID]} 0.50
##### lossweight less,.. more rank? |0.5 mask ema ratios exp| 92acc 55->60 rank (emaratio 0.1->0.01)
##### verify self-distillation results