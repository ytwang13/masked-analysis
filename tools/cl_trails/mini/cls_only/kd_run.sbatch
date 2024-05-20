#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=3:30:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=MMCL_kdrankema_clso
#SBATCH --output=MMCL_kdrankema_clsonly_1-%j.out

module purge
cd /scratch/yw6594/cf/mmcl
source /vast/yw6594/miniconda/bin/activate cl
# MMCL_kdrank_res18_cifar10
# bash /--your-own-dir/tools/cl_trails/baseline_selfKD/test_res_selfkd_var.sh
bash /--your-own-dir/tools/cl_trails/cls_only/test_selfkd_var.sh
