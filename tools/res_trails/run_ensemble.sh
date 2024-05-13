cd  /scratch/yw6594/cf/mmcl
export CUBLAS_WORKSPACE_CONFIG=:16:8 ### 20min - 200 ep
##########################################################################################
#################################### Ensemble-model EXP
##########################################################################################

################ step2: random seeds to ensemble and distill
###### in the cifar10_resnet18_c40_distill.py config we need to modify:
# ensemble_model: to the cifar10_resnet18_c40_seed.py to your config path
# ensemble_ckp_list: to the cifar10_resnet18_c40_seed.py to your config checkpoint path in step1
###### here in model.head.kd_mode, we can use different distillation loss method
# model.head.kd_mode='ens' # ensemble +kd
# model.head.kd_mode='kd' # normal knowledge distillationo
# model.head.loss_weight # to control distill weight

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/ensemble/cifar10_resnet18_c40_distill.py
--cfg-options work_dir='/-yourdirectory-/out/dl/res18-c10/cifarbase/ensemble/all'



################ step1: random seeds to ensemble
python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/ensemble/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=0 work_dir='/-yourdirectory-/out/dl/res18-c10/cifarbase/ensemble/seed0'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/ensemble/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=1 work_dir='/-yourdirectory-/out/dl/res18-c10/cifarbase/ensemble/seed1'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/ensemble/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=2 work_dir='/-yourdirectory-/out/dl/res18-c10/cifarbase/ensemble/seed2'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/ensemble/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=3 work_dir='/-yourdirectory-/out/dl/res18-c10/cifarbase/ensemble/seed3'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/ensemble/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=4 work_dir='/-yourdirectory-/out/dl/res18-c10/cifarbase/ensemble/seed4'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/ensemble/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=5 work_dir='/-yourdirectory-/out/dl/res18-c10/cifarbase/ensemble/seed5'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/ensemble/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=6 work_dir='/-yourdirectory-/out/dl/res18-c10/cifarbase/ensemble/seed6'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/ensemble/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=7 work_dir='/-yourdirectory-/out/dl/res18-c10/cifarbase/ensemble/seed7'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/ensemble/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=8 work_dir='/-yourdirectory-/out/dl/res18-c10/cifarbase/ensemble/seed8'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/ensemble/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=9 work_dir='/-yourdirectory-/out/dl/res18-c10/cifarbase/ensemble/seed9'

