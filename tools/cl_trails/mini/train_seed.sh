cd  /scratch/yw6594/cf/mmcl
export CUBLAS_WORKSPACE_CONFIG=:16:8 ### 20min - 200 ep

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=0 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/seed0'  #conv5x5  bn256

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=1 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/seed1'  #conv5x5  bn256

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=2 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/seed2'  #conv5x5  bn256

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=3 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/seed3'  #conv5x5  bn256

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=4 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/seed4'  #conv5x5  bn256

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=5 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/seed5'  #conv5x5  bn256

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=6 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/seed6'  #conv5x5  bn256

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=7 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/seed7'  #conv5x5  bn256

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=8 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/seed8'  #conv5x5  bn256

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/cifar10_resnet18_c40_seed.py \
--cfg-options randomness.seed=9 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/ensemble/seed9'  #conv5x5  bn256

