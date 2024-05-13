source /vast/yw6594/miniconda/bin/activate cl
cd /scratch/yw6594/cf/mmcl


############ t2

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_clt2.py \
# --cfg-options work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/ratiot2/3.0'              # ratio 3.0 acc 49

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_clt2.py \
# --cfg-options model.head.loss_weight=5.0 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/ratiot2/5.0'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_clt2.py \
# --cfg-options model.head.loss_weight=10.0 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/ratiot2/10.0'

#### try lr_mult ratio =3.0                                                                  # lrmult 0.1 acc 52 lrmult not so useful for short tasks
# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_clt2.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.1 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/ratiot2/lrmult_/0.1'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_clt2.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.3 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/ratiot2/lrmult_/0.3'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_clt2.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.6 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/ratiot2/lrmult_/0.6'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_clt2.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.9 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/ratiot2/lrmult_/0.9'

############ t2




############ t10 33 min 10_ratio 25
#### try lr_mult
# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.1 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/lrmult/0.1'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.9 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/lrmult/0.9'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.5 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/lrmult/0.5'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.3 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/lrmult/0.3'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.05 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/lrmult/0.05'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.08 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/lrmult/0.08'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options model.head.loss_weight=5.0 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/ratio/5.0'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options model.head.loss_weight=3.0 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/ratio/3.0'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options model.head.loss_weight=1.0 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/ratio/1.0'
### seems like longer task is more sensitive to large lwf ratio

#seed0 27
# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options randomness.seed=1 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/lrmult0.3_ratio10/seed1'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options randomness.seed=2 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/lrmult0.3_ratio10/seed2'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options randomness.seed=3 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/lrmult0.3_ratio10/seed3'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options randomness.seed=4 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/lrmult0.3_ratio10/seed4'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options randomness.seed=5 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/lrmult0.3_ratio10/seed5'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options randomness.seed=6 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/lrmult0.3_ratio10/seed6'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options randomness.seed=7 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/lrmult0.3_ratio10/seed7'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100_resnet18_cl.py \
# --cfg-options randomness.seed=8 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/lrmult0.3_ratio10/seed8'

############ t10


# ##### TODO
# 1. DATASET
# 2. CL_THING_MODEL
# 3. LOOP? CLdataset return multiple dataset? build_dataloader return build_dataloaders
# 4. OPtimizer
# 5. method