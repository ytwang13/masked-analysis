source /vast/yw6594/miniconda/bin/activate cl
cd /scratch/yw6594/cf/mmcl

############ t10 61 min 10_res50_ratio_ 25
##### lr_mult0.3 ratio 10 seed 10
# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options randomness.seed=1 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/seed/=1'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options randomness.seed=2 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/seed/=2'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options randomness.seed=3 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/seed/=3'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options randomness.seed=4 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/seed/=4'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options randomness.seed=5 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/seed/=5'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options randomness.seed=6 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/seed/=6'

python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
--cfg-options randomness.seed=7 \
work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/seed/=7'

python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
--cfg-options randomness.seed=8 \
work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/seed/=8'

python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
--cfg-options randomness.seed=9 \
work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/seed/=9'

python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
--cfg-options randomness.seed=10 \
work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/seed/=10'

#### try lr_mult
# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.4 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/0.4'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.3 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/0.3'


# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.7 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/0.7'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.9 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/0.9'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.5 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/0.5'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.3 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/0.3'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.05 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/0.05'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.08 \
# work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ep30_lrmult/0.08'


# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options model.head.loss_weight=20.0 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ratio_/20.0'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options model.head.loss_weight=18.0 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ratio_/18.0'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options model.head.loss_weight=15.0 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ratio_/15.0'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options model.head.loss_weight=13.0 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ratio_/13.0'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options model.head.loss_weight=10.0 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ratio_/10.0' # 23

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options model.head.loss_weight=5.0 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ratio_/5.0'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options model.head.loss_weight=3.0 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ratio_/3.0'

# python tools/train.py /--your-own-dir/cifar-img/cifar100/cifar100_resnet50_cl.py \
# --cfg-options model.head.loss_weight=1.0 work_dir='/scratch/yw6594/out/cl_test/cifar100-lwf/res50_ratio_/1.0'
## seems like longer task is more sensitive to large lwf res50_ratio_

#seed0 27

############ t10


# ##### TODO
# 1. DATASET
# 2. CL_THING_MODEL
# 3. LOOP? CLdataset return multiple dataset? build_dataloader return build_dataloaders
# 4. OPtimizer
# 5. method