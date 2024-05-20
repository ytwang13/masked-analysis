# python tools/train.py /--your-own-dir/cifar-img/cifar100_resnet18.py \
# --cfg-options work_dir='/scratch/yw6594/out/cl_test/mmpretrain-res' \
source /vast/yw6594/miniconda/bin/activate cl
cd /scratch/yw6594/cf/mmcl

# ##### TODO
# 1. DATASET
# 2. CL_THING_MODEL
# 3. LOOP? CLdataset return multiple dataset? build_dataloader return build_dataloaders
# 4. OPtimizer
# 5. method
## 6. accuracy change


########## EP 50
# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltest.py \
# --cfg-options train_cfg.max_epochs=50 optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.5 \
# work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/3.0_ep50_lrmul0.5'
########## EP 50


########## EP 30 
### 3min for t2

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltest.py \
# --cfg-options model.head.loss_weight=3.0 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/3.0_ep30'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltest.py \
# --cfg-options model.head.loss_weight=1.0 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/1.0_ep30'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltest.py \
# --cfg-options model.head.loss_weight=0.5 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/0.5_ep30'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltest.py \
# --cfg-options model.head.loss_weight=0.1 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/0.1_ep30'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltest.py \
# --cfg-options model.head.loss_weight=0.05 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/0.05_ep30'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltest.py \
# --cfg-options model.head.loss_weight=0.01 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/0.01_ep30'

###### 3.0 loss_weight tune lr multi?
# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltest.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.99 \
# work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/3.0_ep30_lrmul0.99'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltest.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.5 \
# work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/3.0_ep30_lrmul0.5'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltest.py \
# --cfg-options optim_wrapper2.paramwise_cfg.custom_keys.backbone.lr_mult=0.1 \
# work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/3.0_ep30_lrmul0.1'
###### 3.0 loss_weight tune lr multi?
######## still 3.0 for lwf_cifar10
# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltestt5.py \
# --cfg-options model.head.loss_weight=10.0 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3_t5/10.0_ep30'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltestt5.py \
# --cfg-options model.head.loss_weight=5.0 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3_t5/5.0_ep30'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltestt5.py \
# --cfg-options model.head.loss_weight=15.0 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3_t5/15.0_ep30'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltestt5.py \
# --cfg-options model.head.loss_weight=3.0 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3_t5/3.0_ep30'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltestt5.py \
# --cfg-options model.head.loss_weight=1.0 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3_t5/1.0_ep30'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltestt5.py \
# --cfg-options model.head.loss_weight=0.5 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3_t5/0.5_ep30'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltestt5.py \
# --cfg-options model.head.loss_weight=0.3 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3_t5/0.3_ep30'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltestt5.py \
# --cfg-options model.head.loss_weight=0.1 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3_t5/0.1_ep30'




########## EP 30




########## EP200
# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cltest.py \
# --cfg-options model.head.loss_weight=3.0 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/testratio/3.0'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cl.py \
# --cfg-options model.head.loss_weight=3.0 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/3.0'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cl.py \
# --cfg-options model.head.loss_weight=1.0 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/1.0'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cl.py \
# --cfg-options model.head.loss_weight=0.5 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/0.5'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cl.py \
# --cfg-options model.head.loss_weight=0.1 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/0.1'

# python tools/train.py /--your-own-dir/cifar-img/cifar_resnet18_cl.py \
# --cfg-options model.head.loss_weight=0.05 work_dir='/scratch/yw6594/out/cl_test/cifar10-lwf/ratiov3/0.05'
########## EP200



