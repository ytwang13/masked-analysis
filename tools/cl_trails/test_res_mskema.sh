cd  /scratch/yw6594/cf/mmcl
export CUBLAS_WORKSPACE_CONFIG=:16:8 
# model.head.mask_ratio=0.13
# model.head.mask_mode='a' model.head.mask_mode='s'
# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskema.py \
# --cfg-options model.head.mask_mode='s' model.head.mask_ratio=0.13  model.head.loss_weight=7.0 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/emamaskwoinv/CSmaskweight/0.13/7.0'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskema.py \
# --cfg-options model.head.mask_mode='s' model.head.mask_ratio=0.13  model.head.loss_weight=5.0 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/emamaskwoinv/CSmaskweight/0.13/5.0'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskema.py \
# --cfg-options model.head.mask_mode='s' model.head.mask_ratio=0.13  model.head.loss_weight=1.0 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/emamaskwoinv/CSmaskweight/0.13/1.0'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskema.py \
# --cfg-options model.head.mask_mode='s' model.head.mask_ratio=0.13  model.head.loss_weight=0.5 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/emamaskwoinv/CSmaskweight/0.13/0.5'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskema.py \
# --cfg-options model.head.mask_mode='s' model.head.mask_ratio=0.13  model.head.loss_weight=0.1 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/emamaskwoinv/CSmaskweight/0.13/0.1'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskema.py \
# --cfg-options model.head.mask_mode='s' model.head.mask_ratio=0.13  model.head.loss_weight=0.050 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/emamaskwoinv/CSmaskweight/0.13/0.050'

#EMA 0.13
# - 1E-4 0.1 0.3 0.01    0.5 0.8 0.13 0.09   
# woinv 0.05 0.1[🌟5.0ratio] 0.5 0.2?[6] 0.15 0.08[]