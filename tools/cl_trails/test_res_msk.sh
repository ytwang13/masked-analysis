cd  /scratch/yw6594/cf/mmcl
export CUBLAS_WORKSPACE_CONFIG=:16:8 
# model.head.mask_inv=True model.head.mask_ratio=0.20
# model.head.mask_mode='a' model.head.mask_mode='s' agf sgf
# model.head.mask_inv=True
python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemask.py \
--cfg-options model.head.mask_mode='s' model.head.mask_inv=True model.head.mask_ratio=0.20  model.head.loss_weight=7.0 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/maskinvold/CSmaskweight/0.20/7.0'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemask.py \
--cfg-options model.head.mask_mode='s' model.head.mask_inv=True model.head.mask_ratio=0.20  model.head.loss_weight=5.0 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/maskinvold/CSmaskweight/0.20/5.0'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemask.py \
--cfg-options model.head.mask_mode='s' model.head.mask_inv=True model.head.mask_ratio=0.20  model.head.loss_weight=1.0 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/maskinvold/CSmaskweight/0.20/1.0'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemask.py \
--cfg-options model.head.mask_mode='s' model.head.mask_inv=True model.head.mask_ratio=0.20  model.head.loss_weight=0.5 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/maskinvold/CSmaskweight/0.20/0.5'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemask.py \
--cfg-options model.head.mask_mode='s' model.head.mask_inv=True model.head.mask_ratio=0.20  model.head.loss_weight=0.1 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/maskinvold/CSmaskweight/0.20/0.1'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemask.py \
--cfg-options model.head.mask_mode='s' model.head.mask_inv=True model.head.mask_ratio=0.20  model.head.loss_weight=0.050 work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/maskinvold/CSmaskweight/0.20/0.050'

#woinv maskratio 0.85 0.20 0.60 0.50 0.35 0.15 0.20

#involdv1 maskratio 0.13 0.10 0.20 0.50?