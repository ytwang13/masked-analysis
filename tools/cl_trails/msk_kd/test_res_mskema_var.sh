cd  /scratch/yw6594/cf/mmcl
export CUBLAS_WORKSPACE_CONFIG=:16:8 
emaratio=${1:-0.1}
mask_ratio=${2:-0.13}
# loss_weight=${3:-7.0}

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskema.py \
# --cfg-options model.head.mask_mode='s' model.head.mask_ratio=$mask_ratio  model.head.loss_weight=$loss_weight default_hooks.emahook.momentum=$emaratio default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/test/ema_ratio='${emaratio}''

# model.head.mask_ratio=$mask_ratio
# model.head.mask_mode='a' model.head.mask_mode='s'
python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskema.py \
--cfg-options model.head.mask_mode='s' model.head.mask_ratio=$mask_ratio  model.head.loss_weight=7.0 default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/emamaskwoinv/CSmaskweight/'${mask_ratio}'/7.0/ema_ratio='${emaratio}''

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskema.py \
--cfg-options model.head.mask_mode='s' model.head.mask_ratio=$mask_ratio  model.head.loss_weight=5.0 default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/emamaskwoinv/CSmaskweight/'${mask_ratio}'/5.0/ema_ratio='${emaratio}''

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskema.py \
--cfg-options model.head.mask_mode='s' model.head.mask_ratio=$mask_ratio  model.head.loss_weight=1.0 default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/emamaskwoinv/CSmaskweight/'${mask_ratio}'/1.0/ema_ratio='${emaratio}''

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskema.py \
--cfg-options model.head.mask_mode='s' model.head.mask_ratio=$mask_ratio  model.head.loss_weight=0.5 default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/emamaskwoinv/CSmaskweight/'${mask_ratio}'/0.5/ema_ratio='${emaratio}''

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskema.py \
--cfg-options model.head.mask_mode='s' model.head.mask_ratio=$mask_ratio  model.head.loss_weight=0.1 default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/emamaskwoinv/CSmaskweight/'${mask_ratio}'/0.1/ema_ratio='${emaratio}''

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_complemaskema.py \
--cfg-options model.head.mask_mode='s' model.head.mask_ratio=$mask_ratio  model.head.loss_weight=0.050 default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/emamaskwoinv/CSmaskweight/'${mask_ratio}'/0.050/ema_ratio='${emaratio}''


# WOINV EMA0.1 MASKRATIO 0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.10
##### regardless the mask setting or other things, the method overall is working
# WOINV MASKRATIO 0.50  EMA 0.95 0.85 0.75 0.50 0.35 0.25 0.15 0.10