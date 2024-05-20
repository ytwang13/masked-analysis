cd  /scratch/yw6594/cf/mmcl
export CUBLAS_WORKSPACE_CONFIG=:16:8 
emaratio=${1:-0.2}
# start_epoch=${2:-1}
# mask_mode=${3:-'a'}
# model.head.mask_inv=True model.head.mask_ratio=$mask_ratio
# model.head.mask_mode='a' model.head.mask_mode='a' agf sgf
# model.head.mask_inv=True
# cifar10_resnet18_c40_kdclso(cls net kd; [_24:default | _1]) cifar10_resnet18_c40_kd(full net kd; [_25:default])
# kd_emaratio_clsonly/ annel_l2sfmx                            kd_emaratio_fullnet/ annel_l2sfmx               
# expname='25_l2sfmx_annel_emaratio'${emaratio}'_lw15.0'

python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/previous+wandb/cifar10_resnet18_c40_kdclso.py \
--cfg-options  model.head.loss_weight=15.0  default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd_emaratio_clsonly/'${emaratio}'/annel_l2sfmx>24/15.0'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/previous+wandb/cifar10_resnet18_c40_kdclso.py \
# --cfg-options  model.head.loss_weight=10.0  default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd_emaratio_clsonly/'${emaratio}'/annel_l2sfmx>24/10.0'

# #above kd version only
# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/previous+wandb/cifar10_resnet18_c40_kdclso.py \
# --cfg-options  model.head.loss_weight=7.0  default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd_emaratio_clsonly/'${emaratio}'/annel_l2sfmx>24/7.00'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/previous+wandb/cifar10_resnet18_c40_kdclso.py \
# --cfg-options  model.head.loss_weight=5.0  default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd_emaratio_clsonly/'${emaratio}'/annel_l2sfmx>24/5.00'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/previous+wandb/cifar10_resnet18_c40_kdclso.py \
# --cfg-options  model.head.loss_weight=1.0  default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd_emaratio_clsonly/'${emaratio}'/annel_l2sfmx>24/1.00'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/previous+wandb/cifar10_resnet18_c40_kdclso.py \
# --cfg-options  model.head.loss_weight=0.5  default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd_emaratio_clsonly/'${emaratio}'/annel_l2sfmx>24/0.500'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/previous+wandb/cifar10_resnet18_c40_kdclso.py \
# --cfg-options  model.head.loss_weight=0.3  default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd_emaratio_clsonly/'${emaratio}'/annel_l2sfmx>24/0.300'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/previous+wandb/cifar10_resnet18_c40_kdclso.py \
# --cfg-options  model.head.loss_weight=0.1  default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd_emaratio_clsonly/'${emaratio}'/annel_l2sfmx>24/0.100'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/previous+wandb/cifar10_resnet18_c40_kdclso.py \
# --cfg-options  model.head.loss_weight=0.090  default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd_emaratio_clsonly/'${emaratio}'/annel_l2sfmx>24/0.090'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/previous+wandb/cifar10_resnet18_c40_kdclso.py \
# --cfg-options  model.head.loss_weight=0.050  default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd_emaratio_clsonly/'${emaratio}'/annel_l2sfmx>24/0.050'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/previous+wandb/cifar10_resnet18_c40_kdclso.py \
# --cfg-options  model.head.loss_weight=0.030  default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd_emaratio_clsonly/'${emaratio}'/annel_l2sfmx>24/0.030'

# python tools/train.py /--your-own-dir/cifar-img/dl_res18_exp/previous+wandb/cifar10_resnet18_c40_kdclso.py \
# --cfg-options  model.head.loss_weight=0.010  default_hooks.emahook.momentum=$emaratio work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kd_emaratio_clsonly/'${emaratio}'/annel_l2sfmx>24/0.010'


### one problem, now we just use >method start and huge method step, 
### this could results in a layback for normal version and ema start epochs.
######### later test >= method start