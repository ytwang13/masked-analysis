cd  /scratch/yw6594/cf/mmcl
export CUBLAS_WORKSPACE_CONFIG=:16:8 
export WANDB_DATA_DIR='/scratch/yw6594/cf/out'
emaratio=${1:-0.2}
# model.head.mask_inv=True model.head.mask_ratio=$mask_ratio
# model.head.mask_mode='a' model.head.mask_mode='a' agf sgf
# model.head.mask_inv=True
python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kdema.py \
--cfg-options  model.head.loss_weight=15.0  default_hooks.emahook.momentum=$emaratio expname='kdema_step25_r'${emaratio}'_lw15.0' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdannelratiowandb/'${emaratio}'/kd_25/15.0'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kdema.py \
--cfg-options  model.head.loss_weight=10.0  default_hooks.emahook.momentum=$emaratio expname='kdema_step25_r'${emaratio}'_lw10.0' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdannelratiowandb/'${emaratio}'/kd_25/10.0'

#above kd version only
python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kdema.py \
--cfg-options  model.head.loss_weight=7.0  default_hooks.emahook.momentum=$emaratio expname='kdema_step25_r'${emaratio}'_lw7.0' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdannelratiowandb/'${emaratio}'/kd_25/7.00'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kdema.py \
--cfg-options  model.head.loss_weight=5.0  default_hooks.emahook.momentum=$emaratio expname='kdema_step25_r'${emaratio}'_lw5.0' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdannelratiowandb/'${emaratio}'/kd_25/5.00'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kdema.py \
--cfg-options  model.head.loss_weight=1.0  default_hooks.emahook.momentum=$emaratio expname='kdema_step25_r'${emaratio}'_lw1.0' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdannelratiowandb/'${emaratio}'/kd_25/1.00'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kdema.py \
--cfg-options  model.head.loss_weight=0.5  default_hooks.emahook.momentum=$emaratio expname='kdema_step25_r'${emaratio}'_lw0.5' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdannelratiowandb/'${emaratio}'/kd_25/0.500'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kdema.py \
--cfg-options  model.head.loss_weight=0.3  default_hooks.emahook.momentum=$emaratio expname='kdema_step25_r'${emaratio}'_lw0.3' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdannelratiowandb/'${emaratio}'/kd_25/0.300'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kdema.py \
--cfg-options  model.head.loss_weight=0.1  default_hooks.emahook.momentum=$emaratio expname='kdema_step25_r'${emaratio}'_lw0.1' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdannelratiowandb/'${emaratio}'/kd_25/0.100'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kdema.py \
--cfg-options  model.head.loss_weight=0.090  default_hooks.emahook.momentum=$emaratio expname='kdema_step25_r'${emaratio}'_lw0.090' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdannelratiowandb/'${emaratio}'/kd_25/0.090'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kdema.py \
--cfg-options  model.head.loss_weight=0.050  default_hooks.emahook.momentum=$emaratio expname='kdema_step25_r'${emaratio}'_lw0.050' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdannelratiowandb/'${emaratio}'/kd_25/0.050'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kdema.py \
--cfg-options  model.head.loss_weight=0.030  default_hooks.emahook.momentum=$emaratio expname='kdema_step25_r'${emaratio}'_lw0.030' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdannelratiowandb/'${emaratio}'/kd_25/0.030'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kdema.py \
--cfg-options  model.head.loss_weight=0.010  default_hooks.emahook.momentum=$emaratio expname='kdema_step25_r'${emaratio}'_lw0.010' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdannelratiowandb/'${emaratio}'/kd_25/0.010'


