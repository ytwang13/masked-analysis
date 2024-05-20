cd  /scratch/yw6594/cf/mmcl
export CUBLAS_WORKSPACE_CONFIG=:16:8 
export WANDB_DATA_DIR='/scratch/yw6594/cf/out'
# model.head.mask_inv=True model.head.mask_ratio=$mask_ratio
# model.head.mask_mode='a' model.head.mask_mode='a' agf sgf
# model.head.mask_inv=True
python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kd.py \
--cfg-options  model.head.loss_weight=7.0  expname='kdbase_step25_ep50_lw7.0' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdbase/0.1/kd_25/7.0'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kd.py \
--cfg-options  model.head.loss_weight=5.0  expname='kdbase_step25_ep50_lw5.0' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdbase/0.1/kd_25/5.0'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kd.py \
--cfg-options  model.head.loss_weight=1.0  expname='kdbase_step25_ep50_lw1.0' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdbase/0.1/kd_25/1.0'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kd.py \
--cfg-options  model.head.loss_weight=0.5  expname='kdbase_step25_ep50_lw0.5' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdbase/0.1/kd_25/0.5'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kd.py \
--cfg-options  model.head.loss_weight=0.3  expname='kdbase_step25_ep50_lw0.3' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdbase/0.1/kd_25/0.3'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kd.py \
--cfg-options  model.head.loss_weight=0.1  expname='kdbase_step25_ep50_lw0.1' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdbase/0.1/kd_25/0.1'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kd.py \
--cfg-options  model.head.loss_weight=0.090  expname='kdbase_step25_ep50_lw0.090' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdbase/0.1/kd_25/0.090'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kd.py \
--cfg-options  model.head.loss_weight=0.050  expname='kdbase_step25_ep50_lw0.050' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdbase/0.1/kd_25/0.050'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kd.py \
--cfg-options  model.head.loss_weight=0.030  expname='kdbase_step25_ep50_lw0.030' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdbase/0.1/kd_25/0.030'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_kd.py \
--cfg-options  model.head.loss_weight=0.010  expname='kdbase_step25_ep50_lw0.010' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdbase/0.1/kd_25/0.010'


