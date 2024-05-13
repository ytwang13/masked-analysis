cd  /scratch/yw6594/cf/mmcl
export CUBLAS_WORKSPACE_CONFIG=:16:8 
# model.head.mask_inv=True model.head.mask_ratio=$mask_ratio
# model.head.mask_mode='a' model.head.mask_mode='a' agf sgf
# model.head.mask_inv=True
python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_cs.py \
--cfg-options  model.head.loss_weight=15.0  work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdema/0.2/kdsfmx_25/15.0'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_cs.py \
--cfg-options  model.head.loss_weight=10.0  work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdema/0.2/kdsfmx_25/10.0'

#above kd version only
python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_cs.py \
--cfg-options  model.head.loss_weight=7.0  work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdema/0.2/kdsfmx_25/7.0'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_cs.py \
--cfg-options  model.head.loss_weight=5.0  work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdema/0.2/kdsfmx_25/5.0'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_cs.py \
--cfg-options  model.head.loss_weight=1.0  work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdema/0.2/kdsfmx_25/1.0'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_cs.py \
--cfg-options  model.head.loss_weight=0.5  work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdema/0.2/kdsfmx_25/0.5'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_cs.py \
--cfg-options  model.head.loss_weight=0.3  work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdema/0.2/kdsfmx_25/0.3'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_cs.py \
--cfg-options  model.head.loss_weight=0.1  work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdema/0.2/kdsfmx_25/0.2'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_cs.py \
--cfg-options  model.head.loss_weight=0.090  work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdema/0.2/kdsfmx_25/0.090'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_cs.py \
--cfg-options  model.head.loss_weight=0.050  work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdema/0.2/kdsfmx_25/0.050'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_cs.py \
--cfg-options  model.head.loss_weight=0.030  work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdema/0.2/kdsfmx_25/0.030'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/cifar10_resnet18_c40_cs.py \
--cfg-options  model.head.loss_weight=0.010  work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/kdema/0.2/kdsfmx_25/0.010'


