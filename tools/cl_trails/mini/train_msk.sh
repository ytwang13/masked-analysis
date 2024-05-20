cd /scratch/yw6594/cf/mmcl

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_mskmulti.py \
# --cfg-options model.head.loss_weight=10.0 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2_amultimsk_weight/msk.85/weight=10.0'

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_mskmulti.py \
# --cfg-options model.head.loss_weight=8.0 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2_amultimsk_weight/msk.85/weight=8.0'

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_mskmulti.py \
# --cfg-options model.head.loss_weight=7.0 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2_amultimsk_weight/msk.85/weight=7.0'

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_mskmulti.py \
# --cfg-options model.head.loss_weight=5.0 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2_amultimsk_weight/msk.85/weight=5.0'

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_mskmulti.py \
# --cfg-options model.head.loss_weight=3.0 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2_amultimsk_weight/msk.85/weight=3.0'

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_mskmulti.py \
# --cfg-options model.head.loss_weight=2.0 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2_amultimsk_weight/msk.85/weight=2.0'

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_mskmulti.py \
# --cfg-options model.head.loss_weight=1.0 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2_amultimsk_weight/msk.85/weight=1.0'


python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_mskmulti.py \
--cfg-options model.head.loss_weight=0.0 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2_amultimsk_weight/msk.50/multi/weight=0.0'

python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_mskmulti.py \
--cfg-options model.head.mask_multi=2 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2_amultimsk_weight/msk.50/multi/2'

python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_mskmulti.py \
--cfg-options model.head.mask_multi=4 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2_amultimsk_weight/msk.50/multi/4'

python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_mskmulti.py \
--cfg-options model.head.mask_multi=6 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2_amultimsk_weight/msk.50/multi/6'

python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_mskmulti.py \
--cfg-options model.head.mask_multi=8 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2_amultimsk_weight/msk.50/multi/8'
# loss_weight=7 10 5 3 15 20 seems like the multi choice is not so significant? multi not so... good
#L2 LOSS_WEIGHT = 3 5 7 10 15 20 / 1.0 0.5 0.1🌟 0.05 0.08 0.3 0.2
