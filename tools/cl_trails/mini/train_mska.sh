cd /scratch/yw6594/cf/mmcl

########## log for mask only without forgetting
#### amsk 0.15_msk ratio    0.1_loss weight
#### smsk 0.15_msk ratio    0.1_loss weight


python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
--cfg-options model.head.loss_weight=0.0 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/weight=0.0'
# #MSK ratio: 0.1 for cs;  3.0 for kd
python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
--cfg-options model.head.loss_weight=10.0 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/weight=10.0'

python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
--cfg-options model.head.loss_weight=5.0 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/weight=5.0'

python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
--cfg-options model.head.loss_weight=3.0 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/weight=3.0'

python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
--cfg-options model.head.loss_weight=1.0 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/weight=1.0'

python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
--cfg-options model.head.loss_weight=0.5 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/weight=0.5'

python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
--cfg-options model.head.loss_weight=0.1 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/weight=0.1'



# s-29 3.0 a-28 5.0 
# cs ratio 3.5 around weight 0.1, try mask mode
# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
# --cfg-options model.head.mask_ratio=0.95 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/ratio/=0.95'

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
# --cfg-options model.head.mask_ratio=0.25 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/ratio/=0.25'

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
# --cfg-options model.head.mask_ratio=0.5 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/ratio/=0.5'

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
# --cfg-options model.head.mask_ratio=0.4 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/ratio/=0.4'

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
# --cfg-options model.head.mask_ratio=0.35 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/ratio/=0.35'

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
# --cfg-options model.head.mask_ratio=0.3 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/ratio/=0.3'

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
# --cfg-options model.head.mask_ratio=0.25 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/ratio/=0.25'

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
# --cfg-options model.head.mask_ratio=0.2 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/ratio/=0.2'

# python tools/train.py /--your-own-dir/cifar-img/mask/cifar100_resnet18_msktest.py \
# --cfg-options model.head.mask_ratio=0.1 work_dir='/scratch/yw6594/out/cl_test/cifar100_mskonly/l2/amsk_kd/lr_mul/msk.10/ratio/=0.1'
