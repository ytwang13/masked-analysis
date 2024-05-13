########## res50
### depth 0 1 2 
# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100/cifar100_resnet18_naive.py \
# --cfg-options model.head.mid_channels=None work_dir='/scratch/yw6594/out/cl_test/cifar100_toy/t5lr_mulWN/res18_d0'

# python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100/cifar100_resnet50_naive.py \
# --cfg-options model.head.mid_channels=None work_dir='/scratch/yw6594/out/cl_test/cifar100_toy/t5lr_mulWN/res50_d0'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100/cifar100_resnet50_naive.py \
--cfg-options model.head.mid_channels=[512] work_dir='/scratch/yw6594/out/cl_test/cifar100_toy/t5lr_mulWN/res50_d1'


python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100/cifar100_resnet50_naive.py \
--cfg-options model.head.mid_channels=[512,512] work_dir='/scratch/yw6594/out/cl_test/cifar100_toy/t5lr_mulWN/res50_d2'
# ### width 

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100/cifar100_resnet50_naive.py \
--cfg-options model.head.mid_channels=[128] work_dir='/scratch/yw6594/out/cl_test/cifar100_toy/t5lr_mulWN/res50_d1w128'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100/cifar100_resnet50_naive.py \
--cfg-options model.head.mid_channels=[256] work_dir='/scratch/yw6594/out/cl_test/cifar100_toy/t5lr_mulWN/res50_d1w256'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/cifar100/cifar100_resnet50_naive.py \
--cfg-options model.head.mid_channels=[1024] work_dir='/scratch/yw6594/out/cl_test/cifar100_toy/t5lr_mulWN/res50_d1w1024'
##### result ep30 in the shared dense layer not optimal.
# 128(512+100)   256(512+100)  512(512+100)     512(512+512+100)=575488 1024(512+100)=626688
# |d1w128|d1w256|d1    |d2   |d1w1024|
# |9.27  |9.09  |9.24  |9.02  |9.39  |#wlrmul
# |8.90  |9.25  |8.92  |8.80  |8.89  |#wolrmul