cd /scratch/yw6594/cf/mmcl
### res50_cifar100
# python tools/train.py /--your-own-dir/cifar-img/draft/cifar100_resnet18_optims.py \
# --cfg-options work_dir='/scratch/yw6594/out/cl_test/cifar100_ub/res18_ep200'

# python tools/train.py /--your-own-dir/cifar-img/draft/cifar100_resnet50.py \
# --cfg-options work_dir='/scratch/yw6594/out/cl_test/cifar100_ub/res50_ep200'

python tools/train.py /--your-own-dir/cifar-img/draft/cifar100_resnet18_optims.py \
--cfg-options work_dir='/scratch/yw6594/out/cl_test/cifar100_ub/res18_ep100'

python tools/train.py /--your-own-dir/cifar-img/draft/cifar100_resnet50.py \
--cfg-options work_dir='/scratch/yw6594/out/cl_test/cifar100_ub/res50_ep100'

# python tools/train.py /--your-own-dir/cifar-img/draft/cifar100_resnet18_optims.py \
# --cfg-options work_dir='/scratch/yw6594/out/cl_test/cifar100_ub/res18_ep50'

# python tools/train.py /--your-own-dir/cifar-img/draft/cifar100_resnet50.py \
# --cfg-options work_dir='/scratch/yw6594/out/cl_test/cifar100_ub/res50_ep50'


# python tools/train.py /--your-own-dir/cifar-img/draft/cifar100_resnet18_optims.py \
# --cfg-options work_dir='/scratch/yw6594/out/cl_test/cifar100_ub/res18_ep30'

# python tools/train.py /--your-own-dir/cifar-img/draft/cifar100_resnet50.py \
# --cfg-options work_dir='/scratch/yw6594/out/cl_test/cifar100_ub/res50_ep30'

# ### res18_cifar100
# python tools/train.py /--your-own-dir/cifar-img/cifar10_resnet18_optims.py \
# --cfg-options optim_wrapper.optimizer.lr=0.08 work_dir='/scratch/yw6594/out/cl_test/cifar10/ub/lr0.08'

# python tools/train.py /--your-own-dir/cifar-img/cifar10_resnet18_optims.py \
# --cfg-options optim_wrapper.optimizer.lr=0.05 work_dir='/scratch/yw6594/out/cl_test/cifar10/ub/lr0.05'

# python tools/train.py /--your-own-dir/cifar-img/cifar10_resnet18_optims.py \
# --cfg-options optim_wrapper.optimizer.lr=0.03 work_dir='/scratch/yw6594/out/cl_test/cifar10/ub/lr0.03'
