cd  /-yourdirectory-/mmcl
# replace //-yourdirectory- to your directory
# export CUDA_VISIBLE_DEVICES=2

##########################################################################################
#################################### Baseline-model base channel EXP
##########################################################################################
# TO RUN Corresponding experiment, just paste the following setting after --cfg-options
# And the saving results will be in work_dir='xxxx'
############### here we just vary the input conv kernel size, [7, 5, 3] (here the spatial size is averged out by the GAP module)
# cifar10_resnet18_original.py    
# cifar10_resnet18_5x5.py
# cifar10_resnet18_3x3.py
############### the training epochs
# param_scheduler.milestones=[25,35] train_cfg.max_epochs=50
# param_scheduler.milestones=[35,50] train_cfg.max_epochs=70
# param_scheduler.milestones=[100,150] train_cfg.max_epochs=200
############### and the base channel size = [64, 40, 32, 16]
# model.base_channels=40 model.base_channels=320
# model.base_channels=32 model.base_channels=256
# model.base_channels=16 model.base_channels=128


python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/base-conv/cifar10_resnet18_original.py \
--cfg-options param_scheduler.milestones=[25,35] train_cfg.max_epochs=50 work_dir='/-yourdirectory-/out/dl/res18-c10/cifarbase/c64'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/base-conv/cifar10_resnet18_original.py \
--cfg-options param_scheduler.milestones=[35,50] train_cfg.max_epochs=70 work_dir='/-yourdirectory-/out/dl/res18-c10/cifarbase/c64'

python tools/train.py /scratch/yw6594/cf/mmcl/cifar-img/dl_res18_exp/base-conv/cifar10_resnet18_original.py \
--cfg-options param_scheduler.milestones=[100,150] train_cfg.max_epochs=200 work_dir='/-yourdirectory-/out/dl/res18-c10/cifarbase/c64'



########## we test baseline resnet18 (7x7 conv1 11.182M) for 50 70 200 overall epochs
# |epoch| acc-1| acc-5|
# |50   | 85.6 | 99.3 |
# |70   | 86.4 | 99.3 |
# |200  | 87.5 | 99.3 |


##########  we test baseline resnet18-cifar (3x3 conv1 11.174M) for 50 70 200 overall epochs
# |epoch| acc-1| acc-5|
# |50   | 92.8 | 99.8 |
# |70   | 93.5 | 99.8 |
# |200  | 94.4 | 99.8 |


########## we test baseline resnet18-cifar (3x3 base_c=40 😊4.384M) for 50 70 200 overall epochs
# |epoch| acc-1| acc-5|
# |50   | 92.29| 99.79|
# |70   | 92.90| 99.83|
# |200  | 94.14| 99.78| ### so far sota? 🛫🛫🛫🛫🛫🛫🛫🛫
