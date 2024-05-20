cd  /scratch/yw6594/cf/mmcl
export CUBLAS_WORKSPACE_CONFIG=:16:8 
export WANDB_DATA_DIR='/scratch/yw6594/cf/out'

python tools/train.py /scratch/yw6594/cf/proj-upload/cifar-img/dl_res18_exp/baseline/cifar10_resnet18_c40_ce.py \
--cfg-options expname='BaseCE_step.5_ep50' work_dir='/scratch/yw6594/cf/out/dl/res18-c10/cifarbase/c40/conv3x3/baselineCE/ep50'

############### change the following lines in config to have ep70 and ep200 runs 
# # learning policy [.5 .7] step
# param_scheduler = dict(
#     type='MultiStepLR', by_epoch=True, milestones=[25, 35], gamma=0.1)

# # train, val, test setting
# train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=5)
